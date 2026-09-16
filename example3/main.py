# dual_domain_pinn_fno_soft_scaled_u1.py
# ------------------------------------------------------------
# 1D Example 3:
#   u_t + c (-Delta)^{alpha/2} u = f
#   u(x,t) = (t+1)*A_alpha*(1-x^2)_+^(alpha/2), x in [-1,1]
#
# New framework:
#   u_pred = u1_PINN + u2_FNO
#   no beta
#
# u1 branch:
#   frequency-domain PINN loss
#   + differentiable inverse Fourier reconstruction
#   + physical-domain GL/fPINNs residual loss
#
# u2 branch:
#   original FNO-style time accumulation
# ------------------------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Supporting modules for data, evaluation, and output organization.
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'other' / 'support'))
import audited_support as audit
import portfolio_io
EXAMPLE_ID = 3
_parser = argparse.ArgumentParser(description="Example 3: PINN + FNO")
_parser.add_argument('--alpha', type=float, default=2.0)
_parser.add_argument('--stage', choices=['train','evaluate'], default='train')
_parser.add_argument('--pinn-epochs', type=int, default=3000)
_parser.add_argument('--fno-epochs', type=int, default=3000)
_parser.add_argument('--out', type=Path)
ARGS = _parser.parse_args() if __name__ == '__main__' else _parser.parse_args([])
if not 0.0 < ARGS.alpha <= 2.0 or abs(ARGS.alpha-1.0) < 1e-12:
    raise ValueError('Require 0 < alpha <= 2; this script does not implement the alpha=1 limiting formula.')
X_LEFT, X_RIGHT = -1.0, 1.0



# ============================================================
# 0. Basic config
# ============================================================

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# Important:
# alpha = 1 会导致 2*cos(pi*alpha/2) 接近 0，不要先用 alpha=1 测。
ALPHA = ARGS.alpha
C = 1.0
T_END = 1.0
X_LEFT = -1.0
X_RIGHT = 1.0

# 原文例1常用 N=6, M=80；这里先给能跑的测试版
N_K = 80          # 频域逆变换采样点数量;
K_MAX = 8.0       # 频率截断
S = 200           # 空间网格数量
M = 80            # 时间步数量

# 训练轮数：先跑小规模看趋势，正式实验再加大
EPOCH_PINN = ARGS.pinn_epochs
EPOCH_FNO = ARGS.fno_epochs

LR_PINN = 3e-4
weight_decay = 1e-6
LR_FNO = 1e-3

# ============================================================
# FNO nondimensionalization / correction settings
# ============================================================
# Keep the original FNO architecture and time-accumulation logic:
#     g = FNO(f,x,t)
#     u2 = cumsum(g) * dt
#
# New idea:
#   1) normalize source input f by F_SCALE;
#   2) let FNO learn dimensionless g_norm;
#   3) recover g = G_SCALE * g_norm;
#   4) train using normalized u2 error:
#          ((u2_pred - u2_target) / U2_SCALE)^2
#
# Optional auxiliary loss:
#   also match normalized g_target = diff(u2_target)/dt.

USE_FNO_SCALE = True
W_FNO_U2 = 1.0
W_FNO_G_AUX = 0.1

# Constrain u2 using its own IC/BC labels; total-u errors are diagnostics.
# Note: in the current decomposition u = u1 + u2 and u2(0)=0.
# Since u1 is detached during FNO training, the IC term is mostly a
# diagnostic/constant term; the BC term can affect FNO through u2.
USE_FNO_ICBC_LOSS = True
W_FNO_IC = 0.1
W_FNO_BC = 0.1

SCALE_EPS = 1e-8

# u1 loss 权重
W_FREQ_EQ = 10
W_FREQ_IC =100
W_PHYS_EQ = 0
W_PHYS_IC = 0

# physical GL residual 采样数量
B_FREQ = 512
B_IC = 512
B_PHYS = 32

# GL 步长与项数
GL_H = 1.0 / 80
GL_TERMS = 80

OUT_DIR = str(ARGS.out or (Path(__file__).resolve().parent / "other" / ("alpha_" + str(ALPHA).replace(".", "_"))))
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# u1 soft scaled PINN switches
# ============================================================
# This keeps the soft-PINN structure:
#   uhat_theta(k,t) = s * NN(k,t)
# There is NO hard IC ansatz:
#   uhat_theta != uhat0 + t*s*NN
USE_OUTPUT_SCALE = True
USE_NORMALIZED_FREQ_LOSS = True

# If True, also normalizes the optional physical IC loss by the
# mean magnitude of u0(x). Default False to keep the original FNO-side
# framework closer to the baseline.
USE_NORMALIZED_PHYS_IC = False


# ============================================================
# 1. Exact solution and source term
# ============================================================


def g_torch(x: torch.Tensor, alpha: float = ALPHA):
    """
    Example 3:
    g(x)=
    2^{-alpha} Gamma(1/2)
    -----------------------------
    Gamma(1+alpha/2) * Gamma(1/2+alpha/2)
    * (1-x^2)^{alpha/2}
    """
    coef = (
        2.0 ** (-alpha)
        * math.gamma(0.5)
        / (math.gamma(1 + alpha/ 2.0) * math.gamma(1/2.0 + alpha/ 2.0))
    )

    return coef * torch.clamp(1-x**2, min=0.0) ** (alpha/2.0)


def u0_torch(x: torch.Tensor) -> torch.Tensor:
    return g_torch(x)


def u_exact_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t+1.0) * g_torch(x)


def source_f_torch(x: torch.Tensor, t: torch.Tensor, alpha: float = ALPHA):
    """
    Example 3:
        f(x,t)=g(x)+t+1
    """
    return g_torch(x, alpha) + C * (t + 1.0)


def true_u1_inverse_fourier(x: torch.Tensor, t: torch.Tensor):
    """Independent reference; valid labels do not depend on PINN predictions."""
    return audit.reference_u1(x, t, globals())

def compute_metrics(pred: torch.Tensor, true: torch.Tensor):
    """
    pred, true shape: [M, S]
    Return MAE / RMSE over domain and at t=1
    """
    err = pred - true

    mae_domain = torch.mean(torch.abs(err)).item()
    rmse_domain = torch.sqrt(torch.mean(err ** 2)).item()

    mae_t1 = torch.mean(torch.abs(err[-1, :])).item()
    rmse_t1 = torch.sqrt(torch.mean(err[-1, :] ** 2)).item()

    maxe_domain = torch.max(torch.abs(err)).item()
    maxe_t1 = torch.max(torch.abs(err[-1, :])).item()

    return {
        "mae_domain": mae_domain,
        "rmse_domain": rmse_domain,
        "maxe_domain": maxe_domain,
        "mae_t1": mae_t1,
        "rmse_t1": rmse_t1,
        "maxe_t1": maxe_t1,
    }

# ============================================================
# 2. Fourier transform of initial data u0
# ============================================================

def u0_hat_torch(k: torch.Tensor, quad_n: int = 512):
    """
    Compute Fourier transform of u0 on [-1,1]:
        \hat u0(k)=1/sqrt(2pi) int_0^1 u0(x) exp(-i k x) dx

    Return:
        real, imag
    """
    xq = torch.linspace(X_LEFT, X_RIGHT, quad_n, device=k.device)
    uq = u0_torch(xq)

    # shape: [B, quad_n]
    phase = k.reshape(-1, 1) * xq.reshape(1, -1)

    real_integrand = uq.reshape(1, -1) * torch.cos(phase)
    imag_integrand = -uq.reshape(1, -1) * torch.sin(phase)

    real = torch.trapz(real_integrand, xq, dim=1) / math.sqrt(2.0 * math.pi)
    imag = torch.trapz(imag_integrand, xq, dim=1) / math.sqrt(2.0 * math.pi)

    return real, imag


def get_uhat_scale() -> torch.Tensor:
    """
    Characteristic Fourier amplitude used only as a numerical scale.

    It changes the parameterization to:
        uhat_theta(k,t) = s * NN(k,t)

    It does NOT impose a hard initial condition.
    """
    with torch.no_grad():
        r0, i0 = u0_hat_torch(torch.tensor([0.0], device=device))
        scale = torch.sqrt(r0[0] ** 2 + i0[0] ** 2)
    return torch.clamp(scale, min=torch.tensor(1e-8, device=device)).detach()


UHAT_SCALE = get_uhat_scale()
print("UHAT_SCALE =", UHAT_SCALE.item())


# ============================================================
# 3. PINN for frequency-domain u1
# ============================================================

class PINNNet(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()

        layers = []
        in_dim = 2
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else width, width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(width, 2))  # real and imag

        self.net = nn.Sequential(*layers)

    def forward(self, kt_raw):
        # Normalize raw [k,t] to roughly [-1,1]^2.
        # This is an input scaling only; it does not impose constraints.
        kt = kt_raw.clone()
        kt[:, 0] = 2.0 * kt[:, 0] / K_MAX - 1.0
        kt[:, 1] = 2.0 * kt[:, 1] / T_END - 1.0
        return self.net(kt)


def pinn_complex_output(net: nn.Module, k: torch.Tensor, t: torch.Tensor):
    """
    Soft scaled PINN output:
        uhat_theta(k,t) = s * NN_theta(k,t)

    No hard IC ansatz is used:
        uhat_theta(k,t) != uhat0(k) + t*s*NN_theta(k,t)
    """
    kt = torch.stack([k, t], dim=1)
    out = net(kt)

    if USE_OUTPUT_SCALE:
        real = UHAT_SCALE * out[:, 0]
        imag = UHAT_SCALE * out[:, 1]
    else:
        real = out[:, 0]
        imag = out[:, 1]

    return real, imag


def frequency_pde_loss(net: nn.Module, alpha: float = ALPHA):
    """
    u1 homogeneous equation in Fourier domain:
        d_t \hat u1 + c |k|^alpha \hat u1 = 0

    If USE_NORMALIZED_FREQ_LOSS=True, residual is divided by UHAT_SCALE.
    This makes the loss dimensionless and asks the network to learn relative
    frequency-domain accuracy. It is not a hard constraint.
    """
    k = torch.rand(B_FREQ, device=device) * K_MAX
    t = torch.rand(B_FREQ, device=device) * T_END

    kt = torch.stack([k, t], dim=1)
    kt.requires_grad_(True)

    k_req = kt[:, 0]
    t_req = kt[:, 1]

    real, imag = pinn_complex_output(net, k_req, t_req)

    grad_real = torch.autograd.grad(
        real.sum(), kt, create_graph=True, retain_graph=True
    )[0]
    grad_imag = torch.autograd.grad(
        imag.sum(), kt, create_graph=True, retain_graph=True
    )[0]

    real_t = grad_real[:, 1]
    imag_t = grad_imag[:, 1]

    r_real = real_t + C * (k_req ** alpha) * real
    r_imag = imag_t + C * (k_req ** alpha) * imag

    if USE_NORMALIZED_FREQ_LOSS:
        return torch.mean((r_real / UHAT_SCALE) ** 2 + (r_imag / UHAT_SCALE) ** 2)

    return torch.mean(r_real ** 2 + r_imag ** 2)


def frequency_ic_loss(net: nn.Module):
    """
    Initial condition in Fourier domain:
        \hat u1(k,0)=\hat u0(k)

    Still a soft IC loss. The only change is optional scale normalization.
    """
    k = torch.rand(B_IC, device=device) * K_MAX
    t = torch.zeros_like(k)

    pred_real, pred_imag = pinn_complex_output(net, k, t)
    true_real, true_imag = u0_hat_torch(k)

    if USE_NORMALIZED_FREQ_LOSS:
        return torch.mean(
            ((pred_real - true_real) / UHAT_SCALE) ** 2
            + ((pred_imag - true_imag) / UHAT_SCALE) ** 2
        )

    return torch.mean((pred_real - true_real) ** 2 + (pred_imag - true_imag) ** 2)


# ============================================================
# 4. Differentiable inverse Fourier reconstruction
# ============================================================

K_GRID = torch.linspace(0.0, K_MAX, N_K, device=device)


def inverse_fourier_reconstruct(net: nn.Module, x: torch.Tensor, t: torch.Tensor):
    """
    Differentiable inverse Fourier reconstruction:
        u1(x,t) ≈ sqrt(2/pi) ∫_0^{Kmax} Re[ \hat u(k,t) exp(i k x) ] dk

    Note:
    这里先用梯形积分版本，稳定、可微。
    原文的指数插值 inverse_fourier_no2pi 可以后续替换进来。
    """
    B = x.shape[0]
    if B > 512:
        return torch.cat([inverse_fourier_reconstruct(net, x[j:j+512], t[j:j+512]) for j in range(0,B,512)])
    Nk = K_GRID.shape[0]

    kk = K_GRID.reshape(1, Nk).repeat(B, 1).reshape(-1)
    tt = t.reshape(B, 1).repeat(1, Nk).reshape(-1)

    real, imag = pinn_complex_output(net, kk, tt)
    real = real.reshape(B, Nk)
    imag = imag.reshape(B, Nk)

    phase = x.reshape(B, 1) * K_GRID.reshape(1, Nk)

    # Re[(a+ib)(cos+i sin)] = a cos - b sin
    integrand = real * torch.cos(phase) - imag * torch.sin(phase)

    u = torch.trapz(integrand, K_GRID, dim=1) * math.sqrt(2.0 / math.pi)
    return u

# ============================================================
# 5. GL / fPINNs fractional operator in physical domain
# ============================================================

def gl_weights(alpha: float, n_terms: int, device=device):
    """
    w_m = (-1)^m binom(alpha,m)
    recurrence:
        w_0=1
        w_m = w_{m-1} * (m-1-alpha)/m
    """
    w = torch.zeros(n_terms, device=device)
    w[0] = 1.0
    for m in range(1, n_terms):
        w[m] = w[m - 1] * ((m - 1 - alpha) / m)
    return w

GL_W = gl_weights(ALPHA, GL_TERMS, device=device)


def fractional_laplacian_gl_reconstructed(
    net: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = ALPHA,
    h: float = GL_H,
):
    """
    Riesz-type GL approximation consistent with the source formula:
        D_alpha u ≈ (D_left^alpha u + D_right^alpha u) / (2 cos(pi alpha/2))

    Zero extension outside [-1,1].
    """
    acc = torch.zeros_like(x)

    for m in range(GL_TERMS):
        shift = m * h
        wm = GL_W[m]

        x_left = x - shift
        x_right = x + shift

        mask_left = ((x_left >= X_LEFT) & (x_left <= X_RIGHT)).float()
        mask_right = ((x_right >= X_LEFT) & (x_right <= X_RIGHT)).float()

        # 即使 x 超出 [-1,1]，reconstruct 也能算；随后乘 mask 做零延拓
        u_left = inverse_fourier_reconstruct(net, x_left, t) * mask_left
        u_right = inverse_fourier_reconstruct(net, x_right, t) * mask_right

        acc = acc + wm * (u_left + u_right)

    denom = 2.0 * math.cos(math.pi * alpha / 2.0)
    return acc / (denom * (h ** alpha))


def physical_pde_loss_u1(net: nn.Module):
    """
    u1 physical-domain homogeneous residual:
        u1_t + c(-Delta)^{alpha/2}u1 = 0

    这里的 fractional operator 用 GL/fPINNs 离散近似。
    """
    # 避开端点，减少 GL 边界奇异/截断影响
    eps = 2.0 * GL_H
    x = X_LEFT + eps + (X_RIGHT-X_LEFT-2.0*eps)*torch.rand(B_PHYS, device=device)
    t = torch.rand(B_PHYS, device=device) * T_END

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = inverse_fourier_reconstruct(net, x, t)

    u_t = torch.autograd.grad(
        u.sum(), t, create_graph=True, retain_graph=True
    )[0]

    frac = fractional_laplacian_gl_reconstructed(net, x, t)

    res = u_t + C * frac
    return torch.mean(res ** 2)

def physical_ic_loss_u1(net: nn.Module):
    """
    u1 physical-domain initial condition:
        u1(x,0)=u0(x)

    This remains a soft loss. By default it keeps the original raw MSE.
    """
    x = X_LEFT + (X_RIGHT-X_LEFT)*torch.rand(B_IC, device=device)
    t = torch.zeros_like(x)

    pred = inverse_fourier_reconstruct(net, x, t)
    true = u0_torch(x)

    if USE_NORMALIZED_PHYS_IC:
        scale = torch.mean(torch.abs(true)).detach() + 1e-8
        return torch.mean(((pred - true) / scale) ** 2)

    return torch.mean((pred - true) ** 2)


# ============================================================
# 6. Train u1 branch
# ============================================================

def train_u1_branch():
    net = PINNNet(width=64, depth=5).to(device)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=LR_PINN,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCH_PINN,
        eta_min=1e-5
    )

    history = []

    for ep in range(1, EPOCH_PINN + 1):
        opt.zero_grad()

        lf_eq = frequency_pde_loss(net)
        lf_ic = frequency_ic_loss(net)

        # 如果 W_PHYS_EQ = 0，就不要计算最慢、最震荡的 GL 残差
        if W_PHYS_EQ != 0:
            lx_eq = physical_pde_loss_u1(net)
        else:
            lx_eq = torch.tensor(0.0, device=device)

        if W_PHYS_IC != 0:
            lx_ic = physical_ic_loss_u1(net)
        else:
            lx_ic = torch.tensor(0.0, device=device)

        loss = (
            W_FREQ_EQ * lf_eq
            + W_FREQ_IC * lf_ic
            + W_PHYS_EQ * lx_eq
            + W_PHYS_IC * lx_ic
        )

        loss.backward()

        # 关键：防止梯度爆冲
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        opt.step()
        scheduler.step()

        if ep % 100 == 0 or ep == 1:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"[u1 PINN] ep={ep:5d} "
                f"lr={current_lr:.2e} "
                f"loss={loss.item():.3e} "
                f"freq_eq={lf_eq.item():.3e} "
                f"freq_ic={lf_ic.item():.3e} "
                f"phys_eq={lx_eq.item():.3e} "
                f"phys_ic={lx_ic.item():.3e}"
            )

        history.append([
            loss.item(),
            lf_eq.item(),
            lf_ic.item(),
            lx_eq.item(),
            lx_ic.item(),
        ])

    return net, np.array(history)

def debug_network_k0(net):
    t = torch.linspace(0.0, 1.0, 6, device=device)
    k = torch.zeros_like(t)

    with torch.no_grad():
        pred_real, pred_imag = pinn_complex_output(net, k, t)
        true_real0, true_imag0 = u0_hat_torch(torch.tensor([0.0], device=device))

    print("\n========== network k=0 mode ==========")
    print("true real k=0 =", true_real0.item())
    for i in range(t.numel()):
        print(
            f"t={t[i].item():.2f}, "
            f"pred_real={pred_real[i].item():.6e}, "
            f"pred_imag={pred_imag[i].item():.6e}"
        )
    print("======================================\n")


# ============================================================
# 7. FNO branch for u2
# ============================================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x, weights):
        # x: [batch, in_channel, modes]
        # weights: [in_channel, out_channel, modes]
        return torch.einsum("bim,iom->bom", x, weights)

    def forward(self, x):
        # x: [batch, channels, spatial]
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(
            x_ft[:, :, :modes],
            self.weights[:, :, :modes],
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1dNoBeta(nn.Module):
    """
    FNO 输入:
        f(x,t), x, t
    输出:
        g(x,t)
    然后 u2 = cumsum(g)*dt
    """
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, f_input, times):
        """
        f_input: [M, S, 1]
        """
        batchsize, size_x, _ = f_input.shape

        x_grid = torch.linspace(X_LEFT, X_RIGHT, size_x, device=f_input.device)
        x_grid = x_grid.reshape(1, size_x, 1).repeat(batchsize, 1, 1)

        t_grid = times.to(device=f_input.device, dtype=f_input.dtype)
        if t_grid.numel() != batchsize:
            raise ValueError("One explicit time is required per source snapshot")
        t_grid = t_grid.reshape(batchsize, 1, 1).repeat(1, size_x, 1)

        x = torch.cat([f_input, x_grid, t_grid], dim=-1)

        x = self.fc0(x)             # [batch, S, width]
        x = x.permute(0, 2, 1)      # [batch, width, S]

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))

        x = x.permute(0, 2, 1)      # [batch, S, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)             # [batch, S, 1]

        return x


# ============================================================
# 8. Data generation and FNO training
# ============================================================

def build_training_grids(net_u1):
    """
    Construct FNO grids.

    Returns both right-time data for the original u2 training and all-time
    data for the added full-solution IC/BC losses.
    """
    x_grid = torch.linspace(X_LEFT, X_RIGHT, S, device=device)
    t_all = torch.linspace(0.0, T_END, M + 1, device=device)

    t_left = t_all[:-1]
    t_right = t_all[1:]
    dt = T_END / M

    xx_left, tt_left = torch.meshgrid(x_grid, t_left, indexing="xy")
    xx_right, tt_right = torch.meshgrid(x_grid, t_right, indexing="xy")
    xx_all, tt_all = torch.meshgrid(x_grid, t_all, indexing="xy")

    # meshgrid indexing="xy" gives [time, space]
    f_train = source_f_torch(xx_left, tt_left).reshape(M, S, 1)
    u_true = u_exact_torch(xx_right, tt_right).reshape(M, S)
    u_true_all = u_exact_torch(xx_all, tt_all).reshape(M + 1, S)

    # u1 prediction at t_right and t_all
    with torch.no_grad():
        x_flat = xx_right.reshape(-1)
        t_flat = tt_right.reshape(-1)
        u1_flat = inverse_fourier_reconstruct(net_u1, x_flat, t_flat)
        u1_pred = u1_flat.reshape(M, S)

        x_all_flat = xx_all.reshape(-1)
        t_all_flat = tt_all.reshape(-1)
        u1_all_flat = inverse_fourier_reconstruct(net_u1, x_all_flat, t_all_flat)
        u1_all = u1_all_flat.reshape(M + 1, S)

    return x_grid, t_all, t_right, f_train, u_true, u_true_all, u1_pred, u1_all, dt

def train_fno_branch(net_u1):
    """Train against independent u2 reference labels and their time increments.

    The PINN is used only to report total-u diagnostics. Both the supervised
    target and the u2 boundary loss are independent of the PINN prediction.
    The zero initial condition of u2 is imposed by the cumulative sum.
    """
    x_grid, t_all, t_right, f_train, u_true, u_true_all, u1_pred, u1_all, dt = build_training_grids(net_u1)

    model = FNO1dNoBeta(modes=16, width=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_FNO, weight_decay=1e-6)

    # ------------------------------------------------------------
    # Build u2 target and scales
    # ------------------------------------------------------------
    # Reference labels are allowed, but may not absorb the PINN's errors.
    xx_ref, tt_ref = torch.meshgrid(x_grid, t_all, indexing="xy")
    with torch.no_grad():
        u1_reference_all = true_u1_inverse_fourier(xx_ref.reshape(-1), tt_ref.reshape(-1)).reshape(M+1,S)
        u2_reference_all = u_true_all - u1_reference_all
        u2_reference_all[0] = 0.0
    u2_target = u2_reference_all[1:].detach()  # independent of net_u1

    # g_target is the discrete increment derivative:
    #   u2_pred[m] = sum_{j<=m} g[j] * dt
    u2_prev = torch.cat(
        [torch.zeros(1, u2_target.shape[1], device=device), u2_target[:-1]],
        dim=0,
    )
    g_target = ((u2_target - u2_prev) / dt).detach()

    f_scale = torch.mean(torch.abs(f_train)).detach().clamp_min(SCALE_EPS)
    u2_scale = torch.mean(torch.abs(u2_target)).detach().clamp_min(SCALE_EPS)
    g_scale = torch.mean(torch.abs(g_target)).detach().clamp_min(SCALE_EPS)
    u_full_scale = torch.mean(torch.abs(u_true_all)).detach().clamp_min(SCALE_EPS)

    if USE_FNO_SCALE:
        f_train_in = f_train / f_scale
    else:
        f_train_in = f_train

    print("\n========== FNO scales ==========")
    print(f"USE_FNO_SCALE     = {USE_FNO_SCALE}")
    print(f"USE_FNO_ICBC_LOSS = {USE_FNO_ICBC_LOSS}")
    print(f"F_SCALE           = {f_scale.item():.6e}")
    print(f"U2_SCALE          = {u2_scale.item():.6e}")
    print(f"G_SCALE           = {g_scale.item():.6e}")
    print(f"U_FULL_SCALE      = {u_full_scale.item():.6e}")
    print(f"W_FNO_U2          = {W_FNO_U2}")
    print(f"W_FNO_G_AUX       = {W_FNO_G_AUX}")
    print(f"W_FNO_IC          = {W_FNO_IC}")
    print(f"W_FNO_BC          = {W_FNO_BC}")
    print("================================\n")

    history = []

    for ep in range(1, EPOCH_FNO + 1):
        opt.zero_grad()

        out = model(f_train_in, t_all[:-1])           # [M,S,1]
        g_norm = out.squeeze(-1)          # [M,S]

        if USE_FNO_SCALE:
            g = g_scale * g_norm
        else:
            g = g_norm

        # original-style time accumulation
        u2_pred = torch.cumsum(g, dim=0) * dt

        # total prediction on t_right
        u_pred = u1_pred.detach() + u2_pred

        # total prediction on t_all, with u2(x,0)=0
        u2_zero = torch.zeros(1, S, device=device)
        u2_all = torch.cat([u2_zero, u2_pred], dim=0)
        u_pred_all = u1_all.detach() + u2_all

        if USE_FNO_SCALE:
            loss_u2 = torch.mean(((u2_pred - u2_target) / u2_scale) ** 2)
            loss_g = torch.mean((g_norm - g_target / g_scale) ** 2)
        else:
            loss_u2 = torch.mean((u2_pred - u2_target) ** 2)
            loss_g = torch.mean((g - g_target) ** 2)

        if USE_FNO_ICBC_LOSS:
            loss_ic = torch.mean((u2_all[0, :] / u2_scale) ** 2)  # hard zero IC; diagnostic only
            loss_bc_left = torch.mean(((u2_all[:, 0] - u2_reference_all[:, 0]) / u2_scale) ** 2)
            loss_bc_right = torch.mean(((u2_all[:, -1] - u2_reference_all[:, -1]) / u2_scale) ** 2)
            loss_bc = loss_bc_left + loss_bc_right
        else:
            loss_ic = torch.tensor(0.0, device=device)
            loss_bc = torch.tensor(0.0, device=device)

        loss = (
            W_FNO_U2 * loss_u2
            + W_FNO_G_AUX * loss_g
            + W_FNO_IC * loss_ic
            + W_FNO_BC * loss_bc
        )

        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == 1:
            mae_u2 = torch.mean(torch.abs(u2_pred - u2_target)).item()
            mae_u = torch.mean(torch.abs(u_pred - u_true)).item()
            raw_mse_u2 = torch.mean((u2_pred - u2_target) ** 2).item()
            raw_ic = torch.mean((u_pred_all[0, :] - u_true_all[0, :]) ** 2).item()
            raw_bc = (
                torch.mean(u_pred_all[:, 0] ** 2)
                + torch.mean(u_pred_all[:, -1] ** 2)
            ).item()
            print(
                f"[FNO scaled+ICBC] ep={ep:5d} "
                f"loss={loss.item():.3e} "
                f"loss_u2={loss_u2.item():.3e} "
                f"loss_g={loss_g.item():.3e} "
                f"loss_ic={loss_ic.item():.3e} "
                f"loss_bc={loss_bc.item():.3e} "
                f"raw_mse_u2={raw_mse_u2:.3e} "
                f"raw_ic={raw_ic:.3e} "
                f"raw_bc={raw_bc:.3e} "
                f"mae_u2={mae_u2:.3e} "
                f"mae_total={mae_u:.3e}"
            )

        history.append([
            loss.item(),
            loss_u2.item(),
            loss_g.item(),
            torch.mean((u2_pred - u2_target) ** 2).item(),
            torch.mean(torch.abs(u2_pred - u2_target)).item(),
            torch.mean(torch.abs(u_pred - u_true)).item(),
            loss_ic.item(),
            loss_bc.item(),
            torch.mean((u_pred_all[0, :] - u_true_all[0, :]) ** 2).item(),
            (
                torch.mean(u_pred_all[:, 0] ** 2)
                + torch.mean(u_pred_all[:, -1] ** 2)
            ).item(),
        ])

    scale_pack = {
        "f_scale": f_scale.detach(),
        "u2_scale": u2_scale.detach(),
        "g_scale": g_scale.detach(),
        "u_full_scale": u_full_scale.detach(),
        "u2_target": u2_target.detach(),
        "u1_reference_all": u1_reference_all.detach(),
        "u2_reference_all": u2_reference_all.detach(),
        "g_target": g_target.detach(),
        "f_train_in": f_train_in.detach(),
        "u_true_all": u_true_all.detach(),
        "u1_all": u1_all.detach(),
        "t_all": t_all.detach(),
    }

    return model, np.array(history), (x_grid, t_all, t_right, f_train, u_true, u_true_all, u1_pred, u1_all, dt, scale_pack)


# ============================================================
# 9. Evaluation and plots
# ============================================================

def evaluate_and_plot(net_u1, fno_model, data_pack):
    # data_pack:
    #   (x_grid, t_all, t_right, f_train, u_true, u_true_all,
    #    u1_pred, u1_all, dt, scale_pack)
    x_grid, t_all, t_right, f_train, u_true, u_true_all, u1_pred, u1_all, dt, scale_pack = data_pack

    with torch.no_grad():
        # -----------------------------
        # 1) FNO gives u2_pred
        # -----------------------------
        if USE_FNO_SCALE:
            f_input = f_train / scale_pack["f_scale"]
        else:
            f_input = f_train

        g_norm = fno_model(f_input, t_all[:-1]).squeeze(-1)      # [M, S]

        if USE_FNO_SCALE:
            g = scale_pack["g_scale"] * g_norm
        else:
            g = g_norm

        u2_pred = torch.cumsum(g, dim=0) * dt        # [M, S]

        # total prediction on t_right
        u_pred = u1_pred + u2_pred                  # [M, S]

        # total prediction on t_all
        u2_all = torch.cat([torch.zeros(1, S, device=device), u2_pred], dim=0)
        u_pred_all = u1_all + u2_all                # [M+1, S]

        # -----------------------------
        # 2) Build true u1 on same grid
        # -----------------------------
        xx, tt = torch.meshgrid(x_grid, t_right, indexing="xy")
        x_flat = xx.reshape(-1)
        t_flat = tt.reshape(-1)

        u1_true_flat = true_u1_inverse_fourier(x_flat, t_flat)
        u1_true = u1_true_flat.reshape(u_true.shape[0], u_true.shape[1])  # [M, S]

        # -----------------------------
        # 3) Build true u2 = u - u1_true
        # -----------------------------
        u2_true = u_true - u1_true

        # -----------------------------
        # 4) Metrics
        # -----------------------------
        metrics_u1 = compute_metrics(u1_pred, u1_true)
        metrics_u2 = compute_metrics(u2_pred, u2_true)
        metrics_u = compute_metrics(u_pred, u_true)

        # extra FNO diagnostics
        u2_train_target = scale_pack["u2_target"]
        train_target_mse = torch.mean((u2_pred - u2_train_target) ** 2).item()
        train_target_mae = torch.mean(torch.abs(u2_pred - u2_train_target)).item()

        ic_mse = torch.mean((u_pred_all[0, :] - u_true_all[0, :]) ** 2).item()
        ic_mae = torch.mean(torch.abs(u_pred_all[0, :] - u_true_all[0, :])).item()
        bc_left_mae = torch.mean(torch.abs(u_pred_all[:, 0])).item()
        bc_right_mae = torch.mean(torch.abs(u_pred_all[:, -1])).item()
        bc_mse = (
            torch.mean(u_pred_all[:, 0] ** 2)
            + torch.mean(u_pred_all[:, -1] ** 2)
        ).item()

    # -----------------------------
    # 5) Print metrics
    # -----------------------------
    audit.export_evaluation(globals(), data_pack, (u2_all, u_pred_all))
    print("\n================ Evaluation ================")

    print("[u1 branch] pred vs true_u1")
    print(f"MAE  over domain = {metrics_u1['mae_domain']:.6e}")
    print(f"RMSE over domain = {metrics_u1['rmse_domain']:.6e}")
    print(f"MAXE over domain = {metrics_u1['maxe_domain']:.6e}")
    print(f"MAE  at t=1      = {metrics_u1['mae_t1']:.6e}")
    print(f"RMSE at t=1      = {metrics_u1['rmse_t1']:.6e}")
    print(f"MAXE at t=1      = {metrics_u1['maxe_t1']:.6e}")
    print()

    print("[u2 branch] pred vs true_u2")
    print(f"MAE  over domain = {metrics_u2['mae_domain']:.6e}")
    print(f"RMSE over domain = {metrics_u2['rmse_domain']:.6e}")
    print(f"MAXE over domain = {metrics_u2['maxe_domain']:.6e}")
    print(f"MAE  at t=1      = {metrics_u2['mae_t1']:.6e}")
    print(f"RMSE at t=1      = {metrics_u2['rmse_t1']:.6e}")
    print(f"MAXE at t=1      = {metrics_u2['maxe_t1']:.6e}")
    print()

    print("[FNO training target diagnostic] pred vs u2_target = u_true - u1_reference")
    print(f"MSE over domain  = {train_target_mse:.6e}")
    print(f"MAE over domain  = {train_target_mae:.6e}")
    print()

    print("[FNO IC/BC diagnostics for total u]")
    print(f"IC  MSE          = {ic_mse:.6e}")
    print(f"IC  MAE          = {ic_mae:.6e}")
    print(f"BC  MSE sum      = {bc_mse:.6e}")
    print(f"BC left  MAE     = {bc_left_mae:.6e}")
    print(f"BC right MAE     = {bc_right_mae:.6e}")
    print()

    print("[total solution] pred vs true_u")
    print(f"MAE  over domain = {metrics_u['mae_domain']:.6e}")
    print(f"RMSE over domain = {metrics_u['rmse_domain']:.6e}")
    print(f"MAXE over domain = {metrics_u['maxe_domain']:.6e}")
    print(f"MAE  at t=1      = {metrics_u['mae_t1']:.6e}")
    print(f"RMSE at t=1      = {metrics_u['rmse_t1']:.6e}")
    print(f"MAXE at t=1      = {metrics_u['maxe_t1']:.6e}")

    print("============================================\n")

    # -----------------------------
    # 6) Convert to numpy
    # -----------------------------
    x_np = x_grid.detach().cpu().numpy()
    t_np = t_right.detach().cpu().numpy()
    t_all_np = t_all.detach().cpu().numpy()

    u1_true_np = u1_true.detach().cpu().numpy()
    u1_pred_np = u1_pred.detach().cpu().numpy()

    u2_true_np = u2_true.detach().cpu().numpy()
    u2_pred_np = u2_pred.detach().cpu().numpy()

    u_true_np = u_true.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_pred_all_np = u_pred_all.detach().cpu().numpy()

    # -----------------------------
    # 7) Plot line comparison at t=1
    # -----------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x_np, u1_true_np[-1], label="true u1(x,1)")
    plt.plot(x_np, u1_pred_np[-1], "--", label="pred u1(x,1)")
    plt.xlabel("x")
    plt.ylabel("u1")
    plt.title(f"u1 comparison at t=1, alpha={ALPHA}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u1_compare_t1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_np, u2_true_np[-1], label="true u2(x,1)")
    plt.plot(x_np, u2_pred_np[-1], "--", label="pred u2(x,1)")
    plt.xlabel("x")
    plt.ylabel("u2")
    plt.title(f"u2 comparison at t=1, alpha={ALPHA}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u2_compare_t1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_np, u_true_np[-1], label="true u(x,1)")
    plt.plot(x_np, u_pred_np[-1], "--", label="pred u(x,1)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Total solution comparison at t=1, alpha={ALPHA}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u_compare_t1.png"), dpi=200)
    plt.close()

    # -----------------------------
    # 8) Plot abs error at t=1
    # -----------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x_np, np.abs(u1_pred_np[-1] - u1_true_np[-1]))
    plt.xlabel("x")
    plt.ylabel("abs error")
    plt.title("u1 absolute error at t=1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u1_abs_error_t1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_np, np.abs(u2_pred_np[-1] - u2_true_np[-1]))
    plt.xlabel("x")
    plt.ylabel("abs error")
    plt.title("u2 absolute error at t=1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u2_abs_error_t1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_np, np.abs(u_pred_np[-1] - u_true_np[-1]))
    plt.xlabel("x")
    plt.ylabel("abs error")
    plt.title("total solution absolute error at t=1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u_abs_error_t1.png"), dpi=200)
    plt.close()

    # -----------------------------
    # 9) Boundary values over t_all
    # -----------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(t_all_np, u_pred_all_np[:, 0], label="u_pred(0,t)")
    plt.plot(t_all_np, u_pred_all_np[:, -1], "--", label="u_pred(1,t)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("t")
    plt.ylabel("boundary value")
    plt.title("FNO IC/BC version: predicted boundary values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boundary_values.png"), dpi=200)
    plt.close()

    # -----------------------------
    # 10) Plot error heatmaps over full domain
    # -----------------------------
    extent = [x_np.min(), x_np.max(), t_np.min(), t_np.max()]

    plt.figure(figsize=(7, 4))
    plt.imshow(np.abs(u1_pred_np - u1_true_np), aspect="auto", origin="lower", extent=extent)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("u1 absolute error over domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u1_error_heatmap.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.imshow(np.abs(u2_pred_np - u2_true_np), aspect="auto", origin="lower", extent=extent)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("u2 absolute error over domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u2_error_heatmap.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.imshow(np.abs(u_pred_np - u_true_np), aspect="auto", origin="lower", extent=extent)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("total solution absolute error over domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u_error_heatmap.png"), dpi=200)
    plt.close()


def plot_losses(pinn_history, fno_history):
    plt.figure(figsize=(7, 4))
    plt.semilogy(pinn_history[:, 0], label="u1 total")
    plt.semilogy(pinn_history[:, 1], label="freq eq")
    plt.semilogy(pinn_history[:, 2], label="freq ic")
    plt.semilogy(pinn_history[:, 3], label="phys eq")
    plt.semilogy(pinn_history[:, 4], label="phys ic")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("u1 soft-scaled PINN losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "u1_losses.png"), dpi=200)
    plt.close()

    fno_history = np.asarray(fno_history)
    plt.figure(figsize=(7, 4))
    plt.semilogy(fno_history[:, 0], label="FNO total loss")
    plt.semilogy(fno_history[:, 1], label="normalized u2 loss")
    plt.semilogy(fno_history[:, 2], label="normalized g aux loss")
    plt.semilogy(fno_history[:, 6], label="normalized IC loss")
    plt.semilogy(fno_history[:, 7], label="normalized BC loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("FNO scaled+ICBC training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fno_scaled_icbc_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(fno_history[:, 3], label="raw u2 MSE")
    plt.semilogy(fno_history[:, 4], label="raw u2 MAE")
    plt.semilogy(fno_history[:, 5], label="raw total MAE")
    plt.semilogy(fno_history[:, 8], label="raw IC MSE")
    plt.semilogy(fno_history[:, 9], label="raw BC MSE sum")
    plt.xlabel("epoch")
    plt.ylabel("raw error")
    plt.title("FNO raw errors during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fno_raw_errors_icbc.png"), dpi=200)
    plt.close()

def plot_freq_ic_check(net):
    k = torch.linspace(0.0, K_MAX, 300, device=device)
    t = torch.zeros_like(k)

    with torch.no_grad():
        pred_real, pred_imag = pinn_complex_output(net, k, t)
        true_real, true_imag = u0_hat_torch(k)

    k_np = k.detach().cpu().numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(k_np, true_real.detach().cpu().numpy(), label="true Re u0_hat")
    plt.plot(k_np, pred_real.detach().cpu().numpy(), "--", label="pred Re u_hat(k,0)")
    plt.xlabel("k")
    plt.ylabel("real")
    plt.title("Frequency IC check: real part")
    plt.savefig(os.path.join(OUT_DIR, "frequency_ic_real.png"), dpi=200)
    plt.legend()
    plt.tight_layout()
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(k_np, true_imag.detach().cpu().numpy(), label="true Im u0_hat")
    plt.plot(k_np, pred_imag.detach().cpu().numpy(), "--", label="pred Im u_hat(k,0)")
    plt.xlabel("k")
    plt.ylabel("imag")
    plt.title("Frequency IC check: imaginary part")
    plt.savefig(os.path.join(OUT_DIR, "frequency_ic_imaginary.png"), dpi=200)
    plt.legend()
    plt.tight_layout()
    plt.close()

# ============================================================
# 10. Main
# ============================================================

def main():
    if W_PHYS_EQ != 0:
        raise ValueError("The legacy zero-extension GL residual is not valid for the full-space u1 branch; keep W_PHYS_EQ=0.")
    audit.run(globals(), ARGS)
    portfolio_io.finalize(Path(__file__).resolve().parent, Path(OUT_DIR))


if __name__ == "__main__":
    main()