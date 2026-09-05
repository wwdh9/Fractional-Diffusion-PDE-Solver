"""
dual_2d_final.py
2D dual-domain PINN+FNO for fractional diffusion (no FFT).
u = u1 (freq PINN, homogeneous) + u2 (FNO, forced)
All inverse Fourier transforms are done by numerical quadrature (vectorized).
"""

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
from torch.special import bessel_j0

# ------------------------------
# 0. Basic config
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

ALPHA = 0.4
C = 1.0
T_END = 1.0

# Space domain
X_MAX = 5.0
Y_MAX = 5.0
NX = 32          # reduced for speed
NY = 32

# Frequency domain (for quadrature)
K_MAX = 8.0
NK = 64                             # quadrature points per dimension

# Time
M = 40
DT = T_END / M

# Training
EPOCH_PINN = 3000
EPOCH_FNO = 3000
LR_PINN = 3e-4
LR_FNO = 1e-3
WEIGHT_DECAY = 1e-6

# Loss weights for u1
W_FREQ_EQ = 10.0
W_FREQ_IC = 100.0

# FNO scaling and losses
USE_FNO_SCALE = True
W_FNO_U2 = 1.0
W_FNO_G_AUX = 0.1
USE_FNO_IC_LOSS = True
W_FNO_IC = 0.1
USE_FNO_BC_LOSS = False
W_FNO_BC = 0.0

SCALE_EPS = 1e-8

# Output directory
OUT_DIR = "dual_2d_paper_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# 1. Exact solution and source term
# ------------------------------
def u0_2d(x, y):
    return torch.exp(-0.5 * (x**2 + y**2))

def u_exact_2d(x, y, t):
    return u0_2d(x, y) * (t + 1.0)

# Cache for h(r) to avoid repeated Bessel integration
_h_cache = {}
def compute_h_r(r, k_max=K_MAX, nk=1000):
    """Compute h(r) = ∫_0^∞ k^{α+1} e^{-k^2/2} J0(k r) dk via trapezoidal rule."""
    # Use caching: store for unique r values (rounded to avoid many keys)
    # Since r is a tensor, we'll compute unique values and map back.
    r_np = r.detach().cpu().numpy()
    # Use rounded values for caching (to avoid floating point mismatch)
    # We'll just use a simple dictionary with tuple keys (not ideal but okay)
    # For simplicity, we compute all at once without caching to keep code clean.
    # But we can cache if r has repeated values (e.g., grid points).
    # We'll implement a simple cache using rounded floats.
    pass

def compute_h_r_vectorized(r, k_max=K_MAX, nk=1000):
    """Vectorized computation of h(r) for all r."""
    k = torch.linspace(0, k_max, nk, device=r.device)
    # r: [N], k: [nk]
    # Use outer product: r.reshape(-1,1) * k.reshape(1,-1)
    # integrand: k^(α+1) * exp(-0.5 k^2) * J0(k*r)
    # We'll compute for all r at once.
    r = r.reshape(-1, 1)          # [N, 1]
    k = k.reshape(1, -1)          # [1, nk]
    kr = k * r                    # [N, nk]
    integrand = (k ** (ALPHA + 1)) * torch.exp(-0.5 * k**2) * bessel_j0(kr)
    h = torch.trapezoid(integrand, k, dim=1)  # [N]
    return h

def compute_f_2d(x, y, t):
    """Source term f(x,y,t) = u_t + c(-Δ)^{α/2}u."""
    r = torch.sqrt(x**2 + y**2)
    h = compute_h_r_vectorized(r)
    f = u0_2d(x, y) + C * (t + 1.0) * h
    return f

# ------------------------------
# 2. Fourier initial condition \hat u0
# ------------------------------
def u0_hat_2d(k1, k2):
    """Fourier transform of exp(-|x|^2/2) in 2D."""
    return 2.0 * math.pi * torch.exp(-0.5 * (k1**2 + k2**2))

# ------------------------------
# 3. 2D frequency-domain PINN model (u1 branch)
# ------------------------------
class FourierPINN2D(nn.Module):
    """
    Frequency-domain PINN for homogeneous fractional diffusion:
        d_t u_hat + C |k|^alpha u_hat = 0
    Input: (kx, ky, t) -> Output: Re(u_hat), Im(u_hat)
    """
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, kx, ky, t):
        inp = torch.stack([kx, ky, t], dim=-1)
        out = self.net(inp)
        return out[:, 0], out[:, 1]   # real, imag

def frequency_pde_loss_2d(model, n_samples=2048):
    """Residual: d_t u_hat + C |k|^alpha u_hat = 0"""
    kx = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    ky = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    t = torch.rand(n_samples, device=device) * T_END
    kx.requires_grad_(True)
    ky.requires_grad_(True)
    t.requires_grad_(True)

    real, imag = model(kx, ky, t)
    real_t = torch.autograd.grad(real.sum(), t, create_graph=True)[0]
    imag_t = torch.autograd.grad(imag.sum(), t, create_graph=True)[0]

    k_norm_alpha = (kx**2 + ky**2 + 1e-12) ** (ALPHA/2)
    res_real = real_t + C * k_norm_alpha * real
    res_imag = imag_t + C * k_norm_alpha * imag
    return torch.mean(res_real**2 + res_imag**2)

def frequency_ic_loss_2d(model, n_samples=2048):
    """Initial condition: u_hat(k,0) = 2π exp(-|k|^2/2)"""
    kx = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    ky = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    t = torch.zeros(n_samples, device=device)
    real, imag = model(kx, ky, t)
    target = u0_hat_2d(kx, ky)
    return torch.mean((real - target)**2 + imag**2)

# ------------------------------
# 4. Vectorized inverse Fourier transform (no FFT, no Python loops)
# ------------------------------
def inverse_fourier_2d(model, x, y, t):
    """
    Continuous inverse Fourier reconstruction by numerical quadrature.
    u(x,y,t)=1/(2π)^2 ∫∫ u_hat(kx,ky,t) exp(i(kx*x+ky*y)) dkx dky
    Fully vectorized using einsum.
    Input:
        x, y: 1D tensors (coordinates) or 2D meshes; if both 1D and same length, treated as point list.
        t: scalar time.
    Returns: 2D array if x,y are meshes, else 1D array.
    """
    # Determine input shape
    if x.dim() == 1 and y.dim() == 1 and x.shape == y.shape:
        # x and y are point lists
        xx = x
        yy = y
        H = 1
        W = x.numel()
        is_point_list = True
    else:
        # meshgrid case
        if x.dim() == 1 and y.dim() == 1:
            xx, yy = torch.meshgrid(x, y, indexing='ij')
        else:
            xx, yy = x, y
        H, W = xx.shape
        is_point_list = False

    # Flatten spatial points
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    P = x_flat.numel()

    # Frequency grid
    k = torch.linspace(-K_MAX, K_MAX, NK, device=device)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    kx_flat = kx.reshape(-1)
    ky_flat = ky.reshape(-1)

    # Evaluate model on frequency grid
    if torch.is_tensor(t):
        t_val = t.item()
    else:
        t_val = t
    t_flat = torch.ones_like(kx_flat) * t_val
    real, imag = model(kx_flat, ky_flat, t_flat)
    u_hat = torch.complex(real, imag).reshape(NK, NK)   # [NK, NK]

    # Precompute exponential matrices: exp(1j * k * x) and exp(1j * k * y)
    # k shape: [NK], x_flat shape: [P]
    exp_x = torch.exp(1j * k.reshape(NK, 1) * x_flat.reshape(1, -1))  # [NK, P]
    exp_y = torch.exp(1j * k.reshape(NK, 1) * y_flat.reshape(1, -1))  # [NK, P]

    # Integrate: sum over ky then over kx
    temp = torch.einsum('ij,jp->ip', u_hat, exp_y)   # [NK, P]
    result = torch.einsum('ip,ip->p', temp, exp_x)   # [P]

    dk = (2*K_MAX) / (NK - 1)
    result = result * (dk**2) / (2*math.pi)**2
    result_real = result.real

    if is_point_list:
        return result_real
    else:
        return result_real.reshape(H, W)

def inverse_fourier_exact_u1_2d(x, y, t):
    """Exact u1 from analytical Fourier transform (for reference)."""
    # Determine shape
    if x.dim() == 1 and y.dim() == 1 and x.shape == y.shape:
        xx = x
        yy = y
        is_point_list = True
        H, W = 1, x.numel()
    else:
        if x.dim() == 1 and y.dim() == 1:
            xx, yy = torch.meshgrid(x, y, indexing='ij')
        else:
            xx, yy = x, y
        H, W = xx.shape
        is_point_list = False

    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    P = x_flat.numel()

    k = torch.linspace(-K_MAX, K_MAX, NK, device=device)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    # Analytical u_hat: exp(-C |k|^alpha t) * 2π exp(-|k|^2/2)
    k_norm = torch.sqrt(kx**2 + ky**2)
    u_hat = torch.exp(-C * (k_norm**ALPHA) * t) * (2*math.pi * torch.exp(-0.5*k_norm**2))
    # u_hat is real, but we treat as complex
    u_hat = u_hat.to(torch.cfloat)

    exp_x = torch.exp(1j * k.reshape(NK, 1) * x_flat.reshape(1, -1))
    exp_y = torch.exp(1j * k.reshape(NK, 1) * y_flat.reshape(1, -1))
    temp = torch.einsum('ij,jp->ip', u_hat, exp_y)
    result = torch.einsum('ip,ip->p', temp, exp_x)
    dk = (2*K_MAX)/(NK-1)
    result = result * (dk**2) / (2*math.pi)**2
    result_real = result.real

    if is_point_list:
        return result_real
    else:
        return result_real.reshape(H, W)

# ------------------------------
# 5. 2D FNO model (no torch.fft)
# ------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W)

        xx = torch.arange(H, device=device)
        yy = torch.arange(W, device=device)
        kx = torch.arange(modes1, device=device)
        ky = torch.arange(modes2, device=device)

        E1 = torch.exp(-2j * math.pi * torch.outer(kx, xx) / H)
        E2 = torch.exp(-2j * math.pi * torch.outer(ky, yy) / W)

        x_ft = torch.einsum('bchw,mh,nw->bcmn', x.to(torch.cfloat), E1, E2)
        out_ft = torch.einsum('bcmn,comn->bomn', x_ft, self.weights1[:, :, :modes1, :modes2])

        Ei1 = torch.conj(E1) / H
        Ei2 = torch.conj(E2) / W
        out = torch.einsum('bomn,mh,nw->bohw', out_ft, Ei1, Ei2).real
        return out

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64):
        super().__init__()
        self.fc0 = nn.Linear(4, width)   # input: (f, x, y, t)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, f_input):
        # f_input: [batch, H, W, 1]
        batch, H, W = f_input.shape[:3]
        x = torch.linspace(-X_MAX, X_MAX, H, device=f_input.device).reshape(1, H, 1, 1).expand(batch, H, W, 1)
        y = torch.linspace(-Y_MAX, Y_MAX, W, device=f_input.device).reshape(1, 1, W, 1).expand(batch, H, W, 1)
        t = torch.linspace(0, T_END, batch, device=f_input.device).reshape(batch, 1, 1, 1).expand(batch, H, W, 1)
        inp = torch.cat([f_input, x, y, t], dim=-1)  # [batch,H,W,4]
        x = self.fc0(inp)          # [batch,H,W,width]
        x = x.permute(0, 3, 1, 2)  # [batch,width,H,W]
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = x.permute(0, 2, 3, 1)  # [batch,H,W,width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)            # [batch,H,W,1]
        return x

# ------------------------------
# 6. Data generation and FNO training
# ------------------------------
def build_training_grids_2d(model_u1):
    x_grid = torch.linspace(-X_MAX, X_MAX, NX, device=device)
    y_grid = torch.linspace(-Y_MAX, Y_MAX, NY, device=device)
    t_all = torch.linspace(0, T_END, M+1, device=device)

    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    # Precompute f for all left time steps
    f_all = []
    for m in range(M):
        t = t_all[m]
        f_val = compute_f_2d(xx_flat, yy_flat, t)
        f_all.append(f_val.reshape(NX, NY))
    f_train = torch.stack(f_all, dim=0)   # [M, NX, NY]

    # True solution at right times
    u_true_all = []
    for m in range(M+1):
        t = t_all[m]
        u_val = u_exact_2d(xx_flat, yy_flat, t)
        u_true_all.append(u_val.reshape(NX, NY))
    u_true_all = torch.stack(u_true_all, dim=0)  # [M+1, NX, NY]
    u_true = u_true_all[1:]   # [M, NX, NY]

    # u1 prediction at all times (using vectorized inverse)
    u1_all = []
    for m in range(M+1):
        t = t_all[m]
        with torch.no_grad():
            # Pass the flattened points; the inverse function returns 1D array
            u1_flat = inverse_fourier_2d(model_u1, xx_flat, yy_flat, t)
            # Check if result is 1D, then reshape
            u1_all.append(u1_flat.reshape(NX, NY))
    u1_all = torch.stack(u1_all, dim=0)  # [M+1, NX, NY]
    u1_pred = u1_all[1:]  # [M, NX, NY]

    return x_grid, y_grid, t_all, f_train, u_true, u_true_all, u1_pred, u1_all

def train_u1_branch():
    model = FourierPINN2D(width=64, depth=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PINN, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_PINN, eta_min=1e-5)
    history = []

    for ep in range(1, EPOCH_PINN+1):
        optimizer.zero_grad()
        lf_eq = frequency_pde_loss_2d(model)
        lf_ic = frequency_ic_loss_2d(model)
        loss = W_FREQ_EQ * lf_eq + W_FREQ_IC * lf_ic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if ep % 100 == 0:
            print(f"[u1 PINN] ep={ep:5d} loss={loss.item():.3e} eq={lf_eq.item():.3e} ic={lf_ic.item():.3e}")
        history.append([loss.item(), lf_eq.item(), lf_ic.item()])
    return model, np.array(history)

def train_fno_branch(model_u1):
    x_grid, y_grid, t_all, f_train, u_true, u_true_all, u1_pred, u1_all = build_training_grids_2d(model_u1)

    model = FNO2d(modes1=12, modes2=12, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FNO, weight_decay=WEIGHT_DECAY)

    # Targets for u2
    u2_target = u_true - u1_pred  # [M, NX, NY]
    u2_prev = torch.cat([torch.zeros(1, NX, NY, device=device), u2_target[:-1]], dim=0)
    g_target = (u2_target - u2_prev) / DT  # [M, NX, NY]

    # Scales
    f_scale = torch.mean(torch.abs(f_train)).clamp_min(SCALE_EPS).detach()
    u2_scale = torch.mean(torch.abs(u2_target)).clamp_min(SCALE_EPS).detach()
    g_scale = torch.mean(torch.abs(g_target)).clamp_min(SCALE_EPS).detach()
    u_full_scale = torch.mean(torch.abs(u_true_all)).clamp_min(SCALE_EPS).detach()

    f_train_in = f_train / f_scale if USE_FNO_SCALE else f_train
    f_train_in = f_train_in.unsqueeze(-1)  # [M, NX, NY, 1]

    history = []
    for ep in range(1, EPOCH_FNO+1):
        optimizer.zero_grad()
        g_norm = model(f_train_in).squeeze(-1)  # [M, NX, NY]
        if USE_FNO_SCALE:
            g = g_scale * g_norm
        else:
            g = g_norm
        u2_pred = torch.cumsum(g, dim=0) * DT

        u_pred = u1_pred + u2_pred
        u2_all = torch.cat([torch.zeros(1, NX, NY, device=device), u2_pred], dim=0)
        u_pred_all = u1_all + u2_all

        if USE_FNO_SCALE:
            loss_u2 = torch.mean(((u2_pred - u2_target) / u2_scale)**2)
            loss_g = torch.mean((g_norm - g_target / g_scale)**2)
        else:
            loss_u2 = torch.mean((u2_pred - u2_target)**2)
            loss_g = torch.mean((g - g_target)**2)

        loss_ic = torch.zeros(1, device=device)
        if USE_FNO_IC_LOSS:
            loss_ic = torch.mean(((u_pred_all[0] - u_true_all[0]) / u_full_scale)**2)

        loss_bc = torch.zeros(1, device=device)
        loss = W_FNO_U2 * loss_u2 + W_FNO_G_AUX * loss_g + W_FNO_IC * loss_ic + W_FNO_BC * loss_bc
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"[FNO] ep={ep:5d} loss={loss.item():.3e} u2={loss_u2.item():.3e} g={loss_g.item():.3e} ic={loss_ic.item():.3e}")
        history.append([loss.item(), loss_u2.item(), loss_g.item(), loss_ic.item(), loss_bc.item()])

    data_pack = {
        'x_grid': x_grid, 'y_grid': y_grid, 't_all': t_all,
        'f_train': f_train, 'u_true': u_true, 'u_true_all': u_true_all,
        'u1_pred': u1_pred, 'u1_all': u1_all, 'u2_target': u2_target,
        'f_scale': f_scale, 'u2_scale': u2_scale, 'g_scale': g_scale,
        'u_full_scale': u_full_scale
    }
    return model, np.array(history), data_pack

# ------------------------------
# 7. Evaluation and plotting
# ------------------------------
def evaluate_2d(model_u1, model_fno, data_pack, hist_u1=None, hist_fno=None):
    xg = data_pack['x_grid']
    yg = data_pack['y_grid']
    t_all = data_pack['t_all']
    u_true_all = data_pack['u_true_all']
    u1_all = data_pack['u1_all']
    f_scale = data_pack['f_scale']

    # Final prediction
    with torch.no_grad():
        f_train_in = data_pack['f_train'].unsqueeze(-1)
        if USE_FNO_SCALE:
            f_train_in = f_train_in / f_scale
        g_norm = model_fno(f_train_in).squeeze(-1)
        if USE_FNO_SCALE:
            g = data_pack['g_scale'] * g_norm
        else:
            g = g_norm
        u2_pred = torch.cumsum(g, dim=0) * DT
        u2_all = torch.cat([torch.zeros(1, NX, NY, device=device), u2_pred], dim=0)
        u_pred_all = u1_all + u2_all

    # Exact u1 for reference
    xx, yy = torch.meshgrid(xg, yg, indexing='ij')
    u1_exact_all = []
    for tt in t_all:
        u1_exact_all.append(inverse_fourier_exact_u1_2d(xx.reshape(-1), yy.reshape(-1), tt).reshape(NX, NY))
    u1_exact_all = torch.stack(u1_exact_all, dim=0)

    err_u1 = u1_all - u1_exact_all
    err_u = u_pred_all - u_true_all

    # ==============================
    # Detailed evaluation table
    # ==============================

    def calc_metrics(err):
        mae = torch.mean(torch.abs(err)).item()
        rmse = torch.sqrt(torch.mean(err ** 2)).item()
        maxe = torch.max(torch.abs(err)).item()
        return mae, rmse, maxe

    print("\n================ Evaluation ================")

    # ---------- u1 ----------
    err_u1 = u1_all - u1_exact_all

    mae, rmse, maxe = calc_metrics(err_u1)

    mae_t1, rmse_t1, maxe_t1 = calc_metrics(err_u1[-1])

    print("[u1 branch] pred vs true_u1")
    print(f"MAE  over domain = {mae:.6e}")
    print(f"RMSE over domain = {rmse:.6e}")
    print(f"MAXE over domain = {maxe:.6e}")
    print(f"MAE  at t=1      = {mae_t1:.6e}")
    print(f"RMSE at t=1      = {rmse_t1:.6e}")
    print(f"MAXE at t=1      = {maxe_t1:.6e}")
    print()

    # ---------- u2 ----------
    u2_exact_all = u_true_all - u1_exact_all
    u2_pred_all = u_pred_all - u1_all

    err_u2 = u2_pred_all - u2_exact_all

    mae, rmse, maxe = calc_metrics(err_u2)

    mae_t1, rmse_t1, maxe_t1 = calc_metrics(err_u2[-1])

    print("[u2 branch] pred vs true_u2")
    print(f"MAE  over domain = {mae:.6e}")
    print(f"RMSE over domain = {rmse:.6e}")
    print(f"MAXE over domain = {maxe:.6e}")
    print(f"MAE  at t=1      = {mae_t1:.6e}")
    print(f"RMSE at t=1      = {rmse_t1:.6e}")
    print(f"MAXE at t=1      = {maxe_t1:.6e}")
    print()

    # ---------- FNO target diagnostic ----------
    u2_target = data_pack["u2_target"]

    err_target = u2_pred_all[1:] - u2_target

    mse_target = torch.mean(err_target ** 2).item()
    mae_target = torch.mean(torch.abs(err_target)).item()

    print("[FNO training target diagnostic] pred vs u2_target = u_true - u1_pred")
    print(f"MSE over domain  = {mse_target:.6e}")
    print(f"MAE over domain  = {mae_target:.6e}")
    print()

    # ---------- Total solution ----------
    err_total = u_pred_all - u_true_all

    mae, rmse, maxe = calc_metrics(err_total)

    mae_t1, rmse_t1, maxe_t1 = calc_metrics(err_total[-1])

    print("[total solution] pred vs true_u")
    print(f"MAE  over domain = {mae:.6e}")
    print(f"RMSE over domain = {rmse:.6e}")
    print(f"MAXE over domain = {maxe:.6e}")
    print(f"MAE  at t=1      = {mae_t1:.6e}")
    print(f"RMSE at t=1      = {rmse_t1:.6e}")
    print(f"MAXE at t=1      = {maxe_t1:.6e}")

    print("============================================\n")

    mae = torch.mean(torch.abs(err_u)).item()
    rmse = torch.sqrt(torch.mean(err_u**2)).item()
    print(f"MAE over full spatiotemporal domain: {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")

    # Plot t=1 slice
    idx_t = M
    u_pred_t1 = u_pred_all[idx_t].cpu().numpy()
    u_true_t1 = u_true_all[idx_t].cpu().numpy()
    x_np = xg.cpu().numpy()
    y_np = yg.cpu().numpy()

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.imshow(u_true_t1, extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()], origin='lower')
    plt.title('True u at t=1')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(u_pred_t1, extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()], origin='lower')
    plt.title('Predicted u at t=1')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(np.abs(u_pred_t1 - u_true_t1), extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()], origin='lower')
    plt.title('Absolute error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '2d_result_t1.png'), dpi=200)
    plt.close()

    # u1 validation
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(u1_exact_all[-1].cpu(), origin='lower')
    plt.title('Exact u1 t=1')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(u1_all[-1].cpu(), origin='lower')
    plt.title('PINN u1 t=1')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(torch.abs(err_u1[-1]).cpu(), origin='lower')
    plt.title('u1 error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'u1_validation.png'), dpi=200)
    plt.close()

    # u2 validation
    u2_exact_all = u_true_all - u1_exact_all
    u2_all = u_pred_all - u1_all   # actually u2_all from FNO
    err_u2 = u2_all - u2_exact_all
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(u2_exact_all[-1].cpu(), origin='lower')
    plt.title('Exact u2 t=1')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(u2_all[-1].cpu(), origin='lower')
    plt.title('FNO u2 t=1')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(torch.abs(err_u2[-1]).cpu(), origin='lower')
    plt.title('u2 error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'u2_validation.png'), dpi=200)
    plt.close()

    # error evolution
    plt.figure(figsize=(6,4))
    plt.semilogy(torch.mean(torch.abs(err_u), dim=(1,2)).cpu())
    plt.title('Space error evolution')
    plt.xlabel('time step')
    plt.ylabel('MAE')
    plt.grid()
    plt.savefig(os.path.join(OUT_DIR,'error_time.png'), dpi=200)
    plt.close()

    # loss curves
    if hist_u1 is not None:
        plt.figure(figsize=(6,4))
        plt.semilogy(hist_u1[:,0], label='total')
        plt.semilogy(hist_u1[:,1], label='equation')
        plt.semilogy(hist_u1[:,2], label='initial')
        plt.legend()
        plt.title('u1 PINN loss')
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR,'u1_losses.png'), dpi=200)
        plt.close()

    if hist_fno is not None:
        plt.figure(figsize=(6,4))
        plt.semilogy(hist_fno[:,0], label='total')
        plt.semilogy(hist_fno[:,1], label='u2')
        plt.semilogy(hist_fno[:,2], label='gradient')
        plt.legend()
        plt.title('FNO loss')
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR,'fno_losses.png'), dpi=200)
        plt.close()



# ------------------------------
# 8. Main
# ------------------------------
def main():
    print("Training u1 PINN (2D frequency domain)...")
    model_u1, hist_u1 = train_u1_branch()
    print("Training FNO (2D)...")
    model_fno, hist_fno, data_pack = train_fno_branch(model_u1)
    evaluate_2d(model_u1, model_fno, data_pack, hist_u1, hist_fno)

    torch.save(model_u1.state_dict(), os.path.join(OUT_DIR, 'u1_2d_pinn.pt'))
    torch.save(model_fno.state_dict(), os.path.join(OUT_DIR, 'fno_2d.pt'))
    print("All done. Results saved to", OUT_DIR)

if __name__ == "__main__":
    main()