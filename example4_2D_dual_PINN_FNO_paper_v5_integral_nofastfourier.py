"""
dual_domain_2d_fno_scaled.py
2D extension of dual-domain PINN+FNO for fractional diffusion.
u = u1 (freq PINN, homogeneous) + u2 (FNO, forced)
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
from torch.special import bessel_j0  # requires torch >= 1.8

# ------------------------------
# 0. Basic config
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

ALPHA = 1.6
C = 1.0
T_END = 1.0

# Space domain
X_MAX = 5.0
Y_MAX = 5.0
NX = 64
NY = 64

# Frequency domain
K_MAX = 8.0
NK1 = 64
NK2 = 64

# Time
M = 80
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
# No physical loss for u1 (2D GL is complex; frequency loss is sufficient)

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

def compute_h_r(r, k_max=K_MAX, nk=1000):
    """Compute h(r) = ∫_0^∞ k^{α+1} e^{-k^2/2} J0(k r) dk via trapezoidal rule."""
    k = torch.linspace(0, k_max, nk, device=r.device)
    k = k.reshape(1, -1)  # [1, nk]
    r = r.reshape(-1, 1)  # [num_points, 1]
    # integrand: k^{α+1} * exp(-k^2/2) * J0(k*r)
    integrand = (k ** (ALPHA + 1)) * torch.exp(-0.5 * k**2) * bessel_j0(k * r)
    h = torch.trapezoid(integrand, k, dim=1)  # [num_points]
    return h

def compute_f_2d(x, y, t):
    """Source term f(x,y,t) = u_t + c(-Δ)^{α/2}u."""
    r = torch.sqrt(x**2 + y**2)
    # Precompute h(r) for all unique r values (caching)
    # Here we assume r is a flattened tensor; we'll compute on the fly.
    # In practice, cache h for unique r to save time.
    h = compute_h_r(r)  # [num_points]
    f = u0_2d(x, y) + C * (t + 1.0) * h
    return f

# ------------------------------
# 2. Fourier initial condition \hat u0
# ------------------------------
def u0_hat_2d(k1, k2):
    """Fourier transform of exp(-|x|^2/2) in 2D."""
    return 2.0 * math.pi * torch.exp(-0.5 * (k1**2 + k2**2))


def inverse_fourier_exact_u1_2d(x, y, t):
    """
    Continuous inverse Fourier reconstruction for the exact u1.

    u1_hat(k,t)=exp(-C|k|^alpha t) * 2*pi*exp(-|k|^2/2)

    The integral is evaluated by trapezoidal quadrature.
    """

    NK = 128
    k = torch.linspace(-K_MAX, K_MAX, NK, device=device)

    k1, k2 = torch.meshgrid(k, k, indexing="ij")
    k_norm = torch.sqrt(k1**2 + k2**2)

    u_hat = torch.exp(-C * (k_norm ** ALPHA) * t) * (
        2 * math.pi * torch.exp(-0.5 * k_norm**2)
    )

    # x,y mesh
    if x.dim() == 1 and y.dim() == 1:
        xx, yy = torch.meshgrid(x, y, indexing="ij")
    else:
        xx, yy = x, y

    result = torch.zeros_like(xx, dtype=torch.cfloat)

    # inverse Fourier integral
    for i in range(NK):
        phase_x = torch.exp(1j * k[i] * xx)
        for j in range(NK):
            phase = phase_x * torch.exp(1j * k[j] * yy)
            result += u_hat[i, j] * phase

    dk = (2 * K_MAX) / (NK - 1)

    result = result * (dk ** 2) / (2 * math.pi) ** 2

    return result.real



def inverse_fourier_2d(model, x, y, t):
    """
    Continuous inverse Fourier reconstruction for PINN predicted spectrum.

    The PINN outputs:
        (kx,ky,t) -> Re(u_hat), Im(u_hat)

    Then:
        u(x,y,t)=1/(2pi)^2 integral u_hat(k,t)exp(i k.x) dk

    No FFT is used. The same quadrature convention is used
    for comparison with the exact solution.
    """

    NK = 128
    k = torch.linspace(-K_MAX, K_MAX, NK, device=device)

    k1, k2 = torch.meshgrid(k, k, indexing="ij")

    k1_flat = k1.reshape(-1)
    k2_flat = k2.reshape(-1)

    # fix scalar t bug
    if torch.is_tensor(t):
        t_value = t.item()
    else:
        t_value = t

    t_flat = torch.ones_like(k1_flat) * t_value

    real, imag = model(k1_flat, k2_flat, t_flat)

    u_hat = torch.complex(real, imag).reshape(NK, NK)

    if x.dim() == 1 and y.dim() == 1:
        xx, yy = torch.meshgrid(x, y, indexing="ij")
    else:
        xx, yy = x, y

    result = torch.zeros_like(xx, dtype=torch.cfloat)

    for i in range(NK):
        phase_x = torch.exp(1j * k[i] * xx)
        for j in range(NK):
            phase = phase_x * torch.exp(1j * k[j] * yy)
            result += u_hat[i,j] * phase

    dk = (2*K_MAX)/(NK-1)

    result = result * (dk**2)/(2*math.pi)**2

    return result.real



# ------------------------------
# 5. 2D FNO model
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
        # x: [batch, channels, H, W]
        batchsize = x.shape[0]
        # Numerical Fourier transform by quadrature (no torch.fft)
        B, C, H, W = x.shape
        device = x.device
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W)

        xx = torch.arange(H, device=device)
        yy = torch.arange(W, device=device)
        kx = torch.arange(modes1, device=device)
        ky = torch.arange(modes2, device=device)

        E1 = torch.exp(-2j*math.pi*torch.outer(kx, xx)/H)
        E2 = torch.exp(-2j*math.pi*torch.outer(ky, yy)/W)

        x_ft = torch.einsum('bchw,mh,nw->bcmn', x, E1, E2)
        out_ft = torch.zeros(B, self.out_channels, modes1, modes2, dtype=torch.cfloat, device=device)
        out_ft = torch.einsum('bcmn,comn->bomn', x_ft, self.weights1[:,:,:modes1,:modes2])

        Ei1 = torch.conj(E1) / H
        Ei2 = torch.conj(E2) / W
        out = torch.einsum('bomn,mh,nw->bohw', out_ft, Ei1, Ei2).real
        return out

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64):
        super().__init__()
        self.fc0 = nn.Linear(4, width)  # input: (f, x, y, t)
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
        t = torch.linspace(0, T_END, batch, device=f_input.device).reshape(batch,1,1,1).expand(batch,H,W,1)
        inp = torch.cat([f_input, x, y, t], dim=-1)  # [batch,H,W,4]
        x = self.fc0(inp)  # [batch, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, H, W]
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # [batch, H, W, 1]
        return x

# ------------------------------
# 6. Data generation and FNO training
# ------------------------------
def build_training_grids_2d(model_u1):
    # Space-time grids
    x_grid = torch.linspace(-X_MAX, X_MAX, NX, device=device)
    y_grid = torch.linspace(-Y_MAX, Y_MAX, NY, device=device)
    t_all = torch.linspace(0, T_END, M+1, device=device)

    # Create meshgrid for all time steps
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    # Precompute f for all time steps (left time points for u2 accumulation)
    f_all = []
    for m in range(M):
        t = t_all[m]
        f_val = compute_f_2d(xx_flat, yy_flat, t)
        f_all.append(f_val.reshape(NX, NY))
    f_train = torch.stack(f_all, dim=0)  # [M, NX, NY]

    # True solution at right time points for u2 training
    u_true_all = []
    for m in range(M+1):
        t = t_all[m]
        u_val = u_exact_2d(xx_flat, yy_flat, t)
        u_true_all.append(u_val.reshape(NX, NY))
    u_true_all = torch.stack(u_true_all, dim=0)  # [M+1, NX, NY]
    u_true = u_true_all[1:]  # [M, NX, NY]

    # u1 prediction at right times and all times
    u1_pred_all = []
    for m in range(M+1):
        t = t_all[m]
        with torch.no_grad():
            u1_flat = inverse_fourier_2d(model_u1, xx_flat, yy_flat, t)
            u1_pred_all.append(u1_flat.reshape(NX, NY))
    u1_pred_all = torch.stack(u1_pred_all, dim=0)  # [M+1, NX, NY]
    u1_pred = u1_pred_all[1:]  # [M, NX, NY]
    u1_all = u1_pred_all  # [M+1, NX, NY]

    return x_grid, y_grid, t_all, f_train, u_true, u_true_all, u1_pred, u1_all

def train_u1_branch():
    model = FourierPINN2D(width=64, depth=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PINN, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_PINN, eta_min=1e-5)
    history = []

    for ep in range(1, EPOCH_PINN + 1):
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


# ------------------------------
# 3. 2D frequency-domain PINN model (u1 branch)
# ------------------------------
class FourierPINN2D(nn.Module):
    """
    Frequency-domain PINN for homogeneous fractional diffusion:
        d_t u_hat + C |k|^alpha u_hat = 0

    Input:
        (kx, ky, t)
    Output:
        Re(u_hat), Im(u_hat)
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
        return out[:, 0:1].squeeze(-1), out[:, 1:2].squeeze(-1)


def frequency_pde_loss_2d(model, n_samples=4096):
    """
    Residual:
        d_t u_hat + C |k|^alpha u_hat = 0
    """
    kx = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    ky = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    t = torch.rand(n_samples, device=device) * T_END

    kx.requires_grad_(True)
    ky.requires_grad_(True)
    t.requires_grad_(True)

    real, imag = model(kx, ky, t)

    real_t = torch.autograd.grad(
        real, t, torch.ones_like(real), create_graph=True
    )[0]
    imag_t = torch.autograd.grad(
        imag, t, torch.ones_like(imag), create_graph=True
    )[0]

    k_norm_alpha = (kx ** 2 + ky ** 2 + 1e-12) ** (ALPHA / 2)

    res_real = real_t + C * k_norm_alpha * real
    res_imag = imag_t + C * k_norm_alpha * imag

    return torch.mean(res_real ** 2 + res_imag ** 2)


def frequency_ic_loss_2d(model, n_samples=4096):
    """
    Initial condition:
        u_hat(k,0)=2*pi*exp(-|k|^2/2)
    """
    kx = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    ky = (2 * torch.rand(n_samples, device=device) - 1) * K_MAX
    t = torch.zeros(n_samples, device=device)

    real, imag = model(kx, ky, t)

    target = u0_hat_2d(kx, ky)

    return torch.mean((real - target) ** 2 + imag ** 2)


# ------------------------------
# 4. 2D inverse Fourier transform for u1
# ------------------------------
def inverse_fourier_2d(model, x, y, t):
    """
    Continuous inverse Fourier reconstruction by numerical quadrature.

    u(x,y,t)=1/(2pi)^2 ∫∫ u_hat(kx,ky,t)exp(i(kx*x+ky*y))dkx dky

    No FFT/ifft is used.
    """
    if x.dim() == 1 and y.dim() == 1:
        xx, yy = torch.meshgrid(x, y, indexing="ij")
    else:
        xx, yy = x, y

    NK = 128
    k = torch.linspace(-K_MAX, K_MAX, NK, device=device)
    k1, k2 = torch.meshgrid(k, k, indexing="ij")

    k1_flat = k1.reshape(-1)
    k2_flat = k2.reshape(-1)

    if torch.is_tensor(t):
        t_value = t.item()
    else:
        t_value = t
    t_flat = torch.ones_like(k1_flat) * t_value

    real, imag = model(k1_flat, k2_flat, t_flat)
    u_hat = torch.complex(real, imag).reshape(NK, NK)

    result = torch.zeros_like(xx, dtype=torch.cfloat)
    for i in range(NK):
        for j in range(NK):
            result += u_hat[i,j] * torch.exp(1j*(k[i]*xx+k[j]*yy))

    dk = (2*K_MAX)/(NK-1)
    result = result * (dk**2)/(2*math.pi)**2
    return result.real

# ------------------------------
# 5. 2D FNO model
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
        # x: [batch, channels, H, W]
        batchsize = x.shape[0]
        # Numerical Fourier transform by quadrature (no torch.fft)
        B, C, H, W = x.shape
        device = x.device
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W)

        xx = torch.arange(H, device=device)
        yy = torch.arange(W, device=device)
        kx = torch.arange(modes1, device=device)
        ky = torch.arange(modes2, device=device)

        E1 = torch.exp(-2j*math.pi*torch.outer(kx, xx)/H)
        E2 = torch.exp(-2j*math.pi*torch.outer(ky, yy)/W)

        x_ft = torch.einsum('bchw,mh,nw->bcmn', x, E1, E2)
        out_ft = torch.zeros(B, self.out_channels, modes1, modes2, dtype=torch.cfloat, device=device)
        out_ft = torch.einsum('bcmn,comn->bomn', x_ft, self.weights1[:,:,:modes1,:modes2])

        Ei1 = torch.conj(E1) / H
        Ei2 = torch.conj(E2) / W
        out = torch.einsum('bomn,mh,nw->bohw', out_ft, Ei1, Ei2).real
        return out

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=64):
        super().__init__()
        self.fc0 = nn.Linear(4, width)  # input: (f, x, y, t)
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
        t = torch.linspace(0, T_END, batch, device=f_input.device).reshape(batch,1,1,1).expand(batch,H,W,1)
        inp = torch.cat([f_input, x, y, t], dim=-1)  # [batch,H,W,4]
        x = self.fc0(inp)  # [batch, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, H, W]
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # [batch, H, W, 1]
        return x

# ------------------------------
# 6. Data generation and FNO training
# ------------------------------
def build_training_grids_2d(model_u1):
    # Space-time grids
    x_grid = torch.linspace(-X_MAX, X_MAX, NX, device=device)
    y_grid = torch.linspace(-Y_MAX, Y_MAX, NY, device=device)
    t_all = torch.linspace(0, T_END, M+1, device=device)

    # Create meshgrid for all time steps
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    # Precompute f for all time steps (left time points for u2 accumulation)
    f_all = []
    for m in range(M):
        t = t_all[m]
        f_val = compute_f_2d(xx_flat, yy_flat, t)
        f_all.append(f_val.reshape(NX, NY))
    f_train = torch.stack(f_all, dim=0)  # [M, NX, NY]

    # True solution at right time points for u2 training
    u_true_all = []
    for m in range(M+1):
        t = t_all[m]
        u_val = u_exact_2d(xx_flat, yy_flat, t)
        u_true_all.append(u_val.reshape(NX, NY))
    u_true_all = torch.stack(u_true_all, dim=0)  # [M+1, NX, NY]
    u_true = u_true_all[1:]  # [M, NX, NY]

    # u1 prediction at right times and all times
    u1_pred_all = []
    for m in range(M+1):
        t = t_all[m]
        with torch.no_grad():
            u1_flat = inverse_fourier_2d(model_u1, xx_flat, yy_flat, t)
            u1_pred_all.append(u1_flat.reshape(NX, NY))
    u1_pred_all = torch.stack(u1_pred_all, dim=0)  # [M+1, NX, NY]
    u1_pred = u1_pred_all[1:]  # [M, NX, NY]
    u1_all = u1_pred_all  # [M+1, NX, NY]

    return x_grid, y_grid, t_all, f_train, u_true, u_true_all, u1_pred, u1_all


def train_fno_branch(model_u1):
    x_grid, y_grid, t_all, f_train, u_true, u_true_all, u1_pred, u1_all = build_training_grids_2d(model_u1)

    model = FNO2d(modes1=12, modes2=12, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FNO, weight_decay=WEIGHT_DECAY)

    # Targets
    u2_target = u_true - u1_pred  # [M, NX, NY]
    # g_target = (u2_target_m - u2_target_{m-1}) / dt, with u2(0)=0
    u2_prev = torch.cat([torch.zeros(1, NX, NY, device=device), u2_target[:-1]], dim=0)
    g_target = (u2_target - u2_prev) / DT  # [M, NX, NY]

    # Scales
    f_scale = torch.mean(torch.abs(f_train)).clamp_min(SCALE_EPS).detach()
    u2_scale = torch.mean(torch.abs(u2_target)).clamp_min(SCALE_EPS).detach()
    g_scale = torch.mean(torch.abs(g_target)).clamp_min(SCALE_EPS).detach()
    u_full_scale = torch.mean(torch.abs(u_true_all)).clamp_min(SCALE_EPS).detach()

    f_train_in = f_train / f_scale if USE_FNO_SCALE else f_train
    # Add channel dimension: [M, NX, NY, 1]
    f_train_in = f_train_in.unsqueeze(-1)

    history = []
    for ep in range(1, EPOCH_FNO + 1):
        optimizer.zero_grad()
        g_norm = model(f_train_in).squeeze(-1)  # [M, NX, NY]
        if USE_FNO_SCALE:
            g = g_scale * g_norm
        else:
            g = g_norm
        u2_pred = torch.cumsum(g, dim=0) * DT

        # Total prediction on right times
        u_pred = u1_pred + u2_pred
        # Total on all times (including t=0)
        u2_all = torch.cat([torch.zeros(1, NX, NY, device=device), u2_pred], dim=0)
        u_pred_all = u1_all + u2_all

        # Losses
        if USE_FNO_SCALE:
            loss_u2 = torch.mean(((u2_pred - u2_target) / u2_scale)**2)
            loss_g = torch.mean((g_norm - g_target / g_scale)**2)
        else:
            loss_u2 = torch.mean((u2_pred - u2_target)**2)
            loss_g = torch.mean((g - g_target)**2)

        loss_ic = torch.zeros(1, device=device)
        if USE_FNO_IC_LOSS:
            loss_ic = torch.mean(((u_pred_all[0] - u_true_all[0]) / u_full_scale)**2)

        # Full-space R^2 Gaussian problem has no Dirichlet boundary.
        # We only keep initial condition consistency.
        loss_bc = torch.zeros(1, device=device)

        loss = W_FNO_U2 * loss_u2 + W_FNO_G_AUX * loss_g + W_FNO_IC * loss_ic + W_FNO_BC * loss_bc
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"[FNO] ep={ep:5d} loss={loss.item():.3e} u2={loss_u2.item():.3e} g={loss_g.item():.3e} ic={loss_ic.item():.3e} bc={loss_bc.item():.3e}")
        history.append([loss.item(), loss_u2.item(), loss_g.item(), loss_ic.item(), loss_bc.item()])

    # Pack data for evaluation
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
    u2_target = data_pack['u2_target']
    f_scale = data_pack['f_scale']

    # Generate final prediction at all times
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
        u_true = u_true_all[1:]
        u1_pred = u1_all[1:]

    # Compute exact u1 reference
    xx, yy = torch.meshgrid(xg, yg, indexing='ij')
    u1_exact_all=[]
    for tt in t_all:
        with torch.no_grad():
            u1_exact_all.append(inverse_fourier_exact_u1_2d(xx.reshape(-1), yy.reshape(-1), tt).reshape(NX,NY))
    u1_exact_all=torch.stack(u1_exact_all,dim=0)

    err_u1=u1_all-u1_exact_all
    err_u=u_pred_all-u_true_all
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

    # u1 PINN verification
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(u1_exact_all[-1].cpu(),origin='lower')
    plt.title('Exact u1 t=1')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(u1_all[-1].cpu(),origin='lower')
    plt.title('PINN u1 t=1')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(torch.abs(err_u1[-1]).cpu(),origin='lower')
    plt.title('u1 error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'u1_validation.png'),dpi=200)
    plt.close()

    # total solution error curve
    plt.figure(figsize=(6,4))
    plt.semilogy(torch.mean(torch.abs(err_u),dim=(1,2)).cpu())
    plt.title('Space error evolution')
    plt.xlabel('time step')
    plt.ylabel('MAE')
    plt.grid()
    plt.savefig(os.path.join(OUT_DIR,'error_time.png'),dpi=200)
    plt.close()

    # ------------------------------
    # Additional paper figures
    # ------------------------------

    # u2 branch validation
    u2_exact_all = u_true_all - u1_exact_all
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


    # total error heatmap
    plt.figure(figsize=(5,4))
    plt.imshow(torch.abs(err_u[-1]).cpu(), origin='lower')
    plt.title('Total solution error t=1')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'u_error_heatmap.png'), dpi=200)
    plt.close()


    # u1 error heatmap
    plt.figure(figsize=(5,4))
    plt.imshow(torch.abs(err_u1[-1]).cpu(), origin='lower')
    plt.title('u1 PINN error t=1')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'u1_error_heatmap.png'), dpi=200)
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
    print("All done.")

if __name__ == "__main__":
    main()