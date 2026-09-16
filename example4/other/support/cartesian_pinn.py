"""Cartesian complex PINN for a constant-coefficient homogeneous spectral PDE.

Only the known initial spectrum and the PDE enter training.  The two output
channels and all three Cartesian inputs remain free; no rotation/reflection
averaging, radial input reduction, or analytic time propagator is used.
"""
from __future__ import annotations
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn


def gaussian_initial(kx, ky):
    """Known initial data for the example, not part of the network architecture."""
    return (2 * math.pi * torch.exp(-.5 * (kx*kx + ky*ky))).to(
        torch.complex128 if kx.dtype == torch.float64 else torch.complex64)


def shifted_elliptic_initial(kx, ky):
    """An independent, nonradial complex initial spectrum for API checks."""
    co, si = math.cos(.4), math.sin(.4)
    ka, kb = co*kx+si*ky, -si*kx+co*ky
    mag = 2*math.pi*.7*1.3*torch.exp(-.5*(.7**2*ka**2+1.3**2*kb**2))
    return mag * torch.exp(-1j*(.7*kx-.3*ky))


class CartesianPINN(nn.Module):
    def __init__(self, alpha, c=1., width=64, depth=4, kscale=2.):
        super().__init__()
        self.alpha, self.c, self.kscale = float(alpha), float(c), float(kscale)
        self.width, self.depth = int(width), int(depth)
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def amplitude(self, kx, ky, t):
        raw = self.net(torch.stack([kx/self.kscale, ky/self.kscale, 2*t-1], -1))
        return torch.stack([1+t*raw[..., 0], t*raw[..., 1]], -1)

    def forward(self, kx, ky, t, initial_spectrum):
        initial = initial_spectrum(kx, ky) if callable(initial_spectrum) else initial_spectrum
        amp = self.amplitude(kx, ky, t)
        return initial * torch.complex(amp[..., 0], amp[..., 1])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_points(n, initial_spectrum, kmax, device, seed):
    """Mixture sampling depends on known IC data, not on any evolved solution.

    One quarter uniform points plus three quarters sampled from |u0_hat|^2.
    Importance weights retain the physical spectral L2 residual objective.
    """
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    pool = engine.draw(65536).double().to(device)
    pool_kx, pool_ky = (2*pool[:,0]-1)*kmax, (2*pool[:,1]-1)*kmax
    magnitude2 = initial_spectrum(pool_kx, pool_ky).abs().square()
    integral = magnitude2.mean() * (2*kmax)**2
    generator = torch.Generator(device=device).manual_seed(seed+1)
    n_importance = 3*n//4
    chosen = torch.multinomial(magnitude2, n_importance, replacement=True, generator=generator)
    extra = engine.draw(n-n_importance).double().to(device)
    kx = torch.cat([pool_kx[chosen], (2*extra[:,0]-1)*kmax])
    ky = torch.cat([pool_ky[chosen], (2*extra[:,1]-1)*kmax])
    t = engine.draw(n).double().to(device)[:,2].detach().requires_grad_(True)
    amplitude2 = initial_spectrum(kx, ky).abs().square()
    density = .25/(2*kmax)**2 + .75*amplitude2/integral
    weights = amplitude2/density
    weights = (weights/weights.sum()).detach()
    return kx.detach(), ky.detach(), t, weights


def residual(model, kx, ky, t, create_graph=True):
    amplitude = model.amplitude(kx, ky, t)
    derivative = torch.stack([
        torch.autograd.grad(amplitude[:,j].sum(), t, create_graph=create_graph,
                            retain_graph=True)[0] for j in range(2)], -1)
    lam = model.c*(kx*kx+ky*ky).pow(model.alpha/2)
    return derivative+lam[:,None]*amplitude


def train_pinn(out, alpha, c=1., epochs=2500, lbfgs_steps=500,
               initial_spectrum=None, device=None, width=64, depth=4,
               kmax=8., seed=20260916, points=4096):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    initial_spectrum = gaussian_initial if initial_spectrum is None else initial_spectrum
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(4)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CartesianPINN(alpha,c,width,depth).double().to(device)
    config = dict(alpha=float(alpha), c=float(c), width=width, depth=depth,
                  kscale=model.kscale, kmax=kmax, adam_epochs=epochs,
                  lbfgs_steps=lbfgs_steps, points=points, seed=seed,
                  dtype='float64', device=str(device), inputs=['kx','ky','t'],
                  outputs=['real_amplitude_correction','imag_amplitude_correction'],
                  hard_ic='u0_hat * (1 + t*a_theta + i*t*b_theta)',
                  sampler='1/4 uniform + 3/4 known initial power; importance corrected',
                  exact_evolution_used_in_training=False,
                  symmetry_reduction=False, symmetry_averaging=False)
    save_json(out/'pinn_config.json', config)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(adam, max(epochs,1), eta_min=1e-5)
    samples = make_points(points, initial_spectrum, kmax, device, seed)
    history=[]; tic=time.perf_counter()
    def objective():
        kx,ky,t,w=samples
        eq = residual(model,kx,ky,t)
        return (w*(eq.square().sum(-1))).sum()
    for step in range(epochs):
        if step and step % 250 == 0:
            samples = make_points(points,initial_spectrum,kmax,device,seed+step)
        adam.zero_grad(set_to_none=True)
        loss=objective(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        adam.step(); schedule.step()
        history.append([step+1,0,float(loss.detach())])
        if (step+1)%250==0:
            print('PINN Adam',step+1,float(loss.detach()),flush=True)
    samples=make_points(points*2,initial_spectrum,kmax,device,seed+99999)
    lbfgs=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=lbfgs_steps,
                          max_eval=max(1,lbfgs_steps*5//4),history_size=60,
                          tolerance_grad=1e-11,tolerance_change=1e-14,
                          line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        lbfgs.zero_grad(set_to_none=True)
        loss=objective();loss.backward();calls+=1
        history.append([epochs+calls,1,float(loss.detach())])
        if calls%100==0:
            print('PINN LBFGS closure',calls,float(loss.detach()),flush=True)
        return loss
    if lbfgs_steps:
        lbfgs.step(closure)
    elapsed=time.perf_counter()-tic
    torch.save(model.state_dict(),out/'pinn.pt')
    np.savetxt(out/'pinn_history.csv',np.asarray(history),delimiter=',',
               header='step,phase_0_adam_1_lbfgs,weighted_pde_residual',comments='')
    kx,ky,t,w=make_points(points*4,initial_spectrum,kmax,device,seed+77777)
    eq=residual(model,kx,ky,t,create_graph=False).detach()
    report=dict(seconds=elapsed,adam_steps=epochs,lbfgs_closure_evaluations=calls,
                independent_pde_points=len(kx),
                independent_weighted_pde_rms=float((w*eq.square().sum(-1)).sum().sqrt()),
                independent_unweighted_amplitude_pde_rms=float(eq.square().mean().sqrt()))
    report.update(functional_checks(model))
    save_json(out/'pinn_checks.json',report)
    print('PINN done',report,flush=True)
    return model


def functional_checks(model):
    """Checks architecture capabilities, not generalization performance."""
    p=next(model.parameters());device,dtype=p.device,p.dtype
    kx=torch.tensor([.4,.9,-.7,1.2],device=device,dtype=dtype)
    ky=torch.tensor([.8,-.3,1.1,-.6],device=device,dtype=dtype)
    initial=shifted_elliptic_initial(kx,ky)
    with torch.no_grad():
        at_zero=model(kx,ky,torch.zeros_like(kx),initial)
        at_half=model(kx,ky,torch.full_like(kx,.5),initial)
        # A deliberately asymmetric random network of the same class is legal.
        torch_rng=torch.random.get_rng_state()
        probe=CartesianPINN(model.alpha,model.c,model.width,model.depth,model.kscale).to(device,dtype)
        nn.init.normal_(probe.net[-1].weight,std=.1)
        same_r=probe.amplitude(torch.tensor([1.,0.],device=device,dtype=dtype),
                              torch.tensor([0.,1.],device=device,dtype=dtype),
                              torch.tensor([.5,.5],device=device,dtype=dtype))
        torch.random.set_rng_state(torch_rng)
    return dict(complex_nonradial_initial_hard_ic_max_error=float((at_zero-initial).abs().max()),
                complex_nonradial_positive_time_imag_max=float(at_half.imag.abs().max()),
                architecture_can_differ_at_equal_radius=bool((same_r[0]-same_r[1]).abs().max()>1e-8),
                architecture_equal_radius_probe_difference=float((same_r[0]-same_r[1]).abs().max()),
                generalization_claim='API/architecture checks only; not a trained cross-source FNO benchmark')


def reference_evaluation(model, out, initial_spectrum=None, nx=33, nt=41, nk=128):
    """Post-training reference only. Never called by an optimizer or selector."""
    from scipy.special import roots_legendre
    initial_spectrum=gaussian_initial if initial_spectrum is None else initial_spectrum
    p=next(model.parameters());device,dtype=p.device,p.dtype
    nodes,weights=roots_legendre(nk);nodes*=8.;weights*=8.
    kxx,kyy=np.meshgrid(nodes,nodes,indexing='ij')
    kx=torch.tensor(kxx.ravel(),device=device,dtype=dtype)
    ky=torch.tensor(kyy.ravel(),device=device,dtype=dtype)
    initial=initial_spectrum(kx,ky)
    lam=model.c*(kx*kx+ky*ky).pow(model.alpha/2)
    x=np.linspace(-5,5,nx);times=np.linspace(0,1,nt)
    basis=np.exp(1j*np.outer(nodes,x))*weights[:,None]/(2*math.pi)
    prediction=[];reference=[]
    with torch.no_grad():
        for time_value in times:
            t=torch.full_like(kx,float(time_value))
            pred=model(kx,ky,t,initial).cpu().numpy().reshape(nk,nk)
            ref=(initial*torch.exp(-lam*t)).cpu().numpy().reshape(nk,nk)
            prediction.append((basis.T@pred@basis).real)
            reference.append((basis.T@ref@basis).real)
    prediction,reference=np.asarray(prediction),np.asarray(reference)
    error=prediction-reference
    report=dict(alpha=model.alpha,grid=[nt,nx,nx],cartesian_quadrature=[nk,nk],
                mae=float(abs(error).mean()),rmse=float(np.mean(error**2)**.5),
                max_abs=float(abs(error).max()),relative_l2=float(np.linalg.norm(error)/np.linalg.norm(reference)),
                t1_mae=float(abs(error[-1]).mean()),t0_max_abs=float(abs(error[0]).max()),
                reference_usage='post-training evaluation only; final iterate, no reference model selection')
    save_json(Path(out)/'pinn_reference_evaluation.json',report)
    np.savez_compressed(Path(out)/'pinn_reference_evaluation.npz',x=x,times=times,prediction=prediction,reference=reference)
    print('PINN reference',report,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--alpha',type=float,default=1.6)
    parser.add_argument('--epochs',type=int,default=2500)
    parser.add_argument('--lbfgs',type=int,default=500)
    args=parser.parse_args()
    model=train_pinn(args.out,args.alpha,epochs=args.epochs,lbfgs_steps=args.lbfgs)
    reference_evaluation(model,args.out)
