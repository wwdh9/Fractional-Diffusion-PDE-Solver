"""Cartesian Fourier quadrature and an independent zero-IC numerical teacher.

Convention: hat(v)(k)=integral v(x) exp(-i k.x) dx and the two-dimensional
inverse has factor (2*pi)**-2. No radial reduction or symmetry projection is
used. The RK4 solver accepts arbitrary complex forcing on the Cartesian grid.
Manufactured-reference functions are separate and are for evaluation only.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import roots_legendre


def cartesian_rule(order: int = 128, kmax: float = 8.0, power: int = 3):
    """Return a split Gauss rule on [-kmax, kmax] for EACH Cartesian axis.

    Splitting at zero and using k=sign(q)*kmax*abs(q)**power resolves limited
    regularity of the fractional symbol without assuming the spectrum is
    even. The Jacobian is included, and both signs are integrated separately.
    """
    if order < 8 or order % 2 or kmax <= 0 or power < 1:
        raise ValueError('order must be even and >=8; kmax and power must be positive')
    z, w = roots_legendre(order // 2)
    q = (z + 1.) / 2.
    positive = kmax * q**power
    weights = w/2. * kmax * power * q**(power-1)
    return np.r_[-positive[::-1], positive], np.r_[weights[::-1], weights]


def wave_mesh(k):
    return np.meshgrid(np.asarray(k, dtype=np.float64),
                       np.asarray(k, dtype=np.float64), indexing='ij')


def inverse_cartesian(spectra, x, y, k, w, real_output=False):
    """Inverse arbitrary (..., nk, nk) spectra on x-by-y output points.

    The two dense complex matrix products are separability of exp(i k.x),
    not separability or symmetry of the solution. Full complex spectra are
    retained, including independent positive and negative frequency values.
    Complex output is the default so imaginary leakage can be inspected.
    Set real_output=True only for an explicitly real-valued physical field.
    """
    spectra = np.asarray(spectra)
    k, w = np.asarray(k), np.asarray(w)
    if spectra.shape[-2:] != (len(k), len(k)) or k.shape != w.shape:
        raise ValueError('spectral axes must match the quadrature rule')
    left = np.exp(1j * np.outer(np.asarray(x), k)) * (w / (2*np.pi))
    right = np.exp(1j * np.outer(np.asarray(y), k)) * (w / (2*np.pi))
    result = (left @ spectra) @ right.T
    return result.real if real_output else result


def gaussian_initial_hat(kx, ky, center=(0., 0.), widths=(1., 1.)):
    """Fourier transform of a supplied shifted anisotropic Gaussian IC."""
    sx, sy = widths
    if sx <= 0 or sy <= 0:
        raise ValueError('Gaussian widths must be positive')
    envelope = 2*np.pi*sx*sy * np.exp(-.5*((sx*kx)**2 + (sy*ky)**2))
    if center == (0., 0.):
        return envelope
    return envelope * np.exp(-1j*(center[0]*kx + center[1]*ky))


def gaussian_source_hat(kx, ky, alpha, c=1., center=(0., 0.), widths=(1., 1.)):
    """Build the given forcing callback from IC spectrum and PDE symbol.

    The teacher receives this callback only; it never receives u, u1, u2,
    a neural prediction, or the analytic homogeneous time evolution.
    """
    phi = gaussian_initial_hat(kx, ky, center, widths)
    lam = c * (kx*kx + ky*ky)**(alpha/2.)
    constant, slope = (1+lam)*phi, lam*phi
    def source(t, _kx=None, _ky=None):
        return constant + t*slope
    return source


def solve_zero_ic_rk4(times, kx, ky, source_hat, alpha, c=1., dtmax=1/1280):
    """Integrate v_t=source_hat(t)-c*|k|**alpha*v, v(0)=0 by RK4.

    source_hat(t,kx,ky) must return the full 2D forcing spectrum. Grid-specific
    constants may be cached in its closure. Returns values and RHS/derivative
    at each requested time. This is a
    generic complex Cartesian solver and contains no analytic exp(-lambda*t)
    evolution, Gaussian solution formula, reference subtraction or PINN call.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or times[0] < 0 or np.any(np.diff(times)<0):
        raise ValueError('times must be a nonempty nondecreasing nonnegative vector')
    if not 0 < alpha <= 2 or c <= 0 or dtmax <= 0:
        raise ValueError('require 0<alpha<=2, c>0, dtmax>0')
    kx, ky = np.broadcast_arrays(kx, ky)
    lam = c * (kx*kx + ky*ky)**(alpha/2.)
    initial_source = np.asarray(source_hat(0., kx, ky))
    if initial_source.shape != lam.shape:
        raise ValueError('source_hat(t) must return the Cartesian spectral grid')
    value = np.zeros_like(initial_source, dtype=np.result_type(initial_source, np.float64))
    values, derivatives = [], []
    current = 0.
    # RK4's real negative stability interval exceeds 2.7; this conservative
    # cap supports other c/kmax choices without silently becoming unstable.
    stable_dt = min(dtmax, 2.0 / max(float(lam.max()), 1e-30))
    for target in times:
        count = max(1, int(math.ceil((target-current)/stable_dt)))
        h = (target-current)/count
        interval_start = current
        for step in range(count):
            t = interval_start + step*h
            f0 = source_hat(t, kx, ky)
            fm = source_hat(t+h/2, kx, ky)
            f1 = source_hat(t+h, kx, ky)
            k1 = f0-lam*value
            k2 = fm-lam*(value+h*k1/2)
            k3 = fm-lam*(value+h*k2/2)
            k4 = f1-lam*(value+h*k3)
            value = value+h*(k1+2*k2+2*k3+k4)/6
        current = float(target)
        values.append(value.copy())
        derivatives.append(source_hat(current,kx,ky)-lam*value)
    return np.asarray(values), np.asarray(derivatives)


def reference_homogeneous(times, x, y, k, w, initial_hat, alpha, c=1.):
    """Evaluation ONLY: analytic spectral evolution followed by 2D inverse."""
    kx, ky = wave_mesh(k)
    lam = c * (kx*kx+ky*ky)**(alpha/2.)
    spectrum = np.asarray(initial_hat(kx, ky))
    ts = np.asarray(times, dtype=np.float64)
    spectra = np.exp(-ts[:, None, None]*lam[None]) * spectrum[None]
    return inverse_cartesian(spectra, x, y, k, w)


def numerical_self_checks(alpha, x, y=None, c=1., orders=(64,96,128),
                          times=(0., .0125, .123, .5, 1.), kmax=8., dtmax=1/1280):
    """Full spatial-grid quadrature refinement and RK4 step-halving checks."""
    if y is None:
        y = x
    outputs = []
    for order in orders:
        k, w = cartesian_rule(order, kmax)
        kx, ky = wave_mesh(k)
        source = gaussian_source_hat(kx, ky, alpha, c)
        value, _ = solve_zero_ic_rk4(times, kx, ky, source, alpha, c, dtmax)
        outputs.append(inverse_cartesian(value, x, y, k, w))
    refined, _ = solve_zero_ic_rk4(times, kx, ky, source, alpha, c, dtmax/2)
    refined = inverse_cartesian(refined, x, y, k, w)
    report = {
        'alpha': float(alpha), 'quadrature_orders_per_axis': list(orders),
        'axis_quadrature_mapping_power': 3,
        'check_times': list(times), 'check_grid_shape': [len(x), len(y)],
        'quadrature_successive_max_differences': [float(np.max(abs(a-b)))
                                                  for a,b in zip(outputs[:-1], outputs[1:])],
        'rk4_step_halving_max_difference': float(np.max(abs(outputs[-1]-refined))),
        'rk4_dtmax': dtmax, 'reference_solution_used_in_teacher': False,
        'teacher_initial_value_max_abs': float(abs(refined[0]).max()) if times[0] == 0 else None,
        'method': 'full complex Cartesian tensor quadrature and zero-IC source-driven RK4',
    }
    return report


def reference_self_checks(alpha, orders=(128,192,256)):
    """Evaluation-reference quadrature refinement, separate from the teacher."""
    x = np.linspace(-5,5,64)
    times = np.array([0.,.0125,.123,.5,1.])
    outputs = []
    for order in orders:
        k,w = cartesian_rule(order)
        outputs.append(reference_homogeneous(times,x,x,k,w,gaussian_initial_hat,alpha))
    xx,yy = np.meshgrid(x,x,indexing='ij')
    return {
        'alpha': float(alpha), 'quadrature_orders_per_axis': list(orders),
        'quadrature_successive_max_differences': [float(np.max(abs(a-b)))
                                                  for a,b in zip(outputs[:-1],outputs[1:])],
        'initial_gaussian_inverse_max_error': float(abs(outputs[-1][0]-np.exp(-.5*(xx*xx+yy*yy))).max()),
        'imaginary_max_abs': float(max(abs(o.imag).max() for o in outputs)),
        'used_for': 'evaluation reference only',
    }


def nonradial_checks(order=128, kmax=8.):
    """A shifted unequal-width Gaussian checks phase/sign and orientation.

    Also checks the zero-IC solver against a complex, non-Gaussian spectral
    polynomial-in-time solution. These are numerical unit diagnostics only,
    never labels for the paper example's models.
    """
    k, w = cartesian_rule(order, kmax)
    kx, ky = wave_mesh(k)
    x, y = np.linspace(-5,5,64), np.linspace(-4.5,5.5,61)
    xx, yy = np.meshgrid(x,y,indexing='ij')
    center, widths = (.6,-.4), (.8,1.3)
    phi = gaussian_initial_hat(kx, ky, center, widths)
    t, c = .37, .7
    inverse = inverse_cartesian(np.exp(-c*t*(kx*kx+ky*ky))*phi, x,y,k,w,
                                real_output=False)
    sx, sy = widths
    ax, ay = sx*sx+2*c*t, sy*sy+2*c*t
    exact = sx*sy/np.sqrt(ax*ay) * np.exp(-.5*((xx-center[0])**2/ax + (yy-center[1])**2/ay))
    beta = .8
    lam = c*(kx*kx+ky*ky)**(beta/2)
    amplitude = (1+.2*kx+1j*.3*ky)*np.exp(-.7*kx*kx-.4*ky*ky)
    forcing = lambda ts, _kx, _ky: (2*ts+lam*ts*ts)*amplitude
    times = np.array([0.,.19,.43])
    solution,_ = solve_zero_ic_rk4(times,kx,ky,forcing,beta,c,dtmax=1/1280)
    target = times[:,None,None]**2*amplitude
    return {
        'diagnostic': 'shifted anisotropic Gaussian heat equation; generic complex polynomial forcing',
        'shift': list(center), 'unequal_widths': list(widths),
        'nonradial_gaussian_inverse_max_error': float(abs(inverse.real-exact).max()),
        'real_field_imaginary_roundoff_max': float(abs(inverse.imag).max()),
        'complex_nongaussian_rk4_max_error': float(abs(solution-target).max()),
        'symmetry_projection_used': False,
    }


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('cartesian_checks.json'))
    parser.add_argument('--alphas', nargs='+', type=float, default=[.4,.8,1.2,1.6,2.])
    args = parser.parse_args()
    reports = {'nonradial': nonradial_checks(), 'alphas': [], 'reference_checks': []}
    print(json.dumps(reports['nonradial']), flush=True)
    for alpha in args.alphas:
        report = numerical_self_checks(alpha, np.linspace(-5,5,64))
        reports['alphas'].append(report)
        print(json.dumps(report), flush=True)
        reports['reference_checks'].append(reference_self_checks(alpha))
        args.output.write_text(json.dumps(reports,indent=2),encoding='utf-8')
