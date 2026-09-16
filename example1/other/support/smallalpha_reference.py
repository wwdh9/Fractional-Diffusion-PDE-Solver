"""Independent low-alpha Fourier reference for the three supplied 1D examples.

Only reference labels/evaluation use this module; it never uses a neural model.
The Bessel transform follows NIST DLMF 10.9.4; the tail expansion is 10.17.3:
https://dlmf.nist.gov/10.9.E4 and https://dlmf.nist.gov/10.17.E3.
"""
import math
from functools import lru_cache
import numpy as np
from scipy.special import jv,roots_legendre
from scipy.integrate import quad,IntegrationWarning

def reference_grid(xs,ts,alpha,beta,amp=1.,center=0.,radius=1.,velocity=0.,c=1.,K=64,order=24,terms=8):
    """Bessel transform, finite quadrature, analytic-phase oscillatory tail."""
    nu=beta+.5
    delta=math.pi*nu/2+math.pi/4
    ac=[1.]
    for m in range(1,terms): ac.append(ac[-1]*(4*nu*nu-(2*m-1)**2)/(8*m))
    pc=[((-1)**(m//2))*ac[m] if m%2==0 else 0. for m in range(terms)]
    qc=[((-1)**((m-1)//2))*ac[m] if m%2 else 0. for m in range(terms)]
    cc=[p*math.cos(delta)+q*math.sin(delta) for p,q in zip(pc,qc)]
    sc=[p*math.sin(delta)-q*math.cos(delta) for p,q in zip(pc,qc)]
    z,w=roots_legendre(order); r=(z+1)/2; wr=w/2
    # k=q^10 avoids alpha<1 derivative singularity at zero.
    k=np.concatenate([r**10,np.concatenate([i+r for i in range(1,K)])])
    wk=np.concatenate([wr*10*r**9,np.tile(wr,K-1)])
    base=amp*2**beta*math.gamma(beta+1)*math.sqrt(2/math.pi)*jv(nu,k)/k**nu*wk
    coeff=amp*2**beta*math.gamma(beta+1)/math.pi
    output=np.empty((len(ts),len(xs)))
    worst=0.
    for it,t in enumerate(ts):
        tau=c*t/radius**alpha
        xx=(np.asarray(xs)-center-velocity*t)/radius
        output[it]=(base*np.exp(-tau*k**alpha))@np.cos(np.outer(k,xx))
        for ix,x in enumerate(xx):
            extra=0.
            for omega in [1+x,1-x]:
                for kind,co in [('cos',cc),('sin',sc)]:
                    def envelope(s):
                        if tau*s**alpha>740: return 0.
                        inv=1/s
                        p=co[-1]
                        for a in co[-2::-1]: p=p*inv+a
                        return s**(-beta-1)*math.exp(-tau*s**alpha)*p
                    if abs(omega)<1e-12:
                        if kind=='sin': continue
                        # The zero-frequency tail is nonoscillatory and extends far
                        # at small alpha; logarithmic integration needs no giant grid.
                        ymax=max(math.log(K)+1,math.log(750/tau)/alpha)
                        val,err=quad(lambda q: envelope(math.exp(q))*math.exp(q),math.log(K),ymax,epsabs=2e-11,epsrel=2e-11)
                    else:
                        val,err=quad(envelope,K,np.inf,weight=kind,wvar=abs(omega),epsabs=2e-11,limlst=200,limit=200)
                        if omega<0 and kind=='sin': val=-val
                    extra+=val
                    worst=max(worst,err*coeff)
            output[it,ix]+=coeff*extra
    return output,worst


@lru_cache(maxsize=8)
def _cached_grid(xs,ts,alpha,c,velocity,case,refinement):
    if case in (1,2):
        beta=3.; amp=1/64.; center=.5; radius=.5
    elif case==3:
        beta=alpha/2
        amp=2**(-alpha)*math.gamma(.5)/(math.gamma(1+beta)*math.gamma(.5+beta))
        center=0.;radius=1.
    else:
        raise ValueError('Low-alpha reference is specific to examples 1, 2 and 3')
    values,tail_error=reference_grid(np.asarray(xs),np.asarray(ts),alpha,beta,
        amp=amp,center=center,radius=radius,velocity=velocity,c=c,
        K=64*refinement,order=24 if refinement==1 else 32)
    if not np.isfinite(values).all() or tail_error>1e-8:
        raise ArithmeticError(f'Low-alpha reference quadrature did not converge: tail error={tail_error}')
    return values


def reference_u1(x,t,scope,refinement=1):
    """Same tensor API as audited_support.reference_u1, for 0<alpha<1.

    Keeping alpha>=1 on the original reference path preserves previous results.
    refinement=2 doubles the finite/tail split and increases its Gauss order.
    """
    import torch
    alpha=float(scope['ALPHA']);c=float(scope['C'])
    if not 0<alpha<1 or c<=0: raise ValueError('Expected 0<alpha<1 and C>0')
    xn=x.detach().cpu().double().numpy().ravel()
    tn=t.detach().cpu().double().numpy().ravel()
    if np.any(tn<0): raise ValueError('Reference time must be nonnegative')
    xs,xi=np.unique(xn,return_inverse=True);ts,ti=np.unique(tn,return_inverse=True)
    result=np.empty((len(ts),len(xs)),dtype=np.float64)
    positive=ts[ts>0]
    if len(positive):
        result[ts>0]=_cached_grid(tuple(xs),tuple(positive),alpha,c,float(scope.get('NU',0.)),int(scope['EXAMPLE_ID']),int(refinement))
    if np.any(ts==0):
        result[ts==0]=scope['u0_torch'](torch.tensor(xs,dtype=torch.float64)).numpy()
    values=torch.tensor(result[ti,xi].reshape(x.shape),device=x.device,dtype=x.dtype)
    zero=t==0
    if torch.any(zero): values[zero]=scope['u0_torch'](x[zero])
    return values
