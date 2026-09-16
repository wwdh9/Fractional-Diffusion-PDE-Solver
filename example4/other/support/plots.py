"""Portfolio figure coverage; uses saved predictions and actual training logs."""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_plots(out):
    out=Path(out); data=np.load(out/'predictions.npz')
    x,t=data['x'],data['times']; mid=int(np.argmin(abs(x)))
    for branch,filename in [('u1','u1_validation.png'),('u2','u2_validation.png'),('total','2d_result_t1.png')]:
        p,r=data[branch],data['reference_'+branch]
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,label,v in zip(axes,['Reference','Network','Absolute error'],[r[-1],p[-1],abs(p[-1]-r[-1])]):
            im=ax.imshow(v.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]])
            ax.set_title(f'{branch}: {label}, t=1'); fig.colorbar(im,ax=ax,shrink=.8)
        fig.savefig(out/filename,dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,idx in zip(axes,[0,1,-1]):
            ax.plot(x,r[idx,:,mid],label='Reference'); ax.plot(x,p[idx,:,mid],'--',label='Network')
            ax.set(xlabel='x',title=f'{branch}, y={x[mid]:.3f}, t={t[idx]:.4f}'); ax.grid(alpha=.25)
        axes[0].legend(); fig.savefig(out/f'{branch}_initial_and_final.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,(left,right) in zip(axes,[(x[0],x[0]+1),(-.8,.8),(x[-1]-1,x[-1])]):
            ix=(x>=left)&(x<=right)
            ax.plot(x[ix],r[-1,ix,mid],label='Reference'); ax.plot(x[ix],p[-1,ix,mid],'--',label='Network')
            ax.set(xlabel='x',title=f'{branch}, y={x[mid]:.3f}, t=1'); ax.grid(alpha=.25)
        axes[0].legend(); fig.savefig(out/f'{branch}_local_zoom.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,label,v in zip(axes,['Reference','Network','Absolute error'],[r[0],p[0],abs(p[0]-r[0])]):
            im=ax.imshow(v.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]])
            ax.set_title(f'{branch}: {label}, t=0'); fig.colorbar(im,ax=ax,shrink=.8)
        fig.savefig(out/f'{branch}_initial_heatmap.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,9),layout='constrained')
    for row,branch in enumerate(['u1','u2','total']):
        edge_points=[(0,mid,f'x={x[0]:g}, y={x[mid]:.3f}'),
                     (-1,mid,f'x={x[-1]:g}, y={x[mid]:.3f}'),
                     (mid,0,f'y={x[0]:g}, x={x[mid]:.3f}'),
                     (mid,-1,f'y={x[-1]:g}, x={x[mid]:.3f}')]
        for col,(ix,iy,label) in enumerate(edge_points):
            ax=axes[row,col]
            ax.plot(t,data['reference_'+branch][:,ix,iy],label='Reference')
            ax.plot(t,data[branch][:,ix,iy],'--',label='Network')
            ax.set(xlabel='t',title=f'{branch}, {label}'); ax.grid(alpha=.25)
    axes[0,0].legend(); fig.savefig(out/'window_edge_values.png',dpi=160); plt.close(fig)
    # These are observation-window edges, not zero Dirichlet boundaries of R^2.
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for branch in ['u1','u2','total']:
        mae=abs(data[branch]-data['reference_'+branch]).mean(axis=(1,2))
        ax.semilogy(t,np.maximum(mae,1e-16),label=branch)
    ax.set(xlabel='t',ylabel='Spatial MAE'); ax.legend(); ax.grid(alpha=.25)
    fig.savefig(out/'error_time.png',dpi=160); plt.close(fig)
    for branch,filename in [('u1','u1_error_heatmap.png'),('total','u_error_heatmap.png')]:
        fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
        im=ax.imshow(abs(data[branch][-1]-data['reference_'+branch][-1]).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='magma')
        ax.set_title(f'{branch}: absolute error at t=1'); fig.colorbar(im,ax=ax)
        fig.savefig(out/filename,dpi=160); plt.close(fig)
    hp=np.loadtxt(out/'pinn_history.csv',delimiter=',',skiprows=1,ndmin=2)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.semilogy(hp[:,0],hp[:,2],label='Weighted complex PDE loss')
    ax.set(xlabel='Objective evaluation / Adam step',ylabel='Loss'); ax.legend(); ax.grid(alpha=.25)
    fig.savefig(out/'u1_losses.png',dpi=160); plt.close(fig)
    first=out/'fno_first_pass_history.csv'
    hf=np.loadtxt(first if first.exists() else out/'fno_history.csv',delimiter=',',ndmin=2)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for col,label in [(1,'Training'),(2,'Numerical validation')]:
        valid=np.isfinite(hf[:,col]); axes[0].semilogy(hf[valid,0],hf[valid,col],label=label)
    axes[0].set(xlabel='Step',ylabel='Normalized MSE',title='FNO first training pass'); axes[0].legend()
    refine=[]
    log=out/'logs/example4_fno_refine.log'
    if log.exists():
        refine=[(int(a),float(b)) for a,b in re.findall(r'FNO LBFGS\s+(\d+)\s+([\deE.+-]+)',log.read_text(encoding='utf-8-sig',errors='replace'))]
    if not refine:
        h=np.loadtxt(out/'fno_history.csv',delimiter=',',ndmin=2)
        refine=[(v[0],v[1]) for v in h if np.isfinite(v[1])]
    if refine:
        h=np.array(refine); axes[1].semilogy(h[:,0],h[:,1],label='Full training-data loss'); axes[1].legend()
    else: axes[1].text(.5,.5,'No recorded refinement trace',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlabel='Objective evaluation',ylabel='Normalized MSE',title='FNO additional L-BFGS pass')
    for ax in axes: ax.grid(alpha=.25)
    fig.savefig(out/'fno_losses.png',dpi=160); plt.close(fig)
    print('Saved complete Example 4 comparison/loss/initial/local/edge figures.',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('out',type=Path)
    make_plots(p.parse_args().out)
