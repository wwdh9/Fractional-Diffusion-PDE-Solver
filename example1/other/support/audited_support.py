"""Result persistence, independent references and plotting for 1D examples."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import roots_legendre
import matplotlib.pyplot as plt


def reference_u1(x,t,scope,refinement=1):
    """Independent double-precision reference; exact supplied initial data at t=0.

    Does not use the PINN, its weights, or its 80-node prediction quadrature.
    Quadrature accuracy is verified separately in reference_checks.json.
    """
    if scope['ALPHA'] < 1:
        import smallalpha_reference
        return smallalpha_reference.reference_u1(x,t,scope,refinement)
    xn=x.detach().cpu().double().numpy().ravel()
    tn=t.detach().cpu().double().numpy().ravel()
    xs,xi=np.unique(xn,return_inverse=True); ts,ti=np.unique(tn,return_inverse=True)
    a,c,nu=scope['ALPHA'],scope['C'],scope.get('NU',0.)
    positive=ts[ts>0]
    result=np.empty((len(ts),len(xs)),dtype=np.float64)
    if len(positive):
        kmax=max(32.,(32/(c*positive.min()))**(1/a))
        nk=max(512,int(np.ceil(4*kmax*(1+abs(nu)))))*refinement
        nk=min(nk,16384)
        z,w=roots_legendre(nk); k=(z+1)*kmax/2; w=w*kmax/2
        nq=max(1024,int(4*kmax))*refinement
        zq,wq=roots_legendre(nq)
        left,right=scope['X_LEFT'],scope['X_RIGHT']
        xq=left+(zq+1)*(right-left)/2; wq=wq*(right-left)/2
        uq=scope['u0_torch'](torch.tensor(xq,dtype=torch.float64)).numpy()
        hat=np.empty(nk,dtype=np.complex128)
        for start in range(0,nk,256):
            hat[start:start+256]=np.exp(-1j*np.outer(k[start:start+256],xq))@(wq*uq)/np.sqrt(2*np.pi)
        spectra=np.exp(-np.outer(positive,c*k**a+1j*nu*k))*hat*w
        result[ts>0]=(spectra@np.exp(1j*np.outer(k,xs))).real*np.sqrt(2/np.pi)
    if np.any(ts==0):
        result[ts==0]=scope['u0_torch'](torch.tensor(xs,dtype=torch.float64)).numpy()
    values=torch.tensor(result[ti,xi].reshape(x.shape),device=x.device,dtype=x.dtype)
    zero=t==0
    if torch.any(zero): values[zero]=scope['u0_torch'](x[zero])
    return values


def metrics(pred,ref):
    e=np.asarray(pred)-np.asarray(ref)
    return dict(mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),
        max_abs=float(abs(e).max()),relative_l2=float(np.linalg.norm(e.ravel())/max(np.linalg.norm(np.asarray(ref).ravel()),1e-30)))


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def export_evaluation(scope,data_pack,predictions):
    xg,tall,_,_,_,u_ref_all,_,u1_all,_,scales=data_pack
    u2_all,u_all=predictions
    to_np=lambda x:x.detach().cpu().double().numpy()
    x,t=to_np(xg),to_np(tall)
    xx,tt=torch.meshgrid(xg,tall,indexing='xy')
    with torch.no_grad():
        u1_ref=scope['true_u1_inverse_fourier'](xx.ravel(),tt.ravel()).reshape(xx.shape)
    arrays={name:to_np(v) for name,v in dict(u1=u1_all,u2=u2_all,total=u_all,
        reference_u1=u1_ref,reference_u2=u_ref_all-u1_ref,reference_total=u_ref_all).items()}
    out=Path(scope['OUT_DIR'])
    np.savez_compressed(out/'predictions.npz',x=x,times=t,**arrays)
    result={'alpha':scope['ALPHA'],'all_times':{},'t1':{},'initial':{},'boundary':{},
        'scope':'single source trajectory; training-grid reconstruction; not an unseen-source benchmark'}
    for branch in ['u1','u2','total']:
        p,r=arrays[branch],arrays['reference_'+branch]
        result['all_times'][branch]=metrics(p,r)
        result['t1'][branch]=metrics(p[-1],r[-1])
        result['initial'][branch]=metrics(p[0],r[0])
        result['boundary'][branch]=metrics(p[:,[0,-1]],r[:,[0,-1]])
    save_json(out/'metrics.json',result)
    np.savez_compressed(out/'fno_training_data.npz',x=x,times=t,
        f_input=to_np(data_pack[3]),u2_label=to_np(scales['u2_target']),g_label=to_np(scales['g_target']))
    extra_plots(out,x,t,arrays)
    # Reference self-convergence, including earliest positive time and endpoints.
    xcheck=torch.linspace(scope['X_LEFT'],scope['X_RIGHT'],21,device=xg.device,dtype=torch.float64)
    tcheck=torch.tensor([0.,scope['T_END']/scope['M'],.1,1.],device=xg.device,dtype=torch.float64)
    xc,tc=torch.meshgrid(xcheck,tcheck,indexing='xy')
    r1=reference_u1(xc.ravel(),tc.ravel(),scope,1)
    r2=reference_u1(xc.ravel(),tc.ravel(),scope,2)
    save_json(out/'reference_checks.json',{'doubling_quadrature_max_difference':float((r1-r2).abs().max()),
        'u2_target_depends_on_pinn':False,'u2_initial_value':0.,
        'prediction_frequency_cutoff':scope['K_MAX'],'prediction_frequency_nodes':scope['N_K'],
        'note':'Prediction cutoff/nodes kept unchanged; references use independent, converged quadrature.'})
    print('AUDITED METRICS',json.dumps(result),flush=True)


def extra_plots(out,x,t,a):
    for name in ['u1','u2','total']:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        p,r=a[name],a['reference_'+name]
        for ax,idx in zip(axes,[0,min(1,len(t)-1),-1]):
            ax.plot(x,r[idx],label='Reference'); ax.plot(x,p[idx],'--',label='Network')
            ax.set_title(f'{name}, t={t[idx]:.4f}'); ax.set_xlabel('x'); ax.grid(alpha=.25)
        axes[0].legend(); fig.savefig(out/f'{name}_initial_and_final.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        width=(x[-1]-x[0])*.12
        regions=[(x[0],x[0]+width),((x[0]+x[-1])/2-width/2,(x[0]+x[-1])/2+width/2),(x[-1]-width,x[-1])]
        for ax,(left,right) in zip(axes,regions):
            ix=(x>=left)&(x<=right)
            ax.plot(x[ix],r[-1,ix],label='Reference'); ax.plot(x[ix],p[-1,ix],'--',label='Network')
            ax.set(xlabel='x',title=f'{name}, t=1: [{left:.2f}, {right:.2f}]'); ax.grid(alpha=.25)
        axes[0].legend(); fig.savefig(out/f'{name}_local_zoom.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
    for row,name in enumerate(['u1','u2','total']):
        for col,idx in enumerate([0,-1]):
            axes[row,col].plot(t,a['reference_'+name][:,idx],label='Reference')
            axes[row,col].plot(t,a[name][:,idx],'--',label='Network')
            axes[row,col].set(xlabel='t',title=f'{name}, x={x[idx]:g}'); axes[row,col].grid(alpha=.25)
    axes[0,0].legend(); fig.savefig(out/'branch_boundary_values.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ['u1','u2','total']:
        ax.semilogy(t,np.maximum(abs(a[name]-a['reference_'+name]).mean(1),1e-16),label=name)
    ax.set(xlabel='t',ylabel='Spatial MAE'); ax.legend(); ax.grid(alpha=.25)
    fig.savefig(out/'error_by_time.png',dpi=160); plt.close(fig)


def run(scope,args):
    import time
    out=Path(scope['OUT_DIR']); start=time.perf_counter()
    if args.stage=='evaluate':
        state=torch.load(out/'run_state.pt',map_location=scope['device'],weights_only=True)
        if state['alpha']!=scope['ALPHA']: raise ValueError('Checkpoint alpha differs from requested alpha')
        net=scope['PINNNet']().to(scope['device'])
        fno=scope['FNO1dNoBeta']().to(scope['device'])
        net.load_state_dict(state['pinn']); fno.load_state_dict(state['fno'])
        data=state['data']; hp=state['pinn_history'].cpu().numpy(); hf=state['fno_history'].cpu().numpy()
    else:
        pstart=time.perf_counter(); net,hp=scope['train_u1_branch'](); pseconds=time.perf_counter()-pstart
        fstart=time.perf_counter(); fno,hf,data=scope['train_fno_branch'](net); fseconds=time.perf_counter()-fstart
        torch.save({'alpha':scope['ALPHA'],'pinn':net.state_dict(),'fno':fno.state_dict(),'data':data,
            'pinn_history':torch.from_numpy(hp),'fno_history':torch.from_numpy(hf)},out/'run_state.pt')
        torch.save(net.state_dict(),out/'u1_soft_scaled_pinn.pt')
        torch.save(fno.state_dict(),out/'fno_scaled_no_beta_icbc.pt')
        np.savetxt(out/'pinn_history.csv',hp,delimiter=',',header='total,frequency_equation,frequency_initial,physical_equation,physical_initial')
        np.savetxt(out/'fno_history.csv',hf,delimiter=',',header='total,u2_scaled,g_scaled,u2_mse,u2_mae,total_mae,u2_initial,u2_boundary,total_initial_mse,total_boundary_mse')
        save_json(out/'config.json',{key:scope[key] for key in ['EXAMPLE_ID','ALPHA','C','X_LEFT','X_RIGHT','N_K','K_MAX','S','M','EPOCH_PINN','EPOCH_FNO','LR_PINN','LR_FNO','B_FREQ','B_IC','W_FREQ_EQ','W_FREQ_IC','W_FNO_U2','W_FNO_G_AUX','W_FNO_IC','W_FNO_BC']})
        save_json(out/'runtime.json',dict(pinn_seconds=pseconds,fno_seconds=fseconds,
            training_seconds=pseconds+fseconds,device=str(scope['device']),seed=0))
    net.eval(); fno.eval()
    scope['evaluate_and_plot'](net,fno,data)
    scope['plot_losses'](hp,hf)
    scope['plot_freq_ic_check'](net)
    print(f'Completed {args.stage} in {time.perf_counter()-start:.1f} s; results: {out}',flush=True)
