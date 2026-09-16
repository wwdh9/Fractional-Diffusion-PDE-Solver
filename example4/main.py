"""Full Cartesian Fourier PINN + 2D FNO: no prescribed solution symmetry.

Each run solves one alpha. Only evaluate() constructs the manufactured total
solution and the analytic homogeneous evolution. Training uses the supplied
initial spectrum/PDE and an independent source-driven numerical dataset.
"""
from __future__ import annotations
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import argparse, csv, json, shutil, sys
import numpy as np
import torch
# Match training precision when evaluating checkpoints in a fresh process.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(4)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'other/support'))
import cartesian_numerics as numerical
import fno_model
import plots

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHAS=[.4,.8,1.2,1.6,2.]


def save_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def initial_hat(kx,ky,config):
    """Supplied initial data, including phase for translated/unequal-width data."""
    return numerical.gaussian_initial_hat(kx,ky,tuple(config['center']),tuple(config['widths']))


def initial_hat_tensor(kx,ky,config):
    sx,sy=config['widths']; mx,my=config['center']
    return 2*np.pi*sx*sy*torch.exp(-.5*((sx*kx)**2+(sy*ky)**2))*torch.exp(-1j*(mx*kx+my*ky))


def forcing(kx,ky,config):
    # Given forcing, not a subtraction involving any reference or network u1.
    phi=initial_hat(kx,ky,config)
    lam=config['c']*(kx*kx+ky*ky)**(config['alpha']/2)
    constant,slope=(1+lam)*phi,lam*phi
    return lambda t,*coordinates: constant+t*slope


def rule(config, order=None):
    return numerical.cartesian_rule(order or config['quadrature_order'],config['kmax'])


def inverse(spectrum,x,k,w):
    return numerical.inverse_cartesian(spectrum,x,x,k,w,real_output=False)


def make_dataset(out,config):
    x=np.linspace(-config['xmax'],config['xmax'],config['nx'])
    m=config['m']; train_t=(np.arange(m)+.5)/m
    val_t=(np.arange(20)+.173)/20
    all_t=np.r_[train_t,val_t]; order=np.argsort(all_t); undo=np.argsort(order)
    k,w=rule(config); kx,ky=numerical.wave_mesh(k)
    source=forcing(kx,ky,config)
    values,derivatives=numerical.solve_zero_ic_rk4(all_t[order],kx,ky,source,config['alpha'],config['c'],config['dtmax'])
    u2=inverse(values,x,k,w)[undo]; g=inverse(derivatives,x,k,w)[undo]
    f=inverse(np.asarray([source(t) for t in all_t]),x,k,w)
    imag=max(float(abs(v.imag).max()) for v in [f,g,u2])
    np.savez_compressed(out/'numerical_training_data.npz',x=x,train_t=train_t,val_t=val_t,
        f_train=f[:m].real,g_train=g[:m].real,u2_train=u2[:m].real,
        f_val=f[m:].real,g_val=g[m:].real,u2_val=u2[m:].real)
    times=np.array([0.,.0125,.123,.5,1.])
    coarse,_=numerical.solve_zero_ic_rk4(times,kx,ky,source,config['alpha'],config['c'],config['dtmax'])
    finer,_=numerical.solve_zero_ic_rk4(times,kx,ky,source,config['alpha'],config['c'],config['dtmax']/2)
    k2,w2=rule(config,config['reference_order']); kx2,ky2=numerical.wave_mesh(k2)
    refined,_=numerical.solve_zero_ic_rk4(times,kx2,ky2,forcing(kx2,ky2,config),config['alpha'],config['c'],config['dtmax']/2)
    checks=dict(alpha=config['alpha'],training_uses_reference=False,pinn_used_by_teacher=False,
        quadrature='full Cartesian tensor-product rule',quadrature_orders=[len(k),len(k2)],
        spatial_shape=[len(x),len(x)],train_times=m,validation_times=20,
        rk4_step_halving_max=float(abs(inverse(coarse-finer,x,k,w)).max()),
        quadrature_refinement_max=float(abs(inverse(finer,x,k,w)-inverse(refined,x,k2,w2)).max()),
        real_data_imaginary_roundoff_max=imag)
    save_json(out/'data_checks.json',checks)
    print('DATA',checks,flush=True)


def predict_u1(model,config,times,x,order=None):
    k,w=rule(config,order or config['prediction_order']); kx,ky=numerical.wave_mesh(k)
    kxt=torch.tensor(kx.ravel(),device=DEVICE,dtype=torch.float64)
    kyt=torch.tensor(ky.ravel(),device=DEVICE,dtype=torch.float64)
    phi=initial_hat_tensor(kxt,kyt,config)
    fields=[]
    with torch.no_grad():
        for time in times:
            predictions=[]
            for start in range(0,len(kxt),4096):
                sl=slice(start,start+4096)
                value=model(kxt[sl],kyt[sl],torch.full_like(kxt[sl],float(time)),phi[sl])
                predictions.append(value.cpu().numpy())
            fields.append(inverse(np.concatenate(predictions).reshape(len(k),len(k)),x,k,w))
    return np.asarray(fields)


def predict_u2(model,config,times,x):
    times=np.asarray(times); dt=np.diff(times); middle=(times[:-1]+times[1:])/2
    ts=np.stack([middle-dt/(2*np.sqrt(3)),middle+dt/(2*np.sqrt(3))],1).ravel()
    k,w=rule(config); kx,ky=numerical.wave_mesh(k); source=forcing(kx,ky,config)
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(ts),8):
            part=ts[start:start+8]
            f=inverse(np.asarray([source(t) for t in part]),x,k,w).real
            out=model(torch.tensor(f,device=DEVICE,dtype=torch.float32),torch.tensor(part,device=DEVICE,dtype=torch.float32))
            outputs.append(out.cpu().double().numpy())
    g=np.concatenate(outputs).reshape(-1,2,len(x),len(x))
    increments=g.mean(1)*dt[:,None,None]
    return np.concatenate([np.zeros((1,len(x),len(x))),np.cumsum(increments,axis=0)])


def metrics(pred,ref):
    error=np.asarray(pred)-np.asarray(ref)
    return dict(mae=float(abs(error).mean()),rmse=float(np.sqrt(np.mean(abs(error)**2))),
        max_abs=float(abs(error).max()),relative_l2=float(np.linalg.norm(error.ravel())/max(np.linalg.norm(np.asarray(ref).ravel()),1e-30)))


def load_pinn(out,config):
    import cartesian_pinn
    saved=json.loads((out/'pinn_config.json').read_text())
    model=cartesian_pinn.CartesianPINN(config['alpha'],config['c'],width=saved.get('width',64),
        depth=saved.get('depth',4),kscale=saved.get('kscale',2.)).double().to(DEVICE)
    model.load_state_dict(torch.load(out/'pinn.pt',map_location=DEVICE,weights_only=True))
    return model.eval()


def evaluate(out,config):
    # Manufactured total and homogeneous reference are used only here.
    x=np.linspace(-config['xmax'],config['xmax'],config['nx']); xx,yy=np.meshgrid(x,x,indexing='ij')
    times=np.linspace(0,1,config['m']+1); k,w=rule(config,config['reference_order']); kx,ky=numerical.wave_mesh(k)
    lam=config['c']*(kx*kx+ky*ky)**(config['alpha']/2)
    ref_u1=inverse(np.exp(-times[:,None,None]*lam)*initial_hat(kx,ky,config),x,k,w).real
    sx,sy=config['widths']; mx,my=config['center']
    ref_u=(1+times[:,None,None])*np.exp(-.5*(((xx-mx)/sx)**2+((yy-my)/sy)**2))
    ref_u2=ref_u-ref_u1
    model=load_pinn(out,config); u1_complex=predict_u1(model,config,times,x); u1=u1_complex.real
    fno=fno_model.load_fno(out); u2=predict_u2(fno,config,times,x); total=u1+u2
    result={'config':config,'all_times':{},'t1':{},'initial':{},'window_edges':{}}
    edge_mask=np.zeros((len(x),len(x)),dtype=bool)
    edge_mask[[0,-1],:]=True; edge_mask[:,[0,-1]]=True
    result['window_edge_sampling']={'points_per_time':int(edge_mask.sum()),
        'corners_counted_once':True,'scope':'all four observation-window edges'}
    for branch,p,r in [('u1',u1,ref_u1),('u2',u2,ref_u2),('total',total,ref_u)]:
        result['all_times'][branch]=metrics(p,r); result['t1'][branch]=metrics(p[-1],r[-1])
        result['initial'][branch]=metrics(p[0],r[0]); result['window_edges'][branch]=metrics(p[:,edge_mask],r[:,edge_mask])
    check_times=times[[0,1,8,40,80]]
    coarse=predict_u1(model,config,check_times,x,config['prediction_order'])
    fine=predict_u1(model,config,check_times,x,config['quadrature_order'])
    result['pinn_inverse_quadrature_refinement_max']=float(abs(coarse-fine).max())
    result['pinn_inverse_imaginary_mae']=float(abs(u1_complex.imag).mean())
    result['pinn_inverse_imaginary_max']=float(abs(u1_complex.imag).max())
    result['pinn_complex_inverse_error']=metrics(u1_complex,ref_u1)
    teacher,_=numerical.solve_zero_ic_rk4(times,kx,ky,forcing(kx,ky,config),config['alpha'],config['c'],config['dtmax'])
    result['numerical_teacher_vs_reference']=metrics(inverse(teacher,x,k,w),ref_u2)
    result['symmetry_projection_used']=False; result['radial_network_reduction_used']=False
    save_json(out/'metrics.json',result)
    np.savez_compressed(out/'predictions.npz',x=x,times=times,u1=u1,u1_imaginary=u1_complex.imag,u2=u2,total=total,
        reference_u1=ref_u1,reference_u2=ref_u2,reference_total=ref_u)
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for row,(name,p,r) in enumerate([('u1',u1,ref_u1),('u2',u2,ref_u2),('total',total,ref_u)]):
        for col,(label,value) in enumerate([('Reference',r[-1]),('Network',p[-1]),('Absolute error',abs(p[-1]-r[-1]))]):
            im=axes[row,col].imshow(value.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='viridis' if col<2 else 'magma')
            axes[row,col].set_title(f'{name}: {label}, t=1'); fig.colorbar(im,ax=axes[row,col],shrink=.8)
    fig.savefig(out/'branch_comparison.png',dpi=160); plt.close(fig)
    plots.make_plots(out)
    print('FINAL METRICS',json.dumps(result['all_times']),flush=True)
    finalize(out,config)


def finalize(out,config):
    standard=out.parent.resolve()==(ROOT/'other').resolve()
    image=ROOT/'images'/out.name if standard else out/'images'
    for f in out.glob('*.png'):
        target=(out/'training_image' if 'loss' in f.name else image)/f.name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(f),str(target))
    rows=[]
    for path in sorted((ROOT/'other').glob('alpha_*/metrics.json')):
        m=json.loads(path.read_text()); row={'alpha':m['config']['alpha']}
        for section,prefix in [('all_times',''),('t1','t1_')]:
            for branch in ['u1','u2','total']:
                for key,value in m[section][branch].items(): row[f'{branch}_{prefix}{key}']=value
        rows.append(row)
    if rows:
        with (ROOT/'results_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alpha',type=float,default=1.6)
    p.add_argument('--stage',choices=['all','data','pinn','fno','evaluate'],default='all')
    p.add_argument('--out',type=Path)
    p.add_argument('--pinn-epochs',type=int,default=2500); p.add_argument('--pinn-lbfgs',type=int,default=500)
    p.add_argument('--fno-steps',type=int,default=2500); p.add_argument('--fno-lbfgs',type=int,default=120)
    p.add_argument('--fno-refine',type=int,default=1200)
    p.add_argument('--shift-x',type=float,default=0.); p.add_argument('--shift-y',type=float,default=0.)
    p.add_argument('--sigma-x',type=float,default=1.); p.add_argument('--sigma-y',type=float,default=1.)
    args=p.parse_args()
    if not 0<args.alpha<=2 or min(args.sigma_x,args.sigma_y)<=0: p.error('Invalid alpha or initial widths')
    config=dict(alpha=args.alpha,c=1.,kmax=8.,xmax=5.,nx=64,m=80,
        quadrature_order=128,reference_order=192,prediction_order=96,dtmax=1/1280,
        center=[args.shift_x,args.shift_y],widths=[args.sigma_x,args.sigma_y],
        symmetry_projection=False,radial_reduction=False)
    out=args.out or ROOT/'other'/('alpha_'+str(args.alpha).replace('.','_')); out.mkdir(parents=True,exist_ok=True)
    cfgpath=out/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=config: raise ValueError('Existing run configuration differs; choose a fresh --out')
    save_json(cfgpath,config)
    if args.stage in ['all','data']: make_dataset(out,config)
    if args.stage in ['all','pinn']:
        import cartesian_pinn
        cartesian_pinn.train_pinn(out,args.alpha,c=config['c'],kmax=config['kmax'],
            epochs=args.pinn_epochs,lbfgs_steps=args.pinn_lbfgs,
            initial_spectrum=lambda kx,ky:initial_hat_tensor(kx,ky,config),device=DEVICE)
    if args.stage in ['all','fno']:
        fno_model.train_fno(out,args.fno_steps,args.fno_lbfgs)
        if args.fno_refine:
            for name in ['fno.pt','fno_history.csv','fno_checks.json']:
                shutil.copy2(out/name,out/name.replace('fno','fno_first_pass',1))
            fno_model.train_fno(out,0,args.fno_refine,True)
    if args.stage in ['all','evaluate']: evaluate(out,config)


if __name__=='__main__': main()
