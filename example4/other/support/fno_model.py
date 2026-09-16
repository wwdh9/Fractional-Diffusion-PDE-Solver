"""Ordinary 2D FNO; no rotations, reflections, averaging or radial inputs."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

class SpectralConv2d(nn.Module):
    """Proper real FFT spectral layer: both positive AND negative kx blocks."""
    def __init__(self,width,modes):
        super().__init__(); self.modes=modes
        self.w1=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)/width)
        self.w2=nn.Parameter(torch.randn(width,width,modes,modes,dtype=torch.cfloat)/width)
    def forward(self,x):
        b,c,h,w=x.shape; m=self.modes
        ft=torch.fft.rfft2(x)
        out=torch.zeros(b,c,h,w//2+1,device=x.device,dtype=ft.dtype)
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',ft[:,:,:m,:m],self.w1)
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',ft[:,:,-m:,:m],self.w2)
        return torch.fft.irfft2(out,s=(h,w))

class FNO2d(nn.Module):
    def __init__(self,width=24,modes=12,depth=3):
        super().__init__()
        self.lift=nn.Conv2d(4,width,1)
        self.spectral=nn.ModuleList([SpectralConv2d(width,modes) for _ in range(depth)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(depth)])
        self.project=nn.Sequential(nn.Conv2d(width,64,1),nn.GELU(),nn.Conv2d(64,1,1))
        self.register_buffer('f_scale',torch.tensor(1.))
        self.register_buffer('g_scale',torch.tensor(1.))
    def forward(self,f,t):
        # Physical t is supplied explicitly. Batch order/size cannot change it.
        b,h,w=f.shape
        xv=torch.linspace(-1,1,h,device=f.device).view(1,1,h,1).expand(b,1,h,w)
        yv=torch.linspace(-1,1,w,device=f.device).view(1,1,1,w).expand(b,1,h,w)
        tv=t.view(b,1,1,1).expand(b,1,h,w)
        inputs=torch.cat([f[:,None]/self.f_scale,xv,yv,tv],1)
        x=self.lift(inputs)
        # Padding reduces periodic-wrap artifacts in the FNO's feature transform.
        x=F.pad(x,(0,8,0,8))
        for spec,local in zip(self.spectral,self.local): x=F.gelu(spec(x)+local(x))
        correction=self.project(x[:,:,:h,:w]).squeeze(1)
        # PDE supplies g(0)=f(0); no reference data enter this hard condition.
        return f+t[:,None,None]*self.g_scale*correction

def load_fno(out, name='fno.pt'):
    cfg=json.loads((out/'fno_config.json').read_text())
    model=FNO2d(**cfg).to(DEVICE)
    model.load_state_dict(torch.load(out/name,map_location=DEVICE,weights_only=True))
    return model.eval()

def train_fno(out, steps, refine_steps, resume=False):
    seed_all(20260916)
    data=np.load(out/'numerical_training_data.npz')
    tensor=lambda a:torch.tensor(a,dtype=torch.float32,device=DEVICE)
    ft,gt,tt=tensor(data['f_train']),tensor(data['g_train']),tensor(data['train_t'])
    fv,gv,tv=tensor(data['f_val']),tensor(data['g_val']),tensor(data['val_t'])
    config={'width':24,'modes':12,'depth':3}
    save_json(out/'fno_config.json',config)
    model=FNO2d(**config).to(DEVICE)
    model.f_scale.copy_(ft.square().mean().sqrt())
    model.g_scale.copy_(gt.square().mean().sqrt())
    if resume: model.load_state_dict(torch.load(out/'fno.pt',map_location=DEVICE,weights_only=True))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(steps,1),eta_min=1e-5)
    history=[]; best=float('inf'); best_state=None; tic=time.perf_counter()
    def normalized_loss(f,g,t):
        return ((model(f,t)-g)/model.g_scale).square().mean()
    def validation():
        with torch.no_grad():
            return sum(normalized_loss(fv[i:i+4],gv[i:i+4],tv[i:i+4]).item()*len(tv[i:i+4]) for i in range(0,len(tv),4))/len(tv)
    if resume:
        best=validation()
        best_state=copy.deepcopy(model.state_dict())
    for step in range(steps):
        ix=torch.randint(0,len(tt),(6,),device=DEVICE)
        opt.zero_grad(set_to_none=True)
        loss=normalized_loss(ft[ix],gt[ix],tt[ix]); loss.backward(); opt.step(); sched.step()
        if (step+1)%100==0 or step==0:
            val=validation()
            history.append([step+1,loss.item(),val])
            if val<best: best=val; best_state=copy.deepcopy(model.state_dict())
            print('FNO ADAM',history[-1],flush=True)
        if step+1==min(500,steps): torch.save(model.state_dict(),out/'fno_early.pt')
    if best_state: model.load_state_dict(best_state)
    # Chunked full-dataset loss keeps 64^2 data and Fourier modes within 6 GB.
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=refine_steps,
        history_size=12,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    calls=[0]
    def closure():
        opt.zero_grad(set_to_none=True); total=0.
        for i in range(0,len(tt),8):
            loss=normalized_loss(ft[i:i+8],gt[i:i+8],tt[i:i+8])*len(tt[i:i+8])/len(tt)
            loss.backward(); total+=loss.item()
        calls[0]+=1
        if calls[0]%10==0:
            history.append([steps+calls[0],total,float('nan')])
            print('FNO LBFGS',calls[0],total,flush=True)
        return torch.tensor(total,device=DEVICE)
    if refine_steps:
        opt.step(closure)
        val=validation()
        history.append([steps+calls[0],float('nan'),val])
        if val<best: best=val; best_state=copy.deepcopy(model.state_dict())
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(),out/'fno.pt')
    np.savetxt(out/'fno_history.csv',history,delimiter=',',header='step,train_normalized_mse,val_normalized_mse')
    report={'seconds':time.perf_counter()-tic,'validation_normalized_mse':best,
            'steps':steps,'lbfgs_closure_calls':calls[0],
            'time_argument_batch_invariance':None,'training_uses_reference':False}
    with torch.no_grad():
        bat=model(fv[:3],tv[:3])
        singles=torch.cat([model(fv[i:i+1],tv[i:i+1]) for i in range(3)])
        report['time_argument_batch_invariance']=float((bat-singles).abs().max())
    save_json(out/'fno_checks.json',report)
    print('FNO DONE',report,flush=True)

