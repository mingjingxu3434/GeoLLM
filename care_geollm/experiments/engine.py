import copy, math, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..model import CAREGeoLLM
from ..metrics import regression_metrics
from ..conformal import ConformalCalibrator
from ..utils import seed_everything

class ExperimentEngine:
    def __init__(self, model_cfg, train_cfg, device=None):
        self.mc=model_cfg; self.tc=train_cfg
        seed_everything(train_cfg.seed)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=CAREGeoLLM(model_cfg).to(self.device)
        # Separate language LR as specified in manuscript.
        lang=list(self.model.language.parameters()); lang_ids={id(p) for p in lang}
        spatial=[p for p in self.model.parameters() if id(p) not in lang_ids]
        self.opt=AdamW([{'params':spatial,'lr':train_cfg.spatial_lr},{'params':lang,'lr':train_cfg.language_lr}],weight_decay=train_cfg.weight_decay)
        self.scheduler=CosineAnnealingLR(self.opt,T_max=max(1,train_cfg.epochs-train_cfg.warmup_epochs))
        self.scaler=torch.amp.GradScaler('cuda',enabled=(train_cfg.mixed_precision and self.device.type=='cuda'))
        self.best_state=None; self.best_val=float('inf')

    def _loss(self,out,g):
        if self.mc.use_uncertainty_head:
            pred=((g.target_delta-out['delta'])**2)/(2*torch.exp(2*out['scale']))+out['scale']
            lp=pred.mean()
        else: lp=F.mse_loss(out['delta'],g.target_delta)
        la=out['delta'].new_tensor(0.)
        if g.align_target is not None:
            k=min(out['masks'].shape[0],g.align_target.shape[0]); n=min(out['masks'].shape[1],g.align_target.shape[1])
            la=F.binary_cross_entropy(out['masks'][:k,:n].clamp(1e-5,1-1e-5),g.align_target[:k,:n])
        lc=out['delta'].new_tensor(0.)
        if g.untreated_mask is not None and g.untreated_mask.any(): lc=((out['hp'][g.untreated_mask]-out['h0'][g.untreated_mask])**2).mean()
        ld=F.relu(-out['delta']).mean()
        return lp+self.tc.lambda_align*la+self.tc.lambda_cons*lc+self.tc.lambda_dir*ld

    @torch.no_grad()
    def predict(self,graphs):
        self.model.eval(); ys=[]; ps=[]; ss=[]
        for g in graphs:
            g=g.to(self.device); out=self.model(g)
            ys.append(g.target_delta.detach().cpu().numpy()); ps.append(out['delta'].cpu().numpy()); ss.append(out['scale'].cpu().numpy())
        if not ys: return np.empty((0,4)),np.empty((0,4)),np.empty((0,4))
        return np.stack(ys),np.stack(ps),np.stack(ss)

    def fit(self,train_graphs,val_graphs=None,checkpoint=None,verbose=True):
        best_epoch=0; stale=0
        for epoch in range(self.tc.epochs):
            self.model.train(); self.opt.zero_grad(set_to_none=True); running=[]
            order=np.random.default_rng(self.tc.seed+epoch).permutation(len(train_graphs))
            for step,idx in enumerate(order):
                g=train_graphs[int(idx)].to(self.device)
                with torch.autocast(device_type=self.device.type,enabled=(self.tc.mixed_precision and self.device.type=='cuda')):
                    out=self.model(g); loss=self._loss(out,g)/max(1,self.tc.gradient_accumulation)
                self.scaler.scale(loss).backward(); running.append(float(loss.detach())*max(1,self.tc.gradient_accumulation))
                if (step+1)%self.tc.gradient_accumulation==0 or step+1==len(order):
                    self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.tc.grad_clip)
                    self.scaler.step(self.opt); self.scaler.update(); self.opt.zero_grad(set_to_none=True)
            if epoch>=self.tc.warmup_epochs: self.scheduler.step()
            if val_graphs:
                yt,yp,_=self.predict(val_graphs); val=regression_metrics(yt,yp)['overall']['RMSE']
            else: val=float(np.mean(running))
            if val<self.best_val:
                self.best_val=val; best_epoch=epoch+1; stale=0; self.best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                if checkpoint: self.save(checkpoint,best_epoch)
            else: stale+=1
            if verbose: print(f'epoch={epoch+1:03d} train_loss={np.mean(running):.5f} val_rmse={val:.5f}')
            if stale>=self.tc.patience: break
        if self.best_state is not None: self.model.load_state_dict(self.best_state)
        return {'best_epoch':best_epoch,'best_val_rmse':self.best_val}

    def fit_calibrator(self,cal_graphs):
        yt,yp,ls=self.predict(cal_graphs)
        if len(yt)==0: return None
        c=ConformalCalibrator(self.tc.coverage)
        c.fit(torch.tensor(yp),torch.tensor(yt),torch.tensor(ls)); return c

    def evaluate(self,graphs,calibrator=None):
        yt,yp,ls=self.predict(graphs); m=regression_metrics(yt,yp)
        if calibrator is not None and len(yt):
            lo,hi=calibrator.interval(torch.tensor(yp),torch.tensor(ls)); y=torch.tensor(yt)
            cov=((y>=lo)&(y<=hi)).float().mean(0).numpy().tolist(); width=(hi-lo).mean(0).numpy().tolist()
            m['coverage90']=cov; m['interval_width']=width
        return m,yt,yp,ls

    def save(self,path,epoch=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':self.model.state_dict(),'model_config':self.mc.to_dict(),'train_config':self.tc.to_dict(),'epoch':epoch},path)
