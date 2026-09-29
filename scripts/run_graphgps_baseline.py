#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, copy
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from care_geollm.baselines import GraphGPSLite
from care_geollm.metrics import regression_metrics
from care_geollm.splits import spatial_group_split
from care_geollm.experiments.io import load_graph_dir,append_summary_row,save_metrics
from care_geollm.utils import seed_everything

def predict(model,graphs,device):
    model.eval(); y=[]; p=[]
    with torch.no_grad():
        for g in graphs:
            g=g.to(device); y.append(g.target_delta.cpu().numpy()); p.append(model(g).cpu().numpy())
    return np.stack(y),np.stack(p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/processed'); ap.add_argument('--city',default='Barcelona'); ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',default='results/baselines'); a=ap.parse_args(); seed_everything(a.seed)
    graphs=[g for g in load_graph_dir(a.data) if g.city.lower()==a.city.lower()]; s=spatial_group_split(graphs); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=GraphGPSLite().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.05); best=None; bestv=1e9
    for ep in range(a.epochs):
        model.train()
        for g in s.get('train',[]): g=g.to(dev); opt.zero_grad(); loss=F.mse_loss(model(g),g.target_delta); loss.backward(); opt.step()
        if s.get('validation'):
            yt,yp=predict(model,s['validation'],dev); v=regression_metrics(yt,yp)['overall']['RMSE']
            if v<bestv: bestv=v; best=copy.deepcopy(model.state_dict())
    if best: model.load_state_dict(best)
    yt,yp=predict(model,s.get('test',[]),dev); m=regression_metrics(yt,yp); Path(a.out).mkdir(parents=True,exist_ok=True); save_metrics(m,Path(a.out)/f'GraphGPS_{a.city}.json'); append_summary_row({'method':'GraphGPSLite','city':a.city,**m['overall']},Path(a.out)/'baselines.csv'); print(m['overall'])
if __name__=='__main__': main()
