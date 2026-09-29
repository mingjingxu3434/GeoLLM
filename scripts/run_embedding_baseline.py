#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from care_geollm.baselines import EmbeddingRegressor,FOUNDATION_BASELINES
from care_geollm.metrics import regression_metrics
from care_geollm.experiments.io import append_summary_row,save_metrics

def main():
    p=argparse.ArgumentParser(description='Shared regression head for pre-extracted official baseline embeddings')
    p.add_argument('--method',required=True,choices=FOUNDATION_BASELINES); p.add_argument('--npz',required=True,help='NPZ with embedding, target, split arrays'); p.add_argument('--epochs',type=int,default=200); p.add_argument('--out',default='results/baselines'); a=p.parse_args()
    z=np.load(a.npz,allow_pickle=True); X=z['embedding'].astype('float32'); y=z['target'].astype('float32'); split=z['split'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=EmbeddingRegressor(X.shape[1]).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.05)
    tr=np.where(split=='train')[0]; te=np.where(split=='test')[0]
    xt=torch.tensor(X[tr],device=dev); yt=torch.tensor(y[tr],device=dev)
    for _ in range(a.epochs): model.train(); opt.zero_grad(); loss=F.mse_loss(model(xt),yt); loss.backward(); opt.step()
    model.eval(); pred=model(torch.tensor(X[te],device=dev)).detach().cpu().numpy(); m=regression_metrics(y[te],pred); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); save_metrics(m,out/f'{a.method}.json'); append_summary_row({'method':a.method,**m['overall']},out/'baselines.csv'); print(m['overall'])
if __name__=='__main__': main()
