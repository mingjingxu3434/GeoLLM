#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
from care_geollm.config import ModelConfig,TrainConfig
from care_geollm.splits import spatial_group_split
from care_geollm.experiments.io import load_graph_dir,save_metrics,append_summary_row
from care_geollm.experiments.engine import ExperimentEngine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed'); p.add_argument('--city'); p.add_argument('--epochs',type=int,default=120); p.add_argument('--seed',type=int,default=7); p.add_argument('--out',default='results/main'); a=p.parse_args()
    graphs=load_graph_dir(a.data)
    if a.city: graphs=[g for g in graphs if g.city.lower()==a.city.lower()]
    if not graphs: raise SystemExit('No graphs found')
    split=spatial_group_split(graphs); mc=ModelConfig(); tc=TrainConfig(epochs=a.epochs,seed=a.seed)
    eng=ExperimentEngine(mc,tc); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    fit=eng.fit(split.get('train',[]),split.get('validation',[]),str(outdir/'best.pt'))
    cal=eng.fit_calibrator(split.get('calibration',[])); metrics,yt,yp,ls=eng.evaluate(split.get('test',[]),cal)
    payload={'fit':fit,'counts':{k:len(v) for k,v in split.items()},'city':a.city or 'ALL'}
    save_metrics(metrics,outdir/'metrics.json',payload)
    row={'experiment':'CARE GeoLLM','city':a.city or 'ALL',**metrics['overall']}; append_summary_row(row,outdir/'summary.csv')
    print(json.dumps({'fit':fit,'overall':metrics['overall'],'counts':payload['counts']},indent=2))
if __name__=='__main__': main()
