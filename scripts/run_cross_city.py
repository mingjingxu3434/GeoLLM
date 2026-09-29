#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from care_geollm.config import ModelConfig,TrainConfig
from care_geollm.splits import leave_one_city_out, spatial_group_split
from care_geollm.experiments.io import load_graph_dir,append_summary_row,save_metrics
from care_geollm.experiments.engine import ExperimentEngine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed'); p.add_argument('--epochs',type=int,default=120); p.add_argument('--seed',type=int,default=7); p.add_argument('--out',default='results/cross_city'); a=p.parse_args()
    graphs=load_graph_dir(a.data); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True); results={}
    for city in ['Barcelona','Rotterdam','Vienna']:
        other,target=leave_one_city_out(graphs,city); s=spatial_group_split(other)
        eng=ExperimentEngine(ModelConfig(),TrainConfig(epochs=a.epochs,seed=a.seed)); eng.fit(s.get('train',[]),s.get('validation',[]),str(outdir/f'holdout_{city}.pt'),verbose=False)
        cal=eng.fit_calibrator(s.get('calibration',[])); m,*_=eng.evaluate(target,cal); results[city]=m
        append_summary_row({'heldout_city':city,**m['overall']},outdir/'cross_city.csv'); print(city,m['overall'])
    save_metrics(results,outdir/'cross_city.json')
if __name__=='__main__': main()
