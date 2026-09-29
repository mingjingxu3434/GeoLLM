#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from care_geollm.config import ModelConfig,TrainConfig
from care_geollm.splits import spatial_group_split
from care_geollm.corruption import spatial_block_corrupt
from care_geollm.experiments.io import load_graph_dir,append_summary_row,save_metrics
from care_geollm.experiments.engine import ExperimentEngine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed'); p.add_argument('--city',default='Barcelona'); p.add_argument('--epochs',type=int,default=120); p.add_argument('--seed',type=int,default=7); p.add_argument('--out',default='results/robustness'); a=p.parse_args()
    graphs=[g for g in load_graph_dir(a.data) if g.city.lower()==a.city.lower()]; split=spatial_group_split(graphs); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    eng=ExperimentEngine(ModelConfig(),TrainConfig(epochs=a.epochs,seed=a.seed)); eng.fit(split.get('train',[]),split.get('validation',[]),str(outdir/'best.pt'),verbose=False); cal=eng.fit_calibrator(split.get('calibration',[]))
    result={}
    for frac in [0.0,0.05,0.10,0.20,0.30]:
        test=[spatial_block_corrupt(g,frac,a.seed+i) for i,g in enumerate(split.get('test',[]))]
        m,*_=eng.evaluate(test,cal); result[str(frac)]=m; append_summary_row({'corruption':frac,'city':a.city,**m['overall']},outdir/'robustness.csv'); print(frac,m['overall'])
    save_metrics(result,outdir/'robustness.json',{'city':a.city})
if __name__=='__main__': main()
