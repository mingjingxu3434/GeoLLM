#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
from care_geollm.config import ModelConfig,TrainConfig
from care_geollm.splits import spatial_group_split
from care_geollm.experiments.io import load_graph_dir,append_summary_row,save_metrics
from care_geollm.experiments.engine import ExperimentEngine

VARIANTS={
 'full':{},
 'no_language_operator':{'use_language_operator':False,'static_plan_conditioning':False},
 'no_consistency':{},
 'static_attention':{'use_language_operator':False,'static_plan_conditioning':True},
 'no_uncertainty':{'use_uncertainty_head':False},
}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed'); p.add_argument('--city',default='Barcelona'); p.add_argument('--epochs',type=int,default=120); p.add_argument('--seed',type=int,default=7); p.add_argument('--out',default='results/ablation'); a=p.parse_args()
    graphs=[g for g in load_graph_dir(a.data) if g.city.lower()==a.city.lower()]; split=spatial_group_split(graphs); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    allres={}
    for name,mods in VARIANTS.items():
        mc=ModelConfig(**mods); tc=TrainConfig(epochs=a.epochs,seed=a.seed)
        if name=='no_consistency': tc.lambda_cons=0.0
        eng=ExperimentEngine(mc,tc); eng.fit(split.get('train',[]),split.get('validation',[]),str(outdir/f'{name}.pt'),verbose=False)
        cal=eng.fit_calibrator(split.get('calibration',[])) if mc.use_uncertainty_head else None
        metrics,*_=eng.evaluate(split.get('test',[]),cal); allres[name]=metrics
        append_summary_row({'variant':name,'city':a.city,**metrics['overall']},outdir/'ablation.csv')
        print(name,metrics['overall'])
    save_metrics(allres,outdir/'ablation.json',{'city':a.city})
if __name__=='__main__': main()
