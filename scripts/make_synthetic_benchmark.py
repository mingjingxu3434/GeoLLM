#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from care_geollm.config import ModelConfig
from care_geollm.synthetic_data import make_benchmark
from care_geollm.data import save_graph

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data/processed'); p.add_argument('--per-city',type=int,default=20); p.add_argument('--seed',type=int,default=7); a=p.parse_args()
    Path(a.out).mkdir(parents=True,exist_ok=True); graphs=make_benchmark(ModelConfig(),a.per_city,a.seed)
    for g in graphs: save_graph(g,str(Path(a.out)/f'{g.city}__{g.community_id}.pt'))
    print(f'saved {len(graphs)} graph files to {a.out}')
if __name__=='__main__': main()
