#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json
from care_geollm.preprocess.build_graph import build_city_community, save_community

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--buildings',required=True); p.add_argument('--city',required=True); p.add_argument('--community-id',required=True)
    p.add_argument('--plan',required=True); p.add_argument('--out-dir',default='data/processed')
    p.add_argument('--target',nargs=4,type=float,default=[0,0,0,0])
    p.add_argument('--wsf-fraction'); p.add_argument('--wsf-height'); p.add_argument('--wsf-volume')
    p.add_argument('--worldpop-total'); p.add_argument('--worldpop-65plus'); p.add_argument('--spatial-group')
    p.add_argument('--bbox',nargs=4,type=float)
    a=p.parse_args()
    g=build_city_community(a.buildings,a.city,a.community_id,a.plan,a.target,
       a.wsf_fraction,a.wsf_height,a.wsf_volume,a.worldpop_total,a.worldpop_65plus,a.bbox,spatial_group=a.spatial_group)
    path=save_community(g,a.out_dir); print(json.dumps({'saved':str(path),'nodes':len(g.x),'edges':g.edge_index.shape[1]}))
if __name__=='__main__': main()
