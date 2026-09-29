#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from urllib.request import urlretrieve

SOURCES = {
    'eubucco': 'https://eubucco.com/data',
    'wsf3d': 'https://download.geoservice.dlr.de/WSF3D/files/',
    'worldpop': 'https://hub.worldpop.org/Global1_2000-2020_AgeSexStructures',
}

def main():
    p=argparse.ArgumentParser(description='Dataset source helper for CARE GeoLLM')
    p.add_argument('--show',action='store_true',help='print manuscript source portals')
    p.add_argument('--url',help='direct downloadable file URL')
    p.add_argument('--out',help='output file for --url')
    a=p.parse_args()
    if a.show or not a.url:
        for k,v in SOURCES.items(): print(f'{k}: {v}')
    if a.url:
        if not a.out: raise SystemExit('--out is required with --url')
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        print('downloading',a.url,'->',a.out); urlretrieve(a.url,a.out); print('done')

if __name__=='__main__': main()
