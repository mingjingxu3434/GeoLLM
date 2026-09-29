#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Template utility for assembling official baseline embeddings.
Input CSV columns: community_id, split, y0,y1,y2,y3, embedding_path (.npy).
"""
import argparse, pandas as pd, numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out',required=True); a=p.parse_args(); df=pd.read_csv(a.csv)
    X=np.stack([np.load(p) for p in df.embedding_path]); y=df[['y0','y1','y2','y3']].to_numpy('float32'); np.savez_compressed(a.out,embedding=X,target=y,split=df.split.astype(str).to_numpy(),community_id=df.community_id.astype(str).to_numpy()); print('saved',a.out,X.shape)
if __name__=='__main__': main()
