from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from ..data import GraphBatch, save_graph
from ..tokenizer import stable_hash_tokenize
from .common import CANONICAL_FEATURES, robust_standardize, normalize_relative_xy, ensure_dir
from .eubucco import load_eubucco
from .raster import sample_raster

def knn_edges(xy, k=8):
    n=len(xy)
    if n < 2: return np.zeros((2,0),np.int64), np.zeros((0,8),np.float32)
    kk=min(k+1,n)
    nbr=NearestNeighbors(n_neighbors=kk).fit(xy)
    dist, ind=nbr.kneighbors(xy)
    edges=[]; attrs=[]
    for i in range(n):
        for d,j in zip(dist[i,1:],ind[i,1:]):
            dx,dy=xy[j]-xy[i]; bearing=np.arctan2(dy,dx)
            edges.append((i,int(j)))
            a=np.zeros(8,np.float32); a[0]=d; a[1]=np.sin(bearing); a[2]=np.cos(bearing); a[3]=1.0
            attrs.append(a)
    return np.asarray(edges,np.int64).T, np.asarray(attrs,np.float32)

def _fit_dim(x, dim):
    if x.shape[1] == dim: return x
    if x.shape[1] > dim: return x[:,:dim]
    return np.pad(x,((0,0),(0,dim-x.shape[1])))

def build_city_community(buildings_path, city, community_id, plan_text,
                         target_delta=None, wsf_fraction=None, wsf_height=None, wsf_volume=None,
                         worldpop_total=None, worldpop_65plus=None, bbox=None,
                         node_feat_dim=24, edge_feat_dim=8, vocab_size=4096, spatial_group=None):
    b=load_eubucco(buildings_path,bbox=bbox)
    raw=b['features'].copy(); xy=b['xy']; n=len(xy)
    if n == 0: raise ValueError('No buildings found for requested community')
    # enrich canonical columns with raster samples at building centroids
    for path,col in [(worldpop_total,18),(worldpop_65plus,19),(wsf_fraction,20),(wsf_height,21),(wsf_volume,22)]:
        if path:
            raw[:,col]=sample_raster(path,xy,b['crs'])
    missing=(~np.isfinite(raw)).astype(np.float32)
    x=robust_standardize(raw)
    x=_fit_dim(x,node_feat_dim); missing=_fit_dim(missing,node_feat_dim)
    pos=normalize_relative_xy(xy)
    edge_index,edge_attr=knn_edges(xy,8); edge_attr=_fit_dim(edge_attr,edge_feat_dim)
    node_type=np.zeros(n,np.int64) # buildings in this base adapter
    provenance=np.zeros(n,np.int64)
    target=np.zeros(4,np.float32) if target_delta is None else np.asarray(target_delta,np.float32)
    graph=GraphBatch(
        x=torch.from_numpy(x), node_type=torch.from_numpy(node_type), provenance=torch.from_numpy(provenance),
        missing=torch.from_numpy(missing), pos=torch.from_numpy(pos), edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr), plan_tokens=stable_hash_tokenize(plan_text,vocab_size),
        target_delta=torch.from_numpy(target), city=city, community_id=community_id,
        intervention_family='mixed', metadata={'spatial_group':spatial_group or community_id,
        'source_crs':b['crs'],'building_count':n,'feature_names':CANONICAL_FEATURES[:node_feat_dim],
        'plan_text':plan_text}
    )
    return graph

def save_community(graph, out_dir):
    ensure_dir(out_dir); path=Path(out_dir)/f'{graph.city}__{graph.community_id}.pt'; save_graph(graph,str(path)); return path
