"""Trainable baseline adapters with a shared quantitative interface.

GraphGPSLite is implemented directly. Foundation-model baselines (SatMAE, Scale-MAE,
UrbanCLIP, RemoteCLIP, GeoChat, UrbanLLM, AnySat) are represented by a common embedding
regressor that consumes pre-extracted official embeddings. This avoids falsely claiming to
reimplement large external checkpoints in a small repository: use scripts/extract_embeddings.py
(or each project's official code) to save one embedding per community, then train the same head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import dense_adjacency

class GraphGPSLite(nn.Module):
    def __init__(self, feat_dim=24, hidden=256, heads=8, layers=4, targets=4, dropout=.1):
        super().__init__(); self.inp=nn.Linear(feat_dim,hidden); self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(hidden,heads,hidden*4,dropout,batch_first=True,norm_first=True,activation='gelu')
            for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(hidden,128),nn.GELU(),nn.Linear(128,targets))
    def forward(self,batch):
        h=self.inp(batch.x).unsqueeze(0)
        # Transformer global channel; local graph information is injected as one-step neighbor mean.
        n=batch.x.size(0); adj=dense_adjacency(batch.edge_index,n,batch.x.device).float()
        deg=adj.sum(-1,keepdim=True).clamp_min(1); local=(adj/deg)@h.squeeze(0); h=h+local.unsqueeze(0)
        for layer in self.layers: h=layer(h)
        return self.head(h.mean(1)).squeeze(0)

class EmbeddingRegressor(nn.Module):
    def __init__(self,input_dim,targets=4,hidden=512):
        super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,256),nn.GELU(),nn.Linear(256,targets))
    def forward(self,x): return self.net(x)

FOUNDATION_BASELINES=['SatMAE','ScaleMAE','UrbanCLIP','RemoteCLIP','GeoChat','UrbanLLM','AnySat']
