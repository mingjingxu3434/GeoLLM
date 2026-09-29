import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import fourier_position, dense_adjacency
from .language import PlanningLanguageAdapter

class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.type_emb = nn.Embedding(cfg.node_type_count, 32)
        self.prov_emb = nn.Embedding(cfg.provenance_count, 16)
        pos_dim = 2 * 2 * 16
        in_dim = cfg.node_feat_dim + cfg.node_feat_dim + 32 + 16 + pos_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, 384), nn.GELU(), nn.LayerNorm(384), nn.Dropout(cfg.dropout),
            nn.Linear(384, cfg.hidden_dim), nn.GELU(), nn.LayerNorm(cfg.hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(cfg.edge_feat_dim, 128), nn.GELU(), nn.Linear(128, cfg.edge_hidden_dim))

    def forward(self, x, node_type, provenance, missing, pos, edge_attr):
        p = fourier_position(pos, 16)
        h = self.node_mlp(torch.cat([x, missing.float(), self.type_emb(node_type), self.prov_emb(provenance), p], -1))
        e = self.edge_mlp(edge_attr)
        return h, e

class EntityAlign(nn.Module):
    def __init__(self, d=256, num_types=4):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.type_proj = nn.Linear(d, num_types)
        self.selector = nn.Sequential(nn.Linear(d,128), nn.GELU(), nn.Linear(128,1))

    def forward(self, q, h, node_type):
        # q [B,K,D], h [N,D]; demo assumes B=1 graph
        q = q[0]
        score = torch.einsum('nd,kd->kn', self.W(h), q) / math.sqrt(h.size(-1))
        allowed_type = self.type_proj(q).softmax(-1)[:, node_type]
        selector_bias = self.selector(h).squeeze(-1).unsqueeze(0)
        return torch.sigmoid(score + selector_bias) * allowed_type

class DirectUpdate(nn.Module):
    def __init__(self, d=256, feat_dim=24):
        super().__init__()
        self.to_update = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,feat_dim), nn.Tanh())
    def forward(self, q, masks):
        u = self.to_update(q[0])                        # [K,F]
        return torch.einsum('kn,kf->nf', masks, u) / max(1, q.size(1))

class HyperNetwork(nn.Module):
    """Produces per-token low-rank factors and bounded magnitude gates."""
    def __init__(self, d=256, rank=8):
        super().__init__()
        self.rank = rank
        self.core = nn.Sequential(nn.Linear(d,256), nn.GELU(), nn.Linear(256,256), nn.GELU())
        self.A = nn.Linear(256, d*rank)
        self.B = nn.Linear(256, d*rank)
        self.alpha = nn.Linear(256,1)
    def forward(self, q):
        z = self.core(q[0])
        K,D = z.size(0), q.size(-1)
        A = self.A(z).view(K,D,self.rank)
        B = self.B(z).view(K,D,self.rank)
        alpha = torch.tanh(self.alpha(z)).squeeze(-1)
        return A,B,alpha

class CounterfactualBlock(nn.Module):
    def __init__(self, d=256, heads=8, ffn=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d,d)
        self.ffn = nn.Sequential(nn.Linear(d,ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn,d))
        self.heads=heads; self.dropout=nn.Dropout(dropout)

    def forward(self, h, adj, modulation=None):
        x=self.norm1(h); N,D=x.shape; H=self.heads; dh=D//H
        q,k,v=self.qkv(x).chunk(3,-1)
        if modulation is not None:
            # localized low-rank delta applied as a state-conditioned projection surrogate
            A,B,gate = modulation                 # [N,D,r],[N,D,r],[N]
            delta = torch.einsum('ndr,ndr,nd->nd', A, B, x) / A.size(-1)
            q = q + gate.unsqueeze(-1)*delta
            v = v + gate.unsqueeze(-1)*delta
        q=q.view(N,H,dh).transpose(0,1); k=k.view(N,H,dh).transpose(0,1); v=v.view(N,H,dh).transpose(0,1)
        attn=torch.einsum('hnd,hmd->hnm',q,k)/math.sqrt(dh)
        attn=attn.masked_fill(~adj.unsqueeze(0), float('-inf'))
        attn=F.softmax(attn,-1)
        y=torch.einsum('hnm,hmd->hnd',self.dropout(attn),v).transpose(0,1).reshape(N,D)
        h=h+self.out(y)
        h=h+self.ffn(self.norm2(h))
        return h

class GeoTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.blocks=nn.ModuleList([CounterfactualBlock(cfg.hidden_dim,cfg.operator_heads,cfg.operator_ffn_dim,cfg.dropout) for _ in range(cfg.operator_layers)])
    def forward(self,h,edge_index,modulation=None):
        adj=dense_adjacency(edge_index,h.size(0),h.device)
        for block in self.blocks: h=block(h,adj,modulation)
        return h

class MultiTargetDecoder(nn.Module):
    def __init__(self,d=256,num_targets=4,dropout=.1):
        super().__init__()
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2)) for _ in range(num_targets)])
    def forward(self,h):
        g=h.mean(0)
        o=torch.stack([head(g) for head in self.heads])
        return o[:,0], o[:,1].clamp(-5,3)

class CAREGeoLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.spatial=SpatialEncoder(cfg)
        self.language=PlanningLanguageAdapter(cfg.language_vocab_size,cfg.language_dim,cfg.language_heads,cfg.language_layers,cfg.language_ffn_dim,cfg.max_intervention_tokens,cfg.intervention_dim,cfg.dropout)
        self.align=EntityAlign(cfg.hidden_dim,cfg.node_type_count)
        self.direct=DirectUpdate(cfg.intervention_dim,cfg.node_feat_dim)
        self.hyper=HyperNetwork(cfg.intervention_dim,cfg.hyper_rank)
        self.operator=GeoTransformer(cfg)
        self.decoder=MultiTargetDecoder(cfg.hidden_dim,cfg.num_targets,cfg.dropout)
        self.plan_to_spatial=nn.Linear(cfg.intervention_dim,cfg.hidden_dim)

    def _localized_modulation(self,A,B,alpha,masks):
        # token low-rank factors -> entity-local factors
        weights=(masks*alpha[:,None]).t()                # [N,K]
        denom=weights.abs().sum(-1,keepdim=True).clamp_min(1e-6)
        w=weights/denom
        Ae=torch.einsum('nk,kdr->ndr',w,A); Be=torch.einsum('nk,kdr->ndr',w,B)
        gate=weights.sum(-1).tanh()
        return Ae,Be,gate

    def forward(self,batch):
        h0,_=self.spatial(batch.x,batch.node_type,batch.provenance,batch.missing,batch.pos,batch.edge_attr)
        q,fields=self.language(batch.plan_tokens)
        masks=self.align(q,h0,batch.node_type)
        A,B,alpha=self.hyper(q)
        hf=self.operator(h0,batch.edge_index,None)
        y0,s0=self.decoder(hf)
        dx=self.direct(q,masks) if self.cfg.use_direct_update else torch.zeros_like(batch.x)
        hp0,_=self.spatial(batch.x+dx,batch.node_type,batch.provenance,batch.missing,batch.pos,batch.edge_attr)
        if self.cfg.static_plan_conditioning:
            hp0 = hp0 + self.plan_to_spatial(q.mean(dim=1)).squeeze(0).unsqueeze(0)
        mod=self._localized_modulation(A,B,alpha,masks) if self.cfg.use_language_operator else None
        hp=self.operator(hp0,batch.edge_index,mod)
        yp,sp=self.decoder(hp)
        scale=0.5*(s0+sp) if self.cfg.use_uncertainty_head else torch.zeros_like(sp)
        return {'delta':yp-y0,'scale':scale,'y0':y0,'yp':yp,'h0':hf,'hp':hp,'masks':masks,'fields':fields}
