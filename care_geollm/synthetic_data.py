import torch
from .data import GraphBatch
from .tokenizer import stable_hash_tokenize

def make_sample(cfg,n=48,seed=7,city='Barcelona',community_id=None):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,cfg.node_feat_dim,generator=g)*.4
    node_type=torch.randint(0,cfg.node_type_count,(n,),generator=g)
    provenance=torch.randint(0,cfg.provenance_count,(n,),generator=g)
    missing=(torch.rand(n,cfg.node_feat_dim,generator=g)<.1).float()
    pos=torch.rand(n,2,generator=g)*2-1
    edges=[]
    for i in range(n): edges += [(i,(i+1)%n),((i+1)%n,i)]
    for _ in range(n*2):
        a=int(torch.randint(0,n,(1,),generator=g)); b=int(torch.randint(0,n,(1,),generator=g))
        if a!=b: edges += [(a,b),(b,a)]
    edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()
    edge_attr=torch.randn(edge_index.size(1),cfg.edge_feat_dim,generator=g)*.2
    plans=[
        'install lifts in stair only blocks and improve insulation in older buildings',
        'shade steep pedestrian routes and preserve mature tree canopy',
        'remove access barriers and electrify heating systems',
        'relocate daily services closer to older residents and retrofit envelopes',
    ]
    plan=plans[seed%len(plans)]
    toks=stable_hash_tokenize(plan,cfg.language_vocab_size,cfg.max_plan_len)
    K=cfg.max_intervention_tokens
    align=torch.zeros(K,n); selected=(node_type==0); align[0,selected]=1.0
    untreated=~selected
    city_bias={'Barcelona':0.00,'Rotterdam':0.02,'Vienna':-0.01}.get(city,0.0)
    base=torch.tensor([.18,.14,.11,.08])+city_bias
    signal=torch.stack([x[:,0].mean(),x[:,1].mean(),x[:,2].mean(),x[:,3].mean()])*.03
    target=base+signal+torch.randn(4,generator=g)*.008
    cid=community_id or f'{city[:3].lower()}_{seed:04d}'
    return GraphBatch(x,node_type,provenance,missing,pos,edge_index,edge_attr,toks,target,align,untreated,
                      city=city,community_id=cid,intervention_family=['access','shade','energy','mixed'][seed%4],
                      metadata={'spatial_group':f'{city[:3]}_{seed//3:03d}','plan_text':plan,'synthetic':True})

def make_benchmark(cfg,per_city=20,seed=7):
    out=[]
    cities=['Barcelona','Rotterdam','Vienna']
    for cidx,city in enumerate(cities):
        for i in range(per_city): out.append(make_sample(cfg,n=40+(i%5)*4,seed=seed+cidx*1000+i,city=city))
    return out
