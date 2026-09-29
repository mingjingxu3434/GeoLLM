from collections import defaultdict
import hashlib

SPLIT_NAMES = ('train', 'calibration', 'validation', 'test')

def _score(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:12],16)

def _allocate_groups(groups):
    groups=sorted(groups,key=lambda g:_score(str(g)))
    n=len(groups)
    if n < 4:
        # Best effort for tiny smoke tests; production datasets should have many spatial groups.
        names=['train','calibration','validation','test']
        return {g:names[min(i,3)] for i,g in enumerate(groups)}
    n_cal=max(1,round(n*.10)); n_val=max(1,round(n*.10)); n_test=max(1,round(n*.10))
    n_train=n-n_cal-n_val-n_test
    if n_train < 1:
        n_train=1
        while n_train+n_cal+n_val+n_test>n:
            if n_test>1:n_test-=1
            elif n_val>1:n_val-=1
            else:n_cal-=1
    cuts=[n_train,n_train+n_cal,n_train+n_cal+n_val]
    out={}
    for i,g in enumerate(groups):
        out[g]='train' if i<cuts[0] else 'calibration' if i<cuts[1] else 'validation' if i<cuts[2] else 'test'
    return out

def spatial_group_split(graphs, group_key='spatial_group'):
    """Deterministic approximate 70/10/10/10 split by spatial group, stratified by city."""
    city_groups=defaultdict(set)
    keyed=[]
    for g in graphs:
        group=str((g.metadata or {}).get(group_key,g.community_id)); city_groups[g.city].add(group); keyed.append((g,group))
    assignment={}
    for city,groups in city_groups.items():
        for group,split in _allocate_groups(groups).items(): assignment[(city,group)]=split
    out=defaultdict(list)
    for g,group in keyed: out[assignment[(g.city,group)]].append(g)
    return dict(out)

def leave_one_city_out(graphs, heldout_city):
    train=[g for g in graphs if g.city.lower()!=heldout_city.lower()]
    test=[g for g in graphs if g.city.lower()==heldout_city.lower()]
    return train,test
