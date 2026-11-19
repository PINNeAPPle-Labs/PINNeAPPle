"""Deterministic parameter sampling and split assignment."""
import hashlib
import json
import numpy as np

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def designs(c):
    bounds={**c.geometry,**c.parameters}
    if len(bounds)!=len(c.geometry)+len(c.parameters):
        raise ValueError("Geometry and physics parameter keys overlap")
    rng=np.random.default_rng(c.seed)
    cols={}
    for name in sorted(bounds):
        lo,hi=map(float,bounds[name])
        vals=(np.arange(c.samples)+rng.random(c.samples))/c.samples
        rng.shuffle(vals)
        cols[name]=lo+(hi-lo)*vals
    return [{k:float(v[i]) for k,v in cols.items()} for i in range(c.samples)]

def assign_splits(n,fractions,seed):
    rng=np.random.default_rng(seed+111)
    ids=rng.permutation(n)
    a=max(1,int(round(n*fractions[0])))
    b=max(1,int(round(n*fractions[1])))
    if a+b>=n: a=n-2;b=1
    out=[""]*n
    for i in ids[:a]:out[i]="train"
    for i in ids[a:a+b]:out[i]="val"
    for i in ids[a+b:]:out[i]="test"
    return out
