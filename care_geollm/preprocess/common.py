import json
from pathlib import Path
import numpy as np

CANONICAL_FEATURES = [
    'footprint_area','perimeter','compactness','height','floors','construction_age_proxy',
    'landuse_residential','envelope_exposure','sky_view_proxy','vegetation_fraction',
    'energy_context','service_distance','edge_length_proxy','slope_proxy','stair_proxy',
    'shade_proxy','vertical_access','service_capacity','population_total','population_65plus',
    'wsf3d_fraction','wsf3d_height','wsf3d_volume','support_score'
]

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def robust_standardize(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            out[:, j] = 0.0; continue
        lo, hi = np.nanpercentile(col[finite], [1, 99])
        clipped = np.clip(col, lo, hi)
        mu = np.nanmean(clipped[finite]); sd = np.nanstd(clipped[finite])
        clipped[~finite] = mu
        out[:, j] = (clipped - mu) / max(float(sd), eps)
    return out.astype(np.float32)

def normalize_relative_xy(xy):
    xy = np.asarray(xy, dtype=np.float32)
    center = np.nanmean(xy, axis=0, keepdims=True)
    rel = xy - center
    r = np.linalg.norm(rel, axis=1)
    scale = np.nanpercentile(r, 95) if len(r) else 1.0
    return (rel / max(float(scale), 1e-6)).astype(np.float32)

def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
