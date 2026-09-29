import numpy as np
import pandas as pd
from .common import CANONICAL_FEATURES

def _first(row, names, default=np.nan):
    for n in names:
        if n in row and pd.notna(row[n]): return row[n]
    return default

def load_eubucco(path, bbox=None, target_crs='EPSG:3035'):
    """Load an EUBUCCO-compatible vector file with geopandas.

    Column names differ by national source. This adapter intentionally uses fallbacks and keeps
    missingness explicit. For strict benchmark reproduction, freeze the exact source version and
    field mapping in a city config file.
    """
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if bbox is not None:
        minx,miny,maxx,maxy = bbox
        gdf = gdf.cx[minx:maxx, miny:maxy]
    if gdf.crs is None: raise ValueError('Building file must have a CRS')
    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    geom = gdf.geometry
    centroid = geom.centroid
    area = geom.area.to_numpy(np.float32)
    perim = geom.length.to_numpy(np.float32)
    compact = (4*np.pi*area / np.maximum(perim**2, 1e-6)).astype(np.float32)
    rows=[]
    for i, (_, r) in enumerate(gdf.iterrows()):
        h=float(_first(r,['height','height_m','building_height'],np.nan))
        floors=float(_first(r,['floors','n_floors','levels'],np.nan))
        age=float(_first(r,['age','construction_year','year'],np.nan))
        if np.isfinite(age) and age > 1800: age=max(0.0, 2026.0-age)
        land=str(_first(r,['type','building','landuse','use'],'')).lower()
        residential=float(any(k in land for k in ['res','apart','house','dwelling']))
        feat=np.zeros(len(CANONICAL_FEATURES),dtype=np.float32)
        feat[0]=area[i]; feat[1]=perim[i]; feat[2]=compact[i]; feat[3]=h; feat[4]=floors
        feat[5]=age; feat[6]=residential
        rows.append(feat)
    features=np.vstack(rows) if rows else np.zeros((0,len(CANONICAL_FEATURES)),np.float32)
    xy=np.column_stack([centroid.x.to_numpy(),centroid.y.to_numpy()]).astype(np.float32)
    ids=np.asarray(gdf.index.astype(str))
    return {'features':features,'xy':xy,'ids':ids,'crs':str(gdf.crs),'gdf':gdf}
