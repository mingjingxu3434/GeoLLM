import numpy as np

def sample_raster(path, xy, src_crs=None, band=1):
    """Sample a raster at x/y coordinates. Requires rasterio.

    xy is assumed to be in src_crs. If src_crs differs from the raster CRS it is transformed.
    """
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(path) as ds:
        xs, ys = np.asarray(xy)[:,0].tolist(), np.asarray(xy)[:,1].tolist()
        if src_crs and str(src_crs) != str(ds.crs):
            xs, ys = transform(src_crs, ds.crs, xs, ys)
        vals = np.array([v[0] for v in ds.sample(zip(xs, ys), indexes=band)], dtype=np.float32)
        nodata = ds.nodata
        if nodata is not None: vals[vals == nodata] = np.nan
        return vals

def sample_multiple(paths, xy, src_crs=None):
    return np.column_stack([sample_raster(p, xy, src_crs) for p in paths]).astype(np.float32)
