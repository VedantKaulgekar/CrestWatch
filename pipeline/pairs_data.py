"""Coarse/fine training pairs for the downscaler.
`from_files` builds real pairs from an ERA5 (coarse) / ERA5-Land or NCMRWF (fine) NetCDF pair fetched by
fetch/era5.py or converted by fetch/ncmrwf.py: matching timestamps, coarse regridded then upsampled onto
the fine grid, fine cropped to fixed-size tiles. `synthetic` (used by train.py while real files are absent)
keeps the exact same return shape so downstream code does not change when real data arrives."""
import numpy as np, glob

def from_files(coarse_glob, fine_glob, var="t2m", tile=64):
    import xarray as xr
    c = xr.open_mfdataset(sorted(glob.glob(coarse_glob)), combine="by_coords")[var]
    f = xr.open_mfdataset(sorted(glob.glob(fine_glob)), combine="by_coords")[var]
    common = np.intersect1d(c.time.values, f.time.values)
    if len(common) == 0: raise SystemExit(f"no overlapping timestamps between {coarse_glob} and {fine_glob}")
    c, f = c.sel(time=common), f.sel(time=common)
    ny, nx = f.latitude.size, f.longitude.size
    fine = f.values.astype(np.float32)
    if fine.shape[1] < tile or fine.shape[2] < tile: raise SystemExit(f"fine grid {fine.shape[1:]} smaller than tile {tile}; lower `tile` or widen the domain")
    y0, x0 = (fine.shape[1]-tile)//2, (fine.shape[2]-tile)//2
    fine = fine[:, y0:y0+tile, x0:x0+tile]
    c_up = c.interp(latitude=f.latitude, longitude=f.longitude).values.astype(np.float32)[:, y0:y0+tile, x0:x0+tile]
    factor = 2  # coarse-pool the upsampled coarse field back down to define the network's "coarse" input resolution
    coarse = c_up.reshape(len(c_up), tile//factor, factor, tile//factor, factor).mean((2, 4))
    if np.isnan(fine).any() or np.isnan(coarse).any(): raise SystemExit("NaNs in pairs; check domain overlap and interpolation")
    return coarse, fine

def synthetic(rng, n, oro):
    from .downscale import random_fine, pool
    fine = random_fine(rng, n, oro); return pool(fine), fine
