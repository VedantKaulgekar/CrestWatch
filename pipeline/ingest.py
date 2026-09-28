"""Ingestion. `load_ensemble` reads real NEPS-G/NCUM files (GRIB2/NetCDF) onto the grid in geo.py;
`make_case` builds a SYNTHETIC ensemble with the same schema so the pipeline runs end to end
without NCMRWF access. Every output carries the provenance flag set here."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .geo import *
SCHEMA = "dims: member, step, lat, lon | vars: msl (hPa), si10 (km/h)"
def load_ensemble(path):
    import xarray as xr
    ds = xr.open_mfdataset(path, combine="by_coords", chunks="auto")
    return ds.interp(latitude=LATS, longitude=LONS)
def make_case(seed, M=23):
    rng = np.random.default_rng(seed); T = len(LEADS); t = LEADS/240
    truth = np.c_[12+10*t**0.9+rng.normal(0, .3), 88+rng.uniform(-1, 1)+2.5*np.sin(np.pi*t*0.8)]
    vt = 35+155*np.sin(np.pi*np.clip(t*1.05, 0, 1))**1.3
    bias = rng.normal(0, 1, (1, 1, 2))*0.9*t[None, :, None]   # shared bias across members
    err = np.cumsum(rng.normal(0, .35, (M, T, 2)), 1); err -= err[:, :1]
    tracks = truth[None]+bias+err
    vmax = vt[None]*(1+rng.normal(0, .08, (M, 1)))+rng.normal(0, 8, (M, T))
    return dict(truth=truth, vt=vt, tracks=tracks, vmax=vmax, rng=rng, provenance="SYNTHETIC")
def mslp_fields(case):
    rng = case["rng"]; tr, vm = case["tracks"], case["vmax"]; M, T, _ = tr.shape
    r = np.sqrt(((LATS[None, None, :, None]-tr[..., 0, None, None])*111.0)**2 +
                ((LONS[None, None, None, :]-tr[..., 1, None, None])*111.0*np.cos(np.radians(tr[..., 0, None, None])))**2)
    dp = 0.5*vm[..., None, None]+4
    noise = gaussian_filter(rng.normal(0, 1, (M, T, NY, NX)), (0, 0, 2, 2))*3
    return 1010-dp*np.exp(-(r/110)**1.3)+noise
def wind_fields(case, ti, rng):
    w = []
    for (la, lo), v in zip(case["tracks"][:, ti], case["vmax"][:, ti]):
        r = radius_km(la, lo)/60.0
        w.append(v*r*np.exp(1-r)+14*rng.weibull(2, (NY, NX)))
    return np.array(w)
