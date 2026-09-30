"""Real-data training entry point (NOT RUN in this build: no network access to CDS/NCEI/NCMRWF from here).
Wires fetch/*.py + config.py into pipeline/gnn.py and pipeline/diffusion_jax.py. Expect a debugging pass on
your machine: variable names, missing timesteps and NaN edges are the most likely issues.

    python fetch/ibtracs.py                 # best-track CSV, ~15 MB
    python -m fetch.era5 tracking            # msl/u10/v10 for TRAIN_YEARS cyclone seasons (config.py)
    python -m fetch.era5 pairs               # 2 m temperature, ERA5 + ERA5-Land, for downscaling pairs
    python train_real.py gnn                 # -> models/gnn_real.npz
    python train_real.py diffusion           # -> models/diffusion_real.npz
"""
import sys, glob, numpy as np, xarray as xr
from config import *
from pipeline import efi as E
from pipeline.geo import LATS, LONS, NY, NX
from pipeline.gnn_jax import GNNTracker
from pipeline.diffusion_jax import DiffusionDownscaler
from pipeline.downscale import pool, F
from fetch import ibtracs

def gnn_cases():
    """Groups ERA5 timesteps by IBTrACS storm id into (T,NY,NX) sequences with a matched centre track,
    the shape pipeline.gnn_jax.GNNTracker.fit expects. Held-out test storms (config.TEST_EVENTS) are skipped."""
    bt = ibtracs.load(); test_sids = ibtracs.sids(bt, TEST_EVENTS)
    files = sorted(glob.glob(str(RAW/"era5_tc/*.nc")))
    if not files: raise SystemExit("no ERA5 tracking files found; run: python -m fetch.era5 tracking")
    cases, clim = [], None
    for fp in files:
        ds = xr.open_dataset(fp)
        msl = ds.msl.values/100.0; u = ds.u10.values; v = ds.v10.values
        times = ds.valid_time.values if "valid_time" in ds else ds.time.values
        wind = np.hypot(u, v)
        if clim is None: clim = wind.reshape(-1, *wind.shape[-2:])[:: max(1, wind.shape[0]//200)]
        for sid, storm in bt[~bt.sid.isin(test_sids)].groupby("sid"):
            idx = [t for t, tt in enumerate(times) if (abs(storm.time - np.datetime64(tt, "ns")) < np.timedelta64(3, "h")).any()]
            if len(idx) < 4: continue
            centre = np.array([[storm.iloc[(storm.time - np.datetime64(times[t], "ns")).abs().argmin()].lat,
                                 storm.iloc[(storm.time - np.datetime64(times[t], "ns")).abs().argmin()].lon] for t in idx])
            ef = np.stack([E.efi(wind[t][None], clim) for t in idx])
            cases.append(dict(mslp=msl[idx], wind=wind[idx], efi=ef, centre=centre))
    print(f"{len(cases)} storm sequences from {len(files)} files, {len(test_sids)} storms held out for evaluation")
    return cases

def diffusion_pairs():
    cf, ff = sorted(glob.glob(str(RAW/"pairs/era5_t2m_*.nc"))), sorted(glob.glob(str(RAW/"pairs/era5land_t2m_*.nc")))
    if not cf: raise SystemExit("no pairs found; run: python -m fetch.era5 pairs")
    fine = np.concatenate([xr.open_dataset(f).t2m.values.reshape(-1, *xr.open_dataset(f).t2m.shape[-2:]) for f in ff], 0) - 273.15
    n = min(fine.shape[1], fine.shape[2]); fine = fine[:, :n - n % F, :n - n % F]
    return fine

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "gnn"
    if what == "gnn":
        GNNTracker().fit(gnn_cases(), steps=4000).save("models/gnn_real.npz")
    else:
        fine = diffusion_pairs(); from pipeline.downscale import make_oro
        oro = make_oro()[: fine.shape[1], : fine.shape[2]] if fine.shape[1] != 64 else make_oro()
        DiffusionDownscaler().fit(pool(fine), fine, oro, steps1=4000, steps2=1000).save("models/diffusion_real.npz")
    print("done")
