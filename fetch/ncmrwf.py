"""NCMRWF data. There is NO public API: IMDAA (12 km, hourly, 1979-2018) and NGFS reanalysis are served by the NCMRWF Reanalysis Data
Service (https://rds.ncmrwf.gov.in, registration required). NCMRWF also produces 'IMDAA-like' products from the 12 km NCUM analysis from
January 2021. NEPS-G / NCUM forecast archives must be requested from NCMRWF or the SIH organisers.
Put the files you receive in data/raw/ncmrwf/ then:
    python -m fetch.ncmrwf inspect data/raw/ncmrwf/file.nc
    python -m fetch.ncmrwf convert data/raw/ncmrwf/file.nc data/raw/era5_tc/imdaa_YYYY.nc   # same schema as the ERA5 files"""
import sys, numpy as np, xarray as xr
from config import *
VAR_MAP = {"msl": ["msl", "prmsl", "air_pressure_at_sea_level", "mslp"], "u10": ["u10", "10u", "uas", "eastward_wind"], "v10": ["v10", "10v", "vas", "northward_wind"]}  # EDIT to match the files
def convert(src, dst):
    ds = xr.open_mfdataset(src) if "*" in src else xr.open_dataset(src); out = {}
    for k, names in VAR_MAP.items():
        n = next((x for x in names if x in ds), None)
        if n is None: raise SystemExit(f"variable for {k} not found; edit VAR_MAP. Available: {list(ds.data_vars)}")
        out[k] = ds[n]
    o = xr.Dataset(out).rename({c: {"lat": "latitude", "lon": "longitude"}.get(c, c) for c in ds.dims}).sortby("latitude")
    if float(o.msl.mean()) < 2000: o["msl"] = o.msl*100        # ERA5 files hold Pa; convert hPa -> Pa if needed
    o = o.interp(latitude=np.arange(DOMAIN["south"], DOMAIN["north"]+RES/2, RES), longitude=np.arange(DOMAIN["west"], DOMAIN["east"]+RES/2, RES))
    o.to_netcdf(dst); print("wrote", dst)
if __name__ == "__main__":
    if sys.argv[1] == "inspect": print(xr.open_dataset(sys.argv[2]))
    else: convert(sys.argv[2], sys.argv[3])
