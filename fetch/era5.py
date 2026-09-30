"""ERA5 / ERA5-Land through the CDS API, cached to disk (no request if the file exists).
Needs a free CDS account, ~/.cdsapirc, and the dataset licences accepted once on the website.
    python -m fetch.era5 static | tracking | pairs | all"""
import sys, pathlib, cdsapi
from config import *
DAYS = [f"{d:02d}" for d in range(1, 32)]
def _get(dataset, req, out):
    out = pathlib.Path(out)
    if out.exists(): print("cached ", out); return
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".part")
    try: cdsapi.Client().retrieve(dataset, {**req, "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}, str(tmp)); tmp.rename(out); print("fetched", out)
    except Exception as e: print("FAILED ", out, "->", e)      # keep going; re-run to retry only the missing files
def static():
    one = dict(year=["2020"], month=["01"], day=["01"], time=["00:00"], variable=["geopotential"])
    _get("reanalysis-era5-land", one, RAW/"static/era5land_z.nc")
def tracking(years=TRAIN_YEARS):        # msl, u10, v10 at 0.25 deg for cyclone seasons: training data for the GNN tracker
    for y in years:
        for s, months in SEASONS.items():
            _get("reanalysis-era5-single-levels", dict(product_type=["reanalysis"], variable=["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"],
                 year=[str(y)], month=months, day=DAYS, time=TC_TIMES), RAW/f"era5_tc/{y}_{s}.nc")
def pairs(years=HEAT_YEARS):            # 2 m temperature: ERA5 0.25 deg (coarse) and ERA5-Land 0.1 deg (fine) = 2.5x downscaling pairs
    for y in years:
        base = dict(year=[str(y)], month=HEAT_MONTHS, day=DAYS, time=[HEAT_TIME])
        _get("reanalysis-era5-single-levels", dict(product_type=["reanalysis"], variable=["2m_temperature"], **base), RAW/f"pairs/era5_t2m_{y}.nc")
        _get("reanalysis-era5-land", dict(variable=["2m_temperature"], **base), RAW/f"pairs/era5land_t2m_{y}.nc")
if __name__ == "__main__":
    for a in (sys.argv[1:] or ["all"]):
        {"static": static, "tracking": tracking, "pairs": pairs, "all": lambda: (static(), tracking(), pairs())}[a]()
