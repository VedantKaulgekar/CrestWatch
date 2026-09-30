"""IBTrACS North Indian Ocean best-track (NOAA NCEI). python -m fetch.ibtracs
For the NI basin the WMO fields are the IMD (New Delhi RSMC) record; check the WMO_AGENCY column."""
import urllib.request, pathlib
from config import *
URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
def download(path=RAW/"ibtracs.NI.csv"):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(URL, path); print("fetched", path)
    return path
def load(path=None):
    import pandas as pd
    df = pd.read_csv(path or RAW/"ibtracs.NI.csv", skiprows=[1], low_memory=False)   # row 1 holds units
    df["time"] = pd.to_datetime(df["ISO_TIME"])
    for c in ("LAT", "LON", "WMO_WIND", "USA_WIND", "WMO_PRES"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wind"] = df["WMO_WIND"].fillna(df["USA_WIND"])
    df = df[(df.time.dt.hour % 6 == 0) & (df.time.dt.minute == 0)].dropna(subset=["LAT", "LON"])
    return df.rename(columns=dict(SID="sid", NAME="name", SEASON="season", LAT="lat", LON="lon", WMO_PRES="pres"))[["sid", "name", "season", "time", "lat", "lon", "wind", "pres"]]
def sids(df, events):
    return {s for n, y in events for s in df[(df.name.str.upper() == n) & (df.season == y)].sid.unique()}
if __name__ == "__main__": download()
