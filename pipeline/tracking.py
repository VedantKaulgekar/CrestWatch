import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from .geo import *
def track_member(fields, gate_km=400):
    out, prev = [], None
    for f in fields:
        sm = gaussian_filter(f, 1.0); ij = np.argwhere((sm == minimum_filter(sm, 9)) & (sm < 1000))
        if not len(ij): out.append((np.nan, np.nan)); continue
        pts = np.c_[LATS[ij[:, 0]], LONS[ij[:, 1]]]
        k = sm[ij[:, 0], ij[:, 1]].argmin() if prev is None else km(pts, np.array(prev)).argmin()
        if prev is not None and km(pts[k], np.array(prev)) > gate_km: out.append((np.nan, np.nan)); continue
        i, j = ij[k]; a, b = max(i-2, 0), max(j-2, 0)
        w = np.clip(1008-f[a:i+3, b:j+3], 0, None); la, lo = LATS[a:i+3], LONS[b:j+3]
        pt = ((w.sum(1)*la).sum()/w.sum(), (w.sum(0)*lo).sum()/w.sum()); prev = pt; out.append(pt)
    return np.array(out)
def track_ensemble(fields): return np.array([track_member(m) for m in fields])
def aggregate(tr):
    mean = np.nanmean(tr, 0); return mean, np.nanpercentile(km(tr, mean[None]), 90, axis=0)
def strike_prob(tr, point, R_km=150): return np.nanmean(km(tr, np.asarray(point)[None, None]) <= R_km, axis=0)
