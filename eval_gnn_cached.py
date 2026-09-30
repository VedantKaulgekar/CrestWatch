"""Evaluate models/gnn.npz against the classical tracker using the cached held-out set. python eval_gnn_cached.py"""
import json, pickle, numpy as np
from pipeline import ingest
from pipeline.gnn_jax import GNNTracker
from pipeline.tracking import track_ensemble, aggregate
from pipeline.geo import km

test = pickle.load(open("data/cache/test_cases.pkl", "rb"))
gnn = GNNTracker.load("models/gnn.npz")
gnn_err, cls_err = [], []
for i, c in enumerate(test):
    gnn_err.append(km(gnn.track(c["mslp"], c["wind"], c["efi"]), c["centre"]))
    c2 = ingest.make_case(1_000_000+i); f = ingest.mslp_fields(c2); tr = track_ensemble(f); m, _ = aggregate(tr)
    cls_err.append(km(m, c2["truth"]))
gnn_err, cls_err = np.array(gnn_err), np.array(cls_err)
result = dict(n_test=len(test), gnn_mean_km=float(np.nanmean(gnn_err)), gnn_median_km=float(np.nanmedian(gnn_err)),
              ensemble_mean_km=float(np.nanmean(cls_err)), ensemble_median_km=float(np.nanmedian(cls_err)),
              gnn_miss_rate=float(np.isnan(gnn_err).mean()))
print(result); json.dump(result, open("models/gnn_eval.json", "w"), indent=1)
