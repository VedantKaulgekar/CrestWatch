"""Evaluate the already-trained models/gnn.npz against the classical tracker on held-out cases,
without retraining.  python eval_gnn.py [n_test]"""
import sys, json, numpy as np
from pipeline import ingest, efi as E
from pipeline.gnn_jax import GNNTracker
from pipeline.tracking import track_ensemble, aggregate
from pipeline.geo import km, LEADS
from train_gnn import build_cases

n_te = int(sys.argv[1]) if len(sys.argv) > 1 else 15
gnn = GNNTracker.load("models/gnn.npz")
test = build_cases(n_te, seed0=1_000_000)
gnn_err, cls_err = [], []
for i, c in enumerate(test):
    gnn_err.append(km(gnn.track(c["mslp"], c["wind"], c["efi"]), c["centre"]))
    c2 = ingest.make_case(1_000_000+i); f = ingest.mslp_fields(c2); tr = track_ensemble(f); m, _ = aggregate(tr)
    cls_err.append(km(m, c2["truth"]))
gnn_err, cls_err = np.array(gnn_err), np.array(cls_err)
result = dict(n_test=n_te, gnn_mean_km=float(np.nanmean(gnn_err)), gnn_median_km=float(np.nanmedian(gnn_err)),
              ensemble_mean_km=float(np.nanmean(cls_err)), ensemble_median_km=float(np.nanmedian(cls_err)),
              gnn_miss_rate=float(np.isnan(gnn_err).mean()))
print(result)
json.dump(result, open("models/gnn_eval.json", "w"), indent=1)
