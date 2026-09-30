"""Train the spatio-temporal GNN anomaly tracker (pipeline/gnn_jax.py) on synthetic labelled cases and
evaluate against the classical minimum-pressure tracker on held-out cases.
    python train_gnn.py [n_cases] [steps] [n_test]
Swap `build_cases` for real IMDAA/ERA5 fields with IBTrACS-matched centres (see fetch/) when available --
`GNNTracker.fit` only needs a list of {mslp, wind, efi, centre} dicts, whatever their source."""
import sys, time, json, numpy as np
from pipeline import ingest, efi as E
from pipeline.gnn_jax import GNNTracker
from pipeline.tracking import track_ensemble, aggregate
from pipeline.geo import km, LEADS

def build_cases(n, seed0, n_clim=15):
    """One case per seed: control-member mslp and wind fields over all leads, EFI per lead computed
    from the full ensemble's wind, and the true track -- the same synthetic generator used everywhere
    else in the pipeline (pipeline/ingest.py), so results are directly comparable to the classical tracker."""
    cases = []
    for s in range(seed0, seed0+n):
        c = ingest.make_case(s); rng = c["rng"]; clim = E.climate(rng, n_clim)
        mslp = ingest.mslp_fields(c)[0].astype(np.float32)                 # control member, (T,NY,NX)
        wind = np.stack([ingest.wind_fields(c, ti, rng).mean(0) for ti in range(len(LEADS))]).astype(np.float32)
        efi = np.stack([E.efi(ingest.wind_fields(c, ti, rng), clim) for ti in range(len(LEADS))]).astype(np.float32)
        cases.append(dict(mslp=mslp, wind=wind, efi=efi, centre=c["truth"].astype(np.float32)))
    return cases

def main():
    n_tr, steps, n_te = (int(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (300, 3000, 60)
    t0 = time.time()
    cases = build_cases(n_tr, seed0=0)
    gnn = GNNTracker().fit(cases, steps=steps, log=lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True))
    gnn.save("models/gnn.npz"); print(f"[{time.time()-t0:5.0f}s] saved models/gnn.npz")

    test = build_cases(n_te, seed0=1_000_000)                              # disjoint seeds: held-out
    gnn_err, cls_err = [], []
    for i, c in enumerate(test):
        gnn_err.append(km(gnn.track(c["mslp"], c["wind"], c["efi"]), c["centre"]))
        # classical baseline needs the full ensemble (member spread), so rebuild the same seed's ensemble
        c2 = ingest.make_case(1_000_000+i); f = ingest.mslp_fields(c2); tr = track_ensemble(f); m, _ = aggregate(tr)
        cls_err.append(km(m, c2["truth"]))
    gnn_err, cls_err = np.array(gnn_err), np.array(cls_err)
    result = dict(n_test=n_te,
                  gnn_mean_km=float(np.nanmean(gnn_err)), gnn_median_km=float(np.nanmedian(gnn_err)),
                  ensemble_mean_km=float(np.nanmean(cls_err)), ensemble_median_km=float(np.nanmedian(cls_err)),
                  gnn_miss_rate=float(np.isnan(gnn_err).mean()), note="gnn_err/ensemble_mean_km are per-lead-averaged track errors on held-out synthetic cases")
    print(f"[{time.time()-t0:5.0f}s]", result)
    import pathlib; pathlib.Path("models/gnn_eval.json").write_text(json.dumps(result, indent=1))

if __name__ == "__main__": main()
