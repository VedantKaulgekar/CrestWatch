# CrestWatch (SIH 2026, PS 26078)

Ensemble anomaly tracking, probabilistic 5 km downscaling and graded alerts for medium-range extreme weather.

## Run
```
pip install numpy scipy jax jaxlib   # jax needed for the trained models; without it run.py falls back to classical/lite methods
python train.py 3000 800             # optional: retrain the downscaler (about 1.5 min on 1 CPU core); weights ship in models/
python cache_cases.py 30 15          # optional: rebuild the GNN's synthetic training/test cases (about 3 min)
python train_gnn_resume.py 700       # optional: train (or continue training) the GNN tracker in short, resumable bursts
python eval_gnn_cached.py            # optional: re-check the GNN against the classical tracker on held-out cases
python run.py                        # ingest -> EFI -> tracking -> downscaling -> alerts -> evaluation (about 3.5 min on 1 CPU core)
open web/index.html                  # landing page; console at web/console.html (works from disk)
pip install fastapi uvicorn && uvicorn api.main:app   # API at :8000, site served at /
```
`run.py` writes `data/results.json` (API) and `web/data/results.js` (site). Every number on the site and API comes from this one file.
The GNN trains in short bursts (`train_gnn_resume.py`) rather than one long run, because a single call in this build's sandbox is time- and memory-limited; run it as many times as you like, it resumes from `models/gnn.npz` each time.

## Layout
| Path | Role |
|---|---|
| `pipeline/ingest.py` | `load_ensemble()` for real NEPS-G/NCUM files (xarray); `make_case()` synthetic ensemble with the same schema |
| `pipeline/efi.py` | Extreme Forecast Index against a climate sample |
| `pipeline/tracking.py` | Per-member detection, gated association, ensemble aggregation, strike probability |
| `pipeline/downscale.py` | Lite statistical residual downscaler (runs on numpy) |
| `pipeline/diffusion_jax.py` | Residual diffusion downscaler (JAX): tail-weighted loss, high-noise fine-tuning stage; `train.py` trains it |
| `pipeline/gnn_jax.py` | Message-passing GNN anomaly tracker (JAX) on a subsampled pixel graph; `train_gnn.py` / `train_gnn_resume.py` train it |
| `pipeline/mesh.py` | Icosahedral and regular mesh builders for a future global-mesh tracker (not yet wired into `gnn_jax.py`) |
| `fetch/era5.py`, `fetch/ibtracs.py`, `fetch/ncmrwf.py` | Real-data fetchers: ERA5/ERA5-Land via the CDS API (cached to disk), IBTrACS best-track, NCMRWF file conversion |
| `config.py` | Study domain (India, Arabian Sea, Bay of Bengal), training years, test events -- edit here, everything else follows |
| `pipeline/downscale_diffusion.py` | Earlier PyTorch draft of the same model, never run; superseded by the JAX version |
| `pipeline/alerts.py` | Exceedance probabilities, Low/Moderate/Severe zones, core coordinate, CAP 1.2, GeoJSON |
| `pipeline/evaluate.py` | PSD, CRPS, peak and tail scores, reliability |
| `api/main.py` | REST endpoints: /tracks /hazards /downscale /alert /alert/cap /alert/geojson /evaluation |
| `web/` | Landing page and forecaster console (static, no build step) |

## What is real and what is not
- **Runs and tested:** ingestion schema, EFI, tracking, lite downscaler, alerts, evaluation, API code, both web pages (loaded in a DOM emulator; not viewed in a real browser).
- **Synthetic:** all data. No NCMRWF data was available, so results verify the software only.
- **Trained diffusion model:** the JAX downscaler, trained on synthetic pairs only (3,800 steps: 3,000 base + 800 high-noise fine-tuning for extremes). On the benchmark in `run.py` (6 held-out synthetic fields, 8 samples each): peak wind ratio 1.00 (bicubic: 0.96), CRPS 1.38 vs 1.77 for bicubic. RMSE is higher by design (a single sample is one realisation, not a mean).
- **Trained GNN tracker:** `pipeline/gnn_jax.py`, trained 2,100 steps on 30 synthetic cases (`cache_cases.py` + `train_gnn_resume.py`). On 15 held-out cases (`eval_gnn_cached.py`): 67.6 km mean track error (54.7 km median) against 65.1 km (55.6 km median) for the classical ensemble tracker. **It does not yet clearly beat the classical baseline** -- report this honestly; more training cases are the likely next lever, not more steps (loss plateaued around 2,100 steps on this case count).
- **Real-data fetchers exist but are untested against real files:** `fetch/era5.py` (CDS API, cached to disk), `fetch/ibtracs.py` (NOAA best-track), `fetch/ncmrwf.py` (schema converter for whatever NCMRWF sends you). None have been run against real data in this build.
- **Toy scale:** the downscaling task is 2x (10 km to 5 km), not 12 km to 5 km. Alert thresholds are illustrative.
- **Domain:** the tracking/EFI grid covers India, the Arabian Sea and the Bay of Bengal (`config.py`), at 0.25 degrees. The offline console map is Natural Earth country borders only, clipped to this box (`tools/build_land.js`).

## Next steps
0. Real data is not reachable from the build sandbox, but the fetchers are written and ready: `fetch/era5.py`, `fetch/ibtracs.py`, `fetch/ncmrwf.py` (see step 1 above). Run them where you have network access to Copernicus CDS and NOAA.
1. Fetch real data: `python -m fetch.era5 all` (needs a free CDS account and `~/.cdsapirc`), `python -m fetch.ibtracs`, and NCMRWF files placed under `data/raw/ncmrwf/` then converted with `python -m fetch.ncmrwf convert ...`.
2. Retrain the GNN on real IMDAA/ERA5 fields with IBTrACS-matched centres (swap `build_cases` in `train_gnn.py`), and the downscaler on real 12 km / 5 km pairs (`python train.py 3000 800 --real ...`); compare both against their current baselines with the same `pipeline/evaluate.py` metrics.
3. Evaluate leave-one-event-out on Amphan, Fani, Yaas and a heatwave using `fetch/ibtracs.py`'s event matching.
4. Replace thresholds with IMD categories; add a basemap layer to the console.
