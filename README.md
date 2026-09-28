# CrestWatch (SIH 2026, PS 26078)

Ensemble anomaly tracking, probabilistic 5 km downscaling and graded alerts for medium-range extreme weather.

## Run
```
pip install numpy scipy jax jaxlib   # jax needed for the diffusion model; without it run.py falls back to the lite downscaler
python train.py 3000 800           # optional: retrain (about 2 min on 1 CPU core); trained weights ship in models/
python run.py                      # ingest -> EFI -> tracking -> downscaling -> alerts -> evaluation (about 18 min on 1 CPU core, sampling dominates)
open web/index.html                # landing page; console at web/console.html (works from disk)
pip install fastapi uvicorn && uvicorn api.main:app   # API at :8000, site served at /
```
`run.py` writes `data/results.json` (API) and `web/data/results.js` (site). Every number on the site and API comes from this one file.

## Layout
| Path | Role |
|---|---|
| `pipeline/ingest.py` | `load_ensemble()` for real NEPS-G/NCUM files (xarray); `make_case()` synthetic ensemble with the same schema |
| `pipeline/efi.py` | Extreme Forecast Index against a climate sample |
| `pipeline/tracking.py` | Per-member detection, gated association, ensemble aggregation, strike probability |
| `pipeline/downscale.py` | Lite statistical residual downscaler (runs on numpy) |
| `pipeline/diffusion_jax.py` | Residual diffusion downscaler (JAX): tail-weighted loss, high-noise fine-tuning stage; `train.py` trains it |
| `pipeline/downscale_diffusion.py` | Earlier PyTorch draft of the same model, never run; superseded by the JAX version |
| `pipeline/alerts.py` | Exceedance probabilities, Low/Moderate/Severe zones, core coordinate, CAP 1.2, GeoJSON |
| `pipeline/evaluate.py` | PSD, CRPS, peak and tail scores, reliability |
| `api/main.py` | REST endpoints: /tracks /hazards /downscale /alert /alert/cap /alert/geojson /evaluation |
| `web/` | Landing page and forecaster console (static, no build step) |

## What is real and what is not
- **Runs and tested:** ingestion schema, EFI, tracking, lite downscaler, alerts, evaluation, API code, both web pages (loaded in a DOM emulator; not viewed in a real browser).
- **Synthetic:** all data. No NCMRWF data was available, so results verify the software only.
- **Trained model:** the JAX diffusion downscaler, trained on synthetic pairs only. On 20 held-out synthetic fields: CRPS 1.86 vs 2.55 for bicubic; peak ratio 1.02 vs 0.97; single-sample RMSE is higher by design. Sampling 32 crops takes about 95 s on one CPU core.
- **Not built:** the GNN tracker; the classical tracker is the baseline it must beat.
- **Toy scale:** the downscaling task is 2x (10 km to 5 km), not 12 km to 5 km. Alert thresholds are illustrative.

## Next steps
0. Real data is not reachable from the build sandbox. Get ERA5 (free Copernicus CDS account) and IBTrACS best-track, or NCMRWF hindcasts.
1. Fill `load_ensemble()` with NEPS-G/NCUM hindcasts and reforecast climatology; set the training target (see the solution design).
2. Retrain with `train.py` on real 12 km / 5 km pairs and compare against bicubic and the lite model with the same `evaluate.py` metrics.
3. Add the mesh GNN and evaluate leave-one-event-out on Amphan, Fani, Yaas and a heatwave.
4. Replace thresholds with IMD categories; add a basemap layer to the console.
