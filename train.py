"""Train the diffusion downscaler.
    python train.py [steps1] [steps2]                                  # synthetic pairs (default)
    python train.py [steps1] [steps2] --real COARSE_GLOB FINE_GLOB VAR  # real NetCDF pairs, e.g.:
    python train.py 3000 800 --real "data/raw/pairs/era5_t2m_*.nc" "data/raw/pairs/era5land_t2m_*.nc" t2m
Swap the synthetic pairs for real 12 km / 5 km pairs via --real once NCMRWF/ERA5-Land files are fetched."""
import sys, time, numpy as np
from pipeline import downscale as D
from pipeline import pairs_data as PD
from pipeline.diffusion_jax import DiffusionDownscaler

argv = sys.argv[1:]
s1, s2 = (int(argv[0]), int(argv[1])) if len(argv) >= 2 and argv[0].isdigit() else (1200, 300)
rng = np.random.default_rng(1); oro = D.make_oro()
if "--real" in argv:
    i = argv.index("--real"); coarse, fine = PD.from_files(argv[i+1], argv[i+2], argv[i+3]); source = f"real: {argv[i+1]} , {argv[i+2]} ({argv[i+3]})"
else:
    coarse, fine = PD.synthetic(rng, 1500, oro); source = "synthetic (pipeline/downscale.py random fields)"
print(f"training pairs: {source}  n={len(fine)}  shape={fine.shape[1:]}")
t0 = time.time()
m = DiffusionDownscaler().fit(coarse, fine, oro, s1, s2, log=lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True))
m.save("models/diffusion.npz"); print("saved models/diffusion.npz  source:", source)
