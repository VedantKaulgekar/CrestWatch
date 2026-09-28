"""Train the diffusion downscaler:  python train.py [steps1] [steps2].  Swap the synthetic pairs for real 12 km / 5 km pairs here."""
import sys, time, numpy as np
from pipeline import downscale as D
from pipeline.diffusion_jax import DiffusionDownscaler
s1, s2 = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1200, 300)
rng = np.random.default_rng(1); oro = D.make_oro(); fine = D.random_fine(rng, 1500, oro); t0 = time.time()
m = DiffusionDownscaler().fit(D.pool(fine), fine, oro, s1, s2, log=lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True))
m.save("models/diffusion.npz"); print("saved models/diffusion.npz")
