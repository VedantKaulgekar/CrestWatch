"""Continue training the GNN tracker for more steps, reusing cached cases and warm-starting from
models/gnn.npz if it exists (fresh Adam moments each call -- a practical compromise for training
across several short, separately-invoked processes instead of one long-running one).
    python cache_cases.py 30 15      # once
    python train_gnn_resume.py 500   # repeat, accumulating steps, until satisfied
    python eval_gnn_cached.py        # check held-out error whenever you like"""
import sys, time, pickle, pathlib
import jax, jax.numpy as jnp
from pipeline.gnn_jax import GNNTracker, init as init_params

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cases = pickle.load(open("data/cache/train_cases.pkl", "rb"))
gnn = GNNTracker()
ckpt = pathlib.Path("models/gnn.npz")
if ckpt.exists():
    p = GNNTracker.load(ckpt).p; print(f"resuming from {ckpt}")
else:
    p = init_params(jax.random.PRNGKey(0)); print("starting from scratch")
opt = (jax.tree_util.tree_map(jnp.zeros_like, p), jax.tree_util.tree_map(jnp.zeros_like, p))
t0 = time.time()
gnn.fit(cases, steps=steps, resume=(p, opt), log=lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True))
gnn.save(ckpt); print(f"[{time.time()-t0:5.0f}s] saved {ckpt}")
