"""Conditional residual diffusion downscaler in JAX (CPU-friendly). Predicts (fine - bicubic(coarse)) given
[bicubic(coarse), orography]. Stage 1: tail-weighted DDPM training. Stage 2: high-noise fine-tuning for extremes."""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from .downscale import bicubic, N
T, S, CH, DIL = 50, 20.0, 20, [1, 2, 4, 2, 1]
BETA = np.linspace(1e-3, 0.2, T).astype(np.float32); AB = np.cumprod(1-BETA).astype(np.float32)
BETA_j, AB_j = jnp.array(BETA), jnp.array(AB)
def init(key):
    dims = [4]+[CH]*len(DIL); ks = jax.random.split(key, len(DIL)+1)
    p = [(jax.random.normal(k, (o, i, 3, 3))*np.sqrt(2/(i*9)), jnp.zeros(o)) for k, i, o in zip(ks, dims[:-1], dims[1:])]
    return p+[(jax.random.normal(ks[-1], (1, CH, 3, 3))*0.02, jnp.zeros(1))]
def conv(x, w, b, d):
    return lax.conv_general_dilated(x, w, (1, 1), "SAME", rhs_dilation=(d, d), dimension_numbers=("NCHW", "OIHW", "NCHW"))+b[None, :, None, None]
def net(p, x, t, cond):
    tt = jnp.broadcast_to((t/T)[:, None, None, None], (x.shape[0], 1)+x.shape[2:])
    h = jax.nn.silu(conv(jnp.concatenate([x, cond, tt], 1), *p[0], DIL[0]))
    for (w, b), d in zip(p[1:len(DIL)], DIL[1:]): h = h+jax.nn.silu(conv(h, w, b, d))
    return conv(h, *p[-1], 1)
def loss_fn(p, x0, cond, t, eps, w):
    ab = AB_j[t][:, None, None, None]; xt = jnp.sqrt(ab)*x0+jnp.sqrt(1-ab)*eps
    return jnp.mean(w*(net(p, xt, t.astype(jnp.float32), cond)-eps)**2)
@jax.jit
def step(p, m, v, i, x0, cond, t, eps, w, lr):
    l, g = jax.value_and_grad(loss_fn)(p, x0, cond, t, eps, w); tm = jax.tree_util.tree_map
    m = tm(lambda a, b: 0.9*a+0.1*b, m, g); v = tm(lambda a, b: 0.999*a+0.001*b*b, v, g)
    p = tm(lambda q, a, b: q-lr*(a/(1-0.9**i))/(jnp.sqrt(b/(1-0.999**i))+1e-8), p, m, v); return p, m, v, l
@jax.jit
def _sample(p, cond, key):
    n = cond.shape[0]; x = jax.random.normal(key, (n, 1)+cond.shape[2:]); ks = jax.random.split(jax.random.fold_in(key, 1), T)
    def body(x, tk):
        t, k = tk; eps = net(p, x, jnp.full((n,), t, jnp.float32), cond); a, ab = 1-BETA_j[t], AB_j[t]
        return (x-(1-a)/jnp.sqrt(1-ab)*eps)/jnp.sqrt(a)+jnp.sqrt(BETA_j[t])*jax.random.normal(k, x.shape)*(t > 0), None
    return lax.scan(body, x, (jnp.arange(T)[::-1], ks[::-1]))[0]
class DiffusionDownscaler:
    def __init__(s, params=None, oro=None): s.p, s.oro = params, oro
    def _cond(s, up): return np.stack([up/100.0, np.broadcast_to(s.oro, up.shape)], 1).astype(np.float32)
    def fit(s, coarse, fine, oro, steps1=1200, steps2=300, B=8, crop=32, lr=2e-3, log=print, seed=0):
        s.oro = oro; up = bicubic(coarse); res = ((fine-up)/S).astype(np.float32); cond = s._cond(up); rng = np.random.default_rng(seed)
        thr = np.quantile(np.abs(res), 0.99); p = init(jax.random.PRNGKey(seed)); m = jax.tree_util.tree_map(jnp.zeros_like, p); v = m
        for i in range(1, steps1+steps2+1):
            hi = i > steps1; idx = rng.integers(0, len(res), B); y, x = rng.integers(0, N-crop+1, 2)
            x0 = res[idx, None, y:y+crop, x:x+crop]; c = cond[idx][:, :, y:y+crop, x:x+crop]
            t = rng.integers(T//2 if hi else 0, T, B); w = 1+4*(np.abs(x0) > thr)
            p, m, v, l = step(p, m, v, float(i), x0, c, t, rng.normal(size=x0.shape).astype(np.float32), w.astype(np.float32), lr*(0.3 if hi else 1)*(1-0.8*i/(steps1+steps2)))
            if i % 100 == 0: log(f"step {i} loss {float(l):.4f} {'stage2-highnoise' if hi else 'stage1'}")
        s.p = p; return s
    def sample(s, coarse, n, rng):
        up = bicubic(coarse); cond = np.repeat(s._cond(up), n, 0)
        x = np.array(_sample(s.p, cond, jax.random.PRNGKey(int(rng.integers(1 << 30)))))[:, 0]
        return np.clip(up+S*x, 0, None)
    def save(s, path): np.savez(path, oro=s.oro, **{f"{k}{i}": np.array(a) for i, (w, b) in enumerate(s.p) for k, a in (("w", w), ("b", b))})
    @classmethod
    def load(cls, path):
        z = np.load(path); n = len([k for k in z.files if k.startswith("w")]); return cls([(jnp.array(z[f"w{i}"]), jnp.array(z[f"b{i}"])) for i in range(n)], z["oro"])
