"""Spatio-temporal GNN anomaly tracker (JAX). Mesh = the regular lat/lon grid in geo.py treated as a graph with
4-neighbour edges (a stand-in for an icosahedral mesh: same message-passing idea, simpler geometry so it trains
fast on CPU). Node features per timestep: mslp anomaly, wind speed, EFI. Two outputs per node: probability the
node is inside the tracked anomaly, and a (dlat, dlon) vote for the anomaly centre -- centroid of the votes,
weighted by probability, is the GNN's track point. This is what Stage 1 of the solution design calls the
'mesh GNN tracker'; pipeline/tracking.py's minimum-pressure detector is the classical baseline it must beat."""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from .geo import LATS, LONS, NY, NX, km

STRIDE = max(1, round(((NY*NX)/4500)**0.5))   # subsample the pixel grid to ~4500 graph nodes regardless of domain size
LATS_S, LONS_S = LATS[::STRIDE], LONS[::STRIDE]
SNY, SNX = LATS_S.size, LONS_S.size

def _edges():
    idx = np.arange(SNY*SNX).reshape(SNY, SNX); e = []
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        s = idx[max(0, -di):SNY-max(0, di), max(0, -dj):SNX-max(0, dj)]
        d = idx[max(0, di):SNY-max(0, -di), max(0, dj):SNX-max(0, -dj)]
        e.append(np.stack([s.ravel(), d.ravel()], 1))
    return np.concatenate(e, 0)
EDGES = _edges(); SRC, DST = jnp.array(EDGES[:, 0]), jnp.array(EDGES[:, 1])
DEG = jnp.array(np.bincount(EDGES[:, 1], minlength=SNY*SNX)).astype(jnp.float32).clip(1)[:, None]

def init(key, cin=3, hid=24, layers=3):
    ks = jax.random.split(key, layers+2)
    enc = (jax.random.normal(ks[0], (cin, hid))*np.sqrt(2/cin), jnp.zeros(hid))
    mp = [(jax.random.normal(k, (2*hid, hid))*np.sqrt(2/(2*hid)), jnp.zeros(hid)) for k in ks[1:1+layers]]
    head = (jax.random.normal(ks[-1], (hid, 3))*0.02, jnp.zeros(3))   # [logit_in_anomaly, dlat_vote, dlon_vote]
    return dict(enc=enc, mp=mp, head=head)

def _mp_layer(h, w, b):
    msg = h[SRC]
    agg = jnp.zeros_like(h).at[DST].add(msg) / DEG
    return jax.nn.relu(jnp.concatenate([h, agg], 1) @ w + b)

def forward(p, x):  # x: (nodes, cin) -> (nodes, 3)
    w, b = p["enc"]; h = jax.nn.relu(x @ w + b)
    for w, b in p["mp"]: h = h + _mp_layer(h, w, b)
    w, b = p["head"]; return h @ w + b

def features(mslp, wind, efi):
    mslp, wind, efi = mslp[::STRIDE, ::STRIDE], wind[::STRIDE, ::STRIDE], jnp.asarray(efi)[::STRIDE, ::STRIDE]
    z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    return jnp.stack([z(mslp).ravel(), z(wind).ravel(), efi.ravel()], 1).astype(jnp.float32)

def centre_from_output(out, thresh=0.0):
    prob = jax.nn.sigmoid(out[:, 0])
    la = jnp.repeat(jnp.array(LATS_S), SNX); lo = jnp.tile(jnp.array(LONS_S), SNY)
    w = jnp.clip(prob - thresh, 0, None) + 1e-6
    return jnp.array([jnp.sum(w * la) / jnp.sum(w), jnp.sum(w * lo) / jnp.sum(w)])

def loss_fn(p, X, mask, centre_lat, centre_lon):
    out = jax.vmap(forward, (None, 0))(p, X)                       # (T, nodes, 3)
    bce = jnp.mean(-mask*jax.nn.log_sigmoid(out[..., 0]) - (1-mask)*jax.nn.log_sigmoid(-out[..., 0]))
    reg = jnp.mean(mask*((out[..., 1]-centre_lat[:, None])**2 + (out[..., 2]-centre_lon[:, None])**2))
    return bce + 0.05*reg

@jax.jit
def train_step(p, opt, X, mask, cla, clo, lr):
    l, g = jax.value_and_grad(loss_fn)(p, X, mask, cla, clo)
    m, v = opt
    m = jax.tree_util.tree_map(lambda a, b: 0.9*a+0.1*b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999*a+0.001*b*b, v, g)
    p = jax.tree_util.tree_map(lambda q, a, b: q-lr*a/(jnp.sqrt(b)+1e-8), p, m, v)
    return p, (m, v), l

class GNNTracker:
    def __init__(s, params=None): s.p = params
    def fit(s, cases, steps=800, lr=3e-3, seed=0, log=print, resume=None):
        """cases: list of dicts with mslp,wind,efi fields (NY,NX) per timestep and true centre track, from
        pipeline.ingest.make_case-style synthetic generation (see train_gnn.py). Swap in real IMDAA/ERA5-labelled
        anomalies (e.g. IBTrACS-matched pressure minima) to train on real cyclones.
        resume: optional (params, opt_state) tuple from a previous run (e.g. GNNTracker.load(...).state()),
        to continue training across several short calls when one call's time budget is too short to converge."""
        rng = np.random.default_rng(seed)
        if resume is not None:
            p, opt = resume
        else:
            p = init(jax.random.PRNGKey(seed))
            opt = (jax.tree_util.tree_map(jnp.zeros_like, p), jax.tree_util.tree_map(jnp.zeros_like, p))
        for i in range(1, steps+1):
            c = cases[rng.integers(len(cases))]
            X = jax.vmap(features)(c["mslp"], c["wind"], c["efi"])
            r = km(np.stack(np.meshgrid(LATS_S, LONS_S, indexing="ij"), -1).reshape(-1, 2)[None], c["centre"][:, None])
            mask = (r < 250).astype(np.float32)
            p, opt, l = train_step(p, opt, X, mask, jnp.array(c["centre"][:, 0]), jnp.array(c["centre"][:, 1]), lr)
            if i % 100 == 0: log(f"gnn step {i} loss {float(l):.4f}")
        s.p, s.opt = p, opt; return s
    def state(s): return (s.p, s.opt)
    def track(s, mslp_t, wind_t, efi_t):
        """One member's field sequence (T,NY,NX) each -> (T,2) GNN track."""
        out = jax.vmap(forward, (None, 0))(s.p, jax.vmap(features)(mslp_t, wind_t, efi_t))
        return np.array(jax.vmap(centre_from_output)(out))
    def save(s, path): np.savez(path, **{f"enc{i}": a for i, a in enumerate(s.p["enc"])},
        **{f"mp{i}_{j}": a for i, l in enumerate(s.p["mp"]) for j, a in enumerate(l)}, **{f"head{i}": a for i, a in enumerate(s.p["head"])})
    @classmethod
    def load(cls, path):
        z = np.load(path); n = len([k for k in z if k.startswith("mp")])//2
        return cls(dict(enc=(jnp.array(z["enc0"]), jnp.array(z["enc1"])), head=(jnp.array(z["head0"]), jnp.array(z["head1"])),
                         mp=[(jnp.array(z[f"mp{i}_0"]), jnp.array(z[f"mp{i}_1"])) for i in range(n)]))
