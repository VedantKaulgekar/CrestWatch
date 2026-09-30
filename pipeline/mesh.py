"""Mesh for the GNN tracker.
`domain_mesh` (used by the trainer below) is a regular mesh over our lat/lon box with 8-neighbour edges -
enough to test the architecture without global tiling. `icosphere` builds a true recursively-subdivided
global icosahedral mesh (as in GraphCast/AIFS); it is provided for the real deployment, where the mesh
must tile the whole sphere rather than one box, and is checked separately in tests/test_mesh.py."""
import numpy as np
from .geo import LATS, LONS

def icosphere(refine=2):
    p = (1+5**0.5)/2
    v = [np.array(x)/np.linalg.norm(x) for x in
         [[-1,p,0],[1,p,0],[-1,-p,0],[1,-p,0],[0,-1,p],[0,1,p],[0,-1,-p],[0,1,-p],[p,0,-1],[p,0,1],[-p,0,-1],[-p,0,1]]]
    f = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
         [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    for _ in range(refine):
        mid, nf = {}, []
        def m(a, b):
            k = tuple(sorted((a, b)))
            if k not in mid: mid[k] = len(v); v.append((v[a]+v[b])/(np.linalg.norm(v[a]+v[b])))
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    edges = {tuple(sorted((int(i), int(j)))) for a, b, c in f for i, j in ((a, b), (b, c), (c, a))}
    ei = np.array(sorted(edges)).T; ei = np.concatenate([ei, ei[::-1]], 1)
    return np.array(v), ei

def domain_mesh(target_nodes=400):
    """A regular mesh over LATS/LONS sized to ~target_nodes, with 8-neighbour edges."""
    ny, nx = LATS.size, LONS.size
    step = max(1, int(round((ny*nx/target_nodes)**0.5)))
    iy, ix = np.arange(0, ny, step), np.arange(0, nx, step)
    la, lo = np.meshgrid(LATS[iy], LONS[ix], indexing="ij")
    nodes = np.c_[la.ravel(), lo.ravel()]; H, W = la.shape
    idx = np.arange(H*W).reshape(H, W); e = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ys, xs = np.clip(np.arange(H)+dy, 0, H-1), np.clip(np.arange(W)+dx, 0, W-1)
            e.append(np.c_[idx.ravel(), idx[ys][:, xs].ravel()])
    ei = np.unique(np.concatenate(e), axis=0).T
    return nodes, ei, (iy, ix)

def grid_to_mesh(field, iy, ix): return field[..., iy[:, None], ix[None, :]].reshape(*field.shape[:-2], -1)
