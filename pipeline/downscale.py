"""Lite statistical residual downscaler (numpy): regression mean + amplitude map + spectrally matched
stochastic residual. Same role as the diffusion model in downscale_diffusion.py."""
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
F, N, DX = 2, 64, 5.0
def make_oro(seed=7):
    o = gaussian_filter(np.random.default_rng(seed).normal(size=(N, N)), 4); return o/np.abs(o).max()
def make_fine(rng, V, R, cy, cx, oro):
    n = len(V); yy, xx = np.mgrid[:N, :N]
    r = np.sqrt((yy[None]-cy[:, None, None])**2+(xx[None]-cx[:, None, None])**2)*DX/R[:, None, None]
    w = (V[:, None, None]*r*np.exp(1-r)+20)*(1+0.12*oro[None])
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))); k[0, 0] = 1
    z = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, N, N)))*k**-1.2).real; z /= z.std((1, 2), keepdims=True)
    return np.clip(w*(1+0.10*z), 0, None)
def random_fine(rng, n, oro):
    return make_fine(rng, rng.uniform(60, 220, n), rng.uniform(25, 60, n), rng.uniform(14, 50, n), rng.uniform(14, 50, n), oro)
def pool(f): return f.reshape(len(f), N//F, F, N//F, F).mean((2, 4))
def bicubic(c): return np.clip(np.array([zoom(x, F, order=3, mode="nearest") for x in c]), 0, None)
class LiteDownscaler:
    def _X(self, up, oro):
        g = np.hypot(*np.gradient(up, axis=(1, 2)))
        return np.stack([np.ones_like(up), up, g, oro[None]+0*up, up*oro[None], up**2/100], -1).reshape(-1, 6)
    def fit(self, coarse, fine, oro):
        up = bicubic(coarse); X = self._X(up, oro); r = (fine-up).ravel()
        self.b = np.linalg.lstsq(X, r, rcond=None)[0]; res = r-X@self.b
        self.a = np.linalg.lstsq(X[:, :3], np.abs(res), rcond=None)[0]
        z = (res/np.clip(X[:, :3]@self.a, .5, None)).reshape(len(up), N, N)[:100]
        self.spec = (np.abs(np.fft.fft2(z))**2).mean(0); self.oro = oro; return self
    def _pred(self, coarse):
        up = bicubic(coarse); X = self._X(up, self.oro)
        return (up.ravel()+X@self.b).reshape(up.shape), np.clip(X[:, :3]@self.a, .5, None).reshape(up.shape)
    def mean(self, coarse): return np.clip(self._pred(coarse)[0], 0, None)
    def sample(self, coarse, n, rng):
        mu, amp = self._pred(coarse)
        z = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, N, N)))*np.sqrt(self.spec/(N*N))[None]).real
        z /= z.std((1, 2), keepdims=True); return np.clip(mu+amp*z, 0, None)
