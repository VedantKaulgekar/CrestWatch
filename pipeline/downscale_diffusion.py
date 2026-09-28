"""Residual diffusion downscaler (PyTorch). NOT EXECUTED in the build sandbox (no disk for torch):
a reviewed reference implementation to run where torch is installed. Same interface role as LiteDownscaler."""
import torch, torch.nn as nn, torch.nn.functional as Fn
T_STEPS = 100
class Eps(nn.Module):
    def __init__(s, c=48):
        super().__init__(); s.i = nn.Conv2d(4, c, 3, padding=1)
        s.b = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for _ in range(4)]); s.o = nn.Conv2d(c, 1, 3, padding=1)
    def forward(s, x, t, cond):
        tt = (t.float()/T_STEPS).view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        h = Fn.silu(s.i(torch.cat([x, cond, tt], 1)))
        for b in s.b: h = h+Fn.silu(b(h))
        return s.o(h)
class ResidualDiffusion:
    def __init__(s):
        s.net = Eps(); s.beta = torch.linspace(1e-4, 0.05, T_STEPS); s.ab = torch.cumprod(1-s.beta, 0)
    def train(s, res, cond, steps=3000, hi_from=0.0, tail_q=0.99, lr=2e-4):
        """res (n,1,H,W) normalised residual; cond (n,2,H,W) = [upsampled coarse, orography].
        hi_from>0 => stage-2 fine-tune on high-noise steps only (t >= hi_from*T) to recover tails."""
        opt = torch.optim.Adam(s.net.parameters(), lr); thr = res.abs().flatten().quantile(tail_q)
        for _ in range(steps):
            i = torch.randint(0, len(res), (32,)); x0, c = res[i], cond[i]
            t = torch.randint(int(hi_from*T_STEPS), T_STEPS, (32,)); e = torch.randn_like(x0); ab = s.ab[t].view(-1, 1, 1, 1)
            w = 1+4*(x0.abs() > thr).float()
            loss = (w*(s.net(ab.sqrt()*x0+(1-ab).sqrt()*e, t, c)-e)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def sample(s, cond, n):
        c = cond.expand(n, -1, -1, -1); x = torch.randn(n, 1, *cond.shape[2:])
        for t in reversed(range(T_STEPS)):
            eps = s.net(x, torch.full((n,), t), c); a, ab = 1-s.beta[t], s.ab[t]
            x = (x-(1-a)/(1-ab).sqrt()*eps)/a.sqrt()+(s.beta[t].sqrt()*torch.randn_like(x) if t else 0)
        return x
