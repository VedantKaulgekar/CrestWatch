import numpy as np
def psd(x):
    n = x.shape[-1]; P = (np.abs(np.fft.fftshift(np.fft.fft2(x-x.mean((1, 2), keepdims=True)), axes=(1, 2)))**2).mean(0)
    yy, xx = np.indices(P.shape); r = np.hypot(yy-n//2, xx-n//2).astype(int)
    return np.arange(1, n//2), np.array([P[r == k].mean() for k in range(1, n//2)])
def crps(S, y):  # S (n,...) samples, y (...)
    return float(np.abs(S-y).mean()-0.5*np.abs(S[:, None]-S[None]).mean())
def field_scores(pred, truth):
    return dict(rmse=float(np.sqrt(((pred-truth)**2).mean())), peak_ratio=float((pred.max((1, 2))/truth.max((1, 2))).mean()),
                q999_err_pct=float(100*(np.quantile(pred, .999)/np.quantile(truth, .999)-1)))
def reliability(p, o, bins=10):
    p, o = np.array(p), np.array(o); e = np.linspace(0, 1, bins+1); i = np.clip(np.digitize(p, e)-1, 0, bins-1)
    return [dict(p=float(p[i == b].mean()), obs=float(o[i == b].mean()), n=int((i == b).sum())) for b in range(bins) if (i == b).sum() >= 5]
