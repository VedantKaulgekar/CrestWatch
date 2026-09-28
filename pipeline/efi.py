import numpy as np
from .geo import *
def climate(rng, n=200): return 14*rng.weibull(2, (n, NY, NX))  # stand-in for reforecast/IMDAA climate
def efi(members, clim):
    """Extreme Forecast Index (Lalaurette 2003): 2/pi * int_0^1 (p-F(p))/sqrt(p(1-p)) dp,
    F(p) = fraction of members below the climate p-quantile."""
    p = np.linspace(0.01, 0.99, 99); q = np.quantile(clim, p, axis=0)
    F = (members[None] < q[:, None]).mean(1)
    return (2/np.pi)*np.trapezoid((p[:, None, None]-F)/np.sqrt(p*(1-p))[:, None, None], p, axis=0)
