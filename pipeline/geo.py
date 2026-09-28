import numpy as np
LAT0, LON0, RES, NY, NX = 8.0, 80.0, 0.25, 65, 65
LATS = LAT0 + RES*np.arange(NY); LONS = LON0 + RES*np.arange(NX)
LEADS = np.arange(0, 241, 12)  # hours
def km(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    la1, lo1, la2, lo2 = map(np.radians, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 12742*np.arcsin(np.sqrt(h))
def radius_km(la, lo):
    yy, xx = LATS[:, None], LONS[None, :]
    return np.sqrt(((yy-la)*111.0)**2 + ((xx-lo)*111.0*np.cos(np.radians(la)))**2)
