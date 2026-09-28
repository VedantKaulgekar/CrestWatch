import numpy as np
from .downscale import DX
LEVELS = [("Low", 62, "yellow"), ("Moderate", 88, "orange"), ("Severe", 118, "red")]  # km/h, illustrative
def build_alert(samples, centre, lead_h, event="Cyclonic storm (synthetic case)"):
    N = samples.shape[1]; P = {v: (samples > v).mean(0) for _, v, _ in LEVELS}; la0, lo0 = centre
    dg = lambda d: d*DX/111.0
    ll = lambda i, j: [round(lo0+dg(j-N/2)/np.cos(np.radians(la0)), 3), round(la0-dg(i-N/2), 3)]
    top = None; feats = []
    for name, v, colour in LEVELS:
        m = P[v] >= 0.5
        if m.any():
            top = (name, colour, v, m); y, x = np.nonzero(m); y0, y1, x0, x1 = y.min(), y.max()+1, x.min(), x.max()+1
            feats.append(dict(type="Feature", properties=dict(kind="zone", category=name, colour=colour, threshold_kmh=v, min_prob=0.5, area_km2=int(m.sum()*DX*DX)),
                              geometry=dict(type="Polygon", coordinates=[[ll(y0, x0), ll(y0, x1), ll(y1, x1), ll(y1, x0), ll(y0, x0)]])))
    if top is None: return dict(alert=dict(category="None", event=event), geojson=dict(type="FeatureCollection", features=[]), cap="")
    name, colour, v, m = top; W = P[v]*m; ii, jj = np.mgrid[:N, :N]
    ci, cj = (W*ii).sum()/W.sum(), (W*jj).sum()/W.sum(); core = ll(ci, cj)  # [lon, lat]
    feats.insert(0, dict(type="Feature", properties=dict(kind="core", category=name), geometry=dict(type="Point", coordinates=core)))
    alert = dict(category=name, colour=colour, event=event, lead_h=int(lead_h), threshold_kmh=v, core=[core[1], core[0]],
                 core_prob=round(float(P[v][int(round(ci)), int(round(cj))]), 3), n_samples=int(len(samples)),
                 guidance="Forecaster guidance from a synthetic verification run; not a public warning.")
    cap = (f'<?xml version="1.0" encoding="UTF-8"?>\n<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2"><identifier>CRW-SYN-{lead_h}</identifier>'
           f'<sender>crestwatch@example.invalid</sender><sent>2020-05-15T00:00:00+00:00</sent><status>Exercise</status><msgType>Alert</msgType><scope>Restricted</scope>\n'
           f'<info><category>Met</category><event>{event}</event><urgency>Future</urgency><severity>{"Extreme" if name=="Severe" else "Moderate"}</severity><certainty>Possible</certainty>\n'
           f'<headline>{name}: probability at least 0.5 of wind above {v} km/h</headline>\n'
           f'<area><areaDesc>5 km zone around {core[1]:.2f}N {core[0]:.2f}E</areaDesc><circle>{core[1]:.3f},{core[0]:.3f} {np.sqrt(m.sum()*DX*DX/np.pi):.1f}</circle></area></info></alert>')
    return dict(alert=alert, geojson=dict(type="FeatureCollection", features=feats), cap=cap, p_low=P[62], p_mod=P[88], p_sev=P[118])
