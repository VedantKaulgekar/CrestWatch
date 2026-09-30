// node tools/build_land.js  (npm i world-atlas topojson-client) -> web/data/land.js, lon/lat rings clipped near the study domain
const topo = require("topojson-client"), fs = require("fs");
const w = JSON.parse(fs.readFileSync("node_modules/world-atlas/countries-50m.json")), fc = topo.feature(w, w.objects.countries);
const D = JSON.parse(fs.readFileSync("web/data/domain.js", "utf8").replace(/^window.DOMAIN=|;$/g, "")), m = 3, out = [];
for (const f of fc.features) { const g = f.geometry, polys = g.type == "Polygon" ? [g.coordinates] : g.coordinates;
  for (const poly of polys) { const r = poly[0]; if (!r.some(([x, y]) => x > D.west-m && x < D.east+m && y > D.south-m && y < D.north+m)) continue;
    let px = 1e9, py = 1e9; const q = r.filter(([x, y]) => { if (Math.abs(x-px)+Math.abs(y-py) < 0.06) return false; px = x; py = y; return true; });
    if (q.length > 3) out.push({ i: f.id == "356" ? 1 : 0, r: q.map(([x, y]) => [+x.toFixed(2), +y.toFixed(2)]) }); } }
fs.writeFileSync("web/data/land.js", "window.LAND=" + JSON.stringify(out) + ";"); console.log("rings", out.length, "bytes", fs.statSync("web/data/land.js").size);
