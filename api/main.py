"""REST API over the pipeline output.  uvicorn api.main:app --reload"""
import json, pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
R = json.loads((pathlib.Path(__file__).parent.parent/"data/results.json").read_text())
app = FastAPI(title="CrestWatch", description="Forecaster guidance API. Data provenance: " + R["meta"]["provenance"])
@app.get("/tracks")
def tracks(): return dict(meta=R["meta"], **R["tracks"])
@app.get("/hazards")
def hazards(): return dict(event=R["meta"]["case"], efi_peak=R["efi"]["peak"], members=R["meta"]["members"], first_lead_h=R["alert"].get("lead_h"), category=R["alert"]["category"])
@app.get("/downscale/{lead_h}")
def downscale(lead_h: int): return dict(lead_h=R["alert"]["lead_h"], requested=lead_h, **R["downscale"])
@app.post("/alert")
def alert(): return R["alert"]
@app.get("/alert/cap")
def cap(): return Response(R["cap"], media_type="application/xml")
@app.get("/alert/geojson")
def geojson(): return R["geojson"]
@app.get("/evaluation")
def evaluation(): return R["metrics"]
app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parent.parent/"web", html=True), name="web")
