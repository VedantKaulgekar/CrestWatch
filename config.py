"""One place for the study domain, events and years. Edit here; everything (fetchers, trainers, website) follows."""
import os, json, pathlib
ROOT = pathlib.Path(__file__).parent
RAW = pathlib.Path(os.environ.get("CW_RAW", ROOT/"data/raw"))          # CW_RAW=data/mock for the smoke test
# India + Arabian Sea + Bay of Bengal + Andaman Sea margin (Kashmir ~37N, Andaman ~93E, Arabian Sea cyclone genesis ~60-70E)
DOMAIN = dict(north=38.0, west=60.0, south=5.0, east=100.0)
AREA = [DOMAIN["north"], DOMAIN["west"], DOMAIN["south"], DOMAIN["east"]]   # CDS order: N, W, S, E
RES = 0.25
SEASONS = {"premonsoon": ["04", "05", "06"], "postmonsoon": ["10", "11", "12"]}   # North Indian Ocean cyclone seasons
TC_TIMES = ["00:00", "06:00", "12:00", "18:00"]
TRAIN_YEARS = [int(y) for y in os.environ["CW_YEARS"].split(",")] if "CW_YEARS" in os.environ else list(range(2010, 2023))
TEST_EVENTS = [("AMPHAN", 2020), ("YAAS", 2021)]                         # held out of GNN training (+-3 days around each)
HEAT_YEARS, HEAT_MONTHS, HEAT_TIME = list(range(2016, 2025)), ["04", "05", "06"], "09:00"   # ERA5 -> ERA5-Land pairs, afternoon peak
def write_domain_js(path=ROOT/"web/data/domain.js"):
    pathlib.Path(path).write_text("window.DOMAIN=" + json.dumps(DOMAIN) + ";")
