from pathlib import Path

BASE_DIR = Path("/Users/laibasabooh/Desktop/Capstone_Pole Gen 2/p1uhs0_1247--p1udt1878").resolve().parent

FEEDER_NAME = "1878"

FEEDER_DIR = BASE_DIR / "p1uhs0_1247--p1udt1878"

MASTER_PATH = FEEDER_DIR / "Master.dss"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pole-generation scenario
AREA_TYPE = "suburban"      # "suburban" or "rural"

SPAN_RANGES_FT = {
    "suburban": (100.0, 150.0),
    "rural": (300.0, 400.0)
}

