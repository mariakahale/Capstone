from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# =========================================================
# Feeders to process
#
# Key   = short feeder name used in output file names
# Value = OpenDSS master file for that feeder
#
# Use a snapshot master file (no yearly simulation, no plots)
# so the feeder compiles and solves quickly.
# =========================================================

# All OpenDSS source data lives in <repo>/data/opendss/
# (one folder per substation, one subfolder per feeder).
DSS_DATA_DIR = REPO_ROOT / "data" / "opendss"

FEEDERS = {
    "1878": (
        DSS_DATA_DIR
        / "p1uhs0_1247"
        / "p1uhs0_1247--p1udt1878"
        / "Master_snapshot.dss"
    ),

    "3777": (
        DSS_DATA_DIR
        / "p48rhs1_25"
        / "p48rhs1_25--p48rdt3777"
        / "Master_snapshot.dss"
    ),
}

# =========================================================
# Fallback bus coordinates (optional, per feeder)
#
# Some SMART-DS feeder folders are missing a few bus
# coordinates in their own Buscoords.dss. The substation-level
# Buscoords.dss has them. Only buses with NO coordinates are
# filled from this file.
#
# 3777: bus p48rdt9397 (end of line
#       l(r:p48rdt14025-p48rdt9397)_cont) is missing.
# =========================================================

FALLBACK_BUSCOORDS = {
    "3777": (
        DSS_DATA_DIR
        / "p48rhs1_25"
        / "Buscoords.dss"
    ),
}

# =========================================================
# Outputs
#
# All feeders and area types are written to this one folder.
# File names already include the feeder and area type, e.g.
#   feeder_1878_poles_suburban.csv
#   feeder_1878_poles_rural.csv
# which is the naming pipeline/asset_adapter.py expects.
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Pole-generation scenarios
#
# Every feeder is generated once per area type listed here.
# =========================================================

AREA_TYPES = ["suburban", "rural"]

SPAN_RANGES_FT = {
    "suburban": (100.0, 150.0),
    "rural": (300.0, 400.0)
}
