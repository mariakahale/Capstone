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
# outputs/
#   feeder_1878/
#     assets/        poles and spans: the pole/span model itself
#                    (read by pipeline/asset_adapter.py for fragility)
#       feeder_1878_poles_suburban.csv
#       feeder_1878_spans_suburban.csv
#       ... and the same for rural
#     mapping/       pole/span -> OpenDSS line links (for power flow)
#       feeder_1878_pole_line_map_suburban.csv
#       feeder_1878_span_line_map_suburban.csv
#       ...
#     intermediate/  working tables, useful for checking the model
#       feeder_1878_overhead.csv
#       feeder_1878_routes.csv
#       feeder_1878_span_plan_suburban.csv
#       ...
#   feeder_3777/
#     (same layout)
#
# File names keep the feeder and area type, so a file still
# makes sense if it is copied out of its folder.
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"

# Which subfolder each output table goes in
OUTPUT_GROUPS = {
    "poles": "assets",
    "spans": "assets",

    "pole_line_map": "mapping",
    "span_line_map": "mapping",

    "overhead": "intermediate",
    "routes": "intermediate",
    "span_plan": "intermediate",
}


def output_path(feeder_name, table_name, area_type=None):
    """
    Full path for one output CSV, creating its folder if needed.

    area_type is left out for tables that don't depend on span
    length (overhead, routes).

    Example:
        output_path("1878", "poles", "rural")
        -> outputs/feeder_1878/assets/feeder_1878_poles_rural.csv
    """
    folder = (
        OUTPUT_DIR
        / f"feeder_{feeder_name}"
        / OUTPUT_GROUPS[table_name]
    )
    folder.mkdir(parents=True, exist_ok=True)

    if area_type is None:
        file_name = f"feeder_{feeder_name}_{table_name}.csv"
    else:
        file_name = f"feeder_{feeder_name}_{table_name}_{area_type}.csv"

    return folder / file_name

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
