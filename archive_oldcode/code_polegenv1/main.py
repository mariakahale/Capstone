import os

from pathlib import Path

from archive_oldcode.code_polegen.config import FEEDER_NAME, MASTER_PATH, PRIMARY_KV_LL, POLE_SPACING_M

from archive_oldcode.code_polegen.grid.feeder import load_feeder, extract_lines

from archive_oldcode.code_polegen.grid.visualization import plot_pole_network

from archive_oldcode.code_polegen.grid.pole_generator import (
    generate_poles_and_spans,
    validate_pole_model
)

from archive_oldcode.code_polegen.grid.line_classifier import (
    get_primary_lines,
    remove_non_conductors,
    get_overhead_lines,
    get_underground_lines,
    get_unclassified_lines
)

# Folder containing main.py
BASE_DIR = Path(__file__).resolve().parent

# Create outputs folder automatically
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make sure output folder exists
os.makedirs("outputs", exist_ok=True)


# -------------------------
# 1. Load feeder
# -------------------------

load_feeder(MASTER_PATH)


# -------------------------
# 2. Extract line data
# -------------------------

lines = extract_lines()

# -------------------------
# Classify primary network
# -------------------------

primary = get_primary_lines(
    lines,
    primary_kv_ll=PRIMARY_KV_LL
)

primary_conductors = remove_non_conductors(primary)

primary_oh = get_overhead_lines(primary_conductors)

primary_ug = get_underground_lines(primary_conductors)

unclassified = get_unclassified_lines(primary_conductors)


# -------------------------
# 3. Basic checks
# -------------------------

# -------------------------
# 4. Save output
# -------------------------

output_file = OUTPUT_DIR / f"feeder{FEEDER_NAME}_lines.csv"

lines.to_csv(output_file, index=False)

primary_oh.to_csv(
    OUTPUT_DIR / f"feeder{FEEDER_NAME}_primary_OH.csv",
    index=False
)

primary_ug.to_csv(
    OUTPUT_DIR / f"feeder{FEEDER_NAME}_primary_UG.csv",
    index=False
)

print("\nSaved:", output_file)

poles, spans = generate_poles_and_spans(
    primary_oh,
    feeder_name=FEEDER_NAME,
    max_span_m=POLE_SPACING_M
)

print("\n--- Synthetic Pole Model ---")

print("OH OpenDSS lines:", len(primary_oh))
print("Unique poles:", len(poles))
print("Synthetic spans:", len(spans))

print(
    "Bus poles:",
    len(poles[poles["pole_type"] == "bus"])
)

print(
    "Intermediate poles:",
    len(poles[poles["pole_type"] == "intermediate"])
)

print(
    "Longest generated span:",
    spans["length_m"].max(),
    "m"
)

poles.to_csv(
    OUTPUT_DIR / f"feeder{FEEDER_NAME}_poles.csv",
    index=False
)

spans.to_csv(
    OUTPUT_DIR / f"feeder{FEEDER_NAME}_spans.csv",
    index=False
)

poles, spans = generate_poles_and_spans(
    primary_oh,
    feeder_name=FEEDER_NAME,
    max_span_m=POLE_SPACING_M
)

validate_pole_model(
    primary_oh,
    poles,
    spans,
    max_span_m=POLE_SPACING_M
)

plot_pole_network(
    poles,
    spans,
    feeder_name=FEEDER_NAME
)