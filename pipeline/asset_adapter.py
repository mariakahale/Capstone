"""
asset_adapter.py
================
Translates Laiba's pole/span generator output (code_polegen2/outputs/*.csv) into the
two standard tables the fragility step reads. This is the ONLY file that knows the
polegen2 CSV layout; if her output changes, only this file changes.

    polegen2 CSVs  ->  asset_adapter.py  ->  pipeline/outputs/poles_for_fragility.csv
                                             pipeline/outputs/spans_for_fragility.csv
                                                     |
                                                     v
                                            fragility lookups (Fig. 4, Fig. 7)

What it does
  * converts span lengths from feet to metres
  * gives each pole its tributary span: half of every span attached to it
    (end pole = one span / 2; junction poles with 3-4 spans = sum / 2)
  * assigns a pole age (polegen2 has none -- see AGE SETTINGS below)
  * checks the tables and flags rows that fall outside the model's assumptions

What it does NOT do
  * no fragility lookups and no rounding into span/age bins -- the lookup files own that
  * no imports from code_polegen2 or code_fragilitycurves (it only reads CSV files)

Run:
    python pipeline/asset_adapter.py                 (from the repo root, or anywhere)

Use from other code:
    from pipeline.asset_adapter import build_asset_tables
    poles, spans = build_asset_tables()
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
POLEGEN_OUTPUT_DIR = REPO_ROOT / "code_polegen2" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEEDER_NAME = "1878"
AREA_TYPE = "suburban"          # must match code_polegen2/config.py AREA_TYPE

# AGE SETTINGS -- polegen2 does not give pole ages, so they are assumed here.
#   "fixed"   : every pole gets FIXED_AGE_YEARS
#   "uniform" : random whole-number ages in [AGE_MIN, AGE_MAX], reproducible via AGE_SEED
AGE_MODE = "fixed"
FIXED_AGE_YEARS = 30
AGE_MIN, AGE_MAX = 0, 60        # the lookups only cover 0-60 yr
AGE_SEED = 2026

FT_TO_M = 0.3048

# Model-validity limits used for flags (they match the lookup span bins)
MIN_BIN_M = 10.0                # lookups round anything shorter up to 10 m
MAX_BIN_M = 150.0               # lookups raise an error above this

# =============================================================================
# OUTPUT CONTRACT -- the columns the fragility step can rely on
# =============================================================================
SPAN_COLUMNS = ["span_id", "start_pole", "end_pole", "span_m",
                "voltage_class", "n_phases", "flag_short_span"]
POLE_COLUMNS = ["pole_id", "pole_type", "x", "y", "age_years", "n_spans", "span_ids",
                "tributary_m", "flag_end_pole", "flag_junction_pole"]


# =============================================================================
# 1. LOAD (the only part that knows polegen2 file names)
# =============================================================================
def polegen_paths(feeder=FEEDER_NAME, area_type=AREA_TYPE, folder=POLEGEN_OUTPUT_DIR):
    return {
        "poles": folder / f"feeder_{feeder}_poles_{area_type}.csv",
        "spans": folder / f"feeder_{feeder}_spans_{area_type}.csv",
    }


def load_polegen(feeder=FEEDER_NAME, area_type=AREA_TYPE, folder=POLEGEN_OUTPUT_DIR):
    paths = polegen_paths(feeder, area_type, folder)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing polegen2 output(s):\n  " + "\n  ".join(missing) +
            "\nRun code_polegen2/main.py first (check FEEDER_NAME / AREA_TYPE match).")
    return pd.read_csv(paths["poles"]), pd.read_csv(paths["spans"])


# =============================================================================
# 2. TRANSLATE (pure functions: DataFrame in, DataFrame out)
# =============================================================================
def build_spans(raw_spans):
    """One row per span (= one wire segment between two poles), lengths in metres."""
    spans = pd.DataFrame({
        "span_id": raw_spans["span_id"],
        "start_pole": raw_spans["start_pole"],
        "end_pole": raw_spans["end_pole"],
        "span_m": raw_spans["length_ft"] * FT_TO_M,
        "voltage_class": raw_spans["voltage_class"],
        "n_phases": raw_spans["construction_phases"].astype(int),
    })
    spans["flag_short_span"] = spans["span_m"] < MIN_BIN_M
    return spans[SPAN_COLUMNS]


def assign_ages(n, mode=AGE_MODE):
    if mode == "fixed":
        return np.full(n, FIXED_AGE_YEARS, dtype=int)
    if mode == "uniform":
        rng = np.random.default_rng(AGE_SEED)
        return rng.integers(AGE_MIN, AGE_MAX, size=n, endpoint=True)
    raise ValueError(f"Unknown AGE_MODE '{mode}' (use 'fixed' or 'uniform').")


def build_poles(raw_poles, spans, age_mode=AGE_MODE):
    """One row per pole, with its tributary span (half of every attached span)."""
    span_len = spans.set_index("span_id")["span_m"]
    span_lists = raw_poles["connected_span_ids"].str.split(";")

    unknown = set(s for ids in span_lists for s in ids) - set(span_len.index)
    if unknown:
        raise ValueError(f"Poles reference {len(unknown)} span_id(s) not in the spans file, "
                         f"e.g. {sorted(unknown)[:5]}")

    n_spans = span_lists.str.len()
    poles = pd.DataFrame({
        "pole_id": raw_poles["pole_id"],
        "pole_type": raw_poles["pole_type"],
        "x": raw_poles["x"],
        "y": raw_poles["y"],
        "age_years": assign_ages(len(raw_poles), age_mode),
        "n_spans": n_spans,
        "span_ids": raw_poles["connected_span_ids"],
        "tributary_m": span_lists.apply(lambda ids: span_len[ids].sum() / 2),
    })
    poles["flag_end_pole"] = n_spans == 1
    # 3+ spans: wires pull in different directions, so summing them overstates the
    # load on the pole (conservative). Flagged so it can be stated as a limitation.
    poles["flag_junction_pole"] = n_spans >= 3
    return poles[POLE_COLUMNS]


# =============================================================================
# 3. CHECK
# =============================================================================
def validate(poles, spans):
    problems = []
    if poles["pole_id"].duplicated().any():
        problems.append("duplicate pole_id")
    if spans["span_id"].duplicated().any():
        problems.append("duplicate span_id")
    if (spans["span_m"] <= 0).any():
        problems.append("span with length <= 0")
    ends = set(spans["start_pole"]) | set(spans["end_pole"])
    if not ends <= set(poles["pole_id"]):
        problems.append("span endpoints not found in the poles file")
    if not poles["age_years"].between(AGE_MIN, AGE_MAX).all():
        problems.append(f"pole age outside {AGE_MIN}-{AGE_MAX} yr (lookups would fail)")
    if (spans["span_m"] > MAX_BIN_M).any() or (poles["tributary_m"] > MAX_BIN_M).any():
        problems.append(f"span or tributary span above {MAX_BIN_M} m (lookups would fail)")
    if problems:
        raise ValueError("asset_adapter checks failed: " + "; ".join(problems))


# =============================================================================
# 4. PUBLIC ENTRY POINT
# =============================================================================
def build_asset_tables(feeder=FEEDER_NAME, area_type=AREA_TYPE, age_mode=AGE_MODE):
    """Returns (poles, spans) DataFrames in the standard format. Does not write files."""
    raw_poles, raw_spans = load_polegen(feeder, area_type)
    spans = build_spans(raw_spans)
    poles = build_poles(raw_poles, spans, age_mode)
    validate(poles, spans)
    return poles, spans


def main():
    poles, spans = build_asset_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{FEEDER_NAME}_{AREA_TYPE}"
    poles.to_csv(OUTPUT_DIR / f"poles_for_fragility_{tag}.csv", index=False)
    spans.to_csv(OUTPUT_DIR / f"spans_for_fragility_{tag}.csv", index=False)

    print(f"Feeder {FEEDER_NAME} ({AREA_TYPE}), ages: {AGE_MODE}")
    print(f"  spans: {len(spans)}  (length {spans.span_m.min():.1f}-{spans.span_m.max():.1f} m, "
          f"{spans.flag_short_span.sum()} under {MIN_BIN_M:g} m)")
    print(f"  poles: {len(poles)}  (tributary {poles.tributary_m.min():.1f}-"
          f"{poles.tributary_m.max():.1f} m, {poles.flag_end_pole.sum()} end poles, "
          f"{poles.flag_junction_pole.sum()} junction poles)")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
