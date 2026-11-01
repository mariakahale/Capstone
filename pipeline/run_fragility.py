"""
run_fragility.py
================
Step 2 of the pipeline: failure probability of every pole and every wire span on
the feeder, for ONE storm scenario.

    asset_adapter.py  ->  run_fragility.py  ->  pipeline/outputs/fragility/
    (poles, spans)        (this file)            pole_fragility_<scenario>.csv
                                                 span_fragility_<scenario>.csv
                                                        |
                                                        v
                                              Monte Carlo (next step) -> power flow

What it does
  * ice storm : pole  -> Fig. 7(a)-(c) lookup  (age, tributary span, ice, concurrent wind)
                wire  -> Fig. 7(d) lookup      (span length, ice, concurrent wind)
  * wind storm: pole  -> Fig. 4 lookup         (age, tributary span, 3-s gust)
                wire  -> 0 (Ma et al.: wires do not fail from wind alone)

For the Monte Carlo, use p_fail (poles) and p_fail_wire (spans), and draw each pole and
each wire separately. A pole is shared by the spans on either side of it, so it must be
drawn once and reused, not drawn again for each span.

p_fail_span_combined (Eq. 11: wire + its two end poles, assumed independent) is for maps
and checks ONLY. Do not feed it into the Monte Carlo: a shared pole would be counted twice.

Run:
    python pipeline/run_fragility.py           (edit SETTINGS below first)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
FRAG_DIR = REPO_ROOT / "code_fragilitycurves"

# The lookup files are plain scripts in separate folders, not packages, so their
# folders are added to Python's search path here.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FRAG_DIR / "figure4_maetall2020"))
sys.path.insert(0, str(FRAG_DIR / "figure7_maetall2020" / "differentspanlengths"))

from pipeline.asset_adapter import build_asset_tables, FEEDER_NAME, AREA_TYPE   # noqa: E402
from figure4_fragility_by_span_lookup import lookup_wind_fragility_pole_span   # noqa: E402
from pole_fig7_lookup import lookup_pole_ice_fragility                          # noqa: E402
from wire_fig7d_lookup import lookup_wire_fragility, _site                       # noqa: E402

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================


#EVENT #1: 12.3 wind speed, ice_mm = 14.7
#EVENT #2 (dec 29, 2025, 1pm): ice_mm =0, 23.8m/s

#Modeling Event #2
SCENARIO = "wind"                # "ice" (ice storm with concurrent wind) or "wind" (wind only)
ICE_MM = 0
WIND_GUST_MS = 50

CONCURRENT_WIND_MS = None       # 3-s gust during the ice storm (m/s); None -> Toronto (31.9)
STRINGING = "slack"             # wire stringing rule: "slack" (default) or "tension"



# Summary only: how many assets are at or above this failure probability
THRESHOLD = 0.10

OUTPUT_DIR = HERE / "outputs" / "fragility"


# =============================================================================
# 1. PER-ASSET FAILURE PROBABILITIES
# =============================================================================
def pole_fragility(poles, scenario):
    if scenario == "ice":
        wind = toronto_concurrent_wind() if CONCURRENT_WIND_MS is None else CONCURRENT_WIND_MS
        p = [lookup_pole_ice_fragility(age, trib, ice_mm=ICE_MM, wind_ms=wind)
             for age, trib in zip(poles["age_years"], poles["tributary_m"])]
    elif scenario == "wind":
        p = [lookup_wind_fragility_pole_span(WIND_GUST_MS, age, trib)
             for age, trib in zip(poles["age_years"], poles["tributary_m"])]
    else:
        raise ValueError(f"Unknown SCENARIO '{scenario}' (use 'ice' or 'wind').")
    out = poles[["pole_id", "pole_type", "x", "y", "age_years", "age_true_years",
                 "flag_age_capped", "section_id", "tributary_m",
                 "n_spans", "flag_end_pole", "flag_junction_pole"]].copy()
    out["p_fail"] = p
    return out


def wire_fragility(spans, scenario):
    if scenario == "ice":
        wind = toronto_concurrent_wind() if CONCURRENT_WIND_MS is None else CONCURRENT_WIND_MS
        p = [lookup_wire_fragility(L, ICE_MM, wind, STRINGING) for L in spans["span_m"]]
    else:
        # Ma et al. (2020): "the failure of the wires is unlikely to occur when only
        # subjected to wind", so wire failure is ignored in the wind-only case.
        p = np.zeros(len(spans))
    out = spans[["span_id", "start_pole", "end_pole", "span_m", "voltage_class",
                 "n_phases", "flag_short_span"]].copy()
    out["p_fail_wire"] = p
    return out


def add_span_combined(span_frag, pole_frag):
    """Eq. 11 for a span: fails if its wire or either end pole fails (independent).
    FOR MAPS / CHECKS ONLY -- not an input to the Monte Carlo (see module docstring)."""
    pp = pole_frag.set_index("pole_id")["p_fail"]
    p_start = span_frag["start_pole"].map(pp).to_numpy()
    p_end = span_frag["end_pole"].map(pp).to_numpy()
    p_wire = span_frag["p_fail_wire"].to_numpy()
    span_frag["p_fail_span_combined"] = 1 - (1 - p_wire) * (1 - p_start) * (1 - p_end)
    return span_frag


def toronto_concurrent_wind():
    return float(_site("toronto")["v_conc"])


# =============================================================================
# 2. RUN ONE SCENARIO
# =============================================================================
def scenario_tag():
    if SCENARIO == "ice":
        wind = toronto_concurrent_wind() if CONCURRENT_WIND_MS is None else CONCURRENT_WIND_MS
        return f"ice{ICE_MM:g}mm_wind{wind:g}ms"
    return f"wind{WIND_GUST_MS:g}ms"


def run_scenario():
    """Returns (pole_frag, span_frag) DataFrames for the scenario in SETTINGS."""
    poles, spans = build_asset_tables()
    pole_frag = pole_fragility(poles, SCENARIO)
    span_frag = wire_fragility(spans, SCENARIO)
    span_frag = add_span_combined(span_frag, pole_frag)
    return pole_frag, span_frag


def main():
    pole_frag, span_frag = run_scenario()
    tag = scenario_tag()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pole_path = OUTPUT_DIR / f"pole_fragility_{FEEDER_NAME}_{AREA_TYPE}_{tag}.csv"
    span_path = OUTPUT_DIR / f"span_fragility_{FEEDER_NAME}_{AREA_TYPE}_{tag}.csv"
    pole_frag.to_csv(pole_path, index=False)
    span_frag.to_csv(span_path, index=False)

    pp, pw, ps = pole_frag["p_fail"], span_frag["p_fail_wire"], span_frag["p_fail_span_combined"]
    print(f"Feeder {FEEDER_NAME} ({AREA_TYPE}), scenario: {tag}")
    print(f"  poles: {len(pp):5d}  mean P_fail {pp.mean():.4f}  max {pp.max():.4f}  "
          f">= {THRESHOLD:g}: {(pp >= THRESHOLD).sum()}")
    print(f"  wires: {len(pw):5d}  mean P_fail {pw.mean():.4f}  max {pw.max():.4f}  "
          f">= {THRESHOLD:g}: {(pw >= THRESHOLD).sum()}")
    print(f"  spans (wire + 2 poles, maps only): >= {THRESHOLD:g}: {(ps >= THRESHOLD).sum()}")
    print(f"  expected failures: {pp.sum():.1f} poles, {pw.sum():.1f} wires")
    print(f"Saved:\n  {pole_path.name}\n  {span_path.name}")


if __name__ == "__main__":
    main()
