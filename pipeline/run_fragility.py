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
  * ice storm : pole  -> Fig. 7(a)-(c) lookup  (age, tributary span, phases, ice, concurrent wind)
                wire  -> Fig. 7(d) lookup      (span length, ice, concurrent wind)
  * wind storm: pole  -> Fig. 4 lookup         (age, tributary span, phases, 3-s gust)
                wire  -> 0 (Ma et al.: wires do not fail from wind alone)

Phases (Nov 2026): a pole's wire load scales with its number of phase conductors
(1-phase pole = 1/3 of Ma's 3-phase wire load; neutral neglected, as in Ma). Poles use
(tributary_eff_m, n_phases) from asset_adapter.py. The wire lookup is per conductor and
does not change with phases: all phase wires in a span see the same storm, so in Ma's
model they fail together and one draw per span is enough.

For the Monte Carlo, use p_fail (poles) and p_fail_wire (spans), and draw each pole and
each wire separately. A pole is shared by the spans on either side of it, so it must be
drawn once and reused, not drawn again for each span.

p_fail_span_combined (Eq. 11: wire + its two end poles, assumed independent) is for maps
and checks ONLY. Do not feed it into the Monte Carlo: a shared pole would be counted twice.

Run:
    python pipeline/run_fragility.py           (choose the storm in get_weather() first)

Structure:
    get_weather()                  -> ice_mm, wind_ms   (WHERE the weather comes from)
    run_fragility(ice_mm, wind_ms) -> pole_frag, span_frag   (the work; saves nothing)
    main()                         -> get_weather -> run_fragility -> save_and_print
From other code (e.g. the GUI):
    from pipeline.run_fragility import run_fragility
    pole_frag, span_frag = run_fragility(ice_mm=17.3, wind_ms=14.9)
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
# The WEATHER (ice thickness, wind speed) is chosen in get_weather() below.
# The settings here are model choices, not weather.

STRINGING = "slack"             # wire stringing rule: "slack" (default) or "tension"

# Summary only: how many assets are at or above this failure probability
THRESHOLD = 0.10

OUTPUT_DIR = HERE / "outputs" / "fragility"


# =============================================================================
# 1. PER-ASSET FAILURE PROBABILITIES
# =============================================================================
def is_ice_storm(ice_mm):
    """Ice on the wire -> ice curves (Fig. 7); no ice -> wind curves (Fig. 4)."""
    return ice_mm > 0


def pole_fragility(poles, ice_mm, wind_ms):
    """p_fail for every pole. Ice storm: wind_ms is the gust during the ice."""
    if is_ice_storm(ice_mm):
        p = [lookup_pole_ice_fragility(age, trib, ice_mm=ice_mm, wind_ms=wind_ms, n_phases=ph)
             for age, trib, ph in zip(poles["age_years"], poles["tributary_eff_m"],
                                      poles["n_phases"])]
    else:
        p = [lookup_wind_fragility_pole_span(wind_ms, age, trib, n_phases=ph)
             for age, trib, ph in zip(poles["age_years"], poles["tributary_eff_m"],
                                      poles["n_phases"])]
    out = poles[["pole_id", "pole_type", "x", "y", "age_years", "age_true_years",
                 "flag_age_capped", "section_id", "tributary_m", "tributary_true_m",
                 "n_phases", "tributary_eff_m", "flag_tributary_capped", "flag_mixed_phase",
                 "n_spans", "flag_end_pole", "flag_junction_pole"]].copy()
    out["p_fail"] = p
    return out


def wire_fragility(spans, ice_mm, wind_ms):
    """p_fail_wire for every span."""
    if is_ice_storm(ice_mm):
        p = [lookup_wire_fragility(L, ice_mm, wind_ms, STRINGING) for L in spans["span_m"]]
    else:
        # Ma et al. (2020): "the failure of the wires is unlikely to occur when only
        # subjected to wind", so wire failure is ignored in the wind-only case.
        p = np.zeros(len(spans))
    out = spans[["span_id", "start_pole", "end_pole", "span_m", "span_true_m",
                 "flag_span_capped", "voltage_class", "n_phases", "flag_short_span"]].copy()
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
    """Toronto's 3-s gust during an ice storm (m/s), for when no measured wind is known."""
    return float(_site("toronto")["v_conc"])


# =============================================================================
# 2. WEATHER INPUT  <-- choose the storm here
# =============================================================================
def get_weather():
    """Decide WHERE the weather comes from. Returns (ice_mm, wind_ms).

    ice_mm  : radial ice on the wire (mm). 0 = wind-only storm.
    wind_ms : 3-s gust (m/s). For an ice storm, the gust DURING the ice.

    Today the numbers are typed here. Later this is where the other sources plug in:
      option 1: the user's weather CSV   (code_weather/find_weather_events.py)
      option 2: a table of historical storms
      option 3: numbers typed in the GUI (the GUI can also call run_fragility() directly)
    """
    # Event #1:  ice 14.7 mm, measured wind 12.3 m/s
    # return 14.7, 12.3
    # Event #1 ice with Toronto's default concurrent wind (31.92 m/s)
    # return 14.7, toronto_concurrent_wind()
    # Test ice storm
    # return 20, toronto_concurrent_wind()
    # Event #2 (Dec 29, 2025, 1 pm): wind only
    # return 0, 23.8
    # Test wind storm
    return 0, 50


def scenario_tag(ice_mm, wind_ms):
    """Name used in the output file names, e.g. ice14.7mm_wind12.3ms or wind50ms."""
    if is_ice_storm(ice_mm):
        return f"ice{ice_mm:g}mm_wind{wind_ms:g}ms"
    return f"wind{wind_ms:g}ms"


# =============================================================================
# 3. RUN
# =============================================================================
def run_fragility(ice_mm, wind_ms):
    """Weather in -> failure probabilities out. Returns (pole_frag, span_frag).
    Saves nothing, so the GUI can call it directly."""
    if ice_mm < 0 or wind_ms < 0:
        raise ValueError(f"ice_mm and wind_ms must be >= 0 (got {ice_mm}, {wind_ms}).")
    poles, spans = build_asset_tables()
    pole_frag = pole_fragility(poles, ice_mm, wind_ms)
    span_frag = wire_fragility(spans, ice_mm, wind_ms)
    span_frag = add_span_combined(span_frag, pole_frag)
    return pole_frag, span_frag


def save_and_print(pole_frag, span_frag, tag):
    """Write the two CSVs to OUTPUT_DIR and print a short summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pole_path = OUTPUT_DIR / f"pole_fragility_{FEEDER_NAME}_{AREA_TYPE}_{tag}.csv"
    span_path = OUTPUT_DIR / f"span_fragility_{FEEDER_NAME}_{AREA_TYPE}_{tag}.csv"
    pole_frag.to_csv(pole_path, index=False)
    span_frag.to_csv(span_path, index=False)

    pp, pw, ps = pole_frag["p_fail"], span_frag["p_fail_wire"], span_frag["p_fail_span_combined"]
    print(f"Feeder {FEEDER_NAME} ({AREA_TYPE}), scenario: {tag}")
    print(f"  poles: {len(pp):5d}  mean P_fail {pp.mean():.4f}  max {pp.max():.4f}  "
          f">= {THRESHOLD:g}: {(pp >= THRESHOLD).sum()}")
    for ph, g in pole_frag.groupby("n_phases")["p_fail"]:
        print(f"    {ph}-phase poles: {len(g):5d}  mean P_fail {g.mean():.4f}  "
              f"expected failures {g.sum():.1f}")
    print(f"  wires: {len(pw):5d}  mean P_fail {pw.mean():.4f}  max {pw.max():.4f}  "
          f">= {THRESHOLD:g}: {(pw >= THRESHOLD).sum()}")
    print(f"  spans (wire + 2 poles, maps only): >= {THRESHOLD:g}: {(ps >= THRESHOLD).sum()}")
    print(f"  expected failures: {pp.sum():.1f} poles, {pw.sum():.1f} wires")
    print(f"Saved:\n  {pole_path.name}\n  {span_path.name}")


def main():
    """The wiring: get weather -> run fragility -> save and print."""
    ice_mm, wind_ms = get_weather()
    pole_frag, span_frag = run_fragility(ice_mm, wind_ms)
    save_and_print(pole_frag, span_frag, scenario_tag(ice_mm, wind_ms))


if __name__ == "__main__":
    main()
