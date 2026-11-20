"""
run_fragility.py
================
Step 2 of the pipeline: failure probability of every pole and every wire span on
ONE feeder, for ONE storm scenario -- and, if asked, the Monte Carlo (step 3) right after.

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
    python pipeline/run_fragility.py
        The terminal asks, in order:
          1. the feeder   (3777 Toronto suburban / 1878 Thunder Bay rural -- pipeline/feeders.py)
          2. the weather  ("user", "csv" or "historical"; for "user", the ice and wind)
          3. whether to run the Monte Carlo, and its threshold
        Press Enter at any question to keep the value in the SETTINGS block.
        If the feeder's polegen2 files for the current version are missing, polegen2 is
        run first (pipeline/feeders.py).

Structure:
    get_weather(source, ...)       -> ice_mm, wind_ms   (WHERE the weather comes from)
    run_fragility(ice_mm, wind_ms, feeder)
                                   -> pole_frag, span_frag   (the work; saves nothing)
    run_monte_carlo_step(...)      -> runs pipeline/monte_carlo_failures.py on the result
    main(ask=True)                 -> ask -> get_weather -> run_fragility -> save_and_print
                                      -> (Monte Carlo)
    main()                         -> same, but uses the SETTINGS block (no prompts; tests)
    The terminal questions live in pipeline/terminal_prompts.py.
From other code (e.g. the GUI):
    from pipeline.run_fragility import run_fragility
    pole_frag, span_frag = run_fragility(ice_mm=17.3, wind_ms=14.9, feeder="3777")
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
sys.path.insert(0, str(REPO_ROOT / "code_weather"))

sys.path.insert(0, str(HERE))
from pipeline.asset_adapter import build_asset_tables                           # noqa: E402
import feeders                                                                   # noqa: E402
from pipeline import monte_carlo_failures as mc                                  # noqa: E402
from figure4_fragility_by_span_lookup import lookup_wind_fragility_pole_span   # noqa: E402
from pole_fig7_lookup import lookup_pole_ice_fragility                          # noqa: E402
from wire_fig7d_lookup import lookup_wire_fragility, _site                       # noqa: E402
import weather_input                                                             # noqa: E402
from pipeline import terminal_prompts                                            # noqa: E402

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
# --- FEEDER: which feeder to model --------------------------------------------
# The feeders and their area types are listed in pipeline/feeders.py:
#   "3777" : Toronto, suburban (spans 100-150 ft)
#   "1878" : Thunder Bay, rural (spans 300-400 ft)
FEEDER = "3777"                 # also the default when the terminal asks
AREA_TYPE = None                # None = the feeder's own area type. Set only to override.

# --- WEATHER: which storm to model -------------------------------------------
# Where the weather comes from (see code_weather/weather_input.py):
#   "user"       : the two numbers below (option 3)
#   "historical" : a storm from the historical catalogue (option 2) -- not implemented yet
#   "csv"        : a storm found in a weather CSV (option 1)       -- not implemented yet
WEATHER_SOURCE = "user"          # also the default when the terminal asks

# "user": ice on the wire (mm; 0 = wind-only storm) and 3-s gust (m/s; for an ice
# storm, the gust DURING the ice). Storms modelled so far:
#   Event #1                       : USER_ICE_MM = 14.7, USER_WIND_MS = 12.3
#   Event #1, Toronto default wind : USER_ICE_MM = 14.7, USER_WIND_MS = 31.92
#   Test ice storm                 : USER_ICE_MM = 20,   USER_WIND_MS = 31.92
#   Event #2 (Dec 29, 2025, 1 pm)  : USER_ICE_MM = 0,    USER_WIND_MS = 23.8
#   Test wind storm                : USER_ICE_MM = 0,    USER_WIND_MS = 50
USER_ICE_MM = 0
USER_WIND_MS = 50

# "historical": the storm_id from weather_input.list_historical_storms()
HISTORICAL_STORM_ID = None
# "csv": the weather CSV and the storm_id from weather_input.list_storms_from_csv()
WEATHER_CSV = None
CSV_STORM_ID = None

# --- MODEL CHOICES (not weather) ---------------------------------------------
STRINGING = "slack"             # wire stringing rule: "slack" (default) or "tension"

# Summary only: how many assets are at or above this failure probability
THRESHOLD = 0.10

# --- MONTE CARLO (step 3, pipeline/monte_carlo_failures.py) -------------------
RUN_MONTE_CARLO = False         # with no prompts (main()): run it after the fragility?
                                # (The terminal asks, and defaults to yes.)
MC_THRESHOLD = mc.THRESHOLD     # sample assets with p_fail >= this (0.01-1.0)

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
def get_weather(source=None, user_ice_mm=None, user_wind_ms=None):
    """Decide WHERE the weather comes from. Returns (ice_mm, wind_ms).

    source       : "user", "historical" or "csv". None = WEATHER_SOURCE from SETTINGS.
    user_ice_mm, user_wind_ms : for "user" only -- the numbers typed in the terminal.
                   None = USER_ICE_MM / USER_WIND_MS from SETTINGS.

    ice_mm  : radial ice on the wire (mm). 0 = wind-only storm.
    wind_ms : 3-s gust (m/s). For an ice storm, the gust DURING the ice.
    The sources live in code_weather/weather_input.py; they all return these two numbers.
    """
    source = WEATHER_SOURCE if source is None else source
    if source == "user":
        ice_mm = USER_ICE_MM if user_ice_mm is None else user_ice_mm
        wind_ms = USER_WIND_MS if user_wind_ms is None else user_wind_ms
        return weather_input.user_storm(ice_mm, wind_ms)
    if source == "historical":
        return weather_input.historical_storm(HISTORICAL_STORM_ID)
    if source == "csv":
        return weather_input.storm_from_csv(WEATHER_CSV, CSV_STORM_ID)
    raise ValueError(f"Unknown WEATHER_SOURCE {source!r} "
                     "(use 'user', 'historical' or 'csv').")


def scenario_tag(ice_mm, wind_ms):
    """Name used in the output file names, e.g. ice14.7mm_wind12.3ms or wind50ms."""
    if is_ice_storm(ice_mm):
        return f"ice{ice_mm:g}mm_wind{wind_ms:g}ms"
    return f"wind{wind_ms:g}ms"


# =============================================================================
# 3. RUN
# =============================================================================
def resolve_feeder(feeder=None, area_type=None):
    """(feeder, area_type) with the SETTINGS / pipeline/feeders.py defaults filled in."""
    feeder = FEEDER if feeder is None else feeder
    if area_type is None:
        area_type = AREA_TYPE if AREA_TYPE is not None else feeders.area_type_of(feeder)
    return feeder, area_type


def run_fragility(ice_mm, wind_ms, feeder=None, area_type=None):
    """Weather in -> failure probabilities out. Returns (pole_frag, span_frag).
    feeder None = FEEDER from SETTINGS; area_type None = the feeder's own.
    Saves nothing, so the GUI can call it directly."""
    if ice_mm < 0 or wind_ms < 0:
        raise ValueError(f"ice_mm and wind_ms must be >= 0 (got {ice_mm}, {wind_ms}).")
    feeder, area_type = resolve_feeder(feeder, area_type)
    poles, spans = build_asset_tables(feeder, area_type)
    pole_frag = pole_fragility(poles, ice_mm, wind_ms)
    span_frag = wire_fragility(spans, ice_mm, wind_ms)
    span_frag = add_span_combined(span_frag, pole_frag)
    return pole_frag, span_frag


def save_and_print(pole_frag, span_frag, tag, feeder=None, area_type=None):
    """Write the two CSVs to OUTPUT_DIR and print a short summary."""
    feeder, area_type = resolve_feeder(feeder, area_type)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pole_path = OUTPUT_DIR / f"pole_fragility_{feeder}_{area_type}_{tag}.csv"
    span_path = OUTPUT_DIR / f"span_fragility_{feeder}_{area_type}_{tag}.csv"
    pole_frag.to_csv(pole_path, index=False)
    span_frag.to_csv(span_path, index=False)

    pp, pw, ps = pole_frag["p_fail"], span_frag["p_fail_wire"], span_frag["p_fail_span_combined"]
    print(f"Feeder {feeder} ({area_type}), scenario: {tag}")
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


def run_monte_carlo_step(pole_frag, span_frag, feeder, area_type, tag, threshold):
    """Step 3 on the fragility just made: pipeline/monte_carlo_failures.py.
    Writes failed_lines_matrix.csv (power-flow input) to the feeder's data/opendss folder
    and mc_summary_<...>.csv to pipeline/outputs/monte_carlo/."""
    print(f"\nMonte Carlo: {mc.N_ITERATIONS:,} runs, threshold {threshold:g} ...")
    span_lines, _ = mc.load_line_map(feeder, area_type)
    settings = dict(threshold=threshold, n=mc.N_ITERATIONS, seed=mc.SEED)
    matrix, summary = mc.run_monte_carlo(pole_frag, span_frag, span_lines, **settings)
    mc.save_and_print(matrix, summary, feeder, area_type, tag, **settings)


def main(ask=False, input_fn=input):
    """The wiring: feeder + weather -> run fragility -> save and print -> (Monte Carlo).
    ask=True : ask in the terminal (what  python run_fragility.py  does).
    ask=False: use the SETTINGS block, no prompts (tests, other scripts)."""
    if ask:
        feeder = terminal_prompts.ask_feeder(
            {f: feeders.describe(f) for f in feeders.FEEDERS}, FEEDER, input_fn)
        feeder, area_type = resolve_feeder(feeder)
        source, ice_mm, wind_ms = terminal_prompts.ask_weather(
            WEATHER_SOURCE, USER_ICE_MM, USER_WIND_MS, input_fn)
        # get the weather BEFORE the next question, so a source that isn't ready
        # (csv / historical for now) stops here instead of after more questions
        ice_mm, wind_ms = get_weather(source, user_ice_mm=ice_mm, user_wind_ms=wind_ms)
        run_mc, mc_threshold = terminal_prompts.ask_monte_carlo(
            MC_THRESHOLD, mc.THRESHOLD_MIN, mc.THRESHOLD_MAX, input_fn=input_fn)
    else:
        feeder, area_type = resolve_feeder()
        run_mc, mc_threshold = RUN_MONTE_CARLO, MC_THRESHOLD
        ice_mm, wind_ms = get_weather()

    tag = scenario_tag(ice_mm, wind_ms)
    pole_frag, span_frag = run_fragility(ice_mm, wind_ms, feeder, area_type)
    save_and_print(pole_frag, span_frag, tag, feeder, area_type)
    if run_mc:
        run_monte_carlo_step(pole_frag, span_frag, feeder, area_type, tag, mc_threshold)


if __name__ == "__main__":
    try:
        main(ask=True)
    except NotImplementedError as e:          # csv / historical are not finished yet
        sys.exit(f"\nThat weather source is not ready yet: {e}")
    except (KeyboardInterrupt, EOFError):     # Ctrl+C / Ctrl+D at a prompt
        sys.exit("\nCancelled.")
