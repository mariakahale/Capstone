"""
asset_adapter.py
================
Translates Laiba's pole/span generator output
(code_polegen2/outputs/feeder_<F>/assets/*.csv) into the
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
  * gives each pole its number of phase conductors (n_phases) and an equivalent
    tributary span (tributary_eff_m) for the phase-dependent pole lookups -- see
    build_poles()
  * assigns a pole age (polegen2 has none -- see AGE SETTINGS below). The default,
    AGE_MODE = "alectra", follows Alectra's 2023 wood pole age distribution with
    line sections sharing an installation age: see agedistribution_alectra/
  * caps spans and tributary spans at 150 m (the longest the lookups cover), keeps the
    true values and flags the capped rows -- only rural junction poles hit this
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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "agedistribution_alectra"))
from alectra_age_model import assign_alectra_ages   # noqa: E402

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
POLEGEN_OUTPUT_DIR = REPO_ROOT / "code_polegen2" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEEDER_NAME = "1878"
AREA_TYPE = "suburban"          # one of AREA_TYPES in code_polegen2/config.py ("suburban", "rural")

# AGE SETTINGS -- polegen2 does not give pole ages, so they are assumed here.
#   "alectra" : Alectra 2023 wood pole age distribution, one installation age per line
#               section + a fraction ALECTRA_RHO of individually replaced poles
#               (see agedistribution_alectra/age_assignment_explanation.txt)
#   "fixed"   : every pole gets FIXED_AGE_YEARS
#   "uniform" : random whole-number ages in [AGE_MIN, AGE_MAX], reproducible via AGE_SEED
AGE_MODE = "alectra"
FIXED_AGE_YEARS = 30
AGE_MIN, AGE_MAX = 0, 60        # the lookups only cover 0-60 yr
AGE_SEED = 2026

ALECTRA_RHO = 0.20              # share of spot-replaced poles (0 = pure cohorts, 1 = independent)
ALECTRA_JITTER = 2              # +/- years of spread within a line section
YEARS_SINCE_2023 = 0            # shift ages forward from the 2023 snapshot (0 = as-is)

# Ages above LOOKUP_MAX_AGE are capped for the fragility lookups (age_years) and the
# true assigned age is kept in age_true_years. Capping UNDERESTIMATES the failure
# probability of these old poles -- a stated limitation of the Ma et al. curves.
LOOKUP_MAX_AGE = 60

FT_TO_M = 0.3048

# Model-validity limits used for flags (they match the lookup span bins)
MIN_BIN_M = 10.0                # lookups round anything shorter up to 10 m
MAX_BIN_M = 150.0               # lookups raise an error above this

# Spans and tributary spans above MAX_BIN_M are CAPPED at MAX_BIN_M for the lookups
# (span_m, tributary_m, tributary_eff_m); the true values are kept in span_true_m,
# tributary_true_m and tributary_eff_true_m, and the rows are flagged
# (flag_span_capped, flag_tributary_capped). Same idea as the age cap above.
# Capping UNDERESTIMATES the failure probability of these poles/spans -- a stated
# limitation. In practice this only happens at rural junction poles, where 3+ long
# spans meet (team decision, Nov 10 2026; see the team decisions doc).

# =============================================================================
# OUTPUT CONTRACT -- the columns the fragility step can rely on
# =============================================================================
SPAN_COLUMNS = ["span_id", "start_pole", "end_pole", "span_m", "span_true_m",
                "flag_span_capped", "voltage_class", "n_phases", "flag_short_span"]
POLE_COLUMNS = ["pole_id", "pole_type", "x", "y", "age_years", "age_true_years",
                "flag_age_capped", "section_id", "flag_spot_replacement",
                "n_spans", "span_ids", "tributary_m", "tributary_true_m", "n_phases",
                "tributary_eff_m", "tributary_eff_true_m", "flag_tributary_capped",
                "flag_mixed_phase", "flag_end_pole", "flag_junction_pole"]
# age_years      : age fed to the fragility lookups (capped at LOOKUP_MAX_AGE)
# age_true_years : age assigned by AGE_MODE before capping
# section_id     : line section (poles between junctions); -1 if AGE_MODE is not "alectra"
# n_phases       : phase conductors on the pole = largest n_phases of its attached spans
# span_m / tributary_m / tributary_eff_m : values fed to the lookups (capped at MAX_BIN_M)
# span_true_m / tributary_true_m / tributary_eff_true_m : real values before capping
# tributary_eff_m: tributary span that, with n_phases wires, gives the same total wire
#                  length as the real attached spans (= tributary_m unless flag_mixed_phase)
# flag_tributary_capped: tributary_true_m or tributary_eff_true_m was above MAX_BIN_M
# flag_mixed_phase: attached spans have different phase counts (e.g. 1-phase tap off 3-phase)
# Pole lookups use (tributary_eff_m, n_phases). Neutral wires are neglected (as in Ma et al.).



# =============================================================================
# 1. LOAD (the only part that knows polegen2 file names)
# =============================================================================
def polegen_paths(feeder=FEEDER_NAME, area_type=AREA_TYPE, folder=POLEGEN_OUTPUT_DIR):
    # polegen2 layout: outputs/feeder_<F>/assets/feeder_<F>_poles_<area>.csv
    folder = folder / f"feeder_{feeder}" / "assets"
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
        "span_true_m": raw_spans["length_ft"] * FT_TO_M,
        "voltage_class": raw_spans["voltage_class"],
        "n_phases": raw_spans["construction_phases"].astype(int),
    })
    spans["span_m"] = np.minimum(spans["span_true_m"], MAX_BIN_M)
    spans["flag_span_capped"] = spans["span_true_m"] > MAX_BIN_M
    spans["flag_short_span"] = spans["span_true_m"] < MIN_BIN_M
    return spans[SPAN_COLUMNS]


def assign_ages(pole_ids, spans, mode=AGE_MODE):
    """Returns a DataFrame (one row per pole, same order as pole_ids) with
    age_true_years, section_id, flag_spot_replacement."""
    n = len(pole_ids)
    if mode == "alectra":
        a = assign_alectra_ages(pole_ids, spans, rho=ALECTRA_RHO, jitter=ALECTRA_JITTER,
                                seed=AGE_SEED, years_since_2023=YEARS_SINCE_2023)
        return pd.DataFrame({"age_true_years": a["age_years"].to_numpy(),
                             "section_id": a["section_id"].to_numpy(),
                             "flag_spot_replacement": a["spot"].to_numpy()})
    if mode == "fixed":
        ages = np.full(n, FIXED_AGE_YEARS, dtype=int)
    elif mode == "uniform":
        rng = np.random.default_rng(AGE_SEED)
        ages = rng.integers(AGE_MIN, AGE_MAX, size=n, endpoint=True)
    else:
        raise ValueError(f"Unknown AGE_MODE '{mode}' (use 'alectra', 'fixed' or 'uniform').")
    return pd.DataFrame({"age_true_years": ages, "section_id": -1,
                         "flag_spot_replacement": False})


def build_poles(raw_poles, spans, age_mode=AGE_MODE):
    """One row per pole, with its tributary span (half of every attached span).

    Phases: the pole models load the pole with (number of phase wires) x (tributary
    span), for both wind area and weight, so only the total length of phase wire on
    the pole matters:
        wire_m          = sum over attached spans of n_phases_i * span_i / 2
        n_phases        = max n_phases_i
        tributary_eff_m = wire_m / n_phases
    For a pole whose spans all have the same phase count, tributary_eff_m = tributary_m.
    For a mixed pole (e.g. 3-phase main + 1-phase tap) this gives exactly the same wire
    load as summing each span with its own phase count."""
    span_len = spans.set_index("span_id")["span_true_m"]   # real lengths, cap applied after
    span_ph = spans.set_index("span_id")["n_phases"]
    span_lists = raw_poles["connected_span_ids"].str.split(";")

    unknown = set(s for ids in span_lists for s in ids) - set(span_len.index)
    if unknown:
        raise ValueError(f"Poles reference {len(unknown)} span_id(s) not in the spans file, "
                         f"e.g. {sorted(unknown)[:5]}")

    n_spans = span_lists.str.len()
    n_phases = span_lists.apply(lambda ids: int(span_ph[ids].max()))
    wire_m = span_lists.apply(lambda ids: float((span_len[ids] * span_ph[ids]).sum() / 2))
    mixed = span_lists.apply(lambda ids: span_ph[ids].nunique() > 1)
    ages = assign_ages(raw_poles["pole_id"], spans, age_mode)
    poles = pd.DataFrame({
        "pole_id": raw_poles["pole_id"],
        "pole_type": raw_poles["pole_type"],
        "x": raw_poles["x"],
        "y": raw_poles["y"],
        "age_years": np.minimum(ages["age_true_years"], LOOKUP_MAX_AGE).to_numpy(),
        "age_true_years": ages["age_true_years"].to_numpy(),
        "flag_age_capped": (ages["age_true_years"] > LOOKUP_MAX_AGE).to_numpy(),
        "section_id": ages["section_id"].to_numpy(),
        "flag_spot_replacement": ages["flag_spot_replacement"].to_numpy(),
        "n_spans": n_spans,
        "span_ids": raw_poles["connected_span_ids"],
        "tributary_true_m": span_lists.apply(lambda ids: span_len[ids].sum() / 2),
        "n_phases": n_phases,
        "tributary_eff_true_m": wire_m / n_phases,
        "flag_mixed_phase": mixed,
    })
    poles["tributary_m"] = np.minimum(poles["tributary_true_m"], MAX_BIN_M)
    poles["tributary_eff_m"] = np.minimum(poles["tributary_eff_true_m"], MAX_BIN_M)
    poles["flag_tributary_capped"] = ((poles["tributary_true_m"] > MAX_BIN_M)
                                      | (poles["tributary_eff_true_m"] > MAX_BIN_M))
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
    if not poles["age_years"].between(0, LOOKUP_MAX_AGE).all():
        problems.append(f"pole age outside 0-{LOOKUP_MAX_AGE} yr (lookups would fail)")
    if not spans["n_phases"].isin([1, 2, 3]).all():
        problems.append("span n_phases not in {1, 2, 3} (lookups would fail)")
    if ((spans["span_m"] > MAX_BIN_M).any() or (poles["tributary_m"] > MAX_BIN_M).any()
            or (poles["tributary_eff_m"] > MAX_BIN_M).any()):
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
    print(f"  caps : {spans.flag_span_capped.sum()} spans and "
          f"{poles.flag_tributary_capped.sum()} poles capped at {MAX_BIN_M:g} m for the lookups "
          f"(max true tributary {poles.tributary_true_m.max():.1f} m)")
    print(f"  phases: spans {spans.n_phases.value_counts().sort_index().to_dict()}, "
          f"poles {poles.n_phases.value_counts().sort_index().to_dict()}, "
          f"{poles.flag_mixed_phase.sum()} mixed-phase poles")
    print(f"  ages : mean {poles.age_true_years.mean():.1f} yr, "
          f"range {poles.age_true_years.min()}-{poles.age_true_years.max()} yr, "
          f"{poles.flag_age_capped.sum()} capped at {LOOKUP_MAX_AGE} yr for the lookups")
    if AGE_MODE == "alectra":
        print(f"         {poles.section_id.nunique()} line sections, "
              f"{poles.flag_spot_replacement.sum()} spot replacements (rho = {ALECTRA_RHO})")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
