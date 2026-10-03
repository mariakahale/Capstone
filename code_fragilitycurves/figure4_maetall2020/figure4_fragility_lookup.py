"""
FRAGILITY LOOKUP TABLE -- AGE-DEPENDENT WOOD POLES (Ma et al. 2020, Fig. 4)
==========================================================================
Simple interface for the team: call a function, get a failure probability.
No need to understand the Monte Carlo model to use this file.

USAGE (for teammates):
    from figure4_fragility_lookup import lookup_wind_fragility_pole

    p_fail = lookup_wind_fragility_pole(wind_speed=50, age_years=30)  # m/s, years
    print(p_fail)  # -> 0.300 (approximately)

Public functions:
    lookup_wind_fragility_pole(wind_speed, age_years)
        P(failure) for a pole of the given age (0-60 years).
    lookup_wind_fragility_pole_new_archived(wind_speed)
        P(failure) for the "New poles (archived)" curve from the original code.
    available_ages()
        The ages (years) that have a curve in the table: 0, 5, ..., 60.

Where the numbers come from:
    figure4_fragility.py (the professor's Fig. 4 code, extended to 5-yr
    intervals) runs the Monte Carlo simulation and writes
    figure4_fragility_curves_5yr.csv: one column per age, wind speeds
    20-90 m/s in 0.1 m/s steps. This file only READS that CSV -- it does
    not re-run the simulation. If you change figure4_fragility.py, re-run it
    to regenerate the CSV.

    Unlike guangyang_fragility_lookup.py, no lognormal/logistic fit is done:
    the CSV is already a dense, smooth curve (500,000 samples), so values are
    read straight off it with linear interpolation between the 0.1 m/s points.

Ages between the 5-yr curves (e.g. 32 years):
    Linearly interpolated between the two neighbouring curves (30 and 35 yr).
    This is a blend of the tabulated curves, not a new simulation.

The CSV is loaded ONCE, automatically, the first time this file is imported
(see `_load_curves()` at the bottom).
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "figure4_fragility_curves_5yr.csv"

_WIND = None        # wind speeds (m/s), shape (n_speeds,)
_AGES = None        # ages (years) with a tabulated curve, sorted
_AGE_CURVES = None  # P(failure), shape (n_ages, n_speeds)
_NEW_ARCHIVED = None


# ---------------------------------------------------------------------
# 1. LOAD THE CSV WRITTEN BY figure4_fragility.py
# ---------------------------------------------------------------------

def _load_curves():
    """Runs once at import time. Reads the CSV and caches the curves."""
    global _WIND, _AGES, _AGE_CURVES, _NEW_ARCHIVED

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"[figure4_fragility_lookup] Missing {CSV_PATH.name}. "
            f"Run figure4_fragility.py first to generate it."
        )

    df = pd.read_csv(CSV_PATH)
    _WIND = df["WindSpeed_mps"].to_numpy()

    # Age columns are named like "0_yr", "5_yr", ..., "60_yr"
    age_cols = {int(c.split("_")[0]): c for c in df.columns if c.endswith("_yr")}
    _AGES = np.array(sorted(age_cols))
    _AGE_CURVES = np.vstack([df[age_cols[a]].to_numpy() for a in _AGES])

    if "New_archived" in df.columns:
        _NEW_ARCHIVED = df["New_archived"].to_numpy()


# ---------------------------------------------------------------------
# 2. SHARED LOOKUP LOGIC
# ---------------------------------------------------------------------

def _check_wind_range(wind_speed):
    lo, hi = _WIND[0], _WIND[-1]
    ws = np.asarray(wind_speed, dtype=float)
    if np.any((ws < lo) | (ws > hi)):
        warnings.warn(
            f"[figure4_fragility_lookup] wind speed outside the tabulated range "
            f"({lo:g}-{hi:g} m/s); result is clamped to the end value "
            f"(~0 below {lo:g} m/s, ~1 above {hi:g} m/s)."
        )
    return ws


def _interp_wind(curve, wind_speed):
    ws = _check_wind_range(wind_speed)
    p = np.interp(ws, _WIND, curve)
    return float(p) if p.ndim == 0 else p


# ---------------------------------------------------------------------
# 3. THE PUBLIC FUNCTIONS -- this is what your team actually calls
# ---------------------------------------------------------------------

def lookup_wind_fragility_pole(wind_speed, age_years):
    """Returns P(failure) for a wood pole of the given age at the given wind speed.

    wind_speed : float or array, m/s (tabulated range 20-90 m/s)
    age_years  : float, 0-60 years. Multiples of 5 use the simulated curve
                 directly; other ages interpolate between the two nearest curves.
    """
    lo, hi = _AGES[0], _AGES[-1]
    if not (lo <= age_years <= hi):
        raise ValueError(
            f"age_years={age_years} is outside the available range ({lo}-{hi} yr)."
        )

    # Find the bracketing 5-yr curves and blend them
    i = np.searchsorted(_AGES, age_years)
    if _AGES[min(i, len(_AGES) - 1)] == age_years:
        curve = _AGE_CURVES[i]
    else:
        a0, a1 = _AGES[i - 1], _AGES[i]
        w = (age_years - a0) / (a1 - a0)
        curve = (1 - w) * _AGE_CURVES[i - 1] + w * _AGE_CURVES[i]

    return _interp_wind(curve, wind_speed)


def lookup_wind_fragility_pole_new_archived(wind_speed):
    """Returns P(failure) for the 'New poles (archived)' curve at the given wind speed (m/s)."""
    if _NEW_ARCHIVED is None:
        raise RuntimeError(
            "The CSV has no 'New_archived' column -- re-run figure4_fragility.py "
            "with INCLUDE_ORIGINAL_NEW_POLE = True."
        )
    return _interp_wind(_NEW_ARCHIVED, wind_speed)


def available_ages():
    """Returns the ages (years) that have a simulated curve: [0, 5, ..., 60]."""
    return _AGES.tolist()


# ---------------------------------------------------------------------
# Load everything once, automatically, as soon as this file is imported
# ---------------------------------------------------------------------
_load_curves()


# ---------------------------------------------------------------------
# Demo / sanity check when run directly (not needed for import use)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Available ages (yr):", available_ages())
    print()
    for age in [0, 30, 60]:
        for v in [40, 50, 60]:
            print(f"{age:>2}-yr pole @ {v} m/s wind: "
                  f"P(fail) = {lookup_wind_fragility_pole(v, age):.4f}")
    print()
    print(f"32-yr pole @ 50 m/s wind (interpolated): "
          f"P(fail) = {lookup_wind_fragility_pole(50, 32):.4f}")
    print(f"New (archived) pole @ 50 m/s wind: "
          f"P(fail) = {lookup_wind_fragility_pole_new_archived(50):.4f}")
