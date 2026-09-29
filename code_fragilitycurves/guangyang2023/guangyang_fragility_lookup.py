"""
FRAGILITY LOOKUP TABLE
=======================
Simple interface for the team: call a function, get a failure probability.
No need to understand curve-fitting to use this file.

USAGE (for teammates):
    from fragility_lookup import lookup_wind_fragility_class4_pole

    p_fail = lookup_wind_fragility_class4_pole(wind_speed=25)  # m/s
    print(p_fail)  # -> 0.412 (for example)

12 functions total:
    6 wind-speed lookups:  class2/4/5 x pole/wire
    6 ice-thickness lookups: class2/4/5 x pole/wire

Each fits the SAME digitized data and SAME lognormal/logistic comparison
logic as fit_fragility_curves.py -- this file just wraps that fitting
into clean, single-purpose functions so nobody has to touch scipy or
curve_fit directly.

All fitting happens ONCE, automatically, the first time this file is
imported (see `_fit_all_curves()` at the bottom). Results are cached in
memory after that -- calling a lookup function 1000 times only fits once.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# 1. SAME FUNCTIONAL FORMS AS fit_fragility_curves.py
# ---------------------------------------------------------------------

def _lognormal_cdf(x, median, beta):
    x = np.maximum(x, 1e-9)
    return norm.cdf(np.log(x / median) / beta)


def _logistic_cdf(x, a, b):
    return 1.0 / (1.0 + np.exp(-(x - a) / b))


def _fit_one_curve(csv_path):
    """Fit both forms to one CSV, return (winner_name, params_tuple)."""
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    x_data = df.iloc[:, 0].to_numpy()
    y_data = df.iloc[:, 1].to_numpy()
    order = np.argsort(x_data)
    x_data, y_data = x_data[order], y_data[order]

    best_winner, best_params, best_sse = None, None, np.inf

    try:
        p0 = [np.median(x_data), 0.3]
        popt, _ = curve_fit(_lognormal_cdf, x_data, y_data, p0=p0,
                             bounds=([1e-6, 1e-3], [np.inf, np.inf]), maxfev=10000)
        sse = np.sum((y_data - _lognormal_cdf(x_data, *popt)) ** 2)
        if sse < best_sse:
            best_winner, best_params, best_sse = "lognormal", popt, sse
    except RuntimeError:
        pass

    try:
        p0 = [np.median(x_data), (x_data.max() - x_data.min()) / 10]
        popt, _ = curve_fit(_logistic_cdf, x_data, y_data, p0=p0, maxfev=10000)
        sse = np.sum((y_data - _logistic_cdf(x_data, *popt)) ** 2)
        if sse < best_sse:
            best_winner, best_params, best_sse = "logistic", popt, sse
    except RuntimeError:
        pass

    if best_winner is None:
        raise RuntimeError(f"Both fits failed for {csv_path}")

    return best_winner, best_params


# ---------------------------------------------------------------------
# 2. CURVE DEFINITIONS -- same 12 files as fit_fragility_curves.py
# ---------------------------------------------------------------------

_CURVE_DEFS = {
    "wind_class2_pole": dict(path="windvfailure/class2pole.csv", valid_range=(12.5, 40)),
    "wind_class2_wire": dict(path="windvfailure/class2wire.csv", valid_range=(12.5, 40)),
    "wind_class4_pole": dict(path="windvfailure/class4pole.csv", valid_range=(12.5, 40)),
    "wind_class4_wire": dict(path="windvfailure/class4wire.csv", valid_range=(12.5, 40)),
    "wind_class5_pole": dict(path="windvfailure/class5pole.csv", valid_range=(12.5, 40)),
    "wind_class5_wire": dict(path="windvfailure/class5wire.csv", valid_range=(12.5, 40)),
    "ice_class2_pole":  dict(path="icethicknessvfailure/class2pole.csv", valid_range=(0.635, 3.81)),
    "ice_class2_wire":  dict(path="icethicknessvfailure/class2wire.csv", valid_range=(0.635, 3.81)),
    "ice_class4_pole":  dict(path="icethicknessvfailure/class4pole.csv", valid_range=(0.635, 3.81)),
    "ice_class4_wire":  dict(path="icethicknessvfailure/class4wire.csv", valid_range=(0.635, 3.81)),
    "ice_class5_pole":  dict(path="icethicknessvfailure/class5pole.csv", valid_range=(0.635, 3.81)),
    "ice_class5_wire":  dict(path="icethicknessvfailure/class5wire.csv", valid_range=(0.635, 3.81)),
}

_FITTED = {}  # populated once by _fit_all_curves() below


def _fit_all_curves():
    """Runs once at import time. Fits every curve, caches winner+params."""
    for key, cfg in _CURVE_DEFS.items():
        full_path = SCRIPT_DIR / cfg["path"]
        if not full_path.exists():
            warnings.warn(f"[fragility_lookup] Missing file, skipping: {cfg['path']}")
            continue
        winner, params = _fit_one_curve(full_path)
        _FITTED[key] = dict(winner=winner, params=params,
                             valid_range=cfg["valid_range"])


def _lookup(key, x_value):
    """Shared logic behind all 12 public functions below."""
    if key not in _FITTED:
        raise RuntimeError(
            f"'{key}' was never fitted -- check that its CSV file exists "
            f"at the path listed in _CURVE_DEFS."
        )
    entry = _FITTED[key]
    lo, hi = entry["valid_range"]
    if not (lo <= x_value <= hi):
        warnings.warn(
            f"[fragility_lookup] {x_value} is outside the validated range "
            f"({lo}-{hi}) for '{key}'. Result may be unreliable."
        )
    if entry["winner"] == "lognormal":
        return float(_lognormal_cdf(x_value, *entry["params"]))
    else:
        return float(_logistic_cdf(x_value, *entry["params"]))


# ---------------------------------------------------------------------
# 3. THE 12 PUBLIC FUNCTIONS -- this is what your team actually calls
# ---------------------------------------------------------------------

def lookup_wind_fragility_class2_pole(wind_speed):
    """Returns P(failure) for a Class 2 POLE at the given wind speed (m/s)."""
    return _lookup("wind_class2_pole", wind_speed)

def lookup_wind_fragility_class2_wire(wind_speed):
    """Returns P(failure) for a Class 2 WIRE at the given wind speed (m/s)."""
    return _lookup("wind_class2_wire", wind_speed)

def lookup_wind_fragility_class4_pole(wind_speed):
    """Returns P(failure) for a Class 4 POLE at the given wind speed (m/s)."""
    return _lookup("wind_class4_pole", wind_speed)

def lookup_wind_fragility_class4_wire(wind_speed):
    """Returns P(failure) for a Class 4 WIRE at the given wind speed (m/s)."""
    return _lookup("wind_class4_wire", wind_speed)

def lookup_wind_fragility_class5_pole(wind_speed):
    """Returns P(failure) for a Class 5 POLE at the given wind speed (m/s)."""
    return _lookup("wind_class5_pole", wind_speed)

def lookup_wind_fragility_class5_wire(wind_speed):
    """Returns P(failure) for a Class 5 WIRE at the given wind speed (m/s)."""
    return _lookup("wind_class5_wire", wind_speed)

def lookup_ice_fragility_class2_pole(ice_thickness_cm):
    """Returns P(failure) for a Class 2 POLE at the given ice thickness (cm)."""
    return _lookup("ice_class2_pole", ice_thickness_cm)

def lookup_ice_fragility_class2_wire(ice_thickness_cm):
    """Returns P(failure) for a Class 2 WIRE at the given ice thickness (cm)."""
    return _lookup("ice_class2_wire", ice_thickness_cm)

def lookup_ice_fragility_class4_pole(ice_thickness_cm):
    """Returns P(failure) for a Class 4 POLE at the given ice thickness (cm)."""
    return _lookup("ice_class4_pole", ice_thickness_cm)

def lookup_ice_fragility_class4_wire(ice_thickness_cm):
    """Returns P(failure) for a Class 4 WIRE at the given ice thickness (cm)."""
    return _lookup("ice_class4_wire", ice_thickness_cm)

def lookup_ice_fragility_class5_pole(ice_thickness_cm):
    """Returns P(failure) for a Class 5 POLE at the given ice thickness (cm)."""
    return _lookup("ice_class5_pole", ice_thickness_cm)

def lookup_ice_fragility_class5_wire(ice_thickness_cm):
    """Returns P(failure) for a Class 5 WIRE at the given ice thickness (cm)."""
    return _lookup("ice_class5_wire", ice_thickness_cm)


# ---------------------------------------------------------------------
# Fit everything once, automatically, as soon as this file is imported
# ---------------------------------------------------------------------
_fit_all_curves()


# ---------------------------------------------------------------------
# Demo / sanity check when run directly (not needed for import use)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Fitted curves:", list(_FITTED.keys()))
    print()
    if "wind_class4_pole" in _FITTED:
        for v in [15, 20, 25, 30, 35]:
            print(f"Class 4 pole @ {v} m/s wind: "
                  f"P(fail) = {lookup_wind_fragility_class4_pole(v):.4f}")
    if "ice_class4_pole" in _FITTED:
        for r in [1.0, 1.5, 2.0, 2.5, 3.0]:
            print(f"Class 4 pole @ {r} cm ice: "
                  f"P(fail) = {lookup_ice_fragility_class4_pole(r):.4f}")
