"""
POLE WIND FRAGILITY LOOKUP, ANY SPAN (Ma et al. 2020, Fig. 4)
=============================================================
Simple interface for the team: give the pole's two adjacent span lengths, its
age and the wind speed, get a failure probability.

USAGE (for teammates):
    from figure4_fragility_by_span_lookup import lookup_wind_fragility_pole_span

    # pole between a 40 m span and a 55 m span, 30 years old, 50 m/s gust
    p = lookup_wind_fragility_pole_span(wind_speed=50, age_years=30,
                                        span_left_m=40, span_right_m=55)

    # end-of-line pole (only one span attached): leave span_right_m out
    p = lookup_wind_fragility_pole_span(50, 30, span_left_m=40)

Public functions:
    lookup_wind_fragility_pole_span(wind_speed, age_years, span_left_m, span_right_m=None)
    tributary_span(span_left_m, span_right_m=None)
    span_bin(tributary_m), age_bin(age_years)
    available_span_bins(), available_ages()

How the span is handled:
    A pole carries half of each adjacent span (Ma Fig. 1), so
        tributary span = (left + right) / 2     (end pole: left / 2)
    which is rounded UP to the next bin (10, 15, ..., 150 m, plus 91.44 m).
    Rounding up is conservative: a longer tributary span means more wind load.

Wind speed is a 3-SECOND GUST (m/s), same as Ma et al. and figure4_fragility_lookup.py.
Ages: tabulated every 5 years (0, 5, ..., 60), the same ages as the Fig. 7 pole lookup.
An age is rounded UP to the next 5-yr bin (32 yr -> 35 yr); no blending.
Rounding up is conservative because older poles are weaker.
(Note: the original figure4_fragility_lookup.py still blends ages; this file does not.)

Where the numbers come from:
    figure4_fragility_by_span.py -> figure4_fragility_curves_by_span.npz.
    This file only READS that file. At 91.44 m it is identical to
    figure4_fragility_curves_5yr.csv.
"""

import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR / "figure4_fragility_curves_by_span.npz"

_SPANS = _AGES = _WIND = _P = None


def _load():
    global _SPANS, _AGES, _WIND, _P
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"[figure4_fragility_by_span_lookup] Missing {NPZ_PATH.name}. "
            f"Run figure4_fragility_by_span.py first to generate it.")
    d = np.load(NPZ_PATH)
    _SPANS, _AGES, _WIND, _P = d["span_bins_m"], d["ages"], d["wind_ms"], d["p_fail"]


def available_span_bins():
    return _SPANS.tolist()


def available_ages():
    return _AGES.tolist()


def tributary_span(span_left_m, span_right_m=None):
    """Half of each adjacent span. End pole (one span): half of that span."""
    if span_right_m is None:
        return span_left_m / 2
    return (span_left_m + span_right_m) / 2


def span_bin(tributary_m):
    """Smallest bin >= tributary span (conservative). Below 10 m -> 10 m bin."""
    if tributary_m > _SPANS[-1]:
        raise ValueError(f"tributary span {tributary_m} m exceeds the largest bin ({_SPANS[-1]} m).")
    return float(_SPANS[np.searchsorted(_SPANS, tributary_m - 1e-9)])


def age_bin(age_years):
    """Round age UP to the next tabulated age (conservative: older poles are weaker).
    No blending between ages -- same binning idea as the span bins."""
    ages = _AGES
    if not (0 <= age_years <= ages[-1]):
        raise ValueError(f"age_years={age_years} is outside {ages[0]}-{ages[-1]} yr.")
    return int(ages[np.searchsorted(ages, age_years - 1e-9)])

#***#
def lookup_wind_fragility_pole_span(wind_speed, age_years, span_left_m, span_right_m=None):
    """P(failure) for a wood pole under extreme wind (no ice).

    wind_speed   : float or array, 3-s gust, m/s (tabulated 20-90 m/s)
    age_years    : float, 0-60, rounded up to the next 5-yr bin
    span_left_m  : span on one side of the pole, m
    span_right_m : span on the other side, m (None for an end pole)
    """
    i = int(np.where(_SPANS == span_bin(tributary_span(span_left_m, span_right_m)))[0][0])

    j = int(np.where(_AGES == age_bin(age_years))[0][0])
    curve = _P[i, j]

    ws = np.asarray(wind_speed, dtype=float)
    if np.any((ws < _WIND[0]) | (ws > _WIND[-1])):
        warnings.warn(f"[figure4_fragility_by_span_lookup] wind speed outside "
                      f"{_WIND[0]:g}-{_WIND[-1]:g} m/s; clamped to the end value.")
    p = np.interp(ws, _WIND, curve)
    return float(p) if p.ndim == 0 else p


_load()

if __name__ == "__main__":
    for left, right in [(40, 40), (40, 55), (91.44, 91.44), (40, None)]:
        L = tributary_span(left, right)
        print(f"spans {left}/{right}: tributary {L:.1f} m -> bin {span_bin(L):g} m")
        for v in (40, 50, 60):
            print(f"   32-yr pole (bin {age_bin(32)}) @ {v} m/s: P(fail) = "
                  f"{lookup_wind_fragility_pole_span(v, 32, left, right):.4f}")
