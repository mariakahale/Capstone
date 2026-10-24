"""
POLE FRAGILITY LOOKUP -- ICE + CONCURRENT WIND, ANY SPAN (Ma et al. 2020, Fig. 7(a)-(c))
======================================================================================
Simple interface for the team: give the pole's two adjacent spans, its age and the
weather, get a failure probability.

USAGE (for teammates):
    from pole_fig7_lookup import lookup_pole_ice_fragility, lookup_pole_ice_annual_pf

    # "If this ice storm happens, how likely is this pole to fail?"  (conditional)
    p = lookup_pole_ice_fragility(age_years=30, span_left_m=40, span_right_m=55,
                                  ice_mm=20, wind_ms=31.9)
    # leave wind out to use Toronto's concurrent gust (concurrentwindspeed.py)
    p = lookup_pole_ice_fragility(30, 40, 55, ice_mm=20)

    # "How likely is it to fail from ice storms in a typical year?"  (Eq. 10)
    pa = lookup_pole_ice_annual_pf(age_years=30, span_left_m=40, span_right_m=55)

    # end-of-line pole: leave span_right_m out

Public functions:
    lookup_pole_ice_fragility(age_years, span_left_m, span_right_m=None, ice_mm=0, wind_ms=None)
    lookup_pole_ice_annual_pf(age_years, span_left_m, span_right_m=None, site="toronto")
    tributary_span(span_left_m, span_right_m=None), span_bin(tributary_m), age_bin(age_years)
    available_span_bins(), available_ages(), available_sites()

How the span is handled:
    tributary span = (left + right) / 2  (end pole: left / 2), rounded UP to the next
    bin (10, 15, ..., 150 m, plus 91.44 m). Conservative: longer = more wire wind load.

Ages:
    Tabulated every 5 years (0, 5, ..., 60), the same ages as the Fig. 4 lookup.
    An age is rounded UP to the next 5-yr bin (32 yr -> 35 yr); no blending.
    Rounding up is conservative because older poles are weaker.

Wind is a 3-SECOND GUST (m/s). Ice is equivalent radial ice thickness on the
WIRES (mm); ice on the pole itself is ignored, as in Ma et al.

Where the numbers come from:
    build_pole_fig7_lookup.py (reconstruct_fig7.py's pole model with SPAN -> tributary
    span) -> pole_fig7_surfaces_by_span.npz. This file only READS that file.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR / "pole_fig7_surfaces_by_span.npz"

_D = None
_INTERP = {}


def _load():
    global _D
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"[pole_fig7_lookup] Missing {NPZ_PATH.name}. "
                                f"Run build_pole_fig7_lookup.py first to generate it.")
    _D = dict(np.load(NPZ_PATH, allow_pickle=False))


def available_span_bins():
    return _D["span_bins_m"].tolist()


def available_ages():
    return _D["ages"].tolist()


def available_sites():
    return [n.replace("site_", "") for n in _D["site_names"]]


def _site(site):
    key = f"site_{site.lower().replace(' ', '_')}"
    if key not in _D:
        raise ValueError(f"Unknown site '{site}'. Available: {available_sites()}")
    a, k, u, v = _D[key]
    return dict(alpha=a, k=k, u=u, v_conc=v)


def tributary_span(span_left_m, span_right_m=None):
    """Half of each adjacent span. End pole (one span): half of that span."""
    return span_left_m / 2 if span_right_m is None else (span_left_m + span_right_m) / 2


def span_bin(tributary_m):
    bins = _D["span_bins_m"]
    if tributary_m > bins[-1]:
        raise ValueError(f"tributary span {tributary_m} m exceeds the largest bin ({bins[-1]} m).")
    return float(bins[np.searchsorted(bins, tributary_m - 1e-9)])


def age_bin(age_years):
    """Round age UP to the next tabulated age (conservative: older poles are weaker).
    No blending between ages -- same binning idea as the span bins."""
    ages = _D["ages"]
    if not (0 <= age_years <= ages[-1]):
        raise ValueError(f"age_years={age_years} is outside {ages[0]}-{ages[-1]} yr.")
    return int(ages[np.searchsorted(ages, age_years - 1e-9)])


def _interp(i_span, j_age):
    key = (i_span, j_age)
    if key not in _INTERP:
        _INTERP[key] = RegularGridInterpolator(
            (_D["ice_mm"], _D["wind_ms"]), _D["p_fail"][i_span, j_age].astype(float),
            bounds_error=False, fill_value=None)
    return _INTERP[key]


def lookup_pole_ice_fragility(age_years, span_left_m, span_right_m=None,
                              ice_mm=0.0, wind_ms=None):
    """P(pole fails | ice thickness on the wires, concurrent gust).

    age_years    : 0-60, rounded up to the next 5-yr bin
    span_left_m  : span on one side (m);  span_right_m : other side (None = end pole)
    ice_mm       : float or array, equivalent radial ice thickness on the wires (mm)
    wind_ms      : float or array, concurrent 3-s gust (m/s). None -> Toronto.
    """
    if wind_ms is None:
        wind_ms = _site("toronto")["v_conc"]
    bins, ages = _D["span_bins_m"], _D["ages"]
    i = int(np.where(bins == span_bin(tributary_span(span_left_m, span_right_m)))[0][0])
    j = int(np.where(ages == age_bin(age_years))[0][0])

    ice, wind = np.broadcast_arrays(np.asarray(ice_mm, float), np.asarray(wind_ms, float))
    if np.any(ice > _D["ice_mm"][-1]) or np.any(wind > _D["wind_ms"][-1]):
        warnings.warn("[pole_fig7_lookup] input outside the tabulated range "
                      f"(ice 0-{_D['ice_mm'][-1]:g} mm, wind 0-{_D['wind_ms'][-1]:g} m/s); "
                      "values are extrapolated and clipped to [0, 1].")
    pts = np.stack([ice.ravel(), wind.ravel()], -1)

    p = np.clip(_interp(i, j)(pts), 0, 1).reshape(ice.shape)
    return float(p) if p.ndim == 0 else p


def lookup_pole_ice_annual_pf(age_years, span_left_m, span_right_m=None, site="toronto"):
    """Annual probability the pole fails from ice + concurrent wind (Eq. 10):
    integral of F(r) f(r) dr, F = fragility at the site's concurrent gust,
    f = site's GPD ice density (same form as reconstruct_fig7.py)."""
    s = _site(site)
    ice = _D["ice_mm"]
    F = lookup_pole_ice_fragility(age_years, span_left_m, span_right_m, ice, s["v_conc"])
    r = np.linspace(0, ice[-1], 10001)
    base = np.clip(1 - s["k"] * (r - s["u"]) / s["alpha"], 1e-300, None)
    f = np.where(r < s["u"], 0.0, base ** (1 / s["k"] - 1) / s["alpha"])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    body = trapz(np.interp(r, ice, F) * f, r)
    end = np.clip(1 - s["k"] * (ice[-1] - s["u"]) / s["alpha"], 0, None) ** (1 / s["k"])
    return float(body + F[-1] * end)


_load()

if __name__ == "__main__":
    v = _site("toronto")["v_conc"]
    print(f"Toronto concurrent gust: {v:.2f} m/s\n")
    for left, right in [(91.44, 91.44), (40, 55), (40, None)]:
        L = tributary_span(left, right)
        print(f"spans {left}/{right}: tributary {L:.1f} m -> bin {span_bin(L):g} m")
        for age in (0, 32, 60):
            p20 = lookup_pole_ice_fragility(age, left, right, ice_mm=20)
            pa = lookup_pole_ice_annual_pf(age, left, right)
            print(f"   {age:>2}-yr pole (bin {age_bin(age)}): P(fail | 20 mm, {v:.1f} m/s) = {p20:.4f}   "
                  f"annual (Toronto) = {pa:.3e}")
