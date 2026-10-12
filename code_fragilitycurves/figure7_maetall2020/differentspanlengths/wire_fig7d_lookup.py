"""
WIRE FRAGILITY LOOKUP -- ICE + CONCURRENT WIND, ANY SPAN (Ma et al. 2020, Fig. 7(d))
===================================================================================
Simple interface for the team: give a span length and the weather, get a
failure probability. No need to understand the cable model to use this file.

USAGE (for teammates):
    from wire_fig7d_lookup import lookup_wire_fragility, lookup_wire_annual_pf

    # "If this storm happens, how likely is this wire to break?"  (conditional)
    p = lookup_wire_fragility(span_m=37.2, ice_mm=45, wind_ms=31.9)

    # Leave wind out to use Toronto's concurrent gust (from concurrentwindspeed.py)
    # 
    p = lookup_wire_fragility(span_m=37.2, ice_mm=45)

    # "How likely is this wire to break in a typical year?"  (Eq. 10, annual)
    pa = lookup_wire_annual_pf(span_m=37.2)                  # Toronto by default
    pa = lookup_wire_annual_pf(span_m=91.44, site="grand_marais")

Public functions:
    lookup_wire_fragility(span_m, ice_mm, wind_ms=None, stringing="slack")
    lookup_wire_curve(span_m, wind_ms=None, stringing="slack") -> (ice_mm, P)
    lookup_wire_annual_pf(span_m, site="toronto", stringing="slack")
    span_bin(span_m)
    available_span_bins(), available_sites()

Where the numbers come from:
    build_wire_fig7d_lookup.py runs wire_fragility_by_span.py for every span bin
    and writes wire_fig7d_surfaces_by_span.npz. This file only READS that file.
    If the model changes, re-run build_wire_fig7d_lookup.py.

How a span is handled:
    Rounded UP to the next bin (37.2 m -> 40 m). Bins are every 5 m from 10 to 150 m,
    plus 91.44 m (Ma's span, so it maps to itself). Shorter spans are stronger,
    so using the next-longer bin is conservative. Spans under 10 m use the 10 m bin.
    Spans over 150 m raise an error (rounding them down would be unconservative).

Between grid points:
    Ice (0.5 mm steps) and wind (0.5 m/s steps) are linearly interpolated.

Stringing rule (how a non-91.44 m span is assumed to be strung):
    "slack"   : same wire-length-to-span ratio as Ma (default, closest to the paper)
    "tension" : same installation tension as Ma's 91.44 m span (more conservative)

Model assumptions and open issues: see NOTE_wire_stiffness_calibration.txt.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR / "wire_fig7d_surfaces_by_span.npz"

_D = None            # loaded data
_INTERP = {}         # (stringing, bin index) -> interpolator


# ---------------------------------------------------------------------
# 1. LOAD THE NPZ WRITTEN BY build_wire_fig7d_lookup.py
# ---------------------------------------------------------------------
def _load():
    global _D
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"[wire_fig7d_lookup] Missing {NPZ_PATH.name}. "
            f"Run build_wire_fig7d_lookup.py first to generate it."
        )
    _D = dict(np.load(NPZ_PATH, allow_pickle=False))


# ---------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------
def available_span_bins():
    return _D["span_bins_m"].tolist()


def available_sites():
    return [n.replace("site_", "") for n in _D["site_names"]]


def _site(site):
    key = f"site_{site.lower().replace(' ', '_')}"
    if key not in _D:
        raise ValueError(f"Unknown site '{site}'. Available: {available_sites()}")
    alpha, k, u, v_conc = _D[key]
    return dict(alpha=alpha, k=k, u=u, v_conc=v_conc)


def span_bin(span_m):
    """Round a span UP to the next bin (conservative)."""
    bins = _D["span_bins_m"]
    if span_m > bins[-1]:
        raise ValueError(f"span {span_m} m is longer than the largest bin ({bins[-1]} m).")
    return float(bins[np.searchsorted(bins, span_m - 1e-9)])   # smallest bin >= span


def _surface_interp(span_m, stringing):
    if stringing not in ("slack", "tension"):
        raise ValueError("stringing must be 'slack' or 'tension'")
    b = span_bin(span_m)
    i = int(np.where(_D["span_bins_m"] == b)[0][0])
    if (stringing, i) not in _INTERP:
        _INTERP[(stringing, i)] = RegularGridInterpolator(
            (_D["ice_mm"], _D["wind_ms"]), _D[f"p_{stringing}"][i],
            bounds_error=False, fill_value=None)
    return _INTERP[(stringing, i)]


def _check_range(ice_mm, wind_ms):
    ice, wind = np.asarray(ice_mm, float), np.asarray(wind_ms, float)
    if np.any(ice < 0) or np.any(wind < 0):
        raise ValueError("ice_mm and wind_ms must be >= 0")
    if np.any(ice > _D["ice_mm"][-1]) or np.any(wind > _D["wind_ms"][-1]):
        warnings.warn(
            f"[wire_fig7d_lookup] input outside the tabulated range "
            f"(ice 0-{_D['ice_mm'][-1]:g} mm, wind 0-{_D['wind_ms'][-1]:g} m/s); "
            f"values are extrapolated and clipped to [0, 1].")
    return ice, wind


def _gpd_pdf(r, alpha, k, u):
    """Same density as reconstruct_fig7.gpd_pdf (Eq. 7-consistent form)."""
    base = np.clip(1 - k * (r - u) / alpha, 1e-300, None)
    return np.where(r < u, 0.0, base ** (1 / k - 1) / alpha)


def _gpd_exceedance(r, alpha, k, u):
    base = np.clip(1 - k * (r - u) / alpha, 0, None)
    return np.where(r < u, 1.0, base ** (1 / k))


_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------
# 3. THE PUBLIC FUNCTIONS -- this is what your team actually calls
# ---------------------------------------------------------------------
def lookup_wire_fragility(span_m, ice_mm, wind_ms=None, stringing="slack"):
    """P(wire breaks | this ice thickness and concurrent wind), for one wire span.

    span_m    : float, span length in m (rounded up to the next bin)
    ice_mm    : float or array, equivalent radial ice thickness, mm
    wind_ms   : float or array, concurrent 3-s gust, m/s. None -> Toronto concurrent gust.
    stringing : "slack" (default) or "tension"
    """
    if wind_ms is None:
        wind_ms = _site("toronto")["v_conc"]
    ice, wind = _check_range(ice_mm, wind_ms)
    ice, wind = np.broadcast_arrays(ice, wind)
    p = _surface_interp(span_m, stringing)(np.stack([ice.ravel(), wind.ravel()], -1))
    p = np.clip(p, 0, 1).reshape(ice.shape)
    return float(p) if p.ndim == 0 else p


def lookup_wire_curve(span_m, wind_ms=None, stringing="slack"):
    """The full fragility curve P(fail) vs ice at one wind speed: a slice of Fig. 7(d).
    Returns (ice_mm array, P array)."""
    ice = _D["ice_mm"]
    return ice, lookup_wire_fragility(span_m, ice, wind_ms, stringing)


def lookup_wire_annual_pf(span_m, site="toronto", stringing="slack"):
    """Annual probability that this wire span breaks from ice + concurrent wind (Eq. 10):
        P_w = integral of F(r) f(r) dr
    F(r) = fragility curve at the site's concurrent wind; f(r) = site's GPD ice density."""
    s = _site(site)
    ice, F = lookup_wire_curve(span_m, s["v_conc"], stringing)
    r = np.linspace(0, ice[-1], 15001)
    body = _trapz(np.interp(r, ice, F) * _gpd_pdf(r, s["alpha"], s["k"], s["u"]), r)
    tail = F[-1] * _gpd_exceedance(ice[-1], s["alpha"], s["k"], s["u"])
    return float(body + tail)


# ---------------------------------------------------------------------
# Load once, automatically, when this file is imported
# ---------------------------------------------------------------------
_load()


if __name__ == "__main__":
    print("Span bins (m):", available_span_bins())
    print("Sites:", available_sites())
    v = _site("toronto")["v_conc"]
    print(f"\nToronto concurrent gust: {v:.2f} m/s\n")
    for L in (37.2, 60, 91.44):
        print(f"Span {L} m (bin {span_bin(L):g} m):")
        for ice in (30, 45, 55, 70):
            print(f"   P(fail | {ice} mm, {v:.1f} m/s) = {lookup_wire_fragility(L, ice):.4f}")
        print(f"   annual P_w Toronto      = {lookup_wire_annual_pf(L):.3e}")
        print(f"   annual P_w Grand Marais = {lookup_wire_annual_pf(L, 'grand_marais'):.3e}")
