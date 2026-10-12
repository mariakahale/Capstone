"""
build_wire_fig7d_lookup.py
==========================
Builds the span-dependent versions of Ma et al. (2020) Fig. 7(d) (wire fragility
under ice + concurrent wind) and saves them to ONE file that teammates read
through wire_fig7d_lookup.py.

    run once:   python build_wire_fig7d_lookup.py
    writes:     wire_fig7d_surfaces_by_span.npz   (next to this script)

Same pattern as Fig. 4:  figure4_fragility.py -> CSV -> figure4_fragility_lookup.py
Here:                    this file            -> NPZ -> wire_fig7d_lookup.py

What's in the NPZ
  span_bins_m   : span bins (m), 10, 15, ..., 150, plus 91.44 (Ma's span)
  ice_mm        : ice grid, 0-150 mm in 0.5 mm steps (Ma used 0-80; extended because
                  short spans only fail above 80 mm)
  wind_ms       : concurrent 3-s gust grid, 0-40 m/s in 0.5 m/s steps (same as Ma)
  p_slack       : P(fail), shape (n_spans, n_ice, n_wind), "slack" stringing rule
  p_tension     : same, "tension" stringing rule
  w_crit_slack, w_crit_tension : critical resultant load per span bin (N/m)
  site_*        : GPD ice parameters + concurrent wind for Seattle, Grand Marais,
                  Toronto, copied from reconstruct_fig7.py so there is one source of truth

Re-run this file if you change wire_fragility_by_span.py, reconstruct_fig7.py's
SITES, toronto_weather_data2.py or concurrentwindspeed.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # .../figure7_maetall2020/differentspanlengths
FIG7_DIR = HERE.parent                            # .../figure7_maetall2020
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FIG7_DIR))

from wire_fragility_by_span import WireFragility, BASE_SPAN   # noqa: E402
import reconstruct_fig7 as r7                                   # noqa: E402  (sites, grids, Eq. 10)

OUT_PATH = HERE / "wire_fig7d_surfaces_by_span.npz"

# 10, 15, ..., 150 m, plus Ma's own 91.44 m so the paper's span maps to itself
SPAN_BINS = np.unique(np.append(np.arange(10, 150.01, 5.0), BASE_SPAN))   # m
ICE_GRID = np.arange(0, 150.01, 0.5)              # mm
WIND_GRID = r7.V_GRID                             # m/s, 0-40 in 0.5 (Ma's grid)


def build_surfaces(stringing):
    frag = WireFragility(stringing=stringing)
    R, V = np.meshgrid(ICE_GRID, WIND_GRID, indexing="ij")
    P = np.stack([frag.p_fail_ice(float(L), R, V) for L in SPAN_BINS])
    wc = np.array([frag.w_crit(float(L)) for L in SPAN_BINS])
    return frag, P, wc


def cross_check_against_reconstruct_fig7(frag):
    """Compare our 91.44 m wire surface with the Fig. 2-polynomial surface from
    reconstruct_fig7.py (fig7_surfaces.npz). Two independent routes to Fig. 7(d)."""
    candidates = [FIG7_DIR / "fig7_surfaces.npz", Path.cwd() / "fig7_surfaces.npz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("  (skipped: run reconstruct_fig7.py first to create fig7_surfaces.npz)")
        return
    d = np.load(path)
    R, V = np.meshgrid(d["r_mm"], d["v_ms"], indexing="ij")
    ours = frag.p_fail_ice(BASE_SPAN, R, V)
    diff = np.abs(ours - d["wire"])
    print(f"  vs {path.name}: mean |diff| = {diff.mean():.4f}  "
          f"(large values only at the cliff edge, where polynomial and circle differ slightly)")
    for v in (0.0, 26.817):
        f_r7 = r7.slice_at_wind(d["wire"], v)
        f_us = r7.slice_at_wind(ours, v)
        c = lambda f: np.interp(0.5, f, d["r_mm"]) if f.max() >= 0.5 else np.nan
        print(f"  50% crossing at {v:6.3f} m/s: reconstruct_fig7 {c(f_r7):.1f} mm, "
              f"span code {c(f_us):.1f} mm")


def main():
    t0 = time.time()
    frag_s, P_s, wc_s = build_surfaces("slack")
    _, P_t, wc_t = build_surfaces("tension")

    site_arrays = {}
    for name, s in r7.SITES.items():
        key = name.lower().replace(" ", "_")
        site_arrays[f"site_{key}"] = np.array([s["alpha"], s["k"], s["u"], s["v_conc"]], float)

    np.savez_compressed(
        OUT_PATH,
        span_bins_m=SPAN_BINS, ice_mm=ICE_GRID, wind_ms=WIND_GRID,
        p_slack=P_s, p_tension=P_t, w_crit_slack=wc_s, w_crit_tension=wc_t,
        calibrated_EA_N=np.array(frag_s.cond.EA),
        site_names=np.array(list(site_arrays)),
        **site_arrays,
    )
    print(f"Saved {OUT_PATH.name}: {len(SPAN_BINS)} span bins x {len(ICE_GRID)} ice x "
          f"{len(WIND_GRID)} wind  ({time.time() - t0:.1f} s)\n")

    print("Cross-check at 91.44 m (span code vs reconstruct_fig7.py):")
    cross_check_against_reconstruct_fig7(frag_s)

    print("\nAnnual wire failure probability (Eq. 10), slack rule:")
    tor = r7.SITES["Toronto"]
    gm = r7.SITES["Grand Marais"]
    print(f"  {'span':>5} | Toronto (v={tor['v_conc']:.1f} m/s) | Grand Marais (v={gm['v_conc']} m/s)")
    for L in (20, 40, 60, 80, 91.44, 100):
        i = int(np.searchsorted(SPAN_BINS, L))   # round up to bin, same as lookup
        out = []
        for s in (tor, gm):
            F = np.array([np.interp(s["v_conc"], WIND_GRID, row) for row in P_s[i]])
            out.append(_annual_pf(F, s))
        print(f"  {L:5.1f} | {out[0]:22.3e} | {out[1]:.3e}")
    a, k = tor["alpha"], tor["k"]
    if k > 0:
        print(f"\n  Note: Toronto's GPD has k > 0, so ice thickness is capped at alpha/k = "
              f"{a / k:.1f} mm.\n  Any span whose wire survives that much ice has an annual "
              f"P_w of exactly 0.")


def _annual_pf(F, site):
    """Eq. 10 on the extended ice grid, using reconstruct_fig7's GPD density."""
    a, k, u = site["alpha"], site["k"], site["u"]
    r = np.linspace(0, ICE_GRID[-1], 15001)
    Fi = np.interp(r, ICE_GRID, F)
    body = r7._trapz(Fi * r7.gpd_pdf(r, a, k, u), r)
    tail = F[-1] * r7.gpd_exceedance(ICE_GRID[-1], a, k, u)
    return float(body + tail)


if __name__ == "__main__":
    main()
