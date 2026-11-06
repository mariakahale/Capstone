"""
build_pole_fig7_lookup.py
=========================
Span-dependent versions of Ma et al. (2020) Fig. 7(a)-(c): POLE fragility under
ice + concurrent wind. Saves them to one file that teammates read through
pole_fig7_lookup.py.

    run once:   python build_pole_fig7_lookup.py
    writes:     pole_fig7_surfaces_by_span.npz   (next to this script)
                p_fail_u16[phase, span, age, ice, wind]   phases = 1, 2, 3
                stored as uint16: P(fail) = p_fail_u16 / p_fail_scale (scale 65535).
                Rounding error <= 7.6e-6, far below the Monte Carlo noise; keeps the
                file ~23 MB instead of ~41 MB as float32.

WHY SPAN MATTERS FOR A POLE
  Ma ignores ice on the pole itself ("not significant according to ASCE 74"), but
  the pole carries the wind on the ICED wires and their weight, for half of each
  adjacent span. In reconstruct_fig7.pole_surface() that is
      A_w   = N_COND * SPAN * (D_COND + 2r)          wind area of iced wires
      axial = N_COND * SPAN * (W_SELF + ice) / AREA_GL
  with SPAN = 91.44 m. Here SPAN is replaced by the pole's tributary span
      L_trib = (left span + right span) / 2      (end pole: one span / 2).
  Everything else is reconstruct_fig7.py's model: same constants, same random
  samples (draw_samples, SEED), same strength model (Eqs. 1-2).

NUMBER OF PHASES (added Nov 2026)
  N_COND = 3 in Ma is three phase conductors. Here N_COND is replaced by the
  number of phases on the line (1, 2 or 3), so a 1-phase pole carries 1/3 of the
  iced-wire wind and weight. Neutral neglected, as in Ma; conductor size unchanged.
  n_phases = 3 reproduces the previous file exactly.

SPEED
  reconstruct_fig7.pole_surface() checks every sample at every wind speed. This
  file uses the equivalent "critical wind speed" form (like figure4_fragility.py):
  for each sample, the gust at which ground-line stress reaches strength, then
  P(fail | v) = fraction of samples with critical speed <= v. Same answer,
  ~50x faster. main() checks it against pole_surface() at 91.44 m.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent             # .../figure7_maetall2020/differentspanlengths
FIG7_DIR = HERE.parent
sys.path.insert(0, str(FIG7_DIR))
import reconstruct_fig7 as r7                      # noqa: E402

OUT_PATH = HERE / "pole_fig7_surfaces_by_span.npz"
P_SCALE = 65535                                    # uint16 storage, see docstring

BASE_SPAN = r7.SPAN                                # 91.44 m
SPAN_BINS = np.unique(np.append(np.arange(10, 150.01, 5.0), BASE_SPAN))
AGES = np.arange(0, 61, 5)                         # 0, 5, ..., 60 yr (same as Fig. 4)
ICE_GRID = np.arange(0, 150.01, 0.5)               # mm (Ma: 0-80; same 0-150 as the wire lookup)
WIND_GRID = r7.V_GRID                              # m/s, 0-40 in 0.5 (Ma's grid)
PHASES = (1, 2, 3)                                 # phase conductors on the pole (Ma = 3)
BASE_PHASES = r7.N_COND


def pole_surface_span(s, age, span, n_phases=BASE_PHASES, r_grid=ICE_GRID, v_grid=WIND_GRID):
    """Same model as reconstruct_fig7.pole_surface(), with SPAN -> `span` and
    N_COND -> `n_phases`. Returns P_f, shape (len(r_grid), len(v_grid))."""
    mean, sd = r7.pole_strength_stats(age)
    strength = r7.lognormal_from_mean_sd(s["z_strength"], mean, sd)

    A_p = (r7.D_GL + r7.D_TOP) / 2 * s["h_tip"]
    arm_p = s["h_tip"] * (r7.D_GL + 2 * r7.D_TOP) / (3 * (r7.D_GL + r7.D_TOP))
    m_pole = 0.613 * s["Kz_p"] * r7.KZT * s["G_p"] * s["Cf_p"] * A_p * arm_p

    n = strength.size
    surf = np.empty((len(r_grid), len(v_grid)))
    for i, r in enumerate(r_grid):
        A_w = n_phases * span * (r7.D_COND + 2 * r / 1000)
        m_wire = 0.613 * s["Kz_w"] * r7.KZT * s["G_w"] * r7.CF_WIRE_ICED * A_w * s["h_wire"]
        c = (m_pole + m_wire) / r7.S_GL                     # bending stress per v^2
        axial = 0.0
        if r7.INCLUDE_WIRE_GRAVITY_ON_POLE:
            axial = n_phases * span * (r7.W_SELF + r7.ice_weight_per_m(r)) / r7.AREA_GL

        # fail when c*v^2 + axial >= strength  ->  v >= v_crit
        need = strength - axial
        v_crit = np.full(n, np.inf)
        v_crit[need <= 0] = 0.0
        ok = (need > 0) & (c > 0)
        v_crit[ok] = np.sqrt(need[ok] / c[ok])
        surf[i] = np.searchsorted(np.sort(v_crit), v_grid, side="right") / n
    return surf


def check_against_reconstruct_fig7(s):
    print("Check vs reconstruct_fig7.pole_surface() at 91.44 m (slow, ~1 min per case):")
    for age, ph in ((0, 3), (60, 3), (30, 1)):
        ref = r7.pole_surface(s, age, n_cond=ph)
        new = pole_surface_span(s, age, BASE_SPAN, ph, r_grid=r7.R_GRID, v_grid=r7.V_GRID)
        print(f"  age {age:>2}, {ph}-phase: max difference = {np.abs(ref - new).max():.2e}")


def main(check=True):
    rng = np.random.default_rng(r7.SEED)
    s = r7.draw_samples(rng, r7.N_SAMPLES)
    if check:
        check_against_reconstruct_fig7(s)

    t0 = time.time()
    P = np.empty((len(PHASES), len(SPAN_BINS), len(AGES), len(ICE_GRID), len(WIND_GRID)),
                 np.float32)
    for k, ph in enumerate(PHASES):
        for i, L in enumerate(SPAN_BINS):
            for j, age in enumerate(AGES):
                P[k, i, j] = pole_surface_span(s, age, float(L), ph)
        print(f"  {ph}-phase done ({time.time() - t0:.0f} s)", flush=True)
    print(f"\nBuilt {len(PHASES)} phases x {len(SPAN_BINS)} spans x {len(AGES)} ages "
          f"({time.time() - t0:.0f} s)")

    site_arrays = {f"site_{n.lower().replace(' ', '_')}":
                   np.array([v["alpha"], v["k"], v["u"], v["v_conc"]], float)
                   for n, v in r7.SITES.items()}
    np.savez_compressed(OUT_PATH, phases=np.array(PHASES), span_bins_m=SPAN_BINS,
                        ages=AGES, ice_mm=ICE_GRID,
                        wind_ms=WIND_GRID,
                        p_fail_u16=np.rint(P.astype(np.float64) * P_SCALE).astype(np.uint16),
                        p_fail_scale=np.array(P_SCALE),
                        site_names=np.array(list(site_arrays)), **site_arrays)
    print(f"Saved {OUT_PATH.name}")

    print("\nAnnual pole failure, ice + concurrent wind (Eq. 10):")
    tor, gm = r7.SITES["Toronto"], r7.SITES["Grand Marais"]
    print(f"  {'phases':>6} | {'trib. span':>10} | {'age':>3} | "
          f"Toronto ({tor['v_conc']:.1f} m/s) | Grand Marais")
    for k, ph in enumerate(PHASES):
        for L in (20, 40, 60, 91.44):
            i = int(np.searchsorted(SPAN_BINS, L - 1e-9))
            for j, age in enumerate(AGES):
                if age not in (0, 30, 60):
                    continue
                out = [annual_pf(P[k, i, j], site) for site in (tor, gm)]
                print(f"  {ph:6d} | {SPAN_BINS[i]:10.2f} | {age:3d} | {out[0]:20.3e} | {out[1]:.3e}")


def annual_pf(surf, site):
    """Eq. 10 with reconstruct_fig7's GPD density; slice at the site's concurrent wind."""
    F = np.array([np.interp(site["v_conc"], WIND_GRID, row) for row in surf])
    r = np.linspace(0, ICE_GRID[-1], 10001)
    a, k, u = site["alpha"], site["k"], site["u"]
    body = r7._trapz(np.interp(r, ICE_GRID, F) * r7.gpd_pdf(r, a, k, u), r)
    tail = F[-1] * r7.gpd_exceedance(ICE_GRID[-1], a, k, u)
    return float(body + tail)


if __name__ == "__main__":
    main()
