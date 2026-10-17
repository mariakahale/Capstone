"""
figure4_fragility_by_span.py
============================
Span-dependent version of figure4_fragility.py (Ma et al. 2020, Fig. 4: pole
fragility under extreme wind, no ice).

WHY SPAN MATTERS FOR A POLE
  A pole carries half of each adjacent span of wire (Ma Fig. 1: "one pole and half
  of all the wires connected"). That tributary length sets two inputs in
  figure4_fragility.py, both fixed there at Ma's 91.44 m:
      Aw ~ N(5.16 m^2, ...)  wind area of the wires on the pole (3 x 0.01882 m x 91.44 m)
      #note: the "3" is the number of wires on the pole, not the number of spans attached
      #0.01882 m = 18.82mm diameter wire, 91.44 m = 300 ft span
      Gi = 7766 * 0.09144    wire gravity load on the pole (per wire)
      #776.6 kg/km
  Here both are scaled by  L_trib / 91.44,  where
      L_trib = (left span + right span) / 2      (end pole: one span / 2).
  Everything else (pole, strength, decay, wind randomization) is identical to
  figure4_fragility.py: same seed, same random draws, same formulas.

RUN ONCE
  python figure4_fragility_by_span.py
  -> writes figure4_fragility_curves_by_span.npz (next to this script)
  -> checks the 91.44 m result against figure4_fragility_curves_5yr.csv

Teammates read the NPZ through figure4_fragility_by_span_lookup.py.
"""

from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "figure4_fragility_curves_by_span.npz"
CSV_91 = OUT_DIR / "figure4_fragility_curves_5yr.csv"

# --- identical settings to figure4_fragility.py ---------------------------------
SEED = 1
N = 500_000
V = np.round(np.arange(20.0, 90.0 + 1e-9, 0.1), 1)
AGES = list(range(0, 65, 5))
BASE_SPAN = 91.44

# Tributary span bins (m): 10, 15, ..., 150 plus Ma's 91.44 m
SPAN_BINS = np.unique(np.append(np.arange(10, 150.01, 5.0), BASE_SPAN))


def aged_pole_strength_kpa(age_years, n, rng):
    """Copied unchanged from figure4_fragility.py."""
    mean_kpa = 52200.0 * (1.0 - (0.014418 * age_years - 0.10683)
                          * (0.00013 * age_years ** 1.846))
    cov = 0.17 + 0.01667 * (age_years / 5.0)
    mu_log = np.log(mean_kpa ** 2 / np.sqrt((cov * mean_kpa) ** 2 + mean_kpa ** 2))
    sigma_log = np.sqrt(np.log(cov ** 2 + 1.0))
    return rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)


def fragility_from_critical_speed(strength_pa, gravity_stress, wind_coeff, wind_speed):
    """Copied unchanged from figure4_fragility.py."""
    required = strength_pa - gravity_stress
    v_crit = np.full(strength_pa.shape, np.inf)
    v_crit[required <= 0] = 0.0
    ok = (required > 0) & (wind_coeff > 0)
    v_crit[ok] = np.sqrt(required[ok] / wind_coeff[ok])
    v_sorted = np.sort(v_crit)
    return np.searchsorted(v_sorted, wind_speed, side="left") / v_crit.size


def pole_wind_fragility(span_trib_m=BASE_SPAN, ages=AGES, n=N, seed=SEED):
    """Fig. 4 curves for a pole with tributary span `span_trib_m`.

    Returns array (len(ages), len(V)). With span_trib_m = 91.44 this reproduces
    figure4_fragility.py exactly (same seed and draw order)."""
    rng = np.random.default_rng(seed)
    scale = span_trib_m / BASE_SPAN

    # Same draws, same order as figure4_fragility.py
    Kzp = rng.normal(0.951, 0.05706, n)
    Kzw = rng.normal(1.024, 0.06144, n)
    Gp = rng.normal(0.948, 0.10428, n)
    Gw = rng.normal(0.801, 0.08811, n)
    Cfp = rng.normal(0.9, 0.108, n)
    Cfw = rng.normal(1.0, 0.12, n)
    Ap = rng.normal(2.66, 0.1596, n)
    Aw = rng.normal(5.16, 0.3098, n) * scale          # <-- span-dependent
    Hp = rng.normal(11.7, 0.351, n)
    Hw = rng.normal(11.1, 0.333, n)

    Kp = 0.613 * Kzp * Gp * Cfp * Ap
    Kw = 0.613 * Kzw * Gw * Cfw * Aw

    D = 0.28
    Gi = 7766 * 0.09144 * scale                       # <-- span-dependent
    gravity_stress = Gi * 3 / ((D / 2) ** 2 * np.pi)
    wind_coeff = 32.0 * (0.5 * Kp * Hp + Kw * Hw) / (np.pi * D ** 3)

    # figure4_fragility.py draws the archived new-pole strength before the aged
    # strengths; draw (and discard) it here too so the random sequence matches.
    rng.lognormal(mean=10.8486, sigma=0.16879, size=n)

    curves = np.empty((len(ages), len(V)))
    for j, age in enumerate(ages):
        strength = aged_pole_strength_kpa(age, n, rng) * 1000.0
        curves[j] = fragility_from_critical_speed(strength, gravity_stress, wind_coeff, V)
    return curves


def check_against_original(curves_91):
    if not CSV_91.exists():
        print(f"  (skipped: {CSV_91.name} not found; run figure4_fragility.py first)")
        return
    import pandas as pd
    df = pd.read_csv(CSV_91)
    worst = max(np.abs(df[f"{a}_yr"].to_numpy() - curves_91[j]).max()
                for j, a in enumerate(AGES))
    print(f"  91.44 m vs {CSV_91.name}: max difference = {worst:.2e}  "
          f"(0 = identical; CSV is rounded to 6 decimals)")


def main():
    P = np.stack([pole_wind_fragility(L) for L in SPAN_BINS])
    np.savez_compressed(NPZ_PATH, span_bins_m=SPAN_BINS, ages=np.array(AGES),
                        wind_ms=V, p_fail=P.astype(np.float32))
    print(f"Saved {NPZ_PATH.name}: {len(SPAN_BINS)} spans x {len(AGES)} ages x {len(V)} speeds")

    print("Check:")
    i91 = int(np.where(SPAN_BINS == BASE_SPAN)[0][0])
    check_against_original(P[i91])

    print("\nWind speed at 50% failure, new pole (m/s):")
    j0 = AGES.index(0)
    for L in (20, 40, 60, 91.44, 120):
        i = int(np.searchsorted(SPAN_BINS, L - 1e-9))
        v50 = np.interp(0.5, P[i, j0], V)
        print(f"  tributary span {SPAN_BINS[i]:6.2f} m: {v50:5.1f} m/s")


if __name__ == "__main__":
    main()
