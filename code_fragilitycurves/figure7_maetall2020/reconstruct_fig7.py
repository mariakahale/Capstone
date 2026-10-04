"""
reconstruct_fig7.py
Reconstruction of Fig. 7 from Ma, Dai & Pang (2020), J. Struct. Eng. 146(7): 04020134
"Reliability Assessment of Electrical Grids Subjected to Wind Hazards and
 Ice Accretion with Concurrent Wind"

What it does
  1. Builds fragility SURFACES P_f(r, v) for poles (age 0/30/60) and wires
     by Monte Carlo over a grid of radial ice thickness r and concurrent gust v.
  2. Takes a slice of each surface at a site's concurrent wind speed.
  3. Integrates that slice against the site's GPD ice hazard (Eq. 10)
     to get the annual failure probability.

Equation numbers in comments refer to the paper.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent     # save outputs next to this script

# Toronto GPD ice parameters come from the hazard-curve fit in the subfolder
sys.path.insert(0, str(OUT_DIR / "hazardcurves_torontoweatherdata_weibull"))
from toronto_weather_data2 import lookup_toronto_ice_gpd_parameters

TORONTO_ICE_ALPHA, TORONTO_ICE_K = lookup_toronto_ice_gpd_parameters()

# =============================================================================
# 0. CONFIGURATION  <-- edit this block
# =============================================================================
N_SAMPLES = 100_000              # paper: "one hundred thousand" samples per grid point
R_GRID = np.arange(0, 80.01, 0.5)   # mm, paper: 0-80 mm in 0.5 mm steps
V_GRID = np.arange(0, 40.01, 0.5)   # m/s, paper: 0-40 m/s in 0.5 m/s steps
AGES = [0, 30, 60]                  # years, Fig. 7(a)-(c)
SEED = 2020

# Ice hazard sites. alpha, k, u in mm; v_conc in m/s (3-s gust).
# alpha/k follow the convention of Eqs. (6)-(7) (see hazard section below).
SITES = {
    # Validation sites: Table 3 + concurrent winds from the "Ice Hazard
    # Probability Analysis" section. Threshold u = 0 (stated for Grand Marais;
    # assumed for Seattle).
    "Seattle":      dict(alpha=0.9087, k=-0.3733, u=0.0, v_conc=9.0),
    "Grand Marais": dict(alpha=6.6357, k=-0.1037, u=0.0, v_conc=26.817),
    # Toronto: alpha/k from toronto_weather_data2.py (same Eq. 6 convention).
    # >>> v_conc still needs to be filled in <<<
    "Toronto":      dict(alpha=TORONTO_ICE_ALPHA, k=TORONTO_ICE_K, u=0.0, v_conc=None),
}

# Values the paper reports, used for the validation printout
PAPER_WIRE_PF = {"Seattle": 1.6415e-4, "Grand Marais": 0.0059}

# =============================================================================
# 1. CONSTANTS FROM THE PAPER
# =============================================================================
G_ACC = 9.81
RHO_ICE = 917.0          # kg/m^3 (paper prints 0.917 kg/m^3 -- a unit typo)

# --- Wires: "Structural Model of Wires" ---
SPAN = 91.44             # m (300 ft)
N_COND = 3               # "three conductor wires at the same height"
D_COND = 0.01882         # m, 18.82 mm ACSR
R_COND = D_COND / 2
W_SELF = 776.6e-3 * G_ACC    # N/m  (776.6 kg/km)
# Check: wire projected area 3 * 91.44 * 0.01882 = 5.16 m^2, matching the paper.

# Fig. 2 failure envelope: allowable vertical load y (N/m) vs wind load x (N/m),
# per conductor. Coefficients copied from the figure, highest power first.
FIG2_POLY = np.array([-6.118e-8, 1.510e-5, -1.312e-3, 4.141e-2, -5.486e-1, 1.185e2])

# --- Poles: "Structural Uncertainty of Poles" ---
R0 = 52.2e6              # Pa, mean fiber strength, new Southern pine (lognormal)
COV0 = 0.17              # COV of new-pole strength
A1, A2 = 0.014418, 0.10683          # linear strength-loss model
B1, B2 = 0.00013, 1.846             # conditional decay percentage model
VAR_T = 0.11                        # variance of capacity loss
H_TIP_MEAN, H_WIRE_MEAN, H_COV = 11.7, 11.1, 0.03   # m
A_POLE_MEAN = 2.66       # m^2 projected pole area

# ASSUMPTION (not given in the paper): pole diameters.
# Class 4 top circumference per ANSI O5.1 = 21 in. The ground-line diameter
# is backed out so the trapezoid area matches the paper's 2.66 m^2.
# --> Replace with whatever the professor's Fig. 4 code uses.
D_TOP = 21 * 0.0254 / np.pi                         # ~0.170 m
D_GL = 2 * A_POLE_MEAN / H_TIP_MEAN - D_TOP         # ~0.285 m
S_GL = np.pi * D_GL**3 / 32                         # section modulus, m^3
AREA_GL = np.pi * D_GL**2 / 4                       # cross-section area, m^2
INCLUDE_WIRE_GRAVITY_ON_POLE = True   # axial term; effect is tiny either way

# --- Wind load parameters (Eq. 5 + Vickery et al. 2000 randomization) ---
# (mean, COV), all normal
G_POLE, G_WIRE, G_COVv = 0.948, 0.801, 0.11
CF_POLE, CF_COVv = 0.9, 0.12
CF_WIRE_ICED = 1.2       # fixed for iced wires (ASCE 7 value for cables)
KZ_POLE, KZ_WIRE, KZ_COVv = 0.951, 1.024, 0.06
KZT = 1.0


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def normal(rng, mean, cov, n):
    return rng.normal(mean, cov * mean, n)


def lognormal_from_mean_sd(z, mean, sd):
    """Map standard normals z to a lognormal with the given mean and sd."""
    s2 = np.log(1 + (sd / mean) ** 2)
    mu = np.log(mean) - s2 / 2
    return np.exp(mu + np.sqrt(s2) * z)


def pole_strength_stats(age):
    """Eqs. (1) and (2): mean and sd of strength after `age` years of decay."""
    loss = np.clip(A1 * age - A2, 0, 1)        # mean strength loss if decayed
    p_dec = np.clip(B1 * age ** B2, 0, 1)      # probability the pole is decayed
    mean = R0 * (1 - loss * p_dec)                                   # Eq. (1)
    var0 = (COV0 * R0) ** 2
    ER2 = (var0 + R0**2) * (1 - p_dec) \
        + (var0 + R0**2) * (VAR_T + (1 - loss) ** 2) * p_dec        # E[R^2]
    var = ER2 - mean**2                                              # Eq. (2)
    return mean, np.sqrt(var)


def wind_pressure(Kz, v):
    """Eq. (5) without G, Cf, A: 0.613 Kz Kzt v^2 (Pa)."""
    return 0.613 * Kz * KZT * v**2


def ice_weight_per_m(r_mm):
    """Eq. (9) divided by span L: weight of an ice tube of thickness r (N/m)."""
    r = r_mm / 1000
    return ((r + R_COND) ** 2 - R_COND**2) * np.pi * RHO_ICE * G_ACC


def fig2_allowable_vertical(x):
    """Fig. 2 envelope. Beyond the polynomial's root, any vertical load fails."""
    y = np.polyval(FIG2_POLY, x)
    return np.where(x < FIG2_X_ROOT, y, -np.inf)


# Wind load at which the Fig. 2 envelope reaches zero (~118 N/m)
_roots = np.roots(FIG2_POLY)
FIG2_X_ROOT = min(rt.real for rt in _roots
                  if abs(rt.imag) < 1e-9 and 50 < rt.real < 200)


# =============================================================================
# 3. FRAGILITY SURFACES (Monte Carlo)
# =============================================================================
def draw_samples(rng, n):
    """Random structural + load parameters, drawn once and reused at every
    grid point (common random numbers -> smooth surfaces)."""
    return dict(
        z_strength=rng.standard_normal(n),
        h_tip=normal(rng, H_TIP_MEAN, H_COV, n),
        h_wire=normal(rng, H_WIRE_MEAN, H_COV, n),
        G_p=normal(rng, G_POLE, G_COVv, n),
        G_w=normal(rng, G_WIRE, G_COVv, n),
        Cf_p=normal(rng, CF_POLE, CF_COVv, n),
        Kz_p=normal(rng, KZ_POLE, KZ_COVv, n),
        Kz_w=normal(rng, KZ_WIRE, KZ_COVv, n),
    )


def pole_surface(s, age):
    """P_f(r, v) for a pole of given age. Returns array [len(R_GRID), len(V_GRID)]."""
    mean, sd = pole_strength_stats(age)
    strength = lognormal_from_mean_sd(s["z_strength"], mean, sd)   # Pa, (N,)

    # Wind on the pole body: trapezoid of height h_tip.
    # Lever arm = centroid of the trapezoid measured from the ground line.
    A_p = (D_GL + D_TOP) / 2 * s["h_tip"]
    arm_p = s["h_tip"] * (D_GL + 2 * D_TOP) / (3 * (D_GL + D_TOP))
    # Moment per unit v^2 from the pole body
    m_pole = 0.613 * s["Kz_p"] * KZT * s["G_p"] * s["Cf_p"] * A_p * arm_p

    v2 = V_GRID**2
    surf = np.zeros((len(R_GRID), len(V_GRID)))
    for i, r in enumerate(R_GRID):
        # Wind on the iced wires: 3 conductors x one span x iced diameter,
        # applied at wire height. Ice only reaches the pole through this term.
        A_w = N_COND * SPAN * (D_COND + 2 * r / 1000)
        m_wire = 0.613 * s["Kz_w"] * KZT * s["G_w"] * CF_WIRE_ICED * A_w * s["h_wire"]

        # Ground-line stress = bending + compression
        bending = np.outer(m_pole + m_wire, v2) / S_GL                 # (N, nV)
        axial = 0.0
        if INCLUDE_WIRE_GRAVITY_ON_POLE:
            axial = N_COND * SPAN * (W_SELF + ice_weight_per_m(r)) / AREA_GL
        stress = bending + axial

        surf[i] = (stress >= strength[:, None]).mean(axis=0)
    return surf


def wire_surface(s):
    """P_f(r, v) for one conductor, using the Fig. 2 envelope."""
    surf = np.zeros((len(R_GRID), len(V_GRID)))
    for i, r in enumerate(R_GRID):
        d_iced = D_COND + 2 * r / 1000
        # Horizontal load per metre (Eq. 5 with A = iced diameter x 1 m)
        wind = np.outer(s["Kz_w"] * s["G_w"] * CF_WIRE_ICED * d_iced,
                        0.613 * KZT * V_GRID**2)                         # (N, nV)
        vertical = W_SELF + ice_weight_per_m(r)                         # scalar
        surf[i] = (vertical > fig2_allowable_vertical(wind)).mean(axis=0)
    return surf


# =============================================================================
# 4. ICE HAZARD (GPD, Eqs. 6-8) AND ANNUAL FAILURE PROBABILITY (Eq. 10)
# =============================================================================
# Eqs. (6) and (7) use the Hosking sign convention:
#   exceedance  S(r) = [1 - k (r-u)/alpha]^(1/k)                     (Eq. 7)
#   return lvl  r(T) = u + (alpha/k) [1 - T^(-k)]                    (Eq. 6, lambda=1)
# The density consistent with those is f(r) = (1/alpha)[1 - k(r-u)/alpha]^(1/k - 1).
# Eq. (8) as printed uses "1 + k(...)" with exponent -1/k-1, which is the
# opposite convention; it doesn't match Eqs. (6)-(7) or Table 3, so the
# consistent form is used here.
def gpd_exceedance(r, alpha, k, u):
    base = np.clip(1 - k * (r - u) / alpha, 0, None)
    return np.where(r < u, 1.0, base ** (1 / k))


def gpd_pdf(r, alpha, k, u):
    base = np.clip(1 - k * (r - u) / alpha, 1e-300, None)
    return np.where(r < u, 0.0, base ** (1 / k - 1) / alpha)


def gpd_return_level(T, alpha, k, u):
    return u + alpha / k * (1 - T ** (-k))


_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renamed trapz


def slice_at_wind(surf, v_conc):
    """Fragility curve in r at a fixed concurrent wind (interpolated)."""
    return np.array([np.interp(v_conc, V_GRID, row) for row in surf])


def annual_pf(frag_r, site):
    """Eq. (10): Pf = integral of F(r) f(r) dr.
    Above 80 mm the fragility is held at its last value."""
    a, k, u = site["alpha"], site["k"], site["u"]
    r_fine = np.linspace(0, R_GRID[-1], 8001)
    F = np.interp(r_fine, R_GRID, frag_r)
    body = _trapz(F * gpd_pdf(r_fine, a, k, u), r_fine)
    tail = frag_r[-1] * gpd_exceedance(R_GRID[-1], a, k, u)
    return body + tail


# =============================================================================
# 5. PLOTTING
# =============================================================================
def plot_fig7(surfaces, fname="fig7_reconstruction.png"):
    Vg, Rg = np.meshgrid(V_GRID, R_GRID)
    titles = ["(a) new poles", "(b) 30-year poles", "(c) 60-year poles",
              "(d) conductor wires"]
    fig = plt.figure(figsize=(13, 10))
    for j, (surf, ttl) in enumerate(zip(surfaces, titles)):
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        ax.plot_surface(Vg, Rg, surf, cmap="viridis", vmin=0, vmax=1,
                        rstride=2, cstride=2, linewidth=0)
        ax.set_xlabel("Concurrent gust speed (m/s)")
        ax.set_ylabel("Equivalent radial ice thickness (mm)")
        ax.set_zlabel("Probability of failure")
        ax.set_zlim(0, 1)
        ax.view_init(elev=25, azim=-130)   # similar angle to the paper
        ax.set_title(ttl)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_reconstruction.png", dpi=150)
    plt.close(fig)


def plot_slices(surfaces, sites, fname="fig7_slices.png"):
    labels = ["Pole, new", "Pole, 30 yr", "Pole, 60 yr", "Wire"]
    active = {n: s for n, s in sites.items() if s["v_conc"] is not None}
    fig, axes = plt.subplots(1, len(active), figsize=(5 * len(active), 4),
                             squeeze=False)
    for ax, (name, site) in zip(axes[0], active.items()):
        for surf, lab in zip(surfaces, labels):
            ax.plot(R_GRID, slice_at_wind(surf, site["v_conc"]), label=lab)
        ax.set_title(f"{name}: v = {site['v_conc']} m/s")
        ax.set_xlabel("Equivalent radial ice thickness (mm)")
        ax.set_ylabel("Probability of failure")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_slices.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    s = draw_samples(rng, N_SAMPLES)

    print("Pole strength after decay (Eqs. 1-2)")
    for age in AGES:
        m, sd = pole_strength_stats(age)
        print(f"  age {age:>2}: mean {m/1e6:6.2f} MPa, sd/R0 {sd/R0:.3f}")
    print("  paper reports for 60 yr: mean 42.34 MPa, variation 0.397\n")

    print(f"Pole geometry (assumed): D_top {D_TOP:.3f} m, D_groundline {D_GL:.3f} m")
    print(f"Fig. 2 envelope: zero-wind capacity {FIG2_POLY[-1]:.1f} N/m, "
          f"zero-capacity wind {FIG2_X_ROOT:.1f} N/m\n")

    surfaces = [pole_surface(s, age) for age in AGES] + [wire_surface(s)]
    np.savez("fig7_surfaces.npz", r_mm=R_GRID, v_ms=V_GRID,
             pole_0=surfaces[0], pole_30=surfaces[1], pole_60=surfaces[2],
             wire=surfaces[3])
    plot_fig7(surfaces)
    plot_slices(surfaces, SITES)

    # Wire cliff location: where the wire slice crosses 50 %
    print("Wire cliff (50 % crossing) vs paper")
    for v, paper in [(0.0, "~57.5 mm"), (9.0, "~57.5 mm"), (26.817, "41-45 mm")]:
        f = slice_at_wind(surfaces[3], v)
        r50 = np.interp(0.5, f, R_GRID) if f.max() >= 0.5 else np.nan
        print(f"  v = {v:6.3f} m/s: {r50:5.1f} mm   (paper {paper})")
    print()

    print("Annual failure probability, ice + concurrent wind (Eq. 10)")
    for name, site in SITES.items():
        if site["alpha"] is None or site["v_conc"] is None:
            print(f"  {name}: skipped (alpha / k / v_conc not set)")
            continue
        r50 = gpd_return_level(50, site["alpha"], site["k"], site["u"])
        print(f"  {name}  (50-yr ice = {r50:.1f} mm, v_conc = {site['v_conc']} m/s)")
        for age, surf in zip(AGES, surfaces[:3]):
            pf = annual_pf(slice_at_wind(surf, site["v_conc"]), site)
            print(f"    pole, {age:>2} yr: {pf:.4e}")
        pf_w = annual_pf(slice_at_wind(surfaces[3], site["v_conc"]), site)
        ref = f"   (paper {PAPER_WIRE_PF[name]:.4e})" if name in PAPER_WIRE_PF else ""
        print(f"    wire       : {pf_w:.4e}{ref}")
    print("\nSaved fig7_reconstruction.png, fig7_slices.png, fig7_surfaces.npz")


if __name__ == "__main__":
    main()
