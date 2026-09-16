"""
HAZARD DISTRIBUTION MODULE
==========================
Step 1 of the capstone pipeline: characterizing wind and ice hazards.

WHY THIS STEP EXISTS:
A fragility curve only answers "if this wind speed occurs, what's the
chance the pole fails?" It says nothing about how LIKELY that wind speed
actually is at a given location. The hazard distribution answers that
second question. You need both -- multiplied together and integrated
across all possible intensities (see Eq. 10 in Ma, Dai & Pang 2020) --
to get the number a utility planner actually cares about: the ANNUAL
probability of failure at a specific location.

This module reproduces the paper's wind (Weibull) and ice (Generalized
Pareto) hazard curves EXACTLY from their published parameters -- this is
pure math, not a black box, so there's nothing to "reconstruct" beyond
plugging numbers into a formula.

HOW TO SWITCH TO ONTARIO DATA:
See the `ONTARIO PLACEHOLDER` section near the bottom. You can either:
  (a) drop in pre-fit alpha/u or alpha/k values directly, or
  (b) supply raw (return_period, value) pairs and let `fit_weibull()` /
      `fit_gpd()` solve for the parameters automatically (same method
      already validated against the paper's own Seattle numbers).
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from toronto_weather_data2 import TORONTO_PARAMETERS


# =================================================================
# WIND HAZARD MODEL (Weibull) -- Eq. 3-4 of Ma, Dai & Pang (2020)
# =================================================================

def weibull_pdf(v, alpha, u):
    """Eq. 3: probability density of extreme wind speed v."""
    v = np.asarray(v, dtype=float)
    return (alpha / u) * (v / u) ** (alpha - 1) * np.exp(-(v / u) ** alpha)


def weibull_speed_for_return_period(T, alpha, u):
    """Eq. 4: wind speed corresponding to a T-year return period."""
    return u * (-np.log(1 / T)) ** (1 / alpha)


def fit_weibull(return_periods, wind_speeds):
    """
    Fits alpha and u from known (return_period, wind_speed) data pairs.
    Use this once you have real Ontario return-period wind data.
    """
    T_data = np.asarray(return_periods, dtype=float)
    v_data = np.asarray(wind_speeds, dtype=float)

    def model(T, alpha, u):
        return weibull_speed_for_return_period(T, alpha, u)

    (alpha_fit, u_fit), _ = curve_fit(model, T_data, v_data, p0=[3.0, 25.0])
    return alpha_fit, u_fit


# =================================================================
# ICE HAZARD MODEL (Generalized Pareto) -- Eq. 6-8 of the paper
# =================================================================

def gpd_pdf(r, alpha, k, u=0):
    """Eq. 8: probability density of extreme equivalent radial ice thickness r."""
    r = np.asarray(r, dtype=float)
    return (1 / alpha) * (1 + k * (r - u) / alpha) ** (-1 / k - 1)


def gpd_thickness_for_return_period(T, alpha, k, u=0, lam=1):
    """Eq. 6: ice thickness corresponding to a T-year return period."""
    return u + (alpha / k) * (1 - (lam * T) ** (-k))


def fit_gpd(return_periods, ice_thicknesses, u=0, lam=1):
    """
    Fits alpha and k from known (return_period, ice_thickness) data pairs.
    Use this once you have real Ontario return-period ice thickness data.
    Same method validated earlier against Seattle's Jones (2002) data.
    """
    T_data = np.asarray(return_periods, dtype=float)
    r_data = np.asarray(ice_thicknesses, dtype=float)

    def model(T, alpha, k):
        return gpd_thickness_for_return_period(T, alpha, k, u=u, lam=lam)

    (alpha_fit, k_fit), _ = curve_fit(model, T_data, r_data, p0=[1.0, -0.3],
                                       maxfev=10000)
    return alpha_fit, k_fit


# =================================================================
# LOCATIONS -- paper's published parameters (exact reproduction)
# =================================================================

LOCATIONS = {
    "Seattle": {
        "wind": {"alpha": 3.369, "u": 23.3742},
        "ice":  {"alpha": 0.9087, "k": -0.3733},
    },
    "Grand Marais": {
        "wind": {"alpha": 3.569, "u": 25.1044},
        "ice":  {"alpha": 6.6357, "k": -0.1037},
    },
    "Toronto": TORONTO_PARAMETERS,
}





# =================================================================
# VALIDATION + DEMO
# =================================================================

if __name__ == "__main__":
    print("=== WIND HAZARD CHECK (should match paper's Fig. 3a) ===")
    for loc in LOCATIONS.keys():
        p = LOCATIONS[loc]["wind"]
        print(f"\n{loc} (alpha={p['alpha']}, u={p['u']}):")
        for T in [10, 25, 50, 100, 300, 700, 1700]:
            v = weibull_speed_for_return_period(T, p["alpha"], p["u"])
            print(f"  T={T:>5} yr -> {v:.2f} m/s")

    print("\n=== ICE HAZARD CHECK (should match paper's Fig. 6a) ===")
    for loc in LOCATIONS.keys():
        p = LOCATIONS[loc]["ice"]
        print(f"\n{loc} (alpha={p['alpha']}, k={p['k']}):")
        for T in [25, 50, 100, 200]:
            r = gpd_thickness_for_return_period(T, p["alpha"], p["k"])
            print(f"  T={T:>5} yr -> {r:.2f} mm")

    # --- Plot both hazard curves, styled like the paper's Fig. 3a/6a ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T_range = np.linspace(2, 1700, 300)
    for loc in LOCATIONS.keys():
        p = LOCATIONS[loc]["wind"]
        v_curve = weibull_speed_for_return_period(T_range, p["alpha"], p["u"])
        axes[0].plot(T_range, v_curve, label=loc)
    axes[0].set_xlabel("Return period (yrs)")
    axes[0].set_ylabel("Extreme wind speed (m/s)")
    axes[0].set_title("Wind hazard (Weibull)")
    axes[0].legend()

    T_range_ice = np.linspace(2, 800, 300)
    for loc in LOCATIONS.keys():
        p = LOCATIONS[loc]["ice"]
        r_curve = gpd_thickness_for_return_period(T_range_ice, p["alpha"], p["k"])
        axes[1].plot(T_range_ice, r_curve, label=loc)
    axes[1].set_xlabel("Return period (yrs)")
    axes[1].set_ylabel("Extreme ice thickness (mm)")
    axes[1].set_title("Ice hazard (Generalized Pareto)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("hazard_curves.png", dpi=150)
    print("\nSaved plot to hazard_curves.png")
