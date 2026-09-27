"""
CONTINUOUS POLE-AGE FRAGILITY MODEL
====================================
Implements Ma, Dai & Pang (2020) Eq. 1-2 EXACTLY (verified against the
paper's own stated checkpoints: R(60)=0.8111 [18.89% strength loss] and
COV(60)=0.3973 [~39.7%] -- both match the text precisely).

CONCEPT: the paper's decay model is continuous in age t. "New/30/60-yr"
in their Figure 4 were just 3 example ages they chose to PLOT -- not a
limitation of the underlying model. This script lets you evaluate the
fragility curve at ANY age, not just those three.

METHOD: anchor theta/beta in wind-speed units using YOUR digitized
new-pole curve (t=0), then shift/widen that curve continuously using
Ma/Dai/Pang's exact R(t) and Var(t) formulas -- physically motivated by
wind force scaling as speed^2 (Eq. 5 of the paper). Validate against
your digitized 60-year curve as an out-of-sample check.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt

# =================================================================
# 1. MA/DAI/PANG'S EXACT DECAY EQUATIONS (Eq. 1-2) -- fully continuous
# =================================================================

A1, A2 = 0.014418, 0.10683
B1, B2 = 0.00013, 1.846
VAR_T = 0.11
COV_NEW = 0.17  # coefficient of variation for a brand-new pole


def strength_fraction(age_years):
    """
    Eq. 1: fraction of a NEW pole's mean strength remaining at a given
    age. Continuous for any age > 0; the paper states this model only
    applies for age > 10, so ages <=10 are treated as undecayed (=1.0).
    """
    if age_years <= 10:
        return 1.0
    t = age_years
    mean_loss = min(max(A1 * t - A2, 0), 1)
    decay_frac = min(max(B1 * t ** B2, 0), 1)
    return 1 - mean_loss * decay_frac


def strength_cov(age_years, R0=1.0, Var0=COV_NEW ** 2):
    """
    Eq. 2: the paper's reported 'COV' at age t is sqrt(Var(t)) relative
    to the ORIGINAL strength R0 (not the current decayed strength) --
    verified numerically to reproduce the paper's exact stated 39.7% at
    age 60. Continuous for any age.
    """
    if age_years <= 10:
        return COV_NEW
    t = age_years
    b_term = min(max(B1 * t ** B2, 0), 1)
    a_term = min(max(A1 * t - A2, 0), 1)
    Rt = strength_fraction(t)
    term1 = (Var0 + R0 ** 2) * (1 - b_term)
    term2 = (Var0 + R0 ** 2) * (VAR_T + (1 - a_term) ** 2) * b_term
    term3 = Rt ** 2
    var_t = term1 + term2 - term3
    return np.sqrt(var_t) / R0


# =================================================================
# 2. FRAGILITY MODEL -- lognormal, anchored to digitized points
# =================================================================

def lognormal_fragility(v, theta, beta):
    return norm.cdf((np.log(v) - np.log(theta)) / beta)


def fit_fragility(wind_speed, probability, p0=(50, 0.15)):
    (theta, beta), _ = curve_fit(
        lognormal_fragility, wind_speed, probability, p0=p0, maxfev=10000
    )
    return theta, beta


def fragility_at_age(v, age_years, theta_new, beta_new):
    """
    Continuous age-shifted fragility, anchored at theta_new/beta_new
    (fit from YOUR digitized new-pole curve). Wind force ~ v^2 (Eq. 5),
    so a pole retaining strength fraction R(t) fails at a proportionally
    lower speed: theta(t) = theta_new * sqrt(R(t)). Dispersion widens in
    proportion to how much the COV has grown relative to a new pole.
    """
    Rt = strength_fraction(age_years)
    cov_t = strength_cov(age_years)
    theta_t = theta_new * np.sqrt(Rt)
    beta_t = beta_new * (cov_t / COV_NEW)
    return lognormal_fragility(v, theta_t, beta_t), theta_t, beta_t


# =================================================================
# 3. REPLACE WITH YOUR DIGITIZED POINTS
# =================================================================

_DIGITIZED_DATA_DIR = Path(__file__).resolve().parent / "csv_source_files_digitizingfig4"

_new_data = np.loadtxt(_DIGITIZED_DATA_DIR / "red.csv", delimiter=",")
digitized_new = {
    "wind_speed": _new_data[:, 0],
    "probability": _new_data[:, 1],
}

_sixty_year_data = np.loadtxt(_DIGITIZED_DATA_DIR / "blue.csv", delimiter=",")
digitized_60yr = {
    "wind_speed": _sixty_year_data[:, 0],
    "probability": _sixty_year_data[:, 1],
}


# =================================================================
# RUN IT
# =================================================================

if __name__ == "__main__":
    # Anchor at t=0 using your digitized new-pole curve
    theta_new, beta_new = fit_fragility(
        digitized_new["wind_speed"], digitized_new["probability"]
    )
    print(f"Anchored at t=0: theta_new={theta_new:.3f} m/s, beta_new={beta_new:.4f}")

    # Validate: does the CONTINUOUS model at t=60 match your DIGITIZED
    # 60-year curve, even though t=60 was never used to fit anything?
    theta_60_fit_direct, beta_60_fit_direct = fit_fragility(
        digitized_60yr["wind_speed"], digitized_60yr["probability"]
    )
    _, theta_60_model, beta_60_model = fragility_at_age(75, 60, theta_new, beta_new)

    print(f"\nValidation at age 60:")
    print(f"  Directly fit to YOUR digitized 60-yr curve: theta={theta_60_fit_direct:.3f}, beta={beta_60_fit_direct:.4f}")
    print(f"  Predicted by CONTINUOUS model (never saw 60-yr data): theta={theta_60_model:.3f}, beta={beta_60_model:.4f}")
    print(f"  theta error: {abs(theta_60_model - theta_60_fit_direct):.3f} m/s")
    print(f"  beta error:  {abs(beta_60_model - beta_60_fit_direct):.4f}")

    # Plot fragility curves at several CONTINUOUS ages -- not just 0/30/60
    fig, ax = plt.subplots(figsize=(9, 6))
    wind_range = np.linspace(20, 95, 300)

    for age in [0, 10, 20, 30, 40, 50, 60]:
        probs, theta_t, beta_t = fragility_at_age(wind_range, age, theta_new, beta_new)
        ax.plot(wind_range, probs, label=f"age={age} (θ={theta_t:.1f}, β={beta_t:.3f})")

    # Overlay your actual digitized points for visual sanity-check
    ax.scatter(digitized_new["wind_speed"], digitized_new["probability"],
               c="black", marker="x", s=10, label="digitized: new pole", zorder=5)
    ax.scatter(digitized_60yr["wind_speed"], digitized_60yr["probability"],
               c="red", marker="x", s=10, label="digitized: 60-yr pole", zorder=5)

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability of failure")
    ax.set_title("Continuous age-shifted fragility curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("continuous_age_fragility.png", dpi=150)
    print("\nSaved plot to continuous_age_fragility.png")
