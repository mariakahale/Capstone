"""
CONTINUOUS POLE-AGE FRAGILITY MODEL -- v2 (calibrated)
========================================================
Fixes the systematic bias seen in v2's plot (generated curves sagging
below the digitized curve for P > ~0.75, worst at age 60).

WHAT CHANGED VS. THE ORIGINAL SCRIPT
-------------------------------------
The original script assumed two physics-motivated but UNVERIFIED rules:
    theta(t) = theta_new * sqrt(R(t))            (pure wind^2-force scaling)
    beta(t)  = beta_new  * (COV(t) / COV_NEW)     (pure ratio scaling)

Checked against the actual digitized 30- and 60-year curves, both rules
were biased: theta(t) didn't drop far enough (mean-vs-median effect of
growing variance was ignored), and beta(t) grew too fast (it ignored a
fixed "demand-side" noise floor that shouldn't scale with pole age).

This version calibrates BOTH relationships against your three known
anchor points (age 0, 30, 60) instead of assuming a fixed physical
exponent:

    theta(t) = theta_new * R(t) ** p              <- p fit from data
    beta(t)  = sqrt( (k * COV(t))**2 + beta_demand**2 )   <- k, beta_demand fit from data

Because R(t) and COV(t) are themselves strongly nonlinear (convex) in
age, this naturally produces the "small jumps early, big jumps late"
behavior -- no manual age-spacing schedule needed.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

# =================================================================
# 1. MA/DAI/PANG'S EXACT DECAY EQUATIONS (Eq. 1-2) -- unchanged
# =================================================================

A1, A2 = 0.014418, 0.10683
B1, B2 = 0.00013, 1.846
VAR_T = 0.11
COV_NEW = 0.17


def strength_fraction(age_years):
    """Eq. 1: fraction of new-pole mean strength remaining at age t."""
    if age_years <= 10:
        return 1.0
    t = age_years
    mean_loss = min(max(A1 * t - A2, 0), 1)
    decay_frac = min(max(B1 * t ** B2, 0), 1)
    return 1 - mean_loss * decay_frac


def strength_cov(age_years, R0=1.0, Var0=COV_NEW ** 2):
    """Eq. 2: strength COV at age t, relative to the ORIGINAL strength."""
    if age_years <= 10:
        return COV_NEW
    t = age_years
    b_term = min(max(B1 * t ** B2, 0), 1)
    a_term = min(max(A1 * t - A2, 0), 1)
    Rt = strength_fraction(t)
    term1 = (Var0 + R0 ** 2) * (1 - b_term)
    term2 = (Var0 + R0 ** 2) * (VAR_T + (1 - a_term) ** 2) * b_term
    term3 = Rt ** 2
    return np.sqrt(term1 + term2 - term3) / R0


# =================================================================
# 2. FRAGILITY MODEL
# =================================================================

def lognormal_fragility(v, theta, beta):
    return norm.cdf((np.log(v) - np.log(theta)) / beta)


def fit_fragility(wind_speed, probability, p0=(50, 0.15)):
    (theta, beta), _ = curve_fit(
        lognormal_fragility, wind_speed, probability, p0=p0, maxfev=10000
    )
    return theta, beta


def calibrate_theta_exponent(theta_new, theta_30, theta_60):
    """Solve theta(t) = theta_new * R(t)^p using both known aged anchors,
    then average the two independent estimates."""
    R30, R60 = strength_fraction(30), strength_fraction(60)
    p30 = np.log(theta_30 / theta_new) / np.log(R30)
    p60 = np.log(theta_60 / theta_new) / np.log(R60)
    return 0.5 * (p30 + p60), p30, p60


def calibrate_beta_params(beta_new, beta_60):
    """Solve beta(t)^2 = (k*COV(t))^2 + beta_demand^2 using the age-0 and
    age-60 anchors (age-30 is left free as an out-of-sample check)."""
    cov60 = strength_cov(60)

    def eqs(vars):
        k, bd = vars
        e1 = (k * COV_NEW) ** 2 + bd ** 2 - beta_new ** 2
        e2 = (k * cov60) ** 2 + bd ** 2 - beta_60 ** 2
        return [e1, e2]

    k, beta_demand = fsolve(eqs, [0.5, 0.07])
    return k, beta_demand


def fragility_at_age(v, age_years, theta_new, p, k, beta_demand):
    Rt = strength_fraction(age_years)
    cov_t = COV_NEW if age_years == 0 else strength_cov(age_years)
    theta_t = theta_new * Rt ** p
    beta_t = np.sqrt((k * cov_t) ** 2 + beta_demand ** 2)
    return lognormal_fragility(v, theta_t, beta_t), theta_t, beta_t


# =================================================================
# 3. LOAD DIGITIZED CURVES
# =================================================================

_DIGITIZED_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "csv_source_files_digitizingfig4"
)

_new_data = np.loadtxt(_DIGITIZED_DATA_DIR / "red.csv", delimiter=",")
digitized_new = {"wind_speed": _new_data[:, 0], "probability": _new_data[:, 1]}

_thirty_year_data = np.loadtxt(_DIGITIZED_DATA_DIR / "black.csv", delimiter=",")
digitized_30yr = {"wind_speed": _thirty_year_data[:, 0], "probability": _thirty_year_data[:, 1]}

_sixty_year_data = np.loadtxt(_DIGITIZED_DATA_DIR / "blue.csv", delimiter=",")
digitized_60yr = {"wind_speed": _sixty_year_data[:, 0], "probability": _sixty_year_data[:, 1]}


# =================================================================
# RUN IT
# =================================================================

if __name__ == "__main__":
    theta_new, beta_new = fit_fragility(digitized_new["wind_speed"], digitized_new["probability"])
    theta_30, beta_30 = fit_fragility(digitized_30yr["wind_speed"], digitized_30yr["probability"])
    theta_60, beta_60 = fit_fragility(digitized_60yr["wind_speed"], digitized_60yr["probability"])

    print(f"Direct fits -> new: theta={theta_new:.3f} beta={beta_new:.4f}")
    print(f"               30yr: theta={theta_30:.3f} beta={beta_30:.4f}")
    print(f"               60yr: theta={theta_60:.3f} beta={beta_60:.4f}")

    p, p30, p60 = calibrate_theta_exponent(theta_new, theta_30, theta_60)
    k, beta_demand = calibrate_beta_params(beta_new, beta_60)

    print(f"\nCalibrated theta exponent p = {p:.4f}  (implied by 30yr: {p30:.4f}, by 60yr: {p60:.4f})")
    print(f"Calibrated beta params: k = {k:.4f}, beta_demand = {beta_demand:.4f}")

    print("\nOut-of-sample check at age 30 (not used to fit beta):")
    _, theta_30_pred, beta_30_pred = fragility_at_age(50, 30, theta_new, p, k, beta_demand)
    print(f"  predicted theta={theta_30_pred:.3f} (actual {theta_30:.3f}), "
          f"beta={beta_30_pred:.4f} (actual {beta_30:.4f})")

    # Plot continuous curves at several ages
    fig, ax = plt.subplots(figsize=(9, 6))
    wind_range = np.linspace(20, 95, 300)

    for age in [0, 10, 20, 30, 40, 50, 60]:
        probs, theta_t, beta_t = fragility_at_age(wind_range, age, theta_new, p, k, beta_demand)
        ax.plot(wind_range, probs, label=f"age={age} (θ={theta_t:.1f}, β={beta_t:.3f})")

    ax.scatter(digitized_new["wind_speed"], digitized_new["probability"],
               c="black", marker="x", s=10, label="digitized: new pole", zorder=5)
    ax.scatter(digitized_30yr["wind_speed"], digitized_30yr["probability"],
               c="green", marker="x", s=10, label="digitized: 30-yr pole", zorder=5)
    ax.scatter(digitized_60yr["wind_speed"], digitized_60yr["probability"],
               c="red", marker="x", s=10, label="digitized: 60-yr pole", zorder=5)

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability of failure")
    ax.set_title("Continuous age-shifted fragility curves (calibrated)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("continuous_age_fragility_v2.png", dpi=150)
    print("\nSaved plot to continuous_age_fragility_v2.png")
