"""
TORONTO WEATHER DATA
====================

Toronto-specific extreme wind and ice return-period data.

SOURCE
------
Krishnasamy, S. & Kulendran, S. (1998), "A procedure for calculating
wind-on-ice loads", Atmospheric Research, 46, 123-129.

The Toronto International Airport values below are entered directly from
the Toronto weather-data paper.

The Ma, Dai & Pang (2020) framework uses:
    - a Weibull distribution for extreme wind speed
    - a Generalized Pareto Distribution (GPD) for extreme ice thickness

This module fits those same functional forms to the Toronto data.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The Toronto source data and the Seattle/Grand Marais data used by
Ma et al. (2020) do not come from exactly the same source/reference
framework. Ma et al. state that their wind parameters were calculated
from ASCE 7-16 design wind speeds, whereas the Toronto paper provides
Toronto weather-station extreme values.

Therefore, this module is an adaptation of Ma et al.'s DISTRIBUTION
FORMULATIONS to the Toronto return-period data. It is not a claim that
Toronto reproduces the ASCE data used by Ma et al.

SOURCE DATA RANGE
-----------------
The Toronto paper provides the following return periods:

    10, 20, 40, 50, 60, 80, 100, 200, 500 years

The fitted curves are therefore supported directly by source data over
10-500 years. Extrapolation beyond 500 years should be labelled as such.

UNITS
-----
Wind:
    source = km/h
    model = m/s

Ice:
    mm

Return period:
    years
"""

import numpy as np
from scipy.optimize import curve_fit


# ============================================================================
# 1. STATIC TORONTO SOURCE DATA
# ============================================================================

# Toronto International Airport.
# Values transcribed directly from the Toronto paper.

TORONTO_RETURN_PERIODS_YEARS = np.array(
    [10, 20, 40, 50, 60, 80, 100, 200, 500],
    dtype=float,
)

TORONTO_ICE_THICKNESS_MM = np.array(
    [13, 15, 18, 19, 19, 20, 21, 23, 27],
    dtype=float,
)

TORONTO_WIND_SPEED_KMH = np.array(
    [79.98, 85.13, 90.28, 92.05, 93.34, 95.43, 97.04, 102.19, 108.79],
    dtype=float,
)

# Convert source wind speeds to SI units for the hazard model.
TORONTO_WIND_SPEED_MS = TORONTO_WIND_SPEED_KMH / 3.6

# Toronto-specific wind-on-ice factor reported by the source paper.
# Keep this separate from the concurrent-wind hazard distribution.
TORONTO_WIND_ON_ICE_FACTOR = 0.55


# ============================================================================
# 2. WEIBULL WIND HAZARD MODEL
# ============================================================================

def weibull_speed_for_return_period(T, alpha, u):
    """
    Weibull extreme wind speed for a T-year return period.

    Based on Eq. (4) in Ma, Dai & Pang (2020):

        v(T) = u * [-ln(1/T)]^(1/alpha)

    Parameters
    ----------
    T : float or array-like
        Return period in years. Must be > 1.
    alpha : float
        Weibull shape parameter.
    u : float
        Weibull scale parameter in m/s.
    """
    T = np.asarray(T, dtype=float)

    if np.any(T <= 1):
        raise ValueError("Return period T must be greater than 1 year.")

    return u * (-np.log(1.0 / T)) ** (1.0 / alpha)


def fit_toronto_weibull():
    """
    Fit Weibull alpha and u using ALL nine Toronto wind observations.

    curve_fit performs nonlinear least squares. The model has two unknown
    parameters and is therefore not restricted to three observations.

    Returns
    -------
    alpha, u
        Weibull shape and scale parameters.
    """

    def model(T, alpha, u):
        return weibull_speed_for_return_period(T, alpha, u)

    (alpha, u), covariance = curve_fit(
        model,
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS,
        p0=[3.0, 25.0],
        bounds=([1e-8, 1e-8], [np.inf, np.inf]),
        maxfev=10000,
    )

    return float(alpha), float(u), covariance


# ============================================================================
# 3. GENERALIZED PARETO ICE HAZARD MODEL
# ============================================================================

def gpd_thickness_for_return_period(T, alpha, k, u=0.0, lam=1.0):
    """
    Generalized Pareto extreme ice thickness for a T-year return period.

    Based on Eq. (6) in Ma, Dai & Pang (2020):

        r(T) = u + (alpha/k) * [1 - (lambda*T)^(-k)]

    The k -> 0 limit is handled as:

        r(T) = u + alpha*ln(lambda*T)

    Parameters
    ----------
    T : float or array-like
        Return period in years.
    alpha : float
        GPD scale parameter in mm.
    k : float
        GPD shape parameter.
    u : float
        Threshold/location parameter in mm.
    lam : float
        Event rate in 1/year.
    """
    T = np.asarray(T, dtype=float)

    if np.any(T <= 0):
        raise ValueError("Return period T must be greater than 0.")
    if lam <= 0:
        raise ValueError("lambda must be greater than 0.")

    if np.isclose(k, 0.0):
        return u + alpha * np.log(lam * T)

    return u + (alpha / k) * (1.0 - (lam * T) ** (-k))


def fit_toronto_gpd(u=0.0, lam=1.0):
    """
    Fit GPD alpha and k using ALL nine Toronto ice observations.

    The Toronto adaptation uses:
        u = 0 mm
        lambda = 1/year

    Returns
    -------
    alpha, k
        GPD scale and shape parameters.
    """

    def model(T, alpha, k):
        return gpd_thickness_for_return_period(
            T, alpha, k, u=u, lam=lam
        )

    (alpha, k), covariance = curve_fit(
        model,
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_ICE_THICKNESS_MM,
        p0=[6.0, 0.1],
        bounds=([1e-8, -0.99], [np.inf, np.inf]),
        maxfev=10000,
    )

    return float(alpha), float(k), covariance


# ============================================================================
# 4. FIT TORONTO PARAMETERS
# ============================================================================

TORONTO_WIND_ALPHA, TORONTO_WIND_U, TORONTO_WIND_COVARIANCE = (
    fit_toronto_weibull()
)

TORONTO_ICE_ALPHA, TORONTO_ICE_K, TORONTO_ICE_COVARIANCE = (
    fit_toronto_gpd(u=0.0, lam=1.0)
)


# This dictionary is what hazard_distribution.py should import.
TORONTO_PARAMETERS = {
    "wind": {
        "alpha": TORONTO_WIND_ALPHA,
        "u": TORONTO_WIND_U,
    },
    "ice": {
        "alpha": TORONTO_ICE_ALPHA,
        "k": TORONTO_ICE_K,
        "u": 0.0,
        "lam": 1.0,
    },
    "wind_on_ice": {
        "factor": TORONTO_WIND_ON_ICE_FACTOR,
    },
}


# ============================================================================
# 5. FITTED CURVES
# ============================================================================

def fitted_wind_speed_ms(T):
    """Return the Toronto fitted Weibull wind speed in m/s."""
    return weibull_speed_for_return_period(
        T,
        TORONTO_WIND_ALPHA,
        TORONTO_WIND_U,
    )


def fitted_wind_speed_kmh(T):
    """Return the Toronto fitted Weibull wind speed in km/h."""
    return fitted_wind_speed_ms(T) * 3.6


def fitted_ice_thickness_mm(T):
    """Return the Toronto fitted GPD ice thickness in mm."""
    return gpd_thickness_for_return_period(
        T,
        TORONTO_ICE_ALPHA,
        TORONTO_ICE_K,
        u=0.0,
        lam=1.0,
    )


# ============================================================================
# 6. SOURCE-DATA COMPARISON / GOODNESS-OF-FIT
# ============================================================================

def calculate_fit_errors():
    """
    Calculate residuals and simple fit statistics for both hazards.

    Returns
    -------
    dict
        Source values, fitted values, residuals, RMSE and R^2.
    """
    wind_fit = fitted_wind_speed_ms(TORONTO_RETURN_PERIODS_YEARS)
    ice_fit = fitted_ice_thickness_mm(TORONTO_RETURN_PERIODS_YEARS)

    wind_residuals = TORONTO_WIND_SPEED_MS - wind_fit
    ice_residuals = TORONTO_ICE_THICKNESS_MM - ice_fit

    wind_rmse = np.sqrt(np.mean(wind_residuals ** 2))
    ice_rmse = np.sqrt(np.mean(ice_residuals ** 2))

    wind_ss_res = np.sum(wind_residuals ** 2)
    ice_ss_res = np.sum(ice_residuals ** 2)

    wind_ss_tot = np.sum(
        (TORONTO_WIND_SPEED_MS - np.mean(TORONTO_WIND_SPEED_MS)) ** 2
    )
    ice_ss_tot = np.sum(
        (TORONTO_ICE_THICKNESS_MM - np.mean(TORONTO_ICE_THICKNESS_MM)) ** 2
    )

    wind_r2 = 1.0 - wind_ss_res / wind_ss_tot
    ice_r2 = 1.0 - ice_ss_res / ice_ss_tot

    return {
        "wind": {
            "source": TORONTO_WIND_SPEED_MS,
            "fit": wind_fit,
            "residuals": wind_residuals,
            "rmse": wind_rmse,
            "r2": wind_r2,
        },
        "ice": {
            "source": TORONTO_ICE_THICKNESS_MM,
            "fit": ice_fit,
            "residuals": ice_residuals,
            "rmse": ice_rmse,
            "r2": ice_r2,
        },
    }


def print_validation_table():
    """Print fitted parameters and source-vs-fit comparison."""

    errors = calculate_fit_errors()

    print("\n" + "=" * 72)
    print("TORONTO HAZARD DISTRIBUTION")
    print("=" * 72)

    print("\nWEIBULL WIND FIT")
    print(f"  alpha = {TORONTO_WIND_ALPHA:.6f}")
    print(f"  u     = {TORONTO_WIND_U:.6f} m/s")
    print(f"  RMSE  = {errors['wind']['rmse']:.6f} m/s")
    print(f"  R^2   = {errors['wind']['r2']:.6f}")

    print("\nGPD ICE FIT")
    print(f"  alpha = {TORONTO_ICE_ALPHA:.6f} mm")
    print(f"  k     = {TORONTO_ICE_K:.6f}")
    print("  u     = 0.000000 mm")
    print("  lambda = 1.000000 / year")
    print(f"  RMSE  = {errors['ice']['rmse']:.6f} mm")
    print(f"  R^2   = {errors['ice']['r2']:.6f}")

    print("\nTORONTO WIND-ON-ICE FACTOR")
    print(f"  factor = {TORONTO_WIND_ON_ICE_FACTOR:.2f}")

    print("\nSOURCE DATA VS FIT")
    print("-" * 72)
    print(
        f"{'T (yr)':>8} | "
        f"{'Wind src':>10} | {'Wind fit':>10} | "
        f"{'Ice src':>9} | {'Ice fit':>9}"
    )
    print("-" * 72)

    for T, wind_src, wind_fit, ice_src, ice_fit in zip(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS,
        errors["wind"]["fit"],
        TORONTO_ICE_THICKNESS_MM,
        errors["ice"]["fit"],
    ):
        print(
            f"{T:8.0f} | "
            f"{wind_src:10.3f} | {wind_fit:10.3f} | "
            f"{ice_src:9.2f} | {ice_fit:9.2f}"
        )


# ============================================================================
# 7. STANDARD RETURN-PERIOD SUMMARY
# ============================================================================

def print_standard_return_period_summary():
    """
    Print fitted Toronto values at the return periods most useful for the
    capstone.

    10-500 years are within the source-data range.
    1000 and 1700 years are extrapolations beyond the Toronto source data.
    """

    return_periods = np.array(
        [10, 25, 50, 100, 200, 500, 1000, 1700],
        dtype=float,
    )

    print("\n" + "=" * 72)
    print("TORONTO FITTED HAZARD CURVE")
    print("=" * 72)
    print("Values at 1000 and 1700 years are extrapolations beyond the")
    print("500-year maximum return period in the Toronto source data.\n")

    print(
        f"{'T (yr)':>8} | "
        f"{'Wind (m/s)':>12} | "
        f"{'Wind (km/h)':>13} | "
        f"{'Ice (mm)':>10}"
    )
    print("-" * 72)

    for T in return_periods:
        print(
            f"{T:8.0f} | "
            f"{fitted_wind_speed_ms(T):12.3f} | "
            f"{fitted_wind_speed_kmh(T):13.2f} | "
            f"{fitted_ice_thickness_mm(T):10.3f}"
        )


# ============================================================================
# 8. PLOT SOURCE DATA + FITTED CURVES
# ============================================================================

def plot_toronto_hazard_curves(
    save_path="toronto_hazard_curves.png",
    show_extrapolation=True,
):
    """
    Plot Toronto source observations against the fitted hazard curves.

    The solid fitted curves cover the 10-500 year source-data range.
    If show_extrapolation=True, the fitted curves continue beyond 500 years
    with a visually different line style to make extrapolation explicit.
    """

    import matplotlib.pyplot as plt

    # Supported source-data range.
    T_supported = np.linspace(10, 500, 500)

    # Optional extrapolation.
    T_extrapolated = np.linspace(500, 1700, 300)

    wind_supported = fitted_wind_speed_ms(T_supported)
    ice_supported = fitted_ice_thickness_mm(T_supported)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ------------------------------------------------------------------
    # WIND
    # ------------------------------------------------------------------
    axes[0].scatter(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS,
        label="Toronto source data",
        zorder=3,
    )

    axes[0].plot(
        T_supported,
        wind_supported,
        label="Weibull fit (10–500 yr)",
    )

    if show_extrapolation:
        axes[0].plot(
            T_extrapolated,
            fitted_wind_speed_ms(T_extrapolated),
            linestyle="--",
            label="Weibull extrapolation (>500 yr)",
        )

    axes[0].set_xlabel("Return period (years)")
    axes[0].set_ylabel("Extreme wind speed (m/s)")
    axes[0].set_title("Toronto Wind Hazard")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # ------------------------------------------------------------------
    # ICE
    # ------------------------------------------------------------------
    axes[1].scatter(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_ICE_THICKNESS_MM,
        label="Toronto source data",
        zorder=3,
    )

    axes[1].plot(
        T_supported,
        ice_supported,
        label="GPD fit (10–500 yr)",
    )

    if show_extrapolation:
        axes[1].plot(
            T_extrapolated,
            fitted_ice_thickness_mm(T_extrapolated),
            linestyle="--",
            label="GPD extrapolation (>500 yr)",
        )

    axes[1].set_xlabel("Return period (years)")
    axes[1].set_ylabel("Extreme ice thickness (mm)")
    axes[1].set_title("Toronto Ice Hazard")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nSaved plot to: {save_path}")


# ============================================================================
# 9. MAIN
# ============================================================================

if __name__ == "__main__":
    print_validation_table()
    print_standard_return_period_summary()
    plot_toronto_hazard_curves()
