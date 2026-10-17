"""
TORONTO WEATHER DATA
====================

Toronto-specific extreme wind and ice return-period data.

SOURCE
------
Krishnasamy, S. & Kulendran, S. (1998), "A procedure for calculating
wind-on-ice loads", Atmospheric Research, 46, 123-129.
"Combined wind and ice loads from historical
extreme wind and ice data"

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

WIND AVERAGING TIME -- VERSION 1 vs VERSION 2
--------------------------------------------
Krishnasamy & Kulendran (1998) do not state the averaging time of their wind
speeds. They are treated here as HOURLY MEAN speeds at 10 m, because:
  - they are AES weather-station annual extremes, and the paper's loads are
    for a conductor at 33 ft (= 10 m), the standard reference height;
  - the 50-yr value (92.05 km/h = 25.6 m/s) is close to the 50-yr hourly-mean
    Toronto wind in Sheng et al. (2023) Table 4 (27.6 m/s);
  - read as 3-s gusts, Toronto's 50-yr wind would be ~30% below Ma et al.'s
    ASCE 7 sites (Seattle 35.0, Grand Marais 36.8 m/s), which is implausible.

Ma et al. (2020) Fig. 4 and Fig. 7 use 3-SECOND GUST speeds (ASCE 7 basis).
So the wind hazard must be converted before it is integrated against Fig. 4.

  VERSION 1 (original functions, unchanged): hourly-mean speeds, as published.
  VERSION 2 (functions ending in _v2): 3-s gust speeds = hourly mean x 1.52.

The 1.52 factor is the Durst (1960) gust factor, as reproduced in the ASCE 7-16
Commentary, Fig. C26.5-1: the ratio of the 3-s gust to the hourly-mean wind
speed in open terrain at 10 m. It is the same factor used in
concurrentwindspeed.py (DURST_3S).

Use VERSION 2 for anything integrated against Ma et al.'s fragility curves.
The ice fit is identical in both versions (ice thickness has no averaging time).

Ice:
    mm

Return period:
    years
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit


SCRIPT_DIR = Path(__file__).resolve().parent
TORONTO_WIND_PARAMETERS_CSV = SCRIPT_DIR / "toronto_wind_parameters.csv"
TORONTO_WIND_XY_CSV = SCRIPT_DIR / "toronto_wind_xy_coordinates.csv"
TORONTO_ICE_PARAMETERS_CSV = SCRIPT_DIR / "toronto_ice_gpd_parameters.csv"
TORONTO_ICE_XY_CSV = SCRIPT_DIR / "toronto_ice_xy_coordinates.csv"


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

# ---- VERSION 2: convert hourly-mean speeds to 3-s gusts --------------------
# Durst (1960) / ASCE 7-16 Commentary Fig. C26.5-1: 3-s gust / hourly mean
# = 1.52 in open terrain at 10 m. Same factor as concurrentwindspeed.py.
HOURLY_TO_3S_GUST = 1.52
TORONTO_WIND_SPEED_MS_V2 = TORONTO_WIND_SPEED_MS * HOURLY_TO_3S_GUST

TORONTO_WIND_PARAMETERS_CSV_V2 = SCRIPT_DIR / "toronto_wind_parameters_v2_3s_gust.csv"
TORONTO_WIND_XY_CSV_V2 = SCRIPT_DIR / "toronto_wind_xy_coordinates_v2_3s_gust.csv"


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


def lookup_toronto_ice_gpd_parameters():
    """
    Return the fitted Toronto GPD ice parameters (alpha, k).

    Usage:
        from toronto_weather_data2 import lookup_toronto_ice_gpd_parameters
        alpha, k = lookup_toronto_ice_gpd_parameters()

    Returns
    -------
    alpha : float
        GPD scale parameter in mm.
    k : float
        GPD shape parameter (dimensionless).

    The fixed values u = 0 mm and lambda = 1/year are not returned; they
    are in TORONTO_PARAMETERS["ice"] if needed.
    """
    return TORONTO_ICE_ALPHA, TORONTO_ICE_K


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


def save_toronto_wind_outputs(
    parameters_path=TORONTO_WIND_PARAMETERS_CSV,
    xy_path=TORONTO_WIND_XY_CSV,
):
    """Save fitted Weibull parameters and the wind-fit XY coordinates.

    The XY file records the nine source coordinates used to estimate the
    Weibull distribution. It also includes each coordinate's fitted value
    and residual so the fit can be reproduced or checked directly.
    """
    parameters_path = Path(parameters_path)
    xy_path = Path(xy_path)

    pd.DataFrame([
        {
            "parameter": "alpha",
            "value": TORONTO_WIND_ALPHA,
            "units": "dimensionless",
            "description": "Weibull shape parameter",
        },
        {
            "parameter": "u",
            "value": TORONTO_WIND_U,
            "units": "m/s",
            "description": "Weibull scale parameter",
        },
    ]).to_csv(parameters_path, index=False)

    fitted_wind = fitted_wind_speed_ms(TORONTO_RETURN_PERIODS_YEARS)
    pd.DataFrame({
        "x_return_period_years": TORONTO_RETURN_PERIODS_YEARS,
        "y_observed_wind_speed_m_per_s": TORONTO_WIND_SPEED_MS,
        "y_fitted_wind_speed_m_per_s": fitted_wind,
        "residual_observed_minus_fitted_m_per_s": (
            TORONTO_WIND_SPEED_MS - fitted_wind
        ),
    }).to_csv(xy_path, index=False)

    print(f"Saved Toronto wind Weibull parameters to: {parameters_path}")
    print(f"Saved Toronto wind XY coordinates to: {xy_path}")


def save_toronto_ice_outputs(
    parameters_path=TORONTO_ICE_PARAMETERS_CSV,
    xy_path=TORONTO_ICE_XY_CSV,
):
    """Save fitted GPD parameters and the ice-fit XY coordinates.

    The XY file records the nine source coordinates used for the GPD fit,
    along with the fitted ice thickness and residual at each return period.
    """
    parameters_path = Path(parameters_path)
    xy_path = Path(xy_path)

    pd.DataFrame([
        {
            "parameter": "alpha",
            "value": TORONTO_ICE_ALPHA,
            "units": "mm",
            "description": "GPD scale parameter",
        },
        {
            "parameter": "k",
            "value": TORONTO_ICE_K,
            "units": "dimensionless",
            "description": "GPD shape parameter",
        },
        {
            "parameter": "u",
            "value": 0.0,
            "units": "mm",
            "description": "GPD threshold/location parameter (fixed)",
        },
        {
            "parameter": "lambda",
            "value": 1.0,
            "units": "1/year",
            "description": "GPD event rate (fixed)",
        },
    ]).to_csv(parameters_path, index=False)

    fitted_ice = fitted_ice_thickness_mm(TORONTO_RETURN_PERIODS_YEARS)
    pd.DataFrame({
        "x_return_period_years": TORONTO_RETURN_PERIODS_YEARS,
        "y_observed_ice_thickness_mm": TORONTO_ICE_THICKNESS_MM,
        "y_fitted_ice_thickness_mm": fitted_ice,
        "residual_observed_minus_fitted_mm": (
            TORONTO_ICE_THICKNESS_MM - fitted_ice
        ),
    }).to_csv(xy_path, index=False)

    print(f"Saved Toronto ice GPD parameters to: {parameters_path}")
    print(f"Saved Toronto ice XY coordinates to: {xy_path}")


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
# 9. VERSION 2 -- WIND AS 3-SECOND GUSTS (hourly mean x 1.52)
# ============================================================================
# Everything in this section mirrors the version 1 function of the same name,
# but uses TORONTO_WIND_SPEED_MS_V2 (3-s gusts). The ice functions are not
# duplicated because the ice fit does not change.
#
# Note: multiplying every wind speed by a constant leaves the Weibull shape
# (alpha) unchanged and multiplies the scale (u) by the same constant, so
# expect alpha_v2 = alpha and u_v2 = 1.52 * u. Refitting confirms this.

def fit_toronto_weibull_v2():
    """Version 2 of fit_toronto_weibull(): fit the 3-s gust speeds."""

    def model(T, alpha, u):
        return weibull_speed_for_return_period(T, alpha, u)

    (alpha, u), covariance = curve_fit(
        model,
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS_V2,
        p0=[3.0, 38.0],
        bounds=([1e-8, 1e-8], [np.inf, np.inf]),
        maxfev=10000,
    )

    return float(alpha), float(u), covariance


TORONTO_WIND_ALPHA_V2, TORONTO_WIND_U_V2, TORONTO_WIND_COVARIANCE_V2 = (
    fit_toronto_weibull_v2()
)

# Version 2 parameter dictionary: gust wind, same ice.
TORONTO_PARAMETERS_V2 = {
    "wind": {
        "alpha": TORONTO_WIND_ALPHA_V2,
        "u": TORONTO_WIND_U_V2,
        "averaging": "3-s gust (hourly mean x 1.52)",
    },
    "ice": TORONTO_PARAMETERS["ice"],
    "wind_on_ice": TORONTO_PARAMETERS["wind_on_ice"],
}


def lookup_toronto_wind_weibull_parameters():
    """Version 1 Weibull (alpha, u): hourly-mean wind, as published."""
    return TORONTO_WIND_ALPHA, TORONTO_WIND_U


def lookup_toronto_wind_weibull_parameters_v2():
    """Version 2 Weibull (alpha, u): 3-s gust wind. Use this with Ma et al.'s
    Fig. 4 / Fig. 7 fragility curves.

    Usage:
        from toronto_weather_data2 import lookup_toronto_wind_weibull_parameters_v2
        alpha, u = lookup_toronto_wind_weibull_parameters_v2()
    """
    return TORONTO_WIND_ALPHA_V2, TORONTO_WIND_U_V2


def fitted_wind_speed_ms_v2(T):
    """Return the Toronto fitted Weibull 3-s gust speed in m/s."""
    return weibull_speed_for_return_period(
        T,
        TORONTO_WIND_ALPHA_V2,
        TORONTO_WIND_U_V2,
    )


def fitted_wind_speed_kmh_v2(T):
    """Return the Toronto fitted Weibull 3-s gust speed in km/h."""
    return fitted_wind_speed_ms_v2(T) * 3.6


def save_toronto_wind_outputs_v2(
    parameters_path=TORONTO_WIND_PARAMETERS_CSV_V2,
    xy_path=TORONTO_WIND_XY_CSV_V2,
):
    """Version 2 of save_toronto_wind_outputs(): 3-s gust Weibull parameters
    and XY coordinates. Also records the hourly-mean source value."""
    parameters_path = Path(parameters_path)
    xy_path = Path(xy_path)

    pd.DataFrame([
        {
            "parameter": "alpha",
            "value": TORONTO_WIND_ALPHA_V2,
            "units": "dimensionless",
            "description": "Weibull shape parameter (3-s gust)",
        },
        {
            "parameter": "u",
            "value": TORONTO_WIND_U_V2,
            "units": "m/s",
            "description": "Weibull scale parameter (3-s gust)",
        },
        {
            "parameter": "hourly_to_3s_gust",
            "value": HOURLY_TO_3S_GUST,
            "units": "dimensionless",
            "description": "Durst (1960) / ASCE 7-16 Commentary Fig. C26.5-1",
        },
    ]).to_csv(parameters_path, index=False)

    fitted_wind = fitted_wind_speed_ms_v2(TORONTO_RETURN_PERIODS_YEARS)
    pd.DataFrame({
        "x_return_period_years": TORONTO_RETURN_PERIODS_YEARS,
        "source_hourly_mean_wind_speed_m_per_s": TORONTO_WIND_SPEED_MS,
        "y_observed_3s_gust_m_per_s": TORONTO_WIND_SPEED_MS_V2,
        "y_fitted_3s_gust_m_per_s": fitted_wind,
        "residual_observed_minus_fitted_m_per_s": (
            TORONTO_WIND_SPEED_MS_V2 - fitted_wind
        ),
    }).to_csv(xy_path, index=False)

    print(f"Saved Toronto wind Weibull parameters (v2, 3-s gust) to: {parameters_path}")
    print(f"Saved Toronto wind XY coordinates (v2, 3-s gust) to: {xy_path}")


def print_validation_table_v2():
    """Version 2 of print_validation_table(): wind fit on 3-s gusts."""
    wind_fit = fitted_wind_speed_ms_v2(TORONTO_RETURN_PERIODS_YEARS)
    res = TORONTO_WIND_SPEED_MS_V2 - wind_fit
    rmse = np.sqrt(np.mean(res ** 2))
    r2 = 1.0 - np.sum(res ** 2) / np.sum(
        (TORONTO_WIND_SPEED_MS_V2 - np.mean(TORONTO_WIND_SPEED_MS_V2)) ** 2
    )

    print("\n" + "=" * 72)
    print("TORONTO HAZARD DISTRIBUTION -- VERSION 2 (wind as 3-s gust)")
    print("=" * 72)
    print(f"\nHourly mean -> 3-s gust factor = {HOURLY_TO_3S_GUST}")
    print("\nWEIBULL WIND FIT (3-s gust)")
    print(f"  alpha = {TORONTO_WIND_ALPHA_V2:.6f}   (v1: {TORONTO_WIND_ALPHA:.6f})")
    print(f"  u     = {TORONTO_WIND_U_V2:.6f} m/s (v1: {TORONTO_WIND_U:.6f} m/s)")
    print(f"  RMSE  = {rmse:.6f} m/s")
    print(f"  R^2   = {r2:.6f}")
    print("\nGPD ICE FIT: unchanged from version 1")

    print("\nSOURCE DATA VS FIT")
    print("-" * 72)
    print(f"{'T (yr)':>8} | {'Hourly src':>10} | {'Gust src':>10} | {'Gust fit':>10}")
    print("-" * 72)
    for T, h, g, f in zip(TORONTO_RETURN_PERIODS_YEARS, TORONTO_WIND_SPEED_MS,
                          TORONTO_WIND_SPEED_MS_V2, wind_fit):
        print(f"{T:8.0f} | {h:10.3f} | {g:10.3f} | {f:10.3f}")


def print_standard_return_period_summary_v2():
    """Version 2 of print_standard_return_period_summary()."""
    return_periods = np.array(
        [10, 25, 50, 100, 200, 500, 1000, 1700],
        dtype=float,
    )

    print("\n" + "=" * 72)
    print("TORONTO FITTED HAZARD CURVE -- VERSION 2 (wind as 3-s gust)")
    print("=" * 72)
    print("Values at 1000 and 1700 years are extrapolations beyond the")
    print("500-year maximum return period in the Toronto source data.\n")

    print(
        f"{'T (yr)':>8} | "
        f"{'Gust (m/s)':>12} | "
        f"{'Gust (km/h)':>13} | "
        f"{'Hourly (m/s)':>13} | "
        f"{'Ice (mm)':>10}"
    )
    print("-" * 72)

    for T in return_periods:
        print(
            f"{T:8.0f} | "
            f"{fitted_wind_speed_ms_v2(T):12.3f} | "
            f"{fitted_wind_speed_kmh_v2(T):13.2f} | "
            f"{fitted_wind_speed_ms(T):13.3f} | "
            f"{fitted_ice_thickness_mm(T):10.3f}"
        )


def plot_toronto_hazard_curves_v2(
    save_path="toronto_hazard_curves_v2_3s_gust.png",
    show_extrapolation=True,
):
    """Version 2 of plot_toronto_hazard_curves().

    Wind panel: the 3-s gust data and fit (v2), with the hourly-mean data and
    fit (v1) shown faintly for comparison. Ice panel: same as version 1.
    """

    import matplotlib.pyplot as plt

    T_supported = np.linspace(10, 500, 500)
    T_extrapolated = np.linspace(500, 1700, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ------------------------------------------------------------------
    # WIND
    # ------------------------------------------------------------------
    axes[0].scatter(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS,
        color="grey",
        alpha=0.5,
        label="Source data, hourly mean (v1)",
        zorder=3,
    )
    axes[0].plot(
        T_supported,
        fitted_wind_speed_ms(T_supported),
        color="grey",
        alpha=0.5,
        label="Weibull fit, hourly mean (v1)",
    )

    axes[0].scatter(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_WIND_SPEED_MS_V2,
        label=f"Source data x {HOURLY_TO_3S_GUST} = 3-s gust (v2)",
        zorder=3,
    )
    axes[0].plot(
        T_supported,
        fitted_wind_speed_ms_v2(T_supported),
        label="Weibull fit, 3-s gust (10–500 yr)",
    )

    if show_extrapolation:
        axes[0].plot(
            T_extrapolated,
            fitted_wind_speed_ms_v2(T_extrapolated),
            linestyle="--",
            label="Weibull extrapolation (>500 yr)",
        )

    axes[0].set_xlabel("Return period (years)")
    axes[0].set_ylabel("Extreme wind speed (m/s)")
    axes[0].set_title("Toronto Wind Hazard (v2: 3-s gust)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # ------------------------------------------------------------------
    # ICE (unchanged)
    # ------------------------------------------------------------------
    axes[1].scatter(
        TORONTO_RETURN_PERIODS_YEARS,
        TORONTO_ICE_THICKNESS_MM,
        label="Toronto source data",
        zorder=3,
    )
    axes[1].plot(
        T_supported,
        fitted_ice_thickness_mm(T_supported),
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
    axes[1].set_title("Toronto Ice Hazard (unchanged)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nSaved plot to: {save_path}")


# ============================================================================
# 10. MAIN
# ============================================================================

if __name__ == "__main__":
    # Version 1: hourly-mean wind, as published
    save_toronto_wind_outputs()
    save_toronto_ice_outputs()
    print_validation_table()
    print_standard_return_period_summary()
    plot_toronto_hazard_curves()

    # Version 2: wind as 3-s gust (use this with Ma et al.'s fragility curves)
    save_toronto_wind_outputs_v2()
    print_validation_table_v2()
    print_standard_return_period_summary_v2()
    plot_toronto_hazard_curves_v2()
