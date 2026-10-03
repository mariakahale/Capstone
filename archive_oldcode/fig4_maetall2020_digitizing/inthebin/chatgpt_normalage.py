"""
Age-dependent fragility approximation following Ma et al. (2020)

Purpose
-------
Approximate the 0-, 30-, and 60-year pole fragility curves from
Ma et al. (2020) without running a large Monte Carlo simulation.

Core idea
---------
1. Use Ma et al.'s conditional decay model to obtain:
       - age-dependent mean pole strength
       - age-dependent variance

2. Represent pole strength using an age-dependent lognormal distribution.

3. Use a digitized Ma et al. fragility curve as the baseline relationship
   between wind speed and structural demand.

4. Convert the baseline failure probability into an equivalent demand
   using the NEW-pole strength distribution.

5. Apply the age-dependent strength distributions to that same demand.

This preserves the shape/information contained in the digitized baseline
curve while incorporating Ma et al.'s age-dependent deterioration model.

IMPORTANT
---------
This is an analytical approximation to Ma et al.'s Monte Carlo procedure.
It is NOT a reproduction of their complete finite-element/mechanical model.

The exact Ma et al. Monte Carlo model randomizes multiple structural/load
parameters. Therefore, exact reproduction of Fig. 4 requires recreating
that full model.

Required packages:
    numpy
    pandas
    scipy
    matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import lognorm
from scipy.interpolate import PchipInterpolator


# ============================================================
# 1. USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# Your digitized Ma et al. curve
# ------------------------------------------------------------

_DIGITIZED_DATA_DIR = (
    Path(__file__).resolve().parent / "csv_source_files_digitizingfig4"
)
DIGITIZED_FILE = _DIGITIZED_DATA_DIR / "red.csv"
DIGITIZED_30YR_FILE = _DIGITIZED_DATA_DIR / "black.csv"
DIGITIZED_60YR_FILE = _DIGITIZED_DATA_DIR / "blue.csv"

# If your CSV contains multiple ages, set this to True.
MULTI_AGE_DIGITIZED_FILE = False

# If using a single digitized curve, what age is it?
# Usually this should be 0 if you digitized Ma et al.'s NEW-pole curve.
DIGITIZED_BASELINE_AGE = 0


# ------------------------------------------------------------
# Ages we want to generate
# ------------------------------------------------------------

AGES = [0, 30, 60]


# ------------------------------------------------------------
# Wind-speed range from Ma et al. Fig. 4
# ------------------------------------------------------------

WIND_MIN = 20.0
WIND_MAX = 90.0

# Ma et al. used 0.1 m/s increments for Fig. 4.
WIND_STEP = 0.1

wind_grid = np.arange(
    WIND_MIN,
    WIND_MAX + WIND_STEP,
    WIND_STEP
)


# ============================================================
# 2. MA ET AL. STRENGTH DETERIORATION PARAMETERS
# ============================================================

# New Southern Pine pole strength
R0 = 52.2          # MPa

# Coefficient of variation of NEW pole strength
COV0 = 0.17

# Parameters reported by Ma et al.
a1 = 0.014418
a2 = 0.10683

b1 = 0.00013
b2 = 1.846

VarT = 0.11


# ============================================================
# 3. MA ET AL. CONDITIONAL DECAY MODEL
# ============================================================

def ma_strength_statistics(age):
    """
    Calculate age-dependent mean and variance of pole strength
    using Ma et al. Eqs. (1) and (2).

    Parameters
    ----------
    age : float
        Pole age in years.

    Returns
    -------
    mean_strength : float
        Age-dependent mean pole strength [MPa].

    variance_strength : float
        Age-dependent variance of pole strength [MPa^2].

    std_strength : float
        Age-dependent standard deviation [MPa].

    cov_strength : float
        Age-dependent coefficient of variation.
    """

    # --------------------------------------------------------
    # Original variance
    # --------------------------------------------------------

    Var0 = (COV0 * R0) ** 2

    # --------------------------------------------------------
    # Conditional decay probability
    #
    # q(t) = min(max(b1*t^b2, 0), 1)
    # --------------------------------------------------------

    q = np.clip(
        b1 * age**b2,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Mean percentage strength loss
    #
    # L(t) = min(max(a1*t-a2, 0), 1)
    # --------------------------------------------------------

    loss = np.clip(
        a1 * age - a2,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Ma et al. Eq. (1)
    # --------------------------------------------------------

    remaining_factor = 1.0 - loss * q

    mean_strength = R0 * remaining_factor

    # --------------------------------------------------------
    # Ma et al. Eq. (2)
    #
    # The published equation contains the R0^2 term in the
    # final variance contribution.
    # --------------------------------------------------------

    variance_strength = (
        (Var0 + R0**2) * (1.0 - q)
        +
        (
            (Var0 + R0**2)
            *
            (
                VarT
                +
                (1.0 - loss)**2
            )
            *
            q
        )
        -
        R0**2 * remaining_factor**2
    )

    # Numerical protection
    variance_strength = max(variance_strength, 1e-12)

    std_strength = np.sqrt(variance_strength)

    cov_strength = std_strength / mean_strength

    return (
        mean_strength,
        variance_strength,
        std_strength,
        cov_strength
    )


# ============================================================
# 4. LOGNORMAL DISTRIBUTION PARAMETERS
# ============================================================

def lognormal_parameters(mean, variance):
    """
    Convert arithmetic mean and variance into scipy's
    lognormal parameters.

    If:

        X ~ Lognormal(mu_ln, sigma_ln)

    then:

        E[X] = mean
        Var[X] = variance
    """

    sigma_ln_squared = np.log(
        1.0 + variance / mean**2
    )

    sigma_ln = np.sqrt(sigma_ln_squared)

    mu_ln = np.log(mean) - 0.5 * sigma_ln_squared

    return mu_ln, sigma_ln


def make_strength_distribution(age):
    """
    Create the age-dependent lognormal pole-strength distribution.
    """

    mean_strength, variance_strength, std_strength, cov_strength = (
        ma_strength_statistics(age)
    )

    mu_ln, sigma_ln = lognormal_parameters(
        mean_strength,
        variance_strength
    )

    distribution = lognorm(
        s=sigma_ln,
        scale=np.exp(mu_ln)
    )

    return {
        "age": age,
        "mean": mean_strength,
        "variance": variance_strength,
        "std": std_strength,
        "cov": cov_strength,
        "mu_ln": mu_ln,
        "sigma_ln": sigma_ln,
        "distribution": distribution
    }


# ============================================================
# 5. PRINT AGE-DEPENDENT STRENGTH CHECK
# ============================================================

print("\n" + "=" * 70)
print("AGE-DEPENDENT POLE STRENGTH")
print("=" * 70)

strength_distributions = {}

for age in AGES:

    info = make_strength_distribution(age)

    strength_distributions[age] = info

    print(
        f"\nAge = {age} years"
    )

    print(
        f"  Mean strength       = {info['mean']:.3f} MPa"
    )

    print(
        f"  Standard deviation  = {info['std']:.3f} MPa"
    )

    print(
        f"  Variance            = {info['variance']:.3f} MPa²"
    )

    print(
        f"  COV                 = {info['cov']:.3f}"
    )


# ============================================================
# 6. LOAD DIGITIZED CURVE
# ============================================================

def load_digitized_curve(filename):
    """
    Load a digitized fragility curve.

    Required columns:
        wind_speed
        failure_probability
    """

    # WebPlotDigitizer exports these files without a header row.
    df = pd.read_csv(
        filename,
        header=None,
        names=["wind_speed", "failure_probability"],
        usecols=[0, 1],
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Sort
    df = df.sort_values("wind_speed")

    # Remove duplicate wind speeds
    df = df.drop_duplicates(
        subset="wind_speed"
    )

    # Make sure probabilities are between 0 and 1
    df["failure_probability"] = np.clip(
        df["failure_probability"],
        1e-8,
        1.0 - 1e-8
    )

    return df


digitized = load_digitized_curve(
    DIGITIZED_FILE
)
digitized_30yr = load_digitized_curve(DIGITIZED_30YR_FILE)
digitized_60yr = load_digitized_curve(DIGITIZED_60YR_FILE)

print("\nDigitized curve loaded:")
print(digitized.head())


# ============================================================
# 7. INTERPOLATE DIGITIZED BASELINE
# ============================================================

baseline_interpolator = PchipInterpolator(
    digitized["wind_speed"].values,
    digitized["failure_probability"].values,
    extrapolate=False
)


# Only use wind speeds inside the digitized range
digitized_min_wind = digitized["wind_speed"].min()
digitized_max_wind = digitized["wind_speed"].max()

valid_wind = (
    (wind_grid >= digitized_min_wind)
    &
    (wind_grid <= digitized_max_wind)
)

calculation_wind = wind_grid[valid_wind]


# Baseline failure probability
baseline_probability = baseline_interpolator(
    calculation_wind
)

baseline_probability = np.clip(
    baseline_probability,
    1e-8,
    1.0 - 1e-8
)


# ============================================================
# 8. CORE AGE-SHIFTING METHOD
# ============================================================

def generate_age_curve(
    baseline_probability,
    baseline_distribution,
    age_distribution
):
    """
    Generate an age-dependent fragility curve.

    The baseline digitized curve tells us:

        P_failure(wind)

    For the new pole:

        P0 = F_C0(D)

    Therefore the equivalent demand is:

        D = F_C0^{-1}(P0)

    For an aged pole:

        Pt = F_Ct(D)

    This is the analytical approximation used here.
    """

    # --------------------------------------------------------
    # Step 1:
    # Convert baseline failure probabilities to equivalent
    # structural demand.
    # --------------------------------------------------------

    equivalent_demand = baseline_distribution.ppf(
        baseline_probability
    )

    # --------------------------------------------------------
    # Step 2:
    # Apply aged capacity distribution to the SAME demand.
    # --------------------------------------------------------

    age_probability = age_distribution.cdf(
        equivalent_demand
    )

    age_probability = np.clip(
        age_probability,
        0.0,
        1.0
    )

    return age_probability


# ============================================================
# 9. GENERATE ALL AGE CURVES
# ============================================================

new_distribution = strength_distributions[
    DIGITIZED_BASELINE_AGE
]["distribution"]


generated_curves = {}

for age in AGES:

    age_distribution = strength_distributions[
        age
    ]["distribution"]

    generated_curves[age] = generate_age_curve(
        baseline_probability,
        new_distribution,
        age_distribution
    )


# ============================================================
# 10. IMPORTANT CHECK:
# NEW POLE MUST MATCH DIGITIZED BASELINE
# ============================================================

new_curve = generated_curves[
    DIGITIZED_BASELINE_AGE
]

new_error = new_curve - baseline_probability

print("\n" + "=" * 70)
print("BASELINE CHECK")
print("=" * 70)

print(
    f"Maximum absolute error: "
    f"{np.max(np.abs(new_error)):.8e}"
)

print(
    f"Mean absolute error: "
    f"{np.mean(np.abs(new_error)):.8e}"
)

if np.max(np.abs(new_error)) < 1e-6:
    print(
        "PASS: The analytical transformation reproduces "
        "the digitized baseline curve."
    )
else:
    print(
        "WARNING: Baseline reproduction is not exact."
    )


# ============================================================
# 11. CHECK AGAINST A DIGITIZED AGED CURVE
# ============================================================

def compare_with_digitized(
    generated_wind,
    generated_probability,
    digitized_wind,
    digitized_probability
):
    """
    Compare generated curve against digitized curve.

    Interpolates the generated curve at the digitized
    wind-speed points.
    """

    # Keep only points inside generated range
    mask = (
        (digitized_wind >= generated_wind.min())
        &
        (digitized_wind <= generated_wind.max())
    )

    x = digitized_wind[mask]
    y_digitized = digitized_probability[mask]

    generated_interpolator = PchipInterpolator(
        generated_wind,
        generated_probability
    )

    y_generated = generated_interpolator(x)

    errors = y_generated - y_digitized

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))

    return {
        "wind_speed": x,
        "digitized": y_digitized,
        "generated": y_generated,
        "error": errors,
        "MAE": mae,
        "RMSE": rmse,
        "MAX": max_error
    }


# ------------------------------------------------------------
# If your digitized CSV is only ONE age, compare it against
# the generated curve corresponding to that age.
# ------------------------------------------------------------

comparison = compare_with_digitized(
    calculation_wind,
    generated_curves[DIGITIZED_BASELINE_AGE],
    digitized["wind_speed"].values,
    digitized["failure_probability"].values
)

print("\n" + "=" * 70)
print(
    f"CHECK AGAINST DIGITIZED {DIGITIZED_BASELINE_AGE}-YEAR CURVE"
)
print("=" * 70)

print(
    f"MAE       = {comparison['MAE']:.6f}"
)

print(
    f"RMSE      = {comparison['RMSE']:.6f}"
)

print(
    f"Max error = {comparison['MAX']:.6f}"
)


# ============================================================
# 12. POINT-BY-POINT CHECKS
# ============================================================

check_winds = [
    25,
    30,
    40,
    50,
    60,
    70,
    75,
    80
]

print("\n" + "=" * 70)
print("POINT-BY-POINT FRAGILITY CHECK")
print("=" * 70)

for age in AGES:

    curve_interpolator = PchipInterpolator(
        calculation_wind,
        generated_curves[age]
    )

    print(f"\nAge = {age} years")

    for wind in check_winds:

        if (
            wind >= calculation_wind.min()
            and
            wind <= calculation_wind.max()
        ):

            probability = float(
                curve_interpolator(wind)
            )

            print(
                f"  Wind = {wind:5.1f} m/s"
                f"    Pf = {probability:.5f}"
            )


# ============================================================
# 13. CHECK PUBLISHED Ma et al. 75 m/s VALUE
# ============================================================

print("\n" + "=" * 70)
print("PUBLISHED Ma et al. CHECK")
print("=" * 70)

"""
Ma et al. report that at approximately 75 m/s:

    New poles:
        Pf = 99.57%

    30/60-year aged poles:
        curves become less steep.

They specifically note that aged poles can have a slightly LOWER
failure probability at high wind speeds because the deterioration
model produces a very large variance.

This is an important sanity check.
"""

published_wind = 75.0
published_new_pf = 0.9957

if (
    published_wind >= calculation_wind.min()
    and
    published_wind <= calculation_wind.max()
):

    new_interp = PchipInterpolator(
        calculation_wind,
        generated_curves[0]
    )

    generated_new_pf = float(
        new_interp(published_wind)
    )

    print(
        f"Ma et al. reported at 75 m/s:"
    )

    print(
        f"    New-pole Pf ≈ {published_new_pf:.4f}"
    )

    print(
        f"Our generated new-pole Pf = "
        f"{generated_new_pf:.4f}"
    )

    print(
        f"Difference = "
        f"{generated_new_pf - published_new_pf:+.4f}"
    )


# ============================================================
# 14. PLOT AGE-DEPENDENT CURVES
# ============================================================

plt.figure(figsize=(10, 6))

for age in AGES:

    plt.plot(
        calculation_wind,
        generated_curves[age],
        label=f"Generated {age}-year curve"
    )


# Digitized baseline
plt.scatter(
    digitized["wind_speed"],
    digitized["failure_probability"],
    s=25,
    label="Digitized new-pole curve"
)

# Digitized Figure 4 curves for direct comparison with the generated ages.
plt.scatter(
    digitized_30yr["wind_speed"],
    digitized_30yr["failure_probability"],
    s=18,
    marker="x",
    label="Digitized 30-year curve"
)
plt.scatter(
    digitized_60yr["wind_speed"],
    digitized_60yr["failure_probability"],
    s=18,
    marker="x",
    label="Digitized 60-year curve"
)

plt.xlabel("Wind speed (m/s)")
plt.ylabel("Probability of pole failure")

plt.title(
    "Age-Dependent Pole Fragility\n"
    "Analytical approximation of Ma et al. (2020)"
)

plt.ylim(0, 1.02)
plt.xlim(WIND_MIN, WIND_MAX)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 15. PLOT THE STRENGTH DISTRIBUTIONS
# ============================================================

strength_values = np.linspace(
    0.1,
    80,
    1000
)

plt.figure(figsize=(10, 6))

for age in AGES:

    distribution = strength_distributions[
        age
    ]["distribution"]

    plt.plot(
        strength_values,
        distribution.pdf(strength_values),
        label=f"{age} years"
    )

plt.xlabel("Pole strength (MPa)")
plt.ylabel("Probability density")

plt.title(
    "Age-Dependent Pole Strength Distributions"
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 16. FINAL SUMMARY TABLE
# ============================================================

summary = []

for age in AGES:

    info = strength_distributions[age]

    summary.append({
        "Age (years)": age,
        "Mean strength (MPa)": info["mean"],
        "Std. dev. (MPa)": info["std"],
        "COV": info["cov"]
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("FINAL AGE-DEPENDENT STRENGTH SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
