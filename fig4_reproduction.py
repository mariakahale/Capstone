"""
MA ET AL. (2020) — FIGURE 4 WIND FRAGILITY
============================================

Purpose
-------
1. Read digitized Ma et al. (2020) Figure 4 data.
2. Recreate Figure 4 as a clean, capstone-ready plot.
3. Fit lognormal fragility parameters (theta, beta).
4. Validate the digitized curves against values explicitly reported
   in Ma et al.
5. Combine the digitized fragility curves with Ma et al.'s Weibull
   wind hazard distributions to reproduce annual failure probabilities
   reported in Table 2.
6. Plot those annual failure probabilities versus pole age as a
   numerical check against Figure 5.

FILES
-----
Place these files in a folder called "figure4":

    figure4/
        red.csv       -> NEW pole
        black.csv     -> 30-year pole
        blue.csv      -> 60-year pole

Each CSV should contain:
    column 1 = wind speed (m/s)
    column 2 = probability of failure

COLOUR ASSIGNMENT FROM YOUR DIGITIZATION
-----------------------------------------
    red   = New pole
    black = 30-year pole
    blue  = 60-year pole
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.integrate import trapezoid


# =================================================================
# 1. FILE LOCATIONS
# =================================================================

NEW_FILE = "figure4/red.csv"
THIRTY_FILE = "figure4/black.csv"
SIXTY_FILE = "figure4/blue.csv"


# =================================================================
# 2. LOAD DIGITIZED DATA
# =================================================================

def load_digitized_curve(filename):
    """
    Reads a WebPlotDigitizer CSV.

    First column:
        wind speed (m/s)

    Second column:
        probability of failure
    """

    data = pd.read_csv(filename, header=None)

    wind_speed = pd.to_numeric(
        data.iloc[:, 0],
        errors="coerce"
    ).to_numpy()

    probability = pd.to_numeric(
        data.iloc[:, 1],
        errors="coerce"
    ).to_numpy()

    # Remove invalid rows
    valid = (
        np.isfinite(wind_speed)
        & np.isfinite(probability)
    )

    wind_speed = wind_speed[valid]
    probability = probability[valid]

    # Sort by wind speed
    order = np.argsort(wind_speed)

    wind_speed = wind_speed[order]
    probability = probability[order]

    # Probability must be between 0 and 1
    probability = np.clip(probability, 0, 1)

    return wind_speed, probability


# Load all three curves
new_v, new_pf = load_digitized_curve(NEW_FILE)
thirty_v, thirty_pf = load_digitized_curve(THIRTY_FILE)
sixty_v, sixty_pf = load_digitized_curve(SIXTY_FILE)


CURVES = {
    "New pole": {
        "wind_speed": new_v,
        "probability": new_pf,
    },

    "30-year pole": {
        "wind_speed": thirty_v,
        "probability": thirty_pf,
    },

    "60-year pole": {
        "wind_speed": sixty_v,
        "probability": sixty_pf,
    },
}


# =================================================================
# 3. LOGNORMAL FRAGILITY MODEL
# =================================================================

def lognormal_fragility(v, theta, beta):
    """
    Lognormal fragility function:

        P_f(V) = Phi[
            ln(V/theta) / beta
        ]

    theta = median failure wind speed
    beta  = logarithmic dispersion
    """

    v = np.asarray(v, dtype=float)

    return norm.cdf(
        (np.log(v) - np.log(theta)) / beta
    )


def fit_fragility(wind_speed, probability):
    """
    Fits theta and beta to the digitized Figure 4 data.
    """

    # Avoid exact 0 and 1 when fitting because the lognormal
    # model approaches these asymptotically.
    mask = (
        (probability > 0)
        & (probability < 1)
        & (wind_speed > 0)
    )

    v = wind_speed[mask]
    p = probability[mask]

    (theta, beta), _ = curve_fit(
        lognormal_fragility,
        v,
        p,
        p0=[50, 0.15],
        bounds=([1, 0.001], [200, 2]),
        maxfev=50000
    )

    return theta, beta


# =================================================================
# 4. FIT ALL THREE CURVES
# =================================================================

FIT_RESULTS = {}

print("\n" + "=" * 70)
print("LOGNORMAL FITS")
print("=" * 70)

for label, data in CURVES.items():

    theta, beta = fit_fragility(
        data["wind_speed"],
        data["probability"]
    )

    FIT_RESULTS[label] = {
        "theta": theta,
        "beta": beta,
    }

    print(
        f"{label:15s}: "
        f"theta = {theta:.3f} m/s, "
        f"beta = {beta:.4f}"
    )


# =================================================================
# 5. RECREATE FIGURE 4
# =================================================================

plt.figure(figsize=(9, 6))

wind_range = np.linspace(20, 90, 500)

for label, data in CURVES.items():

    # Plot digitized data
    plt.scatter(
        data["wind_speed"],
        data["probability"],
        s=12,
        alpha=0.6,
        label=f"{label} — digitized"
    )

    # Plot fitted lognormal representation
    theta = FIT_RESULTS[label]["theta"]
    beta = FIT_RESULTS[label]["beta"]

    plt.plot(
        wind_range,
        lognormal_fragility(
            wind_range,
            theta,
            beta
        ),
        linewidth=2,
        label=f"{label} — lognormal fit"
    )


plt.xlabel("Extreme wind speed (m/s)")
plt.ylabel("Probability of failure")
plt.title("Wind Fragility Curves — Ma et al. (2020), Figure 4")
plt.xlim(20, 90)
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(
    "figure4_recreated.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# # =================================================================
# # 6. DIRECT VALIDATION AGAINST VALUES STATED IN THE PAPER
# # =================================================================
# #
# # Ma et al. explicitly report:
# #
# # New pole:
# #     P(failure | V = 75 m/s) = 0.9957
# #
# # 60-year pole:
# #     P(failure | V = 75 m/s) = 0.9827
# #
# # =================================================================

# ANCHORS = {
#     "New pole": {
#         "wind_speed": 75,
#         "published_probability": 0.9957,
#     },

#     "60-year pole": {
#         "wind_speed": 75,
#         "published_probability": 0.9827,
#     },
# }


# print("\n" + "=" * 70)
# print("FIGURE 4 ANCHOR VALIDATION")
# print("=" * 70)

# for label, anchor in ANCHORS.items():

#     theta = FIT_RESULTS[label]["theta"]
#     beta = FIT_RESULTS[label]["beta"]

#     predicted = lognormal_fragility(
#         anchor["wind_speed"],
#         theta,
#         beta
#     )

#     published = anchor["published_probability"]

#     error = predicted - published

#     percent_error = (
#         error / published * 100
#     )

#     print(f"\n{label}")
#     print(f"  Wind speed       = {anchor['wind_speed']} m/s")
#     print(f"  Digitized fit    = {predicted:.4f}")
#     print(f"  Paper            = {published:.4f}")
#     print(f"  Difference       = {error:+.4f}")
#     print(f"  Percent error    = {percent_error:+.2f}%")


# # =================================================================
# # 7. MA ET AL. WEIBULL WIND HAZARD
# # =================================================================

# LOCATIONS = {

#     "Seattle": {
#         "alpha": 3.369,
#         "u": 23.3742,
#     },

#     "Grand Marais": {
#         "alpha": 3.569,
#         "u": 25.1044,
#     },
# }


# def weibull_pdf(v, alpha, u):
#     """
#     Weibull probability density used for the extreme wind hazard.
#     """

#     v = np.asarray(v, dtype=float)

#     return (
#         (alpha / u)
#         * (v / u) ** (alpha - 1)
#         * np.exp(-(v / u) ** alpha)
#     )


# # =================================================================
# # 8. ANNUAL FAILURE PROBABILITY
# # =================================================================

# def annual_failure_probability(
#     wind_speed,
#     fragility,
#     alpha,
#     u
# ):
#     """
#     Calculates:

#         P_annual = integral[
#             FR(V) * f(V)
#         ] dV

#     using trapezoidal numerical integration.
#     """

#     wind_pdf = weibull_pdf(
#         wind_speed,
#         alpha,
#         u
#     )

#     integrand = (
#         fragility
#         * wind_pdf
#     )

#     return trapezoid(
#         integrand,
#         wind_speed
#     )


# # =================================================================
# # 9. PUBLISHED TABLE 2 VALUES
# # =================================================================
# #
# # These are the values we are trying to reproduce.
# #
# # =================================================================

# PUBLISHED_TABLE_2 = {

#     "Seattle": {
#         "New pole": 2.6388e-5,
#         "30-year pole": 7.2766e-5,
#         "60-year pole": 4.8471e-3,
#     },

#     "Grand Marais": {
#         "New pole": 6.7191e-5,
#         "30-year pole": 2.9581e-4,
#         "60-year pole": 8.1103e-3,
#     },
# }


# # =================================================================
# # 10. CALCULATE ANNUAL FAILURE PROBABILITIES
# # =================================================================

# CALCULATED = {}

# print("\n" + "=" * 70)
# print("ANNUAL FAILURE PROBABILITY — TABLE 2 VALIDATION")
# print("=" * 70)

# for location, parameters in LOCATIONS.items():

#     CALCULATED[location] = {}

#     alpha = parameters["alpha"]
#     u = parameters["u"]

#     for label, data in CURVES.items():

#         # Use the digitized fragility directly.
#         #
#         # This is important:
#         # We are checking the actual digitized Figure 4,
#         # not just the fitted lognormal approximation.

#         calculated = annual_failure_probability(
#             data["wind_speed"],
#             data["probability"],
#             alpha,
#             u
#         )

#         published = PUBLISHED_TABLE_2[
#             location
#         ][label]

#         percent_error = (
#             (calculated - published)
#             / published
#             * 100
#         )

#         CALCULATED[location][label] = calculated

#         print(
#             f"{location:15s} | "
#             f"{label:15s} | "
#             f"Calculated = {calculated:.6e} | "
#             f"Published = {published:.6e} | "
#             f"Error = {percent_error:+.1f}%"
#         )


# # =================================================================
# # 11. PLOT ANNUAL FAILURE PROBABILITY VS POLE AGE
# # =================================================================
# #
# # This is our numerical reconstruction/check of the WIND portion
# # of Ma et al. Figure 5.
# #
# # Figure 5 itself is continuous with age; we only have the three
# # ages for which we digitized Figure 4.
# #
# # =================================================================

# ages = np.array([0, 30, 60])

# plt.figure(figsize=(8, 6))

# for location in LOCATIONS:

#     calculated_values = np.array([
#         CALCULATED[location]["New pole"],
#         CALCULATED[location]["30-year pole"],
#         CALCULATED[location]["60-year pole"],
#     ])

#     published_values = np.array([
#         PUBLISHED_TABLE_2[location]["New pole"],
#         PUBLISHED_TABLE_2[location]["30-year pole"],
#         PUBLISHED_TABLE_2[location]["60-year pole"],
#     ])

#     # Our calculation
#     plt.plot(
#         ages,
#         calculated_values,
#         marker="o",
#         linewidth=2,
#         label=f"{location} — calculated"
#     )

#     # Published Table 2 values
#     plt.plot(
#         ages,
#         published_values,
#         marker="x",
#         linestyle="--",
#         label=f"{location} — Ma et al."
#     )


# plt.xlabel("Pole age (years)")
# plt.ylabel("Annual probability of failure")
# plt.title(
#     "Annual Wind Failure Probability — Table 2 / Figure 5 Check"
# )

# plt.yscale("log")
# plt.xticks([0, 30, 60])
# plt.grid(alpha=0.3, which="both")
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     "annual_failure_validation.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()


# # =================================================================
# # 12. INTEGRAND DIAGNOSTIC
# # =================================================================
# #
# # This shows WHERE the annual failure probability comes from.
# #
# # Area under:
# #
# #       FR(V) * f(V)
# #
# # equals P_annual.
# #
# # =================================================================

# for location, parameters in LOCATIONS.items():

#     alpha = parameters["alpha"]
#     u = parameters["u"]

#     plt.figure(figsize=(9, 6))

#     for label, data in CURVES.items():

#         wind_speed = data["wind_speed"]
#         fragility = data["probability"]

#         wind_pdf = weibull_pdf(
#             wind_speed,
#             alpha,
#             u
#         )

#         contribution = (
#             fragility
#             * wind_pdf
#         )

#         plt.plot(
#             wind_speed,
#             contribution,
#             linewidth=2,
#             label=label
#         )

#     plt.xlabel("Extreme wind speed (m/s)")
#     plt.ylabel("Fragility × wind PDF")
#     plt.title(
#         f"{location}: Contribution to Annual Failure Probability"
#     )

#     plt.grid(alpha=0.3)
#     plt.legend()

#     plt.tight_layout()

#     plt.savefig(
#         f"{location.lower().replace(' ', '_')}_integrand.png",
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.show()


# print("\nDone.")
# print("Generated:")
# print("  figure4_recreated.png")
# print("  annual_failure_validation.png")
# print("  seattle_integrand.png")
# print("  grand_marais_integrand.png")