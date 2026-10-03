from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# ACTUAL fitted parameters
# ============================================================

theta_0 = 55.361
beta_0 = 0.1131

theta_60 = 48.377
beta_60 = 0.2052

theta_30 = 54.567
beta_30 = 0.1267

# ============================================================
# EXPERIMENT 1
# Predict the 30-year parameters using linear interpolation
# between the actual 0-year and 60-year parameters.
# ============================================================

age = 30

theta_30_pred = theta_0 + (age / 60) * (theta_60 - theta_0)

beta_30_pred = beta_0 + (age / 60) * (beta_60 - beta_0)


print("Predicted 30-year parameters:")
print(f"theta_30 = {theta_30_pred:.4f}")
print(f"beta_30  = {beta_30_pred:.4f}")


# ============================================================
# Fragility function
# ============================================================

def fragility_curve(v, theta, beta):
    """
    Lognormal fragility curve.

    v     = wind speed
    theta = median failure wind speed
    beta  = logarithmic standard deviation
    """

    return 0.5 * (
        1 + erf(np.log(v / theta) / (beta * np.sqrt(2)))
    )


# Wind-speed range
v = np.linspace(1, 100, 1000)


# ============================================================
# Generate the predicted 30-year fragility curve
# ============================================================

P_30_pred = fragility_curve(
    v,
    theta_30_pred,
    beta_30_pred
)


# ============================================================
# Plot
# ============================================================

P_30_actual = fragility_curve(
    v,
    theta_30,
    beta_30
)

plt.figure(figsize=(8, 5))

plt.plot(
    v,
    P_30_pred,
    label="Predicted 30-year"
)

plt.plot(
    v,
    P_30_actual,
    "--",
    label="Actual 30-year"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability of Failure")
plt.title("Experiment 1: Predicted vs Actual 30-Year Fragility")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

theta_error = theta_30_pred - theta_30
beta_error = beta_30_pred - beta_30

print(f"Theta error = {theta_error:.4f}")
print(f"Beta error  = {beta_error:.4f}")