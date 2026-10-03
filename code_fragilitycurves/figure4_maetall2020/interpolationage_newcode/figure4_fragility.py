"""
Age-dependent wood-pole fragility curves (reconstruction of Ma et al. 2020, Fig. 4).

Python port of the MATLAB script supplied by the author, extended so it
produces a curve for every age in AGES (default: 0, 5, 10, ..., 60 years)
instead of only new / 30-yr / 60-yr poles.

Units are SI unless noted. Requires numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 1
N = 500_000                                   # Monte Carlo samples
V = np.round(np.arange(20.0, 90.0 + 1e-9, 0.1), 1)   # wind speeds (m/s)
AGES = list(range(0, 65, 5))                  # 0, 5, ..., 60 years
INCLUDE_ORIGINAL_NEW_POLE = True              # also compute the archived "new pole" curve

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------
def aged_pole_strength_kpa(age_years, n, rng):
    """Sample pole bending strength (kPa) for a pole of the given age.

    Mean strength degrades with age; coefficient of variation grows by
    0.01667 every 5 years. Samples are lognormal with that mean and COV.
    """
    mean_kpa = 52200.0 * (1.0 - (0.014418 * age_years - 0.10683)
                          * (0.00013 * age_years ** 1.846))
    cov = 0.17 + 0.01667 * (age_years / 5.0)

    # Convert (mean, COV) of the strength into parameters of the underlying normal
    mu_log = np.log(mean_kpa ** 2 / np.sqrt((cov * mean_kpa) ** 2 + mean_kpa ** 2))
    sigma_log = np.sqrt(np.log(cov ** 2 + 1.0))

    return rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)


def fragility_from_critical_speed(strength_pa, gravity_stress, wind_coeff, wind_speed):
    """P(failure | V) for each V in wind_speed.

    Each sample fails once  wind_coeff * V^2 + gravity_stress >= strength,
    i.e. at the critical speed  V_crit = sqrt((strength - gravity) / wind_coeff).
    The fragility curve is the empirical CDF of V_crit, P(V_crit < V).
    """
    required = strength_pa - gravity_stress
    v_crit = np.full(strength_pa.shape, np.inf)       # never fails (degenerate case)

    v_crit[required <= 0] = 0.0                       # fails under gravity alone

    ok = (required > 0) & (wind_coeff > 0)
    v_crit[ok] = np.sqrt(required[ok] / wind_coeff[ok])

    # Count of samples with V_crit strictly below each V (matches MATLAB histcounts/cumsum)
    v_sorted = np.sort(v_crit)
    return np.searchsorted(v_sorted, wind_speed, side="left") / v_crit.size


# ---------------------------------------------------------------------------
# Random wind-load parameters (shared by every age so curves are comparable)
# ---------------------------------------------------------------------------
Kzp = rng.normal(0.951, 0.05706, N)   # exposure coefficient, pole
Kzw = rng.normal(1.024, 0.06144, N)   # exposure coefficient, wires
Gp  = rng.normal(0.948, 0.10428, N)   # gust factor, pole
Gw  = rng.normal(0.801, 0.08811, N)   # gust factor, wires
Cfp = rng.normal(0.9,   0.108,   N)   # force coefficient, pole
Cfw = rng.normal(1.0,   0.12,    N)   # force coefficient, wires
Ap  = rng.normal(2.66,  0.1596,  N)   # projected area, pole (m^2)
Aw  = rng.normal(5.16,  0.3098,  N)   # projected area, wires (m^2)
Hp  = rng.normal(11.7,  0.351,   N)   # pole height above ground (m)
Hw  = rng.normal(11.1,  0.333,   N)   # wire attachment height (m)

# Force per unit V^2 (N per (m/s)^2): F = 0.613 * Kz * G * Cf * A * V^2
Kp = 0.613 * Kzp * Gp * Cfp * Ap
Kw = 0.613 * Kzw * Gw * Cfw * Aw

# Pole geometry and gravity load
D = 0.28                                            # pole diameter (m)
Gi = 7766 * 0.09144                                 # gravity load from archived code (N)
gravity_stress = Gi * 3 / ((D / 2) ** 2 * np.pi)    # Pa

# Base bending stress per unit V^2:
#   moment = (pole force * Hp/2) + (wire force * Hw),  section modulus = pi*D^3/32
wind_coeff = 32.0 * (0.5 * Kp * Hp + Kw * Hw) / (np.pi * D ** 3)

# ---------------------------------------------------------------------------
# Fragility curves
# ---------------------------------------------------------------------------
curves = {}

if INCLUDE_ORIGINAL_NEW_POLE:
    strength_new = rng.lognormal(mean=10.8486, sigma=0.16879, size=N) * 1000.0  # kPa -> Pa
    curves["New (archived)"] = fragility_from_critical_speed(
        strength_new, gravity_stress, wind_coeff, V)

for age in AGES:
    strength = aged_pole_strength_kpa(age, N, rng) * 1000.0                    # kPa -> Pa
    curves[f"{age} yr"] = fragility_from_critical_speed(
        strength, gravity_stress, wind_coeff, V)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, ax = plt.subplots(figsize=(8, 6), facecolor="w")

colors = plt.cm.viridis(np.linspace(0, 0.95, len(AGES)))
for color, age in zip(colors, AGES):
    ax.plot(V, curves[f"{age} yr"], color=color, lw=1.5, label=f"{age}-yr poles")
if INCLUDE_ORIGINAL_NEW_POLE:
    ax.plot(V, curves["New (archived)"], "r:", lw=1.5, label="New poles (archived)")

ax.set_xlim(20, 90)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(20, 91, 5))
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_xlabel("Extreme wind speed (m/s)")
ax.set_ylabel("Probability of failure")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("reconstructed_figure4_5yr.png", dpi=300)

# ---------------------------------------------------------------------------
# Export CSV: one column per age
# ---------------------------------------------------------------------------
names = list(curves.keys())
header = "WindSpeed_mps," + ",".join(n.replace(" ", "_").replace("(", "").replace(")", "")
                                     for n in names)
data = np.column_stack([V] + [curves[n] for n in names])
np.savetxt("figure4_fragility_curves_5yr.csv", data, delimiter=",",
           header=header, comments="", fmt="%.6f")

if __name__ == "__main__":
    plt.show()
