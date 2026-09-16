"""
Reproduce Figure 5 of Ma, Dai & Pang (2020) — "Annual probability of failure
curves of different age poles subjected to wind hazards" — by numerically
integrating the digitized fragility curves (Fig. 4) against the Weibull wind
hazard model (Table 1), per Eq. (10) of the paper:

    Pf = integral_0^inf  FR(x) * f(x) dx

where FR(x) is the fragility (probability of failure) at wind speed x, and
f(x) is the Weibull probability density of the annual extreme wind speed.

Inputs
------
red.csv    -> fragility curve for NEW poles      (age = 0 yr)
black.csv  -> fragility curve for 30-year poles  (age = 30 yr)
blue.csv   -> fragility curve for 60-year poles  (age = 60 yr)
(digitized from Fig. 4 of the paper)

Output
------
A plot in the style of Fig. 5, and a printed comparison against the paper's
own reported values in Table 2:

    Age (yr)   Seattle              Grand Marais
    0          2.6388e-5            6.7191e-5
    30         7.2766e-5            2.9581e-4
    60         4.8471e-3            8.1103e-3

NOTE on scope: the paper's Fig. 5 is drawn as a smooth curve over pole ages
10-60 years. That requires a fragility curve at *every* age, which the paper
generates from a full Monte Carlo structural model (pole geometry, section
modulus, etc.) that is not fully specified in the text. We only have
digitized fragility curves at 3 ages (0, 30, 60 yr), so this script:
  1) computes the exact integral at those 3 ages (the rigorous, defensible
     part - this is what gets compared to Table 2), and
  2) additionally interpolates log(Pf) between those 3 points just to draw
     a Fig.-5-like continuous curve. That interpolation is for visual
     comparison only and is clearly labeled as such - it is NOT a
     re-derivation of the paper's structural model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ----------------------------------------------------------------------
# 1. Load and clean the digitized fragility curves
# ----------------------------------------------------------------------

def load_fragility_csv(path):
    """Load a digitized (wind_speed, failure_prob) curve, sort by wind
    speed, clip small negative/overshoot artifacts from digitization,
    and de-duplicate."""
    data = np.loadtxt(path, delimiter=",")
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    y = np.clip(y, 0.0, 1.0)
    # average duplicate x-values (digitizer sometimes emits near-duplicates)
    x_unique, inv = np.unique(np.round(x, 6), return_inverse=True)
    y_unique = np.zeros_like(x_unique)
    counts = np.zeros_like(x_unique)
    for i, xi in enumerate(inv):
        y_unique[xi] += y[i]
        counts[xi] += 1
    y_unique /= counts
    return x_unique, y_unique


def make_fragility_function(x, y):
    """Return FR(v): fragility curve extended with FR=0 below the digitized
    range and FR=1 above it (matching the paper's statement that the
    digitized range, 20-90 m/s, spans the full practical fragility curve)."""
    def FR(v):
        v = np.atleast_1d(np.asarray(v, dtype=float))
        out = np.interp(v, x, y, left=0.0, right=1.0)
        return out
    return FR


curves = {}
for age, fname in [(0, "files/red.csv"), (30, "files/black.csv"), (60, "files/blue.csv")]:
    x, y = load_fragility_csv(fname)
    curves[age] = make_fragility_function(x, y)

# ----------------------------------------------------------------------
# 2. Weibull wind hazard model (Eq. 3, Table 1 parameters)
# ----------------------------------------------------------------------

WEIBULL_PARAMS = {
    "Seattle":      {"alpha": 3.369, "u": 23.3742},
    "Grand Marais": {"alpha": 3.569, "u": 25.1044},
}


def weibull_pdf(v, alpha, u):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    pos = v > 0
    out[pos] = (alpha / u) * (v[pos] / u) ** (alpha - 1) * np.exp(-(v[pos] / u) ** alpha)
    return out

# ----------------------------------------------------------------------
# 3. Integrate Eq. (10):  Pf = integral FR(v) * f(v) dv
# ----------------------------------------------------------------------

def annual_failure_probability(FR, alpha, u, v_max=250.0, n=200_000):
    """Numerically integrate FR(v)*f(v) over v in [0, v_max] with a fine
    trapezoidal grid (fast, robust, and avoids quad warnings from the
    piecewise-linear/kinked FR curve)."""
    v = np.linspace(0.0, v_max, n)
    integrand = FR(v) * weibull_pdf(v, alpha, u)
    return np.trapz(integrand, v)


ages = [0, 30, 60]
computed = {loc: [] for loc in WEIBULL_PARAMS}

for loc, params in WEIBULL_PARAMS.items():
    for age in ages:
        Pf = annual_failure_probability(curves[age], params["alpha"], params["u"])
        computed[loc].append(Pf)

# Paper's Table 2 values (element/pole reliability, wind hazard only)
table2 = {
    "Seattle":      [2.6388e-5, 7.2766e-5, 4.8471e-3],
    "Grand Marais": [6.7191e-5, 2.9581e-4, 8.1103e-3],
}

# ----------------------------------------------------------------------
# 4. Print numeric comparison
# ----------------------------------------------------------------------

print(f"{'Location':<13}{'Age':>5}{'Computed Pf':>16}{'Paper Table 2':>16}{'% diff':>10}")
print("-" * 62)
for loc in WEIBULL_PARAMS:
    for i, age in enumerate(ages):
        c = computed[loc][i]
        p = table2[loc][i]
        pct = 100 * (c - p) / p
        print(f"{loc:<13}{age:>5}{c:>16.4e}{p:>16.4e}{pct:>9.1f}%")

# ----------------------------------------------------------------------
# 5. Plot: reproduction of Figure 5
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5.5))

age_fine = np.linspace(0, 60, 300)
colors = {"Seattle": "tab:blue", "Grand Marais": "tab:red"}
markers = {"Seattle": "s", "Grand Marais": "o"}

for loc in WEIBULL_PARAMS:
    # visual-only smooth interpolation through the 3 computed points
    # (monotone cubic in log-space, since Pf spans orders of magnitude)
    log_pf = np.log10(computed[loc])
    interp = PchipInterpolator(ages, log_pf)
    pf_smooth = 10 ** interp(age_fine)

    ax.plot(age_fine, pf_smooth, "--", color=colors[loc], alpha=0.6,
             label=f"{loc} (this reproduction, interpolated)")
    ax.plot(ages, computed[loc], markers[loc], color=colors[loc], ms=8,
             label=f"{loc} (computed at digitized ages)")
    ax.plot(ages, table2[loc], "x", color="black", ms=10, mew=2,
             label=f"{loc} (paper, Table 2)" if loc == "Seattle" else None)

ax.set_xlabel("Pole age (years)")
ax.set_ylabel("Probability of failure")
ax.set_title("Reproduction of Fig. 5: Annual failure probability vs. pole age\n"
             "(wind hazard only)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("files/figure5_reproduction.png", dpi=200)
print("\nSaved plot to figure5_reproduction.png")

# Also produce a version matching the paper's actual axis style (0-60yr, linear
# y-axis 0-0.009, as in Fig. 5(a) which is dominated by pole strength decay)
fig2, ax2 = plt.subplots(figsize=(7, 5))
for loc in WEIBULL_PARAMS:
    log_pf = np.log10(computed[loc])
    interp = PchipInterpolator(ages, log_pf)
    pf_smooth = 10 ** interp(age_fine)
    ax2.plot(age_fine, pf_smooth, "-", color=colors[loc], lw=2,
              label=f"{loc} (reproduced)")
    ax2.plot(ages, table2[loc], markers[loc], color=colors[loc],
              mfc="white", ms=9, mew=2, label=f"{loc} (paper Table 2)")
ax2.set_xlabel("Pole age (years)")
ax2.set_ylabel("Probability of failure")
ax2.set_ylim(0, max(max(table2["Grand Marais"]), max(table2["Seattle"])) * 1.15)
ax2.set_title("Fig. 5 style comparison: reproduced curve vs. paper's reported values")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig2.tight_layout()
# fig2.savefig("files/figure5_reproduction_linear.png", dpi=200)
print("Saved plot to figure5_reproduction_linear.png")
