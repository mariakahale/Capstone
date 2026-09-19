"""
Fit lognormal and logistic fragility curves to digitized (x, P_f) data,
compare goodness of fit, and plot both against the data.

Expected CSV format (no header assumptions beyond first two columns):
    x_value, failure_probability
    e.g.:
    10, 0.02
    15, 0.05
    20, 0.13
    ...

Usage:
    Edit the CSV_FILES list below to point at your digitized files,
    then run: python fit_fragility_curves.py

Each CSV should contain ONE curve (e.g., "Class 2 pole, vs wind speed").
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
from pathlib import Path

# Resolve all input and output paths relative to this script, not the folder
# from which Python happens to be launched.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIRS = {
    "wind": SCRIPT_DIR / "fragilityfit_wind",
    "ice": SCRIPT_DIR / "fragilityfit_ice",
}
# ---------------------------------------------------------------------
# 1. CONFIGURE YOUR FILES HERE
# ---------------------------------------------------------------------
# Each entry: (csv_path, label, x_axis_name, valid_x_range)
# valid_x_range is the range the underlying simulation actually covered
# (used only to print a warning if your data/predictions fall outside it)

CSV_FILES = [
    # --- Wind speed sweep (Fig 5a), ice thickness fixed at 3.2 cm ---
    dict(path="windvfailure/class2pole.csv",  group="wind", label="Class 2 - Pole",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),
    dict(path="windvfailure/class2wire.csv",  group="wind", label="Class 2 - Wire",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),
    dict(path="windvfailure/class4pole.csv",  group="wind", label="Class 4 - Pole",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),
    dict(path="windvfailure/class4wire.csv",  group="wind", label="Class 4 - Wire",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),
    dict(path="windvfailure/class5pole.csv",  group="wind", label="Class 5 - Pole",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),
    dict(path="windvfailure/class5wire.csv",  group="wind", label="Class 5 - Wire",  x_name="Wind speed (m/s)", valid_range=(12.5, 40)),

    # --- Ice thickness sweep (Fig 5b), wind speed fixed at 30 m/s ---
    dict(path="icethicknessvfailure/class2pole.csv",   group="ice", label="Class 2 - Pole",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
    dict(path="icethicknessvfailure/class2wire.csv",   group="ice", label="Class 2 - Wire",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
    dict(path="icethicknessvfailure/class4pole.csv",   group="ice", label="Class 4 - Pole",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
    dict(path="icethicknessvfailure/class4wire.csv",   group="ice", label="Class 4 - Wire",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
    dict(path="icethicknessvfailure/class5pole.csv",   group="ice", label="Class 5 - Pole",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
    dict(path="icethicknessvfailure/class5wire.csv",   group="ice", label="Class 5 - Wire",  x_name="Ice thickness (cm)", valid_range=(0.635, 3.81)),
]


# ---------------------------------------------------------------------
# 2. CANDIDATE FUNCTIONAL FORMS
# ---------------------------------------------------------------------

def lognormal_cdf(x, median, beta):
    """Lognormal fragility function: F(x) = Phi[ln(x/median) / beta]"""
    x = np.maximum(x, 1e-9)  # guard against log(0) or negative x
    return norm.cdf(np.log(x / median) / beta)


def logistic_cdf(x, a, b):
    """Logistic fragility function: F(x) = 1 / (1 + exp(-(x-a)/b))"""
    return 1.0 / (1.0 + np.exp(-(x - a) / b))


# ---------------------------------------------------------------------
# 3. FITTING + COMPARISON
# ---------------------------------------------------------------------

def fit_and_compare(x_data, y_data, label):
    """
    Fit both lognormal and logistic CDFs to the data, return fitted
    params and the sum-of-squared-residuals (SSE) for each, and print
    which one wins.
    """
    results = {}

    # --- Fit lognormal ---
    try:
        # initial guesses: median near the midpoint of x, beta ~ 0.3
        p0_lognorm = [np.median(x_data), 0.3]
        popt_ln, _ = curve_fit(lognormal_cdf, x_data, y_data, p0=p0_lognorm,
                                bounds=([1e-6, 1e-3], [np.inf, np.inf]), maxfev=10000)
        y_pred_ln = lognormal_cdf(x_data, *popt_ln)
        sse_ln = np.sum((y_data - y_pred_ln) ** 2)
        results["lognormal"] = dict(params=popt_ln, sse=sse_ln)
    except RuntimeError:
        results["lognormal"] = dict(params=None, sse=np.inf)

    # --- Fit logistic ---
    try:
        p0_logistic = [np.median(x_data), (x_data.max() - x_data.min()) / 10]
        popt_lg, _ = curve_fit(logistic_cdf, x_data, y_data, p0=p0_logistic,
                                maxfev=10000)
        y_pred_lg = logistic_cdf(x_data, *popt_lg)
        sse_lg = np.sum((y_data - y_pred_lg) ** 2)
        results["logistic"] = dict(params=popt_lg, sse=sse_lg)
    except RuntimeError:
        results["logistic"] = dict(params=None, sse=np.inf)

    # --- Report ---
    print(f"\n=== {label} ===")
    if results["lognormal"]["params"] is not None:
        m, b = results["lognormal"]["params"]
        print(f"  Lognormal fit: median={m:.4f}, beta={b:.4f}  -> SSE = {results['lognormal']['sse']:.5f}")
    else:
        print("  Lognormal fit: FAILED to converge")

    if results["logistic"]["params"] is not None:
        a, b = results["logistic"]["params"]
        print(f"  Logistic fit:  a={a:.4f}, b={b:.4f}          -> SSE = {results['logistic']['sse']:.5f}")
    else:
        print("  Logistic fit: FAILED to converge")

    if results["lognormal"]["sse"] < results["logistic"]["sse"]:
        winner = "lognormal"
    else:
        winner = "logistic"
    print(f"  --> Best fit: {winner.upper()}")

    return results, winner


def plot_fit(x_data, y_data, results, label, x_name, valid_range, outpath):
    """Plot the raw digitized points plus both fitted curves."""
    x_smooth = np.linspace(max(x_data.min() * 0.8, 1e-6), x_data.max() * 1.2, 300)

    plt.figure(figsize=(7, 5))
    plt.scatter(x_data, y_data, color="black", zorder=5, label="Digitized data")

    if results["lognormal"]["params"] is not None:
        y_ln = lognormal_cdf(x_smooth, *results["lognormal"]["params"])
        plt.plot(x_smooth, y_ln, "b-",
                  label=f"Lognormal fit (SSE={results['lognormal']['sse']:.4f})")

    if results["logistic"]["params"] is not None:
        y_lg = logistic_cdf(x_smooth, *results["logistic"]["params"])
        plt.plot(x_smooth, y_lg, "r--",
                  label=f"Logistic fit (SSE={results['logistic']['sse']:.4f})")

    # Shade the region outside the paper's validated simulation range
    if valid_range is not None:
        lo, hi = valid_range
        plt.axvspan(x_smooth.min(), lo, color="gray", alpha=0.15)
        plt.axvspan(hi, x_smooth.max(), color="gray", alpha=0.15,
                     label=f"Outside validated range ({lo}-{hi})")

    plt.xlabel(x_name)
    plt.ylabel("Failure probability")
    plt.title(label)
    plt.ylim(-0.02, 1.05)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------

def main():
    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(exist_ok=True)
    summary_rows = {group: [] for group in OUTPUT_DIRS}

    for entry in CSV_FILES:
        path = SCRIPT_DIR / entry["path"]
        if not os.path.exists(path):
            print(f"\n[SKIPPED] File not found: {path.relative_to(SCRIPT_DIR)}")
            continue

        df = pd.read_csv(path, header=None)
        # If the CSV happens to have a header row that got read as data,
        # drop any row that fails to convert to float.
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        x_data = df.iloc[:, 0].to_numpy()
        y_data = df.iloc[:, 1].to_numpy()

        # sort by x for cleaner plotting/fitting
        order = np.argsort(x_data)
        x_data, y_data = x_data[order], y_data[order]

        label = f"{entry['label']} vs {entry['x_name']}"
        results, winner = fit_and_compare(x_data, y_data, label)

        output_dir = OUTPUT_DIRS[entry["group"]]
        outpath = output_dir / f"{path.stem}_fit.png"
        plot_fit(x_data, y_data, results, label, entry["x_name"],
                  entry["valid_range"], outpath)
        print(f"  Plot saved to: {outpath}")

        summary_rows[entry["group"]].append(dict(
            file=path.relative_to(SCRIPT_DIR),
            label=entry["label"],
            x_axis=entry["x_name"],
            winner=winner,
            lognormal_sse=results["lognormal"]["sse"],
            logistic_sse=results["logistic"]["sse"],
            lognormal_params=results["lognormal"]["params"],
            logistic_params=results["logistic"]["params"],
        ))

    # Final summary table
    for group, rows in summary_rows.items():
        print(f"\n\n========== {group.upper()} SUMMARY ==========")
        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            print(summary_df[["label", "x_axis", "winner",
                              "lognormal_sse", "logistic_sse"]]
                  .to_string(index=False))
            summary_path = OUTPUT_DIRS[group] / "fit_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nFull summary (with fitted params) saved to: {summary_path}")
        else:
            print("No files were processed. Check your CSV_FILES paths.")


if __name__ == "__main__":
    main()
