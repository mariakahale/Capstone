"""
cpdm_analysis.py
================
Criticality Analysis (Subsystem 6): which lines and poles are most often failed in the
Monte Carlo runs that end in a large outage.

    monte_carlo_failures.py  ->  run_sim.py (power flow)  ->  cpdm_analysis.py (this file)
    failed_lines_matrix.csv      cpdm_input.csv                pipeline/outputs/cpdm/

CPDM (Kim et al., Eq. 8), for one element e (a line or a pole) and one outage state:

    CPDM(e | state) = P(F_e | F_sys = state)  ~=  N_e,state / N_state
        N_state   = number of runs whose unserved load falls in the state
        N_e,state = number of those runs in which e failed

Also printed (Requirement 7), over the runs used:
    L_mean = (1/N) * sum(L_k)          L_95 = L_(ceil(0.95 N)), runs sorted low -> high

HOW IT RUNS
  run_sim.py calls run_cpdm() at the end, for the feeder it just ran, with the two files it
  just used:
      run_cpdm(FEEDER_NAME, cpdm_csv, csv_path)
  The feeder is picked once, in run_sim.py (python code_opendsspowerflow/run_sim.py --feeder 1878),
  so the CPDM always uses that feeder's power-flow results and Monte Carlo files.

INPUTS (given by run_sim.py)
  cpdm_input.csv            unserved load of every run that converged
                            (code_opendsspowerflow/outputs/feeder_<F>/)
  failed_lines_matrix.csv   which lines failed in each run (data/opendss/<substation>/<feeder>/)
  failed_lines_matrix_info.txt, next to the matrix, names the Monte Carlo summary, which says
  which POLES failed in each run (only needed for the pole CPDM).
  run_cpdm() stops with an error if the two files disagree on how many lines failed in a run.

STATES (design decision: change STATES below)
  Unserved load as a share of the pre-fault feeder load, in thirds:
  Low < 1/3, Medium 1/3 to 2/3, High >= 2/3. Kim et al.'s state ranges are only examples.

OUTPUTS (pipeline/outputs/cpdm/; <tag> says which feeder and storm, e.g. 1878_suburban_wind50ms_t1)
  cpdm_lines_<tag>.csv and cpdm_poles_<tag>.csv, one row per element, sorted by the CPDM
  in the worst state:
      runs_failed       runs in which it failed (of the runs used)
      failed_share      runs_failed / runs used
      CPDM_<state>      the CPDM in each state
      <worst>_vs_all    CPDM in the worst state / failed_share. Above 1: this element fails
                        more often in big outages than in general, so its failure goes with
                        big outages (a candidate for hardening).
  cpdm_top10_lines_<tag>.png and cpdm_top10_poles_<tag>.png
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
# (name, lower share, upper share) of the pre-fault load; lower bound included
STATES = [
    ("Low", 0.0, 1 / 3),
    ("Medium", 1 / 3, 2 / 3),
    ("High", 2 / 3, 1.0),
]
TOP_N = 10

OUTPUT_DIR = HERE / "outputs" / "cpdm"


# =============================================================================
# 1. INPUTS
# =============================================================================
def load_inputs(cpdm_input_csv, matrix_csv, feeder):
    """Returns (runs, line_names, line_failed, info)
      runs        : cpdm_input.csv (one row per converged run)
      line_names  : the OpenDSS lines in failed_lines_matrix.csv
      line_failed : 0/1 array, lines x runs (column k = run k + 1)
      info        : {"tag": ..., "summary": path of the Monte Carlo summary or None}"""
    matrix_csv = Path(matrix_csv)
    runs = pd.read_csv(cpdm_input_csv)
    matrix = pd.read_csv(matrix_csv, index_col=0)          # first column = line names
    line_failed = matrix.to_numpy(dtype=np.uint8)

    info = {"tag": f"feeder_{feeder}", "summary": None}
    info_path = matrix_csv.parent / "failed_lines_matrix_info.txt"
    if info_path.exists():
        m = re.search(r"Summary: (.+)", info_path.read_text())
        if m:
            info["summary"] = REPO_ROOT / m.group(1).strip()
            info["tag"] = info["summary"].stem.replace("mc_summary_", "")
    return runs, list(matrix.index), line_failed, info


def check_same_run(runs, line_failed):
    """cpdm_input.csv must come from this failed_lines_matrix.csv."""
    counts = line_failed.sum(axis=0)
    k = runs["scenario"].to_numpy() - 1
    if k.max() >= line_failed.shape[1] or (counts[k] != runs["n_lines_failed"].to_numpy()).any():
        raise ValueError("cpdm_input.csv and failed_lines_matrix.csv disagree on how many lines "
                         "failed in some runs. Either cpdm_input.csv is from an older run of "
                         "run_sim.py, or some lines in the matrix are not in the OpenDSS model "
                         "(run_sim.py skips those).")


def load_pole_failures(summary_path, line_failed):
    """Pole failures per run from the Monte Carlo summary: (pole_names, poles x runs 0/1)."""
    if summary_path is None or not Path(summary_path).exists():
        return None, None
    summary = pd.read_csv(summary_path)
    if (len(summary) != line_failed.shape[1]
            or (summary["n_failed_lines"].to_numpy() != line_failed.sum(axis=0)).any()):
        print("NOTE: the Monte Carlo summary does not match the matrix; no pole CPDM.")
        return None, None
    lists = summary["failed_poles"].fillna("").str.split(";")
    names = sorted({p for poles in lists for p in poles if p})
    row = {p: i for i, p in enumerate(names)}
    pole_failed = np.zeros((len(names), len(summary)), dtype=np.uint8)
    for k, poles in enumerate(lists):
        for p in poles:
            if p:
                pole_failed[row[p], k] = 1
    return names, pole_failed


# =============================================================================
# 2. CPDM
# =============================================================================
def state_of(share):
    """The STATES name for one share of the pre-fault load (1.0 and above -> worst)."""
    for name, lo, hi in STATES:
        if lo <= share < hi:
            return name
    return STATES[-1][0] if share >= STATES[-1][2] else STATES[0][0]


def cpdm_table(names, failed, states):
    """CPDM of every element in every state.
    names  : element names (lines or poles)
    failed : 0/1 array, elements x runs (only the runs used)
    states : state name of each of those runs
    Returns a DataFrame sorted by the CPDM in the worst state."""
    states = np.asarray(states)
    n_runs = len(states)
    table = pd.DataFrame({"element": names,
                          "runs_failed": failed.sum(axis=1),
                          "failed_share": failed.sum(axis=1) / n_runs})
    for name, _, _ in STATES:
        in_state = states == name
        table[f"CPDM_{name}"] = (failed[:, in_state].sum(axis=1) / in_state.sum()
                                 if in_state.any() else np.nan)
    worst = STATES[-1][0]
    with np.errstate(divide="ignore", invalid="ignore"):
        table[f"{worst}_vs_all"] = table[f"CPDM_{worst}"] / table["failed_share"]
    table = table[table["runs_failed"] > 0]                 # elements that never failed
    return table.sort_values([f"CPDM_{worst}", "runs_failed"], ascending=False).reset_index(drop=True)


def unserved_summary(unserved_kw):
    """Mean and 95th percentile (L_(ceil(0.95 N))) of the unserved load."""
    srt = np.sort(unserved_kw)
    l95 = srt[math.ceil(0.95 * len(srt)) - 1]
    return float(np.mean(unserved_kw)), float(l95)


# =============================================================================
# 3. CHART
# =============================================================================
def chart(table, kind, path, n_worst, subtitle):
    """Top TOP_N elements by CPDM in the worst state, saved as a PNG.
    Draws on a matplotlib Figure directly (not pyplot), so no window opens and the
    plots of run_sim.py are not affected."""
    from matplotlib.figure import Figure

    ink, ink2, muted, grid, axis = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, orange = "#2a78d6", "#eb6834"
    worst, lo = STATES[-1][0], STATES[-1][1]
    top = table.head(TOP_N).iloc[::-1]
    y = np.arange(len(top))

    fig = Figure(figsize=(8, 0.45 * len(top) + 1.8), dpi=200)
    ax = fig.subplots()
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(y, top[f"CPDM_{worst}"], height=0.55, color=blue,
            label=f"CPDM: failed in this share of {worst}-loss runs")
    ax.scatter(top["failed_share"], y, s=42, color=orange, zorder=3, edgecolor="#fcfcfb",
               linewidth=1.5, label="Failed in this share of all runs used")
    for yi, v, dot in zip(y, top[f"CPDM_{worst}"], top["failed_share"]):
        ax.text(max(v, dot + 0.012) + 0.015, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=8, color=ink2)
    ax.set_yticks(y, top["element"], fontsize=8, color=ink)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Probability", fontsize=9, color=ink2)
    ax.xaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=8)
    ax.set_title(f"Top {len(top)} {kind} by CPDM in the {worst} state "
                 f"(at least {lo:.0%} of load lost, {n_worst} runs)\n{subtitle}",
                 fontsize=10, color=ink, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


# =============================================================================
# 4. RUN (called by run_sim.py)
# =============================================================================
def run_cpdm(feeder, cpdm_input_csv, matrix_csv, output_dir=OUTPUT_DIR):
    """CPDM for one feeder, from the power-flow results and the Monte Carlo matrix run_sim.py
    just used. Prints the results, saves the tables and charts in output_dir and returns
    {"lines": table, "poles": table or None}."""
    runs, line_names, line_failed, info = load_inputs(cpdm_input_csv, matrix_csv, feeder)
    n_all = line_failed.shape[1]
    if runs.empty:
        print(f"\nCPDM for feeder {feeder}: none of the {n_all:,} runs converged, nothing to do.")
        return {"lines": None, "poles": None}
    check_same_run(runs, line_failed)
    used = runs["scenario"].to_numpy() - 1                   # columns of the runs used
    prefault = runs["prefault_load_kw"].to_numpy(float)
    unserved = runs["delta_unserved_kw"].to_numpy(float)
    states = np.array([state_of(s) for s in unserved / prefault])

    mean_kw, l95_kw = unserved_summary(unserved)
    print(f"\nCPDM for feeder {feeder} ({info['tag']}): {len(runs):,} of {n_all:,} runs used "
          f"({n_all - len(runs):,} did not converge)")
    print(f"Unserved load: mean {mean_kw:,.1f} kW ({mean_kw / prefault.mean():.1%}), "
          f"95th percentile {l95_kw:,.1f} kW ({l95_kw / prefault.mean():.1%}) "
          f"of {prefault.mean():,.1f} kW")
    for name, lo, hi in STATES:
        k = int((states == name).sum())
        print(f"  {name:<7} {lo:.0%}-{hi:.0%} of the load lost: {k:,} runs ({k / len(runs):.1%})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worst = STATES[-1][0]
    n_worst = int((states == worst).sum())
    pole_names, pole_failed = load_pole_failures(info["summary"], line_failed)
    tables = {}
    for kind, names, failed in (("lines", line_names, line_failed), ("poles", pole_names, pole_failed)):
        if names is None:
            tables[kind] = None
            continue
        table = cpdm_table(names, failed[:, used], states)
        csv = output_dir / f"cpdm_{kind}_{info['tag']}.csv"
        table.to_csv(csv, index=False, float_format="%.4f")
        print(f"\nTop {TOP_N} {kind} by CPDM in the {worst} state ({n_worst} runs) -> {csv.name}")
        print(table.head(TOP_N).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if n_worst:
            png = output_dir / f"cpdm_top10_{kind}_{info['tag']}.png"
            chart(table, kind, png, n_worst, f"{info['tag']}  |  {len(runs):,} runs used")
            print(f"chart -> {png.name}")
        tables[kind] = table
    print(f"\nCPDM saved in {output_dir}")
    return tables


if __name__ == "__main__":
    print("The CPDM runs at the end of the power flow, for the feeder you pick there:\n"
          "    python code_opendsspowerflow/run_sim.py --feeder 1878")
