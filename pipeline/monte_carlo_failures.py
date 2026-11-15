"""
monte_carlo_failures.py
=======================
Step 3 of the pipeline: which OpenDSS lines fail in each of N Monte Carlo runs,
for ONE storm on ONE feeder.

    run_fragility.py  ->  monte_carlo_failures.py  ->  data/opendss/<substation>/<feeder>/
    (p_fail per pole       (this file)                     failed_lines_matrix.csv
     and per wire)                                         (read by code_opendsspowerflow/run_sim.py)

Run:
    python pipeline/monte_carlo_failures.py
The feeder and area type come from pipeline/asset_adapter.py (FEEDER_NAME, AREA_TYPE) and
the storm from pipeline/run_fragility.py (WEATHER_SOURCE ...), so every step uses the same
settings. Change them there, not here.

Structure (same pattern as run_fragility.py):
    load_line_map(feeder, area_type)        -> which OpenDSS lines each span sits on
    run_monte_carlo(pole_frag, span_frag, span_lines)
                                            -> matrix, summary   (the work; saves nothing)
    save_and_print(matrix, summary, ...)    -> writes the files below
    main()                                  -> get_weather -> run_fragility -> run_monte_carlo -> save
From other code (e.g. the GUI):
    from pipeline.monte_carlo_failures import load_line_map, run_monte_carlo
    span_lines, _ = load_line_map("1878", "suburban")
    matrix, summary = run_monte_carlo(pole_frag, span_frag, span_lines)

OUTPUTS
  1. data/opendss/<substation>/<feeder>/failed_lines_matrix.csv   (the power-flow input)
         line_name,1,2,3,...,10000
         l(r:p1udt1226-p1udt2266),1,0,0,...
     One row per OpenDSS line in the feeder's span-line map, one column per run.
     1 = the line failed in that run. A column of 0s = nothing failed (unserved load 0).
     The feeder folder is found automatically (any folder data/opendss/*/*--*<feeder>),
     so a new feeder only needs its SMART-DS folder there and its polegen2 outputs.
     Next to it, failed_lines_matrix_info.txt says which storm and settings made it.
  2. pipeline/outputs/monte_carlo/mc_summary_<feeder>_<area>_<storm>_t<threshold>.csv
     One row per run: failed poles, spans and lines (input for the CPDM).

HOW IT WORKS
  1. Only assets with p_fail >= THRESHOLD are sampled (Requirement 1).
  2. Each run, each sampled pole and wire fails if u < p, u ~ Uniform(0, 1) (Kim et al.).
  3. A span is down if its wire failed or either end pole failed. Poles are drawn ONCE
     per run, so a failed pole takes down every span attached to it.
     (p_fail_span_combined is NOT used: it would count a shared pole twice.)
  4. Spans -> OpenDSS lines with polegen2's span_line_map (a span can sit on more than
     one line; each failed line is listed once per run).
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import run_fragility as rf                                                  # noqa: E402
from pipeline.asset_adapter import FEEDER_NAME, AREA_TYPE                    # noqa: E402

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
THRESHOLD = 0.01        # sample assets with p_fail >= this. Requirement 1 says 0.05-1.0:
                        # keep the requirement and this value in step.
N_ITERATIONS = 10_000   # Monte Carlo runs (Objective 1)
SEED = 42               # fixed seed: the same settings always give the same runs

POLEGEN_OUTPUT_DIR = REPO_ROOT / "code_polegen2" / "outputs"
OPENDSS_DIR = REPO_ROOT / "data" / "opendss"
OUTPUT_DIR = HERE / "outputs" / "monte_carlo"
POWERFLOW_INPUT_NAME = "failed_lines_matrix.csv"     # the name run_sim.py reads


# =============================================================================
# 1. INPUTS
# =============================================================================
def load_line_map(feeder=FEEDER_NAME, area_type=AREA_TYPE):
    """Which OpenDSS lines each span sits on, from polegen2's span_line_map.
    Returns ({span_id: [line names]}, path of the map)."""
    path = (POLEGEN_OUTPUT_DIR / f"feeder_{feeder}" / "mapping"
            / f"feeder_{feeder}_span_line_map_{area_type}.csv")
    if not path.exists():
        raise FileNotFoundError(f"No span-line map for feeder {feeder} ({area_type}): {path}\n"
                                "Run code_polegen2/main.py for this feeder first.")
    m = pd.read_csv(path)
    span_lines = m.groupby("span_id")["line_name"].apply(lambda s: sorted(set(s))).to_dict()
    return span_lines, path


def feeder_folder(feeder=FEEDER_NAME):
    """The feeder's SMART-DS folder, data/opendss/<substation>/<substation>--<...><feeder>."""
    matches = sorted(p for p in OPENDSS_DIR.glob(f"*/*--*{feeder}") if p.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one folder for feeder {feeder} in {OPENDSS_DIR}, "
                                f"found {[str(p) for p in matches]}")
    return matches[0]


# =============================================================================
# 2. MONTE CARLO
# =============================================================================
def run_monte_carlo(pole_frag, span_frag, span_lines,
                    threshold=THRESHOLD, n=N_ITERATIONS, seed=SEED):
    """Fragility in -> failed lines per run out. Saves nothing.

    pole_frag  : run_fragility output (pole_id, p_fail, ...)
    span_frag  : run_fragility output (span_id, start_pole, end_pole, p_fail_wire, ...)
    span_lines : {span_id: [OpenDSS line names]} from load_line_map()
    Returns
      matrix  : DataFrame, index = line_name (every line in span_lines), columns 1..n,
                values 1 (failed) / 0
      summary : DataFrame, one row per run
    """
    if not 0.01 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.01 and 1.0")
    pole_ids = pole_frag["pole_id"].to_numpy()
    pole_index = {pid: i for i, pid in enumerate(pole_ids)}
    for col in ("start_pole", "end_pole"):
        unknown = set(span_frag[col]) - set(pole_index)
        if unknown:
            raise ValueError(f"spans refer to poles that are not in pole_frag, e.g. {sorted(unknown)[:5]}")
    span_ids = span_frag["span_id"].to_numpy()
    no_line = [s for s in span_ids if s not in span_lines]
    if no_line:
        raise ValueError(f"{len(no_line)} spans have no OpenDSS line in the map, e.g. {no_line[:5]}")

    # 1. assets above the threshold
    p_pole = pole_frag["p_fail"].to_numpy(float)
    p_wire = span_frag["p_fail_wire"].to_numpy(float)
    pole_flag = np.flatnonzero(p_pole >= threshold)
    wire_flag = np.flatnonzero(p_wire >= threshold)

    # 2. draw: fail if u < p
    rng = np.random.default_rng(seed)
    pole_failed = rng.random((n, len(pole_flag))) < p_pole[pole_flag]
    wire_failed = rng.random((n, len(wire_flag))) < p_wire[wire_flag]

    # 3. a span is down if its wire or either end pole failed
    start_i = span_frag["start_pole"].map(pole_index).to_numpy()
    end_i = span_frag["end_pole"].map(pole_index).to_numpy()
    spans_of_pole = {i: np.flatnonzero((start_i == i) | (end_i == i)) for i in pole_flag}

    # 4. spans -> lines
    lines = sorted({x for names in span_lines.values() for x in names})
    row_of = {name: r for r, name in enumerate(lines)}
    span_rows = [np.array([row_of[x] for x in span_lines[s]]) for s in span_ids]
    failed = np.zeros((len(lines), n), dtype=np.uint8)

    summary = []
    for k in range(n):
        down = np.zeros(len(span_ids), dtype=bool)
        failed_poles = pole_flag[pole_failed[k]]
        for i in failed_poles:
            down[spans_of_pole[i]] = True
        down[wire_flag[wire_failed[k]]] = True
        down_idx = np.flatnonzero(down)
        rows = np.unique(np.concatenate([span_rows[j] for j in down_idx])) if len(down_idx) else []
        failed[rows, k] = 1
        summary.append({
            "scenario": k + 1,
            "n_failed_poles": len(failed_poles),
            "n_failed_spans": len(down_idx),
            "n_failed_lines": len(rows),
            "failed_poles": ";".join(pole_ids[failed_poles]),
            "failed_spans": ";".join(span_ids[down_idx]),
            "failed_lines": ";".join(lines[r] for r in rows),
        })

    matrix = pd.DataFrame(failed, index=pd.Index(lines, name="line_name"),
                          columns=range(1, n + 1))
    summary = pd.DataFrame(summary)
    summary.attrs.update(n_poles_sampled=len(pole_flag), n_wires_sampled=len(wire_flag),
                         n_poles=len(p_pole), n_wires=len(p_wire),
                         rate_error=float(np.abs(pole_failed.mean(axis=0) - p_pole[pole_flag]).max())
                         if len(pole_flag) else 0.0)
    return matrix, summary


# =============================================================================
# 3. SAVE
# =============================================================================
def write_matrix(matrix, path):
    """line_name,1,2,...,n then one row of 0/1 per line (fast for 10,000 columns)."""
    cells = np.where(matrix.to_numpy() == 1, "1", "0")
    with open(path, "w", newline="") as f:
        f.write("line_name," + ",".join(str(c) for c in matrix.columns) + "\r\n")
        for name, row in zip(matrix.index, cells):
            f.write(name + "," + ",".join(row) + "\r\n")


def save_and_print(matrix, summary, feeder, area_type, storm_tag,
                   threshold=THRESHOLD, n=N_ITERATIONS, seed=SEED):
    """Write the power-flow input (+ info file) and the summary, and print a short report."""
    tag = f"{feeder}_{area_type}_{storm_tag}_t{threshold * 100:g}"
    folder = feeder_folder(feeder)
    pf_path = folder / POWERFLOW_INPUT_NAME
    try:
        write_matrix(matrix, pf_path)
    except PermissionError:
        sys.exit(f"Can't write {pf_path}\nIt is probably open in Excel. Close it and run again.")
    (folder / "failed_lines_matrix_info.txt").write_text(
        f"{POWERFLOW_INPUT_NAME} was made by pipeline/monte_carlo_failures.py on "
        f"{datetime.now():%Y-%m-%d %H:%M}\n"
        f"  feeder {feeder} ({area_type}), storm {storm_tag}\n"
        f"  threshold {threshold:g}, {n} runs, seed {seed}\n"
        f"  1 = the line failed in that run (a failed pole takes down every span on it).\n"
        f"  Summary: pipeline/outputs/monte_carlo/mc_summary_{tag}.csv\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sum_path = OUTPUT_DIR / f"mc_summary_{tag}.csv"
    summary.to_csv(sum_path, index=False)

    a = summary.attrs
    empty = int((summary["n_failed_lines"] == 0).sum())
    print(f"Feeder {feeder} ({area_type}), storm {storm_tag}, threshold {threshold:g}, "
          f"{n:,} runs (seed {seed})")
    print(f"  sampled: {a['n_poles_sampled']} of {a['n_poles']} poles, "
          f"{a['n_wires_sampled']} of {a['n_wires']} wires")
    print(f"  runs with no failures: {empty:,} ({empty / n:.1%})")
    print(f"  per run: failed poles mean {summary.n_failed_poles.mean():.1f} "
          f"(max {summary.n_failed_poles.max()}), failed lines mean "
          f"{summary.n_failed_lines.mean():.1f} (max {summary.n_failed_lines.max()})")
    if a["n_poles_sampled"]:
        print(f"  check: each sampled pole failed about as often as its p_fail "
              f"(largest difference {a['rate_error']:.4f})")
    print(f"Saved:\n  {pf_path.relative_to(REPO_ROOT)}   <- power-flow input"
          f"\n  {sum_path.relative_to(REPO_ROOT)}")


def main():
    """The wiring: get weather -> run fragility -> run Monte Carlo -> save and print."""
    ice_mm, wind_ms = rf.get_weather()
    pole_frag, span_frag = rf.run_fragility(ice_mm, wind_ms)
    span_lines, _ = load_line_map(FEEDER_NAME, AREA_TYPE)
    matrix, summary = run_monte_carlo(pole_frag, span_frag, span_lines)
    save_and_print(matrix, summary, FEEDER_NAME, AREA_TYPE, rf.scenario_tag(ice_mm, wind_ms))


if __name__ == "__main__":
    main()