"""
MONTE CARLO FAILURE SIMULATION  (Subsystem 4, Block A)
======================================================
Takes the pole and span fragility CSVs from the Vulnerability Analysis,
runs N Monte Carlo iterations, and writes the failed lines of every
iteration in two layouts:

  1. failed_lines_mc.csv  (same format as Melani's power-flow input file)
         line_name
         <line failed in scenario 1>
         <line failed in scenario 1>
         ---
         <line failed in scenario 2>
         ...

  2. failed_lines_matrix.csv  (one row per line, one column per iteration)
         line_name,1,2,3,...,10000
         l(r:p1udt1226-p1udt2266),1,0,0,...
         l(r:p1udt1226-p1udt4516),0,0,1,...
     1 = the line failed in that iteration, 0 = it did not.

Run (Windows / VS Code terminal):
    python monte_carlo_failures.py --poles pole_fragility.csv --spans span_fragility.csv

OUTPUT FOLDER
-------------
All files go into a folder called "Monte Carlo Output" next to this script
(Capstone/pipeline/Monte Carlo Output/), made if it does not exist. They are
named after the pole fragility file and the threshold, so runs for different
storms do not overwrite each other. For pole_fragility_1878_suburban_wind50ms.csv
with --threshold 0.01:
    failed_lines_1878_suburban_wind50ms_t1.csv         (Melani's --- format)
    failed_lines_matrix_1878_suburban_wind50ms_t1.csv  (lines x iterations, 1/0)
    mc_summary_1878_suburban_wind50ms_t1.csv           (one row per iteration)

Options:
    --threshold 0.05   only assets with p_fail >= threshold are sampled (0.01-1.0; Req. 1 is 5-100%)
    --n 10000          number of Monte Carlo iterations (Objective 1)
    --seed 42          fixed random seed, so a run can be repeated exactly
    --line-map map.csv CSV that links spans to OpenDSS line names (see LINE NAMES)
    --out-dir FOLDER   save the files somewhere else
    --out NAME.csv     own name for the Melani-format file
    --matrix NAME.csv  own name for the line x iteration table of 1s and 0s
    --summary NAME.csv own name for the one-row-per-iteration check file
    --lines-dss Lines.dss  also give the matrix a row (all 0s) for every
                       feeder line that has no poles (underground cables, pad
                       switches, fuses). Default: only the overhead lines in
                       the line map (the ones that can fail)
    --skip-empty       leave out iterations where nothing failed (--out file only;
                       the matrix always has every iteration)

HOW IT WORKS
------------
1. Flag the assets above the threshold (Requirement 1):
     poles  with p_fail      >= threshold
     wires  with p_fail_wire >= threshold
2. For every iteration k and flagged asset i, draw u(i,k) ~ Uniform(0,1).
   The asset fails if u(i,k) < p(i)  (Kim et al. [4]).
3. A span is down if its wire failed OR either of its two poles failed.
   Poles are sampled ONCE per iteration, so when a pole fails every span
   attached to it goes down together. (Sampling each span separately with
   p_fail_span_combined would count the same pole failure twice and lose
   that link, so the combined column is NOT used here.)
   Poles are linked to spans through the start_pole / end_pole columns of
   the span fragility CSV.
4. The failed spans of each iteration are written out by span_id (default).

LINE NAMES (optional)
---------------------
To write OpenDSS line names instead of span IDs, give --line-map. For feeder
1878 this is code_polegen2/outputs/feeder_1878_span_line_map_suburban.csv
(columns span_id, line_name; a span can appear on several rows when it sits
on more than one OpenDSS line, and all of them are written). A 'section_id'
column instead of 'span_id' also works. A line is listed once per iteration
even if several of its spans failed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POLE_COLS = {"pole_id", "section_id", "p_fail"}
SPAN_COLS = {"span_id", "start_pole", "end_pole", "p_fail_wire"}
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "Monte Carlo Output"


def output_paths(args):
    """Where the three output files go (folder is made if missing)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.poles).stem
    if stem.startswith("pole_fragility_"):
        stem = stem[len("pole_fragility_"):]
    tag = f"{stem}_t{args.threshold * 100:g}"          # e.g. 1878_suburban_wind50ms_t1
    paths = {
        "out": out_dir / (args.out or f"failed_lines_{tag}.csv"),
        "matrix": out_dir / (args.matrix or f"failed_lines_matrix_{tag}.csv"),
        "summary": out_dir / (args.summary or f"mc_summary_{tag}.csv"),
    }
    for p in paths.values():                 # fail early if Excel has a file open
        if p.exists():
            try:
                with open(p, "a"):
                    pass
            except PermissionError:
                sys.exit(f"Can't write {p}\nIt is probably open in Excel. Close it and run again.")
    return paths


def load(path, needed, what):
    df = pd.read_csv(path)
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"{what} file {path} is missing columns: {sorted(missing)}")
    return df


def span_line_names(spans, poles, line_map_path):
    """Return, for every span (same order as spans), the LIST of names to write
    when that span fails. One span can sit on more than one OpenDSS line."""
    if line_map_path is None:
        print("No --line-map given: writing the failed SPAN IDs (e.g. 1878_S00342).")
        return [[s] for s in spans["span_id"]]

    m = pd.read_csv(line_map_path)
    if "line_name" not in m.columns:
        sys.exit("--line-map file needs a 'line_name' column")
    if "span_id" in m.columns:
        key, ids = "span_id", spans["span_id"]
    elif "section_id" in m.columns:
        key = "section_id"
        ids = spans["end_pole"].map(poles.set_index("pole_id")["section_id"])
    else:
        sys.exit("--line-map file needs a 'span_id' or 'section_id' column")
    lookup = m.groupby(key)["line_name"].apply(lambda s: sorted(set(s))).to_dict()
    names = [lookup.get(i) for i in ids]
    missing = [s for s, n in zip(spans["span_id"], names) if not n]
    if missing:
        sys.exit(f"{len(missing)} spans have no line name in the map, e.g. {missing[:5]}")
    n_multi = sum(len(n) > 1 for n in names)
    print(f"Line map: {len(spans)} spans -> {len({x for n in names for x in n})} OpenDSS lines "
          f"({n_multi} spans sit on more than one line).")
    return names


def write_matrix(path, iter_lines, line_name, lines_dss=None):
    """One row per line, one column per iteration: 1 = failed, 0 = not failed.
    Returns the number of rows written."""
    rows = {x for names in line_name for x in names}
    if lines_dss:
        import re
        with open(lines_dss) as f:
            in_dss = re.findall(r"^\s*new\s+line\.(\S+)", f.read(),
                                flags=re.IGNORECASE | re.MULTILINE)
        have = {r.lower() for r in rows}
        rows |= {x for x in in_dss if x.lower() not in have}
    rows = sorted(rows)
    row_of = {name: i for i, name in enumerate(rows)}

    n = len(iter_lines)
    failed = np.zeros((len(rows), n), dtype=bool)
    for k, names in enumerate(iter_lines):
        for x in names:
            failed[row_of[x], k] = True

    cells = np.where(failed, "1", "0")
    with open(path, "w", newline="") as f:
        f.write("line_name," + ",".join(str(k) for k in range(1, n + 1)) + "\r\n")
        for name, row in zip(rows, cells):
            f.write(name + "," + ",".join(row) + "\r\n")
    if n + 1 > 16_384:
        print(f"  NOTE: {n:,} iterations is more columns than Excel can show (16,384); "
              f"the file is still fine for Python.")
    return len(rows)


def run(args):
    if not 0.01 <= args.threshold <= 1.0:
        sys.exit("Threshold must be between 0.01 and 1.0 (1-100%).")
    if args.threshold < 0.05:
        print(f"NOTE: threshold {args.threshold:.0%} is below Requirement 1 (5-100%).")

    poles = load(args.poles, POLE_COLS, "Pole")
    spans = load(args.spans, SPAN_COLS, "Span")
    paths = output_paths(args)
    pole_index = {pid: i for i, pid in enumerate(poles["pole_id"])}
    for col in ("start_pole", "end_pole"):
        unknown = set(spans[col]) - set(pole_index)
        if unknown:
            sys.exit(f"Span file refers to poles not in the pole file, e.g. {sorted(unknown)[:5]}")

    line_name = span_line_names(spans, poles, args.line_map)

    # 1. flag assets above the threshold ---------------------------------
    p_pole = poles["p_fail"].to_numpy(float)
    p_wire = spans["p_fail_wire"].to_numpy(float)
    pole_flag = np.flatnonzero(p_pole >= args.threshold)
    wire_flag = np.flatnonzero(p_wire >= args.threshold)
    print(f"Threshold {args.threshold:.0%}: {len(pole_flag)} of {len(poles)} poles "
          f"and {len(wire_flag)} of {len(spans)} wires are sampled.")
    if len(pole_flag) + len(wire_flag) == 0:
        print("Nothing is above the threshold, so no asset can fail. "
              "Lower the threshold or check the fragility files.")

    # 2. draw random numbers: fail if u < p ------------------------------
    rng = np.random.default_rng(args.seed)
    pole_failed = rng.random((args.n, len(pole_flag))) < p_pole[pole_flag]
    wire_failed = rng.random((args.n, len(wire_flag))) < p_wire[wire_flag]

    # 3. a span is down if its wire or either end pole failed ------------
    start_i = spans["start_pole"].map(pole_index).to_numpy()
    end_i = spans["end_pole"].map(pole_index).to_numpy()
    # for each flagged pole, the spans attached to it
    spans_of_pole = {i: np.flatnonzero((start_i == i) | (end_i == i)) for i in pole_flag}

    out_lines = ["line_name"]
    summary = []
    iter_lines = []      # failed lines of every iteration, for the matrix
    written = 0
    for k in range(args.n):
        down = np.zeros(len(spans), dtype=bool)
        failed_poles = pole_flag[pole_failed[k]]
        for i in failed_poles:
            down[spans_of_pole[i]] = True
        down[wire_flag[wire_failed[k]]] = True

        span_ids = spans["span_id"].to_numpy()[down]
        # 4. one entry per OpenDSS line, in a fixed order
        lines = sorted({x for j in np.flatnonzero(down) for x in line_name[j]})
        iter_lines.append(lines)

        summary.append({
            "scenario": k + 1,
            "n_failed_poles": len(failed_poles),
            "n_failed_spans": int(down.sum()),
            "n_failed_lines": len(lines),
            "failed_poles": ";".join(poles["pole_id"].to_numpy()[failed_poles]),
            "failed_spans": ";".join(span_ids),
            "failed_lines": ";".join(lines),
        })
        if not lines and args.skip_empty:
            continue
        if written:
            out_lines.append("---")
        out_lines.extend(lines)
        written += 1

    with open(paths["out"], "w", newline="") as f:
        f.write("\r\n".join(out_lines) + "\r\n")     # Windows line endings, like Melani's file
    summ = pd.DataFrame(summary)
    summ.to_csv(paths["summary"], index=False)
    n_rows = write_matrix(paths["matrix"], iter_lines, line_name, args.lines_dss)

    # checks ---------------------------------------------------------------
    empty = int((summ["n_failed_lines"] == 0).sum())
    print(f"\n{args.n:,} iterations (seed {args.seed}). Files saved in {paths['out'].parent}:")
    print(f"  {paths['out'].name}: {written:,} scenarios (--- format)")
    print(f"  {paths['matrix'].name}: {n_rows:,} lines x {args.n:,} iterations (1 = failed)")
    print(f"  {paths['summary'].name}: one row per iteration")
    print(f"  iterations with no failures: {empty:,} ({empty / args.n:.1%})"
          + ("  <- left out (--skip-empty)" if args.skip_empty else ""))
    print(f"  failed poles per iteration: mean {summ.n_failed_poles.mean():.1f}, "
          f"max {summ.n_failed_poles.max()}")
    print(f"  entries written per iteration (spans, or lines with --line-map): "
          f"mean {summ.n_failed_lines.mean():.1f}, "
          f"max {summ.n_failed_lines.max()}")
    if len(pole_flag):
        rate = pole_failed.mean(axis=0)
        err = np.abs(rate - p_pole[pole_flag]).max()
        print(f"  check: each flagged pole failed about as often as its p_fail "
              f"(largest difference {err:.4f})")


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo failure scenarios in Melani's line_name format")
    ap.add_argument("--poles", required=True, help="pole fragility CSV")
    ap.add_argument("--spans", required=True, help="span fragility CSV")
    ap.add_argument("--threshold", type=float, default=0.05, help="0.01-1.0 (default 0.05)")
    ap.add_argument("--n", type=int, default=10_000, help="iterations (default 10000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--line-map", default=None, help="CSV: span_id or section_id -> line_name")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help='folder for all outputs (default: "Monte Carlo Output" next to this script)')
    ap.add_argument("--out", default=None, help="name of the --- format file (default: from the storm)")
    ap.add_argument("--matrix", default=None,
                    help="name of the line x iteration table, 1 = failed (default: from the storm)")
    ap.add_argument("--summary", default=None, help="name of the summary file (default: from the storm)")
    ap.add_argument("--lines-dss", default=None,
                    help="feeder Lines.dss: add a row of 0s for every line with no poles")
    ap.add_argument("--skip-empty", action="store_true",
                    help="leave out iterations with no failures (they still count in the summary)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
