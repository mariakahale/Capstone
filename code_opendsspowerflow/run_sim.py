
import os, sys, csv, re, argparse
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import opendssdirect as dss

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENDSS_DATA_DIR = os.path.join(REPO_ROOT, "data", "opendss")

# Registry of available feeders: feeder_id -> path relative to data/opendss/
AVAILABLE_FEEDERS = {
    "3777": os.path.join("p48rhs1_25", "p48rhs1_25--p48rdt3777"),
    "1878": os.path.join("p1uhs0_1247", "p1uhs0_1247--p1udt1878"),
}


def select_feeder():
    """Select feeder via CLI argument (--feeder) or interactive prompt."""
    parser = argparse.ArgumentParser(description="Run OpenDSS contingency simulation.")
    parser.add_argument(
        "-f", "--feeder",
        choices=list(AVAILABLE_FEEDERS.keys()),
        help=f"Feeder ID to run ({', '.join(AVAILABLE_FEEDERS.keys())})"
    )
    args, _ = parser.parse_known_args()

    if args.feeder:
        return args.feeder

    print("\nAvailable Feeders:")
    for fid, rel_path in AVAILABLE_FEEDERS.items():
        print(f"  - {fid}  ({rel_path})")

    while True:
        choice = input("\nEnter feeder ID to run: ").strip()
        if choice in AVAILABLE_FEEDERS:
            return choice
        print(f"Invalid selection '{choice}'. Choose from: {', '.join(AVAILABLE_FEEDERS.keys())}")


FEEDER_NAME = select_feeder()
FEEDER_DIR = os.path.join(OPENDSS_DATA_DIR, AVAILABLE_FEEDERS[FEEDER_NAME])

script_dir = os.path.dirname(os.path.abspath(__file__))
# Store outputs in a feeder-specific subfolder so feeders don't overwrite each other
OUTPUT_DIR = os.path.join(script_dir, "outputs", f"feeder_{FEEDER_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

master_path = os.path.join(FEEDER_DIR, "Master_snapshot.dss")
csv_path = os.path.join(FEEDER_DIR, "failed_lines_matrix.csv")
results_csv = os.path.join(OUTPUT_DIR, "scenario_results.csv")
cpdm_csv = os.path.join(OUTPUT_DIR, "cpdm_input.csv")
plot_png = os.path.join(OUTPUT_DIR, "unserved_load_analysis.png")
worst_csv = os.path.join(OUTPUT_DIR, "worst_scenarios.csv")
TOP_N_WORST = 20


def make_snapshot_files():
    """Read Master.dss & Loads.dss from FEEDER_DIR and write snapshots to OUTPUT_DIR."""
    master_src = os.path.join(FEEDER_DIR, "Master.dss")
    loads_src = os.path.join(FEEDER_DIR, "Loads.dss")
    master_dst = os.path.join(FEEDER_DIR, "Master_snapshot.dss")
    loads_dst = os.path.join(FEEDER_DIR, "Loads_snapshot.dss")

    if not os.path.exists(loads_dst) and os.path.exists(loads_src):
        with open(loads_src, "r") as f:
            text = f.read()
        text = re.sub(r"\s+(yearly|daily|duty)=\S+", "", text, flags=re.I)
        with open(loads_dst, "w") as f:
            f.write(text)

    if not os.path.exists(master_dst):
        if not os.path.exists(master_src):
            sys.exit(f"Master.dss not found in {FEEDER_DIR}")
        skip = ("redirect loadshapes", "solve", "export", "plot", "show",
                "new monitor", "new energymeter")
        out = []
        with open(master_src, "r") as f:
            for line in f:
                low = line.strip().lower()
                if low.startswith(skip):
                    out.append("// " + line)
                elif low.startswith("redirect loads.dss"):
                    out.append(f'Redirect "{loads_dst}"\n')
                elif low.startswith("redirect "):
                    rel_file = line.strip().split(None, 1)[1].strip('"\'')
                    abs_file = os.path.join(FEEDER_DIR, rel_file)
                    out.append(f'Redirect "{abs_file}"\n')
                else:
                    out.append(line)
        with open(master_dst, "w") as f:
            f.writelines(out)
        print(f"[{FEEDER_NAME}] Created Master_snapshot.dss and Loads_snapshot.dss")

def base_bus(bus_str):
    """Strip phase nodes (e.g., 'p1ulv42.1.2' -> 'p1ulv42')."""
    return bus_str.split(".")[0].lower()


def build_static_elements():
    """Extract all transformers, loads, and static line/switch topology once."""
    lines = {}
    i = dss.Lines.First()
    while i > 0:
        name = dss.Lines.Name().lower()
        b1 = base_bus(dss.Lines.Bus1())
        b2 = base_bus(dss.Lines.Bus2())
        lines[name] = (b1, b2)
        i = dss.Lines.Next()

    xfmrs = {}
    i = dss.Transformers.First()
    while i > 0:
        name = dss.Transformers.Name()
        buses = [base_bus(b) for b in dss.CktElement.BusNames()]
        xfmrs[name] = buses
        i = dss.Transformers.Next()

    loads = {}
    i = dss.Loads.First()
    while i > 0:
        name = dss.Loads.Name()
        bus = base_bus(dss.CktElement.BusNames()[0])
        kw = dss.Loads.kW()
        loads[name] = (bus, kw)
        i = dss.Loads.Next()

    dss.Vsources.First()
    source_bus = base_bus(dss.CktElement.BusNames()[0])
    return source_bus, lines, xfmrs, loads


def get_energized_buses(source_bus, lines, xfmrs, failed_set):
    """BFS from substation Vsource over enabled lines and transformers."""
    adj = defaultdict(list)
    for lname, (b1, b2) in lines.items():
        if lname not in failed_set:
            adj[b1].append(b2)
            adj[b2].append(b1)

    for _, buses in xfmrs.items():
        for idx in range(len(buses) - 1):
            adj[buses[idx]].append(buses[idx + 1])
            adj[buses[idx + 1]].append(buses[idx])

    visited = {source_bus}
    q = deque([source_bus])
    while q:
        curr = q.popleft()
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return visited


def load_scenarios(path):
    names, rows = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            names.append(row[0].strip())
            rows.append([int(float(x)) for x in row[1:]])
    return names, np.array(rows, dtype=np.int8) == 1


def total_kw():
    return dss.Circuit.TotalPower()[0]


def make_outputs(rows, failed_lists, n_total):
    arr = np.array(rows, dtype=float)
    unserved = arr[:, 3]
    n_failed = arr[:, 4]
    ok = arr[:, 5] == 1

    # ---- Worst scenarios -> CSV ----
    idx_ok = np.where(ok)[0]
    worst_idx = idx_ok[np.argsort(unserved[idx_ok])[::-1][:TOP_N_WORST]]
    with open(worst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "scenario", "n_lines_failed", "unserved_kw", "failed_lines"])
        for rank, i in enumerate(worst_idx, start=1):
            w.writerow([rank, int(arr[i, 0]), int(n_failed[i]),
                        f"{unserved[i]:.2f}", ";".join(failed_lists[i])])
    print(f"Saved top {len(worst_idx)} worst scenarios to {worst_csv}")

    # ---- Two-panel figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.scatter(n_failed[ok], unserved[ok], s=14, alpha=0.55, edgecolors="none",
                label="Converged")
    ax1.scatter(n_failed[worst_idx], unserved[worst_idx], s=40, facecolors="none",
                edgecolors="red", linewidths=1.2, label=f"Top {len(worst_idx)} worst")
    if (~ok).any():
        ax1.scatter(n_failed[~ok], np.zeros((~ok).sum()), marker="x", color="gray",
                    s=22, label="Did not converge (plotted at 0)")
    if ok.sum() > 2:
        m, b = np.polyfit(n_failed[ok], unserved[ok], 1)
        xs = np.linspace(n_failed[ok].min(), n_failed[ok].max(), 50)
        ax1.plot(xs, m * xs + b, "k--", linewidth=1, label="Linear trend")
    ax1.set_xlabel("Number of lines failed")
    ax1.set_ylabel("Unserved load (kW)")
    ax1.set_title("Unserved load vs. number of failed lines")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    vals = np.sort(unserved[ok])[::-1]
    exceed = np.arange(1, len(vals) + 1) / len(vals) * 100
    ax2.plot(vals, exceed, linewidth=2)
    for pct in (50, 10, 5):
        v = np.percentile(unserved[ok], 100 - pct)
        ax2.axhline(pct, color="gray", linestyle=":", linewidth=0.8)
        ax2.annotate(f"{pct}% of scenarios > {v:,.0f} kW", (v, pct),
                     textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax2.set_xlabel("Unserved load (kW)")
    ax2.set_ylabel("% of scenarios exceeding")
    ax2.set_title("Exceedance curve")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Failure scenario analysis (N = {n_total})")
    fig.tight_layout()
    fig.savefig(plot_png, dpi=200)
    print(f"Saved plot to {plot_png}")
    plt.show()


def main():
    make_snapshot_files()
    line_names, failed = load_scenarios(csv_path)
    n_scen = failed.shape[1]
    print(f"Loaded {len(line_names)} lines x {n_scen} scenarios from {csv_path}")

    # Compile and relax Vminpu / solver limits for extreme contingencies
    dss.Command(f'Redirect "{master_path}"')
    dss.Command("Batchedit Load..* Vminpu=0.65")
    dss.Command("Set MaxIterations=100 MaxControlIter=100")
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        sys.exit("Baseline failed to converge - fix that first.")

    baseline_kw = total_kw()
    prefault_kw = abs(baseline_kw)
    print(f"Baseline total power: {prefault_kw:.2f} kW")

    source_bus, lines, xfmrs, loads = build_static_elements()

    # Check disabled lines in base case (e.g., normally-open elbswitch)
    base_disabled = set()
    i = dss.Lines.First()
    while i > 0:
        if not dss.CktElement.Enabled():
            base_disabled.add(dss.Lines.Name().lower())
        i = dss.Lines.Next()

    circuit_names = {n.lower(): n for n in dss.Lines.AllNames()}
    dss_name = [circuit_names.get(n.lower()) for n in line_names]

    failed_lists = []
    rows = []
    for s in range(n_scen):
        idx = np.where(failed[:, s])[0]
        to_fail = [dss_name[i] for i in idx if dss_name[i] is not None]
        failed_set = base_disabled | {n.lower() for n in to_fail}

        # 1. Disable failed lines
        for n in to_fail:
            dss.Lines.Name(n)
            dss.CktElement.Enabled(False)

        # 2. Isolate de-energized islands (loads & transformers)
        energized = get_energized_buses(source_bus, lines, xfmrs, failed_set)

        disabled_xfmrs = []
        for xname, buses in xfmrs.items():
            if buses[0] not in energized:
                dss.Transformers.Name(xname)
                dss.CktElement.Enabled(False)
                disabled_xfmrs.append(xname)

        disabled_loads = []
        for ldname, (lbus, _) in loads.items():
            if lbus not in energized:
                dss.Loads.Name(ldname)
                dss.CktElement.Enabled(False)
                disabled_loads.append(ldname)

        # 3. Solve energized subgraph (with Newton fallback if Normal stalls)
        dss.Command("Set Algorithm=Normal")
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            dss.Command("Set Algorithm=Newton")
            dss.Solution.Solve()

        converged = bool(dss.Solution.Converged())
        if converged:
            postfault_kw = abs(total_kw())
            unserved_kw = max(0.0, prefault_kw - postfault_kw)
        else:
            postfault_kw = float("nan")
            unserved_kw = float("nan")

        rows.append((s + 1, prefault_kw, postfault_kw, unserved_kw, len(to_fail), converged))
        failed_lists.append(to_fail)

        # 4. Restore circuit state for next scenario
        for n in to_fail:
            if n.lower() not in base_disabled:
                dss.Lines.Name(n)
                dss.CktElement.Enabled(True)
        for xname in disabled_xfmrs:
            dss.Transformers.Name(xname)
            dss.CktElement.Enabled(True)
        for ldname in disabled_loads:
            dss.Loads.Name(ldname)
            dss.CktElement.Enabled(True)

        if (s + 1) % 250 == 0 or s + 1 == n_scen:
            conv_rate = sum(r[5] for r in rows) / len(rows) * 100
            print(f"  scenario {s + 1}/{n_scen} done (convergence: {conv_rate:.1f}%)")

    # Save outputs
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "prefault_load_kw", "postfault_load_kw",
                    "delta_unserved_kw", "n_lines_failed", "converged"])
        for scen, pre_kw, post_kw, unserved, n_f, conv in rows:
            w.writerow([scen, f"{pre_kw:.2f}",
                        f"{post_kw:.2f}" if conv else "NaN",
                        f"{unserved:.2f}" if conv else "NaN",
                        n_f, conv])

    with open(cpdm_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "prefault_load_kw", "postfault_load_kw",
                    "delta_unserved_kw", "n_lines_failed"])
        for scen, pre_kw, post_kw, unserved, n_f, conv in rows:
            if conv:
                w.writerow([scen, f"{pre_kw:.2f}", f"{post_kw:.2f}",
                            f"{unserved:.2f}", n_f])

    print(f"Saved results to {results_csv} and {cpdm_csv}")
    make_outputs(rows, failed_lists, n_scen)


if __name__ == "__main__":
    main()





# import os, sys, csv, re
# import numpy as np
# import matplotlib.pyplot as plt
# import opendssdirect as dss

# script_dir = os.path.dirname(os.path.abspath(__file__))
# master_path = os.path.join(script_dir, "Master_snapshot.dss")
# csv_path = os.path.join(script_dir, "failed_lines_matrix.csv")
# results_csv = os.path.join(script_dir, "scenario_results.csv")
# cpdm_csv = os.path.join(script_dir, "cpdm_input.csv")
# plot_png = os.path.join(script_dir, "unserved_load_analysis.png")
# worst_csv = os.path.join(script_dir, "worst_scenarios.csv")
# TOP_N_WORST = 20


# def make_snapshot_files(folder):
#     """Create Master_snapshot.dss / Loads_snapshot.dss if missing.
#     Skips load shapes, monitors and meters so the feeder solves as a single
#     snapshot without the (not downloaded) profile CSVs. Originals are untouched."""
#     master_src = os.path.join(folder, "Master.dss")
#     loads_src = os.path.join(folder, "Loads.dss")
#     master_dst = os.path.join(folder, "Master_snapshot.dss")
#     loads_dst = os.path.join(folder, "Loads_snapshot.dss")

#     if not os.path.exists(loads_dst) and os.path.exists(loads_src):
#         with open(loads_src) as f:
#             text = f.read()
#         text = re.sub(r"\s+(yearly|daily|duty)=\S+", "", text, flags=re.I)
#         with open(loads_dst, "w") as f:
#             f.write(text)

#     if not os.path.exists(master_dst):
#         if not os.path.exists(master_src):
#             sys.exit(f"Master.dss not found in {folder}")
#         skip = ("redirect loadshapes", "solve", "export", "plot", "show",
#                 "new monitor", "new energymeter")
#         out = []
#         with open(master_src) as f:
#             for line in f:
#                 low = line.strip().lower()
#                 if low.startswith(skip):
#                     out.append("// " + line)
#                 elif low.startswith("redirect loads.dss"):
#                     out.append("Redirect Loads_snapshot.dss\n")
#                 else:
#                     out.append(line)
#         with open(master_dst, "w") as f:
#             f.writelines(out)
#         print("Created Master_snapshot.dss and Loads_snapshot.dss")


# def load_scenarios(path):
#     """Read the failure matrix: one row per line, one column per scenario.
#     Cell value 1 = line failed, 0 = line not failed.
#     Returns (line_names, failed_matrix) where failed_matrix[i, j] is True if
#     line i fails in scenario j."""
#     names, rows = [], []
#     with open(path, newline="") as f:
#         reader = csv.reader(f)
#         next(reader)                       # header: line_name,1,2,3,...
#         for row in reader:
#             if not row or not row[0].strip():
#                 continue
#             names.append(row[0].strip())
#             rows.append([int(float(x)) for x in row[1:]])
#     return names, np.array(rows, dtype=np.int8) == 1


# def total_kw():
#     return dss.Circuit.TotalPower()[0]   # negative = power delivered


# def set_enabled(line_name, state):
#     """Enable/disable a line by name (works for names with ( ) : characters)."""
#     dss.Lines.Name(line_name)
#     dss.CktElement.Enabled(state)


# def main():
#     make_snapshot_files(script_dir)
#     line_names, failed = load_scenarios(csv_path)
#     n_scen = failed.shape[1]
#     print(f"Loaded {len(line_names)} lines x {n_scen} scenarios from {csv_path}")

#     # Baseline (compile once)
#     dss.Command(f'Redirect "{master_path}"')
#     dss.Solution.Solve()
#     if not dss.Solution.Converged():
#         sys.exit("Baseline failed to converge - fix that first.")
#     baseline_kw = total_kw()
#     prefault_kw = abs(baseline_kw)
#     print(f"Baseline total power: {prefault_kw:.2f} kW")

#     # Match CSV line names to circuit line names (case-insensitive)
#     circuit_names = {n.lower(): n for n in dss.Lines.AllNames()}
#     dss_name = [circuit_names.get(n.lower()) for n in line_names]
#     missing = sum(d is None for d in dss_name)
#     if missing:
#         print(f"Warning: {missing} of {len(line_names)} CSV lines not found in circuit (ignored)")

#     failed_lists = []
#     rows = []   # (scenario, prefault_kw, postfault_kw, delta_unserved_kw, n_failed, converged)
#     for s in range(n_scen):
#         idx = np.where(failed[:, s])[0]
#         to_fail = [dss_name[i] for i in idx if dss_name[i] is not None]

#         for n in to_fail:
#             set_enabled(n, False)
#         dss.Solution.Solve()
#         converged = bool(dss.Solution.Converged())
        
#         if converged:
#             postfault_kw = abs(total_kw())
#             unserved_kw = abs(baseline_kw - total_kw())
#         else:
#             postfault_kw = 0.0
#             unserved_kw = 0.0

#         rows.append((s + 1, prefault_kw, postfault_kw, unserved_kw, len(to_fail), converged))
#         failed_lists.append(to_fail)

#         # Restore the circuit for the next scenario
#         for n in to_fail:
#             set_enabled(n, True)

#         if (s + 1) % 50 == 0 or s + 1 == n_scen:
#             print(f"  scenario {s + 1}/{n_scen} done")

#     # Save requested columns to scenario_results.csv
#     with open(results_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["scenario", "prefault_load_kw", "postfault_load_kw",
#                     "delta_unserved_kw", "n_lines_failed", "converged"])
#         for scen, pre_kw, post_kw, unserved, n_f, conv in rows:
#             w.writerow([scen, f"{pre_kw:.2f}", f"{post_kw:.2f}",
#                         f"{unserved:.2f}", n_f, conv])
#     print(f"Saved results to {results_csv}")

#     # File for the CPDM code:
#     # scenario, prefault load, post-fault load, delta unserved load, # failed lines
#     with open(cpdm_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["scenario", "prefault_load_kw", "postfault_load_kw",
#                     "delta_unserved_kw", "n_lines_failed"])
#         for scen, pre_kw, post_kw, unserved, n_f, conv in rows:
#             if conv:
#                 w.writerow([scen, f"{pre_kw:.2f}", f"{post_kw:.2f}",
#                             f"{unserved:.2f}", n_f])
#             else:
#                 w.writerow([scen, f"{pre_kw:.2f}", "0.00", "0.00", n_f])
#     print(f"Saved CPDM input to {cpdm_csv}")

#     make_outputs(rows, failed_lists, n_scen)


# def make_outputs(rows, failed_lists, n_total):
#     arr = np.array(rows, dtype=float)
#     unserved = arr[:, 3]
#     n_failed = arr[:, 4]
#     ok = arr[:, 5] == 1

#     # ---- Worst scenarios -> CSV ----
#     idx_ok = np.where(ok)[0]
#     worst_idx = idx_ok[np.argsort(unserved[idx_ok])[::-1][:TOP_N_WORST]]
#     with open(worst_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["rank", "scenario", "n_lines_failed", "unserved_kw", "failed_lines"])
#         for rank, i in enumerate(worst_idx, start=1):
#             w.writerow([rank, int(arr[i, 0]), int(n_failed[i]),
#                         f"{unserved[i]:.2f}", ";".join(failed_lists[i])])
#     print(f"Saved top {len(worst_idx)} worst scenarios to {worst_csv}")

#     # ---- Two-panel figure ----
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

#     # Left: lines failed vs unserved load
#     ax1.scatter(n_failed[ok], unserved[ok], s=14, alpha=0.55, edgecolors="none",
#                 label="Converged")
#     ax1.scatter(n_failed[worst_idx], unserved[worst_idx], s=40, facecolors="none",
#                 edgecolors="red", linewidths=1.2, label=f"Top {len(worst_idx)} worst")
#     if (~ok).any():
#         ax1.scatter(n_failed[~ok], np.zeros((~ok).sum()), marker="x", color="gray",
#                     s=22, label="Did not converge (plotted at 0)")
#     if ok.sum() > 2:
#         m, b = np.polyfit(n_failed[ok], unserved[ok], 1)
#         xs = np.linspace(n_failed[ok].min(), n_failed[ok].max(), 50)
#         ax1.plot(xs, m * xs + b, "k--", linewidth=1, label="Linear trend")
#     ax1.set_xlabel("Number of lines failed")
#     ax1.set_ylabel("Unserved load (kW)")
#     ax1.set_title("Unserved load vs. number of failed lines")
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # Right: exceedance curve (fraction of scenarios exceeding each level)
#     vals = np.sort(unserved[ok])[::-1]
#     exceed = np.arange(1, len(vals) + 1) / len(vals) * 100
#     ax2.plot(vals, exceed, linewidth=2)
#     for pct in (50, 10, 5):
#         v = np.percentile(unserved[ok], 100 - pct)
#         ax2.axhline(pct, color="gray", linestyle=":", linewidth=0.8)
#         ax2.annotate(f"{pct}% of scenarios > {v:,.0f} kW", (v, pct),
#                      textcoords="offset points", xytext=(5, 4), fontsize=8)
#     ax2.set_xlabel("Unserved load (kW)")
#     ax2.set_ylabel("% of scenarios exceeding")
#     ax2.set_title("Exceedance curve")
#     ax2.grid(True, alpha=0.3)

#     fig.suptitle(f"Failure scenario analysis (N = {n_total})")
#     fig.tight_layout()
#     fig.savefig(plot_png, dpi=200)
#     print(f"Saved plot to {plot_png}")
#     plt.show()


# if __name__ == "__main__":
#     main()

# import os, sys, csv, re
# from collections import defaultdict, deque
# import numpy as np
# import matplotlib.pyplot as plt
# import opendssdirect as dss


# # Feeder to run. OpenDSS source data lives in <repo>/data/opendss/.
# REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FEEDER_NAME = "3777"
# FEEDER_DIR = os.path.join(REPO_ROOT, "data", "opendss", "p48rhs1_25", "p48rhs1_25--p48rdt3777")
# OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# master_path = os.path.join(script_dir, "Master_snapshot.dss")
# csv_path = os.path.join(script_dir, "failed_lines_matrix.csv")
# results_csv = os.path.join(script_dir, "scenario_results.csv")
# cpdm_csv = os.path.join(script_dir, "cpdm_input.csv")
# plot_png = os.path.join(script_dir, "unserved_load_analysis.png")
# worst_csv = os.path.join(script_dir, "worst_scenarios.csv")
# TOP_N_WORST = 20


# def make_snapshot_files(folder):
#     """Create Master_snapshot.dss / Loads_snapshot.dss if missing."""
#     master_src = os.path.join(FEEDER_DIR , "Master.dss")
#     loads_src = os.path.join(folder, "Loads.dss")
#     master_dst = os.path.join(folder, "Master_snapshot.dss")
#     loads_dst = os.path.join(folder, "Loads_snapshot.dss")

#     if not os.path.exists(loads_dst) and os.path.exists(loads_src):
#         with open(loads_src) as f:
#             text = f.read()
#         text = re.sub(r"\s+(yearly|daily|duty)=\S+", "", text, flags=re.I)
#         with open(loads_dst, "w") as f:
#             f.write(text)

#     if not os.path.exists(master_dst):
#         if not os.path.exists(master_src):
#             sys.exit(f"Master.dss not found in {folder}")
#         skip = ("redirect loadshapes", "solve", "export", "plot", "show",
#                 "new monitor", "new energymeter")
#         out = []
#         with open(master_src) as f:
#             for line in f:
#                 low = line.strip().lower()
#                 if low.startswith(skip):
#                     out.append("// " + line)
#                 elif low.startswith("redirect loads.dss"):
#                     out.append("Redirect Loads_snapshot.dss\n")
#                 else:
#                     out.append(line)
#         with open(master_dst, "w") as f:
#             f.writelines(out)
#         print("Created Master_snapshot.dss and Loads_snapshot.dss")
