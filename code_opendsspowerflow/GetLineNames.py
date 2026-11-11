import os, csv
import opendssdirect as dss

# Feeder to run. OpenDSS source data lives in <repo>/data/opendss/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDER_NAME = "3777"
FEEDER_DIR = os.path.join(REPO_ROOT, "data", "opendss", "p48rhs1_25", "p48rhs1_25--p48rdt3777")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
master_path = os.path.join(FEEDER_DIR, "Master_snapshot.dss")

dss.Command(f'Redirect "{master_path}"')
dss.Solution.Solve()

# --- Lines: name, from-bus, to-bus, length ---
lines_csv = os.path.join(OUTPUT_DIR, f"feeder_{FEEDER_NAME}_line_names.csv")
with open(lines_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["line_name", "bus1", "bus2", "length"])
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        writer.writerow([name, dss.Lines.Bus1(), dss.Lines.Bus2(), dss.Lines.Length()])
print(f"Saved {len(dss.Lines.AllNames())} lines to {lines_csv}")

# --- Buses: name, x, y, base voltage ---
buses_csv = os.path.join(OUTPUT_DIR, f"feeder_{FEEDER_NAME}_bus_names.csv")
with open(buses_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bus_name", "x", "y", "kv_base"])
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        has_xy = dss.Bus.Coorddefined()
        writer.writerow([name,
                         dss.Bus.X() if has_xy else "",
                         dss.Bus.Y() if has_xy else "",
                         dss.Bus.kVBase()])
print(f"Saved {len(dss.Circuit.AllBusNames())} buses to {buses_csv}")