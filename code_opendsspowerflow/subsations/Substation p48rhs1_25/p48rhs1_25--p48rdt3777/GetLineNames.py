import os, csv
import opendssdirect as dss

# Folder this script lives in (the feeder folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
master_path = os.path.join(script_dir, "Master_snapshot.dss")

dss.Command(f'Redirect "{master_path}"')
dss.Solution.Solve()

# --- Lines: name, from-bus, to-bus, length ---
lines_csv = os.path.join(script_dir, "line_names.csv")
with open(lines_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["line_name", "bus1", "bus2", "length"])
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        writer.writerow([name, dss.Lines.Bus1(), dss.Lines.Bus2(), dss.Lines.Length()])
print(f"Saved {len(dss.Lines.AllNames())} lines to {lines_csv}")

# --- Buses: name, x, y, base voltage ---
buses_csv = os.path.join(script_dir, "bus_names.csv")
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