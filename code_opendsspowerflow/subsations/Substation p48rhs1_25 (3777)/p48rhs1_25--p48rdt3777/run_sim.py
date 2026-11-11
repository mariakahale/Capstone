import os, re
from dss import DSS

script_dir = os.path.dirname(os.path.abspath(__file__))

def make_snapshot_files(folder):
    master = os.path.join(folder, "Master.dss")
    loads = os.path.join(folder, "Loads.dss")
    master_snap = os.path.join(folder, "Master_snapshot.dss")
    loads_snap = os.path.join(folder, "Loads_snapshot.dss")

    if not os.path.exists(loads_snap):
        text = re.sub(r"\s+(yearly|daily|duty)=\S+", "", open(loads).read())
        open(loads_snap, "w").write(text)

    if not os.path.exists(master_snap):
        out = []
        for line in open(master):
            low = line.strip().lower()
            if low.startswith(("redirect loadshapes", "solve", "export", "plot", "show",
                   "new monitor", "new energymeter")):
                line = "! " + line
            else:
                line = re.sub(r"(?i)redirect\s+loads\.dss", "Redirect Loads_snapshot.dss", line)
            out.append(line)
        open(master_snap, "w").write("".join(out))

make_snapshot_files(script_dir)
master_path = os.path.join(script_dir, "Master_snapshot.dss")
print("Using:", master_path)


# Initialize OpenDSS engine
dss_engine = DSS
dss_text = dss_engine.Text
dss_circuit = dss_engine.ActiveCircuit

# Compile and run the circuit
print("Compiling OpenDSS circuit...")
dss_text.Command = f'Compile "{master_path}"'

# Check if it solved successfully
if dss_circuit.Solution.Converged:
    print("Simulation converged successfully!")
    # # Print the voltages at each node
    # for name, mag in zip(dss_circuit.AllNodeNames, dss_circuit.AllBusVolts):
    #     print(f"Node {name}: {mag:.2f} V")
else:
    print("Simulation failed to converge.")

#--------get the name of the buses------------- 
import opendssdirect as dss

dss.Command(f'Redirect "{master_path}"')
dss.Solution.Solve()

print(dss.Lines.AllNames())


# -------------------force a failure on a line---------------------------
import opendssdirect as dss

# --- Baseline solve ---
dss.Command(f'Redirect "{master_path}"')
dss.Solution.Solve()

if dss.Solution.Converged():
    baseline_voltages = dss.Circuit.AllBusMagPu()
    print(f"Baseline converged. {len(dss.Circuit.AllNodeNames())} nodes.")
else:
    print("Baseline failed to converge — stop here and debug first.")

# --- Force a line failure ---
target_line = "fuse(r:p48rdt11855-p48rdt17642)p48r_20440"  # replace with a real name from Lines.AllNames()
dss.Command(f"Line.{target_line}.enabled=no")

# --- Re-solve with the failure applied ---
dss.Solution.Solve()

if dss.Solution.Converged():
    print("Post-failure solve converged.")
    failed_voltages = dss.Circuit.AllBusMagPu()
else:
    print("Post-failure solve did NOT converge — likely means the failure islanded part of the circuit with no source.")


# check which loads got disconnected
for name in dss.Loads.AllNames():
    dss.Loads.Name(name)
    bus_name = dss.CktElement.BusNames()[0]
    dss.Circuit.SetActiveBus(bus_name)
    voltage_pu = dss.Bus.puVmagAngle()[0]  # first value is magnitude in pu
    
    status = "DISCONNECTED" if voltage_pu < 0.1 else "OK"
    print(f"{name} (bus {bus_name}): {voltage_pu:.4f} pu — {status}")



dss.Command(f'Redirect "{master_path}"')
dss.Solution.Solve()
baseline_kw = dss.Circuit.TotalPower()[0]  # negative = power delivered

dss.Command(f"Line.{target_line}.enabled=no")
dss.Solution.Solve()
failure_kw = dss.Circuit.TotalPower()[0]

print(f"Baseline total power: {baseline_kw} kW")
print(f"Post-failure total power: {failure_kw} kW")
print(f"Unserved load estimate: {baseline_kw - failure_kw} kW")
