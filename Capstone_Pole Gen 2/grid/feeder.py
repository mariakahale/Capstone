import opendssdirect as dss
import pandas as pd


def load_feeder(master_path):
    """
    Compile the OpenDSS feeder and solve the base case.
    """
    dss.Basic.ClearAll()

    dss.Text.Command(f'Compile "{master_path}"')
    dss.Solution.Solve()

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS power flow did not converge.")

    print("OpenDSS power flow converged.")


def clean_bus_name(bus_full):
    """
    Remove phase/node information from an OpenDSS bus name.

    Example:
        bus123.1.2.3 -> bus123
    """
    return str(bus_full).split(".")[0]


def get_bus_info(bus_full, bus_lookup):
    """
    Get voltage base and coordinates for an OpenDSS bus.
    """
    bus_name = clean_bus_name(bus_full)

    actual_name = bus_lookup.get(bus_name.lower())

    if actual_name is None:
        return None, None, None

    dss.Circuit.SetActiveBus(actual_name)

    kv_base = dss.Bus.kVBase()

    if dss.Bus.Coorddefined():
        x = dss.Bus.X()
        y = dss.Bus.Y()
    else:
        x = None
        y = None

    return kv_base, x, y


def extract_lines():
    """
    Extract every OpenDSS Line element.

    No filtering is performed here.
    Primary, secondary, overhead and underground lines are all retained.
    """

    rows = []

    bus_lookup = {
        bus.lower(): bus
        for bus in dss.Circuit.AllBusNames()
    }

    for line_name in dss.Lines.AllNames():

        dss.Lines.Name(line_name)

        bus1_full = dss.Lines.Bus1()
        bus2_full = dss.Lines.Bus2()

        bus1_name = clean_bus_name(bus1_full)
        bus2_name = clean_bus_name(bus2_full)

        bus1_kv, bus1_x, bus1_y = get_bus_info(
            bus1_full,
            bus_lookup
        )

        bus2_kv, bus2_x, bus2_y = get_bus_info(
            bus2_full,
            bus_lookup
        )

        rows.append({
            "line_name": line_name,

            "bus1": bus1_full,
            "bus2": bus2_full,

            "bus1_name": bus1_name,
            "bus2_name": bus2_name,

            "length_raw": dss.Lines.Length(),
            "length_units_code": dss.Lines.Units(),

            "phases": dss.CktElement.NumPhases(),
            "linecode": dss.Lines.LineCode(),
            "enabled": dss.CktElement.Enabled(),

            "bus1_kvbase": bus1_kv,
            "bus2_kvbase": bus2_kv,

            "bus1_x": bus1_x,
            "bus1_y": bus1_y,
            "bus2_x": bus2_x,
            "bus2_y": bus2_y,
        })

    return pd.DataFrame(rows)