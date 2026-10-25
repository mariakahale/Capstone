#talks to OpenDSS

import opendssdirect as dss
import pandas as pd


def load_feeder(master_path):
    """
    Load an OpenDSS feeder and solve the power flow.
    """

    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{master_path}"')

    # Solve once
    dss.Solution.Solve()

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS power flow did not converge.")

    print("Converged:", dss.Solution.Converged())


def clean_bus_name(bus_full):
    """
    Remove OpenDSS node numbers.

    Example:
    p1udt4233.1.2.3 -> p1udt4233
    """

    return str(bus_full).split(".")[0]


def get_bus_info(bus_full, bus_lookup):
    """
    Get voltage base and coordinates for one OpenDSS bus.
    """

    bus_name = clean_bus_name(bus_full)

    # Bus might not be part of the active circuit
    if bus_name.lower() not in bus_lookup:
        return None, None, None

    actual_bus_name = bus_lookup[bus_name.lower()]

    dss.Circuit.SetActiveBus(actual_bus_name)

    kvbase = dss.Bus.kVBase()

    if dss.Bus.Coorddefined():
        x = dss.Bus.X()
        y = dss.Bus.Y()
    else:
        x = None
        y = None

    return kvbase, x, y


def extract_lines():
    """
    Extract all OpenDSS Line elements into a pandas DataFrame.
    """

    lines = []

    # List of active buses in this feeder
    bus_lookup = {
        bus.lower(): bus
        for bus in dss.Circuit.AllBusNames()
    }

    for line_name in dss.Lines.AllNames():

        # Activate this line
        dss.Lines.Name(line_name)

        # -------------------------
        # Save line information
        # -------------------------

        bus1_full = dss.Lines.Bus1()
        bus2_full = dss.Lines.Bus2()

        bus1_name = clean_bus_name(bus1_full)
        bus2_name = clean_bus_name(bus2_full)

        length_km = dss.Lines.Length()
        linecode = dss.Lines.LineCode()
        phases = dss.CktElement.NumPhases()
        enabled = dss.CktElement.Enabled()

        # -------------------------
        # Bus information
        # -------------------------

        bus1_kv, bus1_x, bus1_y = get_bus_info(
            bus1_full,
            bus_lookup
        )

        bus2_kv, bus2_x, bus2_y = get_bus_info(
            bus2_full,
            bus_lookup
        )

        # -------------------------
        # Store one line
        # -------------------------

        lines.append({
            "line_name": line_name,

            "bus1": bus1_full,
            "bus2": bus2_full,

            "bus1_name": bus1_name,
            "bus2_name": bus2_name,

            "length_km": length_km,
            "phases": phases,
            "linecode": linecode,
            "enabled": enabled,

            "bus1_kvbase": bus1_kv,
            "bus2_kvbase": bus2_kv,

            "bus1_x": bus1_x,
            "bus1_y": bus1_y,
            "bus2_x": bus2_x,
            "bus2_y": bus2_y
        })

    return pd.DataFrame(lines)