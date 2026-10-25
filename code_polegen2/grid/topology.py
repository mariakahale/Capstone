import pandas as pd


def build_overhead_graph(overhead_lines):
    """
    Build an undirected graph of the overhead network.

    Node:
        OpenDSS bus

    Edge:
        Physical overhead OpenDSS conductor

    Each bus stores a list of all overhead lines connected to it.
    """

    graph = {}
    bus_data = {}

    for row in overhead_lines.itertuples(index=False):

        bus1 = row.bus1_name
        bus2 = row.bus2_name

        # -----------------------------------------------------
        # Create buses if they do not already exist
        # -----------------------------------------------------

        if bus1 not in graph:
            graph[bus1] = []

        if bus2 not in graph:
            graph[bus2] = []

        # -----------------------------------------------------
        # Store bus information
        # -----------------------------------------------------

        if bus1 not in bus_data:
            bus_data[bus1] = {
                "x": row.bus1_x,
                "y": row.bus1_y,
                "kvbase": row.bus1_kvbase
            }

        if bus2 not in bus_data:
            bus_data[bus2] = {
                "x": row.bus2_x,
                "y": row.bus2_y,
                "kvbase": row.bus2_kvbase
            }

        # -----------------------------------------------------
        # Information belonging to this conductor
        # -----------------------------------------------------

        edge_1_to_2 = {
            "neighbor": bus2,
            "line_name": row.line_name,
            "length_ft": row.length_ft,
            "voltage_class": row.voltage_class,
            "construction_phases": row.construction_phases,
            "linecode": row.linecode
        }

        edge_2_to_1 = {
            "neighbor": bus1,
            "line_name": row.line_name,
            "length_ft": row.length_ft,
            "voltage_class": row.voltage_class,
            "construction_phases": row.construction_phases,
            "linecode": row.linecode
        }

        # Undirected graph:
        # store the same physical line from both bus directions
        graph[bus1].append(edge_1_to_2)
        graph[bus2].append(edge_2_to_1)

    return graph, bus_data


def classify_bus_type(degree):
    """
    Classify a bus based only on its OH network degree.

    degree 1  -> endpoint
    degree 2  -> pass-through
    degree 3+ -> branch
    """

    if degree == 0:
        return "isolated"

    if degree == 1:
        return "endpoint"

    if degree == 2:
        return "pass_through"

    return "branch"


def find_connected_components(graph):
    """
    Find disconnected overhead network components
    using a simple depth-first search.

    This is useful because OH sections may be separated
    by underground lines, transformers, or other devices.
    """

    visited = set()
    components = []

    for start_bus in graph:

        if start_bus in visited:
            continue

        stack = [start_bus]
        component = []

        while stack:

            bus = stack.pop()

            if bus in visited:
                continue

            visited.add(bus)
            component.append(bus)

            for edge in graph[bus]:
                neighbor = edge["neighbor"]

                if neighbor not in visited:
                    stack.append(neighbor)

        components.append(component)

    return components


def build_bus_topology_table(
    graph,
    bus_data,
    components
):
    """
    Create a DataFrame describing every bus in the
    overhead topology.
    """

    # ---------------------------------------------------------
    # Map each bus to its connected component
    # ---------------------------------------------------------

    component_lookup = {}

    for component_number, buses in enumerate(
        components,
        start=1
    ):

        component_id = f"C{component_number:04d}"

        for bus in buses:
            component_lookup[bus] = component_id

    # ---------------------------------------------------------
    # Build bus table
    # ---------------------------------------------------------

    rows = []

    for bus_name, edges in graph.items():

        degree = len(edges)

        connected_lines = sorted(
            edge["line_name"]
            for edge in edges
        )

        neighbor_buses = sorted(
            edge["neighbor"]
            for edge in edges
        )

        connected_voltage_classes = sorted(
            {
                edge["voltage_class"]
                for edge in edges
            }
        )

        connected_phase_types = sorted(
            {
                int(edge["construction_phases"])
                for edge in edges
                if pd.notna(edge["construction_phases"])
            }
        )

        data = bus_data[bus_name]

        rows.append({
            "bus_name": bus_name,
            "degree": degree,
            "bus_type": classify_bus_type(degree),

            "component_id":
                component_lookup.get(bus_name),

            "x": data["x"],
            "y": data["y"],
            "kvbase": data["kvbase"],

            "connected_lines":
                ";".join(connected_lines),

            "neighbor_buses":
                ";".join(neighbor_buses),

            "connected_voltage_classes":
                ";".join(connected_voltage_classes),

            "connected_phase_types":
                ";".join(
                    str(x)
                    for x in connected_phase_types
                )
        })

    bus_table = pd.DataFrame(rows)

    return bus_table

def find_buses_with_same_coordinates(
    bus_topology,
    decimals=8
):
    """
    Find different OH buses that are mapped to the same
    geographic coordinates.

    Coordinates are rounded slightly before comparison
    to avoid tiny floating-point differences.

    This is only a diagnostic check.
    Buses are NOT automatically merged.
    """

    check = bus_topology.copy()

    check["x_check"] = check["x"].round(decimals)
    check["y_check"] = check["y"].round(decimals)

    grouped = (
        check
        .groupby(
            ["x_check", "y_check"],
            dropna=False
        )
        .agg(
            bus_count=("bus_name", "count"),

            bus_names=(
                "bus_name",
                lambda x: ";".join(sorted(x))
            ),

            degrees=(
                "degree",
                lambda x: ";".join(
                    str(v)
                    for v in sorted(x)
                )
            ),

            kvbases=(
                "kvbase",
                lambda x: ";".join(
                    str(v)
                    for v in sorted(set(x))
                )
            ),

            bus_types=(
                "bus_type",
                lambda x: ";".join(
                    sorted(set(x))
                )
            )
        )
        .reset_index()
    )

    duplicates = grouped[
        grouped["bus_count"] > 1
    ].copy()

    return duplicates