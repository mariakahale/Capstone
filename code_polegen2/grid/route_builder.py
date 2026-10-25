import pandas as pd


def get_anchor_reasons(graph):
    """
    Determine which OH buses should act as route anchors.

    A bus is an anchor when:
        - degree = 1: OH endpoint
        - degree >= 3: OH branch
        - degree = 2 but voltage class changes
        - degree = 2 but construction phase type changes

    A degree-2 bus with matching properties on both sides
    is treated as an ordinary pass-through bus.
    """

    anchors = {}

    for bus_name, edges in graph.items():

        degree = len(edges)
        reasons = []

        # -----------------------------------------------------
        # Topology-based anchors
        # -----------------------------------------------------

        if degree == 1:
            reasons.append("endpoint")

        elif degree >= 3:
            reasons.append("branch")

        # -----------------------------------------------------
        # Degree-2 transition checks
        # -----------------------------------------------------

        elif degree == 2:

            voltage_classes = {
                edge["voltage_class"]
                for edge in edges
            }

            phase_types = {
                edge["construction_phases"]
                for edge in edges
                if pd.notna(edge["construction_phases"])
            }

            if len(voltage_classes) > 1:
                reasons.append("voltage_transition")

            if len(phase_types) > 1:
                reasons.append("phase_transition")

        # -----------------------------------------------------
        # Store only buses that are anchors
        # -----------------------------------------------------

        if reasons:
            anchors[bus_name] = reasons

    return anchors


def build_anchor_table(
    graph,
    bus_data,
    anchors
):
    """
    Create a DataFrame describing all route anchor buses.
    """

    rows = []

    for bus_name, reasons in anchors.items():

        data = bus_data[bus_name]

        rows.append({
            "bus_name": bus_name,
            "degree": len(graph[bus_name]),
            "anchor_reason": ";".join(reasons),
            "x": data["x"],
            "y": data["y"],
            "kvbase": data["kvbase"]
        })

    return pd.DataFrame(rows)


def trace_routes(
    graph,
    anchors
):
    """
    Trace continuous OH routes from anchor to anchor.

    A route begins at an anchor and follows degree-2
    pass-through buses until another anchor is reached.

    Each OpenDSS line is used only once.
    """

    used_lines = set()
    routes = []

    route_number = 1

    # ---------------------------------------------------------
    # Start tracing from every anchor
    # ---------------------------------------------------------

    for start_bus in anchors:

        for starting_edge in graph[start_bus]:

            starting_line = starting_edge["line_name"]

            # This conductor already belongs to a route
            if starting_line in used_lines:
                continue

            route_id = f"R{route_number:04d}"

            route_buses = [start_bus]
            route_lines = []

            route_length_ft = 0.0

            voltage_classes = set()
            phase_types = set()

            current_bus = start_bus
            current_edge = starting_edge

            # -------------------------------------------------
            # Walk until another anchor is reached
            # -------------------------------------------------

            while True:

                line_name = current_edge["line_name"]
                next_bus = current_edge["neighbor"]

                used_lines.add(line_name)

                route_lines.append(line_name)

                route_length_ft += (
                    current_edge["length_ft"]
                )

                voltage_classes.add(
                    current_edge["voltage_class"]
                )

                if pd.notna(
                    current_edge["construction_phases"]
                ):
                    phase_types.add(
                        int(
                            current_edge[
                                "construction_phases"
                            ]
                        )
                    )

                route_buses.append(next_bus)

                # ---------------------------------------------
                # Reached another anchor
                # ---------------------------------------------

                if next_bus in anchors:
                    end_bus = next_bus
                    break

                # ---------------------------------------------
                # Otherwise this should be a degree-2
                # pass-through bus.
                # Find the next unused conductor.
                # ---------------------------------------------

                next_edges = [
                    edge
                    for edge in graph[next_bus]
                    if edge["line_name"] not in used_lines
                ]

                if len(next_edges) != 1:
                    raise RuntimeError(
                        f"Unexpected topology while tracing "
                        f"{route_id} at bus {next_bus}. "
                        f"Expected 1 unused continuation, "
                        f"found {len(next_edges)}."
                    )

                current_bus = next_bus
                current_edge = next_edges[0]

            # -------------------------------------------------
            # Save completed route
            # -------------------------------------------------

            routes.append({
                "route_id": route_id,
                "start_bus": start_bus,
                "end_bus": end_bus,

                "start_anchor_reason":
                    ";".join(anchors[start_bus]),

                "end_anchor_reason":
                    ";".join(anchors[end_bus]),

                "total_length_ft": route_length_ft,

                "num_dss_lines": len(route_lines),

                "num_buses": len(route_buses),

                "voltage_classes":
                    ";".join(
                        sorted(voltage_classes)
                    ),

                "construction_phases":
                    ";".join(
                        str(x)
                        for x in sorted(phase_types)
                    ),

                "bus_sequence":
                    ";".join(route_buses),

                "line_sequence":
                    ";".join(route_lines)
            })

            route_number += 1

    routes_df = pd.DataFrame(routes)

    # ---------------------------------------------------------
    # Determine whether any OH lines were not traced.
    #
    # This can occur in a closed loop consisting entirely
    # of degree-2 buses, because such a loop has no anchor.
    # ---------------------------------------------------------

    all_lines = {
        edge["line_name"]
        for edges in graph.values()
        for edge in edges
    }

    untraced_lines = sorted(
        all_lines - used_lines
    )

    

    return routes_df, untraced_lines