import pandas as pd


def generate_poles_and_spans(
    routes,
    overhead,
    span_plan,
    bus_data,
    feeder_name
):
    """
    Generate physical poles and spans from the route-based
    equal-span plan.

    Returns:
        poles_df
        spans_df
        span_line_map_df
        pole_line_map_df

    Important:
        - Route anchor buses are shared between routes.
        - Intermediate poles are synthetic.
        - Pole coordinates are interpolated using the two
          surrounding OpenDSS bus coordinates.
        - Physical spans are mapped back to all OpenDSS
          Line elements they overlap.
    """

    # ---------------------------------------------------------
    # Lookups
    # ---------------------------------------------------------

    line_lookup = (
        overhead
        .set_index("line_name")
        .to_dict("index")
    )

    plan_lookup = (
        span_plan
        .set_index("route_id")
        .to_dict("index")
    )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    poles = {}
    spans = []
    span_line_map = []

    anchor_to_pole = {}

    next_pole_number = 1
    next_span_number = 1

    # ---------------------------------------------------------
    # Helper: create unique pole ID
    # ---------------------------------------------------------

    def new_pole_id():
        nonlocal next_pole_number

        pole_id = (
            f"{feeder_name}_P"
            f"{next_pole_number:05d}"
        )

        next_pole_number += 1

        return pole_id

    # ---------------------------------------------------------
    # Helper: create unique span ID
    # ---------------------------------------------------------

    def new_span_id():
        nonlocal next_span_number

        span_id = (
            f"{feeder_name}_S"
            f"{next_span_number:05d}"
        )

        next_span_number += 1

        return span_id

    # ---------------------------------------------------------
    # Helper: locate a point along a route
    # ---------------------------------------------------------

    def interpolate_on_route(
        distance_ft,
        segments,
        tolerance=1e-8
    ):
        """
        Determine the coordinate of a point at a given
        cumulative distance along a route.
        """

        for segment in segments:

            start_distance = segment["start_distance_ft"]
            end_distance = segment["end_distance_ft"]

            if (
                distance_ft
                <= end_distance + tolerance
            ):

                line_length = segment["length_ft"]

                if line_length <= 0:
                    raise ValueError(
                        "Encountered non-positive "
                        "OpenDSS line length."
                    )

                local_distance = (
                    distance_ft
                    - start_distance
                )

                fraction = (
                    local_distance
                    / line_length
                )

                # Protect against tiny floating point errors
                fraction = max(
                    0.0,
                    min(1.0, fraction)
                )

                x = (
                    segment["start_x"]
                    + fraction
                    * (
                        segment["end_x"]
                        - segment["start_x"]
                    )
                )

                y = (
                    segment["start_y"]
                    + fraction
                    * (
                        segment["end_y"]
                        - segment["start_y"]
                    )
                )

                return x, y

        raise RuntimeError(
            f"Could not locate route distance "
            f"{distance_ft:.6f} ft."
        )

    # =========================================================
    # Process each route
    # =========================================================

    for route in routes.itertuples(index=False):

        route_id = route.route_id

        bus_sequence = (
            route.bus_sequence.split(";")
        )

        line_sequence = (
            route.line_sequence.split(";")
        )

        if (
            len(bus_sequence)
            != len(line_sequence) + 1
        ):
            raise RuntimeError(
                f"{route_id}: bus/line sequence "
                f"is inconsistent."
            )

        # -----------------------------------------------------
        # Reconstruct oriented OpenDSS segments
        # -----------------------------------------------------

        segments = []

        cumulative_distance = 0.0

        for i, line_name in enumerate(
            line_sequence
        ):

            from_bus = bus_sequence[i]
            to_bus = bus_sequence[i + 1]

            line = line_lookup[line_name]

            line_bus1 = line["bus1_name"]
            line_bus2 = line["bus2_name"]

            # ---------------------------------------------
            # Orient coordinates according to route direction
            # ---------------------------------------------

            if (
                from_bus == line_bus1
                and to_bus == line_bus2
            ):

                start_x = line["bus1_x"]
                start_y = line["bus1_y"]

                end_x = line["bus2_x"]
                end_y = line["bus2_y"]

            elif (
                from_bus == line_bus2
                and to_bus == line_bus1
            ):

                start_x = line["bus2_x"]
                start_y = line["bus2_y"]

                end_x = line["bus1_x"]
                end_y = line["bus1_y"]

            else:

                raise RuntimeError(
                    f"{route_id}: line {line_name} "
                    f"does not connect expected buses "
                    f"{from_bus} and {to_bus}."
                )

            length_ft = line["length_ft"]

            segments.append({
                "line_name": line_name,

                "from_bus": from_bus,
                "to_bus": to_bus,

                "length_ft": length_ft,

                "start_distance_ft":
                    cumulative_distance,

                "end_distance_ft":
                    cumulative_distance
                    + length_ft,

                "start_x": start_x,
                "start_y": start_y,

                "end_x": end_x,
                "end_y": end_y
            })

            cumulative_distance += length_ft

        # -----------------------------------------------------
        # Validate reconstructed route length
        # -----------------------------------------------------

        if abs(
            cumulative_distance
            - route.total_length_ft
        ) > 0.001:

            raise RuntimeError(
                f"{route_id}: reconstructed route "
                f"length does not match route table."
            )

        # -----------------------------------------------------
        # Get planned equal-span information
        # -----------------------------------------------------

        plan = plan_lookup[route_id]

        num_spans = int(
            plan["num_spans"]
        )

        span_length_ft = float(
            plan["span_length_ft"]
        )

        route_length_ft = float(
            route.total_length_ft
        )

        # -----------------------------------------------------
        # Generate cumulative pole positions
        # -----------------------------------------------------

        pole_distances = [
            i * span_length_ft
            for i in range(num_spans)
        ]

        # Force final pole to exactly equal route length
        # to avoid accumulated floating-point error.
        pole_distances.append(
            route_length_ft
        )

        route_poles = []

        # =====================================================
        # Generate poles
        # =====================================================

        for i, distance_ft in enumerate(
            pole_distances
        ):

            is_start = (i == 0)

            is_end = (
                i == len(pole_distances) - 1
            )

            is_anchor = (
                is_start or is_end
            )

            # -------------------------------------------------
            # Route anchor
            # -------------------------------------------------

            if is_anchor:

                if is_start:
                    anchor_bus = route.start_bus
                else:
                    anchor_bus = route.end_bus

                # Reuse an existing physical pole when
                # multiple routes meet at the same anchor bus.
                if anchor_bus in anchor_to_pole:

                    pole_id = (
                        anchor_to_pole[
                            anchor_bus
                        ]
                    )

                    poles[pole_id][
                        "route_ids"
                    ].add(route_id)

                else:

                    pole_id = new_pole_id()

                    anchor_to_pole[
                        anchor_bus
                    ] = pole_id

                    bus = bus_data[
                        anchor_bus
                    ]

                    poles[pole_id] = {
                        "pole_id": pole_id,

                        "pole_type": "anchor",

                        "anchor_bus":
                            anchor_bus,

                        "x": bus["x"],
                        "y": bus["y"],

                        "route_ids":
                            {route_id},

                        "synthetic": True
                    }

            # -------------------------------------------------
            # Synthetic intermediate pole
            # -------------------------------------------------

            else:

                pole_id = new_pole_id()

                x, y = interpolate_on_route(
                    distance_ft,
                    segments
                )

                poles[pole_id] = {
                    "pole_id": pole_id,

                    "pole_type":
                        "intermediate",

                    "anchor_bus": None,

                    "x": x,
                    "y": y,

                    "route_ids":
                        {route_id},

                    "synthetic": True
                }

            route_poles.append({
                "pole_id": pole_id,
                "distance_ft": distance_ft
            })

        # =====================================================
        # Generate physical spans
        # =====================================================

        for i in range(num_spans):

            span_id = new_span_id()

            start_pole = (
                route_poles[i]["pole_id"]
            )

            end_pole = (
                route_poles[i + 1]["pole_id"]
            )

            start_distance = (
                route_poles[i][
                    "distance_ft"
                ]
            )

            end_distance = (
                route_poles[i + 1][
                    "distance_ft"
                ]
            )

            actual_span_length = (
                end_distance
                - start_distance
            )

            spans.append({
                "span_id": span_id,

                "route_id": route_id,

                "start_pole":
                    start_pole,

                "end_pole":
                    end_pole,

                "length_ft":
                    actual_span_length,

                "voltage_class":
                    route.voltage_classes,

                "construction_phases":
                    route.construction_phases,

                "within_preferred_range":
                    plan[
                        "within_preferred_range"
                    ],

                "exception_reason":
                    plan[
                        "exception_reason"
                    ]
            })

            # =================================================
            # Map this physical span to overlapping DSS lines
            # =================================================

            for segment in segments:

                overlap_start = max(
                    start_distance,
                    segment[
                        "start_distance_ft"
                    ]
                )

                overlap_end = min(
                    end_distance,
                    segment[
                        "end_distance_ft"
                    ]
                )

                overlap_ft = (
                    overlap_end
                    - overlap_start
                )

                if overlap_ft > 1e-8:

                    span_line_map.append({
                        "span_id":
                            span_id,

                        "route_id":
                            route_id,

                        "line_name":
                            segment[
                                "line_name"
                            ],

                        "overlap_ft":
                            overlap_ft
                    })

    # =========================================================
    # Convert poles to DataFrame
    # =========================================================

    pole_rows = []

    for pole in poles.values():

        pole_rows.append({
            "pole_id":
                pole["pole_id"],

            "pole_type":
                pole["pole_type"],

            "anchor_bus":
                pole["anchor_bus"],

            "x":
                pole["x"],

            "y":
                pole["y"],

            "route_ids":
                ";".join(
                    sorted(
                        pole["route_ids"]
                    )
                ),

            "synthetic":
                pole["synthetic"]
        })

    poles_df = pd.DataFrame(
        pole_rows
    )

    spans_df = pd.DataFrame(
        spans
    )

    span_line_map_df = pd.DataFrame(
        span_line_map
    )

    # =========================================================
    # Add convenient DSS-line list to spans
    # =========================================================

    span_line_lists = (
        span_line_map_df
        .groupby("span_id")["line_name"]
        .apply(
            lambda x: ";".join(
                sorted(set(x))
            )
        )
    )

    spans_df[
        "dss_lines"
    ] = (
        spans_df["span_id"]
        .map(span_line_lists)
    )

    # =========================================================
    # Build pole -> DSS line mapping
    #
    # Pole failure affects all physical spans touching
    # that pole, so collect the DSS lines associated with
    # all incident spans.
    # =========================================================

    span_to_lines = (
        span_line_map_df
        .groupby("span_id")["line_name"]
        .apply(set)
        .to_dict()
    )

    pole_to_spans = {}

    for span in spans_df.itertuples(
        index=False
    ):

        pole_to_spans.setdefault(
            span.start_pole,
            set()
        ).add(span.span_id)

        pole_to_spans.setdefault(
            span.end_pole,
            set()
        ).add(span.span_id)

    pole_line_rows = []

    connected_span_lists = {}
    connected_line_lists = {}

    for pole_id in poles_df["pole_id"]:

        incident_spans = (
            pole_to_spans.get(
                pole_id,
                set()
            )
        )

        affected_lines = set()

        for span_id in incident_spans:

            affected_lines.update(
                span_to_lines.get(
                    span_id,
                    set()
                )
            )

        connected_span_lists[
            pole_id
        ] = ";".join(
            sorted(incident_spans)
        )

        connected_line_lists[
            pole_id
        ] = ";".join(
            sorted(affected_lines)
        )

        for line_name in sorted(
            affected_lines
        ):

            pole_line_rows.append({
                "pole_id":
                    pole_id,

                "line_name":
                    line_name
            })

    poles_df[
        "connected_span_ids"
    ] = (
        poles_df["pole_id"]
        .map(connected_span_lists)
    )

    poles_df[
        "connected_dss_lines"
    ] = (
        poles_df["pole_id"]
        .map(connected_line_lists)
    )

    pole_line_map_df = pd.DataFrame(
        pole_line_rows
    )

    return (
        poles_df,
        spans_df,
        span_line_map_df,
        pole_line_map_df
    )