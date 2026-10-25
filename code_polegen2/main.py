from config import (
    MASTER_PATH,
    OUTPUT_DIR,
    FEEDER_NAME,
    AREA_TYPE,
    SPAN_RANGES_FT
)

from grid.feeder import (
    load_feeder,
    extract_lines
)

from grid.line_classifier import (
    get_overhead_lines,
    add_length_ft,
    add_voltage_class,
    add_construction_phases,
    find_parallel_lines
)

from grid.topology import (
    build_overhead_graph,
    find_connected_components
)

from grid.route_builder import (
    get_anchor_reasons,
    trace_routes
)

from grid.span_planner import (
    build_span_plan
)

from grid.pole_span_generator import (
    generate_poles_and_spans
)

from grid.validation import (
    validate_physical_model
)


def main():

    # =========================================================
    # 1. Load OpenDSS feeder
    # =========================================================

    print(f"\nProcessing feeder: {FEEDER_NAME}")
    print(f"Area type: {AREA_TYPE}")

    load_feeder(MASTER_PATH)

    # =========================================================
    # 2. Extract and classify overhead conductors
    # =========================================================

    lines = extract_lines()

    overhead = get_overhead_lines(lines)
    overhead = add_length_ft(overhead)
    overhead = add_voltage_class(overhead)
    overhead = add_construction_phases(overhead)

    # Basic input checks
    if overhead["length_ft"].isna().any():
        raise RuntimeError(
            "Some overhead conductors do not have "
            "convertible length values."
        )

    if overhead[
        [
            "bus1_x",
            "bus1_y",
            "bus2_x",
            "bus2_y"
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Some overhead conductors are missing coordinates."
        )

    parallel_pairs = find_parallel_lines(overhead)

    if len(parallel_pairs) > 0:
        raise RuntimeError(
            "Parallel overhead lines were detected. "
            "Current route-building logic assumes one "
            "OH line per bus pair."
        )

    # =========================================================
    # 3. Build overhead topology
    # =========================================================

    graph, bus_data = build_overhead_graph(
        overhead
    )

    components = find_connected_components(
        graph
    )

    # =========================================================
    # 4. Identify route anchors and trace continuous routes
    # =========================================================

    anchors = get_anchor_reasons(
        graph
    )

    routes, untraced_lines = trace_routes(
        graph,
        anchors
    )

    if len(untraced_lines) > 0:
        raise RuntimeError(
            f"{len(untraced_lines)} OH lines were "
            f"not assigned to a route."
        )

    # =========================================================
    # 5. Determine equal span plan
    # =========================================================

    if AREA_TYPE not in SPAN_RANGES_FT:
        raise ValueError(
            f"Unknown AREA_TYPE: {AREA_TYPE}"
        )

    min_span_ft, max_span_ft = (
        SPAN_RANGES_FT[AREA_TYPE]
    )

    span_plan = build_span_plan(
        routes=routes,
        min_span_ft=min_span_ft,
        max_span_ft=max_span_ft
    )

    # =========================================================
    # 6. Generate poles, spans and OpenDSS mappings
    # =========================================================

    (
        poles,
        spans,
        span_line_map,
        pole_line_map
    ) = generate_poles_and_spans(
        routes=routes,
        overhead=overhead,
        span_plan=span_plan,
        bus_data=bus_data,
        feeder_name=FEEDER_NAME
    )

    # =========================================================
    # 7. Validate generated physical model
    # =========================================================

    validate_physical_model(
        overhead=overhead,
        routes=routes,
        span_plan=span_plan,
        poles=poles,
        spans=spans,
        span_line_map=span_line_map,
        num_components=len(components)
    )

    # =========================================================
    # 8. Save outputs
    # =========================================================

    output_files = {
        "overhead":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_overhead.csv",

        "routes":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_routes.csv",

        "span_plan":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_span_plan_{AREA_TYPE}.csv",

        "poles":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_poles_{AREA_TYPE}.csv",

        "spans":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_spans_{AREA_TYPE}.csv",

        "span_line_map":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_span_line_map_{AREA_TYPE}.csv",

        "pole_line_map":
            OUTPUT_DIR /
            f"feeder_{FEEDER_NAME}_pole_line_map_{AREA_TYPE}.csv"
    }

    overhead.to_csv(
        output_files["overhead"],
        index=False
    )

    routes.to_csv(
        output_files["routes"],
        index=False
    )

    span_plan.to_csv(
        output_files["span_plan"],
        index=False
    )

    poles.to_csv(
        output_files["poles"],
        index=False
    )

    spans.to_csv(
        output_files["spans"],
        index=False
    )

    span_line_map.to_csv(
        output_files["span_line_map"],
        index=False
    )

    pole_line_map.to_csv(
        output_files["pole_line_map"],
        index=False
    )

    # =========================================================
    # 9. Final summary
    # =========================================================

    print("\n--- MODEL COMPLETE ---")

    print(
        f"OH conductors: {len(overhead)}"
    )

    print(
        f"Routes: {len(routes)}"
    )

    print(
        f"Poles: {len(poles)}"
    )

    print(
        f"Spans: {len(spans)}"
    )

    print(
        f"Preferred span range: "
        f"{min_span_ft:.0f}-{max_span_ft:.0f} ft"
    )

    print(
        "Exception spans:",
        (
            ~spans["within_preferred_range"]
        ).sum()
    )

    print(
        f"\nOutputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()