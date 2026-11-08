"""
Pole/span generator (polegen2).

For every feeder in config.FEEDERS and every area type in
config.AREA_TYPES, build a synthetic pole/span model from the
feeder's OpenDSS overhead lines and save it as CSV files.

Usage:
    python main.py                              # all feeders, all area types
    python main.py --feeders 1878               # one feeder, all area types
    python main.py --areas rural                # all feeders, rural only
    python main.py --feeders 1878 3777 --areas suburban rural
"""

import argparse

from config import (
    FEEDERS,
    FALLBACK_BUSCOORDS,
    OUTPUT_DIR,
    AREA_TYPES,
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


def build_feeder_routes(feeder_name, master_path):
    """
    Steps 1-4: everything that does NOT depend on span length.

    Load the OpenDSS feeder, keep the overhead conductors,
    build the overhead topology and trace anchor-to-anchor
    routes. This is done once per feeder and shared by every
    area type.
    """

    # =========================================================
    # 1. Load OpenDSS feeder
    # =========================================================

    print(f"\n==============================")
    print(f"Processing feeder: {feeder_name}")
    print(f"Master file: {master_path}")

    if not master_path.exists():
        raise FileNotFoundError(
            f"OpenDSS master file not found: {master_path}"
        )

    load_feeder(
        master_path,
        fallback_buscoords=FALLBACK_BUSCOORDS.get(feeder_name)
    )

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

    # Span-independent outputs: saved once per feeder
    overhead.to_csv(
        OUTPUT_DIR / f"feeder_{feeder_name}_overhead.csv",
        index=False
    )

    routes.to_csv(
        OUTPUT_DIR / f"feeder_{feeder_name}_routes.csv",
        index=False
    )

    print(f"OH conductors: {len(overhead)}")
    print(f"Routes: {len(routes)}")

    return overhead, routes, bus_data, components


def generate_area_version(
    feeder_name,
    area_type,
    overhead,
    routes,
    bus_data,
    components
):
    """
    Steps 5-8: everything that DOES depend on span length.

    Split every route into equal spans using the span range
    for this area type, place poles, validate, and save the
    area-specific CSV files.
    """

    print(f"\n--- Feeder {feeder_name}: {area_type} ---")

    # =========================================================
    # 5. Determine equal span plan
    # =========================================================

    if area_type not in SPAN_RANGES_FT:
        raise ValueError(
            f"Unknown area type: {area_type}. "
            f"Add it to SPAN_RANGES_FT in config.py."
        )

    min_span_ft, max_span_ft = (
        SPAN_RANGES_FT[area_type]
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
        feeder_name=feeder_name
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
    # 8. Save area-specific outputs
    # =========================================================

    tables = {
        "span_plan": span_plan,
        "poles": poles,
        "spans": spans,
        "span_line_map": span_line_map,
        "pole_line_map": pole_line_map
    }

    for table_name, table in tables.items():
        table.to_csv(
            OUTPUT_DIR /
            f"feeder_{feeder_name}_{table_name}_{area_type}.csv",
            index=False
        )

    print(
        f"Preferred span range: "
        f"{min_span_ft:.0f}-{max_span_ft:.0f} ft"
    )
    print(f"Poles: {len(poles)}")
    print(f"Spans: {len(spans)}")
    print(
        "Exception spans:",
        (~spans["within_preferred_range"]).sum()
    )

    return {
        "feeder": feeder_name,
        "area_type": area_type,
        "poles": len(poles),
        "spans": len(spans),
        "exception_spans":
            int((~spans["within_preferred_range"]).sum())
    }


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate synthetic poles and spans "
                    "for each feeder and area type."
    )

    parser.add_argument(
        "--feeders",
        nargs="+",
        default=list(FEEDERS.keys()),
        help=f"Feeders to process (default: all of "
             f"{list(FEEDERS.keys())})"
    )

    parser.add_argument(
        "--areas",
        nargs="+",
        default=AREA_TYPES,
        help=f"Area types to generate (default: {AREA_TYPES})"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    unknown = [f for f in args.feeders if f not in FEEDERS]
    if unknown:
        raise ValueError(
            f"Unknown feeder(s): {unknown}. "
            f"Add them to FEEDERS in config.py."
        )

    summary = []

    for feeder_name in args.feeders:

        overhead, routes, bus_data, components = (
            build_feeder_routes(
                feeder_name,
                FEEDERS[feeder_name]
            )
        )

        for area_type in args.areas:

            summary.append(
                generate_area_version(
                    feeder_name,
                    area_type,
                    overhead,
                    routes,
                    bus_data,
                    components
                )
            )

    # =========================================================
    # 9. Final summary
    # =========================================================

    print("\n==============================")
    print("--- MODEL COMPLETE ---")

    for row in summary:
        print(
            f"Feeder {row['feeder']:>6} | "
            f"{row['area_type']:<9} | "
            f"poles {row['poles']:>5} | "
            f"spans {row['spans']:>5} | "
            f"exceptions {row['exception_spans']:>4}"
        )

    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
