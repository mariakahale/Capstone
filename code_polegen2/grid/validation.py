def validate_physical_model(
    overhead,
    routes,
    span_plan,
    poles,
    spans,
    span_line_map,
    num_components,
    tolerance=1e-4
):
    """
    Strict validation of the generated pole/span model.

    Raises RuntimeError if a validation check fails.
    """

    errors = []

    # =========================================================
    # 1. Unique IDs
    # =========================================================

    if poles["pole_id"].duplicated().any():
        errors.append("Duplicate pole IDs found.")

    if spans["span_id"].duplicated().any():
        errors.append("Duplicate span IDs found.")

    # =========================================================
    # 2. Every span references valid poles
    # =========================================================

    valid_poles = set(poles["pole_id"])

    invalid_start = (
        ~spans["start_pole"].isin(valid_poles)
    ).sum()

    invalid_end = (
        ~spans["end_pole"].isin(valid_poles)
    ).sum()

    if invalid_start > 0 or invalid_end > 0:
        errors.append(
            "Some spans reference nonexistent poles."
        )

    # No span should connect a pole to itself
    if (
        spans["start_pole"]
        == spans["end_pole"]
    ).any():
        errors.append(
            "A span connects a pole to itself."
        )

    # =========================================================
    # 3. Planned span count matches generated span count
    # =========================================================

    expected_spans = int(
        span_plan["num_spans"].sum()
    )

    if len(spans) != expected_spans:
        errors.append(
            f"Expected {expected_spans} spans, "
            f"generated {len(spans)}."
        )

    # =========================================================
    # 4. Each route has correct number of spans
    # =========================================================

    actual_route_counts = (
        spans
        .groupby("route_id")
        .size()
        .to_dict()
    )

    for plan in span_plan.itertuples(index=False):

        actual = actual_route_counts.get(
            plan.route_id,
            0
        )

        if actual != plan.num_spans:
            errors.append(
                f"{plan.route_id}: expected "
                f"{plan.num_spans} spans, "
                f"found {actual}."
            )

    # =========================================================
    # 5. Span lengths reconstruct every route
    # =========================================================

    route_span_lengths = (
        spans
        .groupby("route_id")["length_ft"]
        .sum()
        .to_dict()
    )

    for route in routes.itertuples(index=False):

        generated_length = (
            route_span_lengths.get(
                route.route_id,
                0.0
            )
        )

        if abs(
            generated_length
            - route.total_length_ft
        ) > tolerance:

            errors.append(
                f"{route.route_id}: generated span "
                f"lengths do not reconstruct route."
            )

    # =========================================================
    # 6. Spans are equal within each route
    # =========================================================

    for route_id, group in spans.groupby(
        "route_id"
    ):

        difference = (
            group["length_ft"].max()
            - group["length_ft"].min()
        )

        if difference > tolerance:
            errors.append(
                f"{route_id}: spans are not equal."
            )

    # =========================================================
    # 7. Preferred-range routes actually satisfy range
    # =========================================================

    plan_lookup = (
        span_plan
        .set_index("route_id")
        .to_dict("index")
    )

    for route_id, group in spans.groupby(
        "route_id"
    ):

        plan = plan_lookup[route_id]

        if plan["within_preferred_range"]:

            minimum = plan["preferred_min_ft"]
            maximum = plan["preferred_max_ft"]

            if (
                (
                    group["length_ft"]
                    < minimum - tolerance
                ).any()
                or
                (
                    group["length_ft"]
                    > maximum + tolerance
                ).any()
            ):

                errors.append(
                    f"{route_id}: marked valid but "
                    f"contains span outside preferred range."
                )

    # =========================================================
    # 8. Every physical span maps to OpenDSS
    # =========================================================

    mapped_spans = set(
        span_line_map["span_id"]
    )

    missing_span_maps = (
        set(spans["span_id"])
        - mapped_spans
    )

    if missing_span_maps:
        errors.append(
            f"{len(missing_span_maps)} spans have "
            f"no OpenDSS line mapping."
        )

    # =========================================================
    # 9. Mapping overlap reconstructs every physical span
    # =========================================================

    mapped_span_lengths = (
        span_line_map
        .groupby("span_id")["overlap_ft"]
        .sum()
        .to_dict()
    )

    for span in spans.itertuples(index=False):

        mapped_length = (
            mapped_span_lengths.get(
                span.span_id,
                0.0
            )
        )

        if abs(
            mapped_length
            - span.length_ft
        ) > tolerance:

            errors.append(
                f"{span.span_id}: OpenDSS overlap "
                f"does not equal physical span length."
            )

    # =========================================================
    # 10. Every OH OpenDSS line appears in mapping
    # =========================================================

    original_lines = set(
        overhead["line_name"]
    )

    mapped_lines = set(
        span_line_map["line_name"]
    )

    missing_lines = (
        original_lines - mapped_lines
    )

    extra_lines = (
        mapped_lines - original_lines
    )

    if missing_lines:
        errors.append(
            f"{len(missing_lines)} OH OpenDSS lines "
            f"are missing from span mapping."
        )

    if extra_lines:
        errors.append(
            f"{len(extra_lines)} mapped lines are "
            f"not in the OH conductor dataset."
        )

    # =========================================================
    # 11. Mapping reconstructs each OpenDSS line length
    # =========================================================

    mapped_line_lengths = (
        span_line_map
        .groupby("line_name")["overlap_ft"]
        .sum()
        .to_dict()
    )

    for line in overhead.itertuples(index=False):

        mapped_length = (
            mapped_line_lengths.get(
                line.line_name,
                0.0
            )
        )

        if abs(
            mapped_length
            - line.length_ft
        ) > tolerance:

            errors.append(
                f"{line.line_name}: mapped overlap "
                f"does not reconstruct DSS line length."
            )

    # =========================================================
    # 12. Every pole has coordinates
    # =========================================================

    if poles[["x", "y"]].isna().any().any():
        errors.append(
            "Some generated poles have missing coordinates."
        )

    # =========================================================
    # 13. Forest topology check
    #
    # For a collection of trees:
    # nodes = edges + components
    # =========================================================

    expected_poles = (
        len(spans)
        + num_components
    )

    if len(poles) != expected_poles:
        errors.append(
            f"Forest check failed: expected "
            f"{expected_poles} poles, "
            f"found {len(poles)}."
        )

    # =========================================================
    # Result
    # =========================================================

    if errors:

        print("\nPHYSICAL MODEL VALIDATION: FAILED")

        for error in errors:
            print(" -", error)

        raise RuntimeError(
            "Physical model validation failed."
        )

    print("\n--- PHYSICAL MODEL VALIDATION ---")
    print("Unique pole IDs: PASSED")
    print("Unique span IDs: PASSED")
    print("Span-to-pole references: PASSED")
    print("Planned span counts: PASSED")
    print("Route length reconstruction: PASSED")
    print("Equal spans within routes: PASSED")
    print("Preferred span ranges: PASSED")
    print("Span-to-DSS mapping: PASSED")
    print("DSS line length reconstruction: PASSED")
    print("Pole coordinates: PASSED")
    print("Radial forest topology: PASSED")

    print("\nPHYSICAL MODEL VALIDATION: PASSED")