#creates synthetic poles and spans

import math
import pandas as pd


def generate_poles_and_spans(
    overhead_lines,
    feeder_name,
    max_span_m=60.0
):
    """
    Generate synthetic poles and conductor spans
    for primary overhead OpenDSS lines.
    """

    poles = []
    spans = []

    # Remember which pole belongs to each OpenDSS bus.
    # This prevents duplicate poles at shared buses.
    bus_to_pole = {}

    pole_counter = 1
    span_counter = 1


    def new_pole_id():
        nonlocal pole_counter

        pole_id = f"{feeder_name}_P{pole_counter:04d}"
        pole_counter += 1

        return pole_id


    def new_span_id():
        nonlocal span_counter

        span_id = f"{feeder_name}_S{span_counter:04d}"
        span_counter += 1

        return span_id


    # Go through every primary OH OpenDSS line
    for _, line in overhead_lines.iterrows():

        line_name = line["line_name"]

        bus1 = line["bus1_name"]
        bus2 = line["bus2_name"]

        length_m = line["length_km"] * 1000.0

        x1 = line["bus1_x"]
        y1 = line["bus1_y"]

        x2 = line["bus2_x"]
        y2 = line["bus2_y"]

        linecode = line["linecode"]


        # -------------------------
        # Basic checks
        # -------------------------

        if length_m <= 0:
            raise ValueError(
                f"Line {line_name} has invalid length: {length_m} m"
            )

        if (
            pd.isna(x1)
            or pd.isna(y1)
            or pd.isna(x2)
            or pd.isna(y2)
        ):
            raise ValueError(
                f"Line {line_name} is missing coordinates."
            )


        # -------------------------
        # Pole at Bus 1
        # -------------------------

        if bus1 not in bus_to_pole:

            pole_id = new_pole_id()

            bus_to_pole[bus1] = pole_id

            poles.append({
                "pole_id": pole_id,
                "pole_type": "bus",
                "bus_name": bus1,
                "x": x1,
                "y": y1,
                "synthetic": True
            })

        start_pole = bus_to_pole[bus1]


        # -------------------------
        # Determine number of spans
        # -------------------------

        n_spans = max(
            1,
            math.ceil(length_m / max_span_m)
        )

        span_length_m = length_m / n_spans

        # All poles belonging to this OpenDSS line
        line_poles = [start_pole]


        # -------------------------
        # Intermediate poles
        # -------------------------

        for i in range(1, n_spans):

            fraction = i / n_spans

            # Place pole between the two bus coordinates
            pole_x = x1 + fraction * (x2 - x1)
            pole_y = y1 + fraction * (y2 - y1)

            pole_id = new_pole_id()

            poles.append({
                "pole_id": pole_id,
                "pole_type": "intermediate",
                "bus_name": None,
                "x": pole_x,
                "y": pole_y,
                "synthetic": True
            })

            line_poles.append(pole_id)


        # -------------------------
        # Pole at Bus 2
        # -------------------------

        if bus2 not in bus_to_pole:

            pole_id = new_pole_id()

            bus_to_pole[bus2] = pole_id

            poles.append({
                "pole_id": pole_id,
                "pole_type": "bus",
                "bus_name": bus2,
                "x": x2,
                "y": y2,
                "synthetic": True
            })

        end_pole = bus_to_pole[bus2]

        line_poles.append(end_pole)


        # -------------------------
        # Create conductor spans
        # -------------------------

        for i in range(len(line_poles) - 1):

            spans.append({
                "span_id": new_span_id(),
                "line_name": line_name,
                "start_pole": line_poles[i],
                "end_pole": line_poles[i + 1],
                "length_m": span_length_m,
                "linecode": linecode
            })


    poles_df = pd.DataFrame(poles)
    spans_df = pd.DataFrame(spans)

    return poles_df, spans_df


# ==========================================================
# VALIDATION
# ==========================================================

def validate_pole_model(
    overhead_lines,
    poles_df,
    spans_df,
    max_span_m
):
    """
    Check that the generated pole/span model makes sense.
    """

    errors = []


    # -------------------------
    # 1. No span exceeds limit
    # -------------------------

    too_long = spans_df[
        spans_df["length_m"] > max_span_m + 1e-6
    ]

    if len(too_long) > 0:
        errors.append(
            f"{len(too_long)} spans exceed {max_span_m} m."
        )


    # -------------------------
    # 2. Spans reference real poles
    # -------------------------

    valid_poles = set(poles_df["pole_id"])

    bad_start = spans_df[
        ~spans_df["start_pole"].isin(valid_poles)
    ]

    bad_end = spans_df[
        ~spans_df["end_pole"].isin(valid_poles)
    ]

    if len(bad_start) > 0:
        errors.append(
            f"{len(bad_start)} spans have invalid start poles."
        )

    if len(bad_end) > 0:
        errors.append(
            f"{len(bad_end)} spans have invalid end poles."
        )


    # -------------------------
    # 3. Every OH line has spans
    # -------------------------

    lines_with_spans = set(spans_df["line_name"])

    missing_lines = overhead_lines[
        ~overhead_lines["line_name"].isin(lines_with_spans)
    ]

    if len(missing_lines) > 0:
        errors.append(
            f"{len(missing_lines)} OH lines generated no spans."
        )


    # -------------------------
    # 4. Span lengths reproduce
    # original OpenDSS line length
    # -------------------------

    original_lengths = (
        overhead_lines
        .set_index("line_name")["length_km"]
        * 1000.0
    )

    generated_lengths = (
        spans_df
        .groupby("line_name")["length_m"]
        .sum()
    )

    for line_name, original_length in original_lengths.items():

        generated_length = generated_lengths.get(
            line_name,
            0.0
        )

        if abs(original_length - generated_length) > 1e-6:

            errors.append(
                f"{line_name}: "
                f"original = {original_length:.6f} m, "
                f"generated = {generated_length:.6f} m"
            )


    # -------------------------
    # 5. One bus = one bus pole
    # -------------------------

    bus_poles = poles_df[
        poles_df["pole_type"] == "bus"
    ]

    duplicate_bus_poles = (
        bus_poles["bus_name"]
        .dropna()
        .duplicated()
        .sum()
    )

    if duplicate_bus_poles > 0:

        errors.append(
            f"{duplicate_bus_poles} buses map to "
            "more than one pole."
        )


    # -------------------------
    # Final result
    # -------------------------

    if errors:

        print("\nPole model validation FAILED:")

        for error in errors:
            print("-", error)

        return False


    print("\nPole model validation PASSED")
    print("All spans <=", max_span_m, "m")
    print("All span pole references are valid")
    print("Every OH line has at least one span")
    print("Span lengths reproduce OpenDSS line lengths")
    print("Each bus maps to one endpoint pole")

    return True