import math
import pandas as pd


def choose_span_plan(
    route_length_ft,
    min_span_ft,
    max_span_ft
):
    """
    Choose an integer number of equal spans for one route.

    Preferred behaviour:
        Every span falls between min_span_ft and max_span_ft.

    If that is mathematically impossible:
        Preserve the real route length,
        choose the closest equal-span solution,
        and flag it as an exception.
    """

    if route_length_ft <= 0:
        raise ValueError(
            "Route length must be greater than zero."
        )

    target_span_ft = (
        min_span_ft + max_span_ft
    ) / 2.0

    # ---------------------------------------------------------
    # Determine whether an exact valid division exists
    # ---------------------------------------------------------

    min_num_spans = max(
        1,
        math.ceil(
            route_length_ft / max_span_ft
        )
    )

    max_num_spans = math.floor(
        route_length_ft / min_span_ft
    )

    # ---------------------------------------------------------
    # Normal case:
    # at least one integer number of spans satisfies the range
    # ---------------------------------------------------------

    if min_num_spans <= max_num_spans:

        candidates = range(
            min_num_spans,
            max_num_spans + 1
        )

        num_spans = min(
            candidates,
            key=lambda n: abs(
                route_length_ft / n
                - target_span_ft
            )
        )

        span_length_ft = (
            route_length_ft / num_spans
        )

        return {
            "num_spans": num_spans,
            "span_length_ft": span_length_ft,
            "within_preferred_range": True,
            "exception_reason": ""
        }

    # ---------------------------------------------------------
    # Exception 1:
    # entire route is shorter than minimum preferred span
    # ---------------------------------------------------------

    if route_length_ft < min_span_ft:

        return {
            "num_spans": 1,
            "span_length_ft": route_length_ft,
            "within_preferred_range": False,
            "exception_reason":
                "route_shorter_than_minimum"
        }

    # ---------------------------------------------------------
    # Exception 2:
    # route is long enough, but no integer number of equal
    # spans can fit completely inside the preferred range.
    #
    # Example suburban:
    # route = 170 ft
    #
    # 1 span = 170 ft  (>150)
    # 2 spans = 85 ft  (<100)
    #
    # Neither is valid.
    # ---------------------------------------------------------

    max_candidate_n = max(
        2,
        math.ceil(
            route_length_ft / min_span_ft
        ) + 1
    )

    candidates = range(
        1,
        max_candidate_n + 1
    )

    def candidate_score(n):

        span_length = route_length_ft / n

        # Distance outside allowed range
        if span_length < min_span_ft:
            range_violation = (
                min_span_ft - span_length
            )

        elif span_length > max_span_ft:
            range_violation = (
                span_length - max_span_ft
            )

        else:
            range_violation = 0.0

        # Secondary preference:
        # closest to midpoint of desired range
        midpoint_difference = abs(
            span_length - target_span_ft
        )

        return (
            range_violation,
            midpoint_difference
        )

    num_spans = min(
        candidates,
        key=candidate_score
    )

    span_length_ft = (
        route_length_ft / num_spans
    )

    return {
        "num_spans": num_spans,
        "span_length_ft": span_length_ft,
        "within_preferred_range": False,
        "exception_reason":
            "no_feasible_equal_division"
    }


def build_span_plan(
    routes,
    min_span_ft,
    max_span_ft
):
    """
    Determine the equal-span plan for every OH route.
    """

    rows = []

    for route in routes.itertuples(
        index=False
    ):

        plan = choose_span_plan(
            route_length_ft=route.total_length_ft,
            min_span_ft=min_span_ft,
            max_span_ft=max_span_ft
        )

        rows.append({
            "route_id": route.route_id,

            "voltage_classes":
                route.voltage_classes,

            "construction_phases":
                route.construction_phases,

            "route_length_ft":
                route.total_length_ft,

            "preferred_min_ft":
                min_span_ft,

            "preferred_max_ft":
                max_span_ft,

            "num_spans":
                plan["num_spans"],

            "span_length_ft":
                plan["span_length_ft"],

            "within_preferred_range":
                plan["within_preferred_range"],

            "exception_reason":
                plan["exception_reason"]
        })

    return pd.DataFrame(rows)