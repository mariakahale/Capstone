#identifies primary/OH/UG

import math

# OpenDSS elements that are represented as Lines,
# but are not actual conductor sections

NON_CONDUCTOR_PATTERNS = [
    "padswitch",
    "fuse",
    "elbswitch"
]


def get_primary_lines(lines, primary_kv_ll, tolerance=0.05):
    """
    Return line objects whose two endpoint buses are at the
    feeder's primary voltage.

    primary_kv_ll:
        nominal line-to-line voltage, e.g. 12.47 kV

    OpenDSS Bus.kVBase() returns line-to-neutral voltage,
    so we convert 12.47 kV LL -> about 7.2 kV LN.
    """

    primary_kv_ln = primary_kv_ll / math.sqrt(3)

    lower = primary_kv_ln * (1 - tolerance)
    upper = primary_kv_ln * (1 + tolerance)

    primary = lines[
        lines["bus1_kvbase"].between(lower, upper)
        &
        lines["bus2_kvbase"].between(lower, upper)
    ].copy()

    return primary


def remove_non_conductors(lines):
    """
    Remove switches, fuses, etc. that OpenDSS stores
    as Line objects.
    """

    pattern = "|".join(NON_CONDUCTOR_PATTERNS)

    conductors = lines[
        ~lines["linecode"].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    return conductors


def get_overhead_lines(lines):
    """
    Return overhead conductor sections.
    """

    return lines[
        lines["linecode"].str.contains(
            "_oh_",
            case=False,
            na=False,
            regex=False
        )
    ].copy()


def get_underground_lines(lines):
    """
    Return underground conductor sections.
    """

    return lines[
        lines["linecode"].str.contains(
            "_ug_",
            case=False,
            na=False,
            regex=False
        )
    ].copy()


def get_unclassified_lines(lines):
    """
    Find physical conductor rows that are neither marked OH nor UG.

    Useful as a validation check when processing a new feeder.
    """

    is_oh = lines["linecode"].str.contains(
        "_oh_",
        case=False,
        na=False,
        regex=False
    )

    is_ug = lines["linecode"].str.contains(
        "_ug_",
        case=False,
        na=False,
        regex=False
    )

    return lines[~is_oh & ~is_ug].copy()