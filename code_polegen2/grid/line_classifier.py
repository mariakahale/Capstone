import pandas as pd


# ---------------------------------------------------------
# OpenDSS unit conversion factors to feet
# ---------------------------------------------------------

UNIT_TO_FEET = {
    1: 5280.0,          # miles
    2: 1000.0,          # kft
    3: 3280.839895,     # km
    4: 3.280839895,     # metres
    5: 1.0,             # feet
    6: 1.0 / 12.0,      # inches
    7: 1.0 / 30.48,     # centimetres
    8: 1.0 / 304.8      # millimetres
}


# ---------------------------------------------------------
# OpenDSS Line objects that are devices rather than
# physical conductor sections
# ---------------------------------------------------------

NON_CONDUCTOR_PATTERNS = [
    "padswitch",
    "fuse",
    "elbswitch",
    "disconnect"
]


def get_overhead_lines(lines):
    """
    Keep physical overhead conductor sections.

    SMART-DS linecodes identify overhead construction
    using '_oh_' in the linecode name.

    Device-like OpenDSS Line objects such as switches,
    fuses, and disconnects are removed.
    """

    overhead = lines[
        lines["linecode"].str.contains(
            "_oh_",
            case=False,
            na=False,
            regex=False
        )
    ].copy()

    pattern = "|".join(NON_CONDUCTOR_PATTERNS)

    overhead = overhead[
        ~overhead["line_name"].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    return overhead


def add_length_ft(lines):
    """
    Convert OpenDSS line length to feet.

    Lines with unsupported or unspecified unit codes
    receive a missing length value.
    """

    lines = lines.copy()

    def convert(row):

        unit_code = int(row["length_units_code"])

        conversion = UNIT_TO_FEET.get(unit_code)

        if conversion is None:
            return None

        return row["length_raw"] * conversion

    lines["length_ft"] = lines.apply(
        convert,
        axis=1
    )

    return lines


def add_voltage_class(lines):
    """
    Classify conductors as primary, secondary, mixed,
    or unknown based on OpenDSS bus voltage bases.

    This keeps the classification generic rather than
    hardcoding one exact feeder voltage.
    """

    lines = lines.copy()

    def classify(row):

        kv1 = row["bus1_kvbase"]
        kv2 = row["bus2_kvbase"]

        if pd.isna(kv1) or pd.isna(kv2):
            return "unknown"

        # Primary distribution buses are above 1 kV L-N.
        if kv1 >= 1.0 and kv2 >= 1.0:
            return "primary"

        # Secondary buses are below 1 kV.
        if kv1 < 1.0 and kv2 < 1.0:
            return "secondary"

        # Connection between voltage levels
        return "mixed"

    lines["voltage_class"] = lines.apply(
        classify,
        axis=1
    )

    return lines


def add_construction_phases(lines):
    """
    Extract the construction phase designation from
    SMART-DS linecode names.

    Examples:
        1p_oh_... -> 1
        2p_oh_... -> 2
        3p_oh_... -> 3

    This is stored separately from the raw OpenDSS
    CktElement.NumPhases() value.
    """

    lines = lines.copy()

    def get_phases(linecode):

        name = str(linecode).lower()

        if name.startswith("1p_"):
            return 1

        if name.startswith("2p_"):
            return 2

        if name.startswith("3p_"):
            return 3

        return None

    lines["construction_phases"] = (
        lines["linecode"]
        .apply(get_phases)
    )

    return lines


def find_parallel_lines(lines):
    """
    Find cases where more than one OpenDSS line connects
    the same two buses.

    Bus order is ignored:
        A -> B
    and
        B -> A

    are treated as the same bus pair.
    """

    check = lines.copy()

    check["bus_pair"] = check.apply(
        lambda row: tuple(
            sorted(
                [
                    row["bus1_name"],
                    row["bus2_name"]
                ]
            )
        ),
        axis=1
    )

    counts = (
        check
        .groupby("bus_pair")
        .size()
        .reset_index(name="line_count")
    )

    parallel_pairs = counts[
        counts["line_count"] > 1
    ].copy()

    return parallel_pairs