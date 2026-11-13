"""
weather_input.py
================
The ONE file the pipeline uses to get weather from code_weather.
Every function that gives the pipeline a storm returns the same thing:

    (ice_mm, wind_ms)

    ice_mm  : radial ice on the wire, in MILLIMETRES. 0 = wind-only storm.
              (The event spreadsheets use cm -> multiply by 10 here, not in the pipeline.)
    wind_ms : 3-second gust, in m/s.
              Ice storm (ice_mm > 0): the gust DURING the ice ("gust while iced").
              Wind storm (ice_mm = 0): the strongest gust.

That is the whole contract. pipeline/run_fragility.py only ever sees these two numbers;
it does not know or care which source they came from.

THE THREE SOURCES
    Option 1 - the user's own weather CSV   list_storms_from_csv()   -> storm_from_csv()
    Option 2 - historical storms (catalogue) list_historical_storms() -> historical_storm()
    Option 3 - numbers typed by the user     user_storm()

Options 1 and 2 have two steps because a CSV or a catalogue holds MANY storms: first list
them (the GUI shows this as a dropdown), then get the one the user picked.

STATUS
    Option 3: done.
    Options 1 and 2: stubs -- Tara implements them. Keep the names, inputs and outputs the
    same; only replace the inside. Then run   python code_weather/weather_input.py
    to check the outputs follow the contract.

Ranges the fragility curves cover (the lookups warn outside these; not an error here):
    ice storm : ice 0-150 mm, wind 0-40 m/s   (Ma et al. Fig. 7)
    wind storm: wind 20-90 m/s                (Ma et al. Fig. 4)
"""

import math

# Columns every storm-list table must have (options 1 and 2), in this order.
STORM_LIST_COLUMNS = ["storm_id", "name", "start", "end", "type", "ice_mm", "wind_ms"]
#   storm_id : unique ID, used to pick the storm (e.g. "TOR-2013-12-21")
#   name     : readable label for the GUI (e.g. "Dec 2013 ice storm")
#   start/end: local date-time of the storm
#   type     : "ice", "wind" or "ice + wind"
#   ice_mm, wind_ms : the same two numbers the storm_from_... function returns


# =============================================================================
# CONTRACT CHECK -- every source returns through this
# =============================================================================
def check_weather(ice_mm, wind_ms):
    """Make sure (ice_mm, wind_ms) follows the contract. Returns them as floats."""
    for name, value in (("ice_mm", ice_mm), ("wind_ms", wind_ms)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {value!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return float(ice_mm), float(wind_ms)


# =============================================================================
# OPTION 3 -- the user types the numbers (done)
# =============================================================================
def user_storm(ice_mm, wind_ms):
    """Storm typed in by the user (GUI boxes, or the settings in run_fragility.py).
    Returns (ice_mm, wind_ms)."""
    return check_weather(ice_mm, wind_ms)


# =============================================================================
# OPTION 2 -- historical storms (TODO: Tara)
# =============================================================================
def list_historical_storms():
    """All storms in the historical catalogue, for the GUI dropdown.

    Returns a pandas DataFrame with the columns in STORM_LIST_COLUMNS, one row per storm.
    Ideas for the catalogue: the Dec 2013 Toronto ice storm, Event #1, Event #2 (Dec 29, 2025),
    and storms from the events_toronto_*.xlsx files.
    """
    raise NotImplementedError("list_historical_storms(): not implemented yet (Tara).")


def historical_storm(storm_id):
    """One storm from the historical catalogue, picked by its storm_id.

    Returns (ice_mm, wind_ms) -- finish with:  return check_weather(ice_mm, wind_ms)
    """
    raise NotImplementedError(
        f"historical_storm({storm_id!r}): not implemented yet (Tara).")


# =============================================================================
# OPTION 1 -- the user's own weather CSV (TODO: Tara)
# =============================================================================
def list_storms_from_csv(csv_path):
    """Every storm found in the user's hourly weather CSV, for the GUI dropdown.

    csv_path : hourly weather CSV in the format era5_download.py makes
               (time, latitude, longitude, wind_speed_ms, wind_gust_ms, temp_c, precip_mm, ptype)
    Returns a pandas DataFrame with the columns in STORM_LIST_COLUMNS, one row per storm.
    Likely built on find_weather_events.py (it already finds the storms and their
    peak ice / gust while iced / strongest gust -- in cm, so x10 for ice_mm).
    """
    raise NotImplementedError(
        f"list_storms_from_csv({csv_path!r}): not implemented yet (Tara).")


def storm_from_csv(csv_path, storm_id):
    """One storm from the user's CSV, picked by its storm_id from list_storms_from_csv().

    Returns (ice_mm, wind_ms) -- finish with:  return check_weather(ice_mm, wind_ms)
    """
    raise NotImplementedError(
        f"storm_from_csv({csv_path!r}, {storm_id!r}): not implemented yet (Tara).")


# =============================================================================
# SELF-CHECK:  python code_weather/weather_input.py
# =============================================================================
if __name__ == "__main__":
    def report(label, fn, *args):
        try:
            out = fn(*args)
        except NotImplementedError as e:
            print(f"  [todo] {e}")
            return None
        except Exception as e:                       # noqa: BLE001
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
            return None
        print(f"  [ ok ] {label}")
        return out

    def check_list(label, table):
        if table is None:
            return
        missing = [c for c in STORM_LIST_COLUMNS if c not in table.columns]
        if missing:
            print(f"  [FAIL] {label}: missing columns {missing}")
        elif table["storm_id"].duplicated().any():
            print(f"  [FAIL] {label}: storm_id values are not unique")
        else:
            print(f"         {len(table)} storms")

    print("Option 3 (user):")
    print("   ", report("user_storm(17.3, 14.9)", user_storm, 17.3, 14.9))
    for bad in [(-1, 10), (5, "abc"), (float("nan"), 10)]:
        try:
            user_storm(*bad)
            print(f"  [FAIL] user_storm{bad} should have been rejected")
        except ValueError:
            print(f"  [ ok ] user_storm{bad} rejected")

    print("Option 2 (historical):")
    table = report("list_historical_storms()", list_historical_storms)
    check_list("list_historical_storms()", table)
    if table is not None and len(table):
        sid = table["storm_id"].iloc[0]
        print("   ", report(f"historical_storm({sid!r})", historical_storm, sid))

    print("Option 1 (CSV):")
    from pathlib import Path
    csv = Path(__file__).resolve().parent / "era5_toronto_last_5_years.csv"
    table = report("list_storms_from_csv(...)", list_storms_from_csv, csv)
    check_list("list_storms_from_csv(...)", table)
    if table is not None and len(table):
        sid = table["storm_id"].iloc[0]
        print("   ", report(f"storm_from_csv(..., {sid!r})", storm_from_csv, csv, sid))
