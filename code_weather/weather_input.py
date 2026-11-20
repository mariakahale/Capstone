"""
weather_input.py
================
The ONE file for weather. Everything the pipeline needs from code_weather is here.

WHAT IS IN code_weather
    weather_input.py        this file: the location question and the three weather options
    locations.py            each location's ERA5 grid cells, time zone and FEEDER (lookup table)
    historical_storms.xlsx  option 2's lookup table (made by build_historical_table() below)
    find_weather_events.py  finds the storms in an hourly weather CSV (options 1 and 2 use it)
    era5_download.py        downloads ERA5 weather for a location (to get more data)
    check_weather_csv.py    optional: checks the quality of a weather CSV
    era5_toronto_2013_11_2013_12.csv, era5_toronto_last_5_years.csv   the ERA5 data

WHAT THE USER IS ASKED  (python pipeline/run_fragility.py)
    1. The location        ask_location()  -> e.g. "toronto". Its feeder comes from the
                           lookup table in locations.py (Toronto -> feeder 3777) and the
                           user is told which feeder is being run.
    2. The weather source  pipeline/terminal_prompts.py (Maria): user, csv or historical
    3. The storm:
         csv        ask_csv_storm(location)         OPTION 1: the user's own CSV
         historical ask_historical_storm(location)  OPTION 2: 1 of 5 historical storms
         user       terminal_prompts.py asks        OPTION 3: the user types the numbers

THE CONTRACT: every option returns the same two numbers
    ice_mm  : radial ice on the wire, in MILLIMETRES. 0 = wind-only storm.
    wind_ms : 3-second gust, in m/s.
              Ice storm (ice_mm > 0): the gust DURING the ice ("gust while iced").
              Wind storm (ice_mm = 0): the strongest gust.
pipeline/run_fragility.py only ever sees these two numbers.

OPTION 1 - THE USER'S OWN WEATHER CSV
    The user chooses the file (a file window opens) and two thresholds; every storm in the
    file with ice >= the ice threshold OR a gust >= the gust threshold is listed
    (find_weather_events.py), and the user picks one. What the file must have is in
    CSV_REQUIREMENTS below; the user is shown it before choosing the file.
    No latitude/longitude is needed: the file is taken to be for the location the user
    chose. (A file from era5_download.py has a grid of points; then that location's grid
    cells are used.)
    GUI / other code: list_storms_from_csv(csv, location, ice_mm, gust_ms) -> storm_from_csv(...)

OPTION 2 - A HISTORICAL STORM (lookup table historical_storms.xlsx)
    The user picks one of 5 storms (FEATURED_STORMS). The table is made by
    build_historical_table() from the storms find_weather_events.py finds in the ERA5 data
    (HISTORICAL_CSVS) plus MEASURED_STORMS (storms ERA5 misses). Remake it after
    downloading more data or changing FEATURED_STORMS (it rewrites the file, so change the
    settings below, not the Excel):
        python code_weather/weather_input.py --build-table
    GUI / other code: list_historical_storms(location) -> historical_storm(storm_id)

OPTION 3 - THE USER TYPES THE NUMBERS
    GUI / other code: user_storm(ice_mm, wind_ms)

HOW A STORM BECOMES (ice_mm, wind_ms)   (options 1 and 2, find_weather_events.py)
    type "ice" or "ice + wind" : ice_mm = peak ice on the wire, wind_ms = gust while iced
    type "wind"                : ice_mm = 0, wind_ms = strongest gust
    storm_id = location + start date, e.g. "TOR-2013-12-20" ("-2" if a day has two)

Ranges the fragility curves cover (the lookups warn outside these; not an error here):
    ice storm : ice 0-150 mm, wind 0-40 m/s   (Ma et al. Fig. 7)
    wind storm: wind 20-90 m/s                (Ma et al. Fig. 4)
"""

import functools
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # locations.py, find_weather_events.py
from locations import LOCATIONS, get_location    # noqa: E402

# =============================================================================
# SETTINGS  <-- edit this block
# =============================================================================
DEFAULT_LOCATION = "toronto"         # Enter at the location question

# Option 1: the thresholds offered when the user just presses Enter
DEFAULT_ICE_THRESHOLD_MM = 6.35      # 1/4 inch of radial ice
DEFAULT_GUST_THRESHOLD_MS = 20.0     # 3-s gust; find_weather_events.py allows 12.5-40 m/s

# What the user's CSV must have (option 1). Shown to the user before choosing the file.
CSV_COLUMNS = ["time", "wind_speed_ms", "wind_gust_ms", "temp_c", "precip_mm", "ptype"]
CSV_REQUIREMENTS = """\
  The CSV must have ONE ROW PER HOUR of weather for {place}, with these columns:
    time           the hour, in UTC (e.g. 2013-12-21 05:00:00)
    wind_speed_ms  10 m wind speed (m/s)
    wind_gust_ms   10 m wind gust, the strongest in that hour (m/s)
    temp_c         2 m air temperature (deg C)
    precip_mm      precipitation in that hour (mm)
    ptype          precipitation type (ERA5 codes): 0 none, 1 rain, 3 freezing rain,
                   5 snow, 6 wet snow, 7 rain-snow mix, 8 ice pellets, 12 freezing drizzle
  No latitude/longitude is needed: the file is taken to be weather for {place}.
  (era5_download.py makes files like this. Its files hold a grid of points; then only
  {place}'s grid cells are used.)"""

# Option 2: the lookup table and what build_historical_table() puts in it
HISTORICAL_TABLE = HERE / "historical_storms.xlsx"
HISTORICAL_CSVS = {                  # ERA5 files (in this folder) to look for storms in
    "toronto": ["era5_toronto_2013_11_2013_12.csv", "era5_toronto_last_5_years.csv"],
}
FEATURED_STORMS = {                  # the 5 storms the user chooses from, and why each
    "toronto": {
        "TOR-2013-12-20": "the most ice in the data",
        "TOR-2025-12-28": "ice and strong wind together",
        "TOR-2025-12-29": "strong wind (the team's Event #2)",
        "TOR-2022-05-21": "very strong wind: 120 km/h measured at Pearson (ERA5 misses it)",
        "NS-2022-09-24": "stress test: hurricane-strength gusts (not a Toronto storm)",
    },
}
STORM_NAMES = {                      # a better name for some ERA5 storms, by storm_id
    "TOR-2013-12-20": "Dec 2013 Toronto ice storm",
    "TOR-2025-12-28": "Dec 28 2025 ice storm",
    "TOR-2025-12-29": "Dec 29 2025 wind storm (Event #2)",
}
MEASURED_STORMS = [                  # storms ERA5 misses (ERA5 underestimates sudden gusts)
    # location, storm_id, name, start, end, type, ice_mm, wind_ms, source
    ("toronto", "TOR-2022-05-21", "May 2022 derecho (gust measured at Toronto Pearson)",
     "2022-05-21 11:00", "2022-05-21 14:00", "wind", 0.0, 33.3,
     "Measured: ECCC Top 10 weather stories 2022, peak gust 120 km/h at Toronto Pearson "
     "(ERA5 shows only 14.7 m/s). Times approximate."),
    ("toronto", "NS-2022-09-24", "Hurricane Fiona 2022 (Nova Scotia; stress test only)",
     "2022-09-24 00:00", "2022-09-24 12:00", "wind", 0.0, 49.7,
     "Measured: ECCC Top 10 weather stories 2022, gusts of 179 km/h on Nova Scotia's "
     "North Shore. Not a Toronto storm. Times approximate."),
]

# Columns every storm-list table has (options 1 and 2), in this order.
STORM_LIST_COLUMNS = ["storm_id", "name", "start", "end", "type", "ice_mm", "wind_ms"]
#   storm_id : unique ID, used to pick the storm (e.g. "TOR-2013-12-20")
#   name     : readable label (e.g. "Dec 2013 Toronto ice storm")
#   start/end: local date-time of the storm
#   type     : "ice", "wind" or "ice + wind"
#   ice_mm, wind_ms : the two numbers the storm gives the pipeline


# =============================================================================
# THE CONTRACT -- every option returns through this
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
# THE LOCATION (and its feeder, from the lookup table in locations.py)
# =============================================================================
def feeder_for(location):
    """The feeder run for a location (locations.py). Raises if it has none yet."""
    loc = get_location(location)
    if not loc.get("feeder"):
        raise ValueError(f"{loc['name']} has no feeder yet: set its \"feeder\" in "
                         "code_weather/locations.py.")
    return str(loc["feeder"])


def ask_location(input_fn=input):
    """Ask which location to run for. Tells the user its feeder.
    Returns (location, feeder), e.g. ("toronto", "3777")."""
    keys = list(LOCATIONS)
    print("\nWHICH LOCATION ARE WE RUNNING FOR?")
    for i, key in enumerate(keys, start=1):
        feeder = LOCATIONS[key].get("feeder")
        note = f"feeder {feeder}" if feeder else "no feeder chosen yet"
        mark = "  (default)" if key == DEFAULT_LOCATION else ""
        print(f"  {i}. {LOCATIONS[key]['name']:<12} -> {note}{mark}")
    while True:
        answer = input_fn(f"Choose 1-{len(keys)} or a name [{DEFAULT_LOCATION}]: ").strip().lower()
        key = (DEFAULT_LOCATION if answer == "" else
               keys[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(keys) else
               answer.replace(" ", "_"))
        if key not in LOCATIONS:
            print(f"  '{answer}' is not a location. Try again.")
            continue
        try:
            feeder = feeder_for(key)
        except ValueError as e:
            print(f"  {e} Choose another location.")
            continue
        print(f"Running for {LOCATIONS[key]['name']}: feeder {feeder}")
        return key, feeder


# =============================================================================
# OPTION 1 -- the user's own weather CSV
# =============================================================================
def read_weather_csv(csv_path, location=DEFAULT_LOCATION):
    """Read the user's CSV and check it has what option 1 needs (CSV_REQUIREMENTS).
    Returns the table, ready for find_weather_events.py. Raises ValueError that lists
    every problem found."""
    path = _csv_file(csv_path)
    weather, _ = _prepare(str(path), path.stat().st_mtime, location)
    return weather.copy()


def list_storms_from_csv(csv_path, location=DEFAULT_LOCATION,
                         ice_threshold_mm=DEFAULT_ICE_THRESHOLD_MM,
                         gust_threshold_ms=DEFAULT_GUST_THRESHOLD_MS):
    """Every storm in the user's CSV with ice >= ice_threshold_mm OR a gust >=
    gust_threshold_ms (find_weather_events.py), for the list the user picks from.
    Returns a DataFrame with the columns in STORM_LIST_COLUMNS, one row per storm."""
    path = _csv_file(csv_path)
    return _storms_in_csv(str(path), path.stat().st_mtime, location,
                          float(ice_threshold_mm), float(gust_threshold_ms)).copy()


def storm_from_csv(csv_path, storm_id, location=DEFAULT_LOCATION,
                   ice_threshold_mm=DEFAULT_ICE_THRESHOLD_MM,
                   gust_threshold_ms=DEFAULT_GUST_THRESHOLD_MS):
    """One storm from the user's CSV, picked by its storm_id (same location and thresholds
    as for list_storms_from_csv). Returns (ice_mm, wind_ms)."""
    storms = list_storms_from_csv(csv_path, location, ice_threshold_mm, gust_threshold_ms)
    return _pick(storms, storm_id, Path(csv_path).name)


def ask_csv_storm(location=DEFAULT_LOCATION, input_fn=input, csv_path=None):
    """OPTION 1 in the terminal: show what the CSV needs, let the user choose it (a file
    window opens), ask the two thresholds, list the storms and let the user pick one.
    csv_path: use this file instead of asking. Returns (ice_mm, wind_ms)."""
    place = get_location(location)["name"]
    print(f"\nOPTION 1 - YOUR OWN WEATHER CSV\n{CSV_REQUIREMENTS.format(place=place)}")
    while True:
        path = csv_path or _choose_csv_file(input_fn)
        try:
            read_weather_csv(path, location)
            break
        except (ValueError, FileNotFoundError) as e:
            print(f"\n{e}\nChoose another file.")
            csv_path = None
    print("\nA storm is listed if it passes EITHER threshold.")
    while True:
        ice_thr = _ask_number(f"Ice threshold, mm of radial ice on the wire "
                              f"[{DEFAULT_ICE_THRESHOLD_MM:g}]: ", DEFAULT_ICE_THRESHOLD_MM,
                              input_fn, low=0, low_ok=False)
        gust_thr = _ask_number(f"Gust threshold, m/s, 12.5 to 40 "
                               f"[{DEFAULT_GUST_THRESHOLD_MS:g}]: ", DEFAULT_GUST_THRESHOLD_MS,
                               input_fn, low=12.5, high=40)
        print("  Finding the storms (find_weather_events.py)...")
        storms = list_storms_from_csv(path, location, ice_thr, gust_thr)
        if len(storms):
            break
        print("  No storm in this file passes either threshold. Try lower ones.")
    storm_id = _choose_storm(storms, f"Storms in {Path(path).name} "
                                     f"(ice >= {ice_thr:g} mm or gust >= {gust_thr:g} m/s)",
                             input_fn)
    return storm_from_csv(path, storm_id, location, ice_thr, gust_thr)


# =============================================================================
# OPTION 2 -- a historical storm from the lookup table (historical_storms.xlsx)
# =============================================================================
def build_historical_table(path=None):
    """Make the lookup table historical_storms.xlsx (default path HISTORICAL_TABLE):
        sheet "storms"      the 5 storms per location the user chooses from (FEATURED_STORMS)
        sheet "all storms"  every storm found in HISTORICAL_CSVS + MEASURED_STORMS
        sheet "about"       how the table was made
    A measured storm replaces an ERA5 storm with the same storm_id. Returns "storms"."""
    path = Path(path) if path else HISTORICAL_TABLE
    parts = []
    for location, files in HISTORICAL_CSVS.items():
        for name in files:
            found = list_storms_from_csv(HERE / name, location)
            found.insert(0, "location", location)
            found["source"] = (f"ERA5 {name}, find_weather_events.py (ice >= "
                               f"{DEFAULT_ICE_THRESHOLD_MM:g} mm or gust >= "
                               f"{DEFAULT_GUST_THRESHOLD_MS:g} m/s)")
            parts.append(found)
    parts.append(pd.DataFrame(MEASURED_STORMS, columns=["location"] + STORM_LIST_COLUMNS + ["source"]))
    every = (pd.concat(parts, ignore_index=True)
               .drop_duplicates("storm_id", keep="last")
               .sort_values(["location", "start", "storm_id"]).reset_index(drop=True))
    every["name"] = [STORM_NAMES.get(s, n) for s, n in zip(every["storm_id"], every["name"])]

    featured = []
    for location, chosen in FEATURED_STORMS.items():
        here = every[every["location"] == location].set_index("storm_id")
        missing = [s for s in chosen if s not in here.index]
        if missing:
            raise ValueError(f"FEATURED_STORMS[{location!r}] has storm_ids that are not in "
                             f"the data: {missing}")
        for option, (storm_id, why) in enumerate(chosen.items(), start=1):
            featured.append({"location": location, "option": option, "storm_id": storm_id,
                             **here.loc[storm_id, STORM_LIST_COLUMNS[1:]].to_dict(),
                             "why": why, "source": here.loc[storm_id, "source"]})
    storms = pd.DataFrame(featured)

    about = pd.DataFrame({"": [
        "Lookup table for OPTION 2 in code_weather/weather_input.py.",
        f"Made by build_historical_table() on {datetime.now():%Y-%m-%d %H:%M}. "
        "Do not edit by hand: remaking it rewrites this file.",
        "Sheet 'storms': the 5 storms per location the user chooses from (FEATURED_STORMS).",
        f"Sheet 'all storms': every storm find_weather_events.py finds in the ERA5 files "
        f"(HISTORICAL_CSVS: ice >= {DEFAULT_ICE_THRESHOLD_MM:g} mm or a gust >= "
        f"{DEFAULT_GUST_THRESHOLD_MS:g} m/s), plus MEASURED_STORMS (see the source column).",
        "ice_mm = peak radial ice on the wire; wind_ms = gust while iced (ice storms) or the "
        "strongest gust (wind storms).",
    ]})
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        storms.to_excel(xl, sheet_name="storms", index=False)
        every.to_excel(xl, sheet_name="all storms", index=False)
        about.to_excel(xl, sheet_name="about", index=False, header=False)
        for sheet, widths in (("storms", [10, 7, 16, 36, 17, 17, 11, 8, 8, 58, 60]),
                              ("all storms", [10, 16, 52, 17, 17, 11, 8, 8, 90])):
            for col, width in zip("ABCDEFGHIJK", widths):
                xl.sheets[sheet].column_dimensions[col].width = width
        xl.sheets["about"].column_dimensions["A"].width = 120
    return storms


def list_historical_storms(location=DEFAULT_LOCATION):
    """The 5 storms the user chooses from for this location (sheet "storms" of the lookup
    table). Makes the table first if it isn't there yet.
    Returns a DataFrame with the columns in STORM_LIST_COLUMNS (plus option, why, source)."""
    table = _read_table("storms")
    return table[table["location"] == location].reset_index(drop=True)


def historical_storm(storm_id):
    """Any storm in the lookup table (sheet "all storms"), picked by its storm_id.
    Returns (ice_mm, wind_ms)."""
    return _pick(_read_table("all storms"), storm_id, HISTORICAL_TABLE.name)


def ask_historical_storm(location=DEFAULT_LOCATION, input_fn=input):
    """OPTION 2 in the terminal: show the 5 historical storms, the user picks one.
    Returns (ice_mm, wind_ms)."""
    storms = list_historical_storms(location)
    place = get_location(location)["name"]
    print(f"\nOPTION 2 - A HISTORICAL STORM ({place})")
    for s in storms.itertuples(index=False):
        print(f"  {s.option}. {s.name}\n       ice {s.ice_mm:g} mm, gust {s.wind_ms:g} m/s"
              f" - {s.why}")
    while True:
        answer = input_fn(f"Choose 1-{len(storms)}: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(storms):
            return historical_storm(storms["storm_id"].iloc[int(answer) - 1])
        print(f"  '{answer}' is not one of the options. Try again.")


# =============================================================================
# OPTION 3 -- the user types the numbers (terminal_prompts.py asks them)
# =============================================================================
def user_storm(ice_mm, wind_ms):
    """Storm typed in by the user (terminal, GUI boxes, or the settings in run_fragility.py).
    Returns (ice_mm, wind_ms)."""
    return check_weather(ice_mm, wind_ms)


# =============================================================================
# HELPERS
# =============================================================================
def _csv_file(csv_path):
    """The CSV's full path; a bare file name is also looked for in code_weather."""
    path = Path(str(csv_path).strip().strip('"'))
    if not path.exists() and (HERE / path).exists():
        path = HERE / path
    if not path.is_file():
        raise FileNotFoundError(f"Can't find the weather CSV: {csv_path}")
    return path.resolve()


@functools.lru_cache(maxsize=4)
def _prepare(path, modified_time, location):
    """Read and check one CSV for one location (cached: a 5-year file takes seconds).
    Returns (weather, where) for find_weather_events.find_events: `where` is the location
    key (a file with a grid of points) or a one-point location (a file without one)."""
    loc = get_location(location)
    weather = pd.read_csv(path)
    problems = [f"missing columns: {', '.join(c for c in CSV_COLUMNS if c not in weather.columns)}"
                ] if any(c not in weather.columns for c in CSV_COLUMNS) else []
    if not problems and weather.empty:
        problems.append("the file has no rows")
    if not problems:
        weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
        if weather["time"].isna().any():
            problems.append("some 'time' values are not dates (expected e.g. 2013-12-21 05:00:00)")
        else:
            steps = weather["time"].drop_duplicates().sort_values().diff().dropna()
            if len(steps) and steps.mode().iloc[0] != pd.Timedelta(hours=1):
                problems.append("it is not hourly (one row per hour)")
    if not problems and {"latitude", "longitude"} <= set(weather.columns):
        points = set(zip(weather["latitude"].round(2), weather["longitude"].round(2)))
        cells = [(round(a, 2), round(b, 2)) for a, b in loc["cells"]]
        if len(points) > 1:                          # a grid of points (era5_download.py)
            if not all(c in points for c in cells):
                problems.append(f"it holds several grid points, but not {loc['name']}'s "
                                f"grid cells {cells}")
            else:
                return weather, location
    if problems:
        raise ValueError(f"{Path(path).name} can't be used for option 1:\n  - "
                         + "\n  - ".join(problems)
                         + f"\nWhat the file needs:\n{CSV_REQUIREMENTS.format(place=loc['name'])}")
    # one point (or no coordinates): the whole file is weather for the chosen location
    lat, lon = loc["cells"][0]
    weather["latitude"], weather["longitude"] = lat, lon
    return weather, {**loc, "cells": [(lat, lon)]}


@functools.lru_cache(maxsize=16)
def _storms_in_csv(path, modified_time, location, ice_threshold_mm, gust_threshold_ms):
    """The storm list for one CSV, location and pair of thresholds (cached: picking a storm
    right after listing them does not run find_weather_events.py again)."""
    import find_weather_events as fwe

    weather, where = _prepare(path, modified_time, location)
    events = fwe.find_events(weather.copy(), where, ice_threshold_mm / 10.0,
                             gust_threshold_ms)["events"]
    prefix = get_location(location)["key"].upper()[:3]
    label = {"ice": "Ice storm", "wind": "Wind storm", "ice + wind": "Ice + wind storm"}
    rows, used = [], {}
    for e in events.itertuples(index=False):
        start, end = pd.Timestamp(e.start_local), pd.Timestamp(e.end_local)
        storm_id = f"{prefix}-{start:%Y-%m-%d}"
        used[storm_id] = used.get(storm_id, 0) + 1
        if used[storm_id] > 1:                       # a second storm starting the same day
            storm_id += f"-{used[storm_id]}"
        if e.type == "wind":                         # wind-only storm: the strongest gust
            ice_mm, wind_ms = 0.0, round(e.max_gust_ms, 2)
        else:                                        # ice storm: peak ice, gust WHILE iced
            ice_mm, wind_ms = round(10 * e.peak_ice_cm, 2), round(e.gust_while_iced_ms, 2)
        rows.append({"storm_id": storm_id, "name": f"{label[e.type]}, {start:%d %b %Y}",
                     "start": f"{start:%Y-%m-%d %H:%M}", "end": f"{end:%Y-%m-%d %H:%M}",
                     "type": e.type, "ice_mm": ice_mm, "wind_ms": wind_ms})
    return pd.DataFrame(rows, columns=STORM_LIST_COLUMNS)


def _read_table(sheet):
    """One sheet of the lookup table; makes the table first if it isn't there yet."""
    if not HISTORICAL_TABLE.exists():
        print(f"  {HISTORICAL_TABLE.name} not found: making it (build_historical_table())...")
        build_historical_table()
    table = pd.read_excel(HISTORICAL_TABLE, sheet_name=sheet, dtype={"storm_id": str})
    missing = [c for c in STORM_LIST_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{HISTORICAL_TABLE.name} is missing the columns {missing}. "
                         "Remake it: python code_weather/weather_input.py --build-table")
    return table


def _pick(table, storm_id, where):
    """The (ice_mm, wind_ms) of one storm_id in a storm-list table."""
    row = table[table["storm_id"] == storm_id]
    if row.empty:
        raise ValueError(f"No storm {storm_id!r} in {where}. "
                         f"Choose from: {', '.join(table['storm_id'])}")
    return check_weather(row["ice_mm"].iloc[0], row["wind_ms"].iloc[0])


def _ask_number(question, default, input_fn=input, low=0.0, high=math.inf, low_ok=True):
    """Ask until the user types a number in range. Enter = default."""
    while True:
        answer = input_fn(question).strip()
        if answer == "":
            return float(default)
        try:
            value = float(answer)
        except ValueError:
            value = None
        if value is not None and math.isfinite(value) and value <= high and (
                value > low or (low_ok and value == low)):
            return value
        limits = f"{'>=' if low_ok else '>'} {low:g}" + (f" and <= {high:g}" if high < math.inf else "")
        print(f"  '{answer}' is not a number {limits}. Try again.")


def _choose_storm(table, title, input_fn=input):
    """Show a storm list as a numbered list. Returns the storm_id the user picks
    (typed as its number or as the storm_id)."""
    print(f"\n{title}:")
    for i, s in enumerate(table.itertuples(index=False), start=1):
        print(f"  {i:>3}. {s.storm_id:<16} {s.name:<30} ice {s.ice_mm:>6g} mm, "
              f"gust {s.wind_ms:>5g} m/s")
    ids = list(table["storm_id"])
    while True:
        answer = input_fn(f"Pick a storm (1-{len(ids)} or its storm_id): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(ids):
            return ids[int(answer) - 1]
        if answer in ids:
            return answer
        print(f"  '{answer}' is not on the list. Try again.")


def _choose_csv_file(input_fn=input):
    """A file window to pick the weather CSV. If no window can open (no tkinter or no
    screen) or nothing is picked, the user types the path instead."""
    if input_fn is input:                            # only with a real user at the keyboard
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()                          # no empty main window
            root.attributes("-topmost", True)        # open in front of VS Code
            path = filedialog.askopenfilename(
                title="Choose your hourly weather CSV", initialdir=HERE,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            root.destroy()
            if path:
                return path
        except Exception:                            # noqa: BLE001  no tkinter / no screen
            pass
    while True:
        answer = input_fn("Path of your weather CSV: ")
        try:
            return str(_csv_file(answer))
        except FileNotFoundError as e:
            print(f"  {e}")


# =============================================================================
# RUN THIS FILE
#   python code_weather/weather_input.py                 self-check of the options
#   python code_weather/weather_input.py --build-table   (re)make historical_storms.xlsx
# =============================================================================
if __name__ == "__main__":
    if "--build-table" in sys.argv:
        t = build_historical_table()
        every = pd.read_excel(HISTORICAL_TABLE, sheet_name="all storms")
        print(f"Saved {HISTORICAL_TABLE.name}: {len(t)} storms to choose from, "
              f"{len(every)} in 'all storms'")
        sys.exit()

    def report(label, fn, *args):
        try:
            out = fn(*args)
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

    print("Location:")
    print("   ", report(f"feeder_for({DEFAULT_LOCATION!r})", feeder_for, DEFAULT_LOCATION))

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
    csv = HERE / "era5_toronto_last_5_years.csv"
    table = report("list_storms_from_csv(...)", list_storms_from_csv, csv)
    check_list("list_storms_from_csv(...)", table)
    if table is not None and len(table):
        sid = table["storm_id"].iloc[0]
        print("   ", report(f"storm_from_csv(..., {sid!r})", storm_from_csv, csv, sid))
