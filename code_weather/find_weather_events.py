"""
FIND WEATHER EVENTS THAT PASS AN ICE THRESHOLD AND/OR A GUST THRESHOLD
=====================================================================
Reads a weather CSV you already have (made by era5_download.py) and
writes one Excel file. It never downloads anything.

Files needed in the same folder:
    find_weather_events.py   (this file)
    locations.py             (where each location's grid cells are)
    pip install pandas numpy openpyxl

RUN
---
    python find_weather_events.py --csv "C:\\path\\to\\era5_toronto.csv" --start 2021-01 --end 2025-12 --ice 0.635 --gust 25

    --csv             path to the weather CSV (required; quotes if spaces)
    --start / --end   months to analyse (YYYY-MM, earlier first). Only these
                      months of the CSV are used.
    --location        which cells to use from locations.py (default toronto)
    --ice             ice threshold, cm of radial ice on a wire
    --gust            gust threshold, m/s (12.5 - 40, the fragility-curve range)
    --require         "either" (default): ice >= --ice OR gust >= --gust
                      "both": ice >= --ice AND a gust >= --gust hit the wire
                              WHILE that ice was on it
    --ice-wind-days N keep counting gusts as "on the iced wire" for N days
                      after the freezing rain stops (stops early if the ice
                      melts). Default 3 (Sheng et al. 2023). 0 = only while
                      the freezing rain lasts.
    --no-rain-below-freezing  count ONLY what ERA5 calls freezing rain
                      (Sheng). Default: ALSO count ERA5 rain / drizzle /
                      rain-snow mix at <= 0 C (Jones). Snow never counts.
    --out FILE.xlsx         output name (default: next to the CSV)
  Advanced (each one is a design decision, see the Settings sheet):
    --ice-gap-h 72  --melt-temp 1.0  --melt-hours 1  --wind-gap-h 6
    --min-precip 0.1  --ptype-hour start_or_end|end
    --wind-for-ice hour_average|end

WHICH STORMS ARE KEPT (default --require either)
    storm --> peak ice >= ICE_THR ?  ---yes---+
          --> strongest gust >= GUST_THR ? -yes-+--> EVENT (one is enough)
          --> both "no"  ----------------------> not an event (left out)
    Type = ice (ice only) / wind (wind only) / ice + wind (both)

OUTPUT: one Excel file, ONLY storms that pass:
    Events          one row per event, numbered TOR-001, TOR-002, ...;
                    columns "Passes ice threshold" and "Passes wind
                    threshold" say yes/no
    Event cells     the same events, one row per grid cell that passed
    Settings        every input and design decision, with its reason
    Column guide    the keep-rule above + what every column means
    (--show-below adds a "Below thresholds" sheet of storms that passed
     neither threshold; they are NOT events)

LOGIC (pseudocode)
------------------
Each ERA5 grid cell is treated like one weather station in Sheng et al.
(2023): ice is built up and melted cell by cell. A location (e.g.
Toronto) is a list of cells; storms that hit several cells at the same
time are combined at the end.

FOR EACH location, FOR EACH grid cell of that location:

  1. HOURLY FLAGS (ERA5 values at time t describe the hour ending at t)
       freezing_rain = precip in the hour >= 0.1 mm
                       AND ERA5 precipitation type is freezing rain (3) or
                           freezing drizzle (12) at the START or END of
                           the hour
                       AND temperature <= 1 C (no NEW ice in a warm hour)
                       OR (default, Jones) type is rain / rain-snow mix
                          and temperature <= 0 C. Snow never counts.
       ice_added     = Jones (1998) / Sheng Eq. 4, ONE hour, if freezing_rain:
                       sqrt((P*rho_w)^2 + (3.6*V*W)^2) / (rho_i*pi)
                       P = precip mm, V = wind speed averaged over the hour,
                       W = 0.067*P^0.846, rho_w = 1, rho_i = 0.9
       melt          = temperature > 1 C in this cell for M hours in a row
                       (M = --melt-hours; default 1 = the papers' rule)
       windy         = gust (max 3-s gust in the hour) >= GUST_THR

  2. ICE STORMS (Sheng et al. 2023, p. 130). Walk through the
     freezing-rain hours in time order. Start a NEW ice storm when
       (a) more than 72 h have passed since the last freezing-rain hour, or
       (b) a melt hour happened since the last freezing-rain hour.
     Ice of the storm = SUM of ice_added over its hours.

  3. ICE ON THE WIRE (hour by hour). From the first freezing-rain hour of
     an ice storm, the ice on the wire grows by ice_added each hour. It is
     counted as "on the wire" until N days after the last freezing-rain
     hour (N = --ice-wind-days, default 3), or until the hour before the
     first melt hour, whichever comes first. Outside that, ice = 0.

  4. WIND STORMS. Group the windy hours. Start a NEW wind storm when more
     than 6 h pass without a windy hour.

  5. CELL EVENTS. Ice periods (step 3) and wind storms (step 4) that
     overlap in time are the same event. For each event:
       peak ice             = max ice on the wire
       gust while iced      = max gust in hours with ice on the wire
       strongest gust       = max gust
       ice at strongest gust= ice on the wire in that hour

  6. PASS / FAIL for that cell event
       passes_ice  = peak ice >= ICE_THR
       passes_gust = strongest gust >= GUST_THR
       passes_both = peak ice >= ICE_THR AND gust while iced >= GUST_THR

     type = "ice + wind" if passes_ice AND passes_gust, else "ice" or
            "wind"; gust_on_ice = yes if passes_both

COMBINE CELLS: cell events from any cell of the location that overlap in
time, or are less than 6 h (--wind-gap-h) apart, are one storm. The storm passes if ANY of its cells passes
("either": passes_ice OR passes_gust; "both": passes_both). The row
shows the passing cell furthest above its threshold, so ice and gust
always come from ONE cell. Every cell is listed in Event cells.

FRAGILITY INPUT: two snapshots per storm
    A "most ice":        (peak ice, gust while iced)              -> ice
    B "strongest gust":  (ice at strongest gust, strongest gust)  -> wind
For an ice + wind storm where the gust hit the iced wire (gust_on_ice =
yes), A covers both. Otherwise check both A and B.
Pairing peak ice with the strongest gust while iced follows Jones et al.
(2002, p. 14), who assume the worst wind-on-ice load comes with the most
ice.

TIME NOTE: an event whose first flagged hour is 14:00 started at 13:00
(ERA5 values at 14:00 cover 13:00-14:00).

SOURCES
-------
ECMWF (2026). ERA5: data documentation. Copernicus Knowledge Base.
Jones, K. F. (1998). A simple model for freezing rain ice loads.
    Atmospheric Research, 46, 87-97. (Eq. 4)
Jones, K. F., Thorkildson, R., & Lott, N. (2002). The development of a
    U.S. climatology of extreme ice loads. NOAA/NCDC Tech. Rep. 2002-01.
Sheng, C., Tang, Q., & Hong, H. P. (2023). Estimating and mapping extreme
    ice accretion hazard and load due to freezing rain at Canadian sites.
    Int. J. Disaster Risk Science, 14, 127-142.
"""

import argparse
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from locations import get_location

RHO_WATER, RHO_ICE = 1.0, 0.9      # g/cm^3 (Jones 1998)
HOUR = pd.Timedelta(hours=1)
# If --csv is just a file name and isn't in the current folder, the script
# also looks in its own folder (e.g. code_weather).
HERE = Path(__file__).resolve().parent


# =====================================================================
# DESIGN DECISIONS (each one is listed, with its reason, in the Excel
# "Settings" sheet; change a default here or with the matching flag)
# =====================================================================
@dataclass(frozen=True)
class Settings:
    ice_ptypes: tuple = (3, 12)
    rain_below_freezing: bool = True
    rain_ptypes: tuple = (1, 7)
    freezing_temp_c: float = 0.0
    min_precip_mm: float = 0.1
    ptype_hour: str = "start_or_end"
    wind_for_ice: str = "hour_average"
    melt_temp_c: float = 1.0
    melt_hours: int = 1
    ice_gap_h: float = 72.0
    ice_wind_days: float = 3.0
    wind_gap_h: float = 6.0
    gust_range_ms: tuple = (12.5, 40.0)


# name -> (what it does, why this value / source)
SETTING_NOTES = {
    "ice_ptypes": (
        "ERA5 precipitation types that count as freezing rain "
        "(3 = freezing rain, 12 = freezing drizzle; ECMWF code table 4.201).",
        "Same as Sheng et al. (2023, p. 130) and Jarrett et al. (2019): "
        "freezing rain and freezing drizzle only. Ice pellets (8) are left "
        "out; Jones et al. (2002, p. 7) report results with and without them."),
    "rain_below_freezing": (
        "Also count ERA5 'rain' or 'rain-snow mix' as freezing rain when "
        "the 2 m temperature is at or below freezing_temp_c. Snow is never "
        "counted.",
        "On (default) = Jones et al. (2002, p. 7), whose ice equation we "
        "use: they also accrete rain/drizzle at below-freezing "
        "temperatures and call this 'intentionally conservative'. Snow is "
        "left out because the equation is for liquid drops that freeze on "
        "impact. Off = Sheng et al. (2023). Caveat: ERA5's 2 m temperature "
        "can run 1-2 C too cold near 0 C (e.g. 28 Dec 2025 vs Pearson), "
        "which can turn plain rain into ice."),
    "rain_ptypes": (
        "Types used by rain_below_freezing (1 = rain, 7 = rain-snow mix).",
        "ECMWF code table 4.201."),
    "freezing_temp_c": (
        "Temperature limit for rain_below_freezing (deg C).",
        "0 deg C = freezing point."),
    "min_precip_mm": (
        "An hour needs at least this much precipitation (mm) to count as "
        "freezing rain.",
        "Our choice. Stops trace amounts from keeping a storm 'alive' "
        "under the 72 h rule. Small effect on ice: 0.1 mm in an hour adds "
        "only ~0.04-0.07 mm of radial ice."),
    "ptype_hour": (
        "'start_or_end': the hour counts as freezing rain if ERA5's "
        "precipitation type is freezing rain at the start OR end of the "
        "hour. 'end': only at the end.",
        "ERA5 precipitation is a total over the hour, but precipitation "
        "type is a snapshot at the timestamp (ERA5 data documentation, "
        "Tables 2 and 3). 'start_or_end' catches the last hour of a storm, "
        "when the rain stopped before the snapshot. Jones et al. (2002, "
        "p. 7) likewise treat a whole hour as freezing rain if freezing "
        "rain fell at any time in it."),
    "wind_for_ice": (
        "Wind speed V used in the ice equation. 'hour_average': mean of "
        "the snapshots at the start and end of the hour. 'end': the "
        "snapshot at the end.",
        "Sheng et al. (2023, p. 129) define V as the hourly-mean wind "
        "speed. ERA5 10 m wind is a snapshot each hour, so the average of "
        "the two snapshots around the hour is the closest estimate. Jones "
        "et al. (2002, p. 14) themselves used the 1- or 2-minute average "
        "wind reported once each hour."),
    "melt_temp_c": (
        "Temperature above which (a) no new ice forms in that hour and "
        "(b) existing ice is assumed melted (see melt_hours).",
        "Sheng et al. (2023, p. 130), Jones et al. (2002, p. 9) and "
        "CSA C22.3 (as cited by Sheng et al., p. 129) all use 1 deg C. "
        "Checked separately in every grid cell."),
    "melt_hours": (
        "Ice is assumed melted only after this many hours IN A ROW above "
        "melt_temp_c. Then the ice storm ends and ice on the wire drops "
        "to zero.",
        "1 = the published rule (first hour above 1 deg C: Sheng et al. "
        "2023 p. 130; Jones et al. 2002 p. 9). Jones et al. (p. 9) call "
        "this a simplification: real melting depends on sun and ice "
        "shedding and is 'complex and locally variable'. Values above 1 "
        "are our own sensitivity option; no paper gives a number."),
    "ice_gap_h": (
        "If this many hours pass with no freezing rain (and no melt), the "
        "next freezing rain starts a NEW ice storm.",
        "Sheng et al. (2023, p. 130): 72 consecutive hours (no source "
        "given for the number). CSA C22.3 uses 24 h (Sheng p. 129); "
        "McCray et al. (2019) also merge freezing-rain events less than "
        "24 h apart. Jones et al. (2002, p. 9) use no gap rule; storms "
        "end only when it warms above 1 deg C."),
    "ice_wind_days": (
        "Days after the last freezing-rain hour that ice is still counted "
        "as on the wire, so gusts in that time count as 'while iced'. "
        "Stops early at melt.",
        "3 = the shortest window in Sheng et al. (2023, p. 135), who use "
        "3, 7 and 21 days and find the result 'not very sensitive' to it; "
        "it also matches the 72 h storm rule. 0 = only during the storm. "
        "Jones et al. (2002, p. 13): ice 'may last for days or even weeks "
        "after the freezing rain ends'; they count wind until it melts. "
        "Jones & Mulherin (1998, p. 34) found ice on trees three weeks "
        "after the 1998 storm. Stopping at melt is our addition. If N > 3 "
        "days, a new ice storm can start while old ice is still counted; "
        "the event then reports the larger ice amount, not the sum."),
    "wind_gap_h": (
        "Windy hours less than this many hours apart belong to the same "
        "wind storm. Also: events in neighbouring grid cells less than "
        "this far apart are combined into one storm.",
        "Our choice; no paper. Short enough to keep storms on different "
        "days separate, long enough that a lull of a few hours doesn't "
        "split one storm. Extreme-wind statistics often use much longer "
        "separations (e.g. 48 h in the 'independent storms' method)."),
    "gust_range_ms": (
        "Allowed range for the --gust threshold (m/s).",
        "Gust range of the Hou et al. fragility curves (from Maria)."),
}


# =====================================================================
# STEP 1: hourly flags for ONE grid cell
# =====================================================================
def ice_added_mm(precip_mm, wind_ms):
    """Radial ice added in ONE hour, mm (Jones 1998 Eq. 4 = Sheng Eq. 4)."""
    P = np.asarray(precip_mm, dtype=float)
    V = np.asarray(wind_ms, dtype=float)
    W = 0.067 * P ** 0.846                      # liquid water content, g/m^3
    return np.sqrt((P * RHO_WATER) ** 2 + (3.6 * V * W) ** 2) / (RHO_ICE * np.pi)


def hourly_flags(cell, gust_thr, s):
    """Add freezing_rain, ice_added_mm, melt and windy columns."""
    c = cell.sort_values("time").reset_index(drop=True).copy()
    # previous row is only used if it really is the previous hour
    prev_ok = c["time"].diff() == HOUR
    ptype_prev = c["ptype"].shift(1).where(prev_ok)

    is_ice_type = c["ptype"].isin(s.ice_ptypes)
    if s.ptype_hour == "start_or_end":
        is_ice_type |= ptype_prev.isin(s.ice_ptypes)
    if s.rain_below_freezing:
        is_ice_type |= (c["ptype"].isin(s.rain_ptypes)
                        & (c["temp_c"] <= s.freezing_temp_c))

    # warm hour: no new ice can form. melt: warm for melt_hours in a row.
    warm = c["temp_c"] > s.melt_temp_c
    hours_warm_in_a_row = warm.groupby((~warm).cumsum()).cumsum()
    c["melt"] = hours_warm_in_a_row >= s.melt_hours
    c["freezing_rain"] = ((c["precip_mm"] >= s.min_precip_mm)
                          & is_ice_type & ~warm)

    wind = c["wind_speed_ms"]
    if s.wind_for_ice == "hour_average":
        wind = ((wind + wind.shift(1).where(prev_ok)) / 2).fillna(wind)
    c["ice_added_mm"] = np.where(c["freezing_rain"],
                                 ice_added_mm(c["precip_mm"], wind), 0.0)
    c["windy"] = c["wind_gust_ms"] >= gust_thr
    return c


# =====================================================================
# STEPS 2-4: ice storms, ice on the wire, wind storms (ONE cell)
# =====================================================================
def ice_storms(c, s):
    """Step 2-3. Returns a list of ice storms (row numbers) and the hourly
    ice-on-the-wire series in mm."""
    fr_rows = np.flatnonzero(c["freezing_rain"].to_numpy())
    times = c["time"].to_numpy()
    melt = c["melt"].to_numpy()
    melt_rows = np.flatnonzero(melt)
    melts_so_far = np.cumsum(melt)          # melt hours up to each row
    added = c["ice_added_mm"].to_numpy()
    on_wire = np.zeros(len(c))
    storms = []
    if not len(fr_rows):
        return storms, on_wire

    # (2) group freezing-rain hours into ice storms
    groups = [[fr_rows[0]]]
    for prev, row in zip(fr_rows[:-1], fr_rows[1:]):
        gap_h = (times[row] - times[prev]) / np.timedelta64(1, "h")
        melted_between = melts_so_far[row] > melts_so_far[prev]
        if gap_h > s.ice_gap_h or melted_between:
            groups.append([row])
        else:
            groups[-1].append(row)

    # (3) how long the ice is counted on the wire
    for g in groups:
        first, last = g[0], g[-1]
        until = times[last] + np.timedelta64(int(round(s.ice_wind_days * 24)), "h")
        k = np.searchsorted(melt_rows, last, side="right")
        if k < len(melt_rows):                       # first melt after storm
            until = min(until, times[melt_rows[k]] - np.timedelta64(1, "h"))
        end = int(np.searchsorted(times, until, side="right")) - 1
        end = max(end, last)
        own = added[first:end + 1].copy()
        own[last - first + 1:] = 0.0                 # only THIS storm's ice
        on_wire[first:end + 1] = np.maximum(on_wire[first:end + 1],
                                            np.cumsum(own))
        storms.append({"first": first, "last": last, "end": end,
                       "ice_mm": float(added[first:last + 1].sum()),
                       "fr_hours": len(g)})
    return storms, on_wire


def wind_storms(c, s):
    """Step 4. (first_row, last_row) of each wind storm."""
    rows = np.flatnonzero(c["windy"].to_numpy())
    if not len(rows):
        return []
    times = c["time"].to_numpy()
    out = [[rows[0], rows[0]]]
    for row in rows[1:]:
        gap_h = (times[row] - times[out[-1][1]]) / np.timedelta64(1, "h")
        if gap_h > s.wind_gap_h:
            out.append([row, row])
        else:
            out[-1][1] = row
    return [tuple(w) for w in out]


def merge_intervals(intervals):
    """Join (start, end) intervals that overlap."""
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(m) for m in merged]


# =====================================================================
# STEPS 5-6: events in ONE cell
# =====================================================================
def _event_row(c, on_wire, a, b):
    """Numbers for one event window (rows a..b) in one cell."""
    seg = c.iloc[a:b + 1]
    ice = on_wire[a:b + 1]
    gust = seg["wind_gust_ms"].to_numpy()
    iced = ice > 0
    k = int(np.nanargmax(gust)) if np.isfinite(gust).any() else 0
    gust_iced = (float(np.nanmax(gust[iced]))
                 if iced.any() and np.isfinite(gust[iced]).any() else np.nan)
    return {
        "first_hour": seg["time"].iloc[0],
        "last_hour": seg["time"].iloc[-1],
        "peak_ice_cm": ice.max() / 10.0,
        "gust_while_iced_ms": gust_iced,
        "max_gust_ms": float(gust[k]),
        "ice_at_max_gust_cm": ice[k] / 10.0,
        "max_gust_time": seg["time"].iloc[k],
        "ice_on_wire_hours": int(iced.sum()),
        "freezing_rain_hours": int(seg["freezing_rain"].sum()),
        "windy_hours": int(seg["windy"].sum()),
        "min_temp_c": float(seg["temp_c"].min()),
    }


def cell_events(cell, ice_thr, gust_thr, s):
    """Steps 5-6 for ONE cell: ice periods and wind storms that overlap in
    time are one event. Type = ice / wind / ice + wind."""
    c = hourly_flags(cell, gust_thr, s)
    storms, on_wire = ice_storms(c, s)
    periods = [(st["first"], st["end"]) for st in storms] + wind_storms(c, s)
    ev = pd.DataFrame([_event_row(c, on_wire, a, b)
                       for a, b in merge_intervals(periods)])
    if ev.empty:
        return ev
    ev["passes_ice"] = ev["peak_ice_cm"] >= ice_thr
    ev["passes_gust"] = ev["max_gust_ms"] >= gust_thr
    # passes_both: the strong gust hit the wire while the ice was on it
    ev["passes_both"] = ev["passes_ice"] & (ev["gust_while_iced_ms"] >= gust_thr)
    ev["type"] = np.select(
        [ev["passes_ice"] & ev["passes_gust"], ev["passes_ice"],
         ev["passes_gust"]],
        ["ice + wind", "ice", "wind"], default="below thresholds")
    return ev


# =====================================================================
# MAIN LOGIC: one location (all its cells), then combine cells
# =====================================================================
def find_events(df, location, ice_threshold_cm, gust_threshold_ms,
                require="either", settings=None):
    """Find events for ONE location.

    df       : hourly ERA5 table (time UTC, latitude, longitude,
               wind_speed_ms, wind_gust_ms, temp_c, precip_mm, ptype)
    location : key in locations.LOCATIONS (e.g. "toronto") or a dict
    Returns a dict of DataFrames:
      "events" : storms that pass, numbered TOR-001, TOR-002, ...
      "cells"  : those events, one row per grid cell that passed
      "below"  : storms that passed neither threshold (NOT events)
      "candidates" : every storm, before filtering (for checking)
    """
    s = settings or Settings()
    loc = get_location(location)
    if require not in ("either", "both"):
        raise ValueError("require must be 'either' or 'both'")
    lo, hi = s.gust_range_ms
    if not lo <= gust_threshold_ms <= hi:
        raise ValueError(f"gust threshold must be {lo}-{hi} m/s")
    if ice_threshold_cm <= 0:
        raise ValueError("ice threshold must be > 0")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["_cell"] = list(zip(df["latitude"].round(2), df["longitude"].round(2)))
    wanted = [(round(a, 2), round(b, 2)) for a, b in loc["cells"]]
    missing = [c for c in wanted if c not in set(df["_cell"])]
    if missing:
        raise ValueError(f"{loc['name']}: cells {missing} are not in the "
                         f"weather table. Cells present: "
                         f"{sorted(set(df['_cell']))}")
    df = df[df["_cell"].isin(wanted)]
    # The same hour can appear twice if CSVs were joined, or if a CDS file
    # mixed final ERA5 and preliminary ERA5T data (one copy is then empty).
    # A repeated hour would count its ice twice: keep the most complete copy.
    n_dup = int(df.duplicated(["time", "_cell"]).sum())
    if n_dup:
        print(f"  WARNING: {n_dup} repeated hours removed.")
        values = ["wind_speed_ms", "wind_gust_ms", "temp_c", "precip_mm"]
        df = (df.assign(_gaps=df[values].isna().sum(axis=1))
                .sort_values(["_gaps", "time"])
                .drop_duplicates(["time", "_cell"])
                .drop(columns="_gaps").sort_values("time"))
    # Missing hours would shorten or split storms: say so.
    all_hours = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    for cell in wanted:
        n_have = int((df["_cell"] == cell).sum())
        if n_have < len(all_hours):
            print(f"  WARNING: cell {cell} is missing "
                  f"{len(all_hours) - n_have} of {len(all_hours)} hours "
                  f"between {all_hours[0]:%Y-%m-%d} and "
                  f"{all_hours[-1]:%Y-%m-%d}.")
    n_bad = int((df["ptype"] < 0).sum())
    if n_bad:
        print(f"  WARNING: {n_bad} hours have unknown precipitation type "
              f"(-1). Those hours can never count as freezing rain.")

    # ---- steps 1-6 in each cell ------------------------------------
    parts = []
    for cell in wanted:
        ev = cell_events(df[df["_cell"] == cell], ice_threshold_cm,
                         gust_threshold_ms, s)
        if not ev.empty:
            ev.insert(0, "latitude", cell[0])
            ev.insert(1, "longitude", cell[1])
            parts.append(ev)
    if not parts:
        empty = pd.DataFrame()
        return {"events": empty, "cells": empty, "below": empty,
                "candidates": empty}

    candidates, cells_out = combine_cells(
        pd.concat(parts, ignore_index=True), loc, require,
        ice_threshold_cm, gust_threshold_ms, s)

    # ---- keep ONLY storms that pass; number them 1, 2, 3, ... ---------
    prefix = loc["key"].upper()[:3]
    passed = candidates[candidates["passes"]]
    new_id = {old: f"{prefix}-{i:03d}"
              for i, old in enumerate(passed["event_id"], start=1)}
    events = passed.drop(columns="passes").reset_index(drop=True)
    events["event_id"] = events["event_id"].map(new_id)

    cell_passes = (cells_out["passes_both"] if require == "both"
                   else cells_out["passes_ice"] | cells_out["passes_gust"])
    cells = cells_out[cells_out["event_id"].isin(new_id) & cell_passes].copy()
    cells["event_id"] = cells["event_id"].map(new_id)
    yes_no = {True: "yes", False: "no"}
    cells = cells.rename(columns={"passes_ice": "passes_ice_threshold",
                                  "passes_gust": "passes_wind_threshold",
                                  "passes_both": "gust_on_ice"})
    for col in ("passes_ice_threshold", "passes_wind_threshold",
                "gust_on_ice", "reported"):
        cells[col] = cells[col].map(yes_no)

    failed = candidates[~candidates["passes"]].drop(columns="passes")
    failed = failed.assign(event_id=[f"{prefix}-below-{i:03d}" for i in
                                     range(1, len(failed) + 1)])
    return {"events": events, "cells": cells.reset_index(drop=True),
            "below": failed.reset_index(drop=True), "candidates": candidates}


def combine_cells(cells, loc, require, ice_threshold_cm, gust_threshold_ms, s):
    """Cell events (from any cell of the location) that overlap in time
    are one storm. Returns (one row per storm, one row per cell per storm)."""
    cells["passes"] = (cells["passes_both"] if require == "both"
                       else cells["passes_ice"] | cells["passes_gust"])
    gust_for_rank = (cells["gust_while_iced_ms"].fillna(0)
                     if require == "both" else cells["max_gust_ms"])
    cells["severity"] = np.maximum(cells["peak_ice_cm"] / ice_threshold_cm,
                                   gust_for_rank / gust_threshold_ms)

    # Events in different cells are the same storm if they overlap or are
    # less than wind_gap_h apart (a storm moving from one cell to the next
    # an hour later is still one storm) -- same rule as within one cell.
    cells = cells.sort_values("first_hour").reset_index(drop=True)
    gap = pd.Timedelta(hours=s.wind_gap_h)
    storm_no, current_end = 0, None
    ids = []
    for first, last in zip(cells["first_hour"], cells["last_hour"]):
        if current_end is None or first > current_end + gap:
            storm_no += 1
            current_end = last
        else:
            current_end = max(current_end, last)
        ids.append(storm_no)
    prefix = loc["key"].upper()[:3]
    cells["event_id"] = [f"{prefix}-{i:03d}" for i in ids]

    tz = loc["timezone"]
    local = lambda t: (pd.Timestamp(t).tz_localize("UTC").tz_convert(tz)
                       .tz_localize(None))
    summary = []
    cells["reported"] = False
    for eid, grp in cells.groupby("event_id", sort=False):
        pool = grp[grp["passes"]] if grp["passes"].any() else grp
        best = pool["severity"].idxmax()
        cells.loc[best, "reported"] = True
        r = cells.loc[best]
        start = grp["first_hour"].min() - HOUR
        end = grp["last_hour"].max()
        summary.append({
            "event_id": eid,
            "location": loc["name"],
            "type": r["type"],
            "passes_ice_threshold": "yes" if r["passes_ice"] else "no",
            "passes_wind_threshold": "yes" if r["passes_gust"] else "no",
            "gust_on_ice": "yes" if r["passes_both"] else "no",
            "passes": bool(grp["passes"].any()),
            "start_local": local(start),
            "end_local": local(end),
            "duration_h": int((end - start) / HOUR),
            "peak_ice_cm": round(r["peak_ice_cm"], 3),
            "gust_while_iced_ms": round(r["gust_while_iced_ms"], 1),
            "max_gust_ms": round(r["max_gust_ms"], 1),
            "ice_at_max_gust_cm": round(r["ice_at_max_gust_cm"], 3),
            "max_gust_time_local": local(r["max_gust_time"]),
            "ice_on_wire_hours": int(r["ice_on_wire_hours"]),
            "freezing_rain_hours": int(r["freezing_rain_hours"]),
            "windy_hours": int(r["windy_hours"]),
            "min_temp_c": round(r["min_temp_c"], 1),
            "reported_cell": f"{r['latitude']:.2f}, {r['longitude']:.2f}",
            "cells_passing": int(grp["passes"].sum()),
            "all_cells": " | ".join(
                f"{g.latitude:.2f},{g.longitude:.2f}: "
                f"{g.peak_ice_cm:.2f} cm / {g.max_gust_ms:.1f} m/s"
                for g in grp.itertuples()),
        })
    candidates = pd.DataFrame(summary)

    cells["start_local"] = (cells["first_hour"] - HOUR).map(local)
    cells["end_local"] = cells["last_hour"].map(local)
    cells["max_gust_time_local"] = cells["max_gust_time"].map(local)
    cells["location"] = loc["name"]
    cell_cols = ["event_id", "location", "latitude", "longitude", "type",
                 "passes_ice", "passes_gust", "passes_both",
                 "start_local", "end_local", "peak_ice_cm",
                 "gust_while_iced_ms", "max_gust_ms", "ice_at_max_gust_cm",
                 "max_gust_time_local", "ice_on_wire_hours",
                 "freezing_rain_hours", "windy_hours",
                 "min_temp_c", "reported"]
    cells_out = cells[cell_cols].copy()
    for col in ("peak_ice_cm", "ice_at_max_gust_cm"):
        cells_out[col] = cells_out[col].round(3)
    for col in ("gust_while_iced_ms", "max_gust_ms", "min_temp_c"):
        cells_out[col] = cells_out[col].round(1)
    return candidates, cells_out


# =====================================================================
# Weather data
# =====================================================================
def load_weather(csv, start, end):
    """Read the weather CSV (made earlier by era5_download.py) and keep only
    the months start..end. Nothing is downloaded here."""
    csv = Path(csv)
    if not csv.exists() and (HERE / csv).exists():
        csv = HERE / csv                     # name given without a folder
    if not csv.exists():
        raise SystemExit(f"Can't find the weather CSV: {csv}\n"
                         f"Check the path (put it in quotes if it has spaces).")
    print(f"Reading {csv}")
    df = pd.read_csv(csv, parse_dates=["time"])
    t0 = pd.Period(start, freq="M").start_time
    t1 = (pd.Period(end, freq="M") + 1).start_time
    have0, have1 = df["time"].min(), df["time"].max()
    df = df[(df["time"] >= t0) & (df["time"] < t1)]
    if df.empty:
        raise SystemExit(f"The CSV has no data between {start} and {end}. "
                         f"It covers {have0:%Y-%m-%d} to {have1:%Y-%m-%d}.")
    if have0 > t0 or have1 < t1 - HOUR:
        print(f"  NOTE: you asked for {start} to {end}, but the CSV only "
              f"covers {have0:%Y-%m-%d} to {have1:%Y-%m-%d}.")
    return df, csv


# =====================================================================
# Excel output
# =====================================================================
HEADERS = {
    "event_id": ("Event ID", "Links the same storm across all sheets."),
    "location": ("Location", "Location from locations.py."),
    "latitude": ("Cell lat", "Centre of the ERA5 grid cell."),
    "longitude": ("Cell lon", "Centre of the ERA5 grid cell."),
    "type": ("Type", "ice = passed the ice threshold only; wind = passed "
             "the wind threshold only; ice + wind = passed both in the same "
             "storm."),
    "passes_ice_threshold": ("Passes ice threshold", "yes = peak ice on the "
                             "wire >= the ice threshold."),
    "passes_wind_threshold": ("Passes wind threshold", "yes = strongest gust "
                              ">= the gust threshold."),
    "gust_on_ice": ("Strong gust hit iced wire", "yes = a gust >= the gust "
                    "threshold hit the wire while ice >= the ice threshold "
                    "was on it. no = the ice and the strong gust came at "
                    "different times (or one of them is missing)."),
    "start_local": ("Start (local time)", "Start of the first flagged hour "
                    "in any cell (ERA5 hour-ending, so 1 h before the first "
                    "timestamp)."),
    "end_local": ("End (local time)", "End of the last flagged hour, "
                  "including the --ice-wind-days period."),
    "duration_h": ("Duration (h)", "End minus start."),
    "peak_ice_cm": ("Peak ice on wire (cm)", "Most radial ice on the wire "
                    "during the event (sum of Eq. 4 over the ice storm)."),
    "gust_while_iced_ms": ("Gust while iced (m/s)", "Strongest gust in hours "
                           "with ice on the wire. Blank = no ice."),
    "max_gust_ms": ("Strongest gust (m/s)", "Strongest ERA5 3-s gust in the "
                    "event."),
    "ice_at_max_gust_cm": ("Ice at strongest gust (cm)", "Ice on the wire in "
                           "the hour of the strongest gust (0 = bare wire)."),
    "max_gust_time_local": ("Strongest gust at (local)", "End of the hour "
                            "with the strongest gust."),
    "ice_on_wire_hours": ("Ice on wire (h)", "Hours the wire carried ice in "
                          "this event: from the first freezing-rain hour "
                          "until --ice-wind-days after the last one, or "
                          "until melt (--melt-hours), whichever is first."),
    "freezing_rain_hours": ("Freezing-rain hours", "Hours flagged as "
                            "freezing rain in the reported cell."),
    "windy_hours": ("Windy hours", "Hours with gust >= gust threshold in the "
                    "reported cell."),
    "min_temp_c": ("Min temp (C)", "Lowest 2 m temperature in the event."),
    "reported_cell": ("Reported cell", "Cell whose numbers are shown: the "
                      "passing cell furthest above a threshold."),
    "cells_passing": ("Cells passing", "How many of the location's cells "
                      "passed."),
    "all_cells": ("All cells (ice cm / gust m/s)", "Peak ice and strongest "
                  "gust in every cell the storm touched."),
    "reported": ("Shown on Events sheet", "yes = this cell's numbers are the "
                 "ones on the Events sheet."),
}


def _pretty(df):
    return df.rename(columns={k: v[0] for k, v in HEADERS.items()})


def save_excel(path, results, run_info, s, include_below=False):
    """Write all sheets and make them readable."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def joined(key):
        return pd.concat([r[key] for r in results], ignore_index=True)

    events, cells, below = joined("events"), joined("cells"), joined("below")

    setting_rows = [{"Setting": k, "Value": str(v), "What it does": "",
                     "Why this value / source": ""}
                    for k, v in run_info.items()]
    for f in fields(Settings):
        what, why = SETTING_NOTES[f.name]
        setting_rows.append({"Setting": f.name,
                             "Value": str(getattr(s, f.name)),
                             "What it does": what,
                             "Why this value / source": why})
    settings_df = pd.DataFrame(setting_rows)
    ice_thr = run_info.get("Ice threshold (cm)", "ICE")
    gust_thr = run_info.get("Gust threshold (m/s)", "GUST")
    rule = [
        {"Column": "HOW A STORM BECOMES AN EVENT", "Meaning": ""},
        {"Column": "1. Ice check", "Meaning": f"Peak ice on wire >= {ice_thr} "
                                              f"cm?  -> Passes ice threshold"},
        {"Column": "2. Wind check", "Meaning": f"Strongest gust >= {gust_thr} "
                                               f"m/s?  -> Passes wind threshold"},
        {"Column": "3. Keep it?", "Meaning":
            "Require = either: KEEP if 1 OR 2 is yes (one is enough). "
            "Require = both: KEEP only if the strong gust hit the iced wire."},
        {"Column": "4. Type", "Meaning": "ice (only 1 yes) / wind (only 2 "
                                         "yes) / ice + wind (both yes)"},
        {"Column": "Storms failing BOTH checks", "Meaning":
            "are NOT events and do not appear on the Events or Event cells "
            "sheets."},
        {"Column": "", "Meaning": ""},
    ]
    guide = pd.DataFrame(rule + [{"Column": v[0], "Meaning": v[1]}
                                 for v in HEADERS.values()])

    if events.empty:
        events = pd.DataFrame({"Note": [f"No storm passed the ice threshold "
                                        f"({ice_thr} cm) or the wind "
                                        f"threshold ({gust_thr} m/s)."]})
    sheets = {"Events": _pretty(events), "Event cells": _pretty(cells)}
    if include_below and not below.empty:
        sheets["Below thresholds"] = _pretty(below)
    sheets.update({"Settings": settings_df, "Column guide": guide})

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for i, col in enumerate(d.columns, start=1):
                longest = max([len(str(col))] +
                              [len(str(x)) for x in d[col].head(200)])
                width = min(max(10, longest + 2), 60)
                ws.column_dimensions[get_column_letter(i)].width = width
                if width == 60:
                    for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        row[0].alignment = Alignment(wrap_text=True,
                                                     vertical="top")
                if pd.api.types.is_datetime64_any_dtype(d[col]):
                    for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        row[0].number_format = "yyyy-mm-dd hh:mm"
                    ws.column_dimensions[get_column_letter(i)].width = 17
    return path


# =====================================================================
# Command line
# =====================================================================
def main():
    s0 = Settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", required=True,
                    help="path to the weather CSV made by era5_download.py")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM")
    ap.add_argument("--location", default="toronto",
                    help="which cells to use, from locations.py")
    ap.add_argument("--ice", type=float, default=0.635,
                    help="ice threshold in cm (default 0.635)")
    ap.add_argument("--gust", type=float, default=25.0,
                    help="gust threshold in m/s, 12.5-40 (default 25)")
    ap.add_argument("--require", choices=["either", "both"], default="either")
    ap.add_argument("--ice-wind-days", type=float, default=s0.ice_wind_days)
    ap.add_argument("--rain-below-freezing", default=s0.rain_below_freezing,
                    action=argparse.BooleanOptionalAction,
                    help="also count rain/drizzle/rain-snow mix at <= 0 C as "
                         "freezing rain (Jones; default on). Turn off with "
                         "--no-rain-below-freezing")
    ap.add_argument("--out", default=None,
                    help="Excel output (default: next to the CSV)")
    ap.add_argument("--show-below", action="store_true",
                    help="add a 'Below thresholds' sheet with the storms "
                         "that passed neither threshold (not events)")
    ap.add_argument("--ice-gap-h", type=float, default=s0.ice_gap_h)
    ap.add_argument("--melt-temp", type=float, default=s0.melt_temp_c)
    ap.add_argument("--melt-hours", type=int, default=s0.melt_hours,
                    help="hours in a row above --melt-temp before ice is "
                         "assumed melted (default 1 = Sheng/Jones rule)")
    ap.add_argument("--wind-gap-h", type=float, default=s0.wind_gap_h)
    ap.add_argument("--min-precip", type=float, default=s0.min_precip_mm)
    ap.add_argument("--ptype-hour", choices=["start_or_end", "end"],
                    default=s0.ptype_hour)
    ap.add_argument("--wind-for-ice", choices=["hour_average", "end"],
                    default=s0.wind_for_ice)
    a = ap.parse_args()

    lo, hi = s0.gust_range_ms
    if not lo <= a.gust <= hi:
        ap.error(f"--gust must be between {lo} and {hi} m/s")
    if a.ice <= 0:
        ap.error("--ice must be greater than 0")
    if a.ice_wind_days < 0:
        ap.error("--ice-wind-days can't be negative")
    if a.melt_hours < 1:
        ap.error("--melt-hours must be at least 1")
    if pd.Period(a.start, freq="M") > pd.Period(a.end, freq="M"):
        ap.error("--start is after --end; put the earlier month first")

    s = replace(s0, ice_wind_days=a.ice_wind_days,
                rain_below_freezing=a.rain_below_freezing,
                ice_gap_h=a.ice_gap_h, melt_temp_c=a.melt_temp,
                melt_hours=a.melt_hours,
                wind_gap_h=a.wind_gap_h, min_precip_mm=a.min_precip,
                ptype_hour=a.ptype_hour, wind_for_ice=a.wind_for_ice)

    loc = get_location(a.location)
    df, csv = load_weather(a.csv, a.start, a.end)
    r = find_events(df, loc, a.ice, a.gust, a.require, s)
    results = [r]
    rule = ("ice AND a gust on the iced wire" if a.require == "both"
            else "ice OR wind")
    print(f"\n{loc['name']}: {len(r['events'])} event(s) pass "
          f"({rule}; ice >= {a.ice} cm, gust >= {a.gust} m/s)")
    if len(r["events"]):
        print(r["events"][["event_id", "type", "passes_ice_threshold",
                           "passes_wind_threshold", "start_local",
                           "peak_ice_cm", "max_gust_ms"]]
              .to_string(index=False))
    if len(r["below"]):
        print(f"({len(r['below'])} smaller storm(s) passed neither threshold "
              f"and were left out. Add --show-below to list them.)")

    run_info = {"Run at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Location": a.location,
                "Months": f"{a.start} to {a.end}",
                "Ice threshold (cm)": a.ice,
                "Gust threshold (m/s)": a.gust,
                "Require": a.require,
                "Weather file": str(csv)}
    out = a.out or str(csv.parent / (
        f"events_{a.location}_{a.start}_{a.end}_"
        f"ice{a.ice:g}_gust{a.gust:g}_{a.require}.xlsx"))
    try:
        save_excel(out, results, run_info, s, a.show_below)
    except PermissionError:
        out = out.replace(".xlsx", f"_{datetime.now():%H%M%S}.xlsx")
        print("\nThe Excel file is open, so saving under a new name.")
        save_excel(out, results, run_info, s, a.show_below)
    except ImportError:
        print("\nExcel needs openpyxl:  pip install openpyxl")
        raise SystemExit(1)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()