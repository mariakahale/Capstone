"""
ERA5 EXTREME-EVENT ANALYSIS
===========================
Finds ICE STORMS and WIND STORMS in the hourly weather table made by
era5_download.py, and returns the inputs the fragility lookup needs.
Only needs pandas and numpy -- no Copernicus account or download tools.

    import pandas as pd
    from era5_info import get_info

    df = pd.read_csv("era5_toronto_2021_10_2026_09.csv", parse_dates=["time"])
    info = get_info(df)

    info["extreme_events"]       # every ice storm + wind storm, one row each
    info["fragility_input"]      # biggest ice storm: {"wind_ms","ice_cm",...}

Re-run get_info with different thresholds as often as you like; it does
not download anything.

INPUT TABLE (from era5_download.py)
-----------------------------------
    time (UTC), latitude, longitude, wind_speed_ms, wind_gust_ms,
    temp_c, precip_mm, ptype
ptype codes 3 (freezing rain) and 12 (freezing drizzle) build ice on lines.

WHAT EACH ERA5 VALUE MEANS (ECMWF "ERA5: data documentation")
-------------------------------------------------------------
  wind_speed_ms, temp_c, ptype : instantaneous, AT the timestamp
  precip_mm                    : total over the hour ENDING at the timestamp
  wind_gust_ms                 : maximum gust in the hour ENDING at the
                                 timestamp
So an event whose first flagged timestamp is 14:00 actually started at
13:00; the code subtracts one hour for start times.

GRID POINTS USED
----------------
The download box has 12 ERA5 grid points (3 lats x 4 lons), each the
centre of a ~28 km (N-S) x ~20 km (E-W) cell. Only the two cells that
cover the City of Toronto are used by default (TORONTO_POINTS):
    (43.75, -79.50)  west/central Toronto
    (43.75, -79.25)  east Toronto / Scarborough
The 43.50 row is mostly Lake Ontario, the 44.00 row is York Region, and
the -79.75 / -79.00 columns are Peel / Durham. Pass points=None to use
all 12.

ICE MODEL: CRREL simple model (Jones 1998, Eq. 4)
-------------------------------------------------
K. F. Jones (1998), "A simple model for freezing rain ice loads",
Atmospheric Research 46:87-97. Eq. 4 (p. 89) is the hourly-sum form
below; Eq. 3 is the same thing for N hours of constant weather.
Sheng, Tang & Hong (2023), Int. J. Disaster Risk Science 14:127-142,
doi:10.1007/s13753-023-00466-1, compare this model with others at
Canadian sites (incl. Toronto) and are the source of the event
definition below.

Radial ice thickness added in ONE freezing-rain hour [mm]:

    dR = sqrt( (P * rho_w)^2 + (3.6 * V * W)^2 ) / (rho_i * pi)

    P     precipitation in that hour         [mm/h]
    V     10 m wind speed                    [m/s]
    W     liquid water content of the air    [g/m^3] = 0.067 * P**0.846
    rho_w density of water = 1.0 g/cm^3
    rho_i density of glaze ice = 0.9 g/cm^3

Total ice for an event = SUM of dR over its freezing-rain hours, so the
duration is included through the sum (one term per hour). The first
term is rain falling straight down onto the wire, the second is rain
blown sideways onto it. Assumes ALL water that hits the wire freezes.

The same equation is Eq. 4 in Sheng et al. (2023), p. 129.

EVENT DEFINITIONS
-----------------
Ice storm -- Sheng et al. (2023), Sec. 3.1, p. 130: "the freezing rain
event starts if the flag of freezing rain or freezing drizzle is shown,
and the event ends if the temperature rises above 1 deg C or until there
are no further occurrences of freezing rain for 72 consecutive hours."
CAN/CSA-C22.3 uses the same rule with 24 h instead of 72 h (quoted in
Sheng et al., p. 129). Set with max_gap_h. Counts as an ICE STORM if
total ice >= ice_threshold_cm.

Wind on ice -- Sheng et al. (2023), Sec. 5, p. 135: ice can stay on a
wire long after the freezing rain stops, so the strongest wind on the
iced wire can come AFTER the event; they look at wind "during and within
dt days immediately after a freezing rain event ends" (dt = 3, 7, 21
days; their maps use 7). Set wind_after_ice_days > 0 to do the same.
Our version also stops the window early if that grid point warms above
melt_temp_c (ice gone) -- that part is OUR assumption, not the paper's.

Wind storm (our definition): hours where the gust >= gust_threshold_ms.
The user picks the threshold from GUST_SELECT_RANGE_MS (default 25 m/s
= 90 km/h). Flagged hours less than `wind_gap_h` apart are one storm.

FRAGILITY INPUT (Hou et al. 2023, Reliab. Eng. Syst. Saf. 230:108964)
--------------------------------------------------------------------
Hou et al.'s curves take radial ice thickness [cm] and the 3-s GUST
wind speed [m/s], ice 0.635-3.81 cm and wind 12.5-40 m/s. Values
outside this range are flagged (e.g. a wind storm with no ice).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------

# --- Ice model constants (CRREL simple model) -------------------------
ICE_PTYPES = (3, 12)   # freezing rain, freezing drizzle
RHO_WATER = 1.0        # g/cm^3
RHO_ICE = 0.9          # g/cm^3, glaze ice

# --- Wind: GUST ONLY --------------------------------------------------
GUST_THRESHOLD_MS = 25.0             # default, 90 km/h
GUST_SELECT_RANGE_MS = (12.5, 40.0)  # allowed choices = Hou et al. range

# --- Fragility curve range (Hou et al. 2023) --------------------------
ICE_RANGE_CM = (0.635, 3.81)
WIND_RANGE_MS = (12.5, 40.0)

# --- Grid cells covering the City of Toronto --------------------------
TORONTO_POINTS = [(43.75, -79.50), (43.75, -79.25)]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _jones_ice_mm(precip_mm, wind_ms):
    """CRREL simple model (Jones 1998, one term of Eq. 4):
    radial ice added in ONE hour [mm]."""
    P = np.asarray(precip_mm, dtype=float)
    V = np.asarray(wind_ms, dtype=float)
    W = 0.067 * P ** 0.846                      # g/m^3
    return np.sqrt((P * RHO_WATER) ** 2 + (3.6 * V * W) ** 2) / (RHO_ICE * np.pi)


def _group_hours(times, max_gap_h, break_times=None):
    """Group flagged hourly timestamps into events.

    A new event starts when (a) more than `max_gap_h` hours pass with no
    flagged hour, or (b) one of `break_times` (e.g. hours above 1 deg C,
    when the ice melts off) falls between two flagged hours.
    Returns a list of (first_timestamp, last_timestamp)."""
    t = pd.Series(sorted(pd.unique(pd.to_datetime(times))))
    if t.empty:
        return []
    new_event = t.diff() > pd.Timedelta(hours=max_gap_h)
    if break_times is not None and len(break_times):
        b = np.sort(pd.to_datetime(pd.Series(break_times)).values)
        # number of break hours up to each flagged hour; if it changed
        # since the previous flagged hour, a melt happened in between
        n_breaks = np.searchsorted(b, t.values, side="right")
        new_event |= pd.Series(np.diff(n_breaks, prepend=n_breaks[0]) > 0)
    return [(g.iloc[0], g.iloc[-1]) for _, g in t.groupby(new_event.cumsum())]


def _local(ts):
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert("America/Toronto")


def _fragility_input(gust_ms, ice_cm):
    """Pack the two numbers the fragility lookup needs + range warnings."""
    warnings = []
    if not ICE_RANGE_CM[0] <= ice_cm <= ICE_RANGE_CM[1]:
        warnings.append(f"ice outside Hou et al. range "
                        f"{ICE_RANGE_CM[0]}-{ICE_RANGE_CM[1]} cm")
    if not WIND_RANGE_MS[0] <= gust_ms <= WIND_RANGE_MS[1]:
        warnings.append(f"gust outside Hou et al. range "
                        f"{WIND_RANGE_MS[0]}-{WIND_RANGE_MS[1]} m/s")
    return {"wind_ms": round(float(gust_ms), 2),
            "ice_cm": round(float(ice_cm), 3),
            "out_of_range": warnings}


def _select_points(df, points):
    if points is None:
        return df
    keep = {(round(a, 2), round(b, 2)) for a, b in points}
    key = list(zip(df["latitude"].round(2), df["longitude"].round(2)))
    mask = np.array([k in keep for k in key])
    if not mask.any():
        raise ValueError(f"None of the points {points} are in the table.")
    return df[mask]


# ---------------------------------------------------------------------
# PUBLIC FUNCTION: get_info
# ---------------------------------------------------------------------

def get_info(df, ice_threshold_cm=0.635,
             gust_threshold_ms=GUST_THRESHOLD_MS,
             points=TORONTO_POINTS,
             max_gap_h=72, wind_gap_h=6, melt_temp_c=1.0,
             wind_after_ice_days=0,
             freezing_rule="ptype", freezing_temp_c=0.0,
             min_precip_mm=0.1, verbose=True):
    """Find ice storms and wind storms in the weather table.

    Parameters
    ----------
    df : DataFrame from era5_download.get_weather() (or its CSV).
    ice_threshold_cm : an ice event counts as an ICE STORM if its total
        radial ice >= this. Default 0.635 cm = lower end of Hou et al.
    gust_threshold_ms : an hour is part of a WIND STORM if the gust is
        >= this. Must be within GUST_SELECT_RANGE_MS (12.5-40 m/s).
        Default 25 m/s (90 km/h).
    points : list of (lat, lon) grid points to analyse. Default = the two
        cells covering Toronto. None = every point in the table.
    max_gap_h : ice event ends after this many hours with no freezing
        rain (Sheng et al. 2023: 72; CSA C22.3: 24).
    wind_gap_h : wind-flagged hours closer than this are one wind storm.
    melt_temp_c : ice event also ends once every analysed point is
        warmer than this (ice melts). Sheng et al. 2023: 1 deg C.
    wind_after_ice_days : also look for the strongest gust this many
        days AFTER each ice event ends, while the ice is still on the
        wire (Sheng et al. 2023 Sec. 5 use 3, 7, 21; maps use 7).
        0 = only wind during the freezing rain (old behaviour).
    freezing_rule : "ptype" (only ERA5 ptype 3 / 12) or "ptype_or_temp"
        (ALSO rain / rain-snow mix at 2 m temp <= freezing_temp_c).
    freezing_temp_c : temperature limit for "ptype_or_temp".
    min_precip_mm : ignore trace precipitation below this [mm/h].
    verbose : print a summary.

    Returns
    -------
    dict with
      "ice_events"      every freezing-rain event, biggest ice first
      "ice_storms"      ice_events with ice_cm >= ice_threshold_cm
      "wind_events"     every wind storm, strongest gust first
      "extreme_events"  ice storms + wind storms in one table, by date,
                        each with its fragility input (frag_wind_ms,
                        frag_ice_cm, out_of_range)
      "storm"           the biggest ice storm (dict) or None
      "fragility_input" {"wind_ms", "ice_cm", "out_of_range"} for it, or
                        None if no ice storm passes ice_threshold_cm
      "hourly"          analysed rows with the added per-hour columns
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = _select_points(df, points).copy()

    # -----------------------------------------------------------------
    # 1. WHICH HOURS HAVE FREEZING RAIN?
    # -----------------------------------------------------------------
    wet = df["precip_mm"] >= min_precip_mm
    is_fz_ptype = df["ptype"].isin(ICE_PTYPES)
    if freezing_rule == "ptype":
        df["freezing_rain"] = wet & is_fz_ptype
    elif freezing_rule == "ptype_or_temp":
        cold_rain = (df["ptype"].isin([1, 7])
                     & (df["temp_c"] <= freezing_temp_c))
        df["freezing_rain"] = wet & (is_fz_ptype | cold_rain)
    else:
        raise ValueError("freezing_rule must be 'ptype' or 'ptype_or_temp'")

    # -----------------------------------------------------------------
    # 2. ICE ADDED IN EACH HOUR (Jones 1998 Eq. 4, one term); 0 otherwise
    # -----------------------------------------------------------------
    df["ice_mm_hour"] = np.where(
        df["freezing_rain"],
        _jones_ice_mm(df["precip_mm"], df["wind_speed_ms"]), 0.0)

    # -----------------------------------------------------------------
    # 3. GUST FLAG (gust only -- no sustained-wind check)
    # -----------------------------------------------------------------
    df["gust_flag"] = df["wind_gust_ms"] >= gust_threshold_ms

    # -----------------------------------------------------------------
    # 4. ICE EVENTS: group freezing-rain hours, sum ice per grid point,
    #    keep the point with the most ice
    # -----------------------------------------------------------------
    box_min_t = df.groupby("time")["temp_c"].min()
    melt_hours = box_min_t.index[box_min_t > melt_temp_c]

    rows = []
    for first, last in _group_hours(df.loc[df["freezing_rain"], "time"],
                                    max_gap_h, melt_hours):
        win = df[(df["time"] >= first) & (df["time"] <= last)]
        per_pt = win.groupby(["latitude", "longitude"]).agg(
            ice_mm=("ice_mm_hour", "sum"),
            freezing_rain_hours=("freezing_rain", "sum"),
            gust_hours=("gust_flag", "sum"),
            gust_ms=("wind_gust_ms", "max"),
            precip_mm=("precip_mm", "sum"),
            min_temp_c=("temp_c", "min"),
        )
        lat, lon = per_pt["ice_mm"].idxmax()
        w = per_pt.loc[(lat, lon)]
        start_utc = first - pd.Timedelta(hours=1)  # hour ENDING at `first`
        rows.append({
            "type": "ice",
            "start_local": _local(start_utc),
            "end_local": _local(last),
            "duration_h": int((last - start_utc) / pd.Timedelta(hours=1)),
            "freezing_rain_hours": int(w["freezing_rain_hours"]),
            "ice_mm": round(float(w["ice_mm"]), 2),
            "ice_cm": round(float(w["ice_mm"]) / 10.0, 3),
            "gust_ms": round(float(w["gust_ms"]), 2),
            "gust_hours": int(w["gust_hours"]),
            "precip_mm": round(float(w["precip_mm"]), 1),
            "min_temp_c": round(float(w["min_temp_c"]), 1),
            "latitude": lat, "longitude": lon,
            "start_utc": start_utc, "end_utc": last,
        })
    ice_events = pd.DataFrame(rows)
    if not ice_events.empty:
        ice_events = ice_events.sort_values(
            "ice_mm", ascending=False).reset_index(drop=True)
        ice_storms = ice_events[ice_events["ice_cm"] >= ice_threshold_cm]
    else:
        ice_storms = ice_events

    # -----------------------------------------------------------------
    # 5. WIND STORMS: group gust-flagged hours, keep the point with the
    #    strongest gust
    # -----------------------------------------------------------------
    rows = []
    for first, last in _group_hours(df.loc[df["gust_flag"], "time"],
                                    wind_gap_h):
        win = df[(df["time"] >= first) & (df["time"] <= last)]
        per_pt = win.groupby(["latitude", "longitude"]).agg(
            gust_ms=("wind_gust_ms", "max"),
            gust_hours=("gust_flag", "sum"),
            ice_mm=("ice_mm_hour", "sum"),
            freezing_rain_hours=("freezing_rain", "sum"),
            precip_mm=("precip_mm", "sum"),
            min_temp_c=("temp_c", "min"),
        )
        lat, lon = per_pt["gust_ms"].idxmax()
        w = per_pt.loc[(lat, lon)]
        start_utc = first - pd.Timedelta(hours=1)
        rows.append({
            "type": "wind",
            "start_local": _local(start_utc),
            "end_local": _local(last),
            "duration_h": int((last - start_utc) / pd.Timedelta(hours=1)),
            "freezing_rain_hours": int(w["freezing_rain_hours"]),
            "ice_mm": round(float(w["ice_mm"]), 2),
            "ice_cm": round(float(w["ice_mm"]) / 10.0, 3),
            "gust_ms": round(float(w["gust_ms"]), 2),
            "gust_hours": int(w["gust_hours"]),
            "precip_mm": round(float(w["precip_mm"]), 1),
            "min_temp_c": round(float(w["min_temp_c"]), 1),
            "latitude": lat, "longitude": lon,
            "start_utc": start_utc, "end_utc": last,
        })
    wind_events = pd.DataFrame(rows)
    if not wind_events.empty:
        wind_events = wind_events.sort_values(
            "gust_ms", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------
    # 6. ONE TABLE OF EXTREME EVENTS, each with its fragility input
    # -----------------------------------------------------------------
    extreme = pd.concat([t for t in (ice_storms, wind_events) if not t.empty],
                        ignore_index=True) if not (
        ice_storms.empty and wind_events.empty) else pd.DataFrame()
    if not extreme.empty:
        fi = [_fragility_input(g, i) for g, i in
              zip(extreme["gust_ms"], extreme["ice_cm"])]
        extreme["frag_wind_ms"] = [f["wind_ms"] for f in fi]
        extreme["frag_ice_cm"] = [f["ice_cm"] for f in fi]
        extreme["out_of_range"] = ["; ".join(f["out_of_range"]) for f in fi]
        extreme = extreme.sort_values("start_utc").reset_index(drop=True)

    storm, fragility_input = None, None
    if not ice_storms.empty:
        storm = ice_storms.iloc[0].to_dict()
        fragility_input = _fragility_input(storm["gust_ms"], storm["ice_cm"])

    # -----------------------------------------------------------------
    # 7. PRINT SUMMARY
    # -----------------------------------------------------------------
    if verbose:
        pts = "all points" if points is None else f"points {points}"
        print(f"\nAnalysed {df['time'].nunique():,} hours at {pts}")
        print(f"Freezing-rain events: {len(ice_events)}  |  "
              f"ice storms >= {ice_threshold_cm} cm: {len(ice_storms)}  |  "
              f"wind storms (gust >= {gust_threshold_ms} m/s): "
              f"{len(wind_events)}")
        if storm is not None:
            print("\n=== Biggest ice storm ===")
            print(f"Start (local): {storm['start_local']:%Y-%m-%d %H:%M %Z}")
            print(f"End (local):   {storm['end_local']:%Y-%m-%d %H:%M %Z}")
            print(f"Duration:      {storm['duration_h']} h "
                  f"({storm['freezing_rain_hours']} h of freezing rain)")
            print(f"Ice:           {storm['ice_mm']} mm = {storm['ice_cm']} cm")
            print(f"Max gust:      {storm['gust_ms']} m/s")
            print(f"Grid point:    {storm['latitude']}, {storm['longitude']}")
            print(f"-> fragility_input = {fragility_input}")
        else:
            print("\nNo ice storm above the ice threshold.")
            if not ice_events.empty:
                print("  Largest freezing-rain events (below threshold):")
                print(ice_events[["start_local", "duration_h", "ice_cm",
                                  "gust_ms"]].head().to_string(index=False))
        if wind_events.empty:
            top = df.nlargest(5, "wind_gust_ms")[
                ["time", "latitude", "longitude", "wind_gust_ms"]]
            print(f"\nNo wind storm reached {gust_threshold_ms} m/s. "
                  f"Strongest gust hours found:")
            print(top.to_string(index=False))

    return {
        "ice_events": ice_events,
        "ice_storms": ice_storms,
        "wind_events": wind_events,
        "extreme_events": extreme,
        "storm": storm,
        "fragility_input": fragility_input,
        "hourly": df,
    }


# ---------------------------------------------------------------------
# Run on a CSV saved by era5_download.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    weather = pd.read_csv("era5_toronto_2025_02_04.csv", parse_dates=["time"])
    info = get_info(weather)
    cols = ["type", "start_local", "duration_h", "ice_cm", "gust_ms",
            "out_of_range"]
    if not info["extreme_events"].empty:
        print("\nAll extreme events:")
        print(info["extreme_events"][cols].to_string(index=False))