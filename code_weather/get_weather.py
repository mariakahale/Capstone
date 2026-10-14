"""
GET WEATHER (ERA5 from the Copernicus Climate Data Store)
=========================================================
Two public functions:

    df = get_weather(start, end)          # 1. just download + tidy
    info = get_info(df, ice_threshold_cm=..., wind_threshold_ms=...)
                                          # 2. find ice storms, duration,
                                          #    ice accretion, wind flags,
                                          #    fragility inputs

    get_weather downloads once (cached in era5_raw/ and saved to CSV).
    get_info works on that table, so you can re-run it with different
    thresholds without downloading again:
        df = pd.read_csv("era5_toronto_hourly.csv", parse_dates=["time"])
        info = get_info(df, ice_threshold_cm=0.635)

WHAT ERA5 IS
------------
ERA5 is ECMWF's global reanalysis: a weather model re-run over the past
with real observations blended in. It gives every hour since 1940 on a
0.25 deg grid (~28 km N-S, ~20 km E-W at Toronto's latitude).

VARIABLES WE DOWNLOAD
---------------------
CDS request name                                  -> column in our table
10m_u_component_of_wind  + 10m_v_component_of_wind -> wind_speed_ms
10m_wind_gust_since_previous_post_processing       -> wind_gust_ms
2m_temperature                                     -> temp_c
total_precipitation                                -> precip_mm
precipitation_type                                 -> ptype

ptype codes (ERA5): 0 none, 1 rain, 3 freezing rain, 5 snow, 6 wet snow,
7 rain/snow mix, 8 ice pellets, 12 freezing drizzle.
Codes 3 and 12 are the ones that build ice on lines.

ICE MODEL: CRREL simple model
-----------------------------
Equation used: Sheng, Tang & Hong (2023), "Estimating and Mapping
Extreme Ice Accretion Hazard and Load Due to Freezing Rain at Canadian
Sites", Int. J. Disaster Risk Science 14:127-142, Eq. 4.
Original model: K. F. Jones (1998), "A simple model for freezing rain
ice loads", Atmospheric Research 46:87-97.

Sheng et al. compare three models (Goodwin, Chaine, CRREL) at 266
Canadian stations incl. Toronto, and recommend the CRREL model for
future Canadian design-code work: the Chaine model (current CSA basis)
is "too conservative" and depends on wire diameter.

Equivalent radial ice thickness on a horizontal wire, summed over the
N hours of the event:

    R = (1 / (rho_i * pi)) * SUM_j sqrt( (P_j * rho_w)^2 + (3.6 * V_j * W_j)^2 )

    P     precipitation rate                 [mm/h]
    V     hourly-mean wind speed             [m/s]
    W     liquid water content of the air    [g/m^3] = 0.067 * P**0.846
    rho_w density of water = 1.0 g/cm^3
    rho_i density of glaze ice = 0.9 g/cm^3
    dR    radial ice thickness added that hour [mm]

The first term is rain falling vertically onto the wire, the second is
rain blown sideways onto it. Sum dR over the event -> final ice.
It assumes ALL water that hits the wire freezes. R does not depend on
wire diameter, so one value works for every conductor.

EVENT DEFINITION (Sheng et al. 2023, Sec. 3.1)
-----------------------------------------------
An event starts at the first freezing rain / drizzle hour and ends when
the temperature rises above 1 deg C (ice melts) or when there has been
no freezing rain for `max_gap_h` hours (Sheng: 72 h; CSA C22.3: 24 h).

FRAGILITY INPUT (Hou et al. 2023, Reliab. Eng. Syst. Saf. 230:108964)
--------------------------------------------------------------------
Hou et al.'s curves take radial ice thickness t_i [cm] and the 3-s GUST
wind speed U [m/s]. Their curves cover ice 0.635-3.81 cm and wind
12.5-40 m/s; values outside this range are flagged.

NOTES
-----
* Times from ERA5 are UTC. Event tables also give Toronto local time.
* ERA5 precip/gust at timestamp t = the hour ENDING at t, so an event
  whose first flagged timestamp is 14:00 actually started at 13:00.
* Downloads are cached in `out_dir`: re-running skips months you already
  have, so an interrupted 5-year download picks up where it left off.

ONE-TIME SETUP
--------------
1. Make a free account at https://cds.climate.copernicus.eu
2. Open the "ERA5 hourly data on single levels from 1940 to present"
   dataset page -> "Download" tab -> accept the licence at the bottom.
3. Create the file  C:\\Users\\<you>\\.cdsapirc  containing two lines:
       url: https://cds.climate.copernicus.eu/api
       key: <your personal access token from your CDS profile page>
4. pip install cdsapi xarray netCDF4 pandas numpy
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------

# Box around Toronto: [North, West, South, East].
# Edges sit ON ERA5 grid lines (multiples of 0.25 deg) so CDS returns
# exactly 3 lats x 4 lons = 12 grid points:
#   lats 44.00, 43.75, 43.50     lons -79.75, -79.50, -79.25, -79.00
# NOTE: the 43.50 row is over Lake Ontario.
TORONTO_AREA = [44.00, -79.75, 43.50, -79.00]

DATASET = "reanalysis-era5-single-levels"

# Variables are split into two requests per month. CDS's NetCDF
# converter silently drops variables when one request mixes ERA5
# "analysis" and "forecast" fields -- precipitation_type was being lost
# that way. Asking for it on its own avoids the problem.
REQUEST_GROUPS = {
    "main": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "10m_wind_gust_since_previous_post_processing",
        "2m_temperature",
        "total_precipitation",
    ],
    "ptype": [
        "precipitation_type",
    ],
}

# ERA5 short names can differ between file versions; list every
# spelling we might see and map them to one clean column name.
_RENAME = {
    "u10": "u10", "10u": "u10",
    "v10": "v10", "10v": "v10",
    "fg10": "wind_gust_ms", "10fg": "wind_gust_ms", "i10fg": "wind_gust_ms",
    "t2m": "t2m", "2t": "t2m",
    "tp": "tp",
    "ptype": "ptype",
}

# --- Ice model constants (CRREL simple model) -------------------------
ICE_PTYPES = (3, 12)   # freezing rain, freezing drizzle
RHO_WATER = 1.0        # g/cm^3
RHO_ICE = 0.9          # g/cm^3, glaze ice

# --- Wind thresholds --------------------------------------------------
# Environment Canada wind-warning criteria for southern Ontario
# (sustained 60 km/h and/or gusts 90 km/h). Change these to match the
# range the fragility curve cares about.
WIND_THRESHOLD_MS = 60 / 3.6   # 16.7 m/s sustained
GUST_THRESHOLD_MS = 90 / 3.6   # 25.0 m/s gust


# ---------------------------------------------------------------------
# STEP A: download one month
# ---------------------------------------------------------------------

def _download_month(client, year, month, area, out_dir, group,
                    overwrite=False):
    """Request one month of hourly ERA5 data for one variable group
    and save it to disk.

    Returns the path of the downloaded file (.nc or .zip -- CDS sends a
    zip when the variables have different time-step types).
    """
    suffix = "" if group == "main" else f"_{group}"
    target = Path(out_dir) / f"era5_{year}_{month:02d}{suffix}.download"
    if target.exists() and not overwrite:
        print(f"  {year}-{month:02d} [{group}]: already downloaded, skipping")
        return target

    days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
    request = {
        "product_type": ["reanalysis"],
        "variable": REQUEST_GROUPS[group],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"  {year}-{month:02d} [{group}]: requesting from CDS "
          f"(queued on server)...")
    tmp = target.with_suffix(".part")
    client.retrieve(DATASET, request, str(tmp))
    tmp.rename(target)  # only mark complete once the download finished
    return target


# ---------------------------------------------------------------------
# STEP B: open a downloaded file (handles the zip case)
# ---------------------------------------------------------------------

def _open_download(path):
    """Open one downloaded month as a single xarray Dataset.

    Since late 2024, CDS splits variables by time-step type into separate
    NetCDF files (e.g. ...stepType-instant.nc, ...stepType-accum.nc,
    ...stepType-max.nc) and zips them. We open each and merge them.
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        extract_dir = path.with_suffix("")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        parts = [xr.open_dataset(p).load()
                 for p in sorted(extract_dir.glob("*.nc"))]
        ds = xr.merge(parts, compat="override", join="outer")
    else:
        ds = xr.open_dataset(path).load()

    # Newer files call the time axis "valid_time"; older ones "time".
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.drop_vars("time")
        ds = ds.rename({"valid_time": "time"})

    # Some CDS files give longitude as 0..360 (e.g. 280.25) and others as
    # -180..180 (-79.75). Put everything on -180..180 so the two
    # requests line up when merged -- otherwise ptype lands on its own
    # "phantom" longitudes and every real point reads as ptype 0.
    if "longitude" in ds.coords and float(ds["longitude"].max()) > 180:
        ds = ds.assign_coords(
            longitude=((ds["longitude"] + 180) % 360) - 180
        ).sortby("longitude")
    ds = ds.assign_coords(latitude=ds["latitude"].round(4),
                          longitude=ds["longitude"].round(4))

    # Drop bookkeeping coords we don't need (number, expver, ...).
    keep = {"time", "latitude", "longitude"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep])

    return ds.rename({k: v for k, v in _RENAME.items()
                      if k in ds.data_vars and k != v})


# ---------------------------------------------------------------------
# STEP C: convert raw ERA5 fields into engineering units
# ---------------------------------------------------------------------

def _to_table(ds):
    """Turn the raw Dataset into a tidy DataFrame in the units the
    fragility lookup expects."""
    out = xr.Dataset()

    # Hourly-mean wind speed from its east (u) and north (v) parts.
    out["wind_speed_ms"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    # Max gust in the hour ending at this timestamp (m/s).
    # (NaN column if missing, so the wind flag still works.)
    if "wind_gust_ms" in ds:
        out["wind_gust_ms"] = ds["wind_gust_ms"]
    else:
        out["wind_gust_ms"] = xr.full_like(ds["u10"], np.nan)

    # Kelvin -> Celsius.
    out["temp_c"] = ds["t2m"] - 273.15

    # ERA5 precipitation is metres of water accumulated over the hour.
    # x1000 -> mm in that hour (= mm/h). Clip tiny negative round-off.
    out["precip_mm"] = (ds["tp"] * 1000.0).clip(min=0.0)

    if "ptype" in ds:
        n_missing = int(ds["ptype"].isnull().sum())
        if n_missing:
            print(f"  WARNING: {n_missing} ptype values missing -- "
                  f"set to -1 (unknown), NOT 0.")
        out["ptype"] = ds["ptype"].fillna(-1).astype(int)
    else:
        print("  WARNING: precipitation_type missing from this month's "
              "download -- ptype column set to -1 (unknown).")
        out["ptype"] = xr.full_like(ds["t2m"], -1, dtype=int)

    df = (out.to_dataframe()
             .reset_index()
             .sort_values(["time", "latitude", "longitude"])
             .reset_index(drop=True))
    df["time"] = pd.to_datetime(df["time"])  # UTC
    return df


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _jones_ice_mm(precip_mm, wind_ms):
    """CRREL simple model (Sheng et al. 2023 Eq. 4 / Jones 1998):
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


# ---------------------------------------------------------------------
# PUBLIC FUNCTION 1: get_weather  (download only)
# ---------------------------------------------------------------------

def get_weather(start="2025-02", end="2025-04", area=TORONTO_AREA,
                out_dir="era5_raw", save_csv="era5_toronto_hourly.csv",
                overwrite=False):
    """Download hourly ERA5 weather for a box and return one tidy table.

    Parameters
    ----------
    start, end : "YYYY-MM", first and last month (both inclusive).
    area : [North, West, South, East] in degrees. Default = Toronto.
    out_dir : folder for the raw monthly downloads (acts as a cache).
    save_csv : also write the table to this CSV (None = don't).
    overwrite : re-download months already in out_dir.

    Returns
    -------
    pandas.DataFrame, one row per (hour, grid point):
        time (UTC), latitude, longitude,
        wind_speed_ms   hourly-mean 10 m wind speed      [m/s]
        wind_gust_ms    max 3-s 10 m gust in past hour   [m/s]
        temp_c          2 m air temperature              [deg C]
        precip_mm       precipitation in the past hour   [mm]
        ptype           precipitation type code (see module docstring)
    """
    import cdsapi  # imported here so get_info works without it

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    months = pd.period_range(start, end, freq="M")
    print(f"Fetching {len(months)} month(s) of ERA5, area={area}")

    tables = []
    for m in months:
        parts = [
            _open_download(_download_month(client, m.year, m.month, area,
                                           out_dir, group, overwrite))
            for group in REQUEST_GROUPS
        ]
        ds = xr.merge(parts, compat="override", join="outer")
        if ds.sizes["longitude"] != parts[0].sizes["longitude"]:
            raise ValueError(
                f"{m}: the variable groups have different grids "
                f"({ds.sizes['longitude']} longitudes after merging). "
                f"Check the files in {out_dir}.")
        tables.append(_to_table(ds))
    df = pd.concat(tables, ignore_index=True)

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Saved {len(df):,} rows to {save_csv}")
    return df


# ---------------------------------------------------------------------
# PUBLIC FUNCTION 2: get_info  (ice storms, duration, ice, wind flags)
# ---------------------------------------------------------------------

def get_info(df, ice_threshold_cm=0.635,
             wind_threshold_ms=WIND_THRESHOLD_MS,
             gust_threshold_ms=GUST_THRESHOLD_MS,
             max_gap_h=72, melt_temp_c=1.0,
             freezing_rule="ptype", freezing_temp_c=0.0,
             min_precip_mm=0.1):
    """Find ice storms in a get_weather() table and summarise them.

    Parameters
    ----------
    df : DataFrame from get_weather() (or read back from its CSV).
    ice_threshold_cm : an ice event counts as an ICE STORM if its radial
        ice >= this. Default 0.635 cm (0.25 in) = lower end of the
        Hou et al. 2023 fragility curves.
    wind_threshold_ms, gust_threshold_ms : an hour is wind-flagged if
        sustained wind OR gust is >= these. Default = Environment Canada
        wind-warning levels (60 km/h sustained, 90 km/h gust).
    max_gap_h : an ice event ends after this many hours with no freezing
        rain (Sheng et al. 2023: 72; CSA C22.3: 24).
    melt_temp_c : an ice event also ends once the whole box is warmer
        than this (ice melts). Sheng et al. 2023: 1 deg C.
    freezing_rule : how to decide an hour has freezing rain
        "ptype"          (default) only ERA5 ptype 3 (freezing rain) /
                         12 (freezing drizzle)
        "ptype_or_temp"  ALSO rain (1) or rain/snow mix (7) falling while
                         2 m temp <= freezing_temp_c. Catches hours ERA5
                         labels as rain at sub-zero temps; more ice, but
                         can sweep in mixed/snow events.
    freezing_temp_c : temperature limit for "ptype_or_temp".
    min_precip_mm : ignore trace precipitation below this [mm/h].

    Returns
    -------
    dict with
      "ice_events"      every freezing-rain event, biggest ice first
      "ice_storms"      ice_events with ice_cm >= ice_threshold_cm
      "storm"           the biggest ice storm (dict) or None
      "fragility_input" {"wind_ms", "ice_cm", "out_of_range"} for it
      "wind_hours"      every wind-flagged hour (time, point, wind, gust)
      "hourly"          df with the added per-hour columns
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

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
    # 2. ICE ACCRETION PER HOUR (Sheng et al. 2023 Eq. 4)
    # -----------------------------------------------------------------
    df["ice_mm_hour"] = np.where(
        df["freezing_rain"],
        _jones_ice_mm(df["precip_mm"], df["wind_speed_ms"]), 0.0)

    # -----------------------------------------------------------------
    # 3. WIND FLAG
    # -----------------------------------------------------------------
    df["wind_flag"] = ((df["wind_speed_ms"] >= wind_threshold_ms)
                       | (df["wind_gust_ms"] >= gust_threshold_ms))

    # -----------------------------------------------------------------
    # 4. GROUP FREEZING-RAIN HOURS INTO EVENTS + DURATION
    #    An hour counts if ANY grid point has freezing rain. Numbers for
    #    each event come from the grid point with the most ice.
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
            wind_flag_hours=("wind_flag", "sum"),
            wind_ms=("wind_speed_ms", "max"),
            gust_ms=("wind_gust_ms", "max"),
            precip_mm=("precip_mm", "sum"),
            min_temp_c=("temp_c", "min"),
        )
        lat, lon = per_pt["ice_mm"].idxmax()
        w = per_pt.loc[(lat, lon)]
        start_utc = first - pd.Timedelta(hours=1)  # hour ENDING at `first`
        rows.append({
            "start_local": _local(start_utc),
            "end_local": _local(last),
            "duration_h": int((last - start_utc) / pd.Timedelta(hours=1)),
            "freezing_rain_hours": int(w["freezing_rain_hours"]),
            "ice_mm": round(float(w["ice_mm"]), 2),
            "ice_cm": round(float(w["ice_mm"]) / 10.0, 3),
            "wind_ms": round(float(w["wind_ms"]), 2),
            "gust_ms": round(float(w["gust_ms"]), 2),
            "wind_flag_hours": int(w["wind_flag_hours"]),
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
    # 5. BIGGEST ICE STORM -> FRAGILITY INPUT
    #    Hou et al. 2023 curves take radial ice [cm] + 3-s GUST [m/s].
    # -----------------------------------------------------------------
    storm, fragility_input = None, None
    if not ice_storms.empty:
        storm = ice_storms.iloc[0].to_dict()
        wind_for_curve = (storm["gust_ms"] if pd.notna(storm["gust_ms"])
                          else storm["wind_ms"])
        warnings = []
        if not 0.635 <= storm["ice_cm"] <= 3.81:
            warnings.append("ice outside Hou et al. range 0.635-3.81 cm")
        if not 12.5 <= wind_for_curve <= 40:
            warnings.append("wind outside Hou et al. range 12.5-40 m/s")
        fragility_input = {"wind_ms": wind_for_curve,
                           "ice_cm": storm["ice_cm"],
                           "out_of_range": warnings}

    # -----------------------------------------------------------------
    # 6. PRINT SUMMARY (+ diagnostics if no ice storm was found)
    # -----------------------------------------------------------------
    print(f"\nFreezing-rain events found: {len(ice_events)}  |  "
          f"ice storms >= {ice_threshold_cm} cm: {len(ice_storms)}")
    if storm is not None:
        print("\n=== Biggest ice storm ===")
        print(f"Start (local): {storm['start_local']:%Y-%m-%d %H:%M %Z}")
        print(f"End (local):   {storm['end_local']:%Y-%m-%d %H:%M %Z}")
        print(f"Duration:      {storm['duration_h']} h "
              f"({storm['freezing_rain_hours']} h of freezing rain)")
        print(f"Ice:           {storm['ice_mm']} mm = {storm['ice_cm']} cm")
        print(f"Wind:          {storm['wind_ms']} m/s sustained, "
              f"{storm['gust_ms']} m/s gust "
              f"({storm['wind_flag_hours']} h above threshold)")
        print(f"Grid point:    {storm['latitude']}, {storm['longitude']}")
        print(f"-> fragility_input = {fragility_input}")
    else:
        print("\nNo ice storm above the threshold. Diagnostics:")
        if not ice_events.empty:
            print("  Largest freezing-rain events (below threshold):")
            print(ice_events[["start_local", "duration_h", "ice_cm",
                              "gust_ms"]].head().to_string(index=False))
        print("  Hours by ERA5 ptype (all grid points, wet hours only):")
        print(df.loc[wet, "ptype"].value_counts().sort_index()
                .to_string())
        cold_wet = wet & (df["temp_c"] <= freezing_temp_c)
        print(f"  Wet hours at or below {freezing_temp_c} C, by ptype:")
        print(df.loc[cold_wet, "ptype"].value_counts().sort_index()
                .to_string())

    return {
        "ice_events": ice_events,
        "ice_storms": ice_storms,
        "storm": storm,
        "fragility_input": fragility_input,
        "wind_hours": df.loc[df["wind_flag"],
                             ["time", "latitude", "longitude",
                              "wind_speed_ms", "wind_gust_ms"]],
        "hourly": df,
    }


# ---------------------------------------------------------------------
# Run: Feb-Apr 2025 (contains the 29-31 March 2025 ice storm)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    weather = get_weather(start="2025-02", end="2025-04",
                          save_csv="era5_toronto_2025_02_04.csv")
    info = get_info(weather, ice_threshold_cm=0.635,
                    wind_threshold_ms=WIND_THRESHOLD_MS)
    print("\nAll freezing-rain events:")
    if info["ice_events"].empty:
        print("  (none)")
    else:
        print(info["ice_events"].drop(columns=["start_utc", "end_utc"])
              .head(10).to_string(index=False))
