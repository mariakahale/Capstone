"""
ERA5 DOWNLOAD (Copernicus Climate Data Store)
=============================================
Makes the hourly weather CSV that find_weather_events.py reads.
Raw monthly files are saved in era5_raw/ next to this file and reused, so
an interrupted download picks up where it left off. Locations are in
locations.py.

Download what's missing, then build the CSV:
    python era5_download.py --start 2021-01 --end 2025-12
Build the CSV ONLY from files already in era5_raw (never downloads):
    python era5_download.py --start 2021-01 --end 2025-12 --no-download
Either way the CSV is era5_toronto_2021_01_2025_12.csv next to this file
(change with --csv, other places with --location).

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

OUTPUT TABLE (one row per hour per grid point)
----------------------------------------------
    time (UTC), latitude, longitude,
    wind_speed_ms   10 m wind speed AT the timestamp [m/s]   (instantaneous)
    wind_gust_ms    max 3-s 10 m gust in past hour   [m/s]   (max over hour)
    temp_c          2 m air temperature AT timestamp [deg C] (instantaneous)
    precip_mm       precipitation in the past hour   [mm]    (hour total)
    ptype           precipitation type AT timestamp  [code]  (instantaneous)
Where this is stated: ECMWF "ERA5: data documentation" (Copernicus
Knowledge Base), Table 2 (instantaneous: 10u, 10v, 2t, ptype), Table 3
(accumulations: tp) and Table 5 (min/max: 10fg), plus its notes
"reanalysis: accumulations are over the hour ... ending at the validity
date/time" and "reanalysis: the minimum or maximum values are in the
hour ... ending at the validity date/time". The CDS variable list
describes 10fg as "Maximum 3 second wind at 10 m height as defined by
WMO".

NOTES
-----
* Times from ERA5 are UTC.
* ERA5 precip/gust at timestamp t = the hour ENDING at t.
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

# Box around Toronto: [North, West, South, East] = [44.00, -79.75, 43.50, -79.00].
# Edges sit ON ERA5 grid lines (multiples of 0.25 deg) so CDS returns
# exactly 3 lats x 4 lons = 12 grid points:
#   lats 44.00, 43.75, 43.50     lons -79.75, -79.50, -79.25, -79.00
# = Toronto's two cells (see locations.py) plus a ring of neighbouring
# cells, so you can widen the analysis without re-downloading.
# Other locations: download_area(get_location("<key>")) builds the box.
from locations import download_area, get_location
TORONTO_AREA = download_area(get_location("toronto"))

DATASET = "reanalysis-era5-single-levels"

# Raw downloads and the CSV go in the folder this file is in (e.g.
# code_weather), wherever you run Python from.
HERE = Path(__file__).resolve().parent

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
    # (NOT "i10fg": that is the instantaneous gust, a different quantity)
    "fg10": "wind_gust_ms", "10fg": "wind_gust_ms",
    "t2m": "t2m", "2t": "t2m",
    "tp": "tp",
    "ptype": "ptype",
}


# ---------------------------------------------------------------------
# STEP A: download one month
# ---------------------------------------------------------------------

def _saved_path(out_dir, year, month, group):
    """Where one month of one variable group is saved."""
    suffix = "" if group == "main" else f"_{group}"
    return Path(out_dir) / f"era5_{year}_{month:02d}{suffix}.download"


def _download_month(client, year, month, area, out_dir, group,
                    overwrite=False):
    """Request one month of hourly ERA5 data for one variable group
    and save it to disk.

    Returns the path of the downloaded file (.nc or .zip -- CDS sends a
    zip when the variables have different time-step types).
    """
    target = _saved_path(out_dir, year, month, group)
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

    # Months that mix final ERA5 (expver 1) with preliminary ERA5T
    # (expver 5) can come with an "expver" dimension: every hour appears
    # twice, once filled with NaN. Collapse it to one value per hour,
    # preferring final ERA5. (ECMWF forum: "ERA5 CDS requests which return
    # a mixture of ERA5 and ERA5T data".)
    if "expver" in ds.dims:
        order = sorted(range(ds.sizes["expver"]),
                       key=lambda i: str(ds["expver"].values[i]))
        merged = ds.isel(expver=order[0], drop=True)
        for i in order[1:]:
            merged = merged.combine_first(ds.isel(expver=i, drop=True))
        ds = merged

    # Longitude can be written two ways: -180..180 (Toronto = -79.75) or
    # 0..360 (Toronto = 280.25, i.e. 360 - 79.75). CDS sometimes returns
    # the "main" and "ptype" downloads in different conventions. If they
    # are merged as-is, xarray treats 280.25 and -79.75 as DIFFERENT
    # places, so ptype ends up on separate "phantom" columns and every
    # real grid point reads ptype = missing/0. Convert 0..360 to
    # -180..180:  ((280.25 + 180) % 360) - 180 = -79.75
    # Then round to 4 decimals so tiny float differences (-79.7500001)
    # don't stop the two files from lining up.
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

    # 10 m wind speed at the timestamp (instantaneous) from its east (u)
    # and north (v) parts.
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
# Check a saved month is for the area we asked for
# ---------------------------------------------------------------------

def _check_area(month, ds, area, out_dir, tol=0.01):
    """Saved files are named by month only, so check what they cover.
      same box as asked            -> fine
      smaller box INSIDE it        -> fine, with a note (same place, fewer
                                      grid points, e.g. only Toronto's 2)
      box outside / different area -> error (e.g. Toronto files reused
                                      for Thunder Bay)"""
    n, w, s, e = area
    got = (float(ds["latitude"].max()), float(ds["longitude"].min()),
           float(ds["latitude"].min()), float(ds["longitude"].max()))
    if all(abs(g - x) <= tol for g, x in zip(got, area)):
        return None
    gn, gw, gs, ge = got
    inside = (gn <= n + tol and gs >= s - tol and gw >= w - tol
              and ge <= e + tol)
    if inside:
        return got                       # smaller box: fine, report later
    raise ValueError(
        f"{month}: files in '{out_dir}' cover [N, W, S, E] = {got}, which "
        f"is outside the area you asked for, {area}. They are for another "
        f"place: give each location its own raw_folder in locations.py.")


# ---------------------------------------------------------------------
# PUBLIC FUNCTION: get_weather
# ---------------------------------------------------------------------

def get_weather(start="2025-02", end="2025-04", area=TORONTO_AREA,
                out_dir=HERE / "era5_raw",
                save_csv=HERE / "era5_toronto_hourly.csv",
                overwrite=False, download=True):
    """Build one tidy hourly table from ERA5, downloading what's missing.

    Parameters
    ----------
    start, end : "YYYY-MM", first and last month (both inclusive).
    area : [North, West, South, East] in degrees. Default = Toronto.
    out_dir : folder for the raw monthly downloads (acts as a cache).
    save_csv : also write the table to this CSV (None = don't).
    overwrite : re-download months already in out_dir.
    download : False = never contact CDS; only use files already in
               out_dir (months with no saved files are skipped and listed).

    Returns
    -------
    pandas.DataFrame, one row per (hour, grid point). See the module
    docstring for the columns.
    """
    if pd.Period(start, freq="M") > pd.Period(end, freq="M"):
        raise ValueError(
            f"start ({start}) is after end ({end}). Put the EARLIER month "
            f"first, e.g. get_weather(start='2021-01', end='2025-12').")
    months = pd.period_range(start, end, freq="M")
    out_dir = Path(out_dir)
    if download:
        import cdsapi  # only needed for downloading
        out_dir.mkdir(parents=True, exist_ok=True)
        client = cdsapi.Client()
        print(f"Fetching {len(months)} month(s) of ERA5, area={area}")
    else:
        print(f"Building from saved files in '{out_dir}' (no download): "
              f"{len(months)} month(s)")

    tables, missing, smaller, mixed = [], [], {}, []
    for m in months:
        if download:
            paths = [_download_month(client, m.year, m.month, area, out_dir,
                                     group, overwrite)
                     for group in REQUEST_GROUPS]
        else:
            paths = [_saved_path(out_dir, m.year, m.month, group)
                     for group in REQUEST_GROUPS]
            if not all(p.exists() for p in paths):
                missing.append(str(m))
                continue
        parts = [_open_download(p) for p in paths]
        # Keep the grid points that BOTH variable groups have (they can
        # differ if the two files were downloaded with different boxes).
        ds = xr.merge(parts, compat="override", join="inner")
        sizes = [p.sizes["latitude"] * p.sizes["longitude"] for p in parts]
        n_pts = ds.sizes["latitude"] * ds.sizes["longitude"]
        if n_pts == 0:
            raise ValueError(f"{m}: the saved variable groups have no grid "
                             f"points in common. Check the files in "
                             f"'{out_dir}'.")
        if len(set(sizes + [n_pts])) > 1:
            mixed.append(str(m))
        box = _check_area(m, ds, area, out_dir)
        if box is not None:
            smaller.setdefault((box, n_pts), []).append(str(m))
        tables.append(_to_table(ds))

    def span(ms):
        return ms[0] if len(ms) == 1 else f"{ms[0]} to {ms[-1]} ({len(ms)} months)"

    for (box, n_pts), ms in smaller.items():
        print(f"NOTE: {span(ms)} were saved with a smaller box "
              f"[N, W, S, E] = {box} ({n_pts} grid points) than {area}. "
              f"They are used as they are. Only matters if you need the "
              f"extra grid points: then delete those months' .download "
              f"files in '{out_dir}' and download again.")
    if mixed:
        print(f"NOTE: in {', '.join(mixed)} the two saved files had "
              f"different boxes; only the grid points in both were kept.")
    if missing:
        print(f"WARNING: no saved files for {len(missing)} month(s), so they "
              f"are NOT in the CSV: {', '.join(missing)}")
    if not tables:
        raise ValueError(f"No months found in '{out_dir}' for {start} to {end}.")
    df = pd.concat(tables, ignore_index=True)

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Saved {len(df):,} rows to {save_csv}")
    return df


# ---------------------------------------------------------------------
# Command line
#   python era5_download.py --start 2021-01 --end 2025-12
#   python era5_download.py --start 2021-01 --end 2025-12 --no-download
# ---------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Download ERA5 and/or build the weather CSV.")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM")
    ap.add_argument("--location", default="toronto",
                    help="location from locations.py (default toronto)")
    ap.add_argument("--no-download", action="store_true",
                    help="never contact CDS; build the CSV only from files "
                         "already saved in the raw folder")
    ap.add_argument("--csv", default=None,
                    help="output CSV (default: era5_<location>_<start>_<end>"
                         ".csv next to this file)")
    a = ap.parse_args()
    loc = get_location(a.location)
    csv = Path(a.csv) if a.csv else HERE / (
        f"era5_{loc['key']}_{a.start.replace('-', '_')}_"
        f"{a.end.replace('-', '_')}.csv")
    try:
        get_weather(a.start, a.end, area=download_area(loc),
                    out_dir=HERE / loc["raw_folder"], save_csv=csv,
                    download=not a.no_download)
    except ValueError as err:
        raise SystemExit(f"ERROR: {err}")


if __name__ == "__main__":
    main()