"""
LOCATIONS
=========
Every place we analyse is defined ONCE here. Both era5_download.py and
find_weather_events.py read this file, so adding a new location means
adding one entry below and nothing else.

Each location is a list of ERA5 grid cells. A grid cell is named by its
centre point (latitude, longitude), always a multiple of 0.25 deg. One
cell is ~28 km north-south and ~20 km east-west at Toronto's latitude
(~18 km east-west at Thunder Bay). ERA5 gives ONE value per cell per
hour, so weather can only differ between poles if they are in different
cells.

To add a location:
  1. Find which 0.25 deg cells cover it. A cell centred at (lat, lon)
     covers lat +/- 0.125 and lon +/- 0.125.
  2. Add an entry below with those centres.
  3. Run:  python find_weather_events.py --location <key> ...

What gets downloaded: the cells you list plus a one-cell ring around
them (so you can widen the analysis later without re-downloading).
"""

LOCATIONS = {
    "toronto": {
        "name": "Toronto",
        # The City of Toronto (~43.58-43.86 N, ~79.64-79.12 W) sits
        # mainly in these two cells.
        "cells": [(43.75, -79.50), (43.75, -79.25)],
        "timezone": "America/Toronto",
        # Folder for the raw monthly downloads. Toronto keeps the
        # original folder name so files you already downloaded are reused.
        "raw_folder": "era5_raw",
    },
    "thunder_bay": {
        "name": "Thunder Bay",
        # City coordinates 48 deg 22'56"N 89 deg 14'46"W = 48.382, -89.246.
        # Nearest grid point: (48.50, -89.25), but the city is
        # only ~0.8 km north of the 48.375 N boundary with (48.25, -89.25),
        # so both cells are listed.
        "cells": [(48.50, -89.25), (48.25, -89.25)],
        "timezone": "America/Toronto",   # Thunder Bay uses Eastern time
        "raw_folder": "era5_raw_thunder_bay",
    },
}

GRID_STEP = 0.25   # ERA5 grid spacing, degrees


def get_location(key_or_dict):
    """Return a location dict. Accepts a key from LOCATIONS (e.g.
    "toronto") or a dict with the same fields (for quick tests)."""
    if isinstance(key_or_dict, dict):
        loc = dict(key_or_dict)
        loc.setdefault("key", loc.get("name", "custom").lower().replace(" ", "_"))
    else:
        key = str(key_or_dict).lower()
        if key not in LOCATIONS:
            raise ValueError(f"Unknown location '{key_or_dict}'. "
                             f"Choose from: {', '.join(LOCATIONS)}")
        loc = dict(LOCATIONS[key], key=key)
    loc.setdefault("timezone", "America/Toronto")
    loc.setdefault("raw_folder", f"era5_raw_{loc['key']}")
    for lat, lon in loc["cells"]:
        if abs(lat / GRID_STEP - round(lat / GRID_STEP)) > 1e-6 or \
           abs(lon / GRID_STEP - round(lon / GRID_STEP)) > 1e-6:
            raise ValueError(f"{loc['name']}: cell ({lat}, {lon}) is not on "
                             f"the 0.25 deg ERA5 grid.")
    return loc


def download_area(loc, ring=1):
    """[North, West, South, East] box covering the location's cells plus
    `ring` extra cells on every side. Edges sit on grid lines, so CDS
    returns exactly these grid points (Toronto -> 3 x 4 = 12 points)."""
    lats = [c[0] for c in loc["cells"]]
    lons = [c[1] for c in loc["cells"]]
    pad = ring * GRID_STEP
    return [max(lats) + pad, min(lons) - pad, min(lats) - pad, max(lons) + pad]
