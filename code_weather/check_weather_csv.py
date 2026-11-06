"""
CHECK THE WEATHER CSV (is the data OK? does ERA5 match real measurements?)
=========================================================================
    python check_weather_csv.py era5_toronto_last_5_years.csv

Prints five checks:
  1. Coverage   : dates, missing hours, grid points per year
  2. Precip type: is ptype filled in? (if it's broken, no freezing rain is
                  ever found and ice events disappear)
  3. Freezing rain per winter in the Toronto cells
  4. Strongest ERA5 gust days in the Toronto cells
  5. ERA5 vs MEASURED gusts at Toronto Pearson (CYYZ) on every day the
     airport measured a gust of 40 knots (20.6 m/s) or more, 2021-2025

Measured gusts: daily maximum wind gust at CYYZ from the Iowa Environmental
Mesonet archive of airport (METAR) reports, retrieved 2026-10-20.
Pearson (43.68 N, 79.63 W) sits in ERA5 cell (43.75, -79.75), right on
its edge with cell (43.75, -79.50).
"""

import sys

import numpy as np
import pandas as pd

TORONTO = [(43.75, -79.50), (43.75, -79.25)]
PEARSON_CELL = (43.75, -79.75)
KT = 0.514444                                   # knots -> m/s

# Pearson (CYYZ) daily max gust >= 40 kt, 2021-2025 (IEM daily summaries)
MEASURED_KT = {
    "2021-03-01": 46, "2021-03-11": 42, "2021-04-30": 43, "2021-12-06": 44,
    "2021-12-11": 52, "2021-12-16": 51,
    "2022-01-17": 67, "2022-02-19": 40, "2022-03-06": 46, "2022-03-31": 43,
    "2022-04-15": 43, "2022-05-21": 65, "2022-06-17": 41, "2022-12-03": 43,
    "2022-12-23": 45, "2022-12-24": 43,
    "2023-02-03": 43, "2023-02-15": 41, "2023-03-25": 42,
    "2024-02-28": 49, "2024-03-16": 41, "2024-12-12": 40,
    "2025-02-07": 41, "2025-04-03": 40, "2025-04-29": 55, "2025-05-17": 41,
    "2025-06-19": 41, "2025-11-16": 40, "2025-12-10": 40, "2025-12-19": 40,
    "2025-12-21": 41, "2025-12-29": 42,
}


def main(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df["cell"] = list(zip(df["latitude"].round(2), df["longitude"].round(2)))
    df["local"] = (df["time"].dt.tz_localize("UTC")
                   .dt.tz_convert("America/Toronto").dt.tz_localize(None))
    df["day"] = df["local"].dt.strftime("%Y-%m-%d")
    tor = df[df["cell"].isin(TORONTO)]

    # 1. coverage ------------------------------------------------------
    print("=" * 70, "\n1. COVERAGE")
    hours = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    print(f"   {df['time'].min():%Y-%m-%d %H:%M} to "
          f"{df['time'].max():%Y-%m-%d %H:%M} UTC ({len(hours):,} hours)")
    for c in TORONTO:
        n = int((df["cell"] == c).sum())
        flag = "OK" if n == len(hours) else f"MISSING {len(hours) - n:,} h"
        print(f"   Toronto cell {c}: {n:,} hours  {flag}")
    pts = df.groupby(df["time"].dt.year)["cell"].nunique()
    print("   grid points per year:", pts.to_dict())

    # 2. precipitation type -------------------------------------------
    print("=" * 70, "\n2. PRECIPITATION TYPE (Toronto cells)")
    wet = tor[tor["precip_mm"] >= 0.1]
    print(f"   hours with >= 0.1 mm: {len(wet):,}")
    names = {0: "none", 1: "rain", 3: "freezing rain", 5: "snow",
             6: "wet snow", 7: "rain/snow mix", 8: "ice pellets",
             12: "freezing drizzle", -1: "UNKNOWN (missing)"}
    counts = wet["ptype"].value_counts()
    for code, n in counts.items():
        print(f"     ptype {code:>3} {names.get(code, '?'):<18} {n:>7,} "
              f"({n / len(wet):.0%})")
    bad = counts.get(-1, 0) + counts.get(0, 0)
    if len(wet) and bad / len(wet) > 0.3:
        print("   WARNING: many wet hours have no precipitation type. "
              "ptype may be broken in this CSV -> re-download.")
    else:
        print("   OK: precipitation type is filled in.")

    # 3. freezing rain per winter --------------------------------------
    print("=" * 70, "\n3. FREEZING RAIN PER WINTER (Aug-Jul), Toronto cells")
    fr = tor[(tor["precip_mm"] >= 0.1) & tor["ptype"].isin([3, 12])].copy()
    if fr.empty:
        print("   none at all (suspicious if ptype looked broken above)")
    else:
        fr["winter"] = (fr["local"] - pd.DateOffset(months=7)).dt.year
        summ = fr.groupby(["winter", "cell"]).agg(
            hours=("precip_mm", "size"), total_mm=("precip_mm", "sum"))
        for (w, c), r in summ.iterrows():
            print(f"   {w}-{w + 1}  cell {c}: {int(r.hours):>3} h, "
                  f"{r.total_mm:5.1f} mm  (calm-wind radial ice if it all "
                  f"fell in one storm ~ {r.total_mm / (0.9 * np.pi) / 10:.2f} cm)")
        print("   For 0.635 cm of radial ice, ONE storm needs roughly 15-18 mm "
              "of freezing rain.")

    # 4. strongest ERA5 gust days -------------------------------------
    print("=" * 70, "\n4. STRONGEST ERA5 GUST DAYS (max of the two Toronto cells)")
    daily = tor.groupby("day")["wind_gust_ms"].max().sort_values(ascending=False)
    for d, g in daily.head(10).items():
        print(f"   {d}: {g:5.1f} m/s ({g * 3.6:5.0f} km/h)")
    for thr in (20, 22.5, 25):
        print(f"   days >= {thr} m/s: {int((daily >= thr).sum())}")

    # 5. ERA5 vs measured --------------------------------------------
    print("=" * 70, "\n5. ERA5 vs MEASURED GUST AT PEARSON (local calendar day)")
    print(f"   {'date':<11}{'measured':>10}{'ERA5 Toronto':>14}"
          f"{'ERA5 Pearson cell':>19}")
    pear = df[df["cell"] == PEARSON_CELL].groupby("day")["wind_gust_ms"].max()
    ratios = []
    for d, kt in MEASURED_KT.items():
        if d not in daily.index:
            continue
        obs = kt * KT
        e_t = daily[d]
        e_p = pear.get(d, np.nan)
        ratios.append(e_t / obs)
        mark = "  <- measured >= 25 m/s" if obs >= 25 else ""
        p_txt = f"{e_p:6.1f}" if np.isfinite(e_p) else "   n/a"
        print(f"   {d:<11}{obs:8.1f}  {e_t:12.1f}  {p_txt:>17}{mark}")
    if ratios:
        print(f"   ERA5 / measured, median over {len(ratios)} days: "
              f"{np.median(ratios):.2f}")
        print("   (below 1 = ERA5 lower than the airport measured)")
    else:
        print("   none of the measured dates are in this CSV")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python check_weather_csv.py <weather.csv>")
    main(sys.argv[1])
