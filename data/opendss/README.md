# OpenDSS source data (SMART-DS)

This folder is the **only** place the OpenDSS feeder files live. Every part of the
code (polegen, power flow, pipeline) reads from here through its own config.
Don't keep copies of these files inside code folders.

## Layout

One folder per substation, one subfolder per feeder (SMART-DS naming):

```
data/opendss/
  p1uhs0_1247/                    substation (whole-substation model, incl. Buscoords.dss)
    p1uhs0_1247--p1udt1878/       feeder "1878"
  p48rhs1_25/                     substation
    p48rhs1_25--p48rdt3777/       feeder "3777"
```

| Short name | Substation    | Feeder folder              | Master file used by code |
|------------|---------------|----------------------------|--------------------------|
| 1878       | p1uhs0_1247   | p1uhs0_1247--p1udt1878     | Master_snapshot.dss      |
| 3777       | p48rhs1_25    | p48rhs1_25--p48rdt3777     | Master_snapshot.dss      |

## Rules

1. **Treat SMART-DS files as read-only.** Don't edit `Lines.dss`, `Buscoords.dss`, etc.
2. **Put changes in a separate, clearly named file**, e.g. `Master_snapshot.dss`
   (no yearly solve, no load shapes, no plots) next to the original `Master.dss`.
3. **Fix data problems in code, where they're visible**, not by editing these files.
   Example: bus `p48rdt9397` has no coordinates in the 3777 feeder's `Buscoords.dss`;
   `code_polegen2` fills it from `p48rhs1_25/Buscoords.dss` (see `FALLBACK_BUSCOORDS`
   in `code_polegen2/config.py`).
4. **No outputs here.** Generated files go in each code folder's `outputs/`.
5. **No spaces or brackets in folder names.**

## Notes per feeder

- **1878:** the original SMART-DS `Master.dss` for this feeder is not in the repo.
  `Master_snapshot.dss` is the hand-edited version polegen has always used
  (load shapes commented out, snapshot solve). It was called `Master.dss` before
  the Nov 2026 move.
- **3777:** both the original `Master.dss` (yearly simulation) and
  `Master_snapshot.dss` / `Loads_snapshot.dss` (made by
  `code_opendsspowerflow/run_sim.py`) are here.

## Adding a feeder

Copy the SMART-DS substation folder (with the feeder subfolder inside) here, make a
`Master_snapshot.dss`, then add the feeder to `FEEDERS` in `code_polegen2/config.py`.
