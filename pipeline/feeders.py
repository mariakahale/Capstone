"""
feeders.py
==========
The ONE place that knows which feeders the pipeline can run, and where their
polegen2 files (poles, spans, line maps) live.

    FEEDERS            : the feeders the user can pick, with their location and area type
    feeder_file_paths(): where a feeder's polegen2 CSVs are (for the CURRENT polegen version)
    ensure_feeder_files(): same, but first runs polegen2 if those files are missing

Picking a feeder picks its area type, and the area type picks polegen2's span length
range (code_polegen2/config.py SPAN_RANGES_FT):
    3777  Toronto      suburban   spans 100-150 ft (~30-46 m)
    1878  Thunder Bay  rural      spans 300-400 ft (~91-122 m)

VERSION CHECK
    polegen2 writes  pipeline/input_feedercsvfiles/feeder_<F>/.../feeder_<F>_<table>_<area>_<v>.csv
    where <v> is POLEGEN_VERSION in code_polegen2/config.py. When polegen2 code changes,
    bump POLEGEN_VERSION there (v1 -> v2). The next pipeline run looks for the _v2 files,
    does not find them, and runs polegen2 for that feeder -- so files from an older
    polegen2 are never used by mistake.

This file only reads polegen2's config (for the version and span ranges); it does not
import the rest of polegen2. Regenerating runs  code_polegen2/main.py  as its own process,
exactly as if you typed it in the terminal.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
POLEGEN_DIR = REPO_ROOT / "code_polegen2"

# The feeders the user can pick. Key = feeder ID (as in data/opendss and polegen2).
FEEDERS = {
    "3777": {"location": "Toronto",     "area_type": "suburban"},
    "1878": {"location": "Thunder Bay", "area_type": "rural"},
}

# The polegen2 tables the pipeline reads: table name -> subfolder
#   poles, spans   -> asset_adapter.py (fragility)
#   span_line_map  -> monte_carlo_failures.py (which OpenDSS lines each span sits on)
#   pole_line_map  -> kept with them (power flow / maps)
PIPELINE_TABLES = {
    "poles": "assets",
    "spans": "assets",
    "span_line_map": "mapping",
    "pole_line_map": "mapping",
}


def _load_polegen_config():
    """code_polegen2/config.py, loaded under its own name so it can't clash with any
    other 'config' module. It only defines settings, so loading it is safe and fast."""
    spec = importlib.util.spec_from_file_location("polegen2_config", POLEGEN_DIR / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_polegen = _load_polegen_config()
POLEGEN_VERSION = _polegen.POLEGEN_VERSION        # e.g. "v1"
FEEDER_FILES_DIR = Path(_polegen.OUTPUT_DIR)      # pipeline/input_feedercsvfiles
SPAN_RANGES_FT = _polegen.SPAN_RANGES_FT          # {"suburban": (100, 150), ...}


def area_type_of(feeder):
    """The area type that goes with a feeder in FEEDERS (e.g. "3777" -> "suburban")."""
    if feeder not in FEEDERS:
        raise ValueError(f"Unknown feeder {feeder!r}. Choose from: {', '.join(FEEDERS)}")
    return FEEDERS[feeder]["area_type"]


def describe(feeder):
    """One-line label, e.g. '3777 - Toronto, suburban (spans 100-150 ft)'."""
    area = area_type_of(feeder)
    lo, hi = SPAN_RANGES_FT[area]
    return f"{feeder} - {FEEDERS[feeder]['location']}, {area} (spans {lo:g}-{hi:g} ft)"


def feeder_file_paths(feeder, area_type, version=POLEGEN_VERSION, folder=None):
    """Where polegen2's CSVs for this feeder, area type and version are.
    Returns {table: path} for the tables in PIPELINE_TABLES. Does not check they exist."""
    folder = FEEDER_FILES_DIR if folder is None else Path(folder)
    return {table: folder / f"feeder_{feeder}" / sub
                   / f"feeder_{feeder}_{table}_{area_type}_{version}.csv"
            for table, sub in PIPELINE_TABLES.items()}


def missing_feeder_files(feeder, area_type, version=POLEGEN_VERSION, folder=None):
    """The files from feeder_file_paths() that are not there yet (empty list = all there)."""
    return [p for p in feeder_file_paths(feeder, area_type, version, folder).values()
            if not p.exists()]


def run_polegen(feeder, area_type):
    """Run code_polegen2/main.py for one feeder and area type (writes the current-version
    files into FEEDER_FILES_DIR). Raises RuntimeError if polegen2 fails."""
    print(f"\nRunning polegen2 for feeder {feeder} ({area_type}, {POLEGEN_VERSION}) ...")
    result = subprocess.run(
        [sys.executable, str(POLEGEN_DIR / "main.py"),
         "--feeders", feeder, "--areas", area_type],
        cwd=POLEGEN_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"polegen2 failed for feeder {feeder} ({area_type}):\n"
                           + (result.stderr or result.stdout)[-3000:])
    print("  done.")


def ensure_feeder_files(feeder, area_type=None, version=POLEGEN_VERSION, folder=None,
                        regenerate=run_polegen):
    """Paths to this feeder's polegen2 files, running polegen2 first if any file for the
    current version is missing. area_type None = the one in FEEDERS.
    Returns {table: path} (see feeder_file_paths)."""
    area_type = area_type_of(feeder) if area_type is None else area_type
    missing = missing_feeder_files(feeder, area_type, version, folder)
    if missing:
        print(f"Feeder {feeder} ({area_type}): no polegen2 files for {version} yet "
              f"({len(missing)} missing).")
        regenerate(feeder, area_type)
        still_missing = missing_feeder_files(feeder, area_type, version, folder)
        if still_missing:
            raise FileNotFoundError("polegen2 ran but these files are still missing:\n  "
                                    + "\n  ".join(str(p) for p in still_missing))
    return feeder_file_paths(feeder, area_type, version, folder)


if __name__ == "__main__":
    # python pipeline/feeders.py : show each feeder and whether its files are ready
    print(f"polegen2 version: {POLEGEN_VERSION}   files in {FEEDER_FILES_DIR}")
    for f in FEEDERS:
        missing = missing_feeder_files(f, area_type_of(f))
        status = "ready" if not missing else f"{len(missing)} file(s) missing -> will regenerate"
        print(f"  {describe(f)}: {status}")
