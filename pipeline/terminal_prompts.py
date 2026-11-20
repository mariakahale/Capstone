"""
terminal_prompts.py
===================
The questions  python pipeline/run_fragility.py  asks in the terminal.

This file ONLY talks to the user: it asks, checks the answer is readable, and returns it.
It does not load feeders, fetch weather or run anything -- run_fragility.py does that with
the answers. So run_fragility.py works the same whether the answers come from here, from
the GUI, or from the SETTINGS block. (That is also why this file imports nothing from the
pipeline: whatever it needs to show, such as the feeder list, is passed in.)

    ask_feeder(options, default)          -> "3777" / "1878"
    ask_weather(...)                      -> (source, ice_mm, wind_ms)
                                             ice_mm, wind_ms: typed numbers for "user", else None
    ask_monte_carlo(default_threshold)    -> (run_it, threshold)   threshold None if not run

input_fn is the function that reads one line from the user. It is input() normally;
tests pass a fake one so they can "type" answers without a keyboard.
"""

import math

# The weather sources the user can pick, in menu order, with a one-line description.
SOURCES = {
    "user":       "type the ice and wind yourself",
    "csv":        "a storm from a weather CSV",
    "historical": "a storm from the historical catalogue",
}


# =============================================================================
# Building blocks
# =============================================================================
def ask_menu(title, options, default, input_fn=input):
    """Numbered menu. options = {key: description}. Accepts the number or the key
    (any capitalisation); Enter = default. Returns the key."""
    keys = list(options)
    print(f"\n{title}")
    for i, key in enumerate(keys, start=1):
        mark = "  (default)" if key == default else ""
        print(f"  {i}. {options[key]}{mark}")
    lookup = {str(k).lower(): k for k in keys}
    while True:
        answer = input_fn(f"Choose 1-{len(keys)} [{default}]: ").strip().lower()
        if answer == "":
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]
        if answer in lookup:
            return lookup[answer]
        print(f"  '{answer}' is not an option. Try again.")


def ask_number(question, default, input_fn=input, min_value=0.0, max_value=None):
    """Ask for one number in [min_value, max_value]. Enter = default.
    Asks again on a bad answer."""
    if max_value is None:
        rule = f">= {min_value:g}"
    else:
        rule = f"between {min_value:g} and {max_value:g}"
    while True:
        answer = input_fn(f"{question} [{default:g}]: ").strip()
        if answer == "":
            return float(default)
        try:
            value = float(answer)
        except ValueError:
            print(f"  '{answer}' is not a number. Try again.")
            continue
        if (not math.isfinite(value) or value < min_value
                or (max_value is not None and value > max_value)):
            print(f"  Must be a number {rule}. Try again.")
            continue
        return value


def ask_yes_no(question, default=True, input_fn=input):
    """y/n question. Enter = default. Returns True / False."""
    hint = "Y/n" if default else "y/N"
    while True:
        answer = input_fn(f"{question} [{hint}]: ").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


# =============================================================================
# The questions run_fragility.py asks, in order
# =============================================================================
def ask_feeder(options, default, input_fn=input):
    """Which feeder to run. options = {feeder_id: label}, e.g.
    {"3777": "3777 - Toronto, suburban (spans 100-150 ft)", ...}. Returns the feeder ID."""
    return ask_menu("Which feeder?", options, default, input_fn)


def ask_weather_source(default="user", input_fn=input):
    """Which weather source. Returns "user", "csv" or "historical"."""
    options = {name: f"{name:<11s} - {text}" for name, text in SOURCES.items()}
    return ask_menu("Which weather source?", options, default, input_fn)


def ask_weather(default_source="user", default_ice_mm=0.0, default_wind_ms=0.0,
                input_fn=input):
    """Ask for the source and, for "user", the ice and wind.
    Returns (source, ice_mm, wind_ms); ice_mm and wind_ms are None unless source is "user"."""
    source = ask_weather_source(default_source, input_fn)
    if source != "user":
        return source, None, None
    ice_mm = ask_number("Ice on the wire, mm (0 = wind-only storm)", default_ice_mm, input_fn)
    wind_ms = ask_number("3-s gust, m/s (for ice: the gust DURING the ice)",
                         default_wind_ms, input_fn)
    return source, ice_mm, wind_ms


def ask_monte_carlo(default_threshold, min_threshold=0.01, max_threshold=1.0,
                    default_run=True, input_fn=input):
    """Whether to run the Monte Carlo after the fragility, and with which threshold
    (only assets with p_fail >= threshold are sampled).
    Returns (run_it, threshold); threshold is None when run_it is False."""
    print()
    if not ask_yes_no("Run the Monte Carlo after the fragility?", default_run, input_fn):
        return False, None
    threshold = ask_number("Monte Carlo threshold: sample assets with p_fail >=",
                           default_threshold, input_fn, min_threshold, max_threshold)
    return True, threshold
