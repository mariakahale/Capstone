"""
terminal_prompts.py
===================
The questions  python pipeline/run_fragility.py  asks in the terminal.

This file ONLY talks to the user: it asks, checks the answer is readable, and returns it.
It does not fetch weather or run anything -- run_fragility.py does that with the answers.
That keeps the two jobs apart: run_fragility.py works the same whether the answers come
from here, from the GUI, or from the SETTINGS block.

    ask_weather(...)  ->  (source, ice_mm, wind_ms)
                          source: "user", "csv" or "historical"
                          ice_mm, wind_ms: the typed numbers for "user", else None

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


def ask_weather_source(default="user", input_fn=input):
    """Ask which weather source to use. Returns "user", "csv" or "historical".
    Accepts the number (1-3) or the name; Enter = default."""
    names = list(SOURCES)
    print("\nWhich weather source?")
    for i, name in enumerate(names, start=1):
        mark = "  (default)" if name == default else ""
        print(f"  {i}. {name:<11s} - {SOURCES[name]}{mark}")
    while True:
        answer = input_fn(f"Choose 1-{len(names)} or a name [{default}]: ").strip().lower()
        if answer == "":
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        if answer in SOURCES:
            return answer
        print(f"  '{answer}' is not an option. Try again.")


def ask_number(question, default, input_fn=input):
    """Ask for one number >= 0. Enter = default. Asks again on a bad answer."""
    while True:
        answer = input_fn(f"{question} [{default:g}]: ").strip()
        if answer == "":
            return float(default)
        try:
            value = float(answer)
        except ValueError:
            print(f"  '{answer}' is not a number. Try again.")
            continue
        if not math.isfinite(value) or value < 0:
            print("  Must be a number >= 0. Try again.")
            continue
        return value


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
