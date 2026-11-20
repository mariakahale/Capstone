"""
Tests for pipeline/terminal_prompts.py (the questions asked in the terminal).
A fake input_fn "types" the answers, so no keyboard is needed.

Run all tests from the repo folder:   python3 -m pytest tests
"""
import pytest

from pipeline import terminal_prompts as tp


def _typing(*answers):
    """A fake input() that returns these answers one at a time."""
    it = iter(answers)
    return lambda prompt="": next(it)


@pytest.mark.parametrize("typed, expected", [
    ("1", "user"), ("2", "csv"), ("3", "historical"),
    ("csv", "csv"), ("  Historical ", "historical"),
])
def test_ask_weather_source_accepts_numbers_and_names(typed, expected):
    assert tp.ask_weather_source(input_fn=_typing(typed)) == expected


def test_ask_weather_source_enter_keeps_the_default():
    assert tp.ask_weather_source(default="csv", input_fn=_typing("")) == "csv"


def test_ask_weather_source_asks_again_after_a_bad_answer(capsys):
    assert tp.ask_weather_source(input_fn=_typing("7", "forecast", "1")) == "user"
    assert capsys.readouterr().out.count("not an option") == 2


def test_ask_weather_user_mode_reads_ice_and_wind():
    assert tp.ask_weather(input_fn=_typing("user", "17.3", "14.9")) == ("user", 17.3, 14.9)


def test_ask_weather_other_modes_ask_no_numbers():
    assert tp.ask_weather(input_fn=_typing("csv")) == ("csv", None, None)
    assert tp.ask_weather(input_fn=_typing("3")) == ("historical", None, None)


def test_ask_weather_user_mode_rejects_bad_numbers_then_accepts(capsys):
    out = tp.ask_weather(input_fn=_typing("1", "-2", "abc", "nan", "0", "50"))
    assert out == ("user", 0.0, 50.0)
    printed = capsys.readouterr().out
    assert ">= 0" in printed and "not a number" in printed


def test_ask_weather_user_mode_enter_keeps_the_defaults():
    out = tp.ask_weather(default_ice_mm=14.7, default_wind_ms=12.3,
                         input_fn=_typing("1", "", ""))
    assert out == ("user", 14.7, 12.3)
