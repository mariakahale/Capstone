"""
Tests for code_weather/weather_input.py -- the weather "contract":
every source returns (ice_mm, wind_ms), both numbers >= 0.

Run all tests from the repo folder:   python3 -m pytest tests
"""
import math

import pytest

import weather_input
from weather_input import check_weather, user_storm


# ---------------------------------------------------------------------------
# Option 3: user_storm
# ---------------------------------------------------------------------------
def test_user_storm_returns_the_two_numbers():
    assert user_storm(17.3, 14.9) == (17.3, 14.9)


def test_user_storm_wind_only_storm():
    assert user_storm(0, 23.8) == (0.0, 23.8)


def test_user_storm_accepts_numbers_typed_as_text():
    # GUI text boxes give text, not numbers
    assert user_storm("20", "30.5") == (20.0, 30.5)


def test_user_storm_returns_floats():
    ice_mm, wind_ms = user_storm(20, 30)
    assert isinstance(ice_mm, float) and isinstance(wind_ms, float)


@pytest.mark.parametrize("ice_mm, wind_ms", [
    (-5, 14.9),            # negative ice
    (17.3, -1),            # negative wind
    (17.3, "windy"),       # not a number
    (None, 10),            # missing
    (float("nan"), 10),    # NaN (e.g. an empty spreadsheet cell)
    (10, math.inf),        # infinite
])
def test_user_storm_rejects_bad_input(ice_mm, wind_ms):
    with pytest.raises(ValueError):
        user_storm(ice_mm, wind_ms)


def test_check_weather_error_message_names_the_bad_input():
    with pytest.raises(ValueError, match="ice_mm"):
        check_weather(-5, 10)


# ---------------------------------------------------------------------------
# Options 1 and 2 (Tara). Each test is SKIPPED until its function is written,
# then it automatically starts checking that the output follows the contract.
# ---------------------------------------------------------------------------
def _storm_list_or_skip(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet (Tara)")


def _check_storm_list(table):
    missing = [c for c in weather_input.STORM_LIST_COLUMNS if c not in table.columns]
    assert not missing, f"storm list is missing columns {missing}"
    assert not table["storm_id"].duplicated().any(), "storm_id values must be unique"
    assert len(table) > 0, "storm list is empty"


def test_historical_storms_follow_the_contract():
    table = _storm_list_or_skip(weather_input.list_historical_storms)
    _check_storm_list(table)
    for storm_id in table["storm_id"]:
        ice_mm, wind_ms = weather_input.historical_storm(storm_id)
        assert ice_mm >= 0 and wind_ms >= 0


def test_csv_storms_follow_the_contract():
    csv = weather_input.__file__.replace("weather_input.py", "era5_toronto_last_5_years.csv")
    table = _storm_list_or_skip(weather_input.list_storms_from_csv, csv)
    _check_storm_list(table)
    storm_id = table["storm_id"].iloc[0]
    ice_mm, wind_ms = weather_input.storm_from_csv(csv, storm_id)
    assert ice_mm >= 0 and wind_ms >= 0
