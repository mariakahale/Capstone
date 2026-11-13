"""
Tests for pipeline/run_fragility.py.

  * small tests: one piece at a time (which curves, file names, choosing the source)
  * regression tests: the whole chain, compared with the saved output files, so a change
    that accidentally alters results is caught. Numbers are compared with a tiny tolerance
    because different computers round the last 1-2 digits differently.

Run all tests from the repo folder:   python3 -m pytest tests
"""
from pathlib import Path

import pandas as pd
import pytest

import run_fragility as rf

FRAGILITY_OUTPUTS = Path(rf.OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Small tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ice_mm, expected", [(0, False), (0.1, True), (17.3, True)])
def test_ice_on_the_wire_means_ice_curves(ice_mm, expected):
    assert rf.is_ice_storm(ice_mm) is expected


@pytest.mark.parametrize("ice_mm, wind_ms, tag", [
    (14.7, 12.3, "ice14.7mm_wind12.3ms"),
    (20, 31.92, "ice20mm_wind31.92ms"),
    (0, 23.8, "wind23.8ms"),
    (0, 50, "wind50ms"),
])
def test_output_file_names(ice_mm, wind_ms, tag):
    assert rf.scenario_tag(ice_mm, wind_ms) == tag


def test_get_weather_user_source_returns_the_settings(monkeypatch):
    # monkeypatch = change a setting for this test only; it is put back afterwards
    monkeypatch.setattr(rf, "WEATHER_SOURCE", "user")
    monkeypatch.setattr(rf, "USER_ICE_MM", 17.3)
    monkeypatch.setattr(rf, "USER_WIND_MS", 14.9)
    assert rf.get_weather() == (17.3, 14.9)


def test_get_weather_unknown_source_is_an_error(monkeypatch):
    monkeypatch.setattr(rf, "WEATHER_SOURCE", "forecast")
    with pytest.raises(ValueError, match="WEATHER_SOURCE"):
        rf.get_weather()


def test_run_fragility_rejects_negative_weather():
    with pytest.raises(ValueError):
        rf.run_fragility(-1, 20)


def test_toronto_default_concurrent_wind():
    assert rf.toronto_concurrent_wind() == pytest.approx(31.92)


# ---------------------------------------------------------------------------
# Regression tests: whole chain vs the saved output files
# ---------------------------------------------------------------------------
def _compare_with_saved(frame, saved_path):
    saved = pd.read_csv(saved_path)
    assert list(frame.columns) == list(saved.columns)
    assert len(frame) == len(saved)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), saved,
                                  check_dtype=False, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("ice_mm, wind_ms", [
    (0, 50),          # wind storm (Fig. 4 curves), has pole failures
    (14.7, 31.92),    # ice storm (Fig. 7 curves), has pole failures
])
def test_results_match_saved_outputs(ice_mm, wind_ms):
    pole_frag, span_frag = rf.run_fragility(ice_mm, wind_ms)
    tag = rf.scenario_tag(ice_mm, wind_ms)
    feeder = f"{rf.FEEDER_NAME}_{rf.AREA_TYPE}"
    _compare_with_saved(pole_frag, FRAGILITY_OUTPUTS / f"pole_fragility_{feeder}_{tag}.csv")
    _compare_with_saved(span_frag, FRAGILITY_OUTPUTS / f"span_fragility_{feeder}_{tag}.csv")


def test_wind_storm_breaks_no_wires():
    # Ma et al. (2020): wires don't fail from wind alone
    _, span_frag = rf.run_fragility(0, 50)
    assert (span_frag["p_fail_wire"] == 0).all()


def test_probabilities_are_between_0_and_1():
    pole_frag, span_frag = rf.run_fragility(20, 31.92)
    for p in (pole_frag["p_fail"], span_frag["p_fail_wire"], span_frag["p_fail_span_combined"]):
        assert p.between(0, 1).all()


# ---------------------------------------------------------------------------
# main(): the whole script, as when you run  python3 pipeline/run_fragility.py
# Its output goes to a temporary folder (tmp_path) so the real output files are
# never touched.
# ---------------------------------------------------------------------------
def _run_main(monkeypatch, tmp_path, ice_mm, wind_ms):
    monkeypatch.setattr(rf, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(rf, "WEATHER_SOURCE", "user")
    monkeypatch.setattr(rf, "USER_ICE_MM", ice_mm)
    monkeypatch.setattr(rf, "USER_WIND_MS", wind_ms)
    rf.main()
    feeder = f"{rf.FEEDER_NAME}_{rf.AREA_TYPE}"
    tag = rf.scenario_tag(ice_mm, wind_ms)
    return (tmp_path / f"pole_fragility_{feeder}_{tag}.csv",
            tmp_path / f"span_fragility_{feeder}_{tag}.csv")


@pytest.mark.parametrize("ice_mm, wind_ms", [(0, 50), (14.7, 31.92)])
def test_main_writes_the_same_results_as_the_saved_files(monkeypatch, tmp_path, ice_mm, wind_ms):
    pole_path, span_path = _run_main(monkeypatch, tmp_path, ice_mm, wind_ms)
    assert pole_path.exists() and span_path.exists()
    _compare_with_saved(pd.read_csv(pole_path), FRAGILITY_OUTPUTS / pole_path.name)
    _compare_with_saved(pd.read_csv(span_path), FRAGILITY_OUTPUTS / span_path.name)


def test_main_new_storm_writes_files_and_prints_summary(monkeypatch, tmp_path, capsys):
    # a storm with no saved files yet (Dec 2013): main() must still run and save
    pole_path, span_path = _run_main(monkeypatch, tmp_path, 17.3, 14.9)
    assert pole_path.name.endswith("ice17.3mm_wind14.9ms.csv")
    assert len(pd.read_csv(pole_path)) > 0 and len(pd.read_csv(span_path)) > 0
    printed = capsys.readouterr().out          # capsys = what main() printed
    assert "scenario: ice17.3mm_wind14.9ms" in printed
    assert "expected failures" in printed


def test_main_stops_on_bad_weather_and_writes_nothing(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _run_main(monkeypatch, tmp_path, -5, 14.9)
    assert list(tmp_path.iterdir()) == []
