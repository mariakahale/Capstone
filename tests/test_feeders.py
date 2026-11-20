"""
Tests for pipeline/feeders.py: the feeder list and the polegen2 version check.
The regeneration tests use a temporary folder and a fake polegen2, so they are fast
and never touch the real pipeline/input_feedercsvfiles.

Run all tests from the repo folder:   python3 -m pytest tests
"""
import pytest

import feeders


def test_feeders_and_their_area_types():
    assert feeders.area_type_of("3777") == "suburban"
    assert feeders.area_type_of("1878") == "rural"
    with pytest.raises(ValueError):
        feeders.area_type_of("9999")


def test_file_names_carry_the_polegen_version():
    paths = feeders.feeder_file_paths("1878", "rural", version="v7")
    assert paths["poles"].name == "feeder_1878_poles_rural_v7.csv"
    assert paths["span_line_map"].parent.name == "mapping"


def test_current_version_files_are_there_for_both_feeders():
    for f in feeders.FEEDERS:
        assert feeders.missing_feeder_files(f, feeders.area_type_of(f)) == []


def test_existing_files_are_used_without_regenerating(tmp_path):
    for p in feeders.feeder_file_paths("1878", "rural", "v1", tmp_path).values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    ran = []
    feeders.ensure_feeder_files("1878", version="v1", folder=tmp_path,
                                regenerate=lambda *a: ran.append(a))
    assert ran == []


def test_a_new_version_regenerates(tmp_path):
    # v1 files exist, but the code now says v2 -> polegen2 must run
    for p in feeders.feeder_file_paths("1878", "rural", "v1", tmp_path).values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    ran = []

    def fake_polegen(feeder, area_type):
        ran.append((feeder, area_type))
        for p in feeders.feeder_file_paths(feeder, area_type, "v2", tmp_path).values():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

    paths = feeders.ensure_feeder_files("1878", version="v2", folder=tmp_path,
                                        regenerate=fake_polegen)
    assert ran == [("1878", "rural")]
    assert paths["poles"].name.endswith("_v2.csv")


def test_regenerating_that_makes_nothing_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        feeders.ensure_feeder_files("3777", version="v1", folder=tmp_path,
                                    regenerate=lambda *a: None)
