"""
Two scoring definitions ship in this repo and both are legitimate; they are pinned here so
the distinction stays deliberate.

  pooled          _metrics(): (correct - wrong) / total_fields  -- README.md, and the
                  `weighted_formula` recorded in results/*.json
  macro (25%/bkt) _summary_dict()["scores"]["overall"]: equal-weight mean of the four
                  bucket scores -- what leaderboard.py prints

They disagree whenever buckets differ in size, which is the normal case.
"""
import hashlib
import json
from pathlib import Path

import pytest

import judge as J

RESULTS = Path(__file__).resolve().parent.parent / "results" / "sixtyfour_benchmark_results.json"


def test_pooled_weighted_matches_the_published_formula():
    assert J.EvalRunner._metrics(369, 78, 67)["weighted"] == 56.6      # Sixtyfour High
    assert J.EvalRunner._metrics(229, 68, 217)["weighted"] == 31.3     # Parallel Ultra 8x


def test_published_rows_reproduce_from_their_own_counts():
    rows = json.loads(RESULTS.read_text())["results"]
    for r in rows:
        m = J.EvalRunner._metrics(r["correct"], r["wrong"], r["missing"])
        assert m["weighted"] == r["weighted_accuracy"], r["provider"]
        assert m["accuracy"] == r["accuracy"], r["provider"]
        assert m["precision"] == r["precision"], r["provider"]


def test_pooled_and_macro_disagree_on_unequal_buckets():
    """Not a bug — a documented choice. The test exists so nobody 'fixes' one into the
    other by accident, and so the gap is visible in CI output."""
    r = J.EvalRunner.__new__(J.EvalRunner)
    r.results = [
        {"person": "a", "correct": 90, "wrong": 10, "missing": 0,
         "buckets": {"sales": {"correct": 90, "wrong": 10, "missing": 0}}},
        {"person": "b", "correct": 0, "wrong": 10, "missing": 0,
         "buckets": {"recruiting": {"correct": 0, "wrong": 10, "missing": 0}}},
    ]
    r.name, r.config, r.t_start = "t", {}, 0.0
    s = r._summary_dict()
    pooled = J.EvalRunner._metrics(90, 20, 0)["weighted"]     # (90-20)/110 = +63.6
    macro = s["scores"]["overall"]["weighted"]                # mean(+80.0, -100.0) = -10.0
    assert pooled == 63.6
    assert macro == -10.0
    assert pooled != macro


def test_load_people_rejects_a_dataset_whose_hash_does_not_match(tmp_path, monkeypatch):
    f = tmp_path / "people.json"
    f.write_text(json.dumps([{"person_info": "x", "fields": []}]))
    monkeypatch.setenv("PEOPLE_DATA", str(f))
    monkeypatch.setenv("PEOPLE_DATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        J.load_people()


def test_load_people_accepts_the_matching_hash(tmp_path, monkeypatch):
    f = tmp_path / "people.json"
    f.write_bytes(json.dumps([{"person_info": "x", "fields": []}]).encode())
    monkeypatch.setenv("PEOPLE_DATA", str(f))
    monkeypatch.setenv("PEOPLE_DATA_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    assert J.load_people() == [{"person_info": "x", "fields": []}]


@pytest.mark.parametrize("val,expected", [
    (None, ""), (True, ""), (False, ""), ('""', ""), ("   ", ""), ("{}", ""),
    (0, "0"), (3.5, "3.5"), ("Stanford", "Stanford"),
])
def test_clean_answer_contract(val, expected):
    assert J.clean_answer(val) == expected
