"""Run summaries always carry judge health alongside the score."""
import json

import pytest

import judge as J


def _runner(results):
    r = J.EvalRunner.__new__(J.EvalRunner)   # skip __init__: no OpenAI client, no disk
    r.results = results
    r.name = "test"
    r.config = {}
    r.t_start = 0.0
    return r


def _person(c, w, m, degraded=None, error=None):
    out = {"person": "p", "correct": c, "wrong": w, "missing": m, "verdicts": {}}
    if degraded:
        out["degraded"] = degraded
    if error:
        out["error"] = error
    return out


def test_clean_run_reports_zero_degraded_and_trustworthy():
    s = _runner([_person(8, 1, 1)])._summary_dict()
    assert s["judge_health"] == {
        "degraded_fields": 0, "degraded_pct": 0.0, "breakdown": {}, "trustworthy": True,
    }


def test_degraded_fields_are_summed_across_people_with_a_breakdown():
    s = _runner([
        _person(8, 1, 1, degraded={"substring_fallback": 2}),
        _person(7, 2, 1, degraded={"substring_fallback": 1, "judge_error": 3}),
    ])._summary_dict()
    jh = s["judge_health"]
    assert jh["degraded_fields"] == 6
    assert jh["breakdown"] == {"substring_fallback": 3, "judge_error": 3}
    assert jh["degraded_pct"] == round(6 / 20 * 100, 2) == 30.0


def test_a_run_over_the_threshold_is_flagged_not_trustworthy():
    # 2 degraded of 100 fields = 2%, above JUDGE_DEGRADED_WARN_PCT (1.0)
    people = [_person(9, 1, 0) for _ in range(10)]
    people[0]["degraded"] = {"substring_fallback": 2}
    s = _runner(people)._summary_dict()
    assert s["judge_health"]["degraded_pct"] == 2.0
    assert s["judge_health"]["trustworthy"] is False


def test_threshold_boundary_is_inclusive():
    people = [_person(10, 0, 0) for _ in range(10)]
    people[0]["degraded"] = {"judge_error": 1}      # 1/100 = exactly 1.0%
    s = _runner(people)._summary_dict()
    assert s["judge_health"]["degraded_pct"] == J.JUDGE_DEGRADED_WARN_PCT
    assert s["judge_health"]["trustworthy"] is True


def test_errored_people_contribute_nothing_to_the_denominator():
    """record_error() stores 0/0/0, so an errored person is already score-neutral. Pinned
    here so the disclosed per-row total_fields (e.g. Gemini 505/140) stays explainable."""
    s = _runner([_person(8, 1, 1), _person(0, 0, 0, error="boom")])._summary_dict()
    assert s["total_fields"] == 10
    assert s["completed"] == 1 and s["errors"] == 1


def test_judge_health_survives_json_round_trip():
    s = _runner([_person(8, 1, 1, degraded={"judge_error": 1})])._summary_dict()
    assert json.loads(json.dumps(s))["judge_health"]["degraded_fields"] == 1
