"""Degraded judge verdicts stay tagged and countable."""
import asyncio
import json

import pytest

import judge as J


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeOpenAI:
    """Minimal stand-in for AsyncOpenAI. `script` is a list of str (returned as content)
    or Exception (raised), consumed one entry per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.kwargs = []
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                outer.kwargs.append(kwargs)
                step = outer.script.pop(0) if outer.script else outer.script
                if isinstance(step, Exception):
                    raise step
                return FakeResponse(step)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


FIELDS = [
    {"fieldname": "school", "description": "University attended", "answer": "Stanford University"},
    {"fieldname": "handle", "description": "Early Twitter handle", "answer": "herossgraphics"},
]


async def _run(fake, fields=FIELDS, actual=None):
    actual = actual if actual is not None else {"school": "Stanford", "handle": "herossgraphics"}
    return await J.judge_fields(fake, asyncio.Semaphore(1), "Test Person", actual, fields)


async def test_healthy_judge_leaves_no_degraded_tag():
    fake = FakeOpenAI([json.dumps({
        "school": {"match": "correct", "reason": "same school"},
        "handle": {"match": "correct", "reason": "exact"},
    })])
    verdicts = await _run(fake)
    assert all(v["match"] == "correct" for v in verdicts.values())
    assert all("degraded" not in v for v in verdicts.values()), \
        "a successful judge call must not tag anything as degraded"


async def test_judge_call_is_retried_before_degrading():
    good = json.dumps({"school": {"match": "correct", "reason": "ok"},
                       "handle": {"match": "correct", "reason": "ok"}})
    fake = FakeOpenAI([TimeoutError("transient"), good])
    verdicts = await _run(fake)
    assert len(fake.calls) == 2, "a transient failure must be retried, not degraded immediately"
    assert all("degraded" not in v for v in verdicts.values())


async def test_exhausted_retries_tag_every_field():
    fake = FakeOpenAI([TimeoutError("down")] * J.JUDGE_MAX_ATTEMPTS)
    verdicts = await _run(fake)
    assert len(fake.calls) == J.JUDGE_MAX_ATTEMPTS
    assert all(v.get("degraded") for v in verdicts.values()), \
        "every field scored without the judge must carry a degraded tag"
    assert {v["degraded"] for v in verdicts.values()} <= {"substring_fallback", "judge_error"}


async def test_json_object_response_format_is_requested():
    """Without response_format the only guarantee of JSON is a sentence in the prompt;
    a stray prose preamble then routes a whole person through the fallback."""
    fake = FakeOpenAI([json.dumps({"school": {"match": "correct"}, "handle": {"match": "correct"}})])
    await _run(fake)
    assert fake.kwargs[0].get("response_format") == {"type": "json_object"}


async def test_non_json_reply_is_retried_then_tagged():
    fake = FakeOpenAI(["Sure! Here are the verdicts:"] * J.JUDGE_MAX_ATTEMPTS)
    verdicts = await _run(fake)
    assert len(fake.calls) == J.JUDGE_MAX_ATTEMPTS
    assert all(v.get("degraded") for v in verdicts.values())


async def test_substring_fallback_accepts_a_wrong_answer():
    """The behaviour that makes counting necessary. Ground truth '9th'; the provider says
    '19th'. Containment says CORRECT. Real-shaped answer, wrong fact."""
    fields = [{"fieldname": "rank", "description": "Exam ranking", "answer": "9th"}]
    fake = FakeOpenAI([TimeoutError("down")] * J.JUDGE_MAX_ATTEMPTS)
    verdicts = await J.judge_fields(
        fake, asyncio.Semaphore(1), "Test Person",
        {"rank": "ranked 19th among ISC high scorers"}, fields,
    )
    assert verdicts["rank"]["match"] == "correct"
    assert verdicts["rank"]["degraded"] == "substring_fallback", \
        "a containment guess must never be indistinguishable from a judged verdict"


async def test_substring_fallback_accepts_an_inverted_negation():
    """Ground truth 'herossgraphics'; the provider explicitly says it is NOT that handle.
    Containment finds the string and scores CORRECT."""
    fields = [{"fieldname": "handle", "description": "Early handle", "answer": "herossgraphics"}]
    fake = FakeOpenAI([TimeoutError("down")] * J.JUDGE_MAX_ATTEMPTS)
    verdicts = await J.judge_fields(
        fake, asyncio.Semaphore(1), "Test Person",
        {"handle": "Not found publicly; his current handle is @saarthshah, not herossgraphics"},
        fields,
    )
    assert verdicts["handle"]["match"] == "correct"
    assert verdicts["handle"]["degraded"] == "substring_fallback"


async def test_unrecognised_match_value_is_tagged_not_silently_wrong():
    fake = FakeOpenAI([json.dumps({
        "school": {"match": "partial", "reason": "close enough"},
        "handle": {"match": "correct", "reason": "ok"},
    })])
    verdicts = await _run(fake)
    assert verdicts["school"]["match"] == "wrong"
    assert verdicts["school"]["degraded"] == "unparsed_match"
    assert "degraded" not in verdicts["handle"]


async def test_missing_field_in_judge_reply_is_tagged():
    fake = FakeOpenAI([json.dumps({"school": {"match": "correct", "reason": "ok"}})])
    verdicts = await _run(fake)
    assert verdicts["handle"]["degraded"] == "no_verdict"


async def test_empty_actual_is_missing_and_never_degraded():
    fake = FakeOpenAI([json.dumps({"school": {"match": "correct", "reason": "ok"}})])
    verdicts = await _run(fake, actual={"school": "Stanford", "handle": None})
    assert verdicts["handle"]["match"] == "missing"
    assert "degraded" not in verdicts["handle"], "an honest abstention is not a judge failure"
