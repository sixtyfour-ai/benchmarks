"""
Shared evaluation utilities: lead loading, GPT-4.1-mini judging, result tracking.

All eval scripts import from here. Not meant to be run directly.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
load_dotenv(Path(__file__).parent.parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

# The judge call is retried this many times before a person's fields are degraded to
# substring containment. Degraded fields are tagged and counted, never silently mixed
# in with real verdicts.
JUDGE_MAX_ATTEMPTS = 3
# Fraction of degraded fields above which a run is flagged as not trustworthy.
JUDGE_DEGRADED_WARN_PCT = 1.0

JUDGE_PROMPT = """You are an eval judge comparing enrichment results against verified ground truth.
For each field, decide: CORRECT or WRONG. No partial credit.

FORMAT is IRRELEVANT — judge whether the same INFORMATION is present:
- "$10M" vs "10 million" -> CORRECT
- "Class of 2020" vs "2020" -> CORRECT
- Greek letters vs English for same fraternity -> CORRECT
- "Walnut Creek Dentistry" vs "Walnut Creek Dental" -> CORRECT (same entity)

Expected may list multiple accepted variants separated by " | " — matching ANY variant is CORRECT.

CORRECT: Core factual information matches. Format/wording differences don't matter.
WRONG: Missing key facts, factually incorrect, or empty/irrelevant.

Return JSON: {field_name: {"match": "correct"|"wrong", "reason": "brief explanation"}}
Only JSON, no markdown."""


def load_people(n: int | None = None) -> list[dict]:
    """Load the benchmark dataset: a list of people, each with a `fields` list.

    Reads data/people_data.json by default, or the path in $PEOPLE_DATA if set. Accepts
    either a bare list or a {"meta": ..., "people": [...]} payload, so the distributed
    dataset file can be dropped in unchanged.
    """
    path = Path(os.environ.get("PEOPLE_DATA") or (DATA_DIR / "people_data.json"))
    raw = path.read_bytes()

    # results/*.json publishes a dataset_sha256. Verify against it when the caller supplies
    # one, so a run can't silently be scored against a different dataset than it reports.
    expected_sha = os.environ.get("PEOPLE_DATA_SHA256")
    if expected_sha:
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"dataset hash mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )

    data = json.loads(raw)
    if isinstance(data, dict):
        data = data["people"]
    return data[:n] if n else data


# Serialization fragments some providers leak into field values when their
# structured-output JSON is truncated or malformed (e.g. Exa deep mode).
_STRUCT_JUNK = re.compile(r"top_results|citations\s*:\s*\[|confidence\s*:\s*[\[{]")


def clean_answer(val) -> str:
    """Coerce a raw model field value into a clean answer string.

    Returns "" (i.e. 'no answer' -> scored as missing, never wrong) for nulls,
    booleans, and malformed/structural fragments. Real scalar answers — including
    numbers and JSON-serialized lists/objects — are preserved.
    """
    if val is None or isinstance(val, bool):
        return ""
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (dict, list)):
        try:
            val = json.dumps(val, ensure_ascii=False)
        except (TypeError, ValueError):
            val = str(val)
    s = str(val).strip()
    if not s or s in ('""', "''", '""""', '"', "'"):
        return ""
    if _STRUCT_JUNK.search(s):
        return ""
    # pure JSON-structural punctuation (no letters/digits/other content)
    if re.fullmatch(r"""[\s{}\[\]"';:,.\-]*""", s):
        return ""
    return s


def clean_struct(output: dict, fields: list[dict]) -> dict:
    """Filter a raw output dict to exactly the requested fields, each cleaned.

    Drops leaked/unexpected keys and normalizes every value via clean_answer,
    so downstream judging and stored results never see serialization garbage.
    """
    if not isinstance(output, dict):
        return {}
    return {f["fieldname"]: clean_answer(output.get(f["fieldname"])) for f in fields}


_RETRY_STATUS = {429, 500, 502, 503, 504}


async def post_with_retry(client, url, *, max_retries: int = 6, **kwargs):
    """POST with exponential backoff on rate-limit / transient 5xx responses.

    Shared by the provider runners so each is robust out of the box. Retries on
    429/500/502/503/504 and on transport/timeout errors; raises on other 4xx.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, **kwargs)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_exc = e
            await asyncio.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code in _RETRY_STATUS:
            await asyncio.sleep(min(2 ** attempt, 30))
            continue
        resp.raise_for_status()
        return resp
    if last_exc:
        raise last_exc
    resp.raise_for_status()
    return resp


async def judge_fields(
    oai: AsyncOpenAI,
    sem: asyncio.Semaphore,
    person: str,
    actual: dict,
    fields: list[dict],
) -> dict:
    # ground truth: primary answer plus any accepted variants (accept_also)
    expected = {}
    for f in fields:
        variants = [str(v).strip() for v in [f.get("answer", "")] + (f.get("accept_also") or []) if v and str(v).strip()]
        expected[f["fieldname"]] = " | ".join(variants)
    verdicts = {}
    to_judge = {}

    for field, exp_str in expected.items():
        if not exp_str:
            continue
        act_str = clean_answer(actual.get(field))
        if not act_str:
            verdicts[field] = {"match": "missing", "reason": "actual is empty/null"}
        else:
            to_judge[field] = (act_str, exp_str)

    if not to_judge:
        return verdicts

    lines = []
    for field, (act, exp) in to_judge.items():
        lines.append(f"Field: {field}\n  Expected: {exp}\n  Actual: {act}")
    user_msg = f"Person: {person}\n\n" + "\n\n".join(lines)

    async with sem:
        last_exc: Exception | None = None
        llm_verdicts = None
        for attempt in range(JUDGE_MAX_ATTEMPTS):
            try:
                resp = await oai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=2048,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError(f"judge returned {type(parsed).__name__}, expected object")
                llm_verdicts = parsed
                break
            except Exception as e:
                last_exc = e
                if attempt + 1 < JUDGE_MAX_ATTEMPTS:
                    await asyncio.sleep(min(2 ** attempt, 8))

        if llm_verdicts is None:
            # Every attempt failed. Degrade to substring containment, but TAG every
            # field so the rate is countable downstream — see _summary_dict()["judge_health"].
            llm_verdicts = {}
            for field, (act, exp) in to_judge.items():
                if any(v.strip() and (v.strip().lower() in act.lower() or act.lower() in v.strip().lower())
                       for v in exp.split(" | ")):
                    llm_verdicts[field] = {"match": "correct", "reason": "fallback: substring",
                                           "degraded": "substring_fallback"}
                else:
                    llm_verdicts[field] = {"match": "wrong", "reason": f"judge error: {last_exc}",
                                           "degraded": "judge_error"}

    for field in to_judge:
        v = llm_verdicts.get(field)
        if v and isinstance(v, dict):
            raw_match = str(v.get("match", "wrong")).lower()
            match = raw_match if raw_match in ("correct", "wrong") else "wrong"
            verdicts[field] = {"match": match, "reason": v.get("reason", "")}
            if v.get("degraded"):
                verdicts[field]["degraded"] = v["degraded"]
            elif raw_match not in ("correct", "wrong"):
                # judge answered, but not with a verdict we recognise
                verdicts[field]["degraded"] = "unparsed_match"
        else:
            verdicts[field] = {"match": "wrong", "reason": "no verdict returned",
                               "degraded": "no_verdict"}

    return verdicts


class EvalRunner:
    """Orchestrates eval runs with inline GPT-4.1-mini judging and incremental saves."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=3600.0)
        self.judge_sem = asyncio.Semaphore(20)
        self.results: list[dict] = []
        self.t_start = time.time()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.out_path = RUNS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M}_judged.json"

    async def record(self, item: dict, output: dict, elapsed: float, metadata: dict | None = None) -> dict:
        verdicts = await judge_fields(self.oai, self.judge_sem, item["person_info"], output, item["fields"])
        c = sum(1 for v in verdicts.values() if v["match"] == "correct")
        w = sum(1 for v in verdicts.values() if v["match"] == "wrong")
        m = sum(1 for v in verdicts.values() if v["match"] == "missing")

        # Fields the judge did not actually adjudicate (see judge_fields). Counted so a
        # run can never report a score without also reporting how much of it was guessed.
        degraded: dict[str, int] = {}
        for v in verdicts.values():
            if v.get("degraded"):
                degraded[v["degraded"]] = degraded.get(v["degraded"], 0) + 1

        # per-bucket tallies when the dataset tags fields with a use-case bucket
        field_bucket = {f["fieldname"]: f["bucket"] for f in item["fields"] if f.get("bucket")}
        buckets: dict[str, dict] = {}
        for fname, v in verdicts.items():
            bk = field_bucket.get(fname)
            if not bk:
                continue
            b = buckets.setdefault(bk, {"correct": 0, "wrong": 0, "missing": 0})
            b[v["match"]] += 1

        label = (item.get("name") or item["person_info"])[:30]
        warn = f"  !! JUDGE DEGRADED {sum(degraded.values())} field(s): {degraded}" if degraded else ""
        print(f"  {label:30s} C={c} W={w} M={m} [{elapsed:.0f}s]{warn}", flush=True)

        result = {
            "person": item["person_info"],
            "name": item.get("name", ""),
            "elapsed": round(elapsed, 1),
            "correct": c, "wrong": w, "missing": m,
            **({"degraded": degraded} if degraded else {}),
            **({"buckets": buckets} if buckets else {}),
            "verdicts": verdicts,
            "output": output,
            **(metadata or {}),
        }
        self.results.append(result)
        self._save()
        return result

    async def record_error(self, item: dict, elapsed: float, error: Exception) -> dict:
        label = (item.get("name") or item["person_info"])[:30]
        print(f"  {label:30s} ERROR [{elapsed:.0f}s]: {str(error)[:100]}", flush=True)

        result = {
            "person": item["person_info"],
            "name": item.get("name", ""),
            "elapsed": round(elapsed, 1),
            "error": str(error),
            "correct": 0, "wrong": 0, "missing": 0,
            "verdicts": {}, "output": {},
        }
        self.results.append(result)
        self._save()
        return result

    def _save(self):
        self.out_path.write_text(json.dumps({
            "name": self.name,
            "config": self.config,
            "total_elapsed_s": round(time.time() - self.t_start, 1),
            "summary": self._summary_dict(),
            "results": self.results,
        }, indent=2, default=str))

    @staticmethod
    def _metrics(c: int, w: int, m: int) -> dict | None:
        n = c + w + m
        if not n:
            return None
        return {"n": n, "accuracy": round(c / n * 100, 1),
                "weighted": round((c - w) / n * 100, 1),
                "precision": round(c / (c + w) * 100, 1) if (c + w) else 0.0}

    def _summary_dict(self) -> dict:
        """Complete scoring for the run: raw counts, per-bucket metrics, and the overall
        (equal-weight average of the four bucket scores, 25% per use case)."""
        ok = [r for r in self.results if "error" not in r]
        c = sum(r["correct"] for r in ok)
        w = sum(r["wrong"] for r in ok)
        m = sum(r["missing"] for r in ok)
        t = c + w + m
        degraded: dict[str, int] = {}
        for r in ok:
            for k, v in (r.get("degraded") or {}).items():
                degraded[k] = degraded.get(k, 0) + v
        n_degraded = sum(degraded.values())
        out = {
            "correct": c, "wrong": w, "missing": m, "total_fields": t,
            "accuracy": round(c / t * 100, 1) if t else 0,
            "completed": len(ok), "errors": len(self.results) - len(ok),
            "judge_health": {
                "degraded_fields": n_degraded,
                "degraded_pct": round(n_degraded / t * 100, 2) if t else 0.0,
                "breakdown": degraded,
                "trustworthy": n_degraded / t * 100 <= JUDGE_DEGRADED_WARN_PCT if t else True,
            },
        }
        buckets: dict[str, dict] = {}
        for r in ok:
            for bk, bc in (r.get("buckets") or {}).items():
                b = buckets.setdefault(bk, {"correct": 0, "wrong": 0, "missing": 0})
                for k in b:
                    b[k] += bc.get(k, 0)
        if buckets:
            out["buckets"] = buckets
            per = {bk: mt for bk, b in buckets.items()
                   if (mt := self._metrics(b["correct"], b["wrong"], b["missing"]))}
            if per:
                out["scores"] = {
                    "buckets": per,
                    "overall": {k: round(sum(mt[k] for mt in per.values()) / len(per), 1)
                                for k in ("accuracy", "weighted", "precision")},
                }
        return out

    def summary(self):
        s = self._summary_dict()
        ok = [r for r in self.results if "error" not in r]
        lats = sorted(r["elapsed"] for r in ok if r.get("elapsed"))

        print(f"\n{'='*60}", flush=True)
        print(f"  {self.name} — {s['completed']}/{len(self.results)} completed", flush=True)
        if s["total_fields"]:
            print(f"  Accuracy: {s['correct']}/{s['total_fields']} = {s['accuracy']}%  (C={s['correct']} W={s['wrong']} M={s['missing']})", flush=True)
        sc = s.get("scores")
        if sc:
            for bk, mt in sc["buckets"].items():
                print(f"    {bk:22s} n={mt['n']:4d}  acc={mt['accuracy']:.1f}%  wtd={mt['weighted']:+.1f}%  prec={mt['precision']:.1f}%", flush=True)
            o = sc["overall"]
            print(f"    {'OVERALL (25%/bucket)':22s}        acc={o['accuracy']:.1f}%  wtd={o['weighted']:+.1f}%  prec={o['precision']:.1f}%", flush=True)
        jh = s["judge_health"]
        if jh["degraded_fields"]:
            flag = "OK" if jh["trustworthy"] else "NOT TRUSTWORTHY"
            print(f"  Judge health: {jh['degraded_fields']}/{s['total_fields']} fields "
                  f"({jh['degraded_pct']}%) not adjudicated by the judge -> {flag}", flush=True)
            print(f"    breakdown: {jh['breakdown']}", flush=True)
            if not jh["trustworthy"]:
                print(f"    Above the {JUDGE_DEGRADED_WARN_PCT}% threshold. Do not publish this run.", flush=True)
        else:
            print("  Judge health: all fields adjudicated by the judge (0 degraded)", flush=True)
        if lats:
            print(f"  Median latency: {lats[len(lats)//2]:.0f}s", flush=True)
        print(f"  Total time: {time.time() - self.t_start:.0f}s", flush=True)
        print(f"  Saved: {self.out_path}", flush=True)
        print(f"{'='*60}", flush=True)
