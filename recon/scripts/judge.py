"""
Shared evaluation utilities: lead loading, GPT-4.1-mini judging, result tracking.

All eval scripts import from here. Not meant to be run directly.
"""

import asyncio
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


class JudgmentError(RuntimeError):
    """A judge failure, not a verdict about the researched answer."""

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
    data = json.loads(path.read_text())
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
    return await _request_with_retry(client, "POST", url, max_retries=max_retries, **kwargs)


async def get_with_retry(client, url, *, max_retries: int = 6, **kwargs):
    """GET with the same backoff as post_with_retry, for polling long-running jobs."""
    return await _request_with_retry(client, "GET", url, max_retries=max_retries, **kwargs)


async def _request_with_retry(client, method, url, *, max_retries: int, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.request(method, url, **kwargs)
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

    last_error = None
    for attempt in range(3):
        try:
            async with sem:
                resp = await oai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=2048,
                )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            llm_verdicts = json.loads(raw)
            if not isinstance(llm_verdicts, dict):
                raise ValueError("Judge returned a non-object")
            for field in to_judge:
                v = llm_verdicts.get(field)
                if (not isinstance(v, dict)
                        or not isinstance(v.get("match"), str)
                        or v["match"].lower() not in ("correct", "wrong")
                        or not isinstance(v.get("reason", ""), str)):
                    raise ValueError("Judge returned an incomplete or invalid verdict")
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    else:
        raise JudgmentError("Judge failed after 3 attempts") from last_error

    for field in to_judge:
        v = llm_verdicts[field]
        verdicts[field] = {"match": v["match"].lower(), "reason": v.get("reason", "")}

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
        runs_dir = Path(os.environ.get("RECON_RUNS_DIR") or RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = runs_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}_judged.json"

    async def record(self, item: dict, output: dict, elapsed: float, metadata: dict | None = None) -> dict:
        # Save successful research before making another network request. A judge
        # outage or interrupted process must not force paid research to repeat.
        result = {
            **(metadata or {}),
            "person": item["person_info"], "name": item.get("name", ""),
            "elapsed": round(elapsed, 1), "output": output,
            "correct": 0, "wrong": 0, "missing": 0, "verdicts": {},
            "error": "Judgment pending", "error_type": "judgment_pending",
        }
        self.results.append(result)
        self._save()
        try:
            verdicts = await judge_fields(self.oai, self.judge_sem, item["person_info"], output, item["fields"])
        except JudgmentError as e:
            result.update(error=str(e), error_type="judgment_error")
            self._save()
            print(f"  JUDGMENT ERROR (research saved): {e}", flush=True)
            return result
        c = sum(1 for v in verdicts.values() if v["match"] == "correct")
        w = sum(1 for v in verdicts.values() if v["match"] == "wrong")
        m = sum(1 for v in verdicts.values() if v["match"] == "missing")

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
        print(f"  {label:30s} C={c} W={w} M={m} [{elapsed:.0f}s]", flush=True)

        result.clear()
        result.update({
            **(metadata or {}),
            "person": item["person_info"],
            "name": item.get("name", ""),
            "elapsed": round(elapsed, 1),
            "correct": c, "wrong": w, "missing": m,
            **({"buckets": buckets} if buckets else {}),
            "verdicts": verdicts,
            "output": output,
        })
        self._save()
        return result

    async def record_error(self, item: dict, elapsed: float, error: Exception, metadata: dict | None = None) -> dict:
        label = (item.get("name") or item["person_info"])[:30]
        print(f"  {label:30s} ERROR [{elapsed:.0f}s]: {str(error)[:100]}", flush=True)

        result = {
            **(metadata or {}),
            "person": item["person_info"],
            "name": item.get("name", ""),
            "elapsed": round(elapsed, 1),
            "error": str(error),
            "error_type": "provider_error",
            "correct": 0, "wrong": 0, "missing": 0,
            "verdicts": {}, "output": {},
        }
        self.results.append(result)
        self._save()
        return result

    def _save(self):
        temporary = self.out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "name": self.name,
            "config": self.config,
            "total_elapsed_s": round(time.time() - self.t_start, 1),
            "summary": self._summary_dict(),
            "results": self.results,
        }, indent=2, default=str))
        temporary.replace(self.out_path)

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
        out = {
            "correct": c, "wrong": w, "missing": m, "total_fields": t,
            "accuracy": round(c / t * 100, 1) if t else 0,
            "completed": len(ok), "errors": len(self.results) - len(ok),
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
        if lats:
            print(f"  Median latency: {lats[len(lats)//2]:.0f}s", flush=True)
        print(f"  Total time: {time.time() - self.t_start:.0f}s", flush=True)
        print(f"  Saved: {self.out_path}", flush=True)
        print(f"{'='*60}", flush=True)
