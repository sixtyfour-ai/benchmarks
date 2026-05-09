"""
Shared evaluation utilities: lead loading, GPT-4.1-mini judging, result tracking.

All eval scripts import from here. Not meant to be run directly.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
load_dotenv(Path(__file__).parent.parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

JUDGE_PROMPT = """You are an eval judge comparing enrichment results against verified ground truth.
For each field, decide: CORRECT or WRONG. No partial credit.

FORMAT is IRRELEVANT — judge whether the same INFORMATION is present:
- "$10M" vs "10 million" -> CORRECT
- "Class of 2020" vs "2020" -> CORRECT
- Greek letters vs English for same fraternity -> CORRECT
- "Walnut Creek Dentistry" vs "Walnut Creek Dental" -> CORRECT (same entity)

CORRECT: Core factual information matches. Format/wording differences don't matter.
WRONG: Missing key facts, factually incorrect, or empty/irrelevant.

Return JSON: {field_name: {"match": "correct"|"wrong", "reason": "brief explanation"}}
Only JSON, no markdown."""


def load_people(n: int | None = None) -> list[dict]:
    data = json.loads((DATA_DIR / "people_data.json").read_text())
    return data[:n] if n else data


async def judge_fields(
    oai: AsyncOpenAI,
    sem: asyncio.Semaphore,
    person: str,
    actual: dict,
    fields: list[dict],
) -> dict:
    expected = {f["fieldname"]: f["answer"] for f in fields}
    verdicts = {}
    to_judge = {}

    for field, exp_val in expected.items():
        exp_str = (exp_val or "").strip()
        if not exp_str:
            continue
        act_str = str(actual.get(field) or "").strip()
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
        try:
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
        except Exception as e:
            llm_verdicts = {}
            for field, (act, exp) in to_judge.items():
                if exp.lower() in act.lower() or act.lower() in exp.lower():
                    llm_verdicts[field] = {"match": "correct", "reason": "fallback: substring"}
                else:
                    llm_verdicts[field] = {"match": "wrong", "reason": f"judge error: {e}"}

    for field in to_judge:
        v = llm_verdicts.get(field)
        if v and isinstance(v, dict):
            match = v.get("match", "wrong").lower()
            verdicts[field] = {"match": match if match in ("correct", "wrong") else "wrong", "reason": v.get("reason", "")}
        else:
            verdicts[field] = {"match": "wrong", "reason": "no verdict returned"}

    return verdicts


class EvalRunner:
    """Orchestrates eval runs with inline GPT-4o-mini judging and incremental saves."""

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

        label = (item.get("name") or item["person_info"])[:30]
        print(f"  {label:30s} C={c} W={w} M={m} [{elapsed:.0f}s]", flush=True)

        result = {
            "person": item["person_info"],
            "name": item.get("name", ""),
            "elapsed": round(elapsed, 1),
            "correct": c, "wrong": w, "missing": m,
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
            "config": self.config,
            "total_elapsed_s": round(time.time() - self.t_start, 1),
            "summary": self._summary_dict(),
            "results": self.results,
        }, indent=2, default=str))

    def _summary_dict(self) -> dict:
        ok = [r for r in self.results if "error" not in r]
        c = sum(r["correct"] for r in ok)
        w = sum(r["wrong"] for r in ok)
        m = sum(r["missing"] for r in ok)
        t = c + w + m
        return {
            "correct": c, "wrong": w, "missing": m, "total_fields": t,
            "accuracy": round(c / t * 100, 1) if t else 0,
            "completed": len(ok), "errors": len(self.results) - len(ok),
        }

    def summary(self):
        s = self._summary_dict()
        ok = [r for r in self.results if "error" not in r]
        lats = sorted(r["elapsed"] for r in ok if r.get("elapsed"))

        print(f"\n{'='*60}", flush=True)
        print(f"  {self.name} — {s['completed']}/{len(self.results)} completed", flush=True)
        if s["total_fields"]:
            print(f"  Accuracy: {s['correct']}/{s['total_fields']} = {s['accuracy']}%  (C={s['correct']} W={s['wrong']} M={s['missing']})", flush=True)
        if lats:
            print(f"  Median latency: {lats[len(lats)//2]:.0f}s", flush=True)
        print(f"  Total time: {time.time() - self.t_start:.0f}s", flush=True)
        print(f"  Saved: {self.out_path}", flush=True)
        print(f"{'='*60}", flush=True)
