"""
Evaluate Parallel API on the RECON benchmark.

Submits individual task runs with checkpoint persistence for crash recovery.

Usage:
    python scripts/parallel.py
    python scripts/parallel.py --processor ultra2x --people 5
    python scripts/parallel.py --processor ultra8x --resume
    python scripts/parallel.py --processor ultra --concurrency 20

Requires: PARALLEL_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from judge import load_people, judge_fields, RUNS_DIR

PARALLEL_API_KEY = os.environ["PARALLEL_API_KEY"]
PARALLEL_BASE = "https://api.parallel.ai"
HEADERS = {"x-api-key": PARALLEL_API_KEY, "Content-Type": "application/json"}

COST_PER_1K = {
    "lite": 5, "base": 10, "core": 25, "core2x": 50,
    "pro": 100, "ultra": 300, "ultra2x": 600, "ultra4x": 1200, "ultra8x": 2400,
}


def build_schema(fields: list[dict]) -> dict:
    properties = {}
    required = []
    for f in fields:
        properties[f["fieldname"]] = {"type": ["string", "null"], "description": f["description"]}
        required.append(f["fieldname"])
    return {
        "type": "json",
        "json_schema": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"submitted": {}, "completed": {}}


def save_checkpoint(path: Path, ckpt: dict):
    path.write_text(json.dumps(ckpt, indent=2, default=str))


async def submit_one(client: httpx.AsyncClient, item: dict, processor: str) -> str:
    fields = ", ".join(f["fieldname"] for f in item["fields"])
    resp = await client.post(
        f"{PARALLEL_BASE}/v1/tasks/runs",
        json={
            "task_spec": {
                "output_schema": build_schema(item["fields"]),
                "instructions": (
                    "You are a research agent. Given a description of a person, find specific facts about them. "
                    "For each field, search thoroughly using the description as guidance. "
                    "Cross-reference multiple sources for accuracy. "
                    "Return an empty string for any field you cannot find."
                ),
            },
            "input": (
                f"Person: {item['person_info']}\n\n"
                f"Find the following fields: {fields}\n\n"
                f"For each field, search thoroughly using the field description as guidance. "
                f"If you cannot find a definitive answer, return an empty string for that field."
            ),
            "processor": processor,
        },
    )
    resp.raise_for_status()
    return resp.json()["run_id"]


async def poll_one(
    client: httpx.AsyncClient,
    oai: AsyncOpenAI,
    judge_sem: asyncio.Semaphore,
    poll_sem: asyncio.Semaphore,
    item: dict,
    run_id: str,
    t0: float,
    ckpt: dict,
    ckpt_path: Path,
) -> dict:
    label = (item.get("name") or item["person_info"])[:30]

    async with poll_sem:
        try:
            while True:
                await asyncio.sleep(15)
                resp = await client.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}")
                resp.raise_for_status()
                status = resp.json().get("status", "")
                if status in ("completed", "failed", "cancelled"):
                    break

            elapsed = time.time() - t0

            if status != "completed":
                print(f"  {label:30s} {status.upper()} [{elapsed:.0f}s]", flush=True)
                result = {"person": item["person_info"], "name": item.get("name", ""), "run_id": run_id,
                          "status": status, "elapsed": round(elapsed, 1), "error": status,
                          "correct": 0, "wrong": 0, "missing": 0, "verdicts": {}, "output": {}}
                ckpt["completed"][item["person_info"]] = result
                save_checkpoint(ckpt_path, ckpt)
                return result

            resp = await client.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}/result")
            resp.raise_for_status()
            output = resp.json().get("output", {})
            if isinstance(output, dict) and "content" in output:
                output = output["content"]
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except Exception:
                    output = {}
            if not isinstance(output, dict):
                output = {}

            verdicts = await judge_fields(oai, judge_sem, item["person_info"], output, item["fields"])
            c = sum(1 for v in verdicts.values() if v["match"] == "correct")
            w = sum(1 for v in verdicts.values() if v["match"] == "wrong")
            m = sum(1 for v in verdicts.values() if v["match"] == "missing")

            print(f"  {label:30s} C={c} W={w} M={m} [{elapsed:.0f}s]", flush=True)

            result = {"person": item["person_info"], "name": item.get("name", ""), "run_id": run_id,
                      "status": "completed", "elapsed": round(elapsed, 1),
                      "correct": c, "wrong": w, "missing": m, "verdicts": verdicts, "output": output}
            ckpt["completed"][item["person_info"]] = result
            save_checkpoint(ckpt_path, ckpt)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {label:30s} ERROR [{elapsed:.0f}s]: {str(e)[:100]}", flush=True)
            result = {"person": item["person_info"], "name": item.get("name", ""), "run_id": run_id,
                      "elapsed": round(elapsed, 1), "error": str(e),
                      "correct": 0, "wrong": 0, "missing": 0, "verdicts": {}, "output": {}}
            ckpt["completed"][item["person_info"]] = result
            save_checkpoint(ckpt_path, ckpt)
            return result


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processor", default="ultra",
                   choices=["lite", "base", "core", "core2x", "pro", "ultra", "ultra2x", "ultra4x", "ultra8x"])
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    people = load_people(args.people)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RUNS_DIR / f"parallel_{args.processor}_checkpoint.json"
    ckpt = load_checkpoint(ckpt_path) if args.resume else {"submitted": {}, "completed": {}}

    to_run = [p for p in people if p["person_info"] not in ckpt["completed"]]
    cpt = COST_PER_1K.get(args.processor, 0)

    if args.resume and ckpt["completed"]:
        print(f"Resuming: {len(ckpt['completed'])} done, {len(to_run)} remaining", flush=True)

    print(f"Running Parallel {args.processor} on {len(to_run)} people (of {len(people)}), concurrency={args.concurrency}", flush=True)
    print(f"Est cost: ${len(to_run) * cpt / 1000:.2f}", flush=True)

    oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_sem = asyncio.Semaphore(20)
    poll_sem = asyncio.Semaphore(args.concurrency)
    t_start = time.time()

    async with httpx.AsyncClient(timeout=36000.0, headers=HEADERS) as client:
        tasks = []
        for item in to_run:
            person = item["person_info"]
            if person in ckpt["submitted"] and person not in ckpt["completed"]:
                run_id = ckpt["submitted"][person]["run_id"]
                t0 = float(ckpt["submitted"][person].get("submitted_at", t_start))
                print(f"  Resuming poll: {person[:30]}", flush=True)
            else:
                run_id = await submit_one(client, item, args.processor)
                t0 = time.time()
                ckpt["submitted"][person] = {"run_id": run_id, "submitted_at": t0}
                save_checkpoint(ckpt_path, ckpt)
                print(f"  Submitted: {person[:30]} ({run_id})", flush=True)

            tasks.append(poll_one(client, oai, judge_sem, poll_sem, item, run_id, t0, ckpt, ckpt_path))

        await asyncio.gather(*tasks)

    all_results = []
    for entry in people:
        person = entry["person_info"]
        if person in ckpt["completed"]:
            all_results.append(ckpt["completed"][person])

    ok = [r for r in all_results if r.get("status") == "completed"]
    total_c = sum(r["correct"] for r in ok)
    total_w = sum(r["wrong"] for r in ok)
    total_m = sum(r["missing"] for r in ok)
    total_f = total_c + total_w + total_m
    lats = sorted(r["elapsed"] for r in ok if r.get("elapsed"))

    print(f"\n{'='*60}", flush=True)
    print(f"  Parallel {args.processor} — {len(ok)}/{len(people)} completed — {time.time() - t_start:.0f}s", flush=True)
    if total_f:
        print(f"  Accuracy: {total_c}/{total_f} = {total_c/total_f*100:.1f}%  (C={total_c} W={total_w} M={total_m})", flush=True)
    if lats:
        print(f"  Median latency: {lats[len(lats)//2]:.0f}s", flush=True)
    print(f"  Cost: ${len(ok) * cpt / 1000:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    out_path = RUNS_DIR / f"parallel_{args.processor}_{datetime.now():%Y%m%d_%H%M}_judged.json"
    out_path.write_text(json.dumps({
        "config": vars(args),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "summary": {"correct": total_c, "wrong": total_w, "missing": total_m, "total": total_f,
                     "accuracy": round(total_c / total_f * 100, 1) if total_f else 0},
        "results": all_results,
    }, indent=2, default=str))
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
