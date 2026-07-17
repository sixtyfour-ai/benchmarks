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
from pathlib import Path

import httpx

from judge import load_people, EvalRunner, clean_struct, post_with_retry, RUNS_DIR

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
    resp = await post_with_retry(
        client,
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
    runner: EvalRunner,
    poll_sem: asyncio.Semaphore,
    item: dict,
    run_id: str,
    t0: float,
    ckpt: dict,
    ckpt_path: Path,
) -> dict:
    async with poll_sem:
        try:
            while True:
                await asyncio.sleep(15)
                try:
                    resp = await client.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}")
                except (httpx.TransportError, httpx.TimeoutException):
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    continue
                resp.raise_for_status()
                status = resp.json().get("status", "")
                if status in ("completed", "failed", "cancelled"):
                    break

            elapsed = time.time() - t0

            if status != "completed":
                result = await runner.record_error(item, elapsed, RuntimeError(status))
                ckpt["completed"][item["person_info"]] = result
                save_checkpoint(ckpt_path, ckpt)
                return result

            for attempt in range(6):
                try:
                    resp = await client.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}/result")
                except (httpx.TransportError, httpx.TimeoutException):
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                break
            resp.raise_for_status()
            output = resp.json().get("output", {})
            if isinstance(output, dict) and "content" in output:
                output = output["content"]
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    output = {}
            output = clean_struct(output, item["fields"])

            result = await runner.record(item, output, elapsed, {"run_id": run_id, "status": "completed"})
            ckpt["completed"][item["person_info"]] = result
            save_checkpoint(ckpt_path, ckpt)
            return result

        except Exception as e:
            result = await runner.record_error(item, time.time() - t0, e)
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

    runner = EvalRunner(f"parallel_{args.processor}", vars(args))
    # previously completed (already-judged) results from the checkpoint count toward this run
    for entry in people:
        if entry["person_info"] in ckpt["completed"]:
            runner.results.append(ckpt["completed"][entry["person_info"]])
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

            tasks.append(poll_one(client, runner, poll_sem, item, run_id, t0, ckpt, ckpt_path))

        await asyncio.gather(*tasks)

    ok = [r for r in runner.results if "error" not in r]
    print(f"  Cost: ${len(ok) * cpt / 1000:.2f}", flush=True)
    runner._save()
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
