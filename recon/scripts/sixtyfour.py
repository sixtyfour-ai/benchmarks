"""
Evaluate Sixtyfour People Intelligence on the RECON benchmark.

Uses the async endpoint with polling for production-grade throughput.

Usage:
    python scripts/sixtyfour.py
    python scripts/sixtyfour.py --tier medium --people 5
    python scripts/sixtyfour.py --tier high --concurrency 10

Tiers:
    low     — Baseline, fast and lightweight. Available to all orgs.
    medium  — Deeper research with more sources. Available to all orgs.
    high    — OSINT-grade investigation. Exclusive access — contact sales.

Get an API key: https://app.sixtyfour.ai/keys
Docs: https://docs.sixtyfour.ai/api-reference/endpoint/people-intelligence

Requires: SIXTYFOUR_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import os
import time

import httpx
from judge import load_people, EvalRunner

SIXTYFOUR_API_KEY = os.environ["SIXTYFOUR_API_KEY"]
SIXTYFOUR_BASE = "https://api.sixtyfour.ai"
HEADERS = {
    "x-api-key": SIXTYFOUR_API_KEY,
    "Content-Type": "application/json",
}

TIERS = ["low", "medium", "high"]
RESTRICTED_TIERS = {"high"}


def build_struct(fields: list[dict]) -> dict:
    return {f["fieldname"]: f["description"] for f in fields}


async def submit_async(client: httpx.AsyncClient, item: dict, tier: str) -> str:
    resp = await client.post(
        f"{SIXTYFOUR_BASE}/people-intelligence-async",
        json={
            "lead_info": _parse_lead_info(item["person_info"]),
            "struct": build_struct(item["fields"]),
            "tier": tier,
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()["task_id"]


def _parse_lead_info(person_info: str) -> dict:
    info = {"description": person_info}
    for line in person_info.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if val:
                info[key] = val
    return info


async def poll_result(client: httpx.AsyncClient, task_id: str, timeout: float = 3600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(10)
        resp = await client.get(
            f"{SIXTYFOUR_BASE}/job-status/{task_id}",
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").lower()
        if status == "completed":
            return data.get("result", {})
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Job {status}: {data.get('error', task_id)}")
    raise TimeoutError(f"Job {task_id} did not complete within {timeout}s")


def extract_output(result: dict) -> dict:
    return result.get("structured_data", {})


def extract_metadata(result: dict) -> dict:
    return {
        "confidence_score": result.get("confidence_score"),
        "num_references": len(result.get("references", {})),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=TIERS, default="low")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    if args.tier in RESTRICTED_TIERS:
        print(
            f"NOTE: '{args.tier}' tier requires access enabled on your account.\n"
            f"Contact Sixtyfour sales to request access. Requests on orgs without\n"
            f"access will return 403.\n",
            flush=True,
        )

    people = load_people(args.people)
    runner = EvalRunner(f"sixtyfour_{args.tier}", {**vars(args)})
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Running Sixtyfour tier={args.tier} on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=1800.0) as client:
                    task_id = await submit_async(client, item, args.tier)
                    result = await poll_result(client, task_id)
                output = extract_output(result)
                await runner.record(item, output, time.time() - t0, extract_metadata(result))
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
