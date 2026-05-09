"""
Evaluate Exa search API on the RECON benchmark.

Usage:
    python scripts/exa.py
    python scripts/exa.py --type deep --people 5
    python scripts/exa.py --type deep-lite --concurrency 20

Requires: EXA_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from judge import load_people, EvalRunner

EXA_API_KEY = os.environ["EXA_API_KEY"]
EXA_ENDPOINT = "https://api.exa.ai/search"


def build_schema(fields: list[dict]) -> dict:
    return {
        "type": "object",
        "properties": {f["fieldname"]: {"type": "string", "description": f["description"]} for f in fields},
    }


async def call_api(client: httpx.AsyncClient, item: dict, search_type: str) -> dict:
    fields_desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in item["fields"])
    resp = await client.post(
        EXA_ENDPOINT,
        json={
            "query": (
                f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
                f"Person: {item['person_info']}\n\n"
                f"Find the following fields:\n{fields_desc}\n\n"
                f"For each field, search thoroughly using the description as guidance. "
                f"Cross-reference multiple sources for accuracy. "
                f"If you cannot find a definitive answer, return an empty string for that field."
            ),
            "type": search_type,
            "outputSchema": build_schema(item["fields"]),
            "numResults": 10,
        },
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def extract_output(response: dict) -> dict:
    output = response.get("output", {})
    if isinstance(output, dict) and "content" in output:
        output = output["content"]
    if isinstance(output, str):
        try:
            return json.loads(output)
        except Exception:
            return {}
    return output if isinstance(output, dict) else {}


def extract_metadata(response: dict) -> dict:
    cost = response.get("costDollars", {})
    return {"cost_usd": cost.get("total", 0) if isinstance(cost, dict) else (cost or 0)}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["deep", "deep-reasoning", "deep-lite"], default="deep-reasoning")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    people = load_people(args.people)
    runner = EvalRunner(f"exa_{args.type}", vars(args))
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Running Exa {args.type} on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=3600.0) as client:
                    response = await call_api(client, item, args.type)
                output = extract_output(response)
                metadata = extract_metadata(response)
                await runner.record(item, output, time.time() - t0, metadata)
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
