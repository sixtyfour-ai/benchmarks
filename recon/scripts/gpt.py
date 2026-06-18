"""
Evaluate OpenAI GPT models on the RECON benchmark.

Usage:
    python scripts/gpt.py
    python scripts/gpt.py --reasoning medium --people 5
    python scripts/gpt.py --model gpt-5.4 --reasoning high --concurrency 20

Requires: OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from judge import load_people, EvalRunner, post_with_retry

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"


def build_schema(fields: list[dict]) -> dict:
    properties = {f["fieldname"]: {"type": "string", "description": f["description"]} for f in fields}
    return {
        "type": "json_schema",
        "name": "recon_fields",
        "strict": True,
        "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
    }


async def call_api(client: httpx.AsyncClient, item: dict, model: str, reasoning: str) -> dict:
    fields_desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in item["fields"])
    resp = await post_with_retry(
        client,
        OPENAI_ENDPOINT,
        json={
            "model": model,
            "input": (
                f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
                f"Person: {item['person_info']}\n\n"
                f"Find the following fields:\n{fields_desc}\n\n"
                f"For each field, search thoroughly using the description as guidance. "
                f"Cross-reference multiple sources for accuracy. "
                f"If you cannot find a definitive answer, return an empty string for that field."
            ),
            "reasoning": {"effort": reasoning},
            "tools": [
                {"type": "web_search"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
            "text": {"format": build_schema(item["fields"])},
        },
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    )
    return resp.json()


def extract_output(response: dict) -> dict:
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    try:
                        return json.loads(content["text"])
                    except Exception:
                        pass
    return {}


def extract_metadata(response: dict) -> dict:
    usage = response.get("usage", {})
    output_details = usage.get("output_tokens_details", {})
    web_searches = sum(1 for item in response.get("output", []) if "web_search" in item.get("type", ""))
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        "web_searches": web_searches,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--reasoning", choices=["none", "low", "medium", "high", "xhigh"], default="xhigh")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    people = load_people(args.people)
    runner = EvalRunner(f"gpt_{args.model}_{args.reasoning}", vars(args))
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Running {args.model} (reasoning={args.reasoning}) on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=1800.0) as client:
                    response = await call_api(client, item, args.model, args.reasoning)
                await runner.record(item, extract_output(response), time.time() - t0, extract_metadata(response))
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
