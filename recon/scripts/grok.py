"""
Evaluate xAI Grok models on the RECON benchmark.

Usage:
    python scripts/grok.py
    python scripts/grok.py --model 4.1-fast --people 5
    python scripts/grok.py --model 4.20-ma --concurrency 3

Requires: XAI_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from judge import load_people, EvalRunner, post_with_retry

XAI_API_KEY = os.environ["XAI_API_KEY"]
XAI_ENDPOINT = "https://api.x.ai/v1/responses"

MODELS = {
    "4.3": "grok-4.3",
    "4.20": "grok-4.20-0309-reasoning",
    "4.20-ma": "grok-4.20-multi-agent-0309",
    "4.1-fast": "grok-4-1-fast-reasoning",
}


def build_schema(fields: list[dict]) -> dict:
    properties = {f["fieldname"]: {"type": "string", "description": f["description"]} for f in fields}
    return {
        "type": "json_schema",
        "name": "recon_fields",
        "strict": True,
        "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
    }


async def call_api(client: httpx.AsyncClient, item: dict, model: str) -> dict:
    fields_desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in item["fields"])
    person_key = (item.get("name") or item["person_info"]).replace(" ", "_").lower()
    payload = {
        "model": model,
        "reasoning_effort": "high",
        "input": (
            f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
            f"Person: {item['person_info']}\n\n"
            f"Find the following fields:\n{fields_desc}\n\n"
            f"For each field, search thoroughly using the description as guidance. "
            f"Cross-reference multiple sources for accuracy. "
            f"If you cannot find a definitive answer, return an empty string for that field."
        ),
        "tools": [{"type": "web_search"}, {"type": "x_search"}, {"type": "code_interpreter"}],
        "text": {"format": build_schema(item["fields"])},
        "prompt_cache_key": f"recon_eval_{person_key}",
    }
    resp = await post_with_retry(
        client,
        XAI_ENDPOINT,
        json=payload,
        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
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
    input_details = usage.get("input_tokens_details", {})
    output_details = usage.get("output_tokens_details", {})
    tool_calls = {"web_search_call": 0, "x_search_call": 0, "code_interpreter_call": 0}
    for item in response.get("output", []):
        t = item.get("type", "")
        if t in tool_calls:
            tool_calls[t] += 1
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cached_tokens": input_details.get("cached_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        "cost_usd": usage.get("cost_in_usd_ticks", 0) / 1e10,
        "web_searches": tool_calls["web_search_call"],
        "x_searches": tool_calls["x_search_call"],
        "code_interpreter": tool_calls["code_interpreter_call"],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="4.20-ma")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()

    people = load_people(args.people)
    model_id = MODELS[args.model]
    runner = EvalRunner(f"grok_{args.model}", {**vars(args), "model_id": model_id})
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Running Grok {args.model} ({model_id}) on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=1800.0) as client:
                    response = await call_api(client, item, model_id)
                await runner.record(item, extract_output(response), time.time() - t0, extract_metadata(response))
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
