"""
Evaluate Google Gemini models on the RECON benchmark.

Usage:
    python scripts/gemini.py
    python scripts/gemini.py --thinking medium --people 5
    python scripts/gemini.py --model gemini-3.1-pro-preview --thinking low

Requires: GEMINI_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from judge import load_people, EvalRunner

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def build_schema(fields: list[dict]) -> dict:
    return {
        "type": "OBJECT",
        "properties": {f["fieldname"]: {"type": "STRING", "description": f["description"]} for f in fields},
        "required": [f["fieldname"] for f in fields],
    }


async def call_api(client: httpx.AsyncClient, item: dict, model: str, thinking: str) -> dict:
    fields_desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in item["fields"])
    resp = await client.post(
        f"{GEMINI_BASE}/{model}:generateContent",
        json={
            "contents": [{"parts": [{"text": (
                f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
                f"Person: {item['person_info']}\n\n"
                f"Find the following fields:\n{fields_desc}\n\n"
                f"For each field, search thoroughly using the description as guidance. "
                f"Cross-reference multiple sources for accuracy. "
                f"If you cannot find a definitive answer, return an empty string for that field."
            )}]}],
            "tools": [{"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": build_schema(item["fields"]),
                "thinkingConfig": {"thinkingLevel": thinking.upper()},
            },
        },
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def extract_output(response: dict) -> dict:
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                try:
                    return json.loads(part["text"])
                except Exception:
                    pass
    return {}


def extract_metadata(response: dict) -> dict:
    usage = response.get("usageMetadata", {})
    return {
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "thinking_tokens": usage.get("thoughtsTokenCount", 0),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--thinking", choices=["low", "medium", "high"], default="high")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    people = load_people(args.people)
    runner = EvalRunner(f"gemini_{args.thinking}", vars(args))
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Running {args.model} (thinking={args.thinking}) on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=900.0) as client:
                    response = await call_api(client, item, args.model, args.thinking)
                await runner.record(item, extract_output(response), time.time() - t0, extract_metadata(response))
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
