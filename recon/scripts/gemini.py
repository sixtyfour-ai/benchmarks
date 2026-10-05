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
from judge import load_people, EvalRunner, post_with_retry
from native_model_common import COMPILE_PROMPT

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def build_schema(fields: list[dict]) -> dict:
    return {
        "type": "OBJECT",
        "properties": {f["fieldname"]: {"type": "STRING", "description": f["description"]} for f in fields},
        "required": [f["fieldname"] for f in fields],
    }


# Gemini researches with Google Search and URL context. Code execution is not
# offered: given it, Gemini tries to fetch pages from the code sandbox, which
# has no network access, instead of calling Google Search.
RESEARCH_TOOLS = [{"googleSearch": {}}, {"urlContext": {}}]


async def generate(client: httpx.AsyncClient, model: str, body: dict) -> dict:
    resp = await post_with_retry(
        client,
        f"{GEMINI_BASE}/{model}:generateContent",
        json=body,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
    )
    response = resp.json()
    candidates = response.get("candidates") or []
    if not candidates or not (candidates[0].get("content") or {}).get("parts"):
        reason = candidates[0].get("finishReason") if candidates else response.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no content: {reason}")
    return response


async def call_api(client: httpx.AsyncClient, item: dict, model: str, thinking: str) -> tuple[dict, dict]:
    """Research with tools, then compile the answer with the response schema.

    Gemini does not research when a response schema is set, so the schema is
    applied in a second, tool-free turn over the same conversation.
    """
    fields_desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in item["fields"])
    question = {"role": "user", "parts": [{"text": (
        f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
        f"Person: {item['person_info']}\n\n"
        f"Find the following fields:\n{fields_desc}\n\n"
        f"For each field, search thoroughly using the description as guidance. "
        f"Cross-reference multiple sources for accuracy. "
        f"If you cannot find a definitive answer, return an empty string for that field."
    )}]}
    thinking_config = {"thinkingConfig": {"thinkingLevel": thinking.upper()}}
    research = await generate(client, model, {
        "contents": [question],
        "tools": RESEARCH_TOOLS,
        "generationConfig": thinking_config,
    })
    compiled = await generate(client, model, {
        "contents": [
            question,
            research["candidates"][0]["content"],
            {"role": "user", "parts": [{"text": COMPILE_PROMPT}]},
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": build_schema(item["fields"]),
            **thinking_config,
        },
    })
    return research, compiled


def extract_output(response: dict) -> dict:
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text") and not part.get("thought"):
                try:
                    return json.loads(part["text"])
                except Exception:
                    pass
    return {}


def extract_metadata(research: dict, compiled: dict) -> dict:
    research_usage = research.get("usageMetadata", {})
    compile_usage = compiled.get("usageMetadata", {})
    candidate = research["candidates"][0]
    grounding = candidate.get("groundingMetadata") or {}
    return {
        "input_tokens": research_usage.get("promptTokenCount", 0) + compile_usage.get("promptTokenCount", 0),
        "output_tokens": research_usage.get("candidatesTokenCount", 0) + compile_usage.get("candidatesTokenCount", 0),
        "thinking_tokens": research_usage.get("thoughtsTokenCount", 0) + compile_usage.get("thoughtsTokenCount", 0),
        "tool_use_tokens": research_usage.get("toolUsePromptTokenCount", 0),
        "web_searches": len(grounding.get("webSearchQueries") or []),
        "grounding_sources": len(grounding.get("groundingChunks") or []),
        "research_finish_reason": candidate.get("finishReason"),
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
                # Gemini runs its tool loop server-side within each request. The timeout
                # only catches a stalled connection; Gemini ends its own loop long before it.
                async with httpx.AsyncClient(timeout=httpx.Timeout(7200.0, connect=30.0)) as client:
                    research, compiled = await call_api(client, item, args.model, args.thinking)
                await runner.record(item, extract_output(compiled), time.time() - t0, extract_metadata(research, compiled))
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
