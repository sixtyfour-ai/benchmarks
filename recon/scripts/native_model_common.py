"""Shared runtime for direct native-model benchmark scripts."""

import argparse
import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from judge import EvalRunner, clean_struct, load_people


SYSTEM_PROMPT = """You are a research agent specializing in people intelligence.
Use web search to investigate the exact person in the supplied identity context.
Search thoroughly, follow relevant primary sources, and distinguish same-name people.
Return only facts supported by the evidence you found. If a field cannot be
resolved confidently, return an empty string rather than guessing."""

FINALIZE_PROMPT = """The web-search budget is exhausted. Using the evidence already
collected, return the requested JSON object now. Do not call another tool."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    default_model: str
    key_env: str
    default_reasoning: str
    reasoning_choices: tuple[str, ...]
    default_search_rounds: int | None = None


ProviderCall = Callable[..., Awaitable[tuple[dict, dict]]]


def build_schema(fields: list[dict]) -> dict:
    properties = {
        field["fieldname"]: {
            "type": "string",
            "description": field["description"],
        }
        for field in fields
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_user_prompt(item: dict) -> str:
    requested = "\n".join(
        f"- {field['fieldname']}: {field['description']}"
        for field in item["fields"]
    )
    return (
        f"<person>\n{item['person_info']}\n</person>\n\n"
        f"<requested_fields>\n{requested}\n</requested_fields>\n\n"
        "Research every requested field. Preserve the exact field names in the "
        "final JSON object. Use an empty string for unresolved fields."
    )


def authorization_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def json_content(raw: Any) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("provider returned no final text")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as direct_error:
        # Some native-search endpoints prepend prose despite structured output.
        # Recover an actual JSON object, but never infer field values from prose.
        decoder = json.JSONDecoder()
        parsed = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise ValueError(
                "provider final text contained no JSON object"
            ) from direct_error
    if not isinstance(parsed, dict):
        raise ValueError("provider final output is not a JSON object")
    return parsed


def usage_values(usage: dict | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cached_tokens = (
        (usage.get("input_tokens_details") or {}).get("cached_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    reasoning_tokens = (
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    )
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_tokens": int(cached_tokens),
        "reasoning_tokens": int(reasoning_tokens),
    }


def terminal_output(raw: Any) -> tuple[dict, bool]:
    """Separate a model formatting failure from a provider/runtime failure."""
    try:
        return json_content(raw), True
    except ValueError:
        return {}, False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(config: ProviderConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.default_model)
    parser.add_argument(
        "--reasoning",
        choices=config.reasoning_choices,
        default=config.default_reasoning,
    )
    parser.add_argument("--people", type=positive_int, default=None)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    if config.default_search_rounds is not None:
        parser.add_argument(
            "--max-search-rounds",
            type=positive_int,
            default=config.default_search_rounds,
        )
    else:
        parser.set_defaults(max_search_rounds=None)
    return parser.parse_args()


async def run_provider(config: ProviderConfig, call_provider: ProviderCall) -> None:
    args = parse_args(config)
    api_key = os.getenv(config.key_env)
    if not api_key:
        raise SystemExit(f"Missing {config.key_env}")

    people = load_people(args.people)
    run_config = {
        "provider": config.name,
        "model": args.model,
        "reasoning": args.reasoning,
        "concurrency": args.concurrency,
    }
    if args.max_search_rounds is not None:
        run_config["max_search_rounds"] = args.max_search_rounds

    runner = EvalRunner(
        f"{config.name}_{args.model}_{args.reasoning}", run_config
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 20),
        max_keepalive_connections=max(args.concurrency, 10),
    )
    print(
        f"Running provider={config.name} model={args.model} "
        f"reasoning={args.reasoning} on {len(people)} people, "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    async def process(client: httpx.AsyncClient, item: dict) -> None:
        async with semaphore:
            started = time.time()
            try:
                output, metadata = await call_provider(
                    client,
                    item,
                    api_key=api_key,
                    model=args.model,
                    reasoning=args.reasoning,
                    max_search_rounds=args.max_search_rounds,
                )
                await runner.record(
                    item,
                    clean_struct(output, item["fields"]),
                    time.time() - started,
                    metadata,
                )
            except Exception as exc:
                await runner.record_error(item, time.time() - started, exc)

    async with httpx.AsyncClient(timeout=1800.0, limits=limits) as client:
        await asyncio.gather(*(process(client, item) for item in people))
    runner.summary()
