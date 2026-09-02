"""Evaluate native web-research APIs from Kimi, DeepSeek, and Z.AI.

Each provider receives the same person prompt and strict output schema. Search
stays within the provider: Kimi ``$web_search``, DeepSeek Responses
``web_search``, and GLM function calling backed by the Z.AI Web Search API.

Examples:
    python scripts/native_models.py --provider kimi --reasoning max
    python scripts/native_models.py --provider deepseek --reasoning none
    python scripts/native_models.py --provider glm --reasoning max

Requires OPENAI_API_KEY for judging and one provider key:
MOONSHOT_API_KEY, DEEPSEEK_API_KEY, or ZAI_API_KEY.
"""

import argparse
import asyncio
import json
import os
import time
from typing import Any

import httpx

from judge import EvalRunner, clean_struct, load_people, post_with_retry


PROVIDER_DEFAULTS = {
    "kimi": {
        "model": "kimi-k3",
        "endpoint": "https://api.moonshot.ai/v1/chat/completions",
        "key_env": "MOONSHOT_API_KEY",
        "reasoning": "max",
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "endpoint": "https://api.deepseek.com/responses",
        "key_env": "DEEPSEEK_API_KEY",
        "reasoning": "none",
    },
    "glm": {
        "model": "glm-5.3",
        "endpoint": "https://api.z.ai/api/paas/v4/chat/completions",
        "search_endpoint": "https://api.z.ai/api/paas/v4/web_search",
        "key_env": "ZAI_API_KEY",
        "reasoning": "max",
    },
}

REASONING_CHOICES = ["none", "low", "medium", "high", "xhigh", "max"]

SYSTEM_PROMPT = """You are a research agent specializing in people intelligence.
Use web search to investigate the exact person in the supplied identity context.
Search thoroughly, follow relevant primary sources, and distinguish same-name people.
Return only facts supported by the evidence you found. If a field cannot be
resolved confidently, return an empty string rather than guessing."""

FINALIZE_PROMPT = """The web-search budget is exhausted. Using the evidence already
collected, return the requested JSON object now. Do not call another tool."""


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


def _authorization(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _json_content(raw: Any) -> dict:
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
        # Some native-search endpoints ignore their structured-output setting
        # after tool use and prepend a short answer before the requested JSON.
        # Recover only an actual JSON object; never infer fields from prose.
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
            raise ValueError("provider final text contained no JSON object") from direct_error
    if not isinstance(parsed, dict):
        raise ValueError("provider final output is not a JSON object")
    return parsed


def _usage_values(usage: dict | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cached_tokens = (
        (usage.get("input_tokens_details") or {}).get("cached_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_tokens": int(cached_tokens),
        "reasoning_tokens": int(reasoning_tokens),
    }


def _terminal_output(raw: Any) -> tuple[dict, bool]:
    """Separate a model formatting failure from a provider/runtime failure."""
    try:
        return _json_content(raw), True
    except ValueError:
        # A refusal, prose-only answer, or malformed terminal object is a model
        # result. Score it as missing rather than making the whole run
        # non-publishable as though transport had failed.
        return {}, False


def _deepseek_text_and_searches(payload: dict) -> tuple[str, int]:
    output_items = payload.get("output") or []
    search_calls = sum(
        output_item.get("type") == "web_search_call"
        for output_item in output_items
    )
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text, search_calls

    text_parts = []
    for output_item in output_items:
        if output_item.get("type") != "message":
            continue
        content_items = output_item.get("content") or []
        if isinstance(content_items, str):
            text_parts.append(content_items)
            continue
        for content in content_items:
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, dict) and isinstance(content.get("text"), str):
                text_parts.append(content["text"])
    return "".join(text_parts), search_calls


async def call_kimi(
    client: httpx.AsyncClient,
    item: dict,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    reasoning: str,
    max_search_rounds: int,
) -> tuple[dict, dict]:
    schema = build_schema(item["fields"])
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    search_calls = 0

    for turn in range(max_search_rounds + 1):
        can_search = turn < max_search_rounds
        request_messages = (
            messages
            if can_search
            else [*messages, {"role": "user", "content": FINALIZE_PROMPT}]
        )
        request_payload = {
            "model": model,
            "messages": request_messages,
            "reasoning_effort": reasoning,
            "max_completion_tokens": 131072,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "people_intelligence_fields",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if can_search:
            request_payload["tools"] = [
                {
                    "type": "builtin_function",
                    "function": {"name": "$web_search"},
                }
            ]
        response = await post_with_retry(
            client,
            endpoint,
            json=request_payload,
            headers=_authorization(api_key),
        )
        payload = response.json()
        for key, value in _usage_values(payload.get("usage")).items():
            totals[key] += value
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if choice.get("finish_reason") == "tool_calls" or tool_calls:
            if not can_search:
                raise RuntimeError("Kimi requested a tool during terminal compilation")
            # Kimi requires the complete assistant message, including its
            # reasoning_content, to be returned unchanged on the next turn.
            messages.append(message)
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                if function.get("name") != "$web_search":
                    raise RuntimeError(f"unexpected Kimi tool: {function.get('name')}")
                arguments = json.loads(function.get("arguments") or "{}")
                search_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "$web_search",
                        "content": json.dumps(arguments),
                    }
                )
            continue

        output, terminal_format_valid = _terminal_output(message.get("content"))
        return output, {
            **totals,
            "model": payload.get("model", model),
            "reasoning": reasoning,
            "web_searches": search_calls,
            "provider_status": choice.get("finish_reason"),
            "terminal_format_valid": terminal_format_valid,
        }

    raise RuntimeError("Kimi did not produce a terminal response")


async def call_deepseek(
    client: httpx.AsyncClient,
    item: dict,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    reasoning: str,
    max_search_rounds: int | None,
) -> tuple[dict, dict]:
    if max_search_rounds is not None:
        raise ValueError("DeepSeek controls its server-side web-search limit")
    response = await post_with_retry(
        client,
        endpoint,
        json={
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": build_user_prompt(item),
            "reasoning": {"effort": reasoning},
            "max_output_tokens": 131072,
            "tools": [{"type": "web_search"}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "people_intelligence_fields",
                    "strict": True,
                    "schema": build_schema(item["fields"]),
                }
            },
        },
        headers=_authorization(api_key),
    )
    payload = response.json()
    if payload.get("status") != "completed":
        raise RuntimeError(
            f"DeepSeek response {payload.get('status')}: "
            f"{payload.get('error') or payload.get('incomplete_details')}"
        )
    text, search_calls = _deepseek_text_and_searches(payload)
    if not text.strip():
        output_shape = [
            {
                "type": output_item.get("type"),
                "keys": sorted(output_item),
                "content_types": [
                    content.get("type") if isinstance(content, dict) else type(content).__name__
                    for content in (
                        output_item.get("content")
                        if isinstance(output_item.get("content"), list)
                        else []
                    )
                ],
            }
            for output_item in payload.get("output") or []
        ]
        raise ValueError(f"DeepSeek returned no final text; output shape={output_shape}")
    output, terminal_format_valid = _terminal_output(text)
    totals = _usage_values(payload.get("usage"))
    terminal_repaired = False

    if not terminal_format_valid:
        repair_response = await post_with_retry(
            client,
            endpoint,
            json={
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": [
                    {"role": "user", "content": build_user_prompt(item)},
                    *(payload.get("output") or []),
                    {"role": "user", "content": FINALIZE_PROMPT},
                ],
                "reasoning": {"effort": reasoning},
                "max_output_tokens": 16384,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "people_intelligence_fields",
                        "strict": True,
                        "schema": build_schema(item["fields"]),
                    }
                },
            },
            headers=_authorization(api_key),
        )
        repair_payload = repair_response.json()
        if repair_payload.get("status") != "completed":
            raise RuntimeError(
                f"DeepSeek terminal compilation {repair_payload.get('status')}: "
                f"{repair_payload.get('error') or repair_payload.get('incomplete_details')}"
            )
        repair_text, repair_searches = _deepseek_text_and_searches(repair_payload)
        search_calls += repair_searches
        for key, value in _usage_values(repair_payload.get("usage")).items():
            totals[key] += value
        output, terminal_format_valid = _terminal_output(repair_text)
        payload = repair_payload
        terminal_repaired = terminal_format_valid

    return output, {
        **totals,
        "model": payload.get("model", model),
        "reasoning": reasoning,
        "web_searches": search_calls,
        "provider_status": payload.get("status"),
        "terminal_format_valid": terminal_format_valid,
        "terminal_repaired": terminal_repaired,
    }


async def call_glm(
    client: httpx.AsyncClient,
    item: dict,
    *,
    endpoint: str,
    search_endpoint: str,
    api_key: str,
    model: str,
    reasoning: str,
    max_search_rounds: int,
) -> tuple[dict, dict]:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the public web. Use multiple focused queries and refine "
                    "them when initial results are insufficient."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A focused web search query",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    headers = {**_authorization(api_key), "Accept-Language": "en-US,en"}
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    search_calls = 0
    search_results = 0

    for turn in range(max_search_rounds + 1):
        can_search = turn < max_search_rounds
        request_messages = (
            messages
            if can_search
            else [*messages, {"role": "user", "content": FINALIZE_PROMPT}]
        )
        request_payload = {
            "model": model,
            "messages": request_messages,
            "thinking": {"type": "enabled", "clear_thinking": False},
            "reasoning_effort": reasoning,
            "temperature": 1.0,
            "max_tokens": 131072,
            "response_format": {"type": "json_object"},
        }
        if can_search:
            request_payload.update({"tools": tools, "tool_choice": "auto"})
        response = await post_with_retry(
            client,
            endpoint,
            json=request_payload,
            headers=headers,
        )
        payload = response.json()
        for key, value in _usage_values(payload.get("usage")).items():
            totals[key] += value
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if choice.get("finish_reason") == "tool_calls" or tool_calls:
            if not can_search:
                raise RuntimeError("GLM requested a tool during terminal compilation")
            # Preserve the complete assistant message so GLM can continue its
            # reasoning coherently after each tool result.
            messages.append(message)
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                if function.get("name") != "web_search":
                    raise RuntimeError(f"unexpected GLM tool: {function.get('name')}")
                arguments = function.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                query = str(arguments.get("query") or "").strip()
                if not query:
                    raise RuntimeError("GLM called web_search without a query")
                search_response = await post_with_retry(
                    client,
                    search_endpoint,
                    json={
                        "search_engine": "search-prime",
                        "search_query": query,
                        "count": 10,
                        "search_recency_filter": "noLimit",
                    },
                    headers=headers,
                )
                results = search_response.json().get("search_result") or []
                search_calls += 1
                search_results += len(results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(results, ensure_ascii=False),
                    }
                )
            continue

        output, terminal_format_valid = _terminal_output(message.get("content"))
        return output, {
            **totals,
            "model": payload.get("model", model),
            "reasoning": reasoning,
            "web_searches": search_calls,
            "search_results": search_results,
            "provider_status": choice.get("finish_reason"),
            "terminal_format_valid": terminal_format_valid,
        }

    raise RuntimeError("GLM did not produce a terminal response")


PROVIDER_CALLS = {
    "kimi": call_kimi,
    "deepseek": call_deepseek,
    "glm": call_glm,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def resolve_max_search_rounds(provider: str, requested: int | None) -> int | None:
    if provider in {"kimi", "glm"}:
        return requested or 10
    if requested is not None:
        raise SystemExit(
            "--max-search-rounds is supported for Kimi and GLM; "
            "DeepSeek controls its server-side search limit"
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDER_DEFAULTS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning", choices=REASONING_CHOICES, default=None)
    parser.add_argument("--people", type=positive_int, default=None)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    parser.add_argument("--max-search-rounds", type=positive_int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    defaults = PROVIDER_DEFAULTS[args.provider]
    model = args.model or defaults["model"]
    reasoning = args.reasoning or defaults["reasoning"]
    max_search_rounds = resolve_max_search_rounds(
        args.provider, args.max_search_rounds
    )
    if args.provider == "kimi" and reasoning not in {"low", "high", "max"}:
        raise SystemExit("Kimi K3 reasoning must be low, high, or max")
    if args.provider == "glm" and reasoning not in {"low", "high", "max"}:
        raise SystemExit("GLM 5.3 reasoning must be low, high, or max")

    key_env = defaults["key_env"]
    api_key = os.getenv(key_env)
    if not api_key:
        raise SystemExit(f"Missing {key_env}")

    people = load_people(args.people)
    config = {
        "provider": args.provider,
        "model": model,
        "reasoning": reasoning,
        "concurrency": args.concurrency,
    }
    if max_search_rounds is not None:
        config["max_search_rounds"] = max_search_rounds
    runner = EvalRunner(f"{args.provider}_{model}_{reasoning}", config)
    sem = asyncio.Semaphore(args.concurrency)
    call_provider = PROVIDER_CALLS[args.provider]

    print(
        f"Running provider={args.provider} model={model} reasoning={reasoning} "
        f"on {len(people)} people, concurrency={args.concurrency}",
        flush=True,
    )
    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 20),
        max_keepalive_connections=max(args.concurrency, 10),
    )

    async def process(client: httpx.AsyncClient, item: dict) -> None:
        async with sem:
            started = time.time()
            try:
                output, metadata = await call_provider(
                    client,
                    item,
                    endpoint=defaults["endpoint"],
                    **(
                        {"search_endpoint": defaults["search_endpoint"]}
                        if args.provider == "glm"
                        else {}
                    ),
                    api_key=api_key,
                    model=model,
                    reasoning=reasoning,
                    max_search_rounds=max_search_rounds,
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


if __name__ == "__main__":
    asyncio.run(main())
