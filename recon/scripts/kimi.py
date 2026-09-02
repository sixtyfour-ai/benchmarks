"""Run the benchmark against Kimi K3 with Moonshot's built-in web search."""

import asyncio
import json

import httpx

from judge import post_with_retry
from native_model_common import (
    FINALIZE_PROMPT,
    SYSTEM_PROMPT,
    ProviderConfig,
    authorization_headers,
    build_schema,
    build_user_prompt,
    run_provider,
    terminal_output,
    usage_values,
)


ENDPOINT = "https://api.moonshot.ai/v1/chat/completions"
CONFIG = ProviderConfig(
    name="kimi",
    default_model="kimi-k3",
    key_env="MOONSHOT_API_KEY",
    default_reasoning="max",
    reasoning_choices=("low", "high", "max"),
    default_search_rounds=10,
)


async def call_kimi(
    client: httpx.AsyncClient,
    item: dict,
    *,
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
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
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
            ENDPOINT,
            json=request_payload,
            headers=authorization_headers(api_key),
        )
        payload = response.json()
        for key, value in usage_values(payload.get("usage")).items():
            totals[key] += value
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if choice.get("finish_reason") == "tool_calls" or tool_calls:
            if not can_search:
                raise RuntimeError("Kimi requested a tool during terminal compilation")
            # Moonshot requires the complete assistant message, including its
            # reasoning content, to be returned unchanged on the next turn.
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

        output, terminal_format_valid = terminal_output(message.get("content"))
        return output, {
            **totals,
            "model": payload.get("model", model),
            "reasoning": reasoning,
            "web_searches": search_calls,
            "provider_status": choice.get("finish_reason"),
            "terminal_format_valid": terminal_format_valid,
        }

    raise RuntimeError("Kimi did not produce a terminal response")


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_kimi))
