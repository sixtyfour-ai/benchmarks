"""Run the benchmark against Kimi K3 with Moonshot's native search API."""

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
SEARCH_ENDPOINT = "https://api.moonshot.ai/v1/tools/search_pro"
SEARCH_MAX_RETRIES = 4
RETRYABLE_SEARCH_STATUS = {429, 500, 502, 503, 504}
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web for relevant evidence and source passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused web search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
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
    search_results = 0
    search_errors = 0
    search_error_types: dict[str, int] = {}
    terminal_repair = False
    compile_only = False

    for turn in range(max_search_rounds + 2):
        can_search = turn < max_search_rounds and not terminal_repair and not compile_only
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
        }
        if can_search:
            request_payload["tools"] = [WEB_SEARCH_TOOL]
            request_payload["tool_choice"] = "required" if turn == 0 else "auto"
        else:
            request_payload["tool_choice"] = "none"
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "people_intelligence_fields",
                    "strict": True,
                    "schema": schema,
                },
            }

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
                if terminal_repair:
                    raise RuntimeError("Kimi requested a tool during terminal compilation")
                terminal_repair = True
                continue
            # Moonshot requires the complete assistant message, including its
            # reasoning content, to be returned unchanged on the next turn.
            messages.append(message)
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                if function.get("name") != "web_search":
                    raise RuntimeError(f"unexpected Kimi tool: {function.get('name')}")
                arguments = json.loads(function.get("arguments") or "{}")
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise RuntimeError("Kimi called web_search without a query")
                search_calls += 1
                try:
                    search_response = await post_with_retry(
                        client,
                        SEARCH_ENDPOINT,
                        json={"text_query": query.strip(), "limit": 10, "timeout_seconds": 30},
                        headers=authorization_headers(api_key),
                        max_retries=SEARCH_MAX_RETRIES,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in RETRYABLE_SEARCH_STATUS:
                        raise
                    search_errors += 1
                    error_type = f"http_{exc.response.status_code}"
                    search_error_types[error_type] = search_error_types.get(error_type, 0) + 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": "web_search",
                            "content": json.dumps(
                                {"error": "Search temporarily failed. Try a shorter or differently phrased query."}
                            ),
                        }
                    )
                    continue
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    search_errors += 1
                    error_type = "timeout" if isinstance(exc, httpx.TimeoutException) else type(exc).__name__
                    search_error_types[error_type] = search_error_types.get(error_type, 0) + 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": "web_search",
                            "content": json.dumps(
                                {"error": "Search temporarily failed. Try a shorter or differently phrased query."}
                            ),
                        }
                    )
                    continue
                results = search_response.json().get("search_results")
                if not isinstance(results, list):
                    raise RuntimeError("Kimi search API returned no search_results array")
                search_results += len(results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "web_search",
                        "content": json.dumps(results, ensure_ascii=False),
                    }
                )
            continue

        if can_search:
            messages.append(message)
            compile_only = True
            continue

        output, terminal_format_valid = terminal_output(message.get("content"))
        if not terminal_format_valid:
            if terminal_repair:
                raise RuntimeError("Kimi terminal compilation returned no JSON object")
            messages.append(message)
            terminal_repair = True
            continue
        return output, {
            **totals,
            "model": payload.get("model", model),
            "reasoning": reasoning,
            "web_searches": search_calls,
            "search_results": search_results,
            "search_errors": search_errors,
            "search_error_types": search_error_types,
            "search_backend": "moonshot/search_pro",
            "search_policy": "first_turn_required_then_auto",
            "provider_status": choice.get("finish_reason"),
            "terminal_format_valid": terminal_format_valid,
            "terminal_repaired": terminal_repair,
        }

    raise RuntimeError("Kimi did not produce a terminal response")


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_kimi))
