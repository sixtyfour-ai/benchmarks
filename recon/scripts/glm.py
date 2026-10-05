"""Run the benchmark against GLM-5.3 with Z.AI web search."""

import asyncio
import json

import httpx

from judge import post_with_retry
from native_model_common import (
    COMPILE_PROMPT,
    SYSTEM_PROMPT,
    SEARCH_ERRORS,
    ProviderConfig,
    authorization_headers,
    build_user_prompt,
    run_provider,
    search_failure,
    terminal_output,
    usage_values,
)


CHAT_ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
SEARCH_ENDPOINT = "https://api.z.ai/api/paas/v4/web_search"
CONFIG = ProviderConfig(
    name="glm",
    default_model="glm-5.3",
    key_env="ZAI_API_KEY",
    default_reasoning="max",
    reasoning_choices=("low", "high", "max"),
)

WEB_SEARCH_TOOL = {
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


async def call_glm(
    client: httpx.AsyncClient,
    item: dict,
    *,
    api_key: str,
    model: str,
    reasoning: str,
) -> tuple[dict, dict]:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    headers = {
        **authorization_headers(api_key),
        "Accept-Language": "en-US,en",
    }
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
    search_error_examples: list[str] = []
    terminal_repair = False

    # Research continues until GLM stops calling search.
    while True:
        can_search = not terminal_repair
        request_messages = (
            messages
            if can_search
            else [*messages, {"role": "user", "content": COMPILE_PROMPT}]
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
            request_payload.update(
                {"tools": [WEB_SEARCH_TOOL], "tool_choice": "auto"}
            )
        else:
            request_payload["tool_choice"] = "none"

        response = await post_with_retry(
            client,
            CHAT_ENDPOINT,
            json=request_payload,
            headers=headers,
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
                    raise RuntimeError("GLM requested a tool during terminal compilation")
                terminal_repair = True
                continue
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

                search_calls += 1
                try:
                    search_response = await post_with_retry(
                        client,
                        SEARCH_ENDPOINT,
                        json={
                            "search_engine": "search-prime",
                            "search_query": query,
                            "count": 10,
                            "search_recency_filter": "noLimit",
                        },
                        headers=headers,
                    )
                except SEARCH_ERRORS as exc:
                    error_type, error_message = search_failure(exc)
                    search_errors += 1
                    search_error_types[error_type] = search_error_types.get(error_type, 0) + 1
                    if len(search_error_examples) < 5:
                        search_error_examples.append(error_message)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"error": error_message}),
                        }
                    )
                    continue
                results = search_response.json().get("search_result") or []
                search_results += len(results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(results, ensure_ascii=False),
                    }
                )
            continue

        output, terminal_format_valid = terminal_output(message.get("content"))
        if not terminal_format_valid:
            if terminal_repair:
                raise RuntimeError("GLM terminal compilation returned no JSON object")
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
            "search_error_examples": search_error_examples,
            "provider_status": choice.get("finish_reason"),
            "terminal_format_valid": terminal_format_valid,
            "terminal_repaired": terminal_repair,
        }


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_glm))
