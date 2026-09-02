"""Run the benchmark against GLM-5.3 with Z.AI web search."""

import asyncio
import json

import httpx

from judge import post_with_retry
from native_model_common import (
    FINALIZE_PROMPT,
    SYSTEM_PROMPT,
    ProviderConfig,
    authorization_headers,
    build_user_prompt,
    run_provider,
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
    default_search_rounds=10,
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
    max_search_rounds: int,
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
            request_payload.update(
                {"tools": [WEB_SEARCH_TOOL], "tool_choice": "auto"}
            )

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
                    SEARCH_ENDPOINT,
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

        output, terminal_format_valid = terminal_output(message.get("content"))
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


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_glm))
