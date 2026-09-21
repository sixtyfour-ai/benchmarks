"""Run RECON through DeepSeek with a verified web-search surface.

V4 Pro currently emits provider-side ``web_search_call`` items. DeepSeek Flash
does not: the Responses API ignores built-in tools for Flash, so Flash receives
an explicit function tool backed by Exa search instead.
"""

import asyncio
import json
import os

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


ENDPOINT = "https://api.deepseek.com/responses"
EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
EXA_RESULTS_PER_SEARCH = 5
EXA_TEXT_CHARACTERS = 2000
WEB_SEARCH_FUNCTION = {
    "type": "function",
    "name": "web_search",
    "description": "Search the public web for relevant evidence and source passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused web search query"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
CONFIG = ProviderConfig(
    name="deepseek",
    default_model="deepseek-flash",
    key_env="DEEPSEEK_API_KEY",
    default_reasoning="none",
    reasoning_choices=("none", "low", "medium", "high", "xhigh", "max"),
    default_search_rounds=10,
)


def response_text_and_searches(payload: dict) -> tuple[str, int]:
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


def assert_completed(payload: dict, phase: str) -> None:
    if payload.get("status") != "completed":
        raise RuntimeError(
            f"DeepSeek {phase} {payload.get('status')}: "
            f"{payload.get('error') or payload.get('incomplete_details')}"
        )


async def call_deepseek(
    client: httpx.AsyncClient,
    item: dict,
    *,
    api_key: str,
    model: str,
    reasoning: str,
    max_search_rounds: int,
) -> tuple[dict, dict]:
    if "flash" in model:
        return await call_deepseek_flash(
            client, item, api_key=api_key, model=model, reasoning=reasoning,
            max_search_rounds=max_search_rounds,
        )
    return await call_deepseek_server_search(
        client, item, api_key=api_key, model=model, reasoning=reasoning,
    )


async def call_deepseek_server_search(
    client: httpx.AsyncClient,
    item: dict,
    *,
    api_key: str,
    model: str,
    reasoning: str,
) -> tuple[dict, dict]:
    schema = build_schema(item["fields"])
    response = await post_with_retry(
        client,
        ENDPOINT,
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
                    "schema": schema,
                }
            },
        },
        headers=authorization_headers(api_key),
    )
    payload = response.json()
    assert_completed(payload, "response")
    text, search_calls = response_text_and_searches(payload)
    output, terminal_format_valid = terminal_output(text)
    totals = usage_values(payload.get("usage"))
    terminal_repaired = False

    if not terminal_format_valid:
        # A completed research response may contain only reasoning/search items.
        # Preserve those items for the same single tool-free compilation used
        # for prose responses; never restart the research or invent answers.
        repair_response = await post_with_retry(
            client,
            ENDPOINT,
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
                        "schema": schema,
                    }
                },
            },
            headers=authorization_headers(api_key),
        )
        repair_payload = repair_response.json()
        assert_completed(repair_payload, "terminal compilation")
        repair_text, repair_searches = response_text_and_searches(repair_payload)
        search_calls += repair_searches
        for key, value in usage_values(repair_payload.get("usage")).items():
            totals[key] += value
        output, terminal_format_valid = terminal_output(repair_text)
        if not terminal_format_valid:
            raise RuntimeError("DeepSeek terminal compilation returned no JSON object")
        payload = repair_payload
        terminal_repaired = terminal_format_valid

    return output, {
        **totals,
        "model": payload.get("model", model),
        "reasoning": reasoning,
        "web_searches": search_calls,
        "search_backend": "deepseek/server_web_search",
        "provider_status": payload.get("status"),
        "terminal_format_valid": terminal_format_valid,
        "terminal_repaired": terminal_repaired,
    }


def compact_exa_results(payload: dict) -> list[dict]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Exa search returned no results array")
    compact = []
    for result in results:
        if not isinstance(result, dict):
            continue
        compact.append({
            key: value[:EXA_TEXT_CHARACTERS] if key == "text" and isinstance(value, str) else value
            for key, value in result.items()
            if key in {"title", "url", "publishedDate", "author", "text"}
            and value not in (None, "")
        })
    return compact


async def call_deepseek_flash(
    client: httpx.AsyncClient,
    item: dict,
    *,
    api_key: str,
    model: str,
    reasoning: str,
    max_search_rounds: int,
) -> tuple[dict, dict]:
    exa_key = os.getenv("EXA_API_KEY")
    if not exa_key:
        raise RuntimeError("DeepSeek Flash requires EXA_API_KEY for client-managed search")
    schema = build_schema(item["fields"])
    history: list[dict] = [{"role": "user", "content": build_user_prompt(item)}]
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    search_calls = 0
    search_results = 0
    terminal_repair = False
    compile_only = False

    for turn in range(max_search_rounds + 2):
        can_search = turn < max_search_rounds and not compile_only and not terminal_repair
        request = {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": history if can_search else [*history, {"role": "user", "content": FINALIZE_PROMPT}],
            "reasoning": {"effort": reasoning},
            "max_output_tokens": 131072 if can_search else 16384,
        }
        if can_search:
            request["tools"] = [WEB_SEARCH_FUNCTION]
            # DeepSeek rejects required/named tool choice in thinking mode.
            # The research prompt explicitly directs the model to search; auto
            # preserves high reasoning while still exposing the function tool.
            request["tool_choice"] = "auto"
        else:
            request["tool_choice"] = "none"
            request["text"] = {
                "format": {
                    "type": "json_schema", "name": "people_intelligence_fields",
                    "strict": True, "schema": schema,
                }
            }

        response = await post_with_retry(
            client, ENDPOINT, json=request, headers=authorization_headers(api_key),
        )
        payload = response.json()
        assert_completed(payload, "response")
        for key, value in usage_values(payload.get("usage")).items():
            totals[key] += value
        output_items = payload.get("output") or []
        function_calls = [item for item in output_items if item.get("type") == "function_call"]

        if function_calls:
            if not can_search:
                raise RuntimeError("DeepSeek Flash requested a tool during terminal compilation")
            history.extend(output_items)
            for function_call in function_calls:
                if function_call.get("name") != "web_search":
                    raise RuntimeError(f"unexpected DeepSeek Flash tool: {function_call.get('name')}")
                arguments = json.loads(function_call.get("arguments") or "{}")
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise RuntimeError("DeepSeek Flash called web_search without a query")
                if search_calls >= max_search_rounds:
                    history.append({
                        "type": "function_call_output",
                        "call_id": function_call["call_id"],
                        "output": json.dumps({
                            "error": "Search budget exhausted. Use the evidence already collected."
                        }),
                    })
                    continue
                search_response = await post_with_retry(
                    client,
                    EXA_SEARCH_ENDPOINT,
                    headers={"x-api-key": exa_key, "Content-Type": "application/json"},
                    json={
                        "query": query.strip(), "type": "auto",
                        "numResults": EXA_RESULTS_PER_SEARCH,
                        "contents": {"text": {"maxCharacters": EXA_TEXT_CHARACTERS}},
                    },
                )
                results = compact_exa_results(search_response.json())
                search_calls += 1
                search_results += len(results)
                history.append({
                    "type": "function_call_output",
                    "call_id": function_call["call_id"],
                    "output": json.dumps(results, ensure_ascii=False),
                })
            if search_calls >= max_search_rounds:
                compile_only = True
            continue

        if can_search:
            history.extend(output_items)
            compile_only = True
            continue

        text, _ = response_text_and_searches(payload)
        output, terminal_format_valid = terminal_output(text)
        if not terminal_format_valid:
            if terminal_repair:
                raise RuntimeError("DeepSeek Flash terminal compilation returned no JSON object")
            history.extend(output_items)
            terminal_repair = True
            continue
        return output, {
            **totals,
            "model": payload.get("model", model),
            "reasoning": reasoning,
            "web_searches": search_calls,
            "search_results": search_results,
            "search_backend": "exa/search",
            "search_policy": "auto_in_thinking_mode",
            "provider_status": payload.get("status"),
            "terminal_format_valid": terminal_format_valid,
            "terminal_repaired": terminal_repair,
        }

    raise RuntimeError("DeepSeek Flash did not produce a terminal response")


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_deepseek))
