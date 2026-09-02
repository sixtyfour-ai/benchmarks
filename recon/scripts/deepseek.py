"""Run the benchmark against DeepSeek V4 Flash with server-side web search."""

import asyncio

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
CONFIG = ProviderConfig(
    name="deepseek",
    default_model="deepseek-v4-flash",
    key_env="DEEPSEEK_API_KEY",
    default_reasoning="none",
    reasoning_choices=("none", "low", "medium", "high", "xhigh", "max"),
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
    max_search_rounds: None,
) -> tuple[dict, dict]:
    if max_search_rounds is not None:
        raise ValueError("DeepSeek controls its server-side web-search limit")
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
    if not text.strip():
        output_shape = [
            {
                "type": output_item.get("type"),
                "keys": sorted(output_item),
                "content_types": [
                    content.get("type")
                    if isinstance(content, dict)
                    else type(content).__name__
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

    output, terminal_format_valid = terminal_output(text)
    totals = usage_values(payload.get("usage"))
    terminal_repaired = False

    if not terminal_format_valid:
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


if __name__ == "__main__":
    asyncio.run(run_provider(CONFIG, call_deepseek))
