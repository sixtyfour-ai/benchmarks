"""RECON through Anthropic's native search and code-execution tools.

Requires anthropic, httpx and openai; ANTHROPIC_API_KEY and OPENAI_API_KEY.
Example: python scripts/claude.py --model claude-opus-5 --effort xhigh
"""

import argparse
import asyncio
import time

import anthropic
if int(anthropic.__version__.split(".")[0]) >= 1:
    import httpx2 as httpx
else:
    import httpx

from judge import EvalRunner, load_people
from native_model_common import SYSTEM_PROMPT, build_user_prompt, json_content, positive_int


def tool_definitions():
    # These versions support Haiku 4.5 as well as the larger models.
    return [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch"},
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]


async def stream_message(client, request, attempts=4):
    """Retry interrupted streams without committing partial assistant content."""
    for attempt in range(attempts):
        try:
            async with client.messages.stream(**request) as stream:
                message = await stream.get_final_message()
            if message.stop_reason is None:
                raise RuntimeError("stream ended without a terminal stop reason")
            return message.model_dump(mode="json", exclude_none=True), attempt
        except (anthropic.APIConnectionError, anthropic.APIStatusError,
                httpx.TransportError) as error:
            if isinstance(error, anthropic.APIStatusError) and error.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise
            if attempt + 1 == attempts:
                raise RuntimeError(f"stream failed after {attempts} attempts: {type(error).__name__}: {error}") from error
            await asyncio.sleep(min(2 ** attempt, 30))


def extract_output(response, fields):
    if response.get("stop_reason") == "refusal":
        return {field["fieldname"]: "" for field in fields}
    if response.get("stop_reason") != "end_turn":
        raise ValueError(f"incomplete response: {response.get('stop_reason')}")
    texts = [block["text"] for block in response.get("content", []) if block.get("type") == "text"]
    # Last text block is the answer, not preliminary research narration.
    output = json_content(texts[-1] if texts else "")
    names = {field["fieldname"] for field in fields}
    if set(output) != names or any(not isinstance(value, str) for value in output.values()):
        raise ValueError("final JSON must contain exactly the requested string fields")
    return output


async def call_api(client, item, *, model, effort=None, max_tokens=64000,
                   max_continuations=12, attempts=4):
    if "haiku" in model and effort is not None:
        raise ValueError("Haiku does not support the effort parameter; omit --effort")
    request = {
        "model": model, "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(item)}],
        "tools": tool_definitions(),
    }
    if effort is not None:
        request["output_config"] = {"effort": effort}
        request["thinking"] = {"type": "adaptive"}
    responses = []
    retries = 0
    for continuation in range(max_continuations + 1):
        response, retried = await stream_message(client, request, attempts)
        responses.append(response)
        retries += retried
        if response.get("stop_reason") != "pause_turn":
            output = extract_output(response, item["fields"])
            return output, {
                "refused": response.get("stop_reason") == "refusal",
                "continuations": continuation, "transport_retries": retries,
                "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in responses),
                "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in responses),
                "web_searches": sum(1 for r in responses for b in r.get("content", []) if b.get("type") == "server_tool_use" and b.get("name") == "web_search"),
                "responses": responses,
            }
        if continuation == max_continuations:
            raise RuntimeError(f"pause_turn continuation limit reached ({max_continuations})")
        request["messages"].append({"role": "assistant", "content": response["content"]})
        container_id = response.get("container", {}).get("id")
        if container_id:
            request["container"] = container_id
    raise AssertionError("unreachable")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--people", type=positive_int)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    parser.add_argument("--max-tokens", type=positive_int, default=64000)
    parser.add_argument("--max-continuations", type=positive_int, default=12)
    parser.add_argument("--attempts", type=positive_int, default=4)
    parser.add_argument("--timeout", type=positive_int, default=3600)
    args = parser.parse_args()
    if "haiku" in args.model and args.effort:
        parser.error("Haiku does not support --effort")
    people = load_people(args.people)
    config = {**vars(args), "tools": tool_definitions(), "anthropic_sdk": anthropic.__version__}
    runner = EvalRunner(f"claude_{args.model}_{args.effort or 'default'}", config)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with anthropic.AsyncAnthropic(max_retries=0, timeout=httpx.Timeout(300, connect=30)) as client:
        async def process(item):
            async with semaphore:
                started = time.monotonic()
                try:
                    async with asyncio.timeout(args.timeout):
                        output, metadata = await call_api(client, item, model=args.model, effort=args.effort, max_tokens=args.max_tokens, max_continuations=args.max_continuations, attempts=args.attempts)
                    await runner.record(item, output, time.monotonic() - started, metadata)
                except Exception as error:
                    await runner.record_error(item, time.monotonic() - started, RuntimeError(f"{type(error).__name__}: {error}"))
        await asyncio.gather(*(process(item) for item in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
