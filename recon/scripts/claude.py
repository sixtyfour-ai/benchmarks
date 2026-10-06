"""RECON through Anthropic's native search and code-execution tools.

Requires anthropic, httpx and openai; ANTHROPIC_API_KEY and OPENAI_API_KEY.
Example: python scripts/claude.py --model claude-sonnet-5 --effort xhigh

The runner resumes every ``pause_turn`` until Claude finishes, and each response
may use the model's full output limit. Server-side fallbacks are not enabled, so
every answer comes from the model named on the command line.
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


class ClaudeRunError(RuntimeError):
    """Execution failure retaining completed native responses for diagnosis."""

    def __init__(self, message, responses):
        super().__init__(message)
        self.responses = responses


class TerminalFormatError(ValueError):
    """Completed model response that does not supply the requested answer JSON."""


# Models with adaptive thinking, server-side compaction and the dynamic-filtering
# web tools. Others (Haiku 4.5) use the basic tool versions and context editing.
CURRENT_MODELS = {
    "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6",
}


def context_options(model, enabled=True):
    """Use provider-managed context; never rewrite local history or prompts."""
    if not enabled:
        return {}
    if model in CURRENT_MODELS:
        return {
            "betas": ["compact-2026-01-12"],
            "context_management": {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]},
        }
    # Haiku supports context editing, but not native summary compaction.
    return {
        "betas": ["context-management-2025-06-27"],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 100000}, "keep": {"type": "tool_uses", "value": 3}}]},
    }


def token_usage(responses, key):
    # Compaction billing is omitted from top-level usage; iterations includes it.
    return sum(sum(part.get(key, 0) for part in (r.get("usage", {}).get("iterations") or [r.get("usage", {})])) for r in responses)


def tool_definitions(model):
    """Native research tools for the model."""
    if model in CURRENT_MODELS:
        # These versions run code execution internally to filter results, so a
        # separate code_execution tool would add a second sandbox.
        return [
            {"type": "web_search_20260209", "name": "web_search"},
            {"type": "web_fetch_20260209", "name": "web_fetch"},
        ]
    return [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch"},
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]


async def stream_message(client, request, attempts=4):
    """Retry interrupted streams without committing partial assistant content."""
    for attempt in range(attempts):
        try:
            messages_api = client.beta.messages if request.get("betas") else client.messages
            async with messages_api.stream(**request) as stream:
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
    if any(block.get("type") == "compaction" and not block.get("content") for block in response.get("content", [])):
        raise ValueError("provider returned an empty compaction block; task result is unavailable")
    if response.get("stop_reason") not in {"end_turn", "refusal"}:
        raise ValueError(f"incomplete response: {response.get('stop_reason')}")
    texts = [block["text"] for block in response.get("content", []) if block.get("type") == "text"]
    # Last text block is the answer, not preliminary research narration.
    try:
        output = json_content(texts[-1] if texts else "")
    except ValueError as error:
        raise TerminalFormatError(str(error)) from error
    names = {field["fieldname"] for field in fields}
    if set(output) != names or any(not isinstance(value, str) for value in output.values()):
        raise TerminalFormatError("final JSON must contain exactly the requested string fields")
    return output


def result_metadata(responses, *, continuations, retries, terminal_status,
                    terminal_format_valid, api_refusal=False,
                    continuation_budget_exhausted=False):
    return {
        # These are API/runtime signals, not semantic classifiers of prose.
        "api_refusal": api_refusal,
        "continuation_budget_exhausted": continuation_budget_exhausted,
        "terminal_format_valid": terminal_format_valid,
        "terminal_status": terminal_status,
        "continuations": continuations, "transport_retries": retries,
        "input_tokens": token_usage(responses, "input_tokens"),
        "output_tokens": token_usage(responses, "output_tokens"),
        "cache_read_input_tokens": token_usage(responses, "cache_read_input_tokens"),
        "cache_creation_input_tokens": token_usage(responses, "cache_creation_input_tokens"),
        "context_edits": [r.get("context_management", {}) for r in responses],
        "compactions": sum(1 for r in responses for b in r.get("content", []) if b.get("type") == "compaction"),
        "web_searches": sum(1 for r in responses for b in r.get("content", []) if b.get("type") == "server_tool_use" and b.get("name") == "web_search"),
        "responses": responses,
    }


async def call_api(client, item, *, model, effort=None, max_tokens=64000,
                   max_continuations=None, attempts=4, context_management=True,
                   thinking_budget=None):
    if "haiku" in model and effort is not None:
        raise ValueError("Haiku does not support the effort parameter; omit --effort")
    request = {
        "model": model, "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(item)}],
        "tools": tool_definitions(model),
        "cache_control": {"type": "ephemeral"},
        **context_options(model, context_management),
    }
    if effort is not None:
        request["output_config"] = {"effort": effort}
        request["thinking"] = {"type": "adaptive"}
    elif thinking_budget is not None:
        request["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        # Without this beta, budgeted thinking happens only before the first tool call.
        request["betas"] = [*request.get("betas", []), "interleaved-thinking-2025-05-14"]
    responses = []
    retries = 0
    continuation = 0
    while True:
        try:
            response, retried = await stream_message(client, request, attempts)
        except Exception as error:
            raise ClaudeRunError(f"{type(error).__name__}: {error}", responses) from error
        responses.append(response)
        retries += retried
        if response.get("stop_reason") != "pause_turn":
            format_valid = True
            try:
                output = extract_output(response, item["fields"])
            except TerminalFormatError:
                output = {field["fieldname"]: "" for field in item["fields"]}
                format_valid = False
            except ValueError as error:
                raise ClaudeRunError(str(error), responses) from error
            api_refusal = response.get("stop_reason") == "refusal"
            return output, result_metadata(
                responses, continuations=continuation, retries=retries,
                api_refusal=api_refusal, terminal_format_valid=format_valid,
                terminal_status="api_refusal" if api_refusal else ("structured_answer" if format_valid else "unstructured_nonanswer"),
            )
        if max_continuations is not None and continuation == max_continuations:
            output = {field["fieldname"]: "" for field in item["fields"]}
            return output, result_metadata(
                responses, continuations=continuation, retries=retries,
                terminal_status="continuation_budget_exhausted",
                terminal_format_valid=False,
                continuation_budget_exhausted=True,
            )
        # pause_turn: the server paused its own tool loop. Resend the turn
        # unchanged and it resumes; no extra user message is added.
        request["messages"].append({"role": "assistant", "content": response["content"]})
        container_id = response.get("container", {}).get("id")
        if container_id:
            request["container"] = container_id
        continuation += 1


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--thinking-budget", type=positive_int,
                        help="extended thinking budget for models without adaptive thinking (Haiku 4.5)")
    parser.add_argument("--people", type=positive_int)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    parser.add_argument("--max-tokens", type=positive_int,
                        help="per-response output limit; defaults to the model's maximum")
    parser.add_argument("--max-continuations", type=positive_int,
                        help="optional cap on pause_turn resumptions; unlimited by default")
    parser.add_argument("--attempts", type=positive_int, default=4)
    parser.add_argument("--timeout", type=positive_int,
                        help="optional per-person wall-clock limit in seconds; none by default")
    parser.add_argument("--context-management", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if "haiku" in args.model and args.effort:
        parser.error("Haiku does not support --effort")
    if args.model in CURRENT_MODELS and args.thinking_budget:
        parser.error(f"{args.model} uses adaptive thinking; set --effort instead of --thinking-budget")
    people = load_people(args.people)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with anthropic.AsyncAnthropic(max_retries=0, timeout=httpx.Timeout(300, connect=30)) as client:
        if args.max_tokens is None:
            args.max_tokens = (await client.models.retrieve(args.model)).max_tokens
        if args.thinking_budget is not None and args.thinking_budget >= args.max_tokens:
            parser.error("--thinking-budget must be below the model's output limit")
        config = {**vars(args), "context_options": context_options(args.model, args.context_management), "tools": tool_definitions(args.model), "anthropic_sdk": anthropic.__version__}
        runner = EvalRunner(f"claude_{args.model}_{args.effort or 'default'}", config)

        async def process(item):
            async with semaphore:
                started = time.monotonic()
                try:
                    async with asyncio.timeout(args.timeout):
                        output, metadata = await call_api(client, item, model=args.model, effort=args.effort, max_tokens=args.max_tokens, max_continuations=args.max_continuations, attempts=args.attempts, context_management=args.context_management, thinking_budget=args.thinking_budget)
                    await runner.record(item, output, time.monotonic() - started, metadata)
                except Exception as error:
                    metadata = {"responses": error.responses} if isinstance(error, ClaudeRunError) else None
                    await runner.record_error(item, time.monotonic() - started, RuntimeError(f"{type(error).__name__}: {error}"), metadata=metadata)
        await asyncio.gather(*(process(item) for item in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
