import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import claude
httpx = claude.httpx

ITEM = {"person_info": "Example person", "fields": [{"fieldname": "employer", "description": "First employer", "answer": "SECRET_REFERENCE"}]}


def reply(stop="end_turn", text='{"employer":"Acme"}', **extra):
    return {"stop_reason": stop, "content": [{"type": "text", "text": text}], **extra}


class ClaudeTests(unittest.IsolatedAsyncioTestCase):
    def test_context_strategy_matches_model_support(self):
        self.assertEqual(claude.context_options("claude-haiku-4-5-20251001")["context_management"]["edits"][0]["type"], "clear_tool_uses_20250919")
        self.assertEqual(claude.context_options("claude-opus-5")["context_management"]["edits"][0]["type"], "compact_20260112")
        self.assertEqual(claude.context_options("claude-sonnet-5", False), {})

    def test_token_usage_includes_compaction_without_double_counting(self):
        response = {"usage": {"input_tokens": 12, "output_tokens": 3, "iterations": [{"type": "compaction", "input_tokens": 100, "output_tokens": 10}, {"type": "message", "input_tokens": 12, "output_tokens": 3}]}}
        self.assertEqual(claude.token_usage([response], "input_tokens"), 112)
        self.assertEqual(claude.token_usage([response], "output_tokens"), 13)

    async def test_pause_preserves_container_and_entire_content(self):
        calls = []
        pause = reply("pause_turn", container={"id": "container_123"})
        pause["content"].insert(0, {"type": "compaction", "content": "Retained research summary"})
        pause["content"].append({"type": "server_tool_use", "id": "srv_1", "name": "bash_code_execution", "input": {"command": "echo ok"}})
        responses = [pause, reply()]
        async def stream(client, request, attempts):
            calls.append(copy.deepcopy(request))
            return responses.pop(0), 0
        with patch.object(claude, "stream_message", stream):
            output, meta = await claude.call_api(None, ITEM, model="claude-opus-5", effort="xhigh")
        self.assertEqual(output, {"employer": "Acme"})
        self.assertEqual(calls[1]["container"], "container_123")
        self.assertEqual(calls[1]["messages"][-1]["content"], pause["content"])
        self.assertEqual(calls[0]["tools"], calls[1]["tools"])
        self.assertNotIn("SECRET_REFERENCE", json.dumps(calls))
        self.assertEqual(meta["continuations"], 1)
        self.assertEqual(meta["compactions"], 1)
        self.assertEqual(calls[0]["context_management"], calls[1]["context_management"])

    async def test_continuation_limit_is_explicit_scored_blank(self):
        with patch.object(claude, "stream_message", AsyncMock(return_value=(reply("pause_turn"), 0))):
            output, metadata = await claude.call_api(None, ITEM, model="claude-opus-5", max_continuations=1)
        self.assertEqual(output, {"employer": ""})
        self.assertEqual(metadata["terminal_status"], "continuation_budget_exhausted")
        self.assertTrue(metadata["continuation_budget_exhausted"])
        self.assertFalse(metadata["api_refusal"])
        self.assertFalse(metadata["terminal_format_valid"])
        self.assertEqual(metadata["continuations"], 1)
        self.assertEqual(len(metadata["responses"]), 2)

    async def test_refusal_is_valid_missing(self):
        response = reply("refusal", "Declined")
        with patch.object(claude, "stream_message", AsyncMock(return_value=(response, 0))):
            output, metadata = await claude.call_api(None, ITEM, model="claude-opus-5")
        self.assertEqual(output, {"employer": ""})
        self.assertTrue(metadata["api_refusal"])
        self.assertFalse(metadata["terminal_format_valid"])
        self.assertEqual(metadata["terminal_status"], "api_refusal")

    def test_failed_compaction_is_not_scored_as_task_refusal(self):
        response = {"stop_reason": "refusal", "content": [{"type": "compaction"}]}
        with self.assertRaisesRegex(ValueError, "empty compaction"):
            claude.extract_output(response, ITEM["fields"])

    def test_empty_strings_are_valid(self):
        self.assertEqual(claude.extract_output(reply(text='{"employer":""}'), ITEM["fields"]), {"employer": ""})

    def test_invalid_terminal_is_error(self):
        for response in [reply(text=""), reply(text="I refuse"), reply(text="{}"), reply(text='{"employer":12}'), reply("max_tokens"), reply("tool_use")]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                claude.extract_output(response, ITEM["fields"])

    async def test_haiku_rejects_effort_before_request(self):
        with self.assertRaisesRegex(ValueError, "Haiku"):
            await claude.call_api(None, ITEM, model="claude-haiku-4-5-20251001", effort="high")

    async def test_prose_failure_retains_response_without_repair(self):
        response = reply(text="I cannot provide this information.")
        mocked = AsyncMock(return_value=(response, 0))
        with patch.object(claude, "stream_message", mocked):
            output, metadata = await claude.call_api(None, ITEM, model="claude-sonnet-5")
        self.assertEqual(metadata["responses"], [response])
        self.assertEqual(output, {"employer": ""})
        self.assertFalse(metadata["terminal_format_valid"])
        self.assertFalse(metadata["api_refusal"])
        self.assertEqual(metadata["terminal_status"], "unstructured_nonanswer")
        self.assertEqual(mocked.await_count, 1)

    async def test_terminal_schema_mismatches_are_missing_without_repair(self):
        for text in ['{}', '{"employer":12}', '{"employer":"Acme","extra":"bad"}']:
            response = reply(text=text)
            mocked = AsyncMock(return_value=(response, 0))
            with self.subTest(text=text), patch.object(claude, "stream_message", mocked):
                output, metadata = await claude.call_api(None, ITEM, model="claude-sonnet-5")
            self.assertEqual(output, {"employer": ""})
            self.assertFalse(metadata["terminal_format_valid"])
            self.assertEqual(metadata["responses"], [response])
            self.assertEqual(mocked.await_count, 1)

    async def test_valid_terminal_records_validity(self):
        with patch.object(claude, "stream_message", AsyncMock(return_value=(reply(), 0))):
            output, metadata = await claude.call_api(None, ITEM, model="claude-sonnet-5")
        self.assertEqual(output, {"employer": "Acme"})
        self.assertTrue(metadata["terminal_format_valid"])
        self.assertEqual(metadata["terminal_status"], "structured_answer")

    async def test_continuation_failure_retains_earlier_responses(self):
        pause = reply("pause_turn")
        with patch.object(claude, "stream_message", AsyncMock(side_effect=[(pause, 0), RuntimeError("stream failed")])):
            with self.assertRaises(claude.ClaudeRunError) as raised:
                await claude.call_api(None, ITEM, model="claude-opus-5")
        self.assertEqual(raised.exception.responses, [pause])

    async def test_actual_sdk_assembles_fragmented_stream(self):
        events = [
            {"type": "message_start", "message": {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"employer":'}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '"Acme"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 8}, "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 4, "cleared_input_tokens": 50000}]}},
            {"type": "message_stop"},
        ]
        payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        class Bytes(httpx.AsyncByteStream):
            def __init__(self, interrupt=False):
                self.interrupt = interrupt
            async def __aiter__(self):
                for offset in range(0, len(payload), 17):
                    if self.interrupt and offset > len(payload) // 2:
                        raise httpx.ReadError("synthetic interrupted stream")
                    yield payload[offset:offset + 17]
        attempts = 0
        async def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Bytes(interrupt=attempts == 1))
        async with anthropic.AsyncAnthropic(api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch.object(claude.asyncio, "sleep", AsyncMock()):
                result, retries = await claude.stream_message(client, {"model": "claude-haiku-4-5-20251001", "max_tokens": 100, "messages": [{"role": "user", "content": "test"}], **claude.context_options("claude-haiku-4-5-20251001")})
        self.assertEqual(retries, 1)
        self.assertEqual(claude.extract_output(result, ITEM["fields"]), {"employer": "Acme"})
        self.assertEqual(result["context_management"]["applied_edits"][0]["cleared_tool_uses"], 4)


if __name__ == "__main__":
    unittest.main()
