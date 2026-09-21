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
    async def test_pause_preserves_container_and_entire_content(self):
        calls = []
        pause = reply("pause_turn", container={"id": "container_123"})
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

    async def test_continuation_limit_is_error_not_blank(self):
        with patch.object(claude, "stream_message", AsyncMock(return_value=(reply("pause_turn"), 0))):
            with self.assertRaisesRegex(RuntimeError, "continuation limit"):
                await claude.call_api(None, ITEM, model="claude-opus-5", max_continuations=1)

    def test_refusal_is_valid_missing(self):
        self.assertEqual(claude.extract_output(reply("refusal", "Declined"), ITEM["fields"]), {"employer": ""})

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
            with self.assertRaises(claude.ClaudeRunError) as raised:
                await claude.call_api(None, ITEM, model="claude-sonnet-5")
        self.assertEqual(raised.exception.responses, [response])
        self.assertEqual(mocked.await_count, 1)

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
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 8}},
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
                result, retries = await claude.stream_message(client, {"model": "claude-opus-5", "max_tokens": 100, "messages": [{"role": "user", "content": "test"}]})
        self.assertEqual(retries, 1)
        self.assertEqual(claude.extract_output(result, ITEM["fields"]), {"employer": "Acme"})


if __name__ == "__main__":
    unittest.main()
