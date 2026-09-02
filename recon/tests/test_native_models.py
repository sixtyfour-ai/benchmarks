import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import native_models  # noqa: E402


FIELDS = [
    {"fieldname": "employer", "description": "Current employer"},
    {"fieldname": "hometown", "description": "Childhood hometown"},
]
ITEM = {"person_info": "Ada Example, Engineer at Acme", "fields": FIELDS}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class NativeModelTests(unittest.IsolatedAsyncioTestCase):
    def test_positive_int_rejects_zero_and_negative_values(self):
        self.assertEqual(native_models.positive_int("7"), 7)
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(
                native_models.argparse.ArgumentTypeError
            ):
                native_models.positive_int(value)

    def test_search_round_limit_matches_provider_runtime(self):
        self.assertEqual(native_models.resolve_max_search_rounds("kimi", None), 10)
        self.assertEqual(native_models.resolve_max_search_rounds("kimi", 4), 4)
        self.assertEqual(native_models.resolve_max_search_rounds("glm", None), 10)
        self.assertEqual(native_models.resolve_max_search_rounds("glm", 4), 4)
        self.assertIsNone(native_models.resolve_max_search_rounds("deepseek", None))
        with self.assertRaisesRegex(SystemExit, "DeepSeek controls"):
            native_models.resolve_max_search_rounds("deepseek", 4)

    def test_extracts_embedded_json_without_interpreting_prose(self):
        parsed = native_models._json_content(
            'I researched the person. Final answer:\n'
            '{"employer":"Acme","hometown":"London"}\n'
            'Sources were checked.'
        )
        self.assertEqual(parsed, {"employer": "Acme", "hometown": "London"})

        with self.assertRaisesRegex(ValueError, "no JSON object"):
            native_models._json_content("The employer appears to be Acme.")

        output, valid = native_models._terminal_output("I cannot provide that information.")
        self.assertEqual(output, {})
        self.assertFalse(valid)

    async def test_kimi_round_trips_complete_tool_message_and_strict_schema(self):
        assistant = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should search.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "$web_search",
                        "arguments": '{"query":"Ada Example Acme"}',
                    },
                }
            ],
        }
        responses = [
            FakeResponse(
                {
                    "model": "kimi-k3",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "choices": [{"finish_reason": "tool_calls", "message": assistant}],
                }
            ),
            FakeResponse(
                {
                    "model": "kimi-k3",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"employer":"Acme","hometown":""}',
                            },
                        }
                    ],
                }
            ),
        ]
        requests = []

        async def fake_post(_client, _endpoint, **kwargs):
            requests.append(kwargs)
            return responses.pop(0)

        with patch.object(native_models, "post_with_retry", new=fake_post):
            output, metadata = await native_models.call_kimi(
                object(),
                ITEM,
                endpoint="https://example.test/chat",
                api_key="secret",
                model="kimi-k3",
                reasoning="max",
                max_search_rounds=1,
            )

        self.assertEqual(output, {"employer": "Acme", "hometown": ""})
        self.assertEqual(metadata["web_searches"], 1)
        self.assertTrue(metadata["terminal_format_valid"])
        self.assertEqual(metadata["input_tokens"], 30)
        self.assertEqual(requests[1]["json"]["messages"][2], assistant)
        self.assertEqual(
            json.loads(requests[1]["json"]["messages"][3]["content"]),
            {"query": "Ada Example Acme"},
        )
        response_format = requests[0]["json"]["response_format"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["required"],
            ["employer", "hometown"],
        )
        self.assertNotIn("tools", requests[1]["json"])
        self.assertIn(
            "budget is exhausted", requests[1]["json"]["messages"][-1]["content"]
        )

    async def test_deepseek_uses_server_side_search_and_requested_reasoning(self):
        response = FakeResponse(
            {
                "status": "completed",
                "model": "deepseek-v4-flash",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "output": [
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"employer":"Acme","hometown":"London"}',
                            }
                        ],
                    },
                ],
            }
        )
        fake_post = AsyncMock(return_value=response)
        with patch.object(native_models, "post_with_retry", new=fake_post):
            output, metadata = await native_models.call_deepseek(
                object(),
                ITEM,
                endpoint="https://example.test/responses",
                api_key="secret",
                model="deepseek-v4-flash",
                reasoning="none",
                max_search_rounds=None,
            )

        self.assertEqual(output["hometown"], "London")
        self.assertEqual(metadata["web_searches"], 1)
        body = fake_post.await_args.kwargs["json"]
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertEqual(body["tools"], [{"type": "web_search"}])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertFalse(metadata["terminal_repaired"])

    async def test_deepseek_repairs_prose_with_tool_free_compilation(self):
        research_output = [
            {"type": "web_search_call", "status": "completed", "id": "ws_1"},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Ada works at Acme, but I could not verify a hometown.",
                    }
                ],
            },
        ]
        responses = [
            FakeResponse(
                {
                    "status": "completed",
                    "model": "deepseek-v4-flash",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "output": research_output,
                }
            ),
            FakeResponse(
                {
                    "status": "completed",
                    "model": "deepseek-v4-flash",
                    "usage": {"input_tokens": 40, "output_tokens": 10},
                    "output_text": '{"employer":"Acme","hometown":""}',
                    "output": [],
                }
            ),
        ]
        requests = []

        async def fake_post(_client, _endpoint, **kwargs):
            requests.append(kwargs)
            return responses.pop(0)

        with patch.object(native_models, "post_with_retry", new=fake_post):
            output, metadata = await native_models.call_deepseek(
                object(),
                ITEM,
                endpoint="https://example.test/responses",
                api_key="secret",
                model="deepseek-v4-flash",
                reasoning="none",
                max_search_rounds=None,
            )

        self.assertEqual(output, {"employer": "Acme", "hometown": ""})
        self.assertTrue(metadata["terminal_format_valid"])
        self.assertTrue(metadata["terminal_repaired"])
        self.assertEqual(metadata["input_tokens"], 140)
        self.assertEqual(metadata["web_searches"], 1)
        repair = requests[1]["json"]
        self.assertNotIn("tools", repair)
        self.assertIn(research_output[0], repair["input"])
        self.assertEqual(repair["input"][-1]["content"], native_models.FINALIZE_PROMPT)

    async def test_glm_uses_zai_search_and_thinking_toggle(self):
        assistant = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should search for the employer.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"Ada Example Acme"}',
                    },
                }
            ],
        }
        responses = [
            FakeResponse(
                {
                    "model": "glm-5.3",
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                    "choices": [
                        {"finish_reason": "tool_calls", "message": assistant}
                    ],
                }
            ),
            FakeResponse(
                {
                    "search_result": [
                        {
                            "title": "Ada Example",
                            "content": "Ada works at Acme.",
                            "link": "https://example.test/ada",
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "model": "glm-5.3",
                    "usage": {"prompt_tokens": 80, "completion_tokens": 12},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"employer":"Acme","hometown":""}'
                            },
                        }
                    ],
                }
            ),
        ]
        requests = []

        async def fake_post(_client, request_endpoint, **kwargs):
            requests.append((request_endpoint, kwargs))
            return responses.pop(0)

        with patch.object(native_models, "post_with_retry", new=fake_post):
            output, metadata = await native_models.call_glm(
                object(),
                ITEM,
                endpoint="https://example.test/chat",
                search_endpoint="https://example.test/search",
                api_key="secret",
                model="glm-5.3",
                reasoning="max",
                max_search_rounds=1,
            )

        self.assertEqual(output["employer"], "Acme")
        self.assertEqual(metadata["web_searches"], 1)
        self.assertEqual(metadata["search_results"], 1)
        self.assertEqual(metadata["input_tokens"], 130)
        body = requests[0][1]["json"]
        self.assertEqual(
            body["thinking"], {"type": "enabled", "clear_thinking": False}
        )
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["tools"][0]["function"]["name"], "web_search")
        self.assertEqual(requests[1][0], "https://example.test/search")
        self.assertEqual(requests[1][1]["json"]["search_query"], "Ada Example Acme")
        continuation = requests[2][1]["json"]
        self.assertEqual(
            continuation["thinking"], {"type": "enabled", "clear_thinking": False}
        )
        self.assertEqual(continuation["messages"][2], assistant)
        self.assertEqual(continuation["messages"][3]["tool_call_id"], "call_1")
        self.assertNotIn("tools", continuation)
        self.assertIn(
            "budget is exhausted", continuation["messages"][-1]["content"]
        )


if __name__ == "__main__":
    unittest.main()
