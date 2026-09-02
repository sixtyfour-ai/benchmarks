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
                max_search_rounds=10,
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
                max_search_rounds=10,
            )

        self.assertEqual(output["hometown"], "London")
        self.assertEqual(metadata["web_searches"], 1)
        body = fake_post.await_args.kwargs["json"]
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertEqual(body["tools"], [{"type": "web_search"}])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")

    async def test_glm_uses_zai_search_and_thinking_toggle(self):
        response = FakeResponse(
            {
                "model": "glm-5.3",
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                "web_search": [{"link": "https://example.test"}],
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"employer":"Acme","hometown":""}'
                        },
                    }
                ],
            }
        )
        fake_post = AsyncMock(return_value=response)
        with patch.object(native_models, "post_with_retry", new=fake_post):
            output, metadata = await native_models.call_glm(
                object(),
                ITEM,
                endpoint="https://example.test/chat",
                api_key="secret",
                model="glm-5.3",
                reasoning="max",
                max_search_rounds=10,
            )

        self.assertEqual(output["employer"], "Acme")
        self.assertEqual(metadata["web_searches"], 1)
        body = fake_post.await_args.kwargs["json"]
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["tools"][0]["type"], "web_search")
        self.assertTrue(body["tools"][0]["web_search"]["enable"])


if __name__ == "__main__":
    unittest.main()
