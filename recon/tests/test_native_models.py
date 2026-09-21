import asyncio
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import deepseek  # noqa: E402
import glm  # noqa: E402
import kimi  # noqa: E402
import native_model_common as common  # noqa: E402


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
    async def test_chat_runners_repair_format_once_without_research_tools(self):
        for provider in (kimi, glm):
            with self.subTest(provider=provider.CONFIG.name):
                prose = {"role": "assistant", "content": "Ada works at Acme."}
                responses = [
                    FakeResponse({"choices": [{"finish_reason": "stop", "message": prose}]}),
                    FakeResponse({"choices": [{"finish_reason": "stop", "message": {
                        "role": "assistant", "content": '{"employer":"Acme","hometown":""}'
                    }}]}),
                ]
                fake_post = AsyncMock(side_effect=responses)
                with patch.object(provider, "post_with_retry", fake_post):
                    output, metadata = await getattr(provider, f"call_{provider.CONFIG.name}")(
                        object(), ITEM, api_key="secret", model=provider.CONFIG.default_model,
                        reasoning="max", max_search_rounds=10,
                    )
                self.assertEqual(output["employer"], "Acme")
                self.assertEqual(metadata["terminal_repaired"], provider is glm)
                repair = fake_post.await_args_list[1].kwargs["json"]
                self.assertEqual(repair["tool_choice"], "none")
                self.assertNotIn("tools", repair)
                self.assertIn(prose, repair["messages"])
                self.assertEqual(fake_post.await_count, 2)

    async def test_chat_runners_do_not_score_repeated_format_failure_as_abstention(self):
        for provider in (kimi, glm):
            with self.subTest(provider=provider.CONFIG.name):
                response = FakeResponse({"choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Still not JSON"
                }}]})
                fake_post = AsyncMock(return_value=response)
                with patch.object(provider, "post_with_retry", fake_post):
                    with self.assertRaisesRegex(RuntimeError, "no JSON object"):
                        await getattr(provider, f"call_{provider.CONFIG.name}")(
                            object(), ITEM, api_key="secret", model=provider.CONFIG.default_model,
                            reasoning="max", max_search_rounds=10,
                        )
                self.assertEqual(fake_post.await_count, 3 if provider is kimi else 2)

    async def test_chat_runners_bound_unexpected_terminal_tool_calls(self):
        for provider in (kimi, glm):
            with self.subTest(provider=provider.CONFIG.name):
                unexpected = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "terminal_call", "type": "function", "function": {
                        "name": "unexpected", "arguments": "{}"
                    }
                }]}
                # Exercise only compilation: no provider tools may be dispatched.
                response = FakeResponse({"choices": [{"finish_reason": "tool_calls", "message": unexpected}]})
                fake_post = AsyncMock(return_value=response)
                with patch.object(provider, "post_with_retry", fake_post):
                    with self.assertRaisesRegex(RuntimeError, "tool during terminal"):
                        await getattr(provider, f"call_{provider.CONFIG.name}")(
                            object(), ITEM, api_key="secret", model=provider.CONFIG.default_model,
                            reasoning="max", max_search_rounds=0,
                        )
                self.assertEqual(fake_post.await_count, 2)
                for call in fake_post.await_args_list:
                    body = call.kwargs["json"]
                    self.assertEqual(body["tool_choice"], "none")
                    self.assertNotIn("tools", body)
                    self.assertNotIn(unexpected, body["messages"])

    def test_positive_int_rejects_zero_and_negative_values(self):
        self.assertEqual(common.positive_int("7"), 7)
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(
                common.argparse.ArgumentTypeError
            ):
                common.positive_int(value)

    def test_search_round_limit_is_only_on_client_search_runners(self):
        self.assertEqual(kimi.CONFIG.default_search_rounds, 10)
        self.assertEqual(glm.CONFIG.default_search_rounds, 10)
        self.assertIsNone(deepseek.CONFIG.default_search_rounds)

        with patch.object(sys, "argv", ["kimi.py", "--max-search-rounds", "4"]):
            self.assertEqual(common.parse_args(kimi.CONFIG).max_search_rounds, 4)
        with patch.object(sys, "argv", ["deepseek.py", "--max-search-rounds", "4"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                common.parse_args(deepseek.CONFIG)

    def test_extracts_embedded_json_without_interpreting_prose(self):
        parsed = common.json_content(
            'I researched the person. Final answer:\n'
            '{"employer":"Acme","hometown":"London"}\n'
            'Sources were checked.'
        )
        self.assertEqual(parsed, {"employer": "Acme", "hometown": "London"})

        with self.assertRaisesRegex(ValueError, "no JSON object"):
            common.json_content("The employer appears to be Acme.")

        output, valid = common.terminal_output("I cannot provide that information.")
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
                        "name": "web_search",
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
            FakeResponse({"search_results": [{"url": "https://example.test", "chunks": [{"text": "Ada works at Acme."}]}]}),
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

        with patch.object(kimi, "post_with_retry", new=fake_post):
            output, metadata = await kimi.call_kimi(
                object(),
                ITEM,
                api_key="secret",
                model="kimi-k3",
                reasoning="max",
                max_search_rounds=1,
            )

        self.assertEqual(output, {"employer": "Acme", "hometown": ""})
        self.assertEqual(metadata["web_searches"], 1)
        self.assertTrue(metadata["terminal_format_valid"])
        self.assertEqual(metadata["input_tokens"], 30)
        self.assertEqual(requests[1]["json"], {"text_query": "Ada Example Acme", "limit": 10, "timeout_seconds": 30})
        self.assertEqual(metadata["search_backend"], "moonshot/search_pro")
        self.assertEqual(metadata["search_results"], 1)
        self.assertEqual(requests[2]["json"]["messages"][2], assistant)
        self.assertEqual(
            json.loads(requests[2]["json"]["messages"][3]["content"]),
            [{"url": "https://example.test", "chunks": [{"text": "Ada works at Acme."}]}],
        )
        self.assertNotIn("response_format", requests[0]["json"])
        self.assertEqual(requests[0]["json"]["tools"], [kimi.WEB_SEARCH_TOOL])
        self.assertEqual(requests[0]["json"]["tool_choice"], "required")
        self.assertEqual(requests[2]["json"]["tool_choice"], "none")
        self.assertEqual(metadata["search_policy"], "first_turn_required_then_auto")
        response_format = requests[2]["json"]["response_format"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["required"],
            ["employer", "hometown"],
        )
        self.assertNotIn("tools", requests[2]["json"])
        self.assertIn(
            "budget is exhausted", requests[2]["json"]["messages"][-1]["content"]
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
        with patch.object(deepseek, "post_with_retry", new=fake_post):
            output, metadata = await deepseek.call_deepseek(
                object(),
                ITEM,
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

    async def test_kimi_rejects_invalid_search_arguments_and_response(self):
        for arguments, search_response, expected_calls in (
            ('{"query":""}', None, 1),
            ('{"query":"Ada Example"}', {}, 2),
        ):
            with self.subTest(arguments=arguments):
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function", "function": {
                        "name": "web_search", "arguments": arguments,
                    }
                }]}
                responses = [FakeResponse({"choices": [{"finish_reason": "tool_calls", "message": message}]})]
                if search_response is not None:
                    responses.append(FakeResponse(search_response))
                fake_post = AsyncMock(side_effect=responses)
                with patch.object(kimi, "post_with_retry", fake_post):
                    with self.assertRaises(RuntimeError):
                        await kimi.call_kimi(object(), ITEM, api_key="secret", model="kimi-k3",
                                             reasoning="max", max_search_rounds=10)
                self.assertEqual(fake_post.await_count, expected_calls)

    async def test_kimi_requires_only_first_research_turn(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "web_search", "arguments": '{"query":"Ada Example"}'
            }
        }]}
        final = {"role": "assistant", "content": '{"employer":"","hometown":""}'}
        fake_post = AsyncMock(side_effect=[
            FakeResponse({"choices": [{"finish_reason": "tool_calls", "message": message}]}),
            FakeResponse({"search_results": []}),
            FakeResponse({"choices": [{"finish_reason": "stop", "message": final}]}),
            FakeResponse({"choices": [{"finish_reason": "stop", "message": final}]}),
        ])
        with patch.object(kimi, "post_with_retry", fake_post):
            await kimi.call_kimi(object(), ITEM, api_key="secret", model="kimi-k3",
                                 reasoning="max", max_search_rounds=10)
        chat_requests = [call.kwargs["json"] for call in fake_post.await_args_list
                         if call.args[1] == kimi.ENDPOINT]
        self.assertEqual([request["tool_choice"] for request in chat_requests],
                         ["required", "auto", "none"])
        self.assertNotIn("response_format", chat_requests[1])
        self.assertIn("response_format", chat_requests[2])

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

        with patch.object(deepseek, "post_with_retry", new=fake_post):
            output, metadata = await deepseek.call_deepseek(
                object(),
                ITEM,
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
        self.assertEqual(repair["input"][-1]["content"], common.FINALIZE_PROMPT)

    async def test_deepseek_compiles_completed_research_without_final_text(self):
        research_items = [
            {"type": "reasoning", "id": "reason_1", "summary": []},
            {"type": "web_search_call", "id": "search_1", "status": "completed"},
        ]
        for terminal in ('{"employer":"Acme","hometown":""}', ""):
            with self.subTest(terminal_present=bool(terminal)):
                responses = [
                    FakeResponse({"status": "completed", "output": research_items,
                                  "usage": {"input_tokens": 100, "output_tokens": 20}}),
                    FakeResponse({"status": "completed", "output_text": terminal,
                                  "output": [], "usage": {"input_tokens": 40, "output_tokens": 10}}),
                ]
                fake_post = AsyncMock(side_effect=responses)
                with patch.object(deepseek, "post_with_retry", fake_post):
                    call = deepseek.call_deepseek(
                        object(), ITEM, api_key="secret", model="deepseek-v4-pro",
                        reasoning="high", max_search_rounds=None,
                    )
                    if terminal:
                        output, metadata = await call
                        self.assertEqual(output["employer"], "Acme")
                        self.assertTrue(metadata["terminal_repaired"])
                        self.assertTrue(metadata["terminal_format_valid"])
                        self.assertEqual(metadata["web_searches"], 1)
                        self.assertEqual(metadata["input_tokens"], 140)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "no JSON object"):
                            await call
                self.assertEqual(fake_post.await_count, 2)
                repair = fake_post.await_args_list[1].kwargs["json"]
                self.assertNotIn("tools", repair)
                self.assertEqual(repair["input"][1:-1], research_items)
                self.assertEqual(repair["input"][-1]["content"], common.FINALIZE_PROMPT)

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

        with patch.object(glm, "post_with_retry", new=fake_post):
            output, metadata = await glm.call_glm(
                object(),
                ITEM,
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
        self.assertEqual(requests[1][0], glm.SEARCH_ENDPOINT)
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
