import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("GEMINI_API_KEY", "test-key")
import gemini  # noqa: E402
from native_model_common import COMPILE_PROMPT  # noqa: E402

ITEM = {"person_info": "Ada Example, Engineer at Acme", "fields": [
    {"fieldname": "employer", "description": "Current employer"},
]}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class GeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_researches_with_tools_then_compiles_with_schema(self):
        research_content = {"role": "model", "parts": [
            {"text": "Ada works at Acme.", "thoughtSignature": "sig"},
        ]}
        responses = [
            FakeResponse({
                "candidates": [{"content": research_content, "finishReason": "STOP",
                                "groundingMetadata": {"webSearchQueries": ["Ada Example Acme"],
                                                      "groundingChunks": [{}, {}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10,
                                  "thoughtsTokenCount": 50, "toolUsePromptTokenCount": 900},
            }),
            FakeResponse({
                "candidates": [{"content": {"role": "model", "parts": [{"text": '{"employer":"Acme"}'}]}}],
                "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 5},
            }),
        ]
        fake_post = AsyncMock(side_effect=responses)
        with patch.object(gemini, "post_with_retry", fake_post):
            research, compiled = await gemini.call_api(object(), ITEM, "gemini-3.1-pro-preview", "high")

        self.assertEqual(gemini.extract_output(compiled), {"employer": "Acme"})
        research_body = fake_post.await_args_list[0].kwargs["json"]
        self.assertEqual(research_body["tools"], gemini.RESEARCH_TOOLS)
        self.assertNotIn("responseSchema", research_body["generationConfig"])
        compile_body = fake_post.await_args_list[1].kwargs["json"]
        self.assertNotIn("tools", compile_body)
        self.assertEqual(compile_body["generationConfig"]["responseSchema"]["required"], ["employer"])
        self.assertEqual(compile_body["contents"][1], research_content)
        self.assertEqual(compile_body["contents"][2]["parts"][0]["text"], COMPILE_PROMPT)
        metadata = gemini.extract_metadata(research, compiled)
        self.assertEqual(metadata["web_searches"], 1)
        self.assertEqual(metadata["grounding_sources"], 2)
        self.assertEqual(metadata["input_tokens"], 220)
        self.assertEqual(metadata["tool_use_tokens"], 900)

    async def test_empty_candidate_is_an_error(self):
        fake_post = AsyncMock(return_value=FakeResponse({"candidates": [{"finishReason": "SAFETY"}]}))
        with patch.object(gemini, "post_with_retry", fake_post):
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                await gemini.call_api(object(), ITEM, "gemini-3.1-pro-preview", "high")


if __name__ == "__main__":
    unittest.main()
