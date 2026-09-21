import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import judge

FIELDS = [{"fieldname": "employer", "answer": "Acme"}]
ITEM = {"person_info": "Synthetic test person", "fields": FIELDS}


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def client(*responses):
    create = AsyncMock(side_effect=list(responses))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_failure_never_uses_substring_fallback(self):
        oai = client(*[RuntimeError("provider unavailable") for _ in range(3)])
        with patch.object(judge.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(judge.JudgmentError):
                await judge.judge_fields(oai, asyncio.Semaphore(1), "Person", {"employer": "Acme"}, FIELDS)

    async def test_invalid_verdicts_fail_without_inventing_grades(self):
        for raw in ("{}", "[]", "not json", '{"employer":{"match":"missing"}}', None):
            with self.subTest(raw=raw), patch.object(judge.asyncio, "sleep", new=AsyncMock()):
                oai = client(*[response(raw) for _ in range(3)])
                with self.assertRaises(judge.JudgmentError):
                    await judge.judge_fields(oai, asyncio.Semaphore(1), "Person", {"employer": "Acme"}, FIELDS)
                self.assertEqual(oai.chat.completions.create.await_count, 3)

    async def test_retry_preserves_prompt_and_accepts_valid_grade(self):
        oai = client(response("{}"), response('{"employer":{"match":"correct","reason":"same"}}'))
        with patch.object(judge.asyncio, "sleep", new=AsyncMock()):
            verdicts = await judge.judge_fields(oai, asyncio.Semaphore(1), "Person", {"employer": "Acme"}, FIELDS)
        self.assertEqual(verdicts["employer"]["match"], "correct")
        calls = oai.chat.completions.create.call_args_list
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0].kwargs["messages"][0]["content"], judge.JUDGE_PROMPT)

    async def test_empty_answer_does_not_call_judge(self):
        oai = client()
        verdicts = await judge.judge_fields(oai, asyncio.Semaphore(1), "Person", {}, FIELDS)
        self.assertEqual(verdicts["employer"]["match"], "missing")
        oai.chat.completions.create.assert_not_called()

    async def test_research_is_saved_before_judge_and_failure_is_unscored(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test", "RECON_RUNS_DIR": directory}):
            runner = judge.EvalRunner("test", {})
            async def fail(*args):
                saved = json.loads(runner.out_path.read_text())["results"][0]
                self.assertEqual(saved["output"], {"employer": "Acme"})
                self.assertEqual(saved["error_type"], "judgment_pending")
                raise judge.JudgmentError("Unavailable")
            with patch.object(judge, "judge_fields", side_effect=fail):
                result = await runner.record(ITEM, {"employer": "Acme"}, 2, {"usage": {"tokens": 10}})
            self.assertEqual(result["error_type"], "judgment_error")
            self.assertEqual(result["usage"], {"tokens": 10})
            self.assertEqual(runner._summary_dict()["total_fields"], 0)
            self.assertEqual(len(runner.results), 1)
            await runner.oai.close()

    async def test_success_updates_pending_record_and_run_paths_are_unique(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test", "RECON_RUNS_DIR": directory}):
            runner = judge.EvalRunner("test", {})
            other = judge.EvalRunner("test", {})
            self.assertNotEqual(runner.out_path, other.out_path)
            with patch.object(judge, "judge_fields", new=AsyncMock(return_value={"employer": {"match": "correct", "reason": "same"}})):
                await runner.record(ITEM, {"employer": "Acme"}, 2)
            self.assertEqual(len(runner.results), 1)
            self.assertNotIn("error", runner.results[0])
            self.assertEqual(runner._summary_dict()["correct"], 1)
            await runner.record_error(ITEM, 3, RuntimeError("provider failed"))
            self.assertEqual(runner.results[-1]["error_type"], "provider_error")
            await runner.oai.close()
            await other.oai.close()

    async def test_provider_error_metadata_preserved_without_overriding_status(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test", "RECON_RUNS_DIR": directory}):
            runner = judge.EvalRunner("test", {})
            result = await runner.record_error(ITEM, 3, RuntimeError("provider failed"), {
                "provider_response": {"stop_reason": "max_tokens"},
                "error_type": "incorrect_override", "correct": 99,
            })
            saved = json.loads(runner.out_path.read_text())["results"][0]
            self.assertEqual(saved, result)
            self.assertEqual(saved["provider_response"], {"stop_reason": "max_tokens"})
            self.assertEqual(saved["error_type"], "provider_error")
            self.assertEqual(saved["correct"], 0)
            await runner.oai.close()


if __name__ == "__main__":
    unittest.main()
