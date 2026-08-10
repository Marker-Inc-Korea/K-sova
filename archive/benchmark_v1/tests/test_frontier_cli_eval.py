from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))

from frontier_cli_eval import (  # noqa: E402
    generation_contract,
    parse_codex,
    request_messages,
    reusable_results,
    sha256_json,
)


class FrontierCliEvalTests(unittest.TestCase):
    def event(self, event_type: str, item: dict | None = None, **extra: object) -> str:
        value: dict[str, object] = {"type": event_type, **extra}
        if item is not None:
            value["item"] = item
        return json.dumps(value)

    def test_parse_codex_accepts_reasoning_and_final_message(self) -> None:
        stdout = "\n".join([
            self.event("item.completed", {"id": "r1", "type": "reasoning"}),
            self.event(
                "item.completed",
                {"id": "a1", "type": "agent_message", "text": "answer"},
            ),
            self.event("turn.completed", usage={"output_tokens": 12}),
        ])

        answer, usage, model, tool_events = parse_codex(stdout)

        self.assertEqual(answer, "answer")
        self.assertEqual(usage, {"output_tokens": 12})
        self.assertIsNone(model)
        self.assertEqual(tool_events, [])

    def test_parse_codex_reports_prohibited_tool_once(self) -> None:
        tool = {"id": "cmd1", "type": "command_execution"}
        stdout = "\n".join([
            self.event("item.started", tool),
            self.event("item.completed", tool),
            self.event(
                "item.completed",
                {"id": "a1", "type": "agent_message", "text": "answer"},
            ),
        ])

        answer, _, _, tool_events = parse_codex(stdout)

        self.assertEqual(answer, "answer")
        self.assertEqual(
            tool_events, [{"id": "cmd1", "type": "command_execution"}]
        )

    def test_claude_cli_disables_auxiliary_prompt_suggestions(self) -> None:
        source = (ROOT / "reproduce/frontier_cli_eval.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--prompt-suggestions",\n                    "false"', source)

    def test_request_messages_matches_system_and_prompt_contract(self) -> None:
        row = {"system": "judge", "prompt": "score this"}

        messages = request_messages(row)

        self.assertEqual(messages, [
            {"role": "system", "content": "judge"},
            {"role": "user", "content": "score this"},
        ])
        self.assertEqual(len(sha256_json(messages)), 64)

    def test_generation_contract_records_uncapped_cli_execution(self) -> None:
        class Args:
            provider = "codex"
            model = "gpt-5.6-sol"
            effort = "high"
            instructions = ROOT / "reproduce/frontier_instructions.txt"

        contract = generation_contract(Args())

        self.assertIsNone(contract["max_tokens"])
        self.assertFalse(contract["tools_allowed"])
        self.assertEqual(contract["provider"], "codex_cli")
        self.assertEqual(len(contract["instructions_sha256"]), 64)

    def test_source_keeps_frozen_metadata_separate_from_execution_metadata(self) -> None:
        source = (ROOT / "reproduce/frontier_cli_eval.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"metadata": dict(row.get("metadata") or {})', source)
        self.assertIn('"execution_metadata": {', source)
        self.assertNotIn('**dict(row.get("metadata") or {})', source)

    def test_resume_retains_only_contract_valid_successes(self) -> None:
        request = {"id": "a", "system": "judge", "prompt": "score", "metadata": {"x": 1}}
        args = Namespace(
            provider="claude",
            model="claude-sonnet-5",
            record_model="judge",
            effort="high",
            instructions=ROOT / "reproduce/frontier_judge_instructions.txt",
        )
        contract = generation_contract(args)
        success = {
            "id": "a",
            "model": "judge",
            "response": "{}",
            "finish_reason": "stop",
            "error": None,
            "requested_max_tokens": None,
            "generation_config": contract,
            "messages_sha256": sha256_json(request_messages(request)),
            "metadata": {"x": 1},
        }
        failure = {**success, "id": "b", "response": "", "error": "429"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in (success, failure)) + "\n",
                encoding="utf-8",
            )

            reusable, previous = reusable_results(path, [request], args)

        self.assertEqual(previous, 2)
        self.assertEqual(list(reusable), ["a"])


if __name__ == "__main__":
    unittest.main()
