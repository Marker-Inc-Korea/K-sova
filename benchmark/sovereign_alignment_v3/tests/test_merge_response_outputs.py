from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from merge_response_outputs import response_failure_reason, response_quality_flags  # noqa: E402


class ResponseFailureTests(unittest.TestCase):
    def test_explicit_length_finish_is_retained_and_flagged(self) -> None:
        row = {"response": "partial", "finish_reason": "length"}
        self.assertIsNone(response_failure_reason(row))
        self.assertEqual(response_quality_flags(row), ["truncated_generation"])

    def test_missing_finish_at_budget_is_retained_and_flagged(self) -> None:
        row = {
            "response": "partial",
            "finish_reason": None,
            "requested_max_tokens": 1024,
            "usage": {"completion_tokens": 1024},
        }
        self.assertIsNone(response_failure_reason(row))
        self.assertEqual(response_quality_flags(row), ["truncated_generation"])

    def test_complete_response_is_accepted(self) -> None:
        row = {
            "response": "complete",
            "finish_reason": "stop",
            "requested_max_tokens": 1024,
            "usage": {"completion_tokens": 100},
        }
        self.assertIsNone(response_failure_reason(row))
        self.assertEqual(response_quality_flags(row), [])


class ResponseMergeTests(unittest.TestCase):
    def test_complete_retry_clears_earlier_truncation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "requests.jsonl"
            first = root / "first.jsonl"
            retry = root / "retry.jsonl"
            output = root / "merged.jsonl"
            report = root / "report.json"
            requests.write_text(json.dumps({"id": "item-1"}) + "\n", encoding="utf-8")
            first.write_text(json.dumps({
                "id": "item-1", "model": "test-model", "response": "partial",
                "finish_reason": "length", "requested_max_tokens": 10,
                "usage": {"completion_tokens": 10}, "error": None,
            }) + "\n", encoding="utf-8")
            retry.write_text(json.dumps({
                "id": "item-1", "model": "test-model", "response": "complete",
                "finish_reason": "stop", "requested_max_tokens": 10,
                "usage": {"completion_tokens": 4}, "error": None,
            }) + "\n", encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(TOOLS / "merge_response_outputs.py"),
                "--requests", str(requests), "--responses", str(first), str(retry),
                "--model", "test-model", "--output", str(output), "--report", str(report),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            merged = json.loads(output.read_text(encoding="utf-8"))
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(merged["response"], "complete")
            self.assertEqual(summary["quality_flags"], [])


if __name__ == "__main__":
    unittest.main()
