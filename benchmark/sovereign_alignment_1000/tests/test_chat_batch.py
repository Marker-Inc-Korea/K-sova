import json
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from tools import chat_batch


class ChatBatchTest(unittest.TestCase):
    def test_completed_ids_excludes_length_terminated_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "responses.jsonl"
            rows = [
                {"id": "done", "response": "{}", "error": None, "finish_reason": "stop"},
                {"id": "truncated", "response": "{", "error": None, "finish_reason": "length"},
                {"id": "failed", "response": "", "error": "timeout", "finish_reason": None},
            ]
            output.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(chat_batch.completed_ids(output), {"done"})

    def test_http_error_response_body_is_preserved(self) -> None:
        args = Namespace(
            base_url="http://127.0.0.1:1/v1",
            model="served-model",
            record_model="record-model",
            temperature=0.0,
            force_max_tokens=None,
            max_tokens=128,
            top_p=1.0,
            timeout=1,
            retries=0,
        )
        error = urllib.error.HTTPError(
            args.base_url,
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":"maximum context length exceeded"}'),
        )
        with patch.object(chat_batch.urllib.request, "urlopen", side_effect=error):
            result = chat_batch.call(
                {"id": "request", "prompt": "prompt", "max_tokens": 128},
                args,
                None,
                None,
            )

        self.assertIn("maximum context length exceeded", result["error"])


if __name__ == "__main__":
    unittest.main()
