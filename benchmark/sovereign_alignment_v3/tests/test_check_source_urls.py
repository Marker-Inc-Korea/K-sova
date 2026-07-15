import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from check_source_urls import check  # noqa: E402


class SourceUrlReachabilityTests(unittest.TestCase):
    @patch("check_source_urls.urllib.request.urlopen")
    def test_redirect_response_counts_as_host_reachable(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://example.go.kr/source",
            307,
            "Temporary Redirect",
            {},
            None,
        )

        result = check(
            {"case_id": "case", "id": "source", "url": "https://example.go.kr/source"},
            timeout=1,
        )

        self.assertTrue(result["reachable"])
        self.assertEqual(result["status"], 307)


if __name__ == "__main__":
    unittest.main()
