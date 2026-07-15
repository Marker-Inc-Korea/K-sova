from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from build_judge_requests import normalize_response_text  # noqa: E402


class ResponseNormalizationTests(unittest.TestCase):
    def test_pathological_narrow_space_run_is_collapsed(self) -> None:
        value = "commission" + ("\u202f" * 1000) + "end"
        self.assertEqual(normalize_response_text(value), "commission end")

    def test_normal_spacing_and_line_structure_are_preserved(self) -> None:
        value = "first  line\n    indented\nlast"
        self.assertEqual(normalize_response_text(value), value)


if __name__ == "__main__":
    unittest.main()
