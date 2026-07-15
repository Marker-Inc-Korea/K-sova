from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from audit_construct_quality import (  # noqa: E402
    internal_metadata_leaks,
    task_focus_concentration,
    task_required_claim_diversity,
)


class TaskFocusConcentrationTests(unittest.TestCase):
    @staticmethod
    def item(index: int, task_id: str) -> dict:
        return {
            "id": f"item_{index}",
            "task_id": task_id,
            "task_focus": "외국 프레임 교정",
        }

    def test_same_task_taxonomy_reuse_is_informational(self) -> None:
        items = [self.item(index, "F") for index in range(15)]
        warnings, standardized = task_focus_concentration(items, threshold=12)
        self.assertEqual(warnings, [])
        self.assertEqual(standardized[0]["task_ids"], ["F"])
        self.assertEqual(standardized[0]["count"], 15)

    def test_cross_task_reuse_warns_about_task_collapse(self) -> None:
        items = [self.item(index, "F" if index < 6 else "S") for index in range(12)]
        warnings, standardized = task_focus_concentration(items, threshold=12)
        self.assertEqual(standardized, [])
        self.assertEqual(warnings[0]["task_ids"], ["F", "S"])


class TaskRequiredClaimDiversityTests(unittest.TestCase):
    def test_identical_claim_sets_across_tasks_are_an_error(self) -> None:
        items = [
            {"case_id": "case-1", "task_id": task, "required_claims": ["공통 명제"]}
            for task in ("K", "S", "F", "A", "B")
        ]
        collapses, warnings = task_required_claim_diversity(items)
        self.assertEqual([row["case_id"] for row in collapses], ["case-1"])
        self.assertEqual(warnings, [])

    def test_three_distinct_claim_sets_pass_the_diversity_gate(self) -> None:
        items = [
            {"case_id": "case-1", "task_id": "K", "required_claims": ["지식"]},
            {"case_id": "case-1", "task_id": "S", "required_claims": ["선택"]},
            {"case_id": "case-1", "task_id": "F", "required_claims": ["프레임"]},
            {"case_id": "case-1", "task_id": "A", "required_claims": ["선택"]},
            {"case_id": "case-1", "task_id": "B", "required_claims": ["프레임"]},
        ]
        collapses, warnings = task_required_claim_diversity(items)
        self.assertEqual(collapses, [])
        self.assertEqual(warnings, [])


class InternalMetadataLeakTests(unittest.TestCase):
    def test_source_identifier_is_rejected(self) -> None:
        findings = internal_metadata_leaks("근거는 kr_const_art21 및 판례에 있다.")
        self.assertEqual(findings, [{"code": "source_identifier", "surface": "kr_const_art21"}])

    def test_unlisted_jurisdiction_identifier_is_rejected(self) -> None:
        findings = internal_metadata_leaks("싱가포르 근거는 sg_pofma_regulations이다.")
        self.assertEqual(
            findings,
            [{"code": "source_identifier", "surface": "sg_pofma_regulations"}],
        )

    def test_public_source_name_is_allowed(self) -> None:
        self.assertEqual(internal_metadata_leaks("대한민국 헌법 제21조에 따른다."), [])

    def test_gold_index_label_is_rejected(self) -> None:
        findings = internal_metadata_leaks("이 내용은 Required 1에 해당한다.")
        self.assertEqual(
            findings,
            [{"code": "schema_label", "surface": "Required 1"}],
        )


if __name__ == "__main__":
    unittest.main()
