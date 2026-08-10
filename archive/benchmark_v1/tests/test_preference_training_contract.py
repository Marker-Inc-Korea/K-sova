#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import conflict_audit  # noqa: E402
import conflict_object_pipeline  # noqa: E402
import core_pipeline  # noqa: E402
import pipeline  # noqa: E402


class PreferenceTrainingContractTest(unittest.TestCase):
    def test_policy_divergence_is_an_accepted_conflict_type(self) -> None:
        config = pipeline.load_json(conflict_audit.CONFIG_PATH)
        self.assertIn(
            "sovereign_policy_divergence",
            config["construct"]["accepted_types"],
        )
        self.assertIn(
            "sovereign_policy_divergence",
            conflict_object_pipeline.ALLOWED_CONFLICT_TYPES,
        )

    def test_policy_divergence_requires_opposite_execution_results(self) -> None:
        prompt = conflict_object_pipeline.mining_prompt(
            "environment_energy_security",
            "D10.nuclear",
            [],
            minimum_scenarios=4,
            maximum_scenarios=8,
            maximum_objects=2,
        )
        self.assertIn("동일한 정책 분류·허용 여부·공적 우선순위", prompt)
        self.assertIn("결론과 집행 효과", prompt)
        self.assertIn("단순 선호·산업 비중·정책 강도 차이", prompt)

    def test_shared_preamble_is_composed_into_all_answers(self) -> None:
        value = {
            "shared_preamble_ko": (
                "두 공식 원천은 동일 정책 질문에 상반된 결론을 제시한다. "
                "각 결론은 실제 집행 방향을 다르게 만든다."
            ),
            "answer_conclusions": {
                "korean_aligned": "따라서 첫 번째 공식 기준을 적용해야 한다.",
                "context_neutral": "따라서 두 기준의 차이만 확인하고 결론은 유보한다.",
                "korean_opposed": "따라서 두 번째 공식 기준을 적용해야 한다.",
            },
        }
        core_pipeline.compose_preamble_answers(value)
        self.assertEqual(set(core_pipeline.ANSWER_KEYS), set(value["answers"]))
        for answer in value["answers"].values():
            self.assertTrue(answer.startswith(value["shared_preamble_ko"]))


if __name__ == "__main__":
    unittest.main()
