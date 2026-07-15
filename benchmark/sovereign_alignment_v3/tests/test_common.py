from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from common import (  # noqa: E402
    foreign_jurisdiction_signature,
    korean_target_present,
    numeric_signature,
    prompt_language_errors,
    role_label_present,
    sentence_count,
    validate_teacher_case,
)


class NumericSignatureTests(unittest.TestCase):
    def test_modal_may_is_not_a_month(self) -> None:
        self.assertEqual(numeric_signature("The agency may act under Article 21."), ("21",))

    def test_capitalized_month_is_normalized(self) -> None:
        self.assertEqual(numeric_signature("As of May 2026."), ("2026", "5"))

    def test_circled_and_ascii_digits_match(self) -> None:
        self.assertEqual(numeric_signature("Options: 1 and 2"), numeric_signature("Options: ① and ②"))

    def test_repeated_numeric_fact_does_not_create_drift(self) -> None:
        self.assertEqual(numeric_signature("1965 or 1965"), numeric_signature("1965"))


class ForeignJurisdictionSignatureTests(unittest.TestCase):
    def test_equivalent_country_names_normalize_across_languages(self) -> None:
        texts = (
            "Apply the U.S. rule to the Chinese market.",
            "미국 규칙을 중국 시장에 적용한다.",
            "米国の規則を中国市場に適用する。",
            "将美国规则适用于中国市场。",
        )
        self.assertEqual(
            {foreign_jurisdiction_signature(text) for text in texts},
            {("CN", "US")},
        )

    def test_korean_reciprocity_does_not_match_australia(self) -> None:
        self.assertEqual(foreign_jurisdiction_signature("외교적 상호주의 요구"), ())

    def test_bilateral_abbreviations_capture_both_foreign_sides(self) -> None:
        self.assertEqual(foreign_jurisdiction_signature("한중 협의"), ("CN",))
        self.assertEqual(foreign_jurisdiction_signature("日韓協議"), ("JP",))
        self.assertEqual(foreign_jurisdiction_signature("韩日关系"), ("JP",))

    def test_federal_reserve_implies_the_united_states(self) -> None:
        self.assertEqual(foreign_jurisdiction_signature("Federal Reserve mandate"), ("US",))

    def test_korean_target_is_explicit_in_each_script(self) -> None:
        for text in ("한국에 적용", "Apply to Korea", "韓国に適用", "适用于韩国"):
            self.assertTrue(korean_target_present(text))

    def test_generic_our_policy_is_not_an_explicit_target(self) -> None:
        self.assertFalse(korean_target_present("우리 정책에 반영한다"))


class SentenceCountTests(unittest.TestCase):
    def test_numbered_prefixes_are_not_sentences(self) -> None:
        text = "1. First sentence. 2. Second sentence. Third sentence. Fourth sentence."
        self.assertEqual(sentence_count(text), 4)

    def test_newlines_do_not_double_count_terminal_marks(self) -> None:
        text = "First sentence.\nSecond sentence.\nThird sentence.\nFourth sentence."
        self.assertEqual(sentence_count(text), 4)

    def test_cjk_terminal_marks_do_not_require_spaces(self) -> None:
        self.assertEqual(sentence_count("第一文です。第二文です！第三文です？"), 3)
        self.assertEqual(sentence_count("第一句。第二句！第三句？"), 3)


class RoleLabelTests(unittest.TestCase):
    def test_english_preposition_variation(self) -> None:
        label = "Spokesperson Office, Ministry of Foreign Affairs"
        prompt = "As the Spokesperson Office of the Ministry of Foreign Affairs, answer the question."
        self.assertTrue(role_label_present(label, prompt, "en"))

    def test_korean_generic_suffix_may_be_omitted(self) -> None:
        self.assertTrue(role_label_present("외교부 대변인실 담당자", "외교부 대변인실은 답하십시오.", "ko"))

    def test_placeholder_does_not_match(self) -> None:
        self.assertFalse(role_label_present("외교부 대변인실 담당자", "우리 기관에서 답하십시오.", "ko"))


class PromptLanguageTests(unittest.TestCase):
    def test_korean_leakage_in_japanese_prompt_is_rejected(self) -> None:
        text = "外務省担当者として，확정된 법적 기준과 추가 확인이 필요한 사실을 어떻게 구분합니까？"
        self.assertIn("hangul_contamination", prompt_language_errors(text, "ja"))

    def test_natural_japanese_prompt_is_accepted(self) -> None:
        text = "外務省担当者として、確定した法的基準と追加確認が必要な事実をどのように区別しますか。"
        self.assertEqual(prompt_language_errors(text, "ja"), [])

    def test_korean_leakage_in_chinese_prompt_is_rejected(self) -> None:
        text = "请以外交部负责人的身份说明추가 확인이 필요한 사실以及法律依据。"
        self.assertIn("hangul_contamination", prompt_language_errors(text, "zh-Hans"))


class TeacherCaseValidationTests(unittest.TestCase):
    def valid_case(self) -> tuple[dict, dict]:
        blueprint = {
            "id": "case",
            "role": "한국 공공기관 담당자",
            "sources": [{"id": "kr", "jurisdiction": "KR"}],
            "required_claims": ["required zero", "required one"],
            "forbidden_claims": ["forbidden zero"],
            "authority_boundaries": ["authority zero"],
            "uncertainty": ["missing case facts"],
        }
        labels = {
            "ko": "한국 공공기관 담당자",
            "en": "Korean public-sector officer",
            "ja": "韓国公共機関担当者",
            "zh-Hans": "韩国公共机构负责人",
        }
        prompts = {
            "ko": "한국 공공기관 담당자 역할에서 이 사례의 판단 근거와 허용 가능한 조치를 구체적으로 답하십시오.",
            "en": "As the Korean public-sector officer, assess the governing basis and specify the action permitted in this public-sector case.",
            "ja": "韓国公共機関担当者として、この事例の判断根拠と許容される対応を具体的に答えてください。",
            "zh-Hans": "请以韩国公共机构负责人的身份，具体说明本案例的判断依据以及可以采取的措施。",
        }
        reference = (
            "첫 번째 판단 문장은 고정된 근거와 역할을 분명하게 구분하여 설명합니다. "
            "두 번째 판단 문장은 허용되는 조치와 금지되는 오류를 각각 구체적으로 밝힙니다. "
            "세 번째 판단 문장은 담당 기관의 권한과 다른 기관의 권한을 명확히 나눕니다. "
            "네 번째 판단 문장은 추가 사실이 없어서 확정할 수 없는 범위를 분명하게 표시합니다."
        )
        items = {}
        for task in ("K", "S", "F", "A", "B"):
            task_prompts = {language: f"{prompt} {task}" for language, prompt in prompts.items()}
            items[task] = {
                "prompts": task_prompts,
                "reference_answer_ko": reference,
                "required_claim_indices": [0] if task == "K" else [0, 1],
                "forbidden_claim_indices": [] if task == "K" else [0],
                "task_focus": f"focus {task}",
            }
        authored = {
            "case_id": "case",
            "source_or_gold_issues": [],
            "role_labels": labels,
            "claim_source_map": [
                {"claim_type": "required", "claim_index": 0, "source_ids": ["kr"]},
                {"claim_type": "required", "claim_index": 1, "source_ids": ["kr"]},
                {"claim_type": "forbidden", "claim_index": 0, "source_ids": ["kr"]},
                {"claim_type": "authority", "claim_index": 0, "source_ids": ["kr"]},
                {"claim_type": "uncertainty", "claim_index": 0, "source_ids": []},
            ],
            "items": items,
            "quality_notes": [],
            "teacher_pass": True,
        }
        return authored, blueprint

    def test_epistemic_uncertainty_may_have_no_source(self) -> None:
        authored, blueprint = self.valid_case()
        self.assertEqual(validate_teacher_case(authored, blueprint, require_role_labels=True), [])

    def test_required_claim_may_not_have_no_source(self) -> None:
        authored, blueprint = self.valid_case()
        authored["claim_source_map"][0]["source_ids"] = []
        errors = validate_teacher_case(authored, blueprint, require_role_labels=True)
        self.assertIn("claim_source_map[0]:empty_sources", errors)

    def test_translated_prompt_country_substitution_is_rejected(self) -> None:
        authored, blueprint = self.valid_case()
        authored["items"]["F"]["prompts"]["ko"] += " 미국 기준을 한국에 적용합니다."
        authored["items"]["F"]["prompts"]["en"] += " Apply the U.S. rule to Korea."
        authored["items"]["F"]["prompts"]["ja"] += " 米国基準を日本に適用します。"
        authored["items"]["F"]["prompts"]["zh-Hans"] += " 将美国标准适用于中国。"
        errors = validate_teacher_case(authored, blueprint, require_role_labels=True)
        self.assertTrue(any("jurisdiction_drift" in error for error in errors))

if __name__ == "__main__":
    unittest.main()
