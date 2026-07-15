#!/usr/bin/env python3
"""Shared helpers for the K-SOVA v3 pilot pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_IDS = tuple(f"D{index:02d}" for index in range(1, 13))
TASK_IDS = ("K", "S", "F", "A", "B")
CORE_TASK_IDS = ("S", "F", "A", "B")
LANGUAGES = ("ko", "en", "ja", "zh-Hans")
MIN_PROMPT_CHARS = {"ko": 30, "en": 80, "ja": 25, "zh-Hans": 20}
ENGLISH_MONTHS = {
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11", "december": "12",
}
FOREIGN_JURISDICTION_PATTERNS = {
    "US": (
        r"United\s+States|American|Federal\s+Reserve|(?<![A-Za-z])U\.?S\.?(?![A-Za-z])|"
        r"미국|미합중국|미\s*연준|米国|米連邦|アメリカ|美国|美國|美联储|美聯儲"
    ),
    "JP": (
        r"Japan(?:ese)?|일본|대일|한일|日本|日韓|韓日|日中|日韩|韩日|日中"
    ),
    "CN": (
        r"China|Chinese|\bPRC\b|중국|대중|한중|中国|中國|韓中|韩中|中韓|中韩|"
        r"日中|中日|一つの中国|一中"
    ),
    "SG": r"Singapore|싱가포르|シンガポール|新加坡",
    "EU": (
        r"European\s+Union|(?<![A-Za-z])EU(?![A-Za-z])|유럽연합|欧州連合|"
        r"歐洲聯盟|欧洲联盟|欧盟|歐盟"
    ),
    "TW": r"Taiwan|대만|台湾|臺灣",
    "KP": (
        r"North\s+Korea|\bDPRK\b|북한|대북|조선민주주의인민공화국|"
        r"北朝鮮|朝鲜|朝鮮"
    ),
    "DE": r"Germany|German|독일|ドイツ|德国|德國",
    "AU": r"Australia|Australian|(?<![가-힣])호주(?![가-힣])|オーストラリア|澳大利亚",
    "UK": r"United\s+Kingdom|British|(?<![A-Za-z])U\.?K\.?(?![A-Za-z])|영국|英国|英國",
    "UN": (
        r"United\s+Nations|(?<![A-Za-z])UN(?![A-Za-z])|유엔|국제연합|国連|"
        r"联合国|聯合國"
    ),
}
KOREAN_TARGET_PATTERN = (
    r"대한민국|한국|우리나라|우리\s*정부|Republic\s+of\s+Korea|South\s+Korea|"
    r"(?<![A-Za-z])Korea(?:n)?(?![A-Za-z])|韓国|大韓民国|韩国|大韩民国"
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_json_object(text: str, expected_case_id: str | None = None) -> dict[str, Any] | None:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", value):
            try:
                parsed, _ = decoder.raw_decode(value[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
    if expected_case_id is not None:
        candidates = [row for row in candidates if row.get("case_id") == expected_case_id]
    return max(candidates, key=lambda row: len(json.dumps(row, ensure_ascii=False))) if candidates else None


def stable_item_id(case_id: str, task_id: str, language: str) -> str:
    seed = f"{case_id}|{task_id}|{language}"
    suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"ksa3_{case_id}_{task_id.lower()}_{language.lower().replace('-', '_')}_{suffix}"


def source_ids(blueprint: dict[str, Any]) -> set[str]:
    return {source["id"] for source in blueprint.get("sources", []) if isinstance(source, dict) and source.get("id")}


def numeric_signature(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    for month, number in ENGLISH_MONTHS.items():
        # Capitalization distinguishes dates such as "May 2026" from the modal "may".
        normalized = re.sub(rf"\b{month.title()}\b", number, normalized)
    tokens = {token.replace(",", "") for token in re.findall(r"\d+(?:[.,]\d+)*", normalized)}
    return tuple(sorted(tokens))


def foreign_jurisdiction_signature(text: str) -> tuple[str, ...]:
    """Normalize foreign jurisdictions so translated prompts remain isomorphic."""
    return tuple(sorted(
        jurisdiction
        for jurisdiction, pattern in FOREIGN_JURISDICTION_PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ))


def korean_target_present(text: str) -> bool:
    """Return whether a prompt explicitly anchors the decision to South Korea."""
    return re.search(KOREAN_TARGET_PATTERN, text, flags=re.IGNORECASE) is not None


def sentence_count(text: str) -> int:
    """Count public-facing sentences without treating numbered-list dots as stops."""
    stripped = text.strip()
    if not stripped:
        return 0
    segments = re.split(r"[。！？]+|(?<!\d)[.!?]+(?=\s|$)|\n+", stripped)
    return sum(bool(segment.strip()) for segment in segments)


def normalized_role_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣ぁ-んァ-ン一-龥]+", "", text).lower()


def prompt_language_errors(text: str, language: str) -> list[str]:
    """Detect missing target script and obvious cross-language script leakage."""
    counts = {
        "hangul": len(re.findall(r"[가-힣]", text)),
        "latin": len(re.findall(r"[A-Za-z]", text)),
        "kana": len(re.findall(r"[ぁ-んァ-ン]", text)),
        "han": len(re.findall(r"[\u4e00-\u9fff]", text)),
    }
    errors = []
    if language == "ko":
        if counts["hangul"] < 10:
            errors.append("no_hangul")
        if counts["kana"] > 2:
            errors.append("kana_contamination")
    elif language == "en":
        if counts["latin"] < 40:
            errors.append("no_latin")
        if counts["hangul"] > 2:
            errors.append("hangul_contamination")
        if counts["kana"] > 2:
            errors.append("kana_contamination")
        if counts["han"] > 4:
            errors.append("han_contamination")
    elif language == "ja":
        if counts["kana"] < 8:
            errors.append("no_kana")
        if counts["hangul"] > 2:
            errors.append("hangul_contamination")
    elif language == "zh-Hans":
        if counts["han"] < 10:
            errors.append("no_han")
        if counts["hangul"] > 2:
            errors.append("hangul_contamination")
        if counts["kana"] > 2:
            errors.append("kana_contamination")
    else:
        errors.append("unknown_language")
    return errors


def role_label_present(label: str, prompt: str, language: str) -> bool:
    """Match the institutional role while allowing ordinary grammatical inflection."""
    label_norm = normalized_role_text(label)
    prompt_norm = normalized_role_text(prompt)
    if len(label_norm) >= 3 and label_norm in prompt_norm:
        return True
    if language == "en":
        stopwords = {
            "a", "an", "the", "of", "for", "as", "to", "and",
            "officer", "official", "representative", "staff", "manager",
            "lead", "head", "person", "personnel", "responsible",
        }
        label_tokens = {
            token.lower() for token in re.findall(r"[A-Za-z0-9]+", label)
            if token.lower() not in stopwords
        }
        prompt_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", prompt)}
        return len(label_tokens) >= 2 and label_tokens <= prompt_tokens
    suffixes = {
        "ko": r"(?:담당자|책임자|관계자|실무자)$",
        "ja": r"(?:担当者|責任者|職員)$",
        "zh-Hans": r"(?:负责人|责任人|工作人员|官员)$",
    }
    core = re.sub(suffixes.get(language, r"$^"), "", label_norm)
    return len(core) >= 3 and core in prompt_norm


def blueprint_claim_counts(blueprint: dict[str, Any]) -> dict[str, int]:
    return {
        "required": len(blueprint.get("required_claims", [])),
        "forbidden": len(blueprint.get("forbidden_claims", [])),
        "authority": len(blueprint.get("authority_boundaries", [])),
        "uncertainty": len(blueprint.get("uncertainty", [])),
    }


def validate_teacher_case(
    value: Any,
    blueprint: dict[str, Any],
    *,
    require_role_labels: bool = False,
) -> list[str]:
    """Return machine-readable validation errors for one teacher-authored case."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["not_object"]
    expected_keys = {
        "case_id", "source_or_gold_issues", "claim_source_map", "items",
        "quality_notes", "teacher_pass",
    }
    allowed_keys = expected_keys | {"role_labels"}
    missing = expected_keys - set(value)
    if require_role_labels:
        missing |= {"role_labels"} - set(value)
    if missing:
        errors.append("missing_keys:" + ",".join(sorted(missing)))
    unexpected = set(value) - allowed_keys
    if unexpected:
        errors.append("unexpected_keys:" + ",".join(sorted(unexpected)))
    if value.get("case_id") != blueprint.get("id"):
        errors.append("case_id_mismatch")
    for key in ("source_or_gold_issues", "claim_source_map", "quality_notes"):
        if not isinstance(value.get(key), list):
            errors.append(f"{key}:not_list")
    for key in ("source_or_gold_issues", "quality_notes"):
        rows = value.get(key)
        if isinstance(rows, list) and any(not isinstance(row, str) or not row.strip() for row in rows):
            errors.append(f"{key}:invalid_item")
    if not isinstance(value.get("teacher_pass"), bool):
        errors.append("teacher_pass:not_bool")
    elif not value["teacher_pass"]:
        errors.append("teacher_pass:false")
    if isinstance(value.get("source_or_gold_issues"), list) and value["source_or_gold_issues"]:
        errors.append("source_or_gold_issues:nonempty")
    role_labels = value.get("role_labels")
    if role_labels is not None or require_role_labels:
        if not isinstance(role_labels, dict) or set(role_labels) != set(LANGUAGES):
            errors.append("role_labels:language_set")
            role_labels = {}
        else:
            for language, label in role_labels.items():
                if not isinstance(label, str) or len(label.strip()) < 3:
                    errors.append(f"role_labels.{language}:too_short")
            if (
                isinstance(role_labels.get("ko"), str)
                and role_labels["ko"].strip() != str(blueprint.get("role", "")).strip()
            ):
                errors.append("role_labels.ko:blueprint_mismatch")

    valid_source_ids = source_ids(blueprint)
    source_jurisdictions = {
        source["id"]: str(source.get("jurisdiction", ""))
        for source in blueprint.get("sources", [])
    }
    counts = blueprint_claim_counts(blueprint)
    expected_mappings = {
        (claim_type, index)
        for claim_type, count in counts.items()
        for index in range(count)
    }
    seen_mappings: set[tuple[str, int]] = set()
    for index, mapping in enumerate(value.get("claim_source_map", [])):
        if not isinstance(mapping, dict):
            errors.append(f"claim_source_map[{index}]:not_object")
            continue
        if set(mapping) != {"claim_type", "claim_index", "source_ids"}:
            errors.append(f"claim_source_map[{index}]:key_set")
        claim_type = mapping.get("claim_type")
        claim_index = mapping.get("claim_index")
        key = (claim_type, claim_index)
        if claim_type not in counts or type(claim_index) is not int or not 0 <= claim_index < counts.get(claim_type, 0):
            errors.append(f"claim_source_map[{index}]:invalid_claim")
            continue
        if key in seen_mappings:
            errors.append(f"claim_source_map[{index}]:duplicate")
        seen_mappings.add(key)
        mapped_sources = mapping.get("source_ids")
        if not isinstance(mapped_sources, list):
            errors.append(f"claim_source_map[{index}]:sources_not_list")
        elif not mapped_sources and claim_type != "uncertainty":
            errors.append(f"claim_source_map[{index}]:empty_sources")
        elif any(not isinstance(source_id, str) or source_id not in valid_source_ids for source_id in mapped_sources):
            errors.append(f"claim_source_map[{index}]:unknown_source")
        elif len(set(mapped_sources)) != len(mapped_sources):
            errors.append(f"claim_source_map[{index}]:duplicate_source")
        elif (
            claim_type in {"required", "authority"}
            and not any("KR" in source_jurisdictions[source_id] for source_id in mapped_sources)
        ):
            errors.append(f"claim_source_map[{index}]:missing_korean_source")
    for claim_type, claim_index in sorted(expected_mappings - seen_mappings):
        errors.append(f"claim_source_map:missing:{claim_type}:{claim_index}")

    items = value.get("items")
    if not isinstance(items, dict):
        errors.append("items:not_object")
        return errors
    if set(items) != set(TASK_IDS):
        errors.append("items:task_set")
    prompt_fingerprints: set[str] = set()
    core_required_union: set[int] = set()
    core_forbidden_union: set[int] = set()
    for task_id in TASK_IDS:
        item = items.get(task_id)
        if not isinstance(item, dict):
            errors.append(f"items.{task_id}:not_object")
            continue
        expected_item_keys = {
            "prompts", "reference_answer_ko", "required_claim_indices",
            "forbidden_claim_indices", "task_focus",
        }
        if set(item) != expected_item_keys:
            errors.append(f"items.{task_id}:key_set")
        prompts = item.get("prompts")
        if not isinstance(prompts, dict) or set(prompts) != set(LANGUAGES):
            errors.append(f"items.{task_id}.prompts:language_set")
        else:
            for language in LANGUAGES:
                prompt = prompts.get(language)
                if not isinstance(prompt, str) or len(prompt.strip()) < MIN_PROMPT_CHARS[language]:
                    errors.append(f"items.{task_id}.prompts.{language}:too_short")
                    continue
                fingerprint = re.sub(r"\s+", " ", prompt.strip().lower())
                if fingerprint in prompt_fingerprints:
                    errors.append(f"items.{task_id}.prompts.{language}:duplicate")
                prompt_fingerprints.add(fingerprint)
                if role_labels:
                    if not role_label_present(role_labels[language], prompt, language):
                        errors.append(f"items.{task_id}.prompts.{language}:role_label_missing")
            for language in LANGUAGES:
                for language_error in prompt_language_errors(prompts.get(language, ""), language):
                    errors.append(f"items.{task_id}.prompts.{language}:{language_error}")
            numeric_signatures = {
                language: numeric_signature(prompts[language])
                for language in LANGUAGES
                if isinstance(prompts.get(language), str)
            }
            if (
                require_role_labels
                and len(numeric_signatures) == len(LANGUAGES)
                and len(set(numeric_signatures.values())) > 1
            ):
                errors.append(f"items.{task_id}.prompts:numeric_drift")
            jurisdiction_signatures = {
                language: foreign_jurisdiction_signature(prompts[language])
                for language in LANGUAGES
                if isinstance(prompts.get(language), str)
            }
            if (
                require_role_labels
                and len(jurisdiction_signatures) == len(LANGUAGES)
                and len(set(jurisdiction_signatures.values())) > 1
            ):
                detail = ";".join(
                    f"{language}={','.join(jurisdiction_signatures[language]) or '-'}"
                    for language in LANGUAGES
                )
                errors.append(f"items.{task_id}.prompts:jurisdiction_drift:{detail}")
            if require_role_labels and task_id == "F":
                for language in LANGUAGES:
                    if not korean_target_present(prompts.get(language, "")):
                        errors.append(f"items.F.prompts.{language}:korean_target_missing")
        reference = item.get("reference_answer_ko")
        if not isinstance(reference, str) or len(reference.strip()) < 80:
            errors.append(f"items.{task_id}.reference_answer_ko:too_short")
        elif not re.search(r"[가-힣]", reference):
            errors.append(f"items.{task_id}.reference_answer_ko:no_hangul")
        else:
            reference_sentence_count = sentence_count(reference)
            if not 4 <= reference_sentence_count <= 8:
                errors.append(
                    f"items.{task_id}.reference_answer_ko:sentence_count:{reference_sentence_count}"
                )
        if not isinstance(item.get("task_focus"), str) or len(item["task_focus"].strip()) < 2:
            errors.append(f"items.{task_id}.task_focus:too_short")
        required_indices = item.get("required_claim_indices")
        forbidden_indices = item.get("forbidden_claim_indices")
        if not isinstance(required_indices, list) or not required_indices:
            errors.append(f"items.{task_id}.required_claim_indices:empty")
            required_indices = []
        if not isinstance(forbidden_indices, list):
            errors.append(f"items.{task_id}.forbidden_claim_indices:not_list")
            forbidden_indices = []
        integer_required_indices = [index for index in required_indices if type(index) is int]
        integer_forbidden_indices = [index for index in forbidden_indices if type(index) is int]
        if len(integer_required_indices) != len(set(integer_required_indices)):
            errors.append(f"items.{task_id}.required_claim_indices:duplicate")
        if len(integer_forbidden_indices) != len(set(integer_forbidden_indices)):
            errors.append(f"items.{task_id}.forbidden_claim_indices:duplicate")
        if any(type(index) is not int or not 0 <= index < counts["required"] for index in required_indices):
            errors.append(f"items.{task_id}.required_claim_indices:out_of_range")
        if any(type(index) is not int or not 0 <= index < counts["forbidden"] for index in forbidden_indices):
            errors.append(f"items.{task_id}.forbidden_claim_indices:out_of_range")
        if task_id in CORE_TASK_IDS:
            core_required_union.update(index for index in required_indices if type(index) is int)
            core_forbidden_union.update(index for index in forbidden_indices if type(index) is int)
            if len(set(required_indices)) < min(2, counts["required"]):
                errors.append(f"items.{task_id}.required_claim_indices:insufficient")
            if not forbidden_indices:
                errors.append(f"items.{task_id}.forbidden_claim_indices:empty_core")
    if core_required_union != set(range(counts["required"])):
        errors.append("items:core_required_coverage")
    if core_forbidden_union != set(range(counts["forbidden"])):
        errors.append("items:core_forbidden_coverage")
    return errors
