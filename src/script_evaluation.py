"""Offline, method-blind quality evaluation utilities for generated C11 skits."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    from src.script_validator import (
        DEFAULT_SCHEMA_PATH,
        DEFAULT_WPM,
        _inflection_forms,
        _tokens,
        load_passage,
        validate_script,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from script_validator import (  # type: ignore
        DEFAULT_SCHEMA_PATH,
        DEFAULT_WPM,
        _inflection_forms,
        _tokens,
        load_passage,
        validate_script,
    )


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SCORE_FIELDS = [
    "script_id",
    "passage_id",
    "score_scope",
    "target_word",
    "vocabulary_context_quality_0_3",
    "factual_faithfulness_0_5",
    "dialogue_naturalness_0_5",
    "grade_level_suitability_0_5",
    "story_coherence_0_5",
    "comments",
]
SECONDARY_SCORE_FIELDS = {
    "factual_faithfulness_0_5": 5,
    "dialogue_naturalness_0_5": 5,
    "grade_level_suitability_0_5": 5,
    "story_coherence_0_5": 5,
}
METHOD_MARKERS = (
    "direct generation",
    "plan-then-generate",
    "plan_then_generate",
    "generation_method",
    "vocabulary context plan",
)


def _script_lines(script: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (scene.get("scene_id", ""), line)
        for scene in script.get("scenes", [])
        for line in scene.get("dialogue", [])
        if isinstance(line, dict)
    ]


def _target_occurrences(script: dict[str, Any], targets: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forms_by_word = {word: _inflection_forms(word) for word in targets}
    found_forms = {word: [] for word in targets}
    positions = {word: [] for word in targets}
    for _, line in _script_lines(script):
        line_id = line.get("line_id", "")
        for token in _tokens(line.get("text", "")):
            for word, forms in forms_by_word.items():
                if token in forms:
                    if token not in found_forms[word]:
                        found_forms[word].append(token)
                    if line_id not in positions[word]:
                        positions[word].append(line_id)
    return found_forms, positions


def _source_reference_check(script: dict[str, Any], passage: dict[str, Any]) -> tuple[int, list[dict[str, str]]]:
    valid_ids = {item.get("id") for item in passage.get("sentences", []) if isinstance(item, dict)}
    total = 0
    invalid: list[dict[str, str]] = []
    for _, line in _script_lines(script):
        for sentence_id in line.get("source_sentence_ids", []):
            total += 1
            if sentence_id not in valid_ids:
                invalid.append({"line_id": line.get("line_id", ""), "sentence_id": str(sentence_id)})
    return total, invalid


def _count_anomalies(value: Any) -> int:
    if isinstance(value, dict):
        own = len(value.get("text_anomalies", [])) if isinstance(value.get("text_anomalies"), list) else 0
        return own + sum(_count_anomalies(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_anomalies(item) for item in value)
    return 0


def compute_automatic_metrics(
    script: dict[str, Any] | None,
    passage: dict[str, Any],
    generation_log: dict[str, Any],
    project_root: Path | str = ROOT,
    words_per_minute: int = DEFAULT_WPM,
) -> dict[str, Any]:
    """Compute deterministic metrics without using method identity in the calculation."""

    efficiency = generation_log.get("stats", {}) if isinstance(generation_log, dict) else {}
    if script is None:
        return {
            "passage_id": passage["passage_id"],
            "script_id": "",
            "status": "failed_no_script",
            "schema_valid": False,
            "business_rules_valid": False,
            "duration_compliance": False,
            "duration_is_estimated": True,
            "estimated_duration_sec": None,
            "duration_assumption_wpm": words_per_minute,
            "total_words": 0,
            "line_count": 0,
            "character_count": 0,
            "scene_count": 0,
            "target_word_count": len(passage.get("target_words", [])),
            "target_words_found": 0,
            "vocabulary_coverage_rate": 0.0,
            "target_word_forms": {word: [] for word in passage.get("target_words", [])},
            "target_word_positions": {word: [] for word in passage.get("target_words", [])},
            "source_reference_valid": False,
            "source_reference_count": 0,
            "invalid_source_references": [],
            "api_call_count": efficiency.get("api_call_count", 0),
            "retry_count": efficiency.get("retry_count", 0),
            "prompt_tokens": efficiency.get("prompt_tokens", 0),
            "completion_tokens": efficiency.get("completion_tokens", 0),
            "total_tokens": efficiency.get("total_tokens", 0),
            "elapsed_ms": efficiency.get("elapsed_ms", 0),
            "validation_warnings": [{"category": "error", "message": "no validated script JSON was produced"}],
            "semantic_review_required": True,
        }

    schema_path = Path(project_root) / "schemas" / "script.schema.json"
    validation = validate_script(script, passage, schema_path=schema_path, words_per_minute=words_per_minute)
    targets = list(passage.get("target_words", []))
    found_forms, positions = _target_occurrences(script, targets)
    found_count = sum(bool(found_forms[word]) for word in targets)
    source_count, invalid_refs = _source_reference_check(script, passage)
    stats = validation.get("stats", {})
    estimated_duration = stats.get("estimated_duration_sec")
    min_duration = passage.get("constraints", {}).get("min_duration_sec")
    max_duration = passage.get("constraints", {}).get("max_duration_sec")
    duration_compliance = (
        isinstance(estimated_duration, (int, float))
        and (min_duration is None or estimated_duration >= min_duration)
        and (max_duration is None or estimated_duration <= max_duration)
    )
    return {
        "passage_id": passage["passage_id"],
        "script_id": "",
        "status": "validated" if validation.get("valid") else "invalid",
        "schema_valid": bool(validation.get("schema_valid")),
        "business_rules_valid": bool(validation.get("valid")),
        "duration_compliance": duration_compliance,
        "duration_is_estimated": bool(stats.get("estimated_duration_is_not_audio", True)),
        "estimated_duration_sec": estimated_duration,
        "duration_assumption_wpm": stats.get("duration_assumption_wpm", words_per_minute),
        "total_words": stats.get("total_words", 0),
        "line_count": stats.get("line_count", 0),
        "character_count": stats.get("character_count", 0),
        "scene_count": stats.get("scene_count", 0),
        "target_word_count": len(targets),
        "target_words_found": found_count,
        "vocabulary_coverage_rate": round(found_count / len(targets), 6) if targets else 1.0,
        "target_word_forms": found_forms,
        "target_word_positions": positions,
        "source_reference_valid": not invalid_refs,
        "source_reference_count": source_count,
        "invalid_source_references": invalid_refs,
        "api_call_count": efficiency.get("api_call_count", 0),
        "retry_count": efficiency.get("retry_count", 0),
        "prompt_tokens": efficiency.get("prompt_tokens", 0),
        "completion_tokens": efficiency.get("completion_tokens", 0),
        "total_tokens": efficiency.get("total_tokens", 0),
        "elapsed_ms": efficiency.get("elapsed_ms", 0),
        "validation_warnings": validation.get("warnings", []),
        "semantic_review_required": bool(validation.get("semantic_review_required", True)),
    }


def anonymize_script(script: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove generation metadata while preserving the script shown to blind raters."""

    if script is None:
        return None
    result = copy.deepcopy(script)
    result.pop("generation_method", None)
    return result


def make_anonymous_mapping(seed: int) -> dict[str, str]:
    """Create a reproducible, independently seeded A/B-to-method mapping."""

    methods = ["direct", "plan_then_generate"]
    random.Random(seed).shuffle(methods)
    return {"script_a": methods[0], "script_b": methods[1]}


def make_anonymous_bundle(passage: dict[str, Any], scripts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    bundle = {
        "bundle_version": "1.0",
        "passage_id": passage["passage_id"],
        "title": passage.get("title", ""),
        "language": passage.get("language", "en"),
        "target_words": list(passage.get("target_words", [])),
        "constraints": copy.deepcopy(passage.get("constraints", {})),
        "sentences": copy.deepcopy(passage.get("sentences", [])),
        "scripts": [
            {"script_id": script_id, "script": anonymize_script(scripts[script_id])}
            for script_id in ("script_a", "script_b")
        ],
    }
    serialized = json.dumps(bundle, ensure_ascii=False).casefold()
    if any(marker in serialized for marker in METHOD_MARKERS):
        raise ValueError("anonymous bundle contains generation identity metadata")
    return bundle


def make_evaluation_sheet_rows(script_id: str, passage: dict[str, Any]) -> list[dict[str, str]]:
    rows = [{
        "script_id": script_id,
        "passage_id": passage["passage_id"],
        "score_scope": "script",
        "target_word": "__script__",
        "vocabulary_context_quality_0_3": "",
        "factual_faithfulness_0_5": "",
        "dialogue_naturalness_0_5": "",
        "grade_level_suitability_0_5": "",
        "story_coherence_0_5": "",
        "comments": "",
    }]
    rows.extend({
        "script_id": script_id,
        "passage_id": passage["passage_id"],
        "score_scope": "word",
        "target_word": word,
        "vocabulary_context_quality_0_3": "",
        "factual_faithfulness_0_5": "",
        "dialogue_naturalness_0_5": "",
        "grade_level_suitability_0_5": "",
        "story_coherence_0_5": "",
        "comments": "",
    } for word in passage.get("target_words", []))
    return rows


def _parse_score(raw: str, maximum: int, label: str, warnings: list[str]) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        warnings.append(f"invalid non-integer score for {label}")
        return None
    if value < 0 or value > maximum:
        warnings.append(f"out-of-range score for {label}: {value}; expected 0-{maximum}")
        return None
    return value


def summarize_manual_scores(csv_path: Path | str, passage_id: str) -> dict[str, Any]:
    warnings: list[str] = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    scripts: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("passage_id") != passage_id:
            warnings.append(f"ignored row for unexpected passage_id={row.get('passage_id')!r}")
            continue
        script_id = row.get("script_id", "")
        item = scripts.setdefault(script_id, {"word_scores": {}, "secondary": {}})
        if row.get("score_scope") == "word":
            word = row.get("target_word", "")
            item["word_scores"][word] = _parse_score(
                row.get("vocabulary_context_quality_0_3", ""), 3, f"{script_id}/{word}", warnings
            )
        elif row.get("score_scope") == "script":
            for field, maximum in SECONDARY_SCORE_FIELDS.items():
                item["secondary"][field] = _parse_score(row.get(field, ""), maximum, f"{script_id}/{field}", warnings)
        else:
            warnings.append(f"unknown score_scope for {script_id}: {row.get('score_scope')!r}")

    summaries: dict[str, dict[str, Any]] = {}
    expected_words = [row.get("target_word") for row in records if row.get("score_scope") == "word"]
    expected_words = list(dict.fromkeys(word for word in expected_words if word and word != "__script__"))
    for script_id, item in scripts.items():
        word_scores = item["word_scores"]
        missing = [word for word in expected_words if word not in word_scores or word_scores[word] is None]
        complete = not missing and bool(expected_words)
        if missing:
            warnings.append(f"missing vocabulary score(s) for {script_id}: {', '.join(missing)}")
        vocabulary_mean = round(sum(word_scores[word] for word in expected_words) / len(expected_words), 6) if complete else None
        summaries[script_id] = {
            "vocabulary_scores": word_scores,
            "vocabulary_context_quality_mean": vocabulary_mean,
            "vocabulary_scored_count": sum(value is not None for value in word_scores.values()),
            **item["secondary"],
        }

    pairwise: dict[str, float | None] = {}
    for field in ("vocabulary_context_quality_mean", *SECONDARY_SCORE_FIELDS):
        a = summaries.get("script_a", {}).get(field)
        b = summaries.get("script_b", {}).get(field)
        pairwise[field] = round(a - b, 6) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        if pairwise[field] is None:
            warnings.append(f"missing paired score for {field}")
    return {
        "analysis_unit": "passage_id",
        "passage_id": passage_id,
        "passage_count": 1,
        "scripts": summaries,
        "paired_difference_script_a_minus_script_b": pairwise,
        "warnings": list(dict.fromkeys(warnings)),
    }


RATER_SCORE_FIELDS = ["rater_id", *SCRIPT_SCORE_FIELDS]
RATER_IDS = ("rater_1", "rater_2")
RATER_INSTRUCTIONS_MARKDOWN = """# 开发集人工盲评填写说明

本材料仅用于 C11 开发集的独立人工评价。请分别完成自己的评分表，不要打开或寻找分析人员保管的 `method_mapping.json`，也不要根据脚本风格猜测生成方法。

## 填写步骤

1. 阅读匿名材料中的教材标题、完整原文、句子编号、五个目标词和 Script A / Script B。
2. 对每个脚本的五个目标词分别填写 0–3 分。
3. 每个脚本再填写四项整体指标，范围均为 0–5 分。
4. 在 `comments` 中记录必要的证据、疑问或边界判断；没有备注可以留空。
5. 不确定或尚未完成的项目留空，不要用 0 代替缺失值。请以 UTF-8 保存 CSV。

评分时只依据材料中可见的原文、句子编号和匿名剧本。不要参考 API 用量、耗时、规划内容、既有评价或另一位评价者的表格。

汇总单位是 `passage_id`。五个目标词的分数会先在同一篇文章内汇总，不把五个词当作五篇独立文章进行统计推断。
"""


def make_rater_sheet_rows(script_id: str, passage: dict[str, Any], rater_id: str) -> list[dict[str, str]]:
    """Create an independent, completely blank score sheet for one rater."""

    rows = [
        {
            "rater_id": rater_id,
            **row,
        }
        for row in make_evaluation_sheet_rows(script_id, passage)
    ]
    return rows


def _write_rater_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATER_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_rater_materials(
    project_root: Path | str = ROOT,
    output_dir: Path | str | None = None,
    rater_ids: tuple[str, ...] = RATER_IDS,
) -> dict[str, Any]:
    """Build two separate blind-rater packages from existing dev anonymous data only."""

    project_root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else project_root / "outputs" / "evaluation" / "v1.1" / "development" / "raters"
    source_path = project_root / "outputs" / "evaluation" / "v1.1" / "development" / "dev_anonymous_evaluation.json"
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    serialized = json.dumps(source_payload, ensure_ascii=False).casefold()
    if any(marker in serialized for marker in METHOD_MARKERS):
        raise ValueError("source anonymous evaluation contains method identity metadata")

    passages = {
        item["passage_id"]: item
        for item in json.loads((project_root / "data" / "v0.3" / "dev.json").read_text(encoding="utf-8"))
    }
    manifest: dict[str, Any] = {
        "bundle_version": "1.0",
        "analysis_unit": "passage_id",
        "raters": list(rater_ids),
        "source_anonymous_evaluation": str(source_path),
        "materials": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_rubric.md").write_text(RUBRIC_MARKDOWN, encoding="utf-8")
    (output / "rating_instructions.md").write_text(RATER_INSTRUCTIONS_MARKDOWN, encoding="utf-8")
    for rater_id in rater_ids:
        if rater_id not in {"rater_1", "rater_2"}:
            raise ValueError(f"unsupported rater id: {rater_id}")
        rater_dir = output / rater_id
        rater_dir.mkdir(parents=True, exist_ok=True)
        payload = copy.deepcopy(source_payload)
        payload["rater_material_id"] = rater_id
        json_path = rater_dir / f"dev_anonymous_evaluation_{rater_id}.json"
        _write_json(json_path, payload)
        rows: list[dict[str, str]] = []
        for passage_id in ("p001", "p003", "p005", "p009", "p012"):
            if passage_id not in passages:
                raise ValueError(f"missing development passage: {passage_id}")
            rows.extend(make_rater_sheet_rows("script_a", passages[passage_id], rater_id))
            rows.extend(make_rater_sheet_rows("script_b", passages[passage_id], rater_id))
        sheet_path = rater_dir / f"dev_evaluation_sheet_{rater_id}.csv"
        _write_rater_sheet(sheet_path, rows)
        rubric_path = rater_dir / "evaluation_rubric.md"
        rubric_path.write_text(RUBRIC_MARKDOWN, encoding="utf-8")
        instructions_path = rater_dir / "rating_instructions.md"
        instructions_path.write_text(RATER_INSTRUCTIONS_MARKDOWN, encoding="utf-8")
        manifest["materials"][rater_id] = {
            "anonymous_evaluation": str(json_path),
            "score_sheet": str(sheet_path),
            "rubric": str(rubric_path),
            "instructions": str(instructions_path),
            "row_count": len(rows),
        }
    manifest_path = output / "rater_materials_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def _load_method_mapping(path: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path, dict):
        return path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rater_records(path: Path | str, rater_id: str) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("rater_id", rater_id)
        if row.get("rater_id") and row.get("rater_id") != rater_id:
            warnings.append(f"row rater_id mismatch in {rater_id}: {row.get('rater_id')!r}")
    return rows, warnings


def _summarize_rater_records(
    records: list[dict[str, str]],
    rater_id: str,
    expected_words_by_passage: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        passage_id = row.get("passage_id", "")
        script_id = row.get("script_id", "")
        if not passage_id or not script_id:
            warnings.append(f"missing passage_id or script_id for {rater_id}")
            continue
        item = grouped.setdefault((passage_id, script_id), {"word_scores": {}, "secondary": {}, "raw_rows": []})
        item["raw_rows"].append(dict(row))
        if row.get("score_scope") == "word":
            word = row.get("target_word", "")
            if word in item["word_scores"]:
                warnings.append(f"duplicate vocabulary score for {rater_id}/{passage_id}/{script_id}/{word}")
            item["word_scores"][word] = _parse_score(
                row.get("vocabulary_context_quality_0_3", ""), 3,
                f"{rater_id}/{passage_id}/{script_id}/{word}", warnings,
            )
        elif row.get("score_scope") == "script":
            for field, maximum in SECONDARY_SCORE_FIELDS.items():
                item["secondary"][field] = _parse_score(
                    row.get(field, ""), maximum,
                    f"{rater_id}/{passage_id}/{script_id}/{field}", warnings,
                )
        else:
            warnings.append(f"unknown score_scope for {rater_id}/{passage_id}/{script_id}: {row.get('score_scope')!r}")

    summaries: dict[str, Any] = {}
    for (passage_id, script_id), item in grouped.items():
        words = list(expected_words_by_passage.get(passage_id, [])) if expected_words_by_passage else []
        if not words:
            words = [word for word in item["word_scores"] if word and word != "__script__"]
        missing = [word for word in words if word not in item["word_scores"] or item["word_scores"].get(word) is None]
        if missing:
            warnings.append(f"missing vocabulary score(s) for {rater_id}/{passage_id}/{script_id}: {', '.join(missing)}")
        complete = bool(words) and not missing
        vocab_mean = round(sum(item["word_scores"][word] for word in words) / len(words), 6) if complete else None
        for field in SECONDARY_SCORE_FIELDS:
            if item["secondary"].get(field) is None:
                warnings.append(f"missing score for {rater_id}/{passage_id}/{script_id}/{field}")
        summaries.setdefault(passage_id, {})[script_id] = {
            "vocabulary_scores": item["word_scores"],
            "vocabulary_context_quality_mean": vocab_mean,
            "vocabulary_scored_count": sum(value is not None for value in item["word_scores"].values()),
            **item["secondary"],
            "raw_rows": item["raw_rows"],
        }
    return summaries, warnings


def _agreement_entry(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    if not pairs:
        return {"comparable_count": 0, "exact_count": 0, "percent_agreement": None, "mean_absolute_difference": None}
    exact = sum(a == b for a, b in pairs)
    mad = sum(abs(a - b) for a, b in pairs) / len(pairs)
    return {
        "comparable_count": len(pairs),
        "exact_count": exact,
        "percent_agreement": round(exact / len(pairs), 6),
        "mean_absolute_difference": round(mad, 6),
    }


def summarize_double_rater_scores(
    rater_csv_paths: dict[str, Path | str],
    method_mapping_path: Path | str | dict[str, Any],
    passage_ids: list[str] | None = None,
    expected_words_by_passage: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Validate two blank-or-filled sheets and summarize paired development ratings."""

    all_records: dict[str, list[dict[str, str]]] = {}
    warnings: list[str] = []
    rater_summaries: dict[str, dict[str, Any]] = {}
    for rater_id, path in rater_csv_paths.items():
        records, read_warnings = _read_rater_records(path, rater_id)
        all_records[rater_id] = records
        warnings.extend(read_warnings)
        summary, summary_warnings = _summarize_rater_records(records, rater_id, expected_words_by_passage)
        rater_summaries[rater_id] = summary
        warnings.extend(summary_warnings)

    mapping = _load_method_mapping(method_mapping_path)
    mapped_passages = mapping.get("passages", {})
    ids = passage_ids or sorted({row.get("passage_id", "") for rows in all_records.values() for row in rows if row.get("passage_id")})
    paired: dict[str, Any] = {}
    for passage_id in ids:
        passage_mapping = mapped_passages.get(passage_id, {})
        direct_script = next((script for script, method in passage_mapping.items() if method == "direct"), None)
        plan_script = next((script for script, method in passage_mapping.items() if method == "plan_then_generate"), None)
        if not direct_script or not plan_script:
            warnings.append(f"missing Direct/Plan mapping for {passage_id}")
            continue
        paired[passage_id] = {}
        for rater_id, summaries in rater_summaries.items():
            direct = summaries.get(passage_id, {}).get(direct_script, {})
            planned = summaries.get(passage_id, {}).get(plan_script, {})
            differences: dict[str, float | None] = {}
            for field in ("vocabulary_context_quality_mean", *SECONDARY_SCORE_FIELDS):
                left = direct.get(field)
                right = planned.get(field)
                differences[field] = round(left - right, 6) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
            paired[passage_id][rater_id] = {
                "direct_script_id": direct_script,
                "plan_script_id": plan_script,
                "direct_minus_plan": differences,
            }

    by_metric_pairs: dict[str, list[tuple[int, int]]] = {"vocabulary_context_quality_0_3": [], "vocabulary_context_quality_mean": []}
    by_metric_pairs.update({field: [] for field in SECONDARY_SCORE_FIELDS})
    if len(rater_summaries) >= 2:
        rater_ids = list(rater_summaries)
        left_id, right_id = rater_ids[0], rater_ids[1]
        left_records = {(row.get("passage_id"), row.get("script_id"), row.get("score_scope"), row.get("target_word")): row for row in all_records[left_id]}
        right_records = {(row.get("passage_id"), row.get("script_id"), row.get("score_scope"), row.get("target_word")): row for row in all_records[right_id]}
        for key in sorted(set(left_records) & set(right_records)):
            left_row, right_row = left_records[key], right_records[key]
            if key[2] == "word":
                parsed_left: list[str] = []
                pair: list[tuple[int, int]] = []
                left_value = _parse_score(left_row.get("vocabulary_context_quality_0_3", ""), 3, "agreement", parsed_left)
                right_value = _parse_score(right_row.get("vocabulary_context_quality_0_3", ""), 3, "agreement", parsed_left)
                if left_value is not None and right_value is not None:
                    pair.append((left_value, right_value))
                by_metric_pairs["vocabulary_context_quality_0_3"].extend(pair)
        for passage_id in ids:
            for script_id in {row.get("script_id") for row in all_records[left_id] if row.get("passage_id") == passage_id}:
                left_script = rater_summaries[left_id].get(passage_id, {}).get(script_id, {})
                right_script = rater_summaries[right_id].get(passage_id, {}).get(script_id, {})
                left_mean = left_script.get("vocabulary_context_quality_mean")
                right_mean = right_script.get("vocabulary_context_quality_mean")
                if isinstance(left_mean, (int, float)) and isinstance(right_mean, (int, float)):
                    by_metric_pairs["vocabulary_context_quality_mean"].append((left_mean, right_mean))
                for field in SECONDARY_SCORE_FIELDS:
                    left_value, right_value = left_script.get(field), right_script.get(field)
                    if isinstance(left_value, int) and isinstance(right_value, int):
                        by_metric_pairs[field].append((left_value, right_value))

    agreement = {"by_metric": {field: _agreement_entry(pairs) for field, pairs in by_metric_pairs.items()}}
    comparable = [pair for pairs in by_metric_pairs.values() for pair in pairs]
    agreement["overall"] = _agreement_entry(comparable)
    raw_scores = {rater_id: rows for rater_id, rows in all_records.items()}
    return {
        "analysis_unit": "passage_id",
        "passage_ids": ids,
        "raters": {
            rater_id: {"passages": summaries}
            for rater_id, summaries in rater_summaries.items()
        },
        "paired_comparison": paired,
        "agreement": agreement,
        "raw_scores": raw_scores,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "script_id", "passage_id", "status", "schema_valid", "business_rules_valid", "duration_compliance",
        "duration_is_estimated", "estimated_duration_sec", "duration_assumption_wpm", "total_words",
        "line_count", "character_count", "scene_count", "target_word_count", "target_words_found",
        "vocabulary_coverage_rate", "target_word_forms", "target_word_positions", "source_reference_valid",
        "source_reference_count", "invalid_source_references", "api_call_count", "retry_count", "prompt_tokens",
        "completion_tokens", "total_tokens", "elapsed_ms", "anomaly_count", "semantic_review_required", "validation_warnings",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = {field: metric.get(field, "") for field in fields}
            for field in ("target_word_forms", "target_word_positions", "invalid_source_references", "validation_warnings"):
                row[field] = json.dumps(metric.get(field, {} if "target_word" in field else []), ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def _write_evaluation_sheet(path: Path, passage: dict[str, Any]) -> None:
    rows = make_evaluation_sheet_rows("script_a", passage) + make_evaluation_sheet_rows("script_b", passage)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCRIPT_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


RUBRIC_MARKDOWN = """# C11 剧本质量人工评价 Rubric v1.0

本评价采用匿名配对盲评。评价者只看到 Script A / Script B，不应看到生成方法、Token、耗时、Vocabulary Context Plan 或其他模型评价。主要分析单位是 `passage_id`；五个目标词的评分先汇总为该文章的平均分，不把五个词当成五个独立样本。

## 一、Vocabulary Context Quality（逐词，0–3 分）

对五个目标词分别评分，再计算五词平均值。

| 分数 | 判定标准 |
|---:|---|
| 0 | 目标词缺失、词形不合法、词义用错，或在该语境中明显不成立。 |
| 1 | 词义基本正确，但只是机械出现、罗列、重复原文或没有有效互动。 |
| 2 | 词义正确并处于合理语境，能帮助理解，但语境对词义或记忆的支持有限。 |
| 3 | 词义准确，目标词自然融入人物互动；上下文提供了清楚、具体、适合七年级学习者的理解和记忆线索。 |

不要因为单词“出现了”就自动给高分；要评价它在对话中的语义功能和教学情境。

## 二、次要剧本指标（每项 0–5 分）

### Factual Faithfulness

只根据教材原文和句子编号评价，不把创造性过渡误认为教材事实。

- 0：核心事实错误或与原文矛盾。
- 1：大量事实错误，仅少数片段可对应原文。
- 2：部分核心事实正确，但有明显遗漏或未经支持的关键改写。
- 3：主要事实基本正确，有少量遗漏或轻微改写。
- 4：核心事实准确，遗漏很少，创造内容与教材事实边界清楚。
- 5：核心事实完整、准确，引用和改写均与原文一致，创造性内容不造成混淆。

### Dialogue Naturalness

- 0：不是可理解的对话，或台词严重不连贯。
- 1：大量机械罗列、突兀回应或不自然表达。
- 2：能形成基本对话，但多处像说明文字或目标词练习题。
- 3：大部分回应自然，少数台词略显生硬或重复。
- 4：互动自然、回应有目的，只有轻微不自然处。
- 5：对话流畅、角色回应真实，目标词融入不显刻意。

### Grade-level Suitability

- 0：明显超出七年级水平，几乎无法理解。
- 1：大量生词、复杂句式或不适合学习者的表达。
- 2：部分可理解，但仍有较多超出水平的语言负担。
- 3：整体适合七年级，少量表达需要教师解释。
- 4：词汇、句式和信息量大体适合七年级。
- 5：语言清楚、负担适中，特别适合中国七年级英语学习者配音和学习。

### Story Coherence

- 0：场景和台词无法组成连贯事件。
- 1：存在多个严重跳跃或角色关系不清。
- 2：有基本主题，但转场、动机或结尾明显断裂。
- 3：主线可理解，少数转场或因果关系较弱。
- 4：事件顺序清晰，场景和人物互动服务于主题。
- 5：结构紧凑、转场自然、结尾完整，且没有为了覆盖目标词而破坏故事。

## 三、盲评记录规则

- 先独立完成 Script A 和 Script B，再查看配对差异。
- 对事实判断必须回看原文句子编号。
- 无法确定时保留评论并选择最保守的分数，不要根据生成方法猜测质量。
- 空白分数表示未完成评价；汇总工具不会用默认值或模型分数补齐。
- 若两名或多名评价者参与，应保存各自原始表，再另行计算一致性；本模板不把评价者当作独立 passage 样本。
"""


def _write_comparison_template(path: Path, passage_id: str = "p005") -> None:
    path.write_text(f"""# {passage_id} v1.1 盲评比较模板\n\n本文件只提供分析结构，不包含虚构的人工评分。自动指标见 `automatic_metrics.csv`，人工评分填写完成后再恢复 `method_mapping.json`。\n\n## 配对分析单位\n\n- passage_id：`{passage_id}`\n- Script A 与 Script B 为同一篇教材文章的配对样本。\n- 五个词的词汇质量分数先汇总为文章级平均值。\n\n## 人工评分\n\n| 指标 | Script A | Script B | A - B |\n|---|---:|---:|---:|\n| Vocabulary Context Quality（五词平均，0–3） | 待填写 | 待填写 | 待计算 |\n| Factual Faithfulness（0–5） | 待填写 | 待填写 | 待计算 |\n| Dialogue Naturalness（0–5） | 待填写 | 待填写 | 待计算 |\n| Grade-level Suitability（0–5） | 待填写 | 待填写 | 待计算 |\n| Story Coherence（0–5） | 待填写 | 待填写 | 待计算 |\n\n不得根据单篇结果宣称方法优劣。\n""", encoding="utf-8")


def _load_generation_pair(project_root: Path, passage_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any] | None], dict[str, dict[str, Any]]]:
    passage = load_passage(project_root / "data" / "v0.3" / "dev.json", passage_id)
    base = project_root / "outputs" / "generation_experiments" / "v1.1" / passage_id
    scripts: dict[str, dict[str, Any] | None] = {}
    logs: dict[str, dict[str, Any]] = {}
    for method, filename in (("direct", "script_direct.json"), ("plan_then_generate", "script_planned.json")):
        method_dir = base / method
        script_path = method_dir / filename
        log_path = method_dir / "generation_log.json"
        scripts[method] = json.loads(script_path.read_text(encoding="utf-8")) if script_path.exists() else None
        logs[method] = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"stats": {}}
    return passage, scripts, logs


def build_passage_evaluation(
    project_root: Path | str,
    passage_id: str,
    seed: int,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Build one passage's analyst metrics and independently seeded blind materials."""

    project_root = Path(project_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    passage, method_scripts, method_logs = _load_generation_pair(project_root, passage_id)
    mapping = make_anonymous_mapping(seed)
    anonymous_scripts = {script_id: method_scripts[method] for script_id, method in mapping.items()}
    metrics: list[dict[str, Any]] = []
    for script_id, method in mapping.items():
        item = compute_automatic_metrics(anonymous_scripts[script_id], passage, method_logs[method], project_root)
        item["script_id"] = script_id
        item["anomaly_count"] = _count_anomalies(method_logs[method])
        metrics.append(item)
    _write_metrics_csv(output / "automatic_metrics.csv", metrics)
    _write_json(output / "anonymous_scripts.json", make_anonymous_bundle(passage, anonymous_scripts))
    _write_evaluation_sheet(output / "evaluation_sheet.csv", passage)
    (output / "evaluation_rubric.md").write_text(RUBRIC_MARKDOWN, encoding="utf-8")
    _write_comparison_template(output / "comparison_template.md", passage_id)
    _write_json(output / "method_mapping.json", {
        "warning": "Analyst-only file. Do not provide to blind evaluators.",
        "passage_id": passage_id,
        "seed": seed,
        **mapping,
    })
    _write_json(output / "manual_score_summary.json", summarize_manual_scores(output / "evaluation_sheet.csv", passage_id))
    return {
        "passage": passage,
        "mapping": mapping,
        "metrics": metrics,
        "anonymous_bundle": make_anonymous_bundle(passage, anonymous_scripts),
    }


def _write_development_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "passage_id", "method", "script_id", "status", "schema_valid", "business_rules_valid",
        "vocabulary_coverage_rate", "target_words_found", "target_word_count", "duration_compliance",
        "estimated_duration_sec", "total_words", "line_count", "character_count", "scene_count",
        "source_reference_valid", "api_call_count", "retry_count", "prompt_tokens", "completion_tokens",
        "total_tokens", "elapsed_ms", "anomaly_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_development_evaluation(
    project_root: Path | str = ROOT,
    passage_ids: list[str] | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build the five-passage development summary without reading data/v0.3/test.json."""

    project_root = Path(project_root)
    ids = passage_ids or ["p001", "p003", "p005", "p009", "p012"]
    output = Path(output_dir) if output_dir is not None else project_root / "outputs" / "evaluation" / "v1.1" / "development"
    output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    all_anon: list[dict[str, Any]] = []
    all_sheet_rows: list[dict[str, str]] = []
    mapping_record: dict[str, Any] = {}
    seeds = {passage_id: 4200 + index for index, passage_id in enumerate(ids, start=1)}
    for passage_id in ids:
        result = build_passage_evaluation(project_root, passage_id, seeds[passage_id], output / passage_id)
        mapping_record[passage_id] = {"seed": seeds[passage_id], **result["mapping"]}
        bundle = result["anonymous_bundle"]
        all_anon.append(bundle)
        all_sheet_rows.extend(make_evaluation_sheet_rows("script_a", result["passage"]))
        all_sheet_rows.extend(make_evaluation_sheet_rows("script_b", result["passage"]))
        inverse = {script_id: method for script_id, method in result["mapping"].items()}
        for metric in result["metrics"]:
            item = dict(metric)
            item["method"] = inverse[metric["script_id"]]
            item["anomaly_count"] = item.get("anomaly_count", 0)
            all_metrics.append(item)

    _write_development_metrics_csv(output / "dev_automatic_metrics.csv", all_metrics)
    _write_development_metrics_csv(output / "dev_generation_summary.csv", all_metrics)
    _write_json(output / "dev_anonymous_evaluation.json", {
        "bundle_version": "1.0",
        "analysis_unit": "passage_id",
        "passages": all_anon,
    })
    _write_json(output / "dev_method_mapping.json", {
        "warning": "Analyst-only file. Do not provide to blind evaluators.",
        "passages": mapping_record,
    })
    with (output / "dev_evaluation_sheet.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCRIPT_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(all_sheet_rows)
    status_counts = {"validated": sum(item["status"] == "validated" for item in all_metrics), "failed": sum(item["status"] != "validated" for item in all_metrics)}
    total_calls = sum(item.get("api_call_count", 0) for item in all_metrics)
    total_tokens = sum(item.get("total_tokens", 0) for item in all_metrics)
    total_elapsed = round(sum(item.get("elapsed_ms", 0) for item in all_metrics), 3)
    duration_passes = sum(bool(item.get("duration_compliance")) for item in all_metrics)
    table = [
        "| passage_id | method | schema | coverage | duration | words | lines | calls | total_tokens | elapsed_ms | anomalies |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(all_metrics, key=lambda row: (row["passage_id"], row["method"])):
        table.append(
            f"| {item['passage_id']} | {item['method']} | {'PASS' if item['schema_valid'] else 'FAIL'} | "
            f"{item['target_words_found']}/{item['target_word_count']} | "
            f"{'PASS' if item['duration_compliance'] else 'FAIL'} ({item.get('estimated_duration_sec')}) | "
            f"{item.get('total_words', 0)} | {item.get('line_count', 0)} | {item.get('api_call_count', 0)} | "
            f"{item.get('total_tokens', 0)} | {item.get('elapsed_ms', 0)} | {item.get('anomaly_count', 0)} |"
        )
    (output / "dev_experiment_report.md").write_text(
        "# C11 开发集 v1.1 自动评价汇总\n\n"
        "本报告只包含 P001、P003、P005、P009、P012 的离线自动指标。没有调用 API，没有填写人工评分，也不包含测试集。\n\n"
        f"- passage 数：`{len(ids)}`\n- 方法-文章结果数：`{len(all_metrics)}`\n"
        f"- 结构校验通过：`{status_counts['validated']}`\n- 失败或缺失：`{status_counts['failed']}`\n"
        f"- Duration Compliance 通过：`{duration_passes}/{len(all_metrics)}`\n"
        f"- API 调用总数：`{total_calls}`；Total Tokens：`{total_tokens}`；耗时：`{total_elapsed} ms`\n"
        "- 人工评价：待填写；不得根据自动指标直接宣称方法优劣。\n"
        "- 分析单位：`passage_id`；五个目标词不是五个独立实验样本。\n\n"
        "## 逐文章自动指标\n\n" + "\n".join(table) + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output), "metrics": all_metrics, "mapping": mapping_record}


def build_p005_evaluation(project_root: Path | str = ROOT, output_dir: Path | str | None = None) -> dict[str, Any]:
    project_root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else project_root / "outputs" / "evaluation" / "v1.1" / "p005"
    output.mkdir(parents=True, exist_ok=True)
    passage = load_passage(project_root / "data" / "v0.3" / "dev.json", "p005")
    scripts = {
        "script_a": json.loads((project_root / "outputs" / "generation_experiments" / "v1.1" / "p005" / "direct" / "script_direct.json").read_text(encoding="utf-8")),
        "script_b": json.loads((project_root / "outputs" / "generation_experiments" / "v1.1" / "p005" / "plan_then_generate" / "script_planned.json").read_text(encoding="utf-8")),
    }
    logs = {
        "script_a": json.loads((project_root / "outputs" / "generation_experiments" / "v1.1" / "p005" / "direct" / "generation_log.json").read_text(encoding="utf-8")),
        "script_b": json.loads((project_root / "outputs" / "generation_experiments" / "v1.1" / "p005" / "plan_then_generate" / "generation_log.json").read_text(encoding="utf-8")),
    }
    metrics = []
    for script_id in ("script_a", "script_b"):
        item = compute_automatic_metrics(scripts[script_id], passage, logs[script_id], project_root)
        item["script_id"] = script_id
        metrics.append(item)
    _write_metrics_csv(output / "automatic_metrics.csv", metrics)
    _write_json(output / "anonymous_scripts.json", make_anonymous_bundle(passage, scripts))
    _write_evaluation_sheet(output / "evaluation_sheet.csv", passage)
    (output / "evaluation_rubric.md").write_text(RUBRIC_MARKDOWN, encoding="utf-8")
    _write_comparison_template(output / "comparison_template.md")
    _write_json(output / "method_mapping.json", {
        "warning": "Analyst-only file. Do not provide to blind evaluators.",
        "passage_id": "p005",
        "script_a": "direct",
        "script_b": "plan_then_generate",
    })
    _write_json(output / "manual_score_summary.json", summarize_manual_scores(output / "evaluation_sheet.csv", "p005"))
    return {"output_dir": str(output), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--development", action="store_true", help="build the five-passage development evaluation package")
    args = parser.parse_args()
    result = (
        build_development_evaluation(args.project_root, output_dir=args.output_dir)
        if args.development
        else build_p005_evaluation(args.project_root, args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
