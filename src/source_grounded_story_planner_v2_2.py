"""Build and validate the isolated v2.2 source-grounded story plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "source-grounded-story-planning-v2.2"
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "source_grounded_story_planning_v2_2.txt"
DEFAULT_SCHEMA_PATH = ROOT / "schemas" / "source_grounded_story_plan_v2_2.schema.json"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "script_generation_v2_2.json"
WORD_FIELDS = {"word", "meaning_in_context", "source_sentence_ids", "suggested_dialogue_function", "suggested_story_use"}
CLAIM_FIELDS = {"claim_id", "factual_claim", "source_sentence_ids", "source_statement_type", "importance", "relationship_to_story"}
BEAT_FIELDS = {
    "beat_id", "narrative_function", "character_goal", "trigger_or_obstacle", "character_action",
    "turning_point_or_change", "causal_outcome", "relevant_claim_ids", "target_words_to_integrate",
}
BEAT_FUNCTIONS = ["setup_character_goal", "small_obstacle_or_curiosity", "development_turning_point", "resolution"]


def load_passage_from_dataset(dataset_path: Path | str, passage_id: str) -> dict[str, Any]:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    for passage in data:
        if passage.get("passage_id") == passage_id:
            return passage
    raise ValueError(f"Passage not found: {passage_id}")


def _issue(path: str, message: str) -> dict[str, str]:
    return {"path": path, "message": message}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def build_story_plan_prompt(
    passage: dict[str, Any],
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8").strip()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    sentences = "\n".join(f"[{item['id']}] {item['text']}" for item in passage["sentences"])
    return (
        template
        + "\n\nAUTHORITATIVE INPUT DATA\n"
        + f"passage_id: {passage['passage_id']}\n"
        + f"title: {passage['title']}\n"
        + f"language: {passage.get('language', 'en')}\n"
        + f"learner_level: {passage.get('learner_level', 'grade_7')}\n"
        + f"target_words: {json.dumps(passage['target_words'], ensure_ascii=False)}\n"
        + f"constraints: {json.dumps(passage.get('constraints', {}), ensure_ascii=False, sort_keys=True)}\n"
        + "complete numbered textbook passage:\n"
        + sentences
        + "\n\nSHARED FINAL SCRIPT CONFIGURATION\n"
        + json.dumps(config, ensure_ascii=False, indent=2)
        + "\n\nPLANNING JSON SCHEMA\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\nReturn only the complete v2.2 story-plan JSON object."
    )


def validate_story_plan(plan: Any, passage: dict[str, Any], schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Validate v2.2 structure and source/cross references; semantic quality stays human-reviewed."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("story-plan schema must be a JSON object")
    errors: list[dict[str, str]] = []
    expected_words = list(passage.get("target_words", []))
    sentence_ids = {item.get("id") for item in passage.get("sentences", []) if isinstance(item, dict)}
    if not isinstance(plan, dict):
        return _result([_issue("$", "plan must be a JSON object")], 0, 0, 0, 0)

    allowed_top = {"plan_version", "passage_id", "target_words", "vocabulary_grounding", "source_claims", "story_beats"}
    for key in set(plan) - allowed_top:
        errors.append(_issue(f"$.{key}", "unexpected field"))
    if plan.get("plan_version") != "2.2":
        errors.append(_issue("$.plan_version", "must equal '2.2'"))
    if plan.get("passage_id") != passage.get("passage_id"):
        errors.append(_issue("$.passage_id", "must equal the input passage_id"))
    if plan.get("target_words") != expected_words:
        errors.append(_issue("$.target_words", "must exactly match the V0.3 target_words order and values"))

    vocabulary = plan.get("vocabulary_grounding")
    if not isinstance(vocabulary, list) or len(vocabulary) != 5:
        errors.append(_issue("$.vocabulary_grounding", "must contain exactly five entries"))
        vocabulary = vocabulary if isinstance(vocabulary, list) else []
    seen_words: list[str] = []
    for index, entry in enumerate(vocabulary):
        path = f"$.vocabulary_grounding[{index}]"
        if not isinstance(entry, dict):
            errors.append(_issue(path, "must be an object"))
            continue
        for key in set(entry) - WORD_FIELDS:
            errors.append(_issue(f"{path}.{key}", "unexpected field"))
        word = entry.get("word")
        if not isinstance(word, str) or not word.isalpha() or not word.isascii():
            errors.append(_issue(f"{path}.word", "must be one ASCII alphabetic word"))
        else:
            seen_words.append(word)
        for field in ["meaning_in_context", "suggested_dialogue_function", "suggested_story_use"]:
            if not _nonempty(entry.get(field)):
                errors.append(_issue(f"{path}.{field}", "must not be empty"))
        _check_source_ids(entry.get("source_sentence_ids"), sentence_ids, f"{path}.source_sentence_ids", errors)
    if len(seen_words) != len(set(seen_words)) or set(seen_words) != set(expected_words):
        errors.append(_issue("$.vocabulary_grounding", "each target word must occur exactly once"))

    claims = plan.get("source_claims")
    if not isinstance(claims, list) or not claims:
        errors.append(_issue("$.source_claims", "must contain at least one source-backed claim"))
        claims = claims if isinstance(claims, list) else []
    claim_ids: list[str] = []
    for index, claim in enumerate(claims):
        path = f"$.source_claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(_issue(path, "must be an object"))
            continue
        for key in set(claim) - CLAIM_FIELDS:
            errors.append(_issue(f"{path}.{key}", "unexpected field"))
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.startswith("claim_"):
            errors.append(_issue(f"{path}.claim_id", "must use claim_<number>"))
        else:
            claim_ids.append(claim_id)
        if not _nonempty(claim.get("factual_claim")):
            errors.append(_issue(f"{path}.factual_claim", "must not be empty"))
        if claim.get("source_statement_type") not in {"fact", "question", "invitation", "opinion_or_promise"}:
            errors.append(_issue(f"{path}.source_statement_type", "must distinguish source statement type"))
        if claim.get("importance") not in {"essential", "supporting"}:
            errors.append(_issue(f"{path}.importance", "must be essential or supporting"))
        if not _nonempty(claim.get("relationship_to_story")):
            errors.append(_issue(f"{path}.relationship_to_story", "must not be empty"))
        _check_source_ids(claim.get("source_sentence_ids"), sentence_ids, f"{path}.source_sentence_ids", errors)
    if len(claim_ids) != len(set(claim_ids)):
        errors.append(_issue("$.source_claims", "claim_id values must be unique"))
    claim_id_set = set(claim_ids)

    beats = plan.get("story_beats")
    if not isinstance(beats, list) or len(beats) != 4:
        errors.append(_issue("$.story_beats", "must contain exactly four connected beats"))
        beats = beats if isinstance(beats, list) else []
    beat_ids: list[str] = []
    functions: list[str] = []
    causal_fields = ["character_goal", "trigger_or_obstacle", "character_action", "turning_point_or_change", "causal_outcome"]
    for index, beat in enumerate(beats):
        path = f"$.story_beats[{index}]"
        if not isinstance(beat, dict):
            errors.append(_issue(path, "must be an object"))
            continue
        for key in set(beat) - BEAT_FIELDS:
            errors.append(_issue(f"{path}.{key}", "unexpected field"))
        beat_id = beat.get("beat_id")
        if not isinstance(beat_id, str) or not beat_id.startswith("beat_"):
            errors.append(_issue(f"{path}.beat_id", "must use beat_<number>"))
        else:
            beat_ids.append(beat_id)
        function = beat.get("narrative_function")
        if function not in BEAT_FUNCTIONS:
            errors.append(_issue(f"{path}.narrative_function", "must be one of the four required functions"))
        else:
            functions.append(function)
        for field in causal_fields:
            if not _nonempty(beat.get(field)):
                errors.append(_issue(f"{path}.{field}", "must not be empty"))
        for claim_id in beat.get("relevant_claim_ids", []) if isinstance(beat.get("relevant_claim_ids"), list) else []:
            if claim_id not in claim_id_set:
                errors.append(_issue(f"{path}.relevant_claim_ids", f"unknown claim {claim_id!r}"))
        for word in beat.get("target_words_to_integrate", []) if isinstance(beat.get("target_words_to_integrate"), list) else []:
            if word not in set(expected_words):
                errors.append(_issue(f"{path}.target_words_to_integrate", f"word {word!r} is not a supplied target word"))
    if len(beat_ids) != len(set(beat_ids)):
        errors.append(_issue("$.story_beats", "beat_id values must be unique"))
    if functions and functions != BEAT_FUNCTIONS:
        errors.append(_issue("$.story_beats", "beats must appear in setup, obstacle/curiosity, development/turning-point, resolution order"))

    return _result(errors, len(vocabulary), len(claims), len(beats), len(set(seen_words) & set(expected_words)))


def _check_source_ids(value: Any, valid_ids: set[str], path: str, errors: list[dict[str, str]]) -> None:
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        errors.append(_issue(path, "must be a non-empty unique array"))
        return
    for source_id in value:
        if source_id not in valid_ids:
            errors.append(_issue(path, f"unknown source sentence {source_id!r}"))


def _result(errors: list[dict[str, str]], vocabulary_count: int, claim_count: int, beat_count: int, word_count: int) -> dict[str, Any]:
    return {
        "valid": not errors,
        "schema_valid": not errors,
        "errors": errors,
        "warnings": [],
        "semantic_review_required": True,
        "semantic_review": [
            "Valid source IDs establish traceability but do not prove semantic factual faithfulness.",
            "Causal coherence, meaning-in-context, and the distinction between source fact and creative transition require independent human review.",
        ],
        "stats": {
            "vocabulary_mapping_count": vocabulary_count,
            "target_words_found": word_count,
            "source_claim_count": claim_count,
            "story_beat_count": beat_count,
        },
    }
