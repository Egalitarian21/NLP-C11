"""Schema and input-aware validation for C11 English skits.

The validator deliberately separates deterministic checks from semantic review.
It can confirm structure, IDs, source references, target-word coverage, and
conservative inflectional matching. It cannot prove factual faithfulness or
naturalness without a human or independent model review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = ROOT / "schemas" / "script.schema.json"
EMOTIONS = {"neutral", "happy", "sad", "curious", "surprised", "excited"}
ACTIONS = {"none", "nod", "point", "wave", "look_at", "walk"}
DEFAULT_WPM = 130
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _issue(category: str, path: str, message: str) -> dict[str, str]:
    return {"category": category, "path": path, "message": message}


def _resolve_ref(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported schema reference: {ref}")
    current: Any = schema
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def _validate_schema_node(value: Any, schema: dict[str, Any], root_schema: dict[str, Any], path: str, issues: list[dict[str, str]]) -> None:
    if "$ref" in schema:
        _validate_schema_node(value, _resolve_ref(root_schema, schema["$ref"]), root_schema, path, issues)
        return
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        issues.append(_issue("structural", path, f"expected {expected}"))
        return
    if "const" in schema and value != schema["const"]:
        issues.append(_issue("structural", path, f"must equal {schema['const']!r}"))
    if "enum" in schema and value not in schema["enum"]:
        issues.append(_issue("structural", path, f"must be one of {schema['enum']}"))
    if isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0):
            issues.append(_issue("structural", path, "must not be empty"))
        pattern = schema.get("pattern")
        if pattern and not re.fullmatch(pattern, value):
            issues.append(_issue("structural", path, f"does not match pattern {pattern!r}"))
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            issues.append(_issue("structural", path, f"requires at least {schema['minItems']} item(s)"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(_issue("structural", path, f"allows at most {schema['maxItems']} item(s)"))
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(serialized) != len(set(serialized)):
                issues.append(_issue("structural", path, "items must be unique"))
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema_node(item, item_schema, root_schema, f"{path}[{index}]", issues)
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                issues.append(_issue("structural", f"{path}.{key}", "is required"))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    issues.append(_issue("structural", f"{path}.{key}", "is not allowed"))
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_node(value[key], child_schema, root_schema, f"{path}.{key}", issues)


def _load_schema(schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def _inflection_forms(word: str) -> set[str]:
    """Return conservative, regular forms; never match arbitrary substrings."""

    base = word.casefold()
    forms = {base}
    if base.endswith("y") and len(base) > 1:
        forms.add(base[:-1] + "ies")
        forms.add(base[:-1] + "ied")
    elif base.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(base + "es")
        forms.add(base + "ed")
    else:
        forms.add(base + "s")
        forms.add(base + "ed")
    if base.endswith("e") and not base.endswith("ee"):
        forms.add(base[:-1] + "ing")
    else:
        forms.add(base + "ing")
    if len(base) >= 3 and base[-1] not in "aeiouywx" and base[-2] in "aeiou" and base[-3] not in "aeiou":
        forms.add(base + base[-1] + "ing")
        forms.add(base + base[-1] + "ed")
    return forms


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _line_items(script: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for scene in script.get("scenes", []) if isinstance(script, dict) else []:
        for line in scene.get("dialogue", []) if isinstance(scene, dict) else []:
            if isinstance(line, dict):
                items.append((scene.get("scene_id", ""), line))
    return items


def validate_script(script: dict[str, Any], passage: dict[str, Any], schema_path: Path | str = DEFAULT_SCHEMA_PATH, words_per_minute: int = DEFAULT_WPM) -> dict[str, Any]:
    """Validate one generated script against one V0.3 passage.

    Returns a JSON-serializable report. ``valid`` means all deterministic
    structural and rule checks passed; ``semantic_review_required`` remains
    true because factual faithfulness and naturalness are not fully automatic.
    """

    schema = _load_schema(schema_path)
    structural: list[dict[str, str]] = []
    if not isinstance(script, dict):
        structural.append(_issue("structural", "$", "script must be a JSON object"))
        return {"valid": False, "schema_valid": False, "errors": structural, "semantic_review_required": True, "semantic_review": ["Cannot assess semantic faithfulness before structure is an object."], "warnings": [], "stats": {}}
    _validate_schema_node(script, schema, schema, "$", structural)
    rules: list[dict[str, str]] = []

    expected_id = passage.get("passage_id")
    if script.get("passage_id") != expected_id:
        rules.append(_issue("rule", "$.passage_id", f"passage_id must equal input {expected_id!r}"))
    expected_targets = passage.get("target_words", [])
    if script.get("target_words") != expected_targets:
        rules.append(_issue("rule", "$.target_words", "target_words must exactly match the V0.3 input order and values"))

    characters = script.get("characters", []) if isinstance(script.get("characters", []), list) else []
    character_ids = [item.get("id") for item in characters if isinstance(item, dict)]
    if len(character_ids) != len(set(character_ids)):
        rules.append(_issue("rule", "$.characters", "character ids must be unique"))
    max_characters = int(passage.get("constraints", {}).get("max_characters", 3))
    if len(characters) > max_characters:
        rules.append(_issue("rule", "$.characters", f"character count exceeds input max_characters={max_characters}"))

    scenes = script.get("scenes", []) if isinstance(script.get("scenes", []), list) else []
    scene_ids = [item.get("scene_id") for item in scenes if isinstance(item, dict)]
    if len(scene_ids) != len(set(scene_ids)):
        rules.append(_issue("rule", "$.scenes", "scene ids must be unique"))
    max_scenes = int(passage.get("constraints", {}).get("max_scenes", 3))
    if len(scenes) > max_scenes:
        rules.append(_issue("rule", "$.scenes", f"scene count exceeds input max_scenes={max_scenes}"))

    lines = _line_items(script)
    line_ids = [line.get("line_id") for _, line in lines]
    if len(line_ids) != len(set(line_ids)):
        rules.append(_issue("rule", "$.scenes[*].dialogue[*].line_id", "line_id must be unique across the entire script"))
    valid_sentence_ids = {sentence.get("id") for sentence in passage.get("sentences", []) if isinstance(sentence, dict)}
    for scene_id, line in lines:
        path = f"scene={scene_id}/line={line.get('line_id', '<missing>')}"
        if line.get("speaker_id") not in set(character_ids):
            rules.append(_issue("rule", path, f"speaker_id {line.get('speaker_id')!r} does not refer to a character"))
        for sentence_id in line.get("source_sentence_ids", []) if isinstance(line.get("source_sentence_ids"), list) else []:
            if sentence_id not in valid_sentence_ids:
                rules.append(_issue("rule", path, f"source_sentence_ids contains invalid id {sentence_id!r}"))

    text = " ".join(line.get("text", "") for _, line in lines if isinstance(line.get("text", ""), str))
    token_list = _tokens(text)
    found_forms: dict[str, list[str]] = {}
    for target in expected_targets:
        forms = _inflection_forms(target)
        found = sorted({token for token in token_list if token in forms})
        found_forms[target] = found
        if not found:
            rules.append(_issue("rule", "$.scenes[*].dialogue[*].text", f"Missing target word {target!r}; no legal whole-word form found"))

    total_words = len(token_list)
    estimated_duration = round(total_words / max(words_per_minute, 1) * 60, 2)
    warnings: list[dict[str, str]] = []
    constraints = passage.get("constraints", {})
    min_duration = constraints.get("min_duration_sec")
    max_duration = constraints.get("max_duration_sec")
    if min_duration is not None and estimated_duration < min_duration or max_duration is not None and estimated_duration > max_duration:
        warnings.append(_issue("warning", "$.scenes", f"estimated_duration_sec={estimated_duration} is outside the requested range; this is not an audio measurement"))
    semantic_review = [
        "Fact-level faithfulness to the passage and naturalness of target-word use require human or independent-model review.",
        "source_sentence_ids provide evidence references but do not by themselves prove that a line preserves every source fact.",
    ]
    return {
        "valid": not structural and not rules,
        "schema_valid": not structural,
        "errors": structural + rules,
        "structural_errors": structural,
        "rule_errors": rules,
        "warnings": warnings,
        "semantic_review_required": True,
        "semantic_review": semantic_review,
        "stats": {
            "character_count": len(characters),
            "scene_count": len(scenes),
            "line_count": len(lines),
            "total_words": total_words,
            "target_words_expected": len(expected_targets),
            "target_words_found": sum(bool(found_forms[word]) for word in expected_targets),
            "target_word_forms": found_forms,
            "estimated_duration_sec": estimated_duration,
            "estimated_duration_is_not_audio": True,
            "duration_assumption_wpm": words_per_minute,
        },
    }


def load_passage(dataset_path: Path | str, passage_id: str) -> dict[str, Any]:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    for passage in data:
        if passage.get("passage_id") == passage_id:
            return passage
    raise ValueError(f"Passage not found: {passage_id}")


def write_markdown_report(result: dict[str, Any], output_path: Path | str, script_path: str, passage_id: str) -> None:
    lines = [
        f"# Interface validation report: {passage_id}",
        "",
        f"Script: `{script_path}`",
        f"Deterministic validation: **{'PASS' if result['valid'] else 'FAIL'}**",
        "",
        "## Statistics",
        "",
    ]
    for key, value in result.get("stats", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Structural and rule errors", ""]
    lines.extend(f"- [{item['category']}] `{item['path']}`: {item['message']}" for item in result.get("errors", []))
    if not result.get("errors"):
        lines.append("- None")
    lines += ["", "## Non-blocking warnings", ""]
    lines.extend(f"- {item['message']}" for item in result.get("warnings", []))
    if not result.get("warnings"):
        lines.append("- None")
    lines += ["", "## Semantic review required", ""]
    lines.extend(f"- {item}" for item in result.get("semantic_review", []))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--passage-id", required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "v0.3" / "dataset.json")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "interface_validation_report.md")
    args = parser.parse_args()
    script = json.loads(args.script.read_text(encoding="utf-8"))
    passage = load_passage(args.dataset, args.passage_id)
    result = validate_script(script, passage, args.schema)
    write_markdown_report(result, args.report, str(args.script), args.passage_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
