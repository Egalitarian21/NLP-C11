"""Audits for prompt provenance and unexpected model-output characters."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


ALLOWED_NON_ASCII_PUNCTUATION = {"…", "“", "”", "‘", "’", "—", "–", "‑", "×", "·", "©"}
FORBIDDEN_PROMPT_MARKERS = ("examples/mock_script.json", "script_direct.json", "Welcome to our school club fair!")


def scan_text_anomalies(value: Any, path: str = "$") -> list[dict[str, str]]:
    """Find suspicious characters without treating U+2026 as an error."""
    findings: list[dict[str, str]] = []

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{item_path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")
        elif isinstance(item, str):
            for char in item:
                codepoint = ord(char)
                category = unicodedata.category(char)
                if char in ALLOWED_NON_ASCII_PUNCTUATION:
                    continue
                if char == "\ufffd" or category.startswith("C") and char not in "\n\r\t":
                    findings.append({"path": item_path, "character": char, "codepoint": f"U+{codepoint:04X}", "reason": "replacement or control character"})
                elif codepoint > 127 and category.startswith("L"):
                    findings.append({"path": item_path, "character": char, "codepoint": f"U+{codepoint:04X}", "reason": "unexpected non-ASCII letter in English output"})

    visit(value, path)
    return findings


def audit_request_inputs(request: dict[str, Any], passage: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check that a recorded request contains the intended passage inputs."""
    text = json.dumps(request.get("messages", []), ensure_ascii=False)
    expected_targets = list(passage.get("target_words", []))
    errors: list[str] = []
    if request.get("target_words") != expected_targets:
        errors.append("request target_words do not exactly match V0.3 input")
    missing_targets = [word for word in expected_targets if not re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", text, re.IGNORECASE)]
    if missing_targets:
        errors.append("prompt text is missing target words: " + ", ".join(missing_targets))
    missing_sentence_ids: list[str] = []
    for sentence in passage.get("sentences", []):
        if f"[{sentence['id']}] {sentence['text']}" not in text:
            missing_sentence_ids.append(sentence["id"])
    if missing_sentence_ids:
        errors.append("prompt text is missing authoritative sentence evidence")
    if plan is not None:
        if request.get("generation_method") != "plan_then_generate":
            errors.append("planned request has the wrong generation method")
        for context in plan.get("vocabulary_contexts", []):
            if context.get("word") not in text:
                errors.append(f"planned prompt is missing context for {context.get('word')!r}")
            for source_id in context.get("source_sentence_ids", []):
                if source_id not in text:
                    errors.append(f"planned prompt is missing plan source id {source_id!r}")
    forbidden = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in text]
    if forbidden:
        errors.append("prompt contains forbidden prior-artifact marker(s): " + ", ".join(forbidden))
    return {
        "valid": not errors,
        "errors": errors,
        "target_words": expected_targets,
        "missing_target_words": missing_targets,
        "missing_sentence_ids": missing_sentence_ids,
        "forbidden_markers": forbidden,
        "anomalies": scan_text_anomalies(request.get("messages", [])),
    }
