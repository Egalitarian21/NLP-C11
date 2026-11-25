"""Non-blocking source-evidence hints for Story-driven v2.2 review."""

from __future__ import annotations

import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z]+")
STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "but", "can", "do", "for", "from", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "our", "the", "their", "this", "to", "we", "with", "you",
}


def _content_tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text) if token.casefold() not in STOPWORDS}


def review_source_evidence(script: dict[str, Any], passage: dict[str, Any]) -> dict[str, Any]:
    """Return traceability errors and lexical review hints without judging semantics."""
    sentence_map = {
        item.get("id"): item.get("text", "")
        for item in passage.get("sentences", [])
        if isinstance(item, dict)
    }
    errors: list[dict[str, str]] = []
    weak: list[str] = []
    empty: list[str] = []
    hints: list[dict[str, Any]] = []
    for scene in script.get("scenes", []) if isinstance(script, dict) else []:
        for line in scene.get("dialogue", []) if isinstance(scene, dict) else []:
            if not isinstance(line, dict):
                continue
            line_id = str(line.get("line_id", "<missing>"))
            text = line.get("text", "") if isinstance(line.get("text", ""), str) else ""
            source_ids = line.get("source_sentence_ids", [])
            if not source_ids:
                empty.append(line_id)
                continue
            invalid = [source_id for source_id in source_ids if source_id not in sentence_map]
            if invalid:
                errors.append({"line_id": line_id, "message": f"unknown source sentence IDs: {invalid}"})
                continue
            line_tokens = _content_tokens(text)
            source_tokens = _content_tokens(" ".join(sentence_map[source_id] for source_id in source_ids))
            overlap = len(line_tokens & source_tokens) / max(len(line_tokens), 1)
            if overlap < 0.35:
                weak.append(line_id)
                hints.append({
                    "line_id": line_id,
                    "source_sentence_ids": list(source_ids),
                    "message": "Cited source text has limited lexical support for the line; human semantic review required.",
                    "content_token_overlap": round(overlap, 3),
                })
    return {
        "non_blocking": True,
        "errors": errors,
        "weak_support_line_ids": weak,
        "empty_evidence_line_ids": empty,
        "hints": hints,
    }
