"""Non-blocking heuristic diagnostics for story-driven skits."""

from __future__ import annotations

import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
GOAL_CUES = ("want to", "need to", "would like to", "let's", "let us", "choose", "decide")
OBSTACLE_CUES = ("but", "however", "wonder", "what if", "not sure", "can't", "cannot", "problem", "which")
TURN_CUES = ("then", "suddenly", "discover", "discovered", "find", "found", "realize", "realized", "try", "decide")
RESOLUTION_CUES = ("so", "let's", "let us", "we can", "now we", "finally", "agree", "will join")


def _words(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(cue in lowered for cue in cues)


def diagnose_story(script: dict[str, Any], passage: dict[str, Any]) -> dict[str, Any]:
    lines = [
        line
        for scene in script.get("scenes", [])
        for line in scene.get("dialogue", [])
        if isinstance(line, dict) and isinstance(line.get("text"), str)
    ]
    texts = [line["text"] for line in lines]
    full_text = " ".join(texts)
    target_locations: dict[str, list[str]] = {}
    for target in passage.get("target_words", []):
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(target) + r"(?:s|ed|ing)?(?![A-Za-z])", re.IGNORECASE)
        target_locations[target] = [line.get("line_id", "") for line in lines if pattern.search(line["text"])]
    source_texts = [str(sentence.get("text", "")).casefold() for sentence in passage.get("sentences", [])]
    copied_line_ids = [line.get("line_id", "") for line in lines if line["text"].strip().casefold() in source_texts]
    long_lines = [line.get("line_id", "") for line in lines if len(_words(line["text"])) >= 25]
    repeated_speaker_pairs = []
    for previous, current in zip(lines, lines[1:]):
        if previous.get("speaker_id") == current.get("speaker_id"):
            repeated_speaker_pairs.append((previous.get("line_id", ""), current.get("line_id", "")))
    flags = []
    if not _has_any(full_text, GOAL_CUES):
        flags.append("No obvious goal cue detected by the heuristic.")
    if not _has_any(full_text, OBSTACLE_CUES):
        flags.append("No obvious obstacle or curiosity cue detected by the heuristic.")
    if not _has_any(full_text, TURN_CUES):
        flags.append("No obvious development or turning-point cue detected by the heuristic.")
    if not _has_any(full_text, RESOLUTION_CUES):
        flags.append("No obvious resolution cue detected by the heuristic.")
    if copied_line_ids:
        flags.append("Some dialogue lines exactly match a source sentence; review for rewritten-advertisement behavior.")
    if long_lines:
        flags.append("Some lines are long informational turns; review conversational balance.")
    if repeated_speaker_pairs:
        flags.append("At least one speaker has consecutive dialogue lines; review whether an interaction or response is missing.")
    return {
        "heuristic_only": True,
        "goal_cue_detected": _has_any(full_text, GOAL_CUES),
        "obstacle_or_curiosity_cue_detected": _has_any(full_text, OBSTACLE_CUES),
        "turning_point_cue_detected": _has_any(full_text, TURN_CUES),
        "resolution_cue_detected": _has_any(full_text, RESOLUTION_CUES),
        "target_word_line_locations": target_locations,
        "exact_source_copy_line_ids": copied_line_ids,
        "long_informational_line_ids": long_lines,
        "consecutive_same_speaker_pairs": repeated_speaker_pairs,
        "review_flags": flags,
    }
