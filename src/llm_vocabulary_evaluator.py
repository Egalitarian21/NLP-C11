"""Provider-neutral LLM evaluator with strict, auditable JSON validation.

The module does not select Top 5 words and does not calculate frequency scores.  It only
requests and validates R/C/V semantic judgments.  A standard-library HTTP transport is
used so an OpenAI-compatible client can be configured without adding a dependency.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
import time


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "vocabulary_scoring.txt"
PROMPT_VERSION = "vocabulary-scoring-v1.0"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_ATTEMPTS = 2
SCORE_FIELDS = (
    "thematic_relevance",
    "contextual_learnability",
    "visual_representability",
)
REASON_FIELDS = tuple(f"{field}_reason" for field in SCORE_FIELDS)
REQUIRED_SCORE_FIELDS = (
    "word",
    *SCORE_FIELDS,
    *REASON_FIELDS,
    "contextual_meaning",
    "source_sentence_ids",
)


class ValidationError(ValueError):
    """Raised when a model response cannot be trusted as a scoring result."""


class ConfigurationError(RuntimeError):
    """Raised when the API environment is incomplete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fixed_collocations(word: str, contexts: list[dict[str, Any]]) -> list[str]:
    patterns = {
        "act": r"\bact\s+out\b",
        "fall": r"\bfall\s+in\s+love\s+with\b",
        "hang": r"\bhang\s+out\b",
        "such": r"\bsuch\s+as\b",
        "side": r"\bside\s+by\s+side\b",
    }
    pattern = patterns.get(word.lower())
    if not pattern:
        return []
    found: list[str] = []
    for context in contexts:
        for match in re.finditer(pattern, str(context.get("text", "")), re.IGNORECASE):
            phrase = match.group(0)
            if phrase.lower() not in {item.lower() for item in found}:
                found.append(phrase)
    return found


def _candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    contexts = [
        {
            "sentence_id": context.get("sentence_id"),
            "text": context.get("text", ""),
            "matched_forms": list(context.get("matched_forms", [])),
        }
        for context in candidate.get("matched_contexts", [])
    ]
    result = {
        "word": candidate["word"],
        "textbook_evidence": {
            "textbook_page": candidate.get("textbook_page"),
            "pdf_page": candidate.get("pdf_page"),
            "font_name": candidate.get("font_name"),
            "is_bold": candidate.get("is_bold"),
            "span_text": candidate.get("textbook_span_text"),
        },
        "matched_forms": list(candidate.get("matched_forms", [])),
        "source_sentence_ids": list(candidate.get("source_sentence_ids", [])),
        "source_sentences": contexts,
        "fixed_collocations": _fixed_collocations(candidate["word"], contexts),
    }
    if candidate.get("part_of_speech"):
        result["part_of_speech"] = candidate["part_of_speech"]
    if candidate.get("textbook_meaning"):
        result["textbook_meaning"] = candidate["textbook_meaning"]
    return result


def _passage_text(passage: dict[str, Any]) -> str:
    return "\n".join(
        f"[{sentence['id']}] {sentence['text']}" for sentence in passage.get("sentences", [])
    )


def _prompt_payload(
    passage: dict[str, Any], candidates: list[dict[str, Any]], order: list[str]
) -> dict[str, Any]:
    by_word = {candidate["word"]: candidate for candidate in candidates}
    ordered = [_candidate_for_prompt(by_word[word]) for word in order]
    return {
        "passage_id": passage.get("passage_id"),
        "passage_text": _passage_text(passage),
        "learner_profile": "Chinese Grade 7 English learners",
        "candidate_words": list(order),
        "candidates": ordered,
    }


def build_scoring_request(
    passage: dict[str, Any],
    candidates: list[dict[str, Any]],
    order: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    rubric_text: str | None = None,
) -> dict[str, Any]:
    """Build a request whose visible input cannot leak frequency or prior targets."""

    model_config = model_config or {}
    candidate_words = [candidate["word"] for candidate in candidates]
    order = list(order or sorted(candidate_words))
    if set(order) != set(candidate_words) or len(order) != len(candidate_words):
        raise ValueError("order must be a permutation of the candidate words")
    prompt_text = rubric_text or PROMPT_PATH.read_text(encoding="utf-8")
    payload = _prompt_payload(passage, candidates, order)
    user_content = (
        prompt_text
        + "\n\nINPUT DATA (use only this data):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    request = {
        "prompt_version": PROMPT_VERSION,
        "model": model_config.get("model") or os.getenv("LLM_MODEL", ""),
        "temperature": model_config.get("temperature", DEFAULT_TEMPERATURE),
        "seed": model_config.get("seed"),
        "response_format": {"type": "json_object"},
        "candidate_words": list(order),
        "candidates": payload["candidates"],
        "passage": {
            "passage_id": payload["passage_id"],
            "text": payload["passage_text"],
            "learner_profile": payload["learner_profile"],
        },
        "messages": [
            {
                "role": "system",
                "content": "Return only the required JSON object. Do not select or modify candidates.",
            },
            {"role": "user", "content": user_content},
        ],
    }
    if model_config.get("generation_parameters"):
        request["generation_parameters"] = model_config["generation_parameters"]
    return request


def _parse_json_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict) and "choices" in response:
        choices = response.get("choices") or []
        if not choices:
            raise ValidationError("API response has no choices")
        message = choices[0].get("message", {})
        response = message.get("content")
    elif isinstance(response, dict) and "content" in response:
        response = response["content"]
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(response, dict):
        raise ValidationError("response must be a JSON object")
    return response


def _transport_metadata(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    metadata = {}
    for key in ("model", "usage", "cost", "actual_generation_parameters"):
        if key in response:
            metadata[key] = response[key]
    if "id" in response:
        metadata["response_id"] = response["id"]
    if "elapsed_ms" in response:
        metadata["elapsed_ms"] = response["elapsed_ms"]
    return metadata


def validate_scoring_response(
    payload: Any,
    candidate_words: list[str],
    valid_sentence_ids: set[str],
    candidate_sentence_ids: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Validate a model response without repairing or silently filling it."""

    payload = _parse_json_response(payload)
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise ValidationError("top-level scores must be a list")
    expected = list(candidate_words)
    actual = [item.get("word") if isinstance(item, dict) else None for item in scores]
    if len(actual) != len(expected):
        raise ValidationError(f"candidate count mismatch: expected {len(expected)}, got {len(actual)}")
    if len(set(actual)) != len(actual):
        raise ValidationError("duplicate candidate word in response")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValidationError(f"candidate set mismatch; missing={missing}; extra={extra}")

    validated: list[dict[str, Any]] = []
    for item in scores:
        if not isinstance(item, dict):
            raise ValidationError("each score item must be an object")
        missing_fields = [field for field in REQUIRED_SCORE_FIELDS if field not in item]
        if missing_fields:
            raise ValidationError(f"{item.get('word')}: missing fields {missing_fields}")
        unexpected = sorted(set(item) - set(REQUIRED_SCORE_FIELDS))
        if unexpected:
            raise ValidationError(f"{item.get('word')}: unexpected fields {unexpected}")
        for field in SCORE_FIELDS:
            value = item[field]
            if type(value) is not int or not 0 <= value <= 5:
                raise ValidationError(f"{item['word']}: {field} must be an integer from 0 to 5")
        for field in REASON_FIELDS + ("contextual_meaning",):
            if not _nonempty(item[field]):
                raise ValidationError(f"{item['word']}: {field} must be non-empty")
        source_ids = item["source_sentence_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            raise ValidationError(f"{item['word']}: source_sentence_ids must be a non-empty list")
        if any(not isinstance(sentence_id, str) or sentence_id not in valid_sentence_ids for sentence_id in source_ids):
            raise ValidationError(f"{item['word']}: invalid source sentence ID")
        if len(source_ids) != len(set(source_ids)):
            raise ValidationError(f"{item['word']}: duplicate source sentence ID")
        if candidate_sentence_ids is not None:
            allowed_ids = candidate_sentence_ids.get(item["word"])
            if allowed_ids is None:
                raise ValidationError(f"{item['word']}: missing candidate sentence evidence mapping")
            if any(sentence_id not in allowed_ids for sentence_id in source_ids):
                raise ValidationError(
                    f"{item['word']}: source sentence ID is not supported by candidate evidence"
                )
        validated.append(item)
    return validated


def _safe_raw(response: Any) -> str:
    if isinstance(response, str):
        return response
    try:
        return json.dumps(response, ensure_ascii=False)
    except TypeError:
        return repr(response)


def score_with_retries(
    request: dict[str, Any],
    transport: Callable[[dict[str, Any]], Any],
    candidate_words: list[str],
    valid_sentence_ids: set[str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    candidate_sentence_ids: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Call independently up to a finite limit and preserve every failure trace."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    started_at = utc_now()
    raw_responses: list[str] = []
    transport_metadata: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            raw = transport(request)
            elapsed_ms = (time.perf_counter() - attempt_started) * 1000
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.setdefault("elapsed_ms", elapsed_ms)
            raw_responses.append(_safe_raw(raw))
            transport_metadata.append(_transport_metadata(raw))
            validated = validate_scoring_response(
                raw,
                candidate_words,
                valid_sentence_ids,
                candidate_sentence_ids=candidate_sentence_ids,
            )
            return {
                "status": "validated",
                "attempts": attempt,
                "started_at": started_at,
                "finished_at": utc_now(),
                "validated_scores": validated,
                "raw_responses": raw_responses,
                "transport_metadata": transport_metadata,
                "validation_errors": validation_errors,
            }
        except Exception as exc:  # preserve failure and continue only within the bound
            if len(raw_responses) < attempt:
                raw_responses.append("")
                transport_metadata.append(
                    {"elapsed_ms": (time.perf_counter() - attempt_started) * 1000}
                )
            validation_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    return {
        "status": "failed_validation",
        "attempts": max_attempts,
        "started_at": started_at,
        "finished_at": utc_now(),
        "raw_responses": raw_responses,
        "transport_metadata": transport_metadata,
        "validation_errors": validation_errors,
    }


def load_runtime_env(
    dotenv_path: Path | None = None, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    """Load process variables plus an optional local .env without printing secrets."""

    merged = dict(os.environ if base_env is None else base_env)
    dotenv_path = dotenv_path or ROOT / ".env"
    if not dotenv_path.exists():
        return merged
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        merged.setdefault(name, value)
    return merged


def load_api_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Return safe configuration metadata; never return secret values."""

    env = load_runtime_env() if env is None else env
    required = {"LLM_BASE_URL": env.get("LLM_BASE_URL", ""), "LLM_API_KEY": env.get("LLM_API_KEY", ""), "LLM_MODEL": env.get("LLM_MODEL", "")}
    missing = [name for name, value in required.items() if not str(value).strip()]
    return {
        "configured": not missing,
        "missing": missing,
        "base_url_configured": bool(str(required["LLM_BASE_URL"]).strip()),
        "api_key_configured": bool(str(required["LLM_API_KEY"]).strip()),
        "model_configured": bool(str(required["LLM_MODEL"]).strip()),
    }


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def resolve_generation_parameters(
    base_url: str,
    temperature: int | float = DEFAULT_TEMPERATURE,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Resolve requested versus applied parameters without claiming ignored options worked."""

    host = urlparse(base_url).hostname or ""
    is_deepseek_official = host.lower() in {"api.deepseek.com", "api.deepseek.com.cn"}
    requested = {
        "temperature": temperature,
        "seed": seed,
        "thinking_mode": "disabled",
    }
    if is_deepseek_official:
        return {
            "provider": "deepseek_official",
            "requested": requested,
            "applied": {
                "temperature": temperature,
                "thinking": {"type": "disabled"},
            },
            "support": {
                "json_output": "supported",
                "temperature": "supported_in_non_thinking_mode",
                "thinking_mode": "disabled_explicitly",
                "seed": "not_applied_deepseek_docs_say_ignored",
            },
        }
    return {
        "provider": "openai_compatible_unknown",
        "requested": requested,
        "applied": {"temperature": temperature, "seed": seed},
        "support": {
            "json_output": "requested_provider_specific",
            "temperature": "requested_provider_specific",
            "thinking_mode": "not_sent_to_unknown_provider",
            "seed": "sent_provider_specific_support_unverified",
        },
    }


def openai_compatible_transport(request: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Make one OpenAI-compatible request and preserve safe response metadata."""

    env = load_runtime_env() if env is None else env
    status = load_api_status(env)
    if not status["configured"]:
        raise ConfigurationError("missing API configuration: " + ", ".join(status["missing"]))
    generation = request.get("generation_parameters") or resolve_generation_parameters(
        env["LLM_BASE_URL"], request.get("temperature", DEFAULT_TEMPERATURE), request.get("seed")
    )
    body = {
        "model": request["model"],
        "messages": request["messages"],
        "response_format": request.get("response_format", {"type": "json_object"}),
    }
    body.update(generation.get("applied", {}))
    http_request = urllib.request.Request(
        _chat_completions_url(env["LLM_BASE_URL"]),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + env["LLM_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM connection error: {exc.reason}") from exc
    return {
        "content": data["choices"][0]["message"]["content"],
        "id": data.get("id"),
        "model": data.get("model"),
        "usage": data.get("usage"),
        "cost": data.get("cost"),
        "actual_generation_parameters": generation.get("applied", {}),
        "provider_response": data,
    }


def run_llm_request(
    request: dict[str, Any],
    transport: Callable[[dict[str, Any]], Any] | None = None,
    env: dict[str, str] | None = None,
    candidate_words: list[str] | None = None,
    valid_sentence_ids: set[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    candidate_sentence_ids: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    candidate_words = candidate_words or list(request["candidate_words"])
    valid_sentence_ids = valid_sentence_ids or set()
    if transport is None:
        transport = lambda req: openai_compatible_transport(req, env=env)
    return score_with_retries(
        request,
        transport,
        candidate_words=candidate_words,
        valid_sentence_ids=valid_sentence_ids,
        max_attempts=max_attempts,
        candidate_sentence_ids=candidate_sentence_ids,
    )


def run_connectivity_check(project_root: Path = ROOT, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one non-textbook smoke request and save only secret-free evidence."""

    runtime_env = load_runtime_env() if env is None else env
    status = load_api_status(runtime_env)
    output_path = project_root / "outputs" / "vocabulary_scoring" / "connectivity_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not status["configured"]:
        result = {"status": "blocked", "api_status": status}
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    passage = {
        "passage_id": "connectivity-smoke-test",
        "sentences": [{"id": "smoke-1", "text": "A student reads a book."}],
    }
    candidate = {
        "word": "book",
        "matched_forms": ["book"],
        "source_sentence_ids": ["smoke-1"],
        "matched_contexts": [{"sentence_id": "smoke-1", "text": "A student reads a book."}],
        "textbook_page": None,
        "pdf_page": None,
        "font_name": None,
        "is_bold": False,
    }
    generation_parameters = resolve_generation_parameters(runtime_env["LLM_BASE_URL"], 0, 42)
    request = build_scoring_request(
        passage,
        [candidate],
        ["book"],
        {
            "model": runtime_env["LLM_MODEL"],
            "temperature": 0,
            "seed": 42,
            "generation_parameters": generation_parameters,
        },
    )
    result = run_llm_request(
        request,
        env=runtime_env,
        candidate_words=["book"],
        valid_sentence_ids={"smoke-1"},
        max_attempts=1,
        candidate_sentence_ids={"book": {"smoke-1"}},
    )
    evidence = {
        "status": result["status"],
        "model_configured": runtime_env["LLM_MODEL"],
        "generation_parameters": generation_parameters,
        "request": request,
        "result": result,
        "note": "This request used a generic smoke-test sentence, not a textbook passage or P005 data.",
    }
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence
