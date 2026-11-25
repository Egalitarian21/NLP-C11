"""Direct Generation baseline for one V0.3 passage.

This module performs one independent OpenAI-compatible chat completion at a
time, validates the returned JSON with the existing deterministic validator,
and preserves every attempt without storing credentials.  It intentionally
does not read or send the manual mock script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from src.llm_vocabulary_evaluator import (
        load_runtime_env,
        openai_compatible_transport,
        resolve_generation_parameters,
    )
    from src.prompt_audit import audit_request_inputs, scan_text_anomalies
    from src.script_validator import load_passage, validate_script, write_markdown_report
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI invocation
    from llm_vocabulary_evaluator import (
        load_runtime_env,
        openai_compatible_transport,
        resolve_generation_parameters,
    )
    from prompt_audit import audit_request_inputs, scan_text_anomalies
    from script_validator import load_passage, validate_script, write_markdown_report


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "direct-generation-v1.0"
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "direct_generation.txt"
DEFAULT_SCHEMA_PATH = ROOT / "schemas" / "script.schema.json"
V11_PROMPT_VERSION = "direct-generation-v1.1"
V11_PROMPT_PATH = ROOT / "prompts" / "direct_generation_v1.1.txt"
V11_SHARED_REQUIREMENTS_PATH = ROOT / "prompts" / "shared_script_requirements_v1.1.txt"
V11_SHARED_CONFIG_PATH = ROOT / "configs" / "script_generation_v1.1.json"
DEFAULT_DATASET_PATH = ROOT / "data" / "v0.3" / "dev.json"
DEFAULT_MAX_ATTEMPTS = 2
SECRET_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any, api_key: str = "") -> Any:
    if isinstance(value, str):
        text = value
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
        return SECRET_PATTERN.sub(r"\1[REDACTED]", text)
    if isinstance(value, dict):
        return {key: _redact(item, api_key) for key, item in value.items() if key.lower() not in {"authorization", "api_key", "llm_api_key"}}
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    return value


def load_passage_from_dataset(dataset_path: Path | str, passage_id: str) -> dict[str, Any]:
    return load_passage(dataset_path, passage_id)


def build_direct_prompt(
    passage: dict[str, Any],
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    sentences = "\n".join(f"[{item['id']}] {item['text']}" for item in passage["sentences"])
    constraints = json.dumps(passage.get("constraints", {}), ensure_ascii=False, sort_keys=True)
    targets = json.dumps(passage["target_words"], ensure_ascii=False)
    input_block = (
        "\n\nAUTHORITATIVE INPUT DATA\n"
        f"passage_id: {passage['passage_id']}\n"
        f"title: {passage['title']}\n"
        f"language: {passage.get('language', 'en')}\n"
        f"learner_level: {passage.get('learner_level', 'grade_7')}\n"
        f"target_words: {targets}\n"
        f"constraints: {constraints}\n"
        "complete numbered textbook passage:\n"
        f"{sentences}\n\n"
        "SCHEMA CONTRACT (authoritative):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )
    return template.rstrip() + input_block


def build_direct_prompt_v11(passage: dict[str, Any], project_root: Path | str = ROOT) -> str:
    template = V11_PROMPT_PATH.read_text(encoding="utf-8")
    shared = V11_SHARED_REQUIREMENTS_PATH.read_text(encoding="utf-8").strip()
    schema = json.loads((Path(project_root) / "schemas" / "script.schema.json").read_text(encoding="utf-8"))
    sentences = "\n".join(f"[{item['id']}] {item['text']}" for item in passage["sentences"])
    return (
        template.rstrip()
        + "\n\n"
        + shared
        + "\n\nAUTHORITATIVE INPUT DATA\n"
        + f"passage_id: {passage['passage_id']}\n"
        + f"title: {passage['title']}\n"
        + f"language: {passage.get('language', 'en')}\n"
        + f"learner_level: {passage.get('learner_level', 'grade_7')}\n"
        + f"target_words: {json.dumps(passage['target_words'], ensure_ascii=False)}\n"
        + f"constraints: {json.dumps(passage.get('constraints', {}), ensure_ascii=False, sort_keys=True)}\n"
        + "complete numbered textbook passage:\n"
        + sentences
        + "\n\nSCRIPT JSON SCHEMA\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\nReturn only the complete script JSON object."
    )


def build_generation_request(
    passage: dict[str, Any],
    env: dict[str, str] | None = None,
    project_root: Path | str = ROOT,
    prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    revision: str = "v1.0",
) -> dict[str, Any]:
    runtime_env = load_runtime_env(Path(project_root) / ".env", base_env=env or {})
    base_url = runtime_env.get("LLM_BASE_URL", "")
    generation = resolve_generation_parameters(base_url, temperature=0, seed=42)
    if revision == "v1.1":
        prompt = build_direct_prompt_v11(passage, project_root)
        prompt_version = V11_PROMPT_VERSION
    elif revision == "v1.0":
        prompt = build_direct_prompt(passage, prompt_path=prompt_path, schema_path=Path(project_root) / "schemas" / "script.schema.json")
        prompt_version = PROMPT_VERSION
    else:
        raise ValueError(f"unsupported Direct Generation revision: {revision}")
    return {
        "stage": "direct_generation",
        "prompt_version": prompt_version,
        "generation_method": "direct",
        "model": runtime_env.get("LLM_MODEL", ""),
        "temperature": 0,
        "seed": 42,
        "generation_parameters": generation,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only one JSON object that follows the user-provided contract."},
            {"role": "user", "content": prompt},
        ],
        "passage_id": passage["passage_id"],
        "target_words": list(passage["target_words"]),
    }


def _usage_totals(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _parse_content(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if not isinstance(content, str):
        raise ValueError("response content is not a string")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("model JSON must be an object")
    return parsed


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report_with_generation_stats(result: dict[str, Any], output_path: Path, script_path: str) -> None:
    validation = result.get("validation") or {"valid": False, "errors": []}
    lines = [
        f"# Direct Generation validation report: {result['passage_id']}",
        "",
        f"- Prompt version: `{result['prompt_version']}`",
        f"- Generation method: `{result['generation_method']}`",
        f"- Script file: `{script_path}`",
        f"- Deterministic Schema/business validation: **{'PASS' if validation.get('valid') else 'FAIL'}**",
        "- Semantic faithfulness, naturalness, and teaching value: **pending human or independent-model review**",
        "",
        "## API and generation statistics",
        "",
    ]
    for key, value in result["stats"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Attempts", ""]
    for attempt in result["attempts"]:
        lines.append(
            f"- Attempt {attempt['attempt']}: `{attempt['status']}`, "
            f"model `{attempt.get('model')}`, elapsed `{attempt.get('elapsed_ms')} ms`, "
            f"tokens `{attempt['usage'].get('total_tokens', 0)}`"
        )
        if attempt.get("error"):
            lines.append(f"  - Error: {attempt['error']}")
    lines += ["", "## Validator errors", ""]
    errors = validation.get("errors", [])
    if errors:
        lines.extend(f"- [{item.get('category')}] `{item.get('path')}`: {item.get('message')}" for item in errors)
    else:
        lines.append("- None")
    lines += ["", "## Review boundary", "", "The validator confirms structural and deterministic input-linked rules only. Source ID existence does not prove factual faithfulness; the generated dialogue must be reviewed for accurate paraphrase, natural target-word use, and age-appropriate teaching value.", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_direct(
    passage: dict[str, Any],
    project_root: Path | str = ROOT,
    output_dir: Path | str | None = None,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    env: dict[str, str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    revision: str = "v1.0",
) -> dict[str, Any]:
    project_root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else default_output_dir(project_root, passage["passage_id"], revision)
    output.mkdir(parents=True, exist_ok=True)
    runtime_env = load_runtime_env(project_root / ".env", base_env=env or {})
    request = build_generation_request(passage, runtime_env, project_root, revision=revision)
    api_key = runtime_env.get("LLM_API_KEY", "")
    request_audit = audit_request_inputs(request, passage)
    if not request_audit["valid"]:
        raise ValueError("Direct Generation request audit failed: " + "; ".join(request_audit["errors"]))
    transport_fn = transport or (lambda req: openai_compatible_transport(req, env=runtime_env))
    attempts: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_ms": 0}
    final_script: dict[str, Any] | None = None
    final_validation: dict[str, Any] | None = None

    for attempt_number in range(1, max_attempts + 1):
        started = time.perf_counter()
        record: dict[str, Any] = {
            "attempt": attempt_number,
            "started_at": utc_now(),
            "request": _redact(request, api_key),
            "request_audit": request_audit,
            "status": "request_failed",
            "model": request.get("model"),
            "response_id": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed_ms": 0,
        }
        response: dict[str, Any] | None = None
        try:
            response = transport_fn(request)
            record["response_id"] = response.get("id")
            record["model"] = response.get("model") or record["model"]
            record["usage"] = _usage_totals(response.get("usage"))
            record["actual_generation_parameters"] = response.get("actual_generation_parameters", request.get("generation_parameters"))
            record["raw_response"] = _redact(response, api_key)
            parsed = _parse_content(response)
            record["text_anomalies"] = scan_text_anomalies(parsed)
            validation = validate_script(parsed, passage, schema_path=project_root / "schemas" / "script.schema.json")
            record["validation"] = validation
            if not validation["valid"]:
                record["status"] = "validation_failed"
                record["error"] = "deterministic script validation failed"
            else:
                record["status"] = "validated"
                final_script = parsed
                final_validation = validation
        except json.JSONDecodeError as exc:
            record["status"] = "invalid_json"
            record["error"] = f"invalid JSON: {exc.msg}"
        except Exception as exc:  # request and validation failures are retained, never silently replaced
            record["status"] = "request_failed" if response is None else "processing_failed"
            record["error"] = _redact(str(exc), api_key)
            if response is not None:
                record["raw_response"] = _redact(response, api_key)
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        for key in totals:
            if key == "elapsed_ms":
                totals[key] += record["elapsed_ms"]
            else:
                totals[key] += record["usage"].get(key, 0)
        attempts.append(record)
        if final_script is not None:
            break

    stats = {
        "api_call_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
        "elapsed_ms": round(totals["elapsed_ms"], 3),
        "model_names": sorted({str(item.get("model")) for item in attempts if item.get("model")}),
    }
    result = {
        "status": "validated" if final_script is not None else "failed",
        "valid": final_script is not None,
        "passage_id": passage["passage_id"],
        "generation_method": "direct",
        "prompt_version": request["prompt_version"],
        "stats": stats,
        "attempts": attempts,
        "validation": final_validation or (attempts[-1].get("validation") if attempts else None),
    }
    _write_json(output / "raw_response.json", {"passage_id": passage["passage_id"], "attempts": attempts})
    _write_json(output / "generation_log.json", _redact(result, api_key))
    if final_script is not None:
        _write_json(output / "script_direct.json", final_script)
    _report_with_generation_stats(result, output / "validation_report.md", "script_direct.json")
    return result


def default_output_dir(project_root: Path | str, passage_id: str, revision: str) -> Path:
    """Return a revision-specific default directory without overwriting prior runs."""
    root = Path(project_root)
    if revision == "v1.1":
        return root / "outputs" / "generation_experiments" / "v1.1" / passage_id / "direct"
    if revision == "v1.0":
        return root / "outputs" / "direct_generation" / passage_id
    raise ValueError(f"unsupported Direct Generation revision: {revision}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Direct Generation baseline request.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--passage-id", default="p005")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--revision", choices=["v1.0", "v1.1"], default="v1.1")
    args = parser.parse_args()
    passage = load_passage_from_dataset(args.dataset, args.passage_id)
    result = generate_direct(passage, args.project_root, output_dir=args.output_dir, max_attempts=args.max_attempts, revision=args.revision)
    print(json.dumps({"status": result["status"], "passage_id": result["passage_id"], "stats": result["stats"]}, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
