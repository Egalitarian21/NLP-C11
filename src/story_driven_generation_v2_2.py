"""Version-isolated Story-driven v2.2 Direct and Planned generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from src.direct_generation import _redact, _usage_totals, _write_json
    from src.llm_vocabulary_evaluator import load_runtime_env, openai_compatible_transport, resolve_generation_parameters
    from src.prompt_audit import audit_request_inputs, scan_text_anomalies
    from src.script_validator import load_passage, validate_script
    from src.story_diagnostics import diagnose_story
    from src.story_evidence_review_v2_2 import review_source_evidence
    from src.source_grounded_story_planner_v2_2 import (
        DEFAULT_SCHEMA_PATH as DEFAULT_PLAN_SCHEMA_PATH,
        build_story_plan_prompt,
        validate_story_plan,
    )
except ModuleNotFoundError:  # pragma: no cover
    from direct_generation import _redact, _usage_totals, _write_json
    from llm_vocabulary_evaluator import load_runtime_env, openai_compatible_transport, resolve_generation_parameters
    from prompt_audit import audit_request_inputs, scan_text_anomalies
    from script_validator import load_passage, validate_script
    from story_diagnostics import diagnose_story
    from story_evidence_review_v2_2 import review_source_evidence
    from source_grounded_story_planner_v2_2 import DEFAULT_SCHEMA_PATH as DEFAULT_PLAN_SCHEMA_PATH, build_story_plan_prompt, validate_story_plan


ROOT = Path(__file__).resolve().parents[1]
DIRECT_PROMPT_VERSION = "direct-generation-v2.2"
PLANNING_PROMPT_VERSION = "source-grounded-story-planning-v2.2"
PLANNED_PROMPT_VERSION = "planned-script-generation-v2.2"
DEFAULT_DATASET_PATH = ROOT / "data" / "v0.3" / "dev.json"
DEFAULT_MAX_ATTEMPTS = 2
PROMPT_ROOT = ROOT / "prompts"
SCHEMA_ROOT = ROOT / "schemas"
CONFIG_PATH = ROOT / "configs" / "script_generation_v2_2.json"
SHARED_REQUIREMENTS_PATH = PROMPT_ROOT / "shared_story_requirements_v2_2.txt"
DIRECT_PROMPT_PATH = PROMPT_ROOT / "direct_generation_v2_2.txt"
PLANNING_PROMPT_PATH = PROMPT_ROOT / "source_grounded_story_planning_v2_2.txt"
PLANNED_PROMPT_PATH = PROMPT_ROOT / "planned_script_generation_v2_2.txt"
SCRIPT_SCHEMA_PATH = SCHEMA_ROOT / "script.schema.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_passage_from_dataset(dataset_path: Path | str, passage_id: str) -> dict[str, Any]:
    return load_passage(dataset_path, passage_id)


def assert_development_dataset(dataset_path: Path | str, project_root: Path | str = ROOT) -> None:
    expected = (Path(project_root) / "data" / "v0.3" / "dev.json").resolve()
    actual = Path(dataset_path).resolve()
    if actual != expected:
        raise ValueError(f"v2.2 real Pilot is restricted to the development dataset: {expected}")


def _common_input(passage: dict[str, Any]) -> str:
    sentences = "\n".join(f"[{item['id']}] {item['text']}" for item in passage["sentences"])
    return (
        "AUTHORITATIVE INPUT DATA\n"
        f"passage_id: {passage['passage_id']}\n"
        f"title: {passage['title']}\n"
        f"language: {passage.get('language', 'en')}\n"
        f"learner_level: {passage.get('learner_level', 'grade_7')}\n"
        f"target_words: {json.dumps(passage['target_words'], ensure_ascii=False)}\n"
        f"constraints: {json.dumps(passage.get('constraints', {}), ensure_ascii=False, sort_keys=True)}\n"
        "complete numbered textbook passage:\n"
        f"{sentences}\n"
    )


def _shared_block() -> str:
    return SHARED_REQUIREMENTS_PATH.read_text(encoding="utf-8").strip()


def _schema_block() -> str:
    return json.dumps(json.loads(SCRIPT_SCHEMA_PATH.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)


def _request_base(stage: str, prompt_version: str, method: str, passage: dict[str, Any], runtime_env: dict[str, str]) -> dict[str, Any]:
    return {
        "stage": stage,
        "prompt_version": prompt_version,
        "generation_method": method,
        "revision": "v2.2",
        "model": runtime_env.get("LLM_MODEL", ""),
        "temperature": 0,
        "seed": 42,
        "generation_parameters": resolve_generation_parameters(runtime_env.get("LLM_BASE_URL", ""), temperature=0, seed=42),
        "response_format": {"type": "json_object"},
        "passage_id": passage["passage_id"],
        "target_words": list(passage["target_words"]),
    }


def build_direct_v2_2_request(passage: dict[str, Any], env: dict[str, str] | None = None, project_root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    runtime_env = load_runtime_env(root / ".env", base_env=env or {})
    prompt = (
        DIRECT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        + "\n\n"
        + _shared_block()
        + "\n\n"
        + _common_input(passage)
        + "\nSCRIPT JSON SCHEMA\n"
        + _schema_block()
        + "\nReturn only the complete script JSON object."
    )
    request = _request_base("direct_generation", DIRECT_PROMPT_VERSION, "direct", passage, runtime_env)
    request["messages"] = [{"role": "system", "content": "Return only one complete script JSON object."}, {"role": "user", "content": prompt}]
    return request


def build_story_planning_v2_2_request(passage: dict[str, Any], env: dict[str, str] | None = None, project_root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    runtime_env = load_runtime_env(root / ".env", base_env=env or {})
    prompt = build_story_plan_prompt(passage, PLANNING_PROMPT_PATH, CONFIG_PATH, DEFAULT_PLAN_SCHEMA_PATH)
    request = _request_base("source_grounded_story_planning", PLANNING_PROMPT_VERSION, "plan_then_generate", passage, runtime_env)
    request["messages"] = [{"role": "system", "content": "Return only one v2.2 source-grounded story-plan JSON object."}, {"role": "user", "content": prompt}]
    return request


def build_planned_v2_2_request(passage: dict[str, Any], plan: dict[str, Any], env: dict[str, str] | None = None, project_root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    runtime_env = load_runtime_env(root / ".env", base_env=env or {})
    prompt = (
        PLANNED_PROMPT_PATH.read_text(encoding="utf-8").strip()
        + "\n\n"
        + _shared_block()
        + "\n\n"
        + _common_input(passage)
        + "\nVALIDATED SOURCE-GROUNDED STORY PLAN v2.2\n"
        + json.dumps(plan, ensure_ascii=False, indent=2)
        + "\n\nSCRIPT JSON SCHEMA\n"
        + _schema_block()
        + "\nReturn only the complete script JSON object."
    )
    request = _request_base("planned_script_generation", PLANNED_PROMPT_VERSION, "plan_then_generate", passage, runtime_env)
    request["messages"] = [{"role": "system", "content": "Return only one complete script JSON object."}, {"role": "user", "content": prompt}]
    return request


def _audit_v2_2_request(request: dict[str, Any], passage: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    audit = audit_request_inputs(request, passage)
    errors = list(audit["errors"])
    text = json.dumps(request.get("messages", []), ensure_ascii=False)
    if "v2.0" in text or "v1.1" in text or "v2.1" in text:
        errors.append("older prompt marker found in v2.2 request")
    if request.get("stage") == "direct_generation" and ("VALIDATED SOURCE-GROUNDED STORY PLAN" in text or "story_beats" in text):
        errors.append("Direct request contains a planning artifact")
    if plan is not None:
        for beat in plan.get("story_beats", []):
            if beat.get("beat_id") not in text:
                errors.append(f"planned request is missing beat {beat.get('beat_id')!r}")
    audit["errors"] = errors
    audit["valid"] = not errors and not audit.get("anomalies")
    return audit


def _parse_json_response(response: dict[str, Any]) -> Any:
    content = response.get("content")
    if not isinstance(content, str):
        raise ValueError("response content is not a string")
    return json.loads(content)


def _run_stage_v2_2(request: dict[str, Any], transport: Callable[[dict[str, Any]], dict[str, Any]], validator: Callable[[Any], dict[str, Any]], passage: dict[str, Any], api_key: str, max_attempts: int, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    request_audit = _audit_v2_2_request(request, passage, plan)
    if not request_audit["valid"]:
        raise ValueError("v2.2 request audit failed: " + "; ".join(request_audit["errors"]))
    attempts: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_ms": 0.0}
    value: Any = None
    final_validation: dict[str, Any] | None = None
    for number in range(1, max_attempts + 1):
        started = time.perf_counter()
        record: dict[str, Any] = {
            "attempt": number, "stage": request["stage"], "revision": "v2.2", "started_at": utc_now(),
            "request": _redact(request, api_key), "request_audit": request_audit, "status": "request_failed",
            "model": request.get("model"), "response_id": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "elapsed_ms": 0,
        }
        response: dict[str, Any] | None = None
        try:
            response = transport(request)
            record["response_id"] = response.get("id")
            record["model"] = response.get("model") or record["model"]
            record["usage"] = _usage_totals(response.get("usage"))
            record["actual_generation_parameters"] = response.get("actual_generation_parameters", request.get("generation_parameters"))
            record["raw_response"] = _redact(response, api_key)
            parsed = _parse_json_response(response)
            record["text_anomalies"] = scan_text_anomalies(parsed)
            validation = validator(parsed)
            record["validation"] = validation
            if record["text_anomalies"]:
                record["status"] = "anomaly_detected"
                record["error"] = "unexpected output character(s)"
            elif validation.get("valid"):
                record["status"] = "validated"
                value = parsed
                final_validation = validation
            else:
                record["status"] = "validation_failed"
                record["error"] = "deterministic validation failed"
        except json.JSONDecodeError as exc:
            record["status"] = "invalid_json"
            record["error"] = f"invalid JSON: {exc.msg}"
        except Exception as exc:
            record["status"] = "request_failed" if response is None else "processing_failed"
            record["error"] = _redact(str(exc), api_key)
            if response is not None:
                record["raw_response"] = _redact(response, api_key)
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        for key in totals:
            totals[key] += record["elapsed_ms"] if key == "elapsed_ms" else record["usage"].get(key, 0)
        attempts.append(record)
        if value is not None:
            break
    stats = {
        "api_call_count": len(attempts), "retry_count": max(0, len(attempts) - 1),
        "prompt_tokens": totals["prompt_tokens"], "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"], "elapsed_ms": round(totals["elapsed_ms"], 3),
        "model_names": sorted({str(item.get("model")) for item in attempts if item.get("model")}),
    }
    return {"valid": value is not None, "value": value, "validation": final_validation or (attempts[-1].get("validation") if attempts else None), "attempts": attempts, "stats": stats}


def _ensure_new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing v2.2 result directory: {output}")


def _script_validator(expected_method: str, passage: dict[str, Any]) -> Callable[[Any], dict[str, Any]]:
    def validate(value: Any) -> dict[str, Any]:
        result = validate_script(value, passage, schema_path=SCRIPT_SCHEMA_PATH)
        if isinstance(value, dict) and value.get("generation_method") != expected_method:
            result["valid"] = False
            result.setdefault("errors", []).append({"category": "rule", "path": "$.generation_method", "message": f"must equal {expected_method!r}"})
        result["business_rule_valid"] = bool(result.get("valid"))
        stats = result.get("stats", {})
        result["duration_compliant"] = not any(item.get("path") == "$.scenes" for item in result.get("warnings", []))
        result["human_review_status"] = "pending"
        return result
    return validate


def _write_report(result: dict[str, Any], output: Path, planned: bool) -> None:
    validation = result.get("validation") or result.get("generation", {}).get("validation") or {}
    lines = [
        f"# Story-driven v2.2 validation report: {result['passage_id']}", "",
        f"- Overall deterministic result: **{'PASS' if result['valid'] else 'FAIL'}**",
        f"- Revision: `v2.2`",
        f"- Prompt versions: `{result.get('prompt_version') or result.get('planning_prompt_version')}`, `{result.get('script_prompt_version', '')}`",
        f"- API calls: `{result['stats']['api_call_count']}`; retries: `{result['stats']['retry_count']}`",
        f"- Prompt tokens: `{result['stats']['prompt_tokens']}`; completion tokens: `{result['stats']['completion_tokens']}`; total tokens: `{result['stats']['total_tokens']}`",
        f"- Elapsed ms: `{result['stats']['elapsed_ms']}`",
        "- Semantic factual faithfulness, causal quality, naturalness, creativity, and pedagogical value: **pending independent human review**",
        "",
        "## Deterministic validation",
        "",
        f"- Schema validity: `{validation.get('schema_valid')}`",
        f"- Business-rule validity: `{validation.get('business_rule_valid', validation.get('valid'))}`",
        f"- Duration compliance (estimated at 130 WPM): `{validation.get('duration_compliant')}`",
        f"- Factual/narrative human review status: `{validation.get('human_review_status', 'pending')}`",
        f"- Errors: `{len(validation.get('errors', []))}`",
        f"- Stats: `{json.dumps(validation.get('stats', {}), ensure_ascii=False)}`",
        f"- Evidence review: `{json.dumps(result.get('evidence_review'), ensure_ascii=False)}`",
        f"- Heuristic story diagnostics: `{json.dumps(result.get('story_diagnostics'), ensure_ascii=False)}`",
        "",
        "Valid source IDs and lexical evidence hints provide traceability only; they do not prove semantic factual faithfulness.",
    ]
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_direct_v2_2(passage: dict[str, Any], project_root: Path | str = ROOT, output_dir: Path | str | None = None, transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None, env: dict[str, str] | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict[str, Any]:
    root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else default_v2_2_output_dir(root, passage["passage_id"], "direct")
    _ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime_env = load_runtime_env(root / ".env", base_env=env or {})
    request = build_direct_v2_2_request(passage, runtime_env, root)
    call = transport or (lambda req: openai_compatible_transport(req, env=runtime_env))
    stage = _run_stage_v2_2(request, call, _script_validator("direct", passage), passage, runtime_env.get("LLM_API_KEY", ""), max_attempts)
    evidence = review_source_evidence(stage["value"], passage) if stage["value"] is not None else None
    validation = stage["validation"] or {}
    status = "validated" if stage["valid"] and validation.get("duration_compliant", True) else ("duration_noncompliant" if stage["valid"] else "failed")
    result = {"status": status, "valid": stage["valid"], "passage_id": passage["passage_id"], "generation_method": "direct", "revision": "v2.2", "prompt_version": DIRECT_PROMPT_VERSION, "stats": stage["stats"], "attempts": stage["attempts"], "validation": validation, "evidence_review": evidence, "story_diagnostics": diagnose_story(stage["value"], passage) if stage["value"] is not None else None}
    _write_json(output / "raw_response.json", {"passage_id": passage["passage_id"], "attempts": stage["attempts"]})
    _write_json(output / "generation_log.json", _redact(result, runtime_env.get("LLM_API_KEY", "")))
    if stage["value"] is not None:
        _write_json(output / "script_direct.json", stage["value"])
    _write_report(result, output, planned=False)
    return result


def generate_plan_v2_2(passage: dict[str, Any], project_root: Path | str = ROOT, output_dir: Path | str | None = None, planning_transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None, generation_transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None, env: dict[str, str] | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict[str, Any]:
    root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else default_v2_2_output_dir(root, passage["passage_id"], "plan_then_generate")
    _ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime_env = load_runtime_env(root / ".env", base_env=env or {})
    planning_request = build_story_planning_v2_2_request(passage, runtime_env, root)
    planning_call = planning_transport or (lambda req: openai_compatible_transport(req, env=runtime_env))
    planning_stage = _run_stage_v2_2(planning_request, planning_call, lambda value: validate_story_plan(value, passage, DEFAULT_PLAN_SCHEMA_PATH), passage, runtime_env.get("LLM_API_KEY", ""), max_attempts)
    plan = planning_stage["value"]
    if plan is None:
        generation_stage = {"valid": False, "value": None, "validation": None, "attempts": [], "stats": {"api_call_count": 0, "retry_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_ms": 0.0, "model_names": []}}
    else:
        generation_request = build_planned_v2_2_request(passage, plan, runtime_env, root)
        generation_call = generation_transport or (lambda req: openai_compatible_transport(req, env=runtime_env))
        generation_stage = _run_stage_v2_2(generation_request, generation_call, _script_validator("plan_then_generate", passage), passage, runtime_env.get("LLM_API_KEY", ""), max_attempts, plan)
    stages = [planning_stage["stats"], generation_stage["stats"]]
    combined = {"api_call_count": sum(item["api_call_count"] for item in stages), "retry_count": sum(item["retry_count"] for item in stages), "prompt_tokens": sum(item["prompt_tokens"] for item in stages), "completion_tokens": sum(item["completion_tokens"] for item in stages), "total_tokens": sum(item["total_tokens"] for item in stages), "elapsed_ms": round(sum(item["elapsed_ms"] for item in stages), 3), "model_names": sorted(set(planning_stage["stats"]["model_names"] + generation_stage["stats"]["model_names"])), "planning": planning_stage["stats"], "generation": generation_stage["stats"]}
    evidence = review_source_evidence(generation_stage["value"], passage) if generation_stage["value"] is not None else None
    generation_validation = generation_stage["validation"] or {}
    status = "validated" if generation_stage["valid"] and generation_validation.get("duration_compliant", True) else ("duration_noncompliant" if generation_stage["valid"] else "failed")
    result = {"status": status, "valid": generation_stage["valid"], "passage_id": passage["passage_id"], "generation_method": "plan_then_generate", "revision": "v2.2", "planning_prompt_version": PLANNING_PROMPT_VERSION, "script_prompt_version": PLANNED_PROMPT_VERSION if plan is not None else None, "planning": planning_stage, "generation": generation_stage, "stats": combined, "evidence_review": evidence, "story_diagnostics": diagnose_story(generation_stage["value"], passage) if generation_stage["value"] is not None else None}
    _write_json(output / "raw_planning_response.json", {"passage_id": passage["passage_id"], "attempts": planning_stage["attempts"]})
    _write_json(output / "raw_generation_response.json", {"passage_id": passage["passage_id"], "attempts": generation_stage["attempts"]})
    if plan is not None:
        _write_json(output / "source_grounded_story_plan.json", plan)
    if generation_stage["value"] is not None:
        _write_json(output / "script_planned.json", generation_stage["value"])
    _write_json(output / "generation_log.json", _redact(result, runtime_env.get("LLM_API_KEY", "")))
    _write_report(result, output, planned=True)
    return result


def default_v2_2_output_dir(project_root: Path | str, passage_id: str, method: str) -> Path:
    if method not in {"direct", "plan_then_generate"}:
        raise ValueError(f"unsupported v2.2 method: {method}")
    return Path(project_root) / "outputs" / "generation_experiments" / "v2.2" / passage_id / method


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated v2.2 development story-driven Pilot.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--passage-id", default="p005")
    parser.add_argument("--method", choices=["direct", "plan_then_generate"], required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()
    if args.passage_id not in {"p003", "p005"}:
        raise SystemExit("v2.2 real Pilot is restricted to development passages P003 and P005")
    assert_development_dataset(args.dataset, args.project_root)
    passage = load_passage_from_dataset(args.dataset, args.passage_id)
    if args.method == "direct":
        result = generate_direct_v2_2(passage, args.project_root, args.output_dir, max_attempts=args.max_attempts)
    else:
        result = generate_plan_v2_2(passage, args.project_root, args.output_dir, max_attempts=args.max_attempts)
    print(json.dumps({"status": result["status"], "passage_id": result["passage_id"], "stats": result["stats"]}, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
