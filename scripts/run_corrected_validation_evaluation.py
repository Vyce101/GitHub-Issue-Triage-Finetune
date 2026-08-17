"""Regenerate only historically truncated validation rows with safe prompt truncation."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from validation_evaluation.config import (
    ADAPTER_PATH,
    CONDITION_BASE_FEW_SHOT,
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    CONDITIONS,
    CORRECTED_RAW_ARTIFACT_DIRECTORY,
    CORRECTED_REPORT_PATH,
    MAX_INPUT_TOKENS,
    RAW_ARTIFACT_DIRECTORY,
    REPORT_PATH,
    TOTAL_CONTEXT_TOKENS,
)
from validation_evaluation.data import load_frozen_validation_rows, load_recorded_few_shot_rows
from validation_evaluation.generation import evaluate_selected_rows
from validation_evaluation.models import load_adapter_model, load_base_model, verify_adapter_artifact
from validation_evaluation.prompt_checks import run_prompt_regression_checks
from validation_evaluation.runner import (
    _parser_smoke_test,
    _progress,
    _release_cuda_memory,
    _resource_preflight,
)


def _utc_now() -> str:
    """Return a UTC timestamp for the corrected report."""
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    """Return a stable repository-relative path."""
    return str(path.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one complete historical prediction artifact."""
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _validate_historical_records(
    condition: str,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Verify that a historical artifact is complete and aligned before selective regeneration."""
    if len(records) != len(rows):
        raise RuntimeError(f"Historical {condition} artifact has {len(records)} rows, expected {len(rows)}")
    for evaluation_index, (row, record) in enumerate(zip(rows, records)):
        expected_identity = {
            "evaluation_index": evaluation_index,
            "issue_id": row["issue_id"],
            "repository": row["repository"],
            "expected_category": row["target_category"],
        }
        for field, expected_value in expected_identity.items():
            if record.get(field) != expected_value:
                raise RuntimeError(f"Historical {condition} identity mismatch at {evaluation_index}: {field}")


def _load_historical_records(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Load all three original artifacts without modifying them."""
    records = {}
    for condition in CONDITIONS:
        artifact = RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl"
        records[condition] = _load_jsonl(artifact)
        _validate_historical_records(condition, rows, records[condition])
    return records


def _git_snapshot() -> dict[str, Any]:
    """Capture the source revision used for corrected evaluation."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit_sha": commit, "working_tree_dirty": bool(status), "status_lines": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"capture_error": f"{type(error).__name__}: {error}"}


def _pairwise_deltas(condition_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Calculate the requested corrected-condition metric deltas."""
    fine = condition_results[CONDITION_FINE_TUNED_ZERO_SHOT]["metrics"]
    base_zero = condition_results[CONDITION_BASE_ZERO_SHOT]["metrics"]
    base_few = condition_results[CONDITION_BASE_FEW_SHOT]["metrics"]
    return {
        "fine_tuned_zero_shot_minus_base_zero_shot_accuracy_delta": round(fine["accuracy"] - base_zero["accuracy"], 6),
        "fine_tuned_zero_shot_minus_base_zero_shot_macro_f1_delta": round(fine["macro_f1"] - base_zero["macro_f1"], 6),
        "fine_tuned_zero_shot_minus_base_few_shot_accuracy_delta": round(fine["accuracy"] - base_few["accuracy"], 6),
        "fine_tuned_zero_shot_minus_base_few_shot_macro_f1_delta": round(fine["macro_f1"] - base_few["macro_f1"], 6),
        "base_few_shot_minus_base_zero_shot_accuracy_delta": round(base_few["accuracy"] - base_zero["accuracy"], 6),
        "base_few_shot_minus_base_zero_shot_macro_f1_delta": round(base_few["macro_f1"] - base_zero["macro_f1"], 6),
    }


def run_corrected_evaluation() -> dict[str, Any]:
    """Run prompt checks, then regenerate only historically truncated rows."""
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": _utc_now(),
        "evaluation_type": "generation_based_validation_evaluation_corrected_truncation",
        "full_validation_inference_rerun": False,
        "test_accessed_by_evaluator": False,
        "historical_evaluation_report": _relative(REPORT_PATH),
        "historical_raw_artifact_directory": _relative(RAW_ARTIFACT_DIRECTORY),
        "corrected_raw_artifact_directory": _relative(CORRECTED_RAW_ARTIFACT_DIRECTORY),
        "git": _git_snapshot(),
        "errors": [],
    }
    model = None
    tokenizer = None
    try:
        report["resource_preflight"] = _resource_preflight()
        validation_rows, validation_boundary = load_frozen_validation_rows()
        few_shot_rows, few_shot_definition = load_recorded_few_shot_rows()
        historical_report = _load_json(REPORT_PATH)
        historical_records = _load_historical_records(validation_rows)
        report["validation_boundary"] = validation_boundary
        report["few_shot_definition"] = few_shot_definition
        report["adapter_verification"] = verify_adapter_artifact()
        report["historical_metrics"] = {
            condition: historical_report["conditions"][condition]["metrics"]
            for condition in CONDITIONS
        }

        selected_indices = {
            condition: {
                record["evaluation_index"]
                for record in historical_records[condition]
                if bool(record.get("input_truncated"))
            }
            for condition in CONDITIONS
        }
        report["regeneration_plan"] = {
            condition: {
                "row_count": len(indices),
                "evaluation_indices_are_historical_truncations": True,
                "unaffected_rows_reused": len(validation_rows) - len(indices),
            }
            for condition, indices in selected_indices.items()
        }

        tokenizer_only = AutoTokenizer.from_pretrained(
            str(ADAPTER_PATH),
            local_files_only=True,
            use_fast=True,
        )
        report["tokenizer_only_prompt_regression_checks"] = run_prompt_regression_checks(
            tokenizer_only,
            validation_rows,
            few_shot_rows,
            historical_records,
        )
        report["parser_contract_check_before_inference"] = _parser_smoke_test()
        report["corrected_context_policy"] = {
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
            "generation_output_reserve_tokens": 16,
            "maximum_prompt_input_tokens": MAX_INPUT_TOKENS,
            "truncation_implementation": "prompt_preserving_v2",
            "truncation_order": "current_issue_body_right_then_current_issue_title_right",
            "structural_chat_content_preserved": True,
            "frozen_few_shot_demonstrations_unchanged": True,
            "generation_prompt_boundary_preserved": True,
        }

        condition_results: dict[str, dict[str, Any]] = {}
        device = torch.device("cuda:0")

        print("Loading locked base model after prompt checks...", flush=True)
        model, tokenizer, base_load = load_base_model()
        report["base_model_load"] = base_load
        try:
            base_tokenizer_checks = run_prompt_regression_checks(
                tokenizer,
                validation_rows,
                few_shot_rows,
                historical_records,
            )
            report["base_model_tokenizer_prompt_regression_checks"] = base_tokenizer_checks
            for condition in (CONDITION_BASE_ZERO_SHOT, CONDITION_BASE_FEW_SHOT):
                output_path = CORRECTED_RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl"
                condition_results[condition] = evaluate_selected_rows(
                    model,
                    tokenizer,
                    validation_rows,
                    historical_records[condition],
                    selected_indices[condition],
                    condition,
                    few_shot_rows,
                    device=device,
                    output_path=output_path,
                    progress_callback=_progress,
                )
                condition_results[condition]["raw_predictions_path"] = _relative(output_path)
        finally:
            model = None
            tokenizer = None
            _release_cuda_memory()

        print("Loading completed LoRA adapter after prompt checks...", flush=True)
        model, tokenizer, adapter_load = load_adapter_model()
        report["adapter_model_load"] = adapter_load
        try:
            adapter_tokenizer_checks = run_prompt_regression_checks(
                tokenizer,
                validation_rows,
                few_shot_rows,
                historical_records,
            )
            report["adapter_tokenizer_prompt_regression_checks"] = adapter_tokenizer_checks
            output_path = CORRECTED_RAW_ARTIFACT_DIRECTORY / f"{CONDITION_FINE_TUNED_ZERO_SHOT}.jsonl"
            condition_results[CONDITION_FINE_TUNED_ZERO_SHOT] = evaluate_selected_rows(
                model,
                tokenizer,
                validation_rows,
                historical_records[CONDITION_FINE_TUNED_ZERO_SHOT],
                selected_indices[CONDITION_FINE_TUNED_ZERO_SHOT],
                CONDITION_FINE_TUNED_ZERO_SHOT,
                few_shot_rows,
                device=device,
                output_path=output_path,
                progress_callback=_progress,
            )
            condition_results[CONDITION_FINE_TUNED_ZERO_SHOT]["raw_predictions_path"] = _relative(output_path)
        finally:
            model = None
            tokenizer = None
            _release_cuda_memory()

        report["conditions"] = condition_results
        report["pairwise_deltas"] = _pairwise_deltas(condition_results)
        report["timing_policy"] = {
            "unaffected_rows": "historical inference_time_seconds preserved exactly in the corrected artifact",
            "regenerated_rows": "new inference_time_seconds measured during corrected generation",
            "total_inference_runtime": "sum of mixed historical and regenerated row timings; regenerated runtime is separately recorded per condition",
        }
        report["raw_artifacts"] = {
            "format": "complete sequential JSONL; unaffected historical records copied and affected records marked regenerated",
            "conditions": {
                condition: _relative(CORRECTED_RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl")
                for condition in CONDITIONS
            },
        }
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - persist diagnostics before surfacing failure.
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if model is not None and tokenizer is not None:
            model = None
            tokenizer = None
            _release_cuda_memory()
        report["completed_at_utc"] = _utc_now()
        CORRECTED_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the corrected selective validation command."""
    report = run_corrected_evaluation()
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": _relative(CORRECTED_REPORT_PATH),
                "full_validation_inference_rerun": report["full_validation_inference_rerun"],
                "regeneration_plan": report.get("regeneration_plan"),
                "pairwise_deltas": report.get("pairwise_deltas"),
                "errors": report.get("errors", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
