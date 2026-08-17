"""Run the smoke-gated, three-condition frozen validation evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from classification.parser import parse_classification_output

from .config import (
    ADAPTER_PATH,
    CONDITION_BASE_FEW_SHOT,
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    CONDITIONS,
    CONFIG_PATH,
    GENERATION_MAX_NEW_TOKENS,
    MAX_INPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    RAW_ARTIFACT_DIRECTORY,
    REPORT_PATH,
    TOTAL_CONTEXT_TOKENS,
)
from .data import load_frozen_validation_rows, load_recorded_few_shot_rows
from .generation import evaluate_condition
from .models import load_adapter_model, load_base_model, verify_adapter_artifact


def _relative(path: Path) -> str:
    """Return a repository-relative path with stable separators."""
    return str(path.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/")


def _utc_now() -> str:
    """Return an explicit UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _git_snapshot() -> dict[str, Any]:
    """Capture the source revision used to launch the evaluation."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CONFIG_PATH.parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=CONFIG_PATH.parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit_sha": commit, "working_tree_dirty": bool(status), "status_lines": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"capture_error": f"{type(error).__name__}: {error}"}


def _resource_preflight() -> dict[str, Any]:
    """Verify the isolated interpreter, CUDA, and absence of a competing GPU workload."""
    interpreter = Path(sys.executable).resolve()
    venv_active = sys.prefix != sys.base_prefix and ".venv" in interpreter.parts
    if not venv_active:
        raise RuntimeError(f"Evaluation must run inside the project .venv: {interpreter}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Evaluation requires Python 3.11, got {sys.version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpu_name = torch.cuda.get_device_name(0)
    try:
        process_output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Could not inspect GPU processes: {error}") from error
    processes = []
    for line in process_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            processes.append({"pid": fields[0], "name": fields[1], "used_memory_mib": fields[2]})
    substantial_processes = []
    for process in processes:
        try:
            used_memory_mib = int(process["used_memory_mib"])
        except ValueError:
            continue
        if used_memory_mib >= 512:
            substantial_processes.append(process)
    if substantial_processes:
        raise RuntimeError(f"A substantial compute workload is already using the GPU: {substantial_processes}")

    total_gpu_memory_mib = 0
    for process in processes:
        try:
            total_gpu_memory_mib += int(process["used_memory_mib"])
        except ValueError:
            continue

    return {
        "python_executable": _relative(interpreter),
        "python_version": sys.version,
        "venv_active": venv_active,
        "torch_version": torch.__version__,
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_process_count_before_evaluation": len(processes),
        "gpu_memory_mib_before_evaluation": total_gpu_memory_mib,
        "substantial_gpu_process_count_before_evaluation": len(substantial_processes),
        "unrelated_substantial_gpu_workload": False,
        "environment_variable_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _parser_smoke_test() -> dict[str, Any]:
    """Exercise the canonical parser's required acceptance and rejection boundaries."""
    cases = {
        "plain_valid": ("{\"type\":\"bug\"}", True),
        "empty_think_wrapper_valid": ("<think></think>{\"type\":\"bug\"}", True),
        "nonempty_thinking_rejected": ("<think>reason</think>{\"type\":\"bug\"}", False),
        "extra_prose_rejected": ("Answer: {\"type\":\"bug\"}", False),
        "extra_key_rejected": ("{\"type\":\"bug\",\"extra\":1}", False),
        "extra_object_rejected": ("{\"type\":\"bug\"}{\"type\":\"feature\"}", False),
        "unapproved_category_rejected": ("{\"type\":\"other\"}", False),
    }
    outcomes = {}
    for name, (raw_output, expected_valid) in cases.items():
        result = parse_classification_output(raw_output)
        if result.schema_valid != expected_valid:
            raise RuntimeError(f"Parser smoke case failed: {name}: {result}")
        outcomes[name] = {
            "schema_valid": result.schema_valid,
            "parse_error": result.parse_error,
            "normalized_output": result.normalized_output,
        }
    return {"passed": True, "cases": outcomes}


def _release_cuda_memory() -> None:
    """Collect released model objects and return their GPU memory to the allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _progress(condition: str, completed: int, total: int, elapsed: float) -> None:
    """Print bounded progress so a long evaluation remains observable."""
    if completed == total or completed % 500 == 0:
        print(f"[{condition}] {completed}/{total} examples; elapsed={elapsed:.1f}s", flush=True)


def _run_smoke_test(
    validation_rows: list[dict[str, Any]],
    few_shot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove all three model conditions reload and call the canonical parser."""
    smoke_rows = validation_rows[:1]
    device = torch.device("cuda:0")
    condition_results = {}

    print("Loading locked base model for smoke test...", flush=True)
    model, tokenizer, base_load = load_base_model()
    try:
        for condition in (CONDITION_BASE_ZERO_SHOT, CONDITION_BASE_FEW_SHOT):
            condition_results[condition] = evaluate_condition(
                model,
                tokenizer,
                smoke_rows,
                condition,
                few_shot_rows,
                device=device,
                output_path=None,
            )
    finally:
        model = None
        tokenizer = None
        _release_cuda_memory()

    print("Loading completed LoRA adapter for smoke test...", flush=True)
    adapter_model, adapter_tokenizer, adapter_load = load_adapter_model()
    try:
        condition_results[CONDITION_FINE_TUNED_ZERO_SHOT] = evaluate_condition(
            adapter_model,
            adapter_tokenizer,
            smoke_rows,
            CONDITION_FINE_TUNED_ZERO_SHOT,
            few_shot_rows,
            device=device,
            output_path=None,
        )
    finally:
        adapter_model = None
        adapter_tokenizer = None
        _release_cuda_memory()

    for condition, result in condition_results.items():
        if result["metrics"]["example_count"] != 1:
            raise RuntimeError(f"Smoke test did not produce one result for {condition}")

    return {
        "passed": True,
        "validation_row_used": {
            "issue_id": smoke_rows[0]["issue_id"],
            "repository": smoke_rows[0]["repository"],
            "source_split": smoke_rows[0]["source_split"],
        },
        "parser_contract": _parser_smoke_test(),
        "base_model_load": base_load,
        "adapter_model_load": adapter_load,
        "conditions": {
            condition: {
                "schema_valid": result["metrics"]["valid_output_count"] == 1,
                "predicted_category": result["metrics"]["predicted_class_distribution"],
                "raw_predictions_path": result["raw_predictions_path"],
            }
            for condition, result in condition_results.items()
        },
    }


def _pairwise_deltas(condition_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return the requested accuracy and macro-F1 comparisons."""
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


def run_validation_evaluation(*, smoke_only: bool = False) -> dict[str, Any]:
    """Run the required smoke test and, unless requested otherwise, full validation."""
    started_at = _utc_now()
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": started_at,
        "evaluation_type": "generation_based_validation_evaluation",
        "full_evaluation_started": False,
        "test_accessed_by_evaluator": False,
        "config_path": _relative(CONFIG_PATH),
        "git": _git_snapshot(),
        "errors": [],
    }
    model = None
    tokenizer = None
    try:
        report["resource_preflight"] = _resource_preflight()
        validation_rows, validation_boundary = load_frozen_validation_rows()
        few_shot_rows, few_shot_definition = load_recorded_few_shot_rows()
        report["validation_boundary"] = validation_boundary
        report["few_shot_definition"] = few_shot_definition
        report["adapter_verification"] = verify_adapter_artifact()
        report["context_policy"] = {
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
            "generation_output_reserve_tokens": GENERATION_MAX_NEW_TOKENS,
            "maximum_prompt_input_tokens": MAX_INPUT_TOKENS,
            "truncation_side": "right",
            "truncation_scope": "prompt_input_only",
            "expected_output_region_truncated": False,
            "policy_applied_to_all_conditions": True,
        }
        report["generation_contract"] = {
            "do_sample": False,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
            "chain_of_thought_requested": False,
            "parser": "scripts/classification/parser.py:parse_classification_output",
        }
        report["smoke_test"] = _run_smoke_test(validation_rows, few_shot_rows)
        if smoke_only:
            report["status"] = "smoke_passed"
            return report

        report["full_evaluation_started"] = True
        condition_results: dict[str, dict[str, Any]] = {}
        device = torch.device("cuda:0")

        print("Loading locked base model for full validation...", flush=True)
        model, tokenizer, base_load = load_base_model()
        report["base_model_load"] = base_load
        try:
            base_zero_path = RAW_ARTIFACT_DIRECTORY / "base_zero_shot.jsonl"
            condition_results[CONDITION_BASE_ZERO_SHOT] = evaluate_condition(
                model,
                tokenizer,
                validation_rows,
                CONDITION_BASE_ZERO_SHOT,
                few_shot_rows,
                device=device,
                output_path=base_zero_path,
                progress_callback=_progress,
            )
            condition_results[CONDITION_BASE_ZERO_SHOT]["raw_predictions_path"] = _relative(base_zero_path)
            base_few_path = RAW_ARTIFACT_DIRECTORY / "base_few_shot.jsonl"
            condition_results[CONDITION_BASE_FEW_SHOT] = evaluate_condition(
                model,
                tokenizer,
                validation_rows,
                CONDITION_BASE_FEW_SHOT,
                few_shot_rows,
                device=device,
                output_path=base_few_path,
                progress_callback=_progress,
            )
            condition_results[CONDITION_BASE_FEW_SHOT]["raw_predictions_path"] = _relative(base_few_path)
        finally:
            model = None
            tokenizer = None
            _release_cuda_memory()

        print("Loading completed LoRA adapter for full validation...", flush=True)
        model, tokenizer, adapter_load = load_adapter_model()
        report["adapter_model_load"] = adapter_load
        try:
            fine_tuned_path = RAW_ARTIFACT_DIRECTORY / "fine_tuned_zero_shot.jsonl"
            condition_results[CONDITION_FINE_TUNED_ZERO_SHOT] = evaluate_condition(
                model,
                tokenizer,
                validation_rows,
                CONDITION_FINE_TUNED_ZERO_SHOT,
                few_shot_rows,
                device=device,
                output_path=fine_tuned_path,
                progress_callback=_progress,
            )
            condition_results[CONDITION_FINE_TUNED_ZERO_SHOT]["raw_predictions_path"] = _relative(fine_tuned_path)
        finally:
            model = None
            tokenizer = None
            _release_cuda_memory()

        report["conditions"] = condition_results
        report["pairwise_deltas"] = _pairwise_deltas(condition_results)
        report["historical_prompt_development_evidence"] = {
            "source": "results/baselines/metrics.json",
            "evaluation_split": "train",
            "example_count_per_condition": 400,
            "not_used_as_primary_validation_comparison": True,
        }
        report["raw_artifacts"] = {
            "directory": _relative(RAW_ARTIFACT_DIRECTORY),
            "format": "compact JSONL; one complete result for every validation row per condition",
            "conditions": {
                condition: _relative(RAW_ARTIFACT_DIRECTORY / filename)
                for condition, filename in {
                    CONDITION_BASE_ZERO_SHOT: "base_zero_shot.jsonl",
                    CONDITION_BASE_FEW_SHOT: "base_few_shot.jsonl",
                    CONDITION_FINE_TUNED_ZERO_SHOT: "fine_tuned_zero_shot.jsonl",
                }.items()
            },
        }
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - write a diagnostic report before surfacing failure.
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        if model is not None and tokenizer is not None:
            model = None
            tokenizer = None
            _release_cuda_memory()
        report["completed_at_utc"] = _utc_now()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the smoke-gated evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run resource, adapter, parser, and three-condition smoke checks without full validation.",
    )
    args = parser.parse_args()
    report = run_validation_evaluation(smoke_only=args.smoke_only)
    summary = {
        "status": report["status"],
        "report_path": _relative(REPORT_PATH),
        "full_evaluation_started": report["full_evaluation_started"],
        "errors": report.get("errors", []),
    }
    if report["status"] == "passed":
        summary["pairwise_deltas"] = report.get("pairwise_deltas")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
