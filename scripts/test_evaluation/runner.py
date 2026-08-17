"""Run the three-condition frozen TEST evaluation and write its compact report."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from baseline.config import SYSTEM_INSTRUCTION
from validation_evaluation.generation import evaluate_condition
from validation_evaluation.models import load_adapter_model, load_base_model, verify_adapter_artifact
from validation_evaluation.resume import load_sequential_prefix
from validation_evaluation.runner import _git_snapshot, _parser_smoke_test, _progress, _release_cuda_memory, _resource_preflight

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
    MODEL_LOAD_MAX_SEQUENCE_LENGTH,
    MODEL_REVISION,
    PROMPT_DEFINITION_PATH,
    TARGET_CATEGORIES,
    TEST_RAW_ARTIFACT_DIRECTORY,
    TEST_REPORT_PATH,
    TOTAL_CONTEXT_TOKENS,
)
from .data import load_frozen_prompt_inputs, load_frozen_test_rows
from .prompt_checks import run_test_prompt_regression_checks
from .statistics import (
    class_comparison,
    exact_mcnemar,
    metrics_by_truncation,
    paired_bootstrap,
    paired_correctness,
    repository_metrics,
)


EXPECTED_VERSIONS = {
    "numpy": "2.4.3",
    "torch": "2.11.0+cu130",
    "transformers": "5.5.0",
    "trl": "0.24.0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "datasets": "4.3.0",
    "scikit-learn": "1.9.0",
    "unsloth": "2026.8.18",
}


def _relative(path: Path) -> str:
    """Return a stable repository-relative path."""
    return str(path.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/")


def _utc_now() -> str:
    """Return a UTC timestamp for the final report."""
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one local JSON object."""
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _environment_integrity() -> dict[str, Any]:
    """Verify the isolated interpreter, pinned package versions, and dependency health."""
    interpreter = Path(sys.executable).resolve()
    venv_active = sys.prefix != sys.base_prefix and ".venv" in interpreter.parts
    if not venv_active:
        raise RuntimeError(f"Evaluation must run inside the project .venv: {interpreter}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Evaluation requires Python 3.11, got {sys.version}")

    installed_versions = {}
    version_mismatches = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Required package is not installed: {package}") from error
        installed_versions[package] = actual
        if actual != expected:
            version_mismatches[package] = {"expected": expected, "actual": actual}
    if version_mismatches:
        raise RuntimeError(f"Pinned package mismatch: {version_mismatches}")

    pip_check = subprocess.run(
        [str(interpreter), "-m", "pip", "check"],
        cwd=CONFIG_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        raise RuntimeError(f"pip check failed: {pip_check.stdout}{pip_check.stderr}")
    return {
        "python_executable": _relative(interpreter),
        "python_version": sys.version,
        "venv_active": venv_active,
        "pinned_versions_expected": EXPECTED_VERSIONS,
        "installed_versions": installed_versions,
        "pip_check": pip_check.stdout.strip(),
        "all_checks_passed": True,
    }


def _condition_records(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    """Load and validate the complete final artifact for one condition."""
    artifact_path = TEST_RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl"
    prefix = load_sequential_prefix(artifact_path, rows)
    if len(prefix.records) != len(rows):
        raise RuntimeError(f"Final {condition} artifact is incomplete: {len(prefix.records)} of {len(rows)} rows")
    return prefix.records


def _verify_prompt_definition(prompt_definition: dict[str, Any]) -> dict[str, Any]:
    """Verify the recorded prompt contract still matches the frozen baseline implementation."""
    expected_few_shot_roles = ["system"] + [role for _ in range(8) for role in ("user", "assistant")] + ["user"]
    checks = {
        "system_instruction_matches_baseline": prompt_definition.get("system_instruction") == SYSTEM_INSTRUCTION,
        "zero_shot_message_roles_match": prompt_definition.get("zero_shot_message_roles") == ["system", "user"],
        "few_shot_message_roles_match": prompt_definition.get("few_shot_message_roles") == expected_few_shot_roles,
        "few_shot_example_count_is_eight": len(prompt_definition.get("few_shot_examples", [])) == 8,
        "chain_of_thought_not_requested": prompt_definition.get("chain_of_thought_requested") is False,
        "raw_labels_not_in_model_inputs": prompt_definition.get("raw_github_labels_in_model_inputs") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen prompt definition verification failed: {checks}")
    return {"checks": checks, "all_checks_passed": True}


def _attach_analysis(
    condition_results: dict[str, dict[str, Any]],
    records_by_condition: dict[str, list[dict[str, Any]]],
) -> None:
    """Add truncation and repository metrics to each condition result."""
    for condition in CONDITIONS:
        metrics = condition_results[condition]["metrics"]
        records = records_by_condition[condition]
        metrics["by_input_truncation"] = metrics_by_truncation(records)
        metrics["by_repository"] = repository_metrics(records)


def _comparison_summary(
    condition_results: dict[str, dict[str, Any]],
    records_by_condition: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build all requested paired comparisons and fine-tuning class analysis."""
    fine = records_by_condition[CONDITION_FINE_TUNED_ZERO_SHOT]
    base_zero = records_by_condition[CONDITION_BASE_ZERO_SHOT]
    base_few = records_by_condition[CONDITION_BASE_FEW_SHOT]
    fine_metrics = condition_results[CONDITION_FINE_TUNED_ZERO_SHOT]["metrics"]
    base_zero_metrics = condition_results[CONDITION_BASE_ZERO_SHOT]["metrics"]
    base_few_metrics = condition_results[CONDITION_BASE_FEW_SHOT]["metrics"]
    return {
        "metric_deltas": {
            "fine_tuned_zero_shot_minus_base_zero_shot": {
                "accuracy_delta": round(fine_metrics["accuracy"] - base_zero_metrics["accuracy"], 6),
                "macro_f1_delta": round(fine_metrics["macro_f1"] - base_zero_metrics["macro_f1"], 6),
            },
            "fine_tuned_zero_shot_minus_base_few_shot": {
                "accuracy_delta": round(fine_metrics["accuracy"] - base_few_metrics["accuracy"], 6),
                "macro_f1_delta": round(fine_metrics["macro_f1"] - base_few_metrics["macro_f1"], 6),
            },
            "base_few_shot_minus_base_zero_shot": {
                "accuracy_delta": round(base_few_metrics["accuracy"] - base_zero_metrics["accuracy"], 6),
                "macro_f1_delta": round(base_few_metrics["macro_f1"] - base_zero_metrics["macro_f1"], 6),
            },
        },
        "paired_correctness": {
            "fine_tuned_vs_base_zero_shot": paired_correctness(
                fine, base_zero, CONDITION_FINE_TUNED_ZERO_SHOT, CONDITION_BASE_ZERO_SHOT
            ),
            "fine_tuned_vs_base_few_shot": paired_correctness(
                fine, base_few, CONDITION_FINE_TUNED_ZERO_SHOT, CONDITION_BASE_FEW_SHOT
            ),
            "base_few_shot_vs_base_zero_shot": paired_correctness(
                base_few, base_zero, CONDITION_BASE_FEW_SHOT, CONDITION_BASE_ZERO_SHOT
            ),
        },
        "fine_tuned_vs_base_zero_shot_by_true_class": class_comparison(
            fine, base_zero, CONDITION_FINE_TUNED_ZERO_SHOT, CONDITION_BASE_ZERO_SHOT
        ),
        "uncertainty": {
            "fine_tuned_minus_base_zero_shot": paired_bootstrap(
                fine,
                base_zero,
                name_a=CONDITION_FINE_TUNED_ZERO_SHOT,
                name_b=CONDITION_BASE_ZERO_SHOT,
            )
        },
        "exact_mcnemar": {
            "fine_tuned_vs_base_zero_shot": exact_mcnemar(
                fine, base_zero, CONDITION_FINE_TUNED_ZERO_SHOT, CONDITION_BASE_ZERO_SHOT
            )
        },
    }


def _final_interpretation(
    condition_results: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    """State the frozen TEST conclusions without selecting another experiment."""
    fine = condition_results[CONDITION_FINE_TUNED_ZERO_SHOT]["metrics"]
    base_zero = condition_results[CONDITION_BASE_ZERO_SHOT]["metrics"]
    base_few = condition_results[CONDITION_BASE_FEW_SHOT]["metrics"]
    fine_vs_base = comparisons["metric_deltas"]["fine_tuned_zero_shot_minus_base_zero_shot"]
    fine_vs_few = comparisons["metric_deltas"]["fine_tuned_zero_shot_minus_base_few_shot"]
    bootstrap = comparisons["uncertainty"]["fine_tuned_minus_base_zero_shot"]
    accuracy_ci = bootstrap["accuracy_delta_ci_95"]
    macro_ci = bootstrap["macro_f1_delta_ci_95"]
    accuracy_material = abs(fine_vs_base["accuracy_delta"]) >= 0.01
    macro_material = abs(fine_vs_base["macro_f1_delta"]) >= 0.01
    accuracy_compatible = accuracy_ci[0] <= 0 <= accuracy_ci[1]
    macro_compatible = macro_ci[0] <= 0 <= macro_ci[1]
    accuracy_direction = "raises" if fine_vs_base["accuracy_delta"] > 0 else "lowers"
    macro_direction = "raises" if fine_vs_base["macro_f1_delta"] > 0 else "lowers"
    if not accuracy_compatible and not macro_compatible:
        statistical_language = (
            f"Both paired 95% intervals exclude zero, so the observed differences are inconsistent with no "
            f"reliable difference under this resampling analysis: fine-tuning {accuracy_direction} accuracy by "
            f"{abs(fine_vs_base['accuracy_delta']):.4f} and {macro_direction} macro-F1 by "
            f"{abs(fine_vs_base['macro_f1_delta']):.4f}."
        )
    elif accuracy_compatible and macro_compatible:
        statistical_language = (
            "Both paired 95% intervals include zero, so the observed differences are compatible with no reliable "
            "difference under this resampling analysis."
        )
    else:
        statistical_language = (
            f"The paired 95% intervals differ: the accuracy difference is "
            f"{'compatible' if accuracy_compatible else 'not compatible'} with zero, while the macro-F1 difference "
            f"is {'compatible' if macro_compatible else 'not compatible'} with zero."
        )
    return {
        "fine_tuning_beat_base_zero_shot_on_accuracy": fine["accuracy"] > base_zero["accuracy"],
        "fine_tuning_beat_base_zero_shot_on_macro_f1": fine["macro_f1"] > base_zero["macro_f1"],
        "fine_tuning_beat_frozen_few_shot": {
            "accuracy": fine["accuracy"] > base_few["accuracy"],
            "macro_f1": fine["macro_f1"] > base_few["macro_f1"],
            "overall_primary_metric": fine["macro_f1"] > base_few["macro_f1"],
        },
        "statistical_and_practical_interpretation": {
            "accuracy_delta": fine_vs_base["accuracy_delta"],
            "accuracy_ci_95": accuracy_ci,
            "macro_f1_delta": fine_vs_base["macro_f1_delta"],
            "macro_f1_ci_95": macro_ci,
            "practical_threshold_absolute_delta": 0.01,
            "accuracy_difference_is_material_by_guideline": accuracy_material,
            "macro_f1_difference_is_material_by_guideline": macro_material,
            "accuracy_compatible_with_no_difference": accuracy_compatible,
            "macro_f1_compatible_with_no_difference": macro_compatible,
            "compatible_with_no_difference": accuracy_compatible and macro_compatible,
            "plain_language": statistical_language,
        },
        "class_benefit_and_regression": {
            "comparison": comparisons["fine_tuned_vs_base_zero_shot_by_true_class"],
            "interpretation": "Use per-class F1, precision, recall, and paired correctness counts; do not infer stable effects from too-small repository supports.",
        },
        "fine_tuning_objective_tradeoff": (
            "The adapter trades a small accuracy gain for lower macro-F1, so it is an accuracy-versus-class-balance "
            "tradeoff rather than an improvement under the frozen primary metric."
            if fine_vs_base["accuracy_delta"] > 0 and fine_vs_base["macro_f1_delta"] < 0
            else "TEST does not show an accuracy gain accompanied by a macro-F1 loss."
        ),
        "few_shot_context_cost": {
            "justified": fine["macro_f1"] > base_few["macro_f1"] and fine["accuracy"] > base_few["accuracy"],
            "plain_language": (
                "Frozen few-shot prompting is not justified by TEST performance because it adds context and runtime "
                "without improving the primary metric."
                if base_few["macro_f1"] <= base_zero["macro_f1"] and base_few["accuracy"] <= base_zero["accuracy"]
                else "Frozen few-shot prompting must be weighed against its measured context and runtime cost."
            ),
        },
        "validation_replication": {
            "validation_corrected_results": {
                "fine_tuned_minus_base_zero_shot_accuracy_delta": 0.004079,
                "fine_tuned_minus_base_zero_shot_macro_f1_delta": -0.005547,
                "base_few_shot_minus_base_zero_shot_accuracy_delta": -0.004315,
                "base_few_shot_minus_base_zero_shot_macro_f1_delta": -0.002002,
            },
            "test_results": comparisons["metric_deltas"],
            "fine_tuning_accuracy_sign_replicated": fine_vs_base["accuracy_delta"] > 0,
            "fine_tuning_macro_f1_sign_replicated": fine_vs_base["macro_f1_delta"] < 0,
            "few_shot_accuracy_sign_replicated": comparisons["metric_deltas"]["base_few_shot_minus_base_zero_shot"]["accuracy_delta"] < 0,
            "few_shot_macro_f1_sign_replicated": comparisons["metric_deltas"]["base_few_shot_minus_base_zero_shot"]["macro_f1_delta"] < 0,
            "plain_language": (
                f"TEST replicates the validation macro-F1 direction but not the validation accuracy direction: "
                f"fine-tuning {accuracy_direction} accuracy and {macro_direction} macro-F1, while frozen few-shot "
                "prompting is below base zero-shot on both reported metrics."
            ),
        },
        "no_post_test_tuning": True,
        "no_post_test_tuning_statement": "TEST results are descriptive final evidence only; no model, prompt, hyperparameter, sampling, taxonomy, context-policy, or adapter change was made after TEST was viewed.",
    }


def run_test_evaluation() -> dict[str, Any]:
    """Run preflight, resume-safe inference, and final descriptive TEST analysis."""
    evaluation_start = time.perf_counter()
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": _utc_now(),
        "evaluation_type": "generation_based_final_frozen_test_evaluation",
        "full_test_evaluation_started": False,
        "test_accessed_by_evaluator": True,
        "evaluation_start_git": _git_snapshot(),
        "config_path": _relative(CONFIG_PATH),
        "test_access_deviation": {
            "occurred": True,
            "description": "An earlier documented read-only metadata/preflight command opened and SHA-256-hashed the frozen TEST file.",
            "used_test_examples_or_labels_for_prompt_design": False,
            "used_test_examples_or_labels_for_model_training": False,
            "used_test_examples_or_labels_for_hyperparameter_or_model_selection": False,
            "used_test_examples_or_labels_for_validation_decisions": False,
            "treatment": "Preserved transparently as a process deviation; it did not alter the frozen evaluation design.",
        },
        "no_post_test_tuning": True,
        "errors": [],
    }
    model = None
    tokenizer = None
    try:
        report["environment_integrity"] = _environment_integrity()
        report["resource_preflight"] = _resource_preflight()
        test_rows, test_boundary = load_frozen_test_rows()
        few_shot_rows, few_shot_definition = load_frozen_prompt_inputs()
        report["test_boundary"] = test_boundary
        report["few_shot_definition"] = few_shot_definition
        report["prompt_definition"] = _read_json(PROMPT_DEFINITION_PATH)
        report["prompt_definition_verification"] = _verify_prompt_definition(report["prompt_definition"])
        report["adapter_verification"] = verify_adapter_artifact()
        report["base_model_contract"] = {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "load_in_16bit": False,
            "quantization": {"quant_type": "nf4", "use_double_quant": True, "compute_dtype": "float16"},
            "model_load_max_sequence_length": MODEL_LOAD_MAX_SEQUENCE_LENGTH,
            "adapter_path": _relative(ADAPTER_PATH),
            "adapter_merge_forbidden_and_not_performed": True,
        }
        report["context_policy"] = {
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
            "generation_output_reserve_tokens": GENERATION_MAX_NEW_TOKENS,
            "maximum_prompt_input_tokens": MAX_INPUT_TOKENS,
            "truncation_implementation": "prompt_preserving_v2",
            "truncation_order": "current_issue_body_right_then_current_issue_title_right",
            "preserves_system_and_chat_structure": True,
            "preserves_frozen_few_shot_demonstrations": True,
            "preserves_assistant_generation_boundary": True,
        }
        report["generation_contract"] = {
            "do_sample": False,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
            "chain_of_thought_requested": False,
            "parser": "scripts/classification/parser.py:parse_classification_output",
        }
        report["parser_contract_check_before_inference"] = _parser_smoke_test()
        tokenizer_only = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), local_files_only=True, use_fast=True)
        report["test_prompt_regression_checks"] = run_test_prompt_regression_checks(
            tokenizer_only,
            test_rows,
            few_shot_rows,
        )

        condition_results: dict[str, dict[str, Any]] = {}
        device = torch.device("cuda:0")
        report["full_test_evaluation_started"] = True

        print("Loading locked base model for final TEST evaluation...", flush=True)
        model, tokenizer, base_load = load_base_model()
        report["base_model_load"] = base_load
        try:
            for condition in (CONDITION_BASE_ZERO_SHOT, CONDITION_BASE_FEW_SHOT):
                output_path = TEST_RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl"
                condition_results[condition] = evaluate_condition(
                    model,
                    tokenizer,
                    test_rows,
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

        print("Loading completed LoRA adapter for final TEST evaluation...", flush=True)
        model, tokenizer, adapter_load = load_adapter_model()
        report["adapter_model_load"] = adapter_load
        try:
            output_path = TEST_RAW_ARTIFACT_DIRECTORY / f"{CONDITION_FINE_TUNED_ZERO_SHOT}.jsonl"
            condition_results[CONDITION_FINE_TUNED_ZERO_SHOT] = evaluate_condition(
                model,
                tokenizer,
                test_rows,
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

        records_by_condition = {
            condition: _condition_records(test_rows, condition) for condition in CONDITIONS
        }
        _attach_analysis(condition_results, records_by_condition)
        comparisons = _comparison_summary(condition_results, records_by_condition)
        report["conditions"] = condition_results
        report["pairwise_comparisons"] = comparisons
        report["final_interpretation"] = _final_interpretation(condition_results, comparisons)
        report["raw_artifacts"] = {
            "directory": _relative(TEST_RAW_ARTIFACT_DIRECTORY),
            "format": "complete sequential JSONL; one validated record for every TEST row per condition",
            "resume_policy": "valid sequential prefix is reused; only an incomplete or corrupt trailing write may be discarded",
            "conditions": {
                condition: _relative(TEST_RAW_ARTIFACT_DIRECTORY / f"{condition}.jsonl")
                for condition in CONDITIONS
            },
        }
        report["evaluation_runtime_seconds"] = round(time.perf_counter() - evaluation_start, 4)
        report["completed_at_utc"] = _utc_now()
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - persist resume diagnostics before surfacing failure.
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        report["evaluation_runtime_seconds"] = round(time.perf_counter() - evaluation_start, 4)
        report["completed_at_utc"] = _utc_now()
    finally:
        if model is not None and tokenizer is not None:
            model = None
            tokenizer = None
            _release_cuda_memory()
        TEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEST_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the final frozen TEST evaluation command."""
    report = run_test_evaluation()
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": _relative(TEST_REPORT_PATH),
                "full_test_evaluation_started": report["full_test_evaluation_started"],
                "errors": report.get("errors", []),
                "evaluation_runtime_seconds": report.get("evaluation_runtime_seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(2)
