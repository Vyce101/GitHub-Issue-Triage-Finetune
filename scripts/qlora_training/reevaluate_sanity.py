"""Re-evaluate the saved 64-example sanity adapter under the JSON contract."""

from __future__ import annotations

import gc
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data import sanity_dataset_records, select_sanity_examples
from .evaluation import evaluate_model
from .run_sanity import _compact_report, _write_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/training_sanity_config.json"


def _assess_overfit(pre_metrics: dict[str, Any], post_metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed sanity success criterion to the corrected post-evaluation."""
    return {
        "success_criterion": "post accuracy >= 0.90, post macro-F1 >= 0.90, valid JSON >= 100%, and relative macro-F1 improvement >= 50%",
        "accuracy_delta": round(post_metrics["accuracy"] - pre_metrics["accuracy"], 6),
        "macro_f1_delta": round(post_metrics["macro_f1"] - pre_metrics["macro_f1"], 6),
        "macro_f1_relative_improvement": round(
            (post_metrics["macro_f1"] - pre_metrics["macro_f1"]) / pre_metrics["macro_f1"],
            6,
        ),
        "passed": bool(
            post_metrics["accuracy"] >= 0.90
            and post_metrics["macro_f1"] >= 0.90
            and post_metrics["valid_output_percentage"] >= 100.0
            and (
                post_metrics["macro_f1"] - pre_metrics["macro_f1"]
            ) / pre_metrics["macro_f1"] >= 0.50
        ),
    }


def reevaluate_saved_adapter(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Re-evaluate the saved adapter without training or accessing held-out splits."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report_path = PROJECT_ROOT / config["artifacts"]["tracked_report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("full_training_started") or report.get("validation_accessed") or report.get("test_accessed"):
        raise RuntimeError("The saved report is outside the train-only sanity scope")
    output_dir = PROJECT_ROOT / config["artifacts"]["temporary_output_dir"]
    train_path = PROJECT_ROOT / config["data"]["source_path"]
    model = None
    tokenizer = None
    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
        examples = select_sanity_examples(
            train_path,
            tokenizer,
            categories=tuple(config["data"]["target_categories"]),
            examples_per_category=config["data"]["examples_per_category"],
            max_sequence_length=config["model"]["max_sequence_length"],
            short_sequence_token_ceiling=config["data"]["short_sequence_token_ceiling"],
            chat_template_kwargs=config["data"]["chat_template_kwargs"],
        )
        records = sanity_dataset_records(
            examples,
            chat_template_kwargs=config["data"]["chat_template_kwargs"],
        )
        counts = Counter(record["target_category"] for record in records)
        expected_counts = Counter(
            {category: config["data"]["examples_per_category"] for category in config["data"]["target_categories"]}
        )
        if counts != expected_counts:
            raise AssertionError(f"Unexpected sanity class counts: {counts}")

        model_config = config["model"]
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_config["model_id"],
            revision=model_config["revision"],
            max_seq_length=model_config["max_sequence_length"],
            dtype=torch.float16,
            load_in_4bit=model_config["load_in_4bit"],
            load_in_8bit=model_config["load_in_8bit"],
            load_in_16bit=model_config["load_in_16bit"],
            device_map=model_config["device_map"],
            trust_remote_code=model_config["trust_remote_code"],
            use_gradient_checkpointing=False,
            local_files_only=model_config["local_files_only"],
        )
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=False)
        FastLanguageModel.for_inference(model)
        post_eval = evaluate_model(
            model,
            tokenizer,
            records,
            device=next(model.parameters()).device,
            categories=tuple(config["data"]["target_categories"]),
            max_length=config["model"]["max_sequence_length"],
            max_new_tokens=config["evaluation"]["max_new_tokens"],
            chat_template_kwargs=config["data"]["chat_template_kwargs"],
        )
        report["post_training_evaluation"] = post_eval
        report["overfit_assessment"] = _assess_overfit(
            report["pre_training_evaluation"]["metrics"],
            post_eval["metrics"],
        )
        report["evaluation_contract"] = {
            "empty_think_wrapper_policy": "strip only an empty <think>...</think> template wrapper; reject non-empty think content",
            "empty_think_wrappers_stripped_post_training": post_eval["metrics"]["empty_think_wrapper_count"],
            "nonempty_think_outputs_post_training": post_eval["metrics"]["nonempty_think_output_count"],
            "json_evaluated_after_empty_wrapper_normalization": True,
        }
        report["status"] = "passed_sanity" if report["overfit_assessment"]["passed"] else "failed_overfit"
        report["post_evaluation_only"] = True
        report["post_evaluation_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["diagnosis"] = (
            "The first post-training evaluation emitted Qwen's empty think template wrapper before the correct JSON. "
            "The evaluator now strips only that empty wrapper and rejects non-empty think content."
        )
        _compact_report(report)
        _write_report(report_path, report)
        return report
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    """Run the post-training evaluation-only correction and print its status."""
    report = reevaluate_saved_adapter()
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": "results/training_sanity_check.json",
                "training_restarted": False,
                "full_training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
