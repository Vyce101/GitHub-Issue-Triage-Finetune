"""Run token analysis, train-only prompt selection, and prompting baselines."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer

from baseline.config import (
    BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
    PROMPT_DEVELOPMENT_MAX_ZERO_SHOT_INPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    RESULTS_DIRECTORY,
    TARGET_CATEGORIES,
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
)
from baseline.data import (
    read_jsonl,
    selection_record,
    select_few_shot_rows,
    select_prompt_development_rows,
    token_length_analysis,
)
from baseline.evaluate import evaluate_condition, load_locked_model
from baseline.prompts import prompt_definition, zero_shot_messages


def _write_json(path, value) -> None:
    """Write readable deterministic JSON under the baseline results directory."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _assert_train_only(prompt_rows, train_rows, name: str) -> None:
    """Ensure every selected prompt-development row comes from the train split."""
    train_keys = {(row["source_split"], row["source_row_index"]) for row in train_rows}
    selected_keys = {(row["source_split"], row["source_row_index"]) for row in prompt_rows}
    if not selected_keys.issubset(train_keys) or any(row["source_split"] != "train" for row in prompt_rows):
        raise RuntimeError(f"{name} contains a non-train example")


def main() -> None:
    """Run the complete prompting-only baseline experiment."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(TRAIN_SPLIT_PATH)
    validation_rows = read_jsonl(VALIDATION_SPLIT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    length_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "train": token_length_analysis(tokenizer, train_rows, zero_shot_messages),
        "validation": token_length_analysis(tokenizer, validation_rows, zero_shot_messages),
        "validation_used_for_prompt_development": False,
        "test_split_accessed": False,
    }
    _write_json(RESULTS_DIRECTORY / "token_length_analysis.json", length_report)

    prompt_development_rows = select_prompt_development_rows(train_rows, tokenizer)
    few_shot_rows = select_few_shot_rows(train_rows, prompt_development_rows, tokenizer)
    _assert_train_only(prompt_development_rows, train_rows, "prompt-development subset")
    _assert_train_only(few_shot_rows, train_rows, "few-shot examples")
    development_keys = {(row["source_split"], row["source_row_index"]) for row in prompt_development_rows}
    few_shot_keys = {(row["source_split"], row["source_row_index"]) for row in few_shot_rows}
    if development_keys & few_shot_keys:
        raise RuntimeError("Few-shot examples overlap the prompt-development subset")
    if len(prompt_development_rows) != 400 or len(few_shot_rows) != 8:
        raise RuntimeError("Prompt-development selection has the wrong size")
    if {row["target_category"] for row in prompt_development_rows} != set(TARGET_CATEGORIES):
        raise RuntimeError("Prompt-development selection is missing a category")
    selection_report = {
        "selection_method": "Deterministic lexical repository round-robin for development rows; shortest title/body token candidates with distinct repositories for demonstrations.",
        "prompt_development_zero_shot_input_token_cap": PROMPT_DEVELOPMENT_MAX_ZERO_SHOT_INPUT_TOKENS,
        "prompt_development_length_cap_is_final_training_context": False,
        "source_split": "train",
        "prompt_development_row_count": len(prompt_development_rows),
        "prompt_development_class_counts": {
            category: sum(row["target_category"] == category for row in prompt_development_rows)
            for category in TARGET_CATEGORIES
        },
        "prompt_development_repository_count": len({row["repository"] for row in prompt_development_rows}),
        "prompt_development_examples": selection_record(prompt_development_rows),
        "few_shot_example_count": len(few_shot_rows),
        "few_shot_class_counts": {
            category: sum(row["target_category"] == category for row in few_shot_rows)
            for category in TARGET_CATEGORIES
        },
        "few_shot_repository_count": len({row["repository"] for row in few_shot_rows}),
        "few_shot_examples": selection_record(few_shot_rows),
        "few_shot_excluded_from_prompt_development": True,
        "validation_or_test_examples_selected": False,
    }
    _write_json(RESULTS_DIRECTORY / "prompt_development_selection.json", selection_report)
    _write_json(RESULTS_DIRECTORY / "prompts.json", prompt_definition(few_shot_rows))

    del tokenizer
    gc.collect()
    model, model_tokenizer = load_locked_model()
    try:
        zero_summary = evaluate_condition(
            model,
            model_tokenizer,
            prompt_development_rows,
            "zero_shot",
            few_shot_rows,
            RESULTS_DIRECTORY / "zero_shot_results.jsonl",
        )
        few_summary = evaluate_condition(
            model,
            model_tokenizer,
            prompt_development_rows,
            "few_shot",
            few_shot_rows,
            RESULTS_DIRECTORY / "few_shot_results.jsonl",
        )
    finally:
        del model
        del model_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    metrics_report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "evaluation_split": "train",
        "evaluation_subset": "prompt_development_selection.json",
        "validation_evaluated": False,
        "test_evaluated": False,
        "temporary_baseline_inference_ceiling_tokens": BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
        "temporary_baseline_inference_ceiling_is_final_training_context": False,
        "zero_shot": zero_summary,
        "few_shot": few_summary,
        "comparison": {
            "accuracy_delta_few_shot_minus_zero_shot": round(few_summary["metrics"]["accuracy"] - zero_summary["metrics"]["accuracy"], 6),
            "macro_f1_delta_few_shot_minus_zero_shot": round(few_summary["metrics"]["macro_f1"] - zero_summary["metrics"]["macro_f1"], 6),
            "valid_output_percentage_delta_few_shot_minus_zero_shot": round(few_summary["metrics"]["valid_output_percentage"] - zero_summary["metrics"]["valid_output_percentage"], 4),
            "average_input_token_delta_few_shot_minus_zero_shot": round(few_summary["metrics"]["average_input_tokens"] - zero_summary["metrics"]["average_input_tokens"], 4),
            "average_inference_time_delta_few_shot_minus_zero_shot": round(few_summary["metrics"]["average_inference_time_seconds_per_issue"] - zero_summary["metrics"]["average_inference_time_seconds_per_issue"], 6),
        },
        "raw_github_labels_in_model_inputs": False,
        "training_performed": False,
        "lora_adapter_created": False,
    }
    _write_json(RESULTS_DIRECTORY / "metrics.json", metrics_report)
    print(json.dumps(metrics_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
