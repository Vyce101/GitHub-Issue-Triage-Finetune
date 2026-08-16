"""Load the locked model and evaluate deterministic prompt conditions in batches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .config import (
    BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
    EVALUATION_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_CATEGORIES,
)
from .prompts import few_shot_messages, zero_shot_messages
from classification.parser import parse_classification_output


def load_locked_model() -> tuple[Any, Any]:
    """Load the untouched locked model with 4-bit weights and FP16 compute."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        device_map="sequential",
        trust_remote_code=False,
        use_gradient_checkpointing=False,
    )
    FastLanguageModel.for_inference(model)
    if sorted({str(parameter.device) for parameter in model.parameters()}) != ["cuda:0"]:
        raise RuntimeError("Baseline model is not fully placed on cuda:0")
    return model, tokenizer


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render one chat conversation with thinking disabled when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generated_text_and_count(
    tokenizer: Any,
    sequence: torch.Tensor,
    padded_input_length: int,
) -> tuple[str, int]:
    """Decode the generated suffix and count tokens through the first EOS."""
    response_ids = sequence[padded_input_length:].tolist()
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and eos_token_id in response_ids:
        response_ids = response_ids[: response_ids.index(eos_token_id) + 1]
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return text, len(response_ids)


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate balanced classification metrics and an invalid-output bucket."""
    invalid_label = "__invalid_output__"
    y_true = [record["expected_category"] for record in records]
    y_pred = [record["predicted_category"] or invalid_label for record in records]
    labels = list(TARGET_CATEGORIES) + [invalid_label]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(TARGET_CATEGORIES),
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    valid_count = sum(record["schema_valid"] for record in records)
    correct_count = sum(
        record["schema_valid"] and record["predicted_category"] == record["expected_category"]
        for record in records
    )
    return {
        "example_count": len(records),
        "accuracy": round(correct_count / len(records), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(TARGET_CATEGORIES), average="macro", zero_division=0)), 6),
        "valid_output_count": valid_count,
        "valid_output_percentage": round(100 * valid_count / len(records), 4),
        "invalid_output_count": len(records) - valid_count,
        "per_class": {
            category: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, category in enumerate(TARGET_CATEGORIES)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "average_input_tokens": round(sum(record["input_token_count"] for record in records) / len(records), 4),
        "average_output_tokens": round(sum(record["output_token_count"] for record in records) / len(records), 4),
        "average_inference_time_seconds_per_issue": round(sum(record["inference_time_seconds"] for record in records) / len(records), 6),
        "average_batch_tokens_per_second": round(sum(record["batch_tokens_per_second"] for record in records) / len(records), 6),
        "input_truncated_count": sum(record["input_truncated"] for record in records),
        "input_truncated_percentage": round(100 * sum(record["input_truncated"] for record in records) / len(records), 4),
        "maximum_full_input_tokens": max(record["full_input_token_count"] for record in records),
        "maximum_fed_input_tokens": max(record["input_token_count"] for record in records),
    }


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Run one deterministic prompt condition and write per-issue JSONL results."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    records = []
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        batch_start = 0
        while batch_start < len(rows):
            batch_rows = rows[batch_start : batch_start + EVALUATION_BATCH_SIZE]
            if condition == "zero_shot":
                conversations = [zero_shot_messages(row) for row in batch_rows]
            elif condition == "few_shot":
                conversations = [few_shot_messages(row, few_shot_rows) for row in batch_rows]
            else:
                raise ValueError(f"Unknown baseline condition: {condition}")
            prompt_texts = [_render_prompt(tokenizer, conversation) for conversation in conversations]
            full_input_token_counts = [
                len(tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"])
                for prompt_text in prompt_texts
            ]
            if max(full_input_token_counts) > 2048 and len(batch_rows) > 1:
                batch_rows = batch_rows[:1]
                continue
            encoded = tokenizer(
                prompt_texts,
                add_special_tokens=False,
                truncation=True,
                max_length=BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            padded_input_length = int(encoded["input_ids"].shape[1])
            if padded_input_length > BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH:
                raise RuntimeError(
                    f"Prompt input length {padded_input_length} exceeds the temporary baseline "
                    f"inference ceiling {BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH}"
                )
            start_time = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - start_time
            batch_generated_tokens = 0
            batch_records = []
            for row, full_input_token_count, input_token_count, sequence in zip(batch_rows, full_input_token_counts, input_token_counts, sequences):
                raw_output, output_token_count = _generated_text_and_count(
                    tokenizer,
                    sequence,
                    padded_input_length,
                )
                parsed_output = parse_classification_output(raw_output)
                batch_generated_tokens += output_token_count
                batch_records.append(
                    {
                        "issue_id": row["issue_id"],
                        "repository": row["repository"],
                        "source_split": row["source_split"],
                        "source_row_index": row["source_row_index"],
                        "expected_category": row["target_category"],
                        "raw_model_output": raw_output,
                        "normalized_json_output": parsed_output.normalized_output,
                        "empty_think_wrapper_stripped": parsed_output.empty_think_wrapper_stripped,
                        "nonempty_thinking_content": parsed_output.nonempty_thinking_content,
                        "predicted_category": parsed_output.predicted_category,
                        "schema_valid": parsed_output.schema_valid,
                        "parse_error": parsed_output.parse_error,
                        "input_token_count": int(input_token_count),
                        "full_input_token_count": full_input_token_count,
                        "input_truncated": full_input_token_count > int(input_token_count),
                        "output_token_count": output_token_count,
                        "batch_index": batch_start // EVALUATION_BATCH_SIZE,
                    }
                )
            batch_tokens_per_second = batch_generated_tokens / batch_elapsed if batch_elapsed else 0.0
            for record in batch_records:
                record["batch_elapsed_seconds"] = batch_elapsed
                record["inference_time_seconds"] = batch_elapsed / len(batch_records)
                record["batch_tokens_per_second"] = batch_tokens_per_second
                output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                records.append(record)
            output_file.flush()
            batch_start += len(batch_rows)
    tokenizer.padding_side = original_padding_side
    return {
        "condition": condition,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        },
        "batch_size": EVALUATION_BATCH_SIZE,
        "temporary_input_limit": BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH,
        "temporary_input_limit_is_final_training_context": False,
        "metrics": _metrics(records),
    }
