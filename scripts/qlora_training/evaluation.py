"""Evaluate the sanity adapter with the existing deterministic JSON contract."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from classification.parser import parse_classification_output


def _token_ids(encoded: Any) -> list[int]:
    """Normalize one tokenizer result to a flat token sequence."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _render_prompt_ids(tokenizer: Any, messages: list[dict[str, str]], chat_template_kwargs: dict[str, Any]) -> list[int]:
    """Render the same zero-shot generation prompt used by the baseline."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return _token_ids(encoded)


def _metrics(records: list[dict[str, Any]], categories: tuple[str, ...]) -> dict[str, Any]:
    """Calculate the baseline-compatible accuracy, macro-F1, and validity metrics."""
    invalid_label = "__invalid_output__"
    y_true = [record["expected_category"] for record in records]
    y_pred = [record["predicted_category"] or invalid_label for record in records]
    labels = list(categories) + [invalid_label]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(categories),
        zero_division=0,
    )
    valid_count = sum(record["schema_valid"] for record in records)
    correct_count = sum(
        record["schema_valid"] and record["predicted_category"] == record["expected_category"]
        for record in records
    )
    return {
        "example_count": len(records),
        "accuracy": round(correct_count / len(records), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(categories), average="macro", zero_division=0)), 6),
        "valid_output_count": valid_count,
        "valid_output_percentage": round(100 * valid_count / len(records), 4),
        "invalid_output_count": len(records) - valid_count,
        "empty_think_wrapper_count": sum(record["empty_think_wrapper_stripped"] for record in records),
        "nonempty_think_output_count": sum(record["nonempty_thinking_content"] for record in records),
        "per_class": {
            category: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "accuracy": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, category in enumerate(categories)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "average_input_tokens": round(
            sum(record["input_token_count"] for record in records) / len(records), 4
        ),
        "input_truncated_count": sum(record["input_truncated"] for record in records),
        "input_truncated_percentage": round(
            100 * sum(record["input_truncated"] for record in records) / len(records), 4
        ),
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    categories: tuple[str, ...],
    max_length: int,
    max_new_tokens: int,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run deterministic one-example-at-a-time generation on the selected records."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config = deepcopy(generation_config)
        generation_config.max_new_tokens = None
        generation_config.do_sample = False
        generation_config.use_cache = True
        generation_config.pad_token_id = tokenizer.pad_token_id
    output_records = []
    try:
        for index, record in enumerate(records):
            prompt_ids = _render_prompt_ids(tokenizer, record["prompt"], chat_template_kwargs)
            full_input_token_count = len(prompt_ids)
            input_ids = prompt_ids[:max_length]
            input_truncated = len(input_ids) < full_input_token_count
            encoded = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
            }
            start_time = time.perf_counter()
            with torch.inference_mode():
                if generation_config is not None:
                    generation_config.max_length = len(input_ids) + max_new_tokens
                    sequence = model.generate(**encoded, generation_config=generation_config)[0]
                else:
                    sequence = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            response_ids = sequence[len(input_ids) :].tolist()
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and eos_token_id in response_ids:
                response_ids = response_ids[: response_ids.index(eos_token_id) + 1]
            raw_output = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            parsed_output = parse_classification_output(raw_output)
            output_records.append(
                {
                    "issue_id": record["issue_id"],
                    "repository": record["repository"],
                    "source_split": record["source_split"],
                    "source_row_index": record["source_row_index"],
                    "expected_category": record["target_category"],
                    "raw_model_output": raw_output,
                    "normalized_json_output": parsed_output.normalized_output,
                    "empty_think_wrapper_stripped": parsed_output.empty_think_wrapper_stripped,
                    "nonempty_thinking_content": parsed_output.nonempty_thinking_content,
                    "predicted_category": parsed_output.predicted_category,
                    "schema_valid": parsed_output.schema_valid,
                    "parse_error": parsed_output.parse_error,
                    "input_token_count": len(input_ids),
                    "full_input_token_count": full_input_token_count,
                    "input_truncated": input_truncated,
                    "output_token_count": len(response_ids),
                    "inference_time_seconds": round(elapsed, 6),
                    "evaluation_index": index,
                }
            )
    finally:
        tokenizer.padding_side = original_padding_side
    return {
        "metrics": _metrics(output_records, categories),
        "records": output_records,
    }
