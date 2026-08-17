"""Calculate the shared validation metrics and error summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .config import TARGET_CATEGORIES


INVALID_LABEL = "__invalid_output__"


def _error_family(error: str) -> str:
    """Collapse detailed parser messages into stable report categories."""
    return error.split(":", 1)[0]


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate identical metrics for valid outputs, parse errors, and inference errors."""
    if not records:
        raise ValueError("Cannot calculate metrics for zero records")

    y_true = [record["expected_category"] for record in records]
    y_pred = [record["predicted_category"] or INVALID_LABEL for record in records]
    labels = list(TARGET_CATEGORIES) + [INVALID_LABEL]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(TARGET_CATEGORIES),
        zero_division=0,
    )
    valid_count = sum(bool(record["schema_valid"]) for record in records)
    correct_count = sum(
        bool(record["schema_valid"]) and record["predicted_category"] == record["expected_category"]
        for record in records
    )
    parse_errors = Counter(
        _error_family(record["parse_error"])
        for record in records
        if record["status"] == "ok" and record["parse_error"]
    )
    inference_errors = Counter(
        record.get("error_type", "inference_error")
        for record in records
        if record["status"] == "inference_error"
    )
    predicted_distribution = Counter(y_pred)
    input_truncated_count = sum(bool(record["input_truncated"]) for record in records)
    total_inference_runtime = sum(float(record["inference_time_seconds"]) for record in records)

    return {
        "example_count": len(records),
        "total_examples": len(records),
        "accuracy": round(correct_count / len(records), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(TARGET_CATEGORIES), average="macro", zero_division=0)), 6),
        "primary_metric": "macro_f1",
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
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predicted_class_distribution": {
            category: int(predicted_distribution[category]) for category in labels
        },
        "input_truncated_count": input_truncated_count,
        "input_truncated_percentage": round(100 * input_truncated_count / len(records), 4),
        "average_input_tokens": round(
            sum(int(record["input_token_count"]) for record in records) / len(records), 4
        ),
        "average_full_input_tokens": round(
            sum(int(record["full_input_token_count"]) for record in records) / len(records), 4
        ),
        "maximum_full_input_tokens": max(int(record["full_input_token_count"]) for record in records),
        "maximum_fed_input_tokens": max(int(record["input_token_count"]) for record in records),
        "average_output_tokens": round(
            sum(int(record["output_token_count"]) for record in records) / len(records), 4
        ),
        "average_inference_time_seconds_per_example": round(total_inference_runtime / len(records), 6),
        "total_inference_runtime_seconds": round(total_inference_runtime, 4),
        "status_counts": {
            status: int(sum(record["status"] == status for record in records))
            for status in ("ok", "inference_error")
        },
        "output_parsing_error_count": int(sum(
            record["status"] == "ok" and not record["schema_valid"] for record in records
        )),
        "parse_error_counts": dict(sorted(parse_errors.items())),
        "inference_error_counts": dict(sorted(inference_errors.items())),
        "empty_think_wrapper_count": int(sum(record["empty_think_wrapper_stripped"] for record in records)),
        "nonempty_think_output_count": int(sum(record["nonempty_thinking_content"] for record in records)),
    }
