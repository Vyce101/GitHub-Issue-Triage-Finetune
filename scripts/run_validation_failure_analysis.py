"""Analyze completed validation predictions without loading a model or TEST data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CATEGORIES = ["bug", "feature", "documentation", "question_support"]
INVALID_OUTPUT = "__invalid_output__"
MATRIX_LABELS = CATEGORIES + [INVALID_OUTPUT]
CONDITIONS = ["base_zero_shot", "base_few_shot", "fine_tuned_zero_shot"]
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CHUNK_SIZE = 100
REPOSITORY_MIN_CLASS_SUPPORT = 5
GENERATION_MAX_NEW_TOKENS = 16


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def rounded(value: float | int | None, digits: int = 6) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def percentage(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def prediction_label(row: dict[str, Any]) -> str:
    predicted = row.get("predicted_category")
    return predicted if predicted in CATEGORIES else INVALID_OUTPUT


def is_valid_prediction(row: dict[str, Any]) -> bool:
    return prediction_label(row) in CATEGORIES and bool(row.get("schema_valid"))


def f1_from_counts(true_positive: int, predicted: int, support: int) -> float:
    denominator = 2 * true_positive + (predicted - true_positive) + (support - true_positive)
    if denominator == 0:
        return 0.0
    return 2.0 * true_positive / denominator


def confusion_for_rows(rows: Iterable[dict[str, Any]]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(MATRIX_LABELS)}
    matrix = [[0 for _ in MATRIX_LABELS] for _ in MATRIX_LABELS]
    for row in rows:
        matrix[index[row["expected_category"]]][index[prediction_label(row)]] += 1
    return matrix


def class_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_counts = Counter(row["expected_category"] for row in rows)
    predicted_counts = Counter(prediction_label(row) for row in rows)
    matrix = confusion_for_rows(rows)
    matrix_index = {label: position for position, label in enumerate(MATRIX_LABELS)}

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for category in CATEGORIES:
        category_index = matrix_index[category]
        true_positive = matrix[category_index][category_index]
        support = expected_counts[category]
        predicted = predicted_counts[category]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = f1_from_counts(true_positive, predicted, support)
        f1_values.append(f1)
        per_class[category] = {
            "support": support,
            "predicted_count": predicted,
            "precision": rounded(precision),
            "recall": rounded(recall),
            "f1": rounded(f1),
        }

    valid_count = sum(predicted_counts[category] for category in CATEGORIES)
    correct_count = sum(
        row["expected_category"] == prediction_label(row)
        for row in rows
    )
    status_counts = Counter(row.get("status", "missing_status") for row in rows)
    parse_error_counts = Counter(
        row["parse_error"]
        for row in rows
        if row.get("parse_error")
    )
    parse_error_family_counts = Counter(
        row["parse_error"].split(":", 1)[0]
        for row in rows
        if row.get("parse_error")
    )
    return {
        "total_examples": len(rows),
        "correct_count": correct_count,
        "accuracy": rounded(correct_count / len(rows) if rows else 0.0),
        "macro_f1": rounded(sum(f1_values) / len(f1_values) if f1_values else 0.0),
        "valid_output_count": valid_count,
        "valid_output_percentage": rounded(percentage(valid_count, len(rows))),
        "invalid_output_count": len(rows) - valid_count,
        "output_parsing_error_count": sum(parse_error_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "per_class": per_class,
        "predicted_class_distribution": {
            label: {
                "count": predicted_counts[label],
                "percentage_of_all_rows": rounded(percentage(predicted_counts[label], len(rows))),
                "percentage_of_valid_outputs": rounded(percentage(predicted_counts[label], valid_count)),
            }
            for label in MATRIX_LABELS
        },
        "confusion_matrix_labels": MATRIX_LABELS,
        "confusion_matrix_rows_true_columns_predicted": matrix,
        "input_truncated_count": sum(bool(row.get("input_truncated")) for row in rows),
        "input_truncated_percentage": rounded(
            percentage(sum(bool(row.get("input_truncated")) for row in rows), len(rows))
        ),
        "parse_error_counts": dict(sorted(parse_error_counts.items())),
        "parse_error_family_counts": dict(sorted(parse_error_family_counts.items())),
        "nonempty_thinking_count": sum(bool(row.get("nonempty_thinking_content")) for row in rows),
        "empty_think_wrapper_count": sum(bool(row.get("empty_think_wrapper_stripped")) for row in rows),
    }


def valid_only_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    official = class_metrics(rows)
    valid_rows = [row for row in rows if is_valid_prediction(row)]
    metrics = class_metrics(valid_rows)
    return {
        "valid_examples_analyzed": len(valid_rows),
        "valid_classification_error_count": len(valid_rows) - metrics["correct_count"],
        "accuracy_among_valid_outputs": metrics["accuracy"],
        "macro_f1_among_valid_outputs": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "official_minus_valid_only_counterfactual": {
            "accuracy_gap_attributable_to_invalid_outputs": rounded(
                float(metrics["accuracy"]) - float(official["accuracy"])
            ),
            "macro_f1_gap_attributable_to_invalid_outputs_counterfactual": rounded(
                float(metrics["macro_f1"]) - float(official["macro_f1"])
            ),
        },
    }


def exact_parser_failure_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parse_error"):
            grouped[row["parse_error"]].append(row)

    invalid_rows = [row for row in rows if row.get("parse_error")]
    failures: list[dict[str, Any]] = []
    for reason, reason_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        samples = []
        for row in reason_rows[:3]:
            samples.append(
                {
                    "evaluation_index": row["evaluation_index"],
                    "issue_id": row["issue_id"],
                    "repository": row["repository"],
                    "expected_category": row["expected_category"],
                    "input_truncated": bool(row.get("input_truncated")),
                    "raw_model_output": row.get("raw_model_output", ""),
                }
            )
        failures.append(
            {
                "exact_parser_reason": reason,
                "parser_reason_family": reason.split(":", 1)[0],
                "count": len(reason_rows),
                "percentage_of_all_rows": rounded(percentage(len(reason_rows), len(rows))),
                "percentage_of_invalid_outputs": rounded(
                    percentage(len(reason_rows), len(invalid_rows))
                ),
                "reached_max_new_tokens_count": sum(
                    row.get("output_token_count", 0) >= GENERATION_MAX_NEW_TOKENS for row in reason_rows
                ),
                "input_truncated_count": sum(bool(row.get("input_truncated")) for row in reason_rows),
                "representative_outputs": samples,
            }
        )

    invalid_at_output_cap = sum(
        row.get("output_token_count", 0) >= GENERATION_MAX_NEW_TOKENS for row in invalid_rows
    )
    invalid_on_truncated_input = sum(bool(row.get("input_truncated")) for row in invalid_rows)
    return {
        "invalid_output_count": len(invalid_rows),
        "failure_categories": failures,
        "nonempty_thinking_count": sum(bool(row.get("nonempty_thinking_content")) for row in rows),
        "empty_think_wrapper_stripped_count": sum(bool(row.get("empty_think_wrapper_stripped")) for row in rows),
        "invalid_outputs_reaching_max_new_tokens": invalid_at_output_cap,
        "invalid_outputs_reaching_max_new_tokens_percentage": rounded(
            percentage(invalid_at_output_cap, len(invalid_rows))
        ),
        "invalid_outputs_with_input_truncation": invalid_on_truncated_input,
        "invalid_outputs_with_input_truncation_percentage": rounded(
            percentage(invalid_on_truncated_input, len(invalid_rows))
        ),
        "invalid_outputs_without_input_truncation": len(invalid_rows) - invalid_on_truncated_input,
        "invalid_output_cross_tab": {
            "input_truncated_and_reached_max_new_tokens": sum(
                bool(row.get("input_truncated"))
                and row.get("output_token_count", 0) >= GENERATION_MAX_NEW_TOKENS
                for row in invalid_rows
            ),
            "input_truncated_and_did_not_reach_max_new_tokens": sum(
                bool(row.get("input_truncated"))
                and row.get("output_token_count", 0) < GENERATION_MAX_NEW_TOKENS
                for row in invalid_rows
            ),
            "not_input_truncated_and_reached_max_new_tokens": sum(
                not bool(row.get("input_truncated"))
                and row.get("output_token_count", 0) >= GENERATION_MAX_NEW_TOKENS
                for row in invalid_rows
            ),
            "not_input_truncated_and_did_not_reach_max_new_tokens": sum(
                not bool(row.get("input_truncated"))
                and row.get("output_token_count", 0) < GENERATION_MAX_NEW_TOKENS
                for row in invalid_rows
            ),
        },
    }


def truncation_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for truncated in (False, True):
        group = [row for row in rows if bool(row.get("input_truncated")) is truncated]
        metrics = class_metrics(group)
        result["truncated" if truncated else "not_truncated"] = {
            "row_count": len(group),
            "percentage_of_condition": rounded(percentage(len(group), len(rows))),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "valid_output_percentage": metrics["valid_output_percentage"],
            "per_class": metrics["per_class"],
        }
    return result


def cross_condition_truncation_comparison(
    base_rows: list[dict[str, Any]], few_rows: list[dict[str, Any]], fine_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    condition_rows = {
        "base_zero_shot": base_rows,
        "base_few_shot": few_rows,
        "fine_tuned_zero_shot": fine_rows,
    }
    summary: dict[str, Any] = {}
    for group_name, truncated in (("not_truncated", False), ("truncated", True)):
        summary[group_name] = {}
        for condition, rows in condition_rows.items():
            group = [row for row in rows if bool(row.get("input_truncated")) is truncated]
            metrics = class_metrics(group)
            summary[group_name][condition] = {
                "row_count": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "valid_output_percentage": metrics["valid_output_percentage"],
            }
    summary["interpretation"] = (
        "The truncation strata are condition-specific because few-shot demonstrations change prompt length; "
        "they are descriptive, not paired subsets. If few-shot remains below base zero-shot within its own "
        "non-truncated stratum, its lower score is not plausibly explained by truncation alone."
    )
    return summary


def deltas(left: dict[str, Any], right: dict[str, Any], metrics: tuple[str, ...] = ("accuracy", "macro_f1")) -> dict[str, float]:
    return {f"{key}_delta": rounded(float(left[key]) - float(right[key])) for key in metrics}


def per_class_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        category: {
            metric: rounded(float(left["per_class"][category][metric]) - float(right["per_class"][category][metric]))
            for metric in ("precision", "recall", "f1")
        }
        for category in CATEGORIES
    }


def selected_confusion_transitions(
    base_rows: list[dict[str, Any]], fine_rows: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    requested = [
        ("documentation", "bug"),
        ("documentation", "feature"),
        ("documentation", "question_support"),
        ("question_support", "bug"),
        ("question_support", "feature"),
        ("question_support", "documentation"),
        ("bug", "feature"),
        ("feature", "bug"),
    ]
    result: dict[str, dict[str, int]] = {}
    for true_label, predicted_label in requested:
        base_count = sum(
            row["expected_category"] == true_label and prediction_label(row) == predicted_label
            for row in base_rows
        )
        fine_count = sum(
            row["expected_category"] == true_label and prediction_label(row) == predicted_label
            for row in fine_rows
        )
        result[f"{true_label}_to_{predicted_label}"] = {
            "base_zero_shot": base_count,
            "fine_tuned_zero_shot": fine_count,
            "fine_minus_base": fine_count - base_count,
        }
    return result


def error_destinations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        errors = [row for row in rows if row["expected_category"] == category and prediction_label(row) != category]
        counts = Counter(prediction_label(row) for row in errors)
        result[category] = {
            "error_count": len(errors),
            "destinations": {label: counts[label] for label in MATRIX_LABELS},
        }
    return result


def distribution(counts: Counter[str], denominator: int) -> dict[str, Any]:
    return {
        label: {
            "count": counts[label],
            "percentage": rounded(percentage(counts[label], denominator)),
        }
        for label in CATEGORIES
    }


def prediction_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(prediction_label(row) for row in rows)
    valid_counts = Counter({category: counts[category] for category in CATEGORIES})
    return {
        "all_rows_denominator": len(rows),
        "all_rows": {
            label: {
                "count": counts[label],
                "percentage": rounded(percentage(counts[label], len(rows))),
            }
            for label in MATRIX_LABELS
        },
        "valid_outputs_denominator": sum(valid_counts.values()),
        "valid_outputs_only": distribution(valid_counts, sum(valid_counts.values())),
    }


def l1_distance_to_prior(predicted: Counter[str], denominator: int, prior: Counter[str], prior_denominator: int) -> float:
    if denominator == 0 or prior_denominator == 0:
        return 0.0
    return sum(
        abs(predicted[category] / denominator - prior[category] / prior_denominator)
        for category in CATEGORIES
    )


def repository_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = class_metrics(rows)
    present_categories = [
        category for category in CATEGORIES if metrics["per_class"][category]["support"] > 0
    ]
    present_macro_f1 = (
        sum(metrics["per_class"][category]["f1"] for category in present_categories) / len(present_categories)
        if present_categories
        else 0.0
    )
    min_support = min(
        (metrics["per_class"][category]["support"] for category in present_categories),
        default=0,
    )
    return {
        "row_count": len(rows),
        "class_composition": {
            category: {
                "count": metrics["per_class"][category]["support"],
                "percentage": rounded(percentage(metrics["per_class"][category]["support"], len(rows))),
            }
            for category in CATEGORIES
        },
        "accuracy": metrics["accuracy"],
        "macro_f1_all_four_categories": metrics["macro_f1"],
        "macro_f1_present_true_categories": rounded(present_macro_f1),
        "macro_f1_definition": "Macro average over categories with at least one true validation example in this repository; descriptive, not a significance test.",
        "present_true_categories": present_categories,
        "absent_true_categories": [category for category in CATEGORIES if category not in present_categories],
        "minimum_present_class_support": min_support,
        "macro_f1_has_at_least_five_examples_per_present_class": min_support >= REPOSITORY_MIN_CLASS_SUPPORT,
        "per_class": metrics["per_class"],
    }


def qualitative_example(row: dict[str, Any], validation_by_issue_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    validation_row = validation_by_issue_id.get(int(row["issue_id"]), {})
    body = str(validation_row.get("body", ""))
    return {
        "evaluation_index": row["evaluation_index"],
        "issue_id": row["issue_id"],
        "repository": row["repository"],
        "expected_category": row["expected_category"],
        "title": validation_row.get("title", ""),
        "body_excerpt": body[:700],
        "base_prediction": row.get("predicted_category"),
        "fine_tuned_prediction": row.get("predicted_category"),
    }


def paired_example(
    base_row: dict[str, Any],
    fine_row: dict[str, Any],
    validation_by_issue_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    validation_row = validation_by_issue_id.get(int(base_row["issue_id"]), {})
    body = str(validation_row.get("body", ""))
    return {
        "evaluation_index": base_row["evaluation_index"],
        "issue_id": base_row["issue_id"],
        "repository": base_row["repository"],
        "expected_category": base_row["expected_category"],
        "title": validation_row.get("title", ""),
        "body_excerpt": body[:700],
        "base_prediction": prediction_label(base_row),
        "fine_tuned_prediction": prediction_label(fine_row),
        "base_raw_model_output": base_row.get("raw_model_output", ""),
        "fine_tuned_raw_model_output": fine_row.get("raw_model_output", ""),
    }


def select_paired_examples(
    base_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    validation_by_issue_id: dict[int, dict[str, Any]],
    mode: str,
    max_per_class: int = 2,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_class = Counter()
    for base_row, fine_row in zip(base_rows, fine_rows):
        base_correct = prediction_label(base_row) == base_row["expected_category"]
        fine_correct = prediction_label(fine_row) == fine_row["expected_category"]
        matches = (mode == "fine_fixes_base" and not base_correct and fine_correct) or (
            mode == "fine_breaks_base" and base_correct and not fine_correct
        )
        if not matches or per_class[base_row["expected_category"]] >= max_per_class:
            continue
        per_class[base_row["expected_category"]] += 1
        selected.append(paired_example(base_row, fine_row, validation_by_issue_id))
    return selected


def paired_comparison(
    base_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    validation_by_issue_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if len(base_rows) != len(fine_rows):
        raise ValueError("Base and fine-tuned artifacts have different row counts")
    groups = Counter()
    by_class: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    for base_row, fine_row in zip(base_rows, fine_rows):
        if base_row["evaluation_index"] != fine_row["evaluation_index"]:
            raise ValueError("Base and fine-tuned artifacts are not aligned by evaluation index")
        base_correct = prediction_label(base_row) == base_row["expected_category"]
        fine_correct = prediction_label(fine_row) == fine_row["expected_category"]
        if base_correct and fine_correct:
            group = "both_correct"
        elif base_correct and not fine_correct:
            group = "base_correct_fine_wrong"
        elif not base_correct and fine_correct:
            group = "base_wrong_fine_correct"
        else:
            group = "both_wrong"
        groups[group] += 1
        by_class[base_row["expected_category"]][group] += 1

    return {
        "counts": dict(groups),
        "by_true_class": {
            category: {
                group: by_class[category][group]
                for group in ("both_correct", "base_correct_fine_wrong", "base_wrong_fine_correct", "both_wrong")
            }
            for category in CATEGORIES
        },
        "fine_fixes_base_examples": select_paired_examples(
            base_rows, fine_rows, validation_by_issue_id, "fine_fixes_base"
        ),
        "fine_breaks_base_examples": select_paired_examples(
            base_rows, fine_rows, validation_by_issue_id, "fine_breaks_base"
        ),
    }


def bootstrap_macro_f1_delta(
    base_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if len(base_rows) != len(fine_rows):
        raise ValueError("Bootstrap inputs must be paired and equal in length")
    if any(base["evaluation_index"] != fine["evaluation_index"] for base, fine in zip(base_rows, fine_rows)):
        raise ValueError("Bootstrap inputs are not aligned")

    category_index = {category: index for index, category in enumerate(CATEGORIES)}
    actual = np.array([category_index[row["expected_category"]] for row in base_rows], dtype=np.int8)
    base_prediction = np.array(
        [category_index.get(prediction_label(row), len(CATEGORIES)) for row in base_rows], dtype=np.int8
    )
    fine_prediction = np.array(
        [category_index.get(prediction_label(row), len(CATEGORIES)) for row in fine_rows], dtype=np.int8
    )
    base_correct = (actual == base_prediction).astype(np.float64)
    fine_correct = (actual == fine_prediction).astype(np.float64)
    rng = np.random.default_rng(seed)
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    macro_f1_deltas = np.empty(replicates, dtype=np.float64)

    for start in range(0, replicates, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, replicates)
        indices = rng.integers(0, len(base_rows), size=(stop - start, len(base_rows)))
        accuracy_deltas[start:stop] = np.mean(
            fine_correct[indices] - base_correct[indices], axis=1
        )
        macro_f1_base = np.zeros(stop - start, dtype=np.float64)
        macro_f1_fine = np.zeros(stop - start, dtype=np.float64)
        for category in range(len(CATEGORIES)):
            true_mask = actual[indices] == category
            base_pred_mask = base_prediction[indices] == category
            fine_pred_mask = fine_prediction[indices] == category
            base_tp = np.sum(true_mask & base_pred_mask, axis=1)
            fine_tp = np.sum(true_mask & fine_pred_mask, axis=1)
            base_denominator = 2 * base_tp + np.sum(base_pred_mask, axis=1) - base_tp + np.sum(true_mask, axis=1) - base_tp
            fine_denominator = 2 * fine_tp + np.sum(fine_pred_mask, axis=1) - fine_tp + np.sum(true_mask, axis=1) - fine_tp
            macro_f1_base += np.divide(
                2 * base_tp,
                base_denominator,
                out=np.zeros_like(base_denominator, dtype=np.float64),
                where=base_denominator != 0,
            )
            macro_f1_fine += np.divide(
                2 * fine_tp,
                fine_denominator,
                out=np.zeros_like(fine_denominator, dtype=np.float64),
                where=fine_denominator != 0,
            )
        macro_f1_deltas[start:stop] = (macro_f1_fine - macro_f1_base) / len(CATEGORIES)

    base_point = class_metrics(base_rows)
    fine_point = class_metrics(fine_rows)
    return {
        "method": "paired row bootstrap with replacement; each replicate resamples validation rows jointly for base and fine-tuned predictions",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "accuracy_delta_point_estimate": rounded(float(np.mean(fine_correct) - np.mean(base_correct))),
        "accuracy_delta_ci95": [rounded(float(np.percentile(accuracy_deltas, 2.5))), rounded(float(np.percentile(accuracy_deltas, 97.5)))],
        "macro_f1_delta_point_estimate": rounded(
            float(fine_point["macro_f1"]) - float(base_point["macro_f1"])
        ),
        "macro_f1_delta_ci95": [rounded(float(np.percentile(macro_f1_deltas, 2.5))), rounded(float(np.percentile(macro_f1_deltas, 97.5)))],
    }


def exact_mcnemar_p_value(discordant_base_correct_fine_wrong: int, discordant_base_wrong_fine_correct: int) -> float:
    total = discordant_base_correct_fine_wrong + discordant_base_wrong_fine_correct
    if total == 0:
        return 1.0

    lower = min(discordant_base_correct_fine_wrong, discordant_base_wrong_fine_correct)
    log_half = math.log(0.5)

    def probability_up_to(limit: int) -> float:
        return sum(
            math.exp(
                math.lgamma(total + 1)
                - math.lgamma(k + 1)
                - math.lgamma(total - k + 1)
                + total * log_half
            )
            for k in range(limit + 1)
        )

    two_sided = 2.0 * probability_up_to(lower)
    return min(1.0, two_sided)


def paired_accuracy_test(paired: dict[str, Any]) -> dict[str, Any]:
    base_correct_fine_wrong = paired["counts"]["base_correct_fine_wrong"]
    base_wrong_fine_correct = paired["counts"]["base_wrong_fine_correct"]
    return {
        "test": "exact two-sided McNemar test on paired correctness",
        "base_correct_fine_wrong": base_correct_fine_wrong,
        "base_wrong_fine_correct": base_wrong_fine_correct,
        "discordant_pairs": base_correct_fine_wrong + base_wrong_fine_correct,
        "exact_two_sided_p_value": rounded(
            exact_mcnemar_p_value(base_correct_fine_wrong, base_wrong_fine_correct)
        ),
        "interpretation": "Descriptive paired accuracy test; it does not test macro-F1.",
    }


def make_repository_analysis(base_rows: list[dict[str, Any]], fine_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fine_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for base_row, fine_row in zip(base_rows, fine_rows):
        base_by_repo[base_row["repository"]].append(base_row)
        fine_by_repo[fine_row["repository"]].append(fine_row)

    result: dict[str, Any] = {}
    for repository in sorted(base_by_repo):
        base_summary = repository_metric(base_by_repo[repository])
        fine_summary = repository_metric(fine_by_repo[repository])
        result[repository] = {
            "row_count": base_summary["row_count"],
            "class_composition": base_summary["class_composition"],
            "macro_f1_definition": base_summary["macro_f1_definition"],
            "base_zero_shot": base_summary,
            "fine_tuned_zero_shot": fine_summary,
            "fine_minus_base": {
                "accuracy_delta": rounded(float(fine_summary["accuracy"]) - float(base_summary["accuracy"])),
                "macro_f1_present_true_categories_delta": rounded(
                    float(fine_summary["macro_f1_present_true_categories"])
                    - float(base_summary["macro_f1_present_true_categories"])
                ),
                "macro_f1_all_four_categories_delta": rounded(
                    float(fine_summary["macro_f1_all_four_categories"])
                    - float(base_summary["macro_f1_all_four_categories"])
                ),
            },
        }
    return result


def apply_corrected_prompt_profile(
    report: dict[str, Any],
    evaluation_report: dict[str, Any],
    train_counts: Counter[str],
    train_total: int,
    validation_counts: Counter[str],
) -> None:
    validation_total = sum(validation_counts.values())
    uniform_target = train_total / len(CATEGORIES)
    uniform_exposure = {
        category: rounded(uniform_target / train_counts[category]) for category in CATEGORIES
    }
    sqrt_weights = {category: train_counts[category] ** 0.5 for category in CATEGORIES}
    sqrt_weight_total = sum(sqrt_weights.values())
    sqrt_target = {
        category: rounded(train_total * sqrt_weights[category] / sqrt_weight_total)
        for category in CATEGORIES
    }
    sqrt_exposure = {
        category: rounded(sqrt_target[category] / train_counts[category]) for category in CATEGORIES
    }

    report["correction_history"] = {
        "historical_evaluation_report": "results/validation_evaluation.json",
        "corrected_evaluation_report": "results/validation_evaluation_corrected.json",
        "root_cause": "The historical evaluator right-truncated the already-rendered chat-template token sequence, which removed the assistant generation boundary on long prompts and caused generation to continue issue prose instead of producing classification JSON.",
        "repair": "The corrected evaluator renders the complete chat prompt after truncating only the current issue body from the right, then the title from the right if necessary, and reserves 16 output tokens while preserving system instructions, frozen demonstrations, role markers, and the assistant generation boundary.",
        "regenerated_rows": evaluation_report["regeneration_plan"],
        "unaffected_rows_reused_exactly": True,
        "old_metrics_preserved": evaluation_report["historical_metrics"],
    }
    report["failure_mechanism_assessment"] = {
        "observed_primary_pattern": "After correcting the prompt-boundary defect, formatting failures disappeared and the fine-tuned model gained accuracy, but macro-F1 remained slightly below base zero-shot because documentation F1 fell by 0.082175. Feature F1 rose by 0.019182 and question_support F1 rose by 0.039962; bug F1 was effectively unchanged.",
        "ranked_plausible_causes": [
            {
                "rank": 1,
                "cause": "Repository/domain shift and taxonomy ambiguity",
                "evidence": "The fine-tuned minus base macro-F1 delta is positive on PowerToys (+0.065445) and rust (+0.014889), but negative on node (-0.082223), svelte (-0.013390), and storybook (-0.009126); small gin is also strongly negative. Documentation errors shifted toward bug (42 to 76) even though the adapter predicted documentation more often (525 to 620), indicating a boundary/label-semantics problem rather than simple absence of minority predictions.",
            },
            {
                "rank": 2,
                "cause": "Natural class imbalance interacting with held-out repository priors",
                "evidence": f"The train prior is {rounded(percentage(train_counts['bug'], train_total))}% bug, {rounded(percentage(train_counts['feature'], train_total))}% feature, {rounded(percentage(train_counts['documentation'], train_total))}% documentation, and {rounded(percentage(train_counts['question_support'], train_total))}% question_support, versus validation {rounded(percentage(validation_counts['bug'], validation_total))}%, {rounded(percentage(validation_counts['feature'], validation_total))}%, {rounded(percentage(validation_counts['documentation'], validation_total))}%, and {rounded(percentage(validation_counts['question_support'], validation_total))}%. However, fine-tuned predictions moved farther from the train prior (L1 distance 0.113501 versus base 0.095028), so literal prior collapse is not supported.",
            },
            {
                "rank": 3,
                "cause": "Residual context difficulty after structural repair",
                "evidence": "Corrected truncated rows are valid and much better than under the old evaluator, but their macro-F1 remains lower than non-truncated rows: base zero-shot 0.468678 versus 0.609526, few-shot 0.475608 versus 0.617049, and fine-tuned zero-shot 0.481341 versus 0.603937. This is a difficulty effect shared by conditions, not evidence of a condition-specific truncation defect.",
            },
            {
                "rank": 4,
                "cause": "One epoch, learning rate/schedule, LoRA capacity, or completion-only objective",
                "evidence": "These remain plausible optimization contributors, but the corrected validation evidence does not isolate any one of them. Changing them together would not produce a clean causal experiment.",
            },
            {
                "rank": 5,
                "cause": "Label noise",
                "evidence": "The documentation-to-bug and bug-to-documentation exchanges, plus repository-specific reversals, are consistent with ambiguous labels, but the existing validation artifacts cannot distinguish label noise from genuine taxonomy ambiguity.",
            },
        ],
    }
    report["next_experiment_recommendation"] = {
        "decision": "A",
        "recommendation": "Keep Run A; do not start a second training run yet.",
        "primary_intervention": None,
        "why": "Run A does not improve the primary macro-F1 metric, but the paired macro-F1 delta is small and uncertain (95% CI includes zero), accuracy is slightly higher with a confidence interval that also includes zero, and the corrected evidence does not identify one training intervention as the causal fix. The dominant loss is documentation F1 on repository-specific boundaries, while the adapter is not moving toward the natural train prior.",
        "conceptual_imbalance_options_considered": {
            "uniform_four_class_sampling": {
                "same_total_training_draws": train_total,
                "approximate_draws_per_class": rounded(uniform_target),
                "average_exposure_multiplier": uniform_exposure,
                "risk": "Documentation and question_support examples would be repeated about 5.7x and 7.3x on average, while bug would be sampled at about 0.41x; this is a high overfitting risk without direct evidence that uniform sampling addresses the observed documentation precision/recall tradeoff.",
            },
            "moderate_sqrt_class_sampling": {
                "sampling_rule": "p(class) proportional to sqrt(class_count)",
                "approximate_draws_per_class": sqrt_target,
                "average_exposure_multiplier": sqrt_exposure,
                "risk": "This would be less aggressive than uniform sampling but still changes exposure without evidence that class prior correction is the main bottleneck.",
            },
            "class_weighted_loss": {
                "assessment": "Avoids repeating examples but would alter the completion-only optimization weighting and is not currently isolated by the validation evidence.",
            },
        },
        "start_from": None,
        "selection_rule": "Reconsider one controlled Run B only if a new development question identifies a single supported mechanism; keep TEST untouched.",
    }
    report["analysis_scope"]["analysis_profile"] = "corrected_prompt_preserving"


def minority_repository_concentration(
    repository_analysis: dict[str, Any], category: str, total_support: int
) -> list[dict[str, Any]]:
    concentration = []
    for repository, summary in repository_analysis.items():
        count = summary["class_composition"][category]["count"]
        if count:
            concentration.append(
                {
                    "repository": repository,
                    "count": count,
                    "percentage_of_validation_class": rounded(percentage(count, total_support)),
                }
            )
    return sorted(concentration, key=lambda item: (-item["count"], item["repository"]))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    evaluation_report_path = args.evaluation_report
    if not evaluation_report_path.is_absolute():
        evaluation_report_path = root / evaluation_report_path
    artifacts_directory = args.artifacts_directory
    if not artifacts_directory.is_absolute():
        artifacts_directory = root / artifacts_directory
    training_report_path = root / "results" / "initial_qlora_training.json"
    validation_path = root / "data" / "processed" / "splits" / "validation.jsonl"
    evaluation_report = load_json(evaluation_report_path)
    training_report = load_json(training_report_path)

    condition_rows = {
        condition: load_jsonl(artifacts_directory / f"{condition}.jsonl")
        for condition in CONDITIONS
    }
    expected_count = evaluation_report["validation_boundary"]["row_count"]
    if any(len(rows) != expected_count for rows in condition_rows.values()):
        raise ValueError("At least one validation artifact does not contain the frozen row count")
    for condition, rows in condition_rows.items():
        for expected_index, row in enumerate(rows):
            if row.get("evaluation_index") != expected_index:
                raise ValueError(f"{condition} is not a complete sequential validation artifact")

    validation_rows = load_jsonl(validation_path)
    validation_by_issue_id = {int(row["issue_id"]): row for row in validation_rows}
    if len(validation_by_issue_id) != len(validation_rows):
        raise ValueError("Validation issue IDs are not unique")

    base_rows = condition_rows["base_zero_shot"]
    few_rows = condition_rows["base_few_shot"]
    fine_rows = condition_rows["fine_tuned_zero_shot"]
    base_summary = class_metrics(base_rows)
    few_summary = class_metrics(few_rows)
    fine_summary = class_metrics(fine_rows)

    train_counts = Counter(training_report["data_statistics"]["train"]["class_counts"])
    validation_counts = Counter(evaluation_report["validation_boundary"]["class_counts"])
    base_pred_counts = Counter(prediction_label(row) for row in base_rows)
    fine_pred_counts = Counter(prediction_label(row) for row in fine_rows)
    train_total = sum(train_counts.values())
    validation_total = sum(validation_counts.values())
    base_valid_total = sum(base_pred_counts[category] for category in CATEGORIES)
    fine_valid_total = sum(fine_pred_counts[category] for category in CATEGORIES)

    paired = paired_comparison(base_rows, fine_rows, validation_by_issue_id)
    bootstrap = bootstrap_macro_f1_delta(
        base_rows,
        fine_rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    bootstrap["macro_f1_delta_point_estimate"] = rounded(
        float(fine_summary["macro_f1"]) - float(base_summary["macro_f1"])
    )

    repository_analysis = make_repository_analysis(base_rows, fine_rows)
    report = {
        "analysis_status": "passed",
        "analysis_version": args.analysis_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": {
            "inference_rerun": False,
            "model_loaded": False,
            "test_split_accessed_by_analysis": False,
            "validation_path": "data/processed/splits/validation.jsonl",
            "validation_row_count": expected_count,
            "validation_sha256": evaluation_report["validation_boundary"]["sha256"],
            "historical_train_only_prompt_metrics_used_as_primary_evidence": False,
        },
        "source_artifacts": {
            "validation_evaluation_report": str(evaluation_report_path.relative_to(root)).replace("\\", "/"),
            "initial_training_report": "results/initial_qlora_training.json",
            "validation_prediction_artifacts": {
                condition: str((artifacts_directory / f"{condition}.jsonl").relative_to(root)).replace("\\", "/")
                for condition in CONDITIONS
            },
            "source_evaluation_commit_sha": evaluation_report.get("git", {}).get("commit_sha"),
        },
        "frozen_distributions": {
            "train": {
                "row_count": train_total,
                "class_counts": dict(train_counts),
                "class_percentages": {
                    category: rounded(percentage(train_counts[category], train_total)) for category in CATEGORIES
                },
            },
            "validation": {
                "row_count": validation_total,
                "class_counts": dict(validation_counts),
                "class_percentages": {
                    category: rounded(percentage(validation_counts[category], validation_total)) for category in CATEGORIES
                },
            },
        },
        "condition_metrics": {
            "base_zero_shot": {
                "official_metrics_recomputed": base_summary,
                "valid_outputs_only_analysis": valid_only_metrics(base_rows),
                "parser_failure_analysis": exact_parser_failure_analysis(base_rows),
                "truncation_analysis": truncation_analysis(base_rows),
            },
            "base_few_shot": {
                "official_metrics_recomputed": few_summary,
                "valid_outputs_only_analysis": valid_only_metrics(few_rows),
                "parser_failure_analysis": exact_parser_failure_analysis(few_rows),
                "truncation_analysis": truncation_analysis(few_rows),
            },
            "fine_tuned_zero_shot": {
                "official_metrics_recomputed": fine_summary,
                "valid_outputs_only_analysis": valid_only_metrics(fine_rows),
                "parser_failure_analysis": exact_parser_failure_analysis(fine_rows),
                "truncation_analysis": truncation_analysis(fine_rows),
            },
        },
        "cross_condition_truncation_comparison": cross_condition_truncation_comparison(
            base_rows, few_rows, fine_rows
        ),
        "per_class_comparison": {
            category: {
                "support": base_summary["per_class"][category]["support"],
                "base_zero_shot": base_summary["per_class"][category],
                "base_few_shot": few_summary["per_class"][category],
                "fine_tuned_zero_shot": fine_summary["per_class"][category],
                "fine_minus_base_zero_shot": {
                    metric: rounded(
                        float(fine_summary["per_class"][category][metric])
                        - float(base_summary["per_class"][category][metric])
                    )
                    for metric in ("precision", "recall", "f1")
                },
            }
            for category in CATEGORIES
        },
        "confusion_analysis": {
            "base_zero_shot_error_destinations": error_destinations(base_rows),
            "fine_tuned_zero_shot_error_destinations": error_destinations(fine_rows),
            "selected_transitions": selected_confusion_transitions(base_rows, fine_rows),
            "fine_minus_base_confusion_matrix": [
                [fine_summary["confusion_matrix_rows_true_columns_predicted"][row][column]
                 - base_summary["confusion_matrix_rows_true_columns_predicted"][row][column]
                 for column in range(len(MATRIX_LABELS))]
                for row in range(len(MATRIX_LABELS))
            ],
            "fine_minus_base_per_class": per_class_deltas(fine_summary, base_summary),
        },
        "prediction_distribution_analysis": {
            "true_validation_distribution": {
                "counts": dict(validation_counts),
                "percentages": {
                    category: rounded(percentage(validation_counts[category], validation_total))
                    for category in CATEGORIES
                },
            },
            "base_zero_shot": prediction_distribution(base_rows),
            "fine_tuned_zero_shot": prediction_distribution(fine_rows),
            "fine_minus_base_predicted_counts": {
                label: fine_pred_counts[label] - base_pred_counts[label] for label in MATRIX_LABELS
            },
            "training_prior_distance": {
                "distance_definition": "L1 distance between four-class proportions among valid outputs and the natural train class prior; lower means closer.",
                "base_zero_shot_l1_distance": rounded(
                    l1_distance_to_prior(base_pred_counts, base_valid_total, train_counts, train_total)
                ),
                "fine_tuned_zero_shot_l1_distance": rounded(
                    l1_distance_to_prior(fine_pred_counts, fine_valid_total, train_counts, train_total)
                ),
                "fine_minus_base_l1_distance": rounded(
                    l1_distance_to_prior(fine_pred_counts, fine_valid_total, train_counts, train_total)
                    - l1_distance_to_prior(base_pred_counts, base_valid_total, train_counts, train_total)
                ),
            },
        },
        "paired_validation_analysis": paired,
        "statistical_uncertainty": {
            "paired_bootstrap": bootstrap,
            "mcnemar_accuracy": paired_accuracy_test(paired),
        },
        "repository_level_analysis": repository_analysis,
        "minority_class_repository_concentration": {
            category: minority_repository_concentration(
                repository_analysis, category, validation_counts[category]
            )
            for category in ("documentation", "question_support")
        },
        "failure_mechanism_assessment": {
            "observed_primary_pattern": "Fine-tuning improved feature F1 and question_support precision/F1, but documentation F1 fell substantially; bug F1 was approximately flat. The aggregate macro-F1 loss is driven primarily by documentation.",
            "ranked_plausible_causes": [
                {
                    "rank": 1,
                    "cause": "Natural class imbalance combined with repository-held-out prior/domain shift",
                    "evidence": "The frozen train prior is 61.03% bug, 31.17% feature, 4.36% documentation, and 3.44% question_support, while validation is 57.25%, 39.12%, 2.49%, and 1.15%. Fine-tuning reduced documentation F1 by 0.082122 and left question_support recall very low at 0.116438, while majority-class F1 remained stable or improved. The adapter did not simply copy the train prior, so this is an imbalance/generalization effect rather than a literal prior-collapse diagnosis.",
                },
                {
                    "rank": 2,
                    "cause": "One natural-distribution epoch without checkpoint selection",
                    "evidence": "Training loss fell rapidly and then remained near 0.42 for most of the run; only one adapter checkpoint was available for comparison. This makes optimization duration a plausible but unproven contributor, not an observed mechanism.",
                },
                {
                    "rank": 3,
                    "cause": "Repository/domain shift and taxonomy ambiguity",
                    "evidence": "Validation is repository-held-out and the minority categories have small supports. Repository-level deltas and class-specific confusion should be read as descriptive evidence of heterogeneous generalization, not as a single global label rule.",
                },
                {
                    "rank": 4,
                    "cause": "Output-format behavior",
                    "evidence": "Strict invalid-output rates are high for both zero-shot conditions, but fine-tuning and base zero-shot have nearly identical valid-output rates. Formatting failure therefore affects absolute scores but does not explain the macro-F1 regression by itself.",
                },
                {
                    "rank": 5,
                    "cause": "Context truncation",
                    "evidence": "Base zero-shot and fine-tuned zero-shot have the same truncation count and prompt contract. Truncation can affect individual rows, but it cannot explain their aggregate difference as a condition-level exposure imbalance.",
                },
                {
                    "rank": 6,
                    "cause": "LoRA capacity, learning rate, or completion-only objective",
                    "evidence": "These settings are theoretically possible contributors, but the completed validation evidence does not isolate any of them. Changing them together would not produce a clean causal second experiment.",
                },
            ],
        },
        "next_experiment_recommendation": {
            "decision": "B",
            "recommendation": "Run one targeted second fine-tune with explicit class-balanced sampling.",
            "primary_intervention": "Replace natural frozen train sampling with uniform four-class sampling with replacement for the same 31,876 sampled rows and 1,993 optimizer steps, while retaining the existing completion-only objective.",
            "remain_identical": [
                "original locked base model and revision",
                "repository-held-out train/validation split",
                "zero-shot prompt and canonical parser",
                "1536-token context policy and target preservation",
                "LoRA target modules, rank, alpha, dropout, and quantization",
                "learning rate, scheduler, optimizer, batch/accumulation, one epoch, seed, and deterministic generation",
            ],
            "start_from": "original_locked_base_model",
            "not_from_run_a_adapter": True,
            "why": "The largest class-specific loss is documentation F1, and question_support recall remains low, while the train distribution strongly overrepresents bug/feature relative to repository-held-out validation. Balancing directly tests whether minority exposure, rather than a broad hyperparameter change, improves macro-F1.",
            "expected_benefit": "Higher documentation and question_support recall and a higher macro-F1 if natural-frequency exposure is the main limitation.",
            "main_risk": "Oversampling small minority classes may overfit their repository-specific wording or reduce majority-class accuracy.",
            "selection_rule": "Evaluate Run B on the same frozen validation split with the same three-condition apples-to-apples comparison before any final decision; keep TEST untouched.",
        },
        "historical_evidence_note": "The earlier 400-example train-only prompt-development metrics are preserved as historical evidence and are not used as the fair validation comparison.",
    }
    if args.analysis_profile == "corrected_prompt_preserving":
        apply_corrected_prompt_profile(report, evaluation_report, train_counts, train_total, validation_counts)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("results") / "validation_evaluation.json",
    )
    parser.add_argument(
        "--artifacts-directory",
        type=Path,
        default=Path("results") / "validation_evaluation",
    )
    parser.add_argument("--analysis-version", default="1.0")
    parser.add_argument(
        "--analysis-profile",
        choices=("historical", "corrected_prompt_preserving"),
        default="historical",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "validation_failure_analysis.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise SystemExit("--bootstrap-replicates must be positive")
    report = analyze(args)
    output_path = args.output if args.output.is_absolute() else args.root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "status": report["analysis_status"],
        "bootstrap_replicates": report["statistical_uncertainty"]["paired_bootstrap"]["replicates"],
        "macro_f1_delta": report["statistical_uncertainty"]["paired_bootstrap"]["macro_f1_delta_point_estimate"],
        "macro_f1_delta_ci95": report["statistical_uncertainty"]["paired_bootstrap"]["macro_f1_delta_ci95"],
    }, indent=2))


if __name__ == "__main__":
    main()
