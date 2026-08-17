"""Compute final TEST metrics, paired comparisons, repository analysis, and uncertainty."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binomtest

from validation_evaluation.config import TARGET_CATEGORIES
from validation_evaluation.metrics import calculate_metrics

from .config import (
    BOOTSTRAP_REPLICATES,
    LIMITED_CLASS_SUPPORT_THRESHOLD,
    STABLE_CLASS_SUPPORT_THRESHOLD,
    STATISTICAL_SEED,
)


def metrics_by_truncation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the same metric bundle for truncated and non-truncated rows."""
    grouped = {
        "truncated": [record for record in records if record["input_truncated"]],
        "non_truncated": [record for record in records if not record["input_truncated"]],
    }
    result = {}
    for name, subset in grouped.items():
        result[name] = calculate_metrics(subset) if subset else {"example_count": 0, "unavailable": True}
    return result


def _support_interpretation(support: int) -> str:
    """Label class support with a fixed descriptive stability guideline."""
    if support >= STABLE_CLASS_SUPPORT_THRESHOLD:
        return "stable_guidance"
    if support >= LIMITED_CLASS_SUPPORT_THRESHOLD:
        return "limited_guidance"
    return "too_little_support"


def repository_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report overall and per-class metrics for each TEST repository."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["repository"])].append(record)

    result = {}
    for repository in sorted(grouped):
        subset = grouped[repository]
        metrics = calculate_metrics(subset)
        class_support = {
            category: metrics["per_class"][category]["support"] for category in TARGET_CATEGORIES
        }
        result[repository] = {
            "row_count": len(subset),
            "class_support": class_support,
            "class_support_interpretation": {
                category: _support_interpretation(support)
                for category, support in class_support.items()
            },
            "metrics": metrics,
        }
    return result


def _is_correct(record: dict[str, Any]) -> bool:
    """Return whether one row has a valid prediction matching its frozen target."""
    return bool(record["schema_valid"]) and record["predicted_category"] == record["expected_category"]


def paired_correctness(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], name_a: str, name_b: str) -> dict[str, Any]:
    """Count paired correctness outcomes on identical TEST rows."""
    if len(records_a) != len(records_b):
        raise ValueError("Paired artifacts have different row counts")
    for index, (record_a, record_b) in enumerate(zip(records_a, records_b)):
        if record_a["evaluation_index"] != record_b["evaluation_index"]:
            raise ValueError(f"Paired artifacts are misaligned at evaluation index {index}")
    a_correct_b_wrong = sum(_is_correct(a) and not _is_correct(b) for a, b in zip(records_a, records_b))
    a_wrong_b_correct = sum(not _is_correct(a) and _is_correct(b) for a, b in zip(records_a, records_b))
    both_correct = sum(_is_correct(a) and _is_correct(b) for a, b in zip(records_a, records_b))
    both_wrong = sum(not _is_correct(a) and not _is_correct(b) for a, b in zip(records_a, records_b))
    return {
        "condition_a": name_a,
        "condition_b": name_b,
        "total_examples": len(records_a),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
    }


def _macro_f1(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Compute four-class macro-F1 while treating invalid predictions as non-category output."""
    scores = []
    for category_index in range(len(TARGET_CATEGORIES)):
        true_positive = np.sum((true_values == category_index) & (predicted_values == category_index))
        false_positive = np.sum((true_values != category_index) & (predicted_values == category_index))
        false_negative = np.sum((true_values == category_index) & (predicted_values != category_index))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def paired_bootstrap(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """Use fixed-seed paired bootstrap resampling for accuracy and macro-F1 deltas."""
    if len(records_a) != len(records_b):
        raise ValueError("Bootstrap artifacts have different row counts")
    n = len(records_a)
    category_to_index = {category: index for index, category in enumerate(TARGET_CATEGORIES)}
    true_values = np.array([category_to_index[record["expected_category"]] for record in records_a], dtype=np.int8)
    predicted_a = np.array(
        [category_to_index.get(record["predicted_category"], -1) for record in records_a], dtype=np.int8
    )
    predicted_b = np.array(
        [category_to_index.get(record["predicted_category"], -1) for record in records_b], dtype=np.int8
    )
    correct_a = predicted_a == true_values
    correct_b = predicted_b == true_values
    observed_accuracy_delta = float(np.mean(correct_a) - np.mean(correct_b))
    observed_macro_f1_delta = _macro_f1(true_values, predicted_a) - _macro_f1(true_values, predicted_b)

    rng = np.random.default_rng(STATISTICAL_SEED)
    accuracy_deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    macro_f1_deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample_indices = rng.integers(0, n, size=n)
        sampled_true = true_values[sample_indices]
        sampled_a = predicted_a[sample_indices]
        sampled_b = predicted_b[sample_indices]
        accuracy_deltas[replicate] = np.mean(sampled_a == sampled_true) - np.mean(sampled_b == sampled_true)
        macro_f1_deltas[replicate] = _macro_f1(sampled_true, sampled_a) - _macro_f1(sampled_true, sampled_b)

    return {
        "condition_a": name_a,
        "condition_b": name_b,
        "method": "paired bootstrap over identical TEST row indices",
        "seed": STATISTICAL_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "accuracy_delta": round(observed_accuracy_delta, 6),
        "accuracy_delta_ci_95": [
            round(float(np.quantile(accuracy_deltas, 0.025)), 6),
            round(float(np.quantile(accuracy_deltas, 0.975)), 6),
        ],
        "macro_f1_delta": round(observed_macro_f1_delta, 6),
        "macro_f1_delta_ci_95": [
            round(float(np.quantile(macro_f1_deltas, 0.025)), 6),
            round(float(np.quantile(macro_f1_deltas, 0.975)), 6),
        ],
    }


def exact_mcnemar(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], name_a: str, name_b: str) -> dict[str, Any]:
    """Run the exact two-sided paired correctness test for one condition pair."""
    paired = paired_correctness(records_a, records_b, name_a, name_b)
    a_correct_b_wrong = paired["a_correct_b_wrong"]
    a_wrong_b_correct = paired["a_wrong_b_correct"]
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_correct_b_wrong, a_wrong_b_correct), n=discordant, p=0.5).pvalue
    )
    return {
        "test": "exact two-sided McNemar via binomial test",
        "condition_a": name_a,
        "condition_b": name_b,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": discordant,
        "p_value": round(p_value, 8),
    }


def class_comparison(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], name_a: str, name_b: str) -> dict[str, Any]:
    """Report per-class correctness gains and losses for a paired comparison."""
    result = {}
    for category in TARGET_CATEGORIES:
        subset_a = [record for record in records_a if record["expected_category"] == category]
        subset_b = [record for record in records_b if record["expected_category"] == category]
        a_correct_b_wrong = sum(_is_correct(a) and not _is_correct(b) for a, b in zip(subset_a, subset_b))
        a_wrong_b_correct = sum(not _is_correct(a) and _is_correct(b) for a, b in zip(subset_a, subset_b))
        result[category] = {
            "support": len(subset_a),
            "a_correct_b_wrong": a_correct_b_wrong,
            "a_wrong_b_correct": a_wrong_b_correct,
            "net_correctness_delta": a_correct_b_wrong - a_wrong_b_correct,
        }
    return {
        "condition_a": name_a,
        "condition_b": name_b,
        "by_true_class": result,
    }
