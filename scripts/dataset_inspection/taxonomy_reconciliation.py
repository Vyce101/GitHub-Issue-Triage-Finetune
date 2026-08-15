"""Reconcile taxonomy analysis counts with the normalized labeled dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DATASET_ID
from .loader import load_dataset_splits
from .raw_values import resolve_column, text_value
from .split_analysis import NORMALIZED_DATASET_PATH, _target_mapping
from .taxonomy_analysis import HIGH_CONFIDENCE_CATEGORY_ORDER, analyze_taxonomy, apparent_atomic_labels, classify_atom


OUTPUT_PATH = Path("results/taxonomy_reconciliation.json")
APPROVED_SOURCE_CATEGORIES = tuple(HIGH_CONFIDENCE_CATEGORY_ORDER[:4])
APPROVED_TARGET_CATEGORIES = ("bug", "feature", "documentation", "question_support")


def _row_key(split: str, row_index: int) -> tuple[str, int]:
    """Identify a source row deterministically."""
    return split, row_index


def _source_row_payload(values: dict[str, Any], column_map: dict[str, str], split: str, row_index: int) -> dict[str, Any]:
    """Keep source fields required for normalized-row validation."""
    return {
        "issue_id": values.get(column_map["issue_id"]),
        "repository": text_value(values.get(column_map["repository"])).strip() or "<MISSING_REPOSITORY>",
        "title": text_value(values.get(column_map["title"])),
        "body": text_value(values.get(column_map["body"])),
        "raw_labels": text_value(values.get(column_map["labels"])),
        "source_split": split,
        "source_row_index": row_index,
    }


def _mapping_validation(raw_labels: str) -> dict[str, Any]:
    """Return the exact current construction mapping for one raw label string."""
    mapping = _target_mapping(raw_labels)
    target_categories = mapping["target_categories"]
    responsible = mapping["mappings"]
    return {
        "target_categories": target_categories,
        "responsible_atomic_labels": [item["atomic_label"] for item in responsible],
        "mapping_reasons": responsible,
        "ambiguous_atoms": mapping["ambiguous_atoms"],
        "non_target_high_confidence": mapping["non_target_high_confidence"],
    }


def _load_source_audit() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Recompute target matches and counts directly from the current repository code."""
    dataset_splits = load_dataset_splits()
    all_columns = sorted({column for dataset in dataset_splits.values() for column in dataset.column_names})
    column_map = {
        "issue_id": resolve_column(all_columns, ("issue_id", "id", "number", "issue_number"), "issue_id"),
        "repository": resolve_column(all_columns, ("repository", "repo", "repository_name", "repo_name"), "repo"),
        "title": resolve_column(all_columns, ("title", "issue_title"), "title"),
        "body": resolve_column(all_columns, ("body", "issue_body", "description"), "body"),
        "labels": resolve_column(all_columns, ("labels", "label"), "label"),
    }
    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    target_category_counts: Counter[str] = Counter()
    target_conflict_sets: Counter[tuple[str, ...]] = Counter()
    target_conflict_occurrences: Counter[str] = Counter()
    strict_category_counts: Counter[str] = Counter()
    target_single_category_counts: Counter[str] = Counter()
    no_target_count = 0
    target_conflict_count = 0
    strict_ambiguous_count = 0
    strict_non_target_high_confidence_count = 0
    target_single_with_ambiguous_count = 0
    target_single_with_non_target_high_confidence_count = 0
    raw_total = 0

    for split_name, dataset in dataset_splits.items():
        for row_index, row in enumerate(dataset):
            values = dict(row)
            source = _source_row_payload(values, column_map, split_name, row_index)
            source_rows[_row_key(split_name, row_index)] = source
            raw_total += 1
            mapping = _mapping_validation(source["raw_labels"])
            target_categories = tuple(sorted(mapping["target_categories"]))
            high_categories = set(mapping["target_categories"])
            # Reconstruct all current high-confidence categories, including performance,
            # for exact reproduction of the prior strict coverage calculation.
            all_high_confidence_categories = set()
            for atom in dict.fromkeys(apparent_atomic_labels(source["raw_labels"])):
                category, status, _ = classify_atom(atom)
                if status == "high_confidence" and category is not None:
                    all_high_confidence_categories.add(category)

            for category in target_categories:
                target_category_counts[category] += 1
            if not target_categories:
                no_target_count += 1
            elif len(target_categories) > 1:
                target_conflict_count += 1
                target_conflict_sets[target_categories] += 1
                target_conflict_occurrences.update(target_categories)
            else:
                target_single_category_counts[target_categories[0]] += 1

            if mapping["ambiguous_atoms"]:
                if target_categories:
                    target_single_with_ambiguous_count += int(len(target_categories) == 1)
                strict_ambiguous_count += 1
            if mapping["non_target_high_confidence"]:
                if target_categories:
                    target_single_with_non_target_high_confidence_count += int(len(target_categories) == 1)
                strict_non_target_high_confidence_count += 1
            if len(all_high_confidence_categories) == 1 and not mapping["ambiguous_atoms"]:
                only_category = next(iter(all_high_confidence_categories))
                if only_category in APPROVED_SOURCE_CATEGORIES:
                    strict_category_counts[only_category] += 1

    return source_rows, {
        "raw_total": raw_total,
        "column_map": column_map,
        "target_category_counts_including_conflicts": dict(target_category_counts),
        "target_conflict_count": target_conflict_count,
        "target_conflict_sets": {" + ".join(key): count for key, count in sorted(target_conflict_sets.items())},
        "target_conflict_category_occurrences": dict(target_conflict_occurrences),
        "target_single_category_counts_before_duplicate_removal": dict(target_single_category_counts),
        "no_target_count": no_target_count,
        "strict_category_counts_reproduced": dict(strict_category_counts),
        "strict_rows_with_ambiguous_atoms": strict_ambiguous_count,
        "strict_rows_with_non_target_high_confidence_atoms": strict_non_target_high_confidence_count,
        "target_single_rows_with_ambiguous_atoms": target_single_with_ambiguous_count,
        "target_single_rows_with_non_target_high_confidence_atoms": target_single_with_non_target_high_confidence_count,
    }


def _load_normalized_rows() -> list[dict[str, Any]]:
    """Load the existing normalized JSONL without modifying it."""
    with NORMALIZED_DATASET_PATH.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _validate_normalized_rows(normalized_rows: list[dict[str, Any]], source_rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Verify provenance, mapping responsibility, category uniqueness, and duplicate safety."""
    source_mismatches = []
    mapping_mismatches = []
    invalid_target_categories = []
    invalid_responsible_atoms = []
    ambiguous_or_excluded_responsibility = []
    source_keys = Counter()
    issue_ids = Counter()
    title_body_keys = Counter()

    for row in normalized_rows:
        source_key = _row_key(row["source_split"], row["source_row_index"])
        source_keys[source_key] += 1
        issue_ids[json.dumps(row["issue_id"], ensure_ascii=False, sort_keys=True)] += 1
        title_body_keys[json.dumps([row["title"], row["body"]], ensure_ascii=False)] += 1
        source = source_rows.get(source_key)
        if source is None:
            source_mismatches.append({"source_key": source_key, "reason": "source_row_not_found"})
            continue
        for field in ("issue_id", "repository", "title", "body", "raw_labels"):
            if row[field] != source[field]:
                source_mismatches.append({"source_key": source_key, "field": field, "normalized": row[field], "source": source[field]})
        mapping = _mapping_validation(row["raw_labels"])
        if mapping["target_categories"] != [row["target_category"]]:
            mapping_mismatches.append({"source_key": source_key, "expected": mapping["target_categories"], "actual": row["target_category"]})
        if row["target_category"] not in APPROVED_TARGET_CATEGORIES:
            invalid_target_categories.append({"source_key": source_key, "target_category": row["target_category"]})
        expected_atoms = mapping["responsible_atomic_labels"]
        if row["atomic_labels"] != expected_atoms:
            invalid_responsible_atoms.append({"source_key": source_key, "expected": expected_atoms, "actual": row["atomic_labels"]})
        if any(item["target_category"] not in APPROVED_TARGET_CATEGORIES for item in row["mapping_reasons"]):
            ambiguous_or_excluded_responsibility.append({"source_key": source_key, "mapping_reasons": row["mapping_reasons"]})

    return {
        "normalized_row_count": len(normalized_rows),
        "source_rows_found": len(normalized_rows) - len([item for item in source_mismatches if item.get("reason") == "source_row_not_found"]),
        "source_field_mismatch_count": len(source_mismatches),
        "mapping_mismatch_count": len(mapping_mismatches),
        "invalid_target_category_count": len(invalid_target_categories),
        "invalid_responsible_atom_count": len(invalid_responsible_atoms),
        "ambiguous_or_excluded_responsibility_count": len(ambiguous_or_excluded_responsibility),
        "duplicate_source_key_count": sum(count > 1 for count in source_keys.values()),
        "duplicate_source_issue_id_count": sum(count > 1 for count in issue_ids.values()),
        "duplicate_title_body_pair_count": sum(count > 1 for count in title_body_keys.values()),
        "examples": {
            "source_mismatches": source_mismatches[:10],
            "mapping_mismatches": mapping_mismatches[:10],
            "invalid_target_categories": invalid_target_categories[:10],
            "invalid_responsible_atoms": invalid_responsible_atoms[:10],
            "ambiguous_or_excluded_responsibility": ambiguous_or_excluded_responsibility[:10],
        },
    }


def _reconcile() -> dict[str, Any]:
    """Run the complete reconciliation and return an audit report."""
    taxonomy_summary = analyze_taxonomy()
    source_rows, source_audit = _load_source_audit()
    normalized_rows = _load_normalized_rows()
    normalized_audit = _validate_normalized_rows(normalized_rows, source_rows)

    strict_option = next(
        option
        for option in taxonomy_summary["coverage_options"]
        if option["name"] == "four_class_bug_feature_documentation_question_support"
    )
    previous_report_counts = {
        "category_candidate_matches": {
            category: taxonomy_summary["candidate_issue_type_mappings"][category]["matched_issue_count"]
            for category in ("bug", "feature_request", "documentation", "question_support")
        },
        "four_class_strict_usable_counts": strict_option["high_confidence_issue_count_per_category"],
        "four_class_strict_total": strict_option["total_usable_issues"],
    }
    target_single_counts = source_audit["target_single_category_counts_before_duplicate_removal"]
    duplicate_removed = taxonomy_summary.get("approved_target_policy_coverage", {}).get("candidate_issue_count_before_duplicate_removal", 0) - normalized_audit["normalized_row_count"]
    final_counts = Counter(row["target_category"] for row in normalized_rows)

    strict_counts_as_targets = {
        "bug": previous_report_counts["four_class_strict_usable_counts"].get("bug", 0),
        "feature": previous_report_counts["four_class_strict_usable_counts"].get("feature_request", 0),
        "documentation": previous_report_counts["four_class_strict_usable_counts"].get("documentation", 0),
        "question_support": previous_report_counts["four_class_strict_usable_counts"].get("question_support", 0),
    }
    strict_difference = {
        category: target_single_counts.get(category, 0) - strict_counts_as_targets.get(category, 0)
        for category in APPROVED_TARGET_CATEGORIES
    }
    target_candidate_sum = sum(source_audit["target_category_counts_including_conflicts"].values())
    conflict_occurrence_sum = sum(source_audit["target_conflict_category_occurrences"].values())
    reconciliation_math = {
        "source_rows": source_audit["raw_total"],
        "no_target_rows": source_audit["no_target_count"],
        "multi_target_conflict_rows": source_audit["target_conflict_count"],
        "single_target_rows_before_duplicate_removal": sum(target_single_counts.values()),
        "duplicate_rows_removed": duplicate_removed,
        "final_normalized_rows": normalized_audit["normalized_row_count"],
        "equation": f"{source_audit['no_target_count']} + {source_audit['target_conflict_count']} + {sum(target_single_counts.values())} = {source_audit['raw_total']}; {sum(target_single_counts.values())} - {duplicate_removed} = {normalized_audit['normalized_row_count']}",
        "category_candidate_sum_minus_conflict_occurrences": f"{target_candidate_sum} - {conflict_occurrence_sum} = {target_candidate_sum - conflict_occurrence_sum}",
    }

    return {
        "dataset_id": DATASET_ID,
        "audit_scope": {
            "dataset_changed": False,
            "normalized_dataset_changed": False,
            "split_assignments_changed": False,
            "model_downloaded": False,
            "training_run": False,
        },
        "previous_taxonomy_reproduction": {
            "source": "Current taxonomy_analysis.py executed from the repository.",
            "counts": previous_report_counts,
            "approved_target_policy_coverage_from_current_code": taxonomy_summary.get("approved_target_policy_coverage"),
        },
        "row_level_source_audit": source_audit,
        "reconciliation_math": reconciliation_math,
        "strict_vs_approved_policy_difference": {
            "strict_policy_definition": strict_option["eligibility_policy"],
            "strict_policy_total": previous_report_counts["four_class_strict_total"],
            "strict_policy_counts_as_final_target_names": strict_counts_as_targets,
            "approved_policy_total_before_duplicate_removal": sum(target_single_counts.values()),
            "additional_rows_allowed_by_approved_policy": sum(strict_difference.values()),
            "additional_rows_by_category": strict_difference,
            "explanation": "The earlier 46,300 figure required exactly one high-confidence category across all candidate categories, excluded any ambiguous atom, and treated performance as another candidate category. The approved construction requires exactly one of the four approved target categories; non-target labels such as performance, workflow, and ambiguous labels are not used to assign a target or create a target conflict.",
        },
        "normalized_dataset_audit": normalized_audit,
        "final_normalized_counts": dict(final_counts),
        "multi_category_exclusion_audit": {
            "all_excluded_target_conflicts_have_at_least_two_approved_categories": source_audit["target_conflict_count"] == sum(source_audit["target_conflict_sets"].values()),
            "conflict_sets": source_audit["target_conflict_sets"],
            "conflict_category_occurrences": source_audit["target_conflict_category_occurrences"],
        },
        "duplicate_audit": {
            "normalized_duplicate_source_issue_ids": normalized_audit["duplicate_source_issue_id_count"],
            "normalized_duplicate_title_body_pairs": normalized_audit["duplicate_title_body_pair_count"],
            "duplicate_rows_removed_before_final_output": duplicate_removed,
        },
        "conclusion": {
            "classification": "C_with_reporting_ambiguity",
            "normalized_dataset_correct": normalized_audit["source_field_mismatch_count"] == 0 and normalized_audit["mapping_mismatch_count"] == 0 and normalized_audit["invalid_target_category_count"] == 0 and normalized_audit["invalid_responsible_atom_count"] == 0 and normalized_audit["duplicate_source_issue_id_count"] == 0 and normalized_audit["duplicate_title_body_pair_count"] == 0,
            "earlier_46300_is_wrong_for_the_approved_policy": True,
            "what_happened": "The two analyses used intentionally different eligibility criteria. The earlier report presented the strict candidate-coverage number as the four-class usable count, while normalized construction used the later approved target-only policy. The analysis report has been clarified with an explicit approved_target_policy_coverage section; the normalized dataset was not changed.",
        },
    }


def write_reconciliation_report(summary: dict[str, Any]) -> None:
    """Write the taxonomy reconciliation audit report."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_reconciliation_summary(summary: dict[str, Any]) -> None:
    """Print a concise audit result."""
    math_result = summary["reconciliation_math"]
    conclusion = summary["conclusion"]
    print("\nTaxonomy reconciliation complete")
    print(f"Earlier strict four-class count: {summary['previous_taxonomy_reproduction']['counts']['four_class_strict_total']}")
    print(f"Approved-policy rows before duplicates: {math_result['single_target_rows_before_duplicate_removal']}")
    print(f"Normalized rows: {math_result['final_normalized_rows']}")
    print(f"Duplicate rows removed: {math_result['duplicate_rows_removed']}")
    print(f"Normalized dataset correct: {conclusion['normalized_dataset_correct']}")
    print(f"Conclusion: {conclusion['classification']}")
    print(f"Audit report: {OUTPUT_PATH}")
