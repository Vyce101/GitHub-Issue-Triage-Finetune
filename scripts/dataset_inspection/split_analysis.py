"""Construct the normalized label set and analyze deterministic split strategies."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DATASET_ID
from .loader import load_dataset_splits
from .raw_values import is_null, resolve_column, text_value
from .taxonomy_analysis import apparent_atomic_labels, classify_atom


NORMALIZED_DATASET_PATH = Path("data/processed/normalized_labeled_dataset.jsonl")
TARGET_CATEGORY_MAP = {
    "bug": "bug",
    "feature_request": "feature",
    "documentation": "documentation",
    "question_support": "question_support",
}
TARGET_CATEGORY_ORDER = tuple(TARGET_CATEGORY_MAP.values())
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REPO_SEARCH_SEEDS = tuple(range(64))
REPO_CANDIDATE_REPORT_LIMIT = 5
EXAMPLE_LIMIT = 10


def _stable_json(value: Any) -> str:
    """Serialize a value for deterministic hashing and duplicate detection."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    """Return a stable UTF-8 SHA-256 fingerprint."""
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _append_capped(values: list[Any], value: Any, limit: int = EXAMPLE_LIMIT) -> None:
    """Keep deterministic examples bounded."""
    if value not in values and len(values) < limit:
        values.append(value)


def _empty_row_example(record: dict[str, Any]) -> dict[str, Any]:
    """Serialize source identity without copying the full body into analysis output."""
    return {
        "issue_id": record["issue_id"],
        "repository": record["repository"],
        "source_split": record["source_split"],
        "source_row_index": record["source_row_index"],
        "title": record["title"],
        "raw_labels": record["raw_labels"],
        "target_category": record.get("target_category"),
        "atomic_labels": record.get("atomic_labels", []),
    }


def _target_mapping(raw_labels: Any) -> dict[str, Any]:
    """Apply only the previously established high-confidence mappings."""
    atoms = apparent_atomic_labels(raw_labels)
    mappings = []
    non_target_high_confidence = []
    ambiguous_atoms = []
    for atom in dict.fromkeys(atoms):
        category, status, reason = classify_atom(atom)
        if status == "high_confidence" and category in TARGET_CATEGORY_MAP:
            mappings.append(
                {
                    "atomic_label": atom,
                    "source_category": category,
                    "target_category": TARGET_CATEGORY_MAP[category],
                    "mapping_reason": reason,
                }
            )
        elif status == "high_confidence" and category is not None:
            non_target_high_confidence.append({"atomic_label": atom, "category": category, "reason": reason})
        elif status == "ambiguous":
            ambiguous_atoms.append({"atomic_label": atom, "reason": reason})

    target_categories = sorted({mapping["target_category"] for mapping in mappings})
    return {
        "apparent_atomic_labels": atoms,
        "mappings": mappings,
        "target_categories": target_categories,
        "non_target_high_confidence": non_target_high_confidence,
        "ambiguous_atoms": ambiguous_atoms,
    }


def _load_candidate_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load raw rows and retain only rows with one target category before deduplication."""
    dataset_splits = load_dataset_splits()
    all_columns = sorted({column for dataset in dataset_splits.values() for column in dataset.column_names})
    column_map = {
        "issue_id": resolve_column(all_columns, ("issue_id", "id", "number", "issue_number"), "issue_id"),
        "repository": resolve_column(all_columns, ("repository", "repo", "repository_name", "repo_name"), "repo"),
        "title": resolve_column(all_columns, ("title", "issue_title"), "title"),
        "body": resolve_column(all_columns, ("body", "issue_body", "description"), "body"),
        "labels": resolve_column(all_columns, ("labels", "label"), "label"),
    }
    required = [name for name, column in column_map.items() if column is None]
    if required:
        raise ValueError(f"Required columns could not be resolved: {required}")

    candidate_rows: list[dict[str, Any]] = []
    exclusion_counts = Counter()
    exclusion_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    non_target_counts = Counter()
    ambiguous_counts = Counter()
    total_rows = 0
    split_row_counts = {}

    for split_name, dataset in dataset_splits.items():
        split_row_counts[split_name] = len(dataset)
        for row_index, row in enumerate(dataset):
            values = dict(row)
            total_rows += 1
            raw_labels = text_value(values.get(column_map["labels"]))
            mapping = _target_mapping(raw_labels)
            source_record = {
                "issue_id": values.get(column_map["issue_id"]),
                "repository": text_value(values.get(column_map["repository"])).strip() or "<MISSING_REPOSITORY>",
                "title": text_value(values.get(column_map["title"])),
                "body": text_value(values.get(column_map["body"])),
                "raw_labels": raw_labels,
                "source_split": split_name,
                "source_row_index": row_index,
            }
            for item in mapping["non_target_high_confidence"]:
                non_target_counts[item["category"]] += 1
            for item in mapping["ambiguous_atoms"]:
                ambiguous_counts[item["atomic_label"]] += 1

            if not mapping["target_categories"]:
                exclusion_counts["no_high_confidence_target_category"] += 1
                if len(exclusion_examples["no_high_confidence_target_category"]) < EXAMPLE_LIMIT:
                    exclusion_examples["no_high_confidence_target_category"].append(_empty_row_example({**source_record, "target_category": None, "atomic_labels": mapping["apparent_atomic_labels"]}))
                continue
            if len(mapping["target_categories"]) > 1:
                exclusion_counts["conflicting_target_categories"] += 1
                if len(exclusion_examples["conflicting_target_categories"]) < EXAMPLE_LIMIT:
                    exclusion_examples["conflicting_target_categories"].append(
                        _empty_row_example(
                            {
                                **source_record,
                                "target_category": mapping["target_categories"],
                                "atomic_labels": mapping["apparent_atomic_labels"],
                            }
                        )
                    )
                continue

            target_category = mapping["target_categories"][0]
            candidate_rows.append(
                {
                    **source_record,
                    "target_category": target_category,
                    "atomic_labels": [item["atomic_label"] for item in mapping["mappings"]],
                    "mapping_reasons": mapping["mappings"],
                    "non_target_high_confidence": mapping["non_target_high_confidence"],
                    "ambiguous_atoms": mapping["ambiguous_atoms"],
                }
            )

    return candidate_rows, {
        "total_source_rows": total_rows,
        "source_rows_per_split": split_row_counts,
        "column_mapping": column_map,
        "exclusion_counts_before_duplicate_removal": dict(exclusion_counts),
        "exclusion_examples_before_duplicate_removal": dict(exclusion_examples),
        "non_target_high_confidence_atom_counts_in_retained_candidates": dict(non_target_counts),
        "ambiguous_atom_counts_in_retained_candidates": dict(ambiguous_counts),
    }


def _deduplicate_rows(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove exact title/body duplicates by keeping the earliest source row."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        duplicate_key = _sha256([row["title"], row["body"]])
        row["title_body_sha256"] = duplicate_key
        groups[duplicate_key].append(row)

    retained_rows = []
    duplicate_groups = []
    removed_rows = []
    for duplicate_key, rows in groups.items():
        rows.sort(key=lambda row: (row["source_split"], row["source_row_index"], str(row["issue_id"])))
        retained_rows.append(rows[0])
        if len(rows) <= 1:
            continue
        group = {
            "title": rows[0]["title"],
            "body_sha256": _sha256(rows[0]["body"]),
            "body_length": len(rows[0]["body"]),
            "title_body_sha256": duplicate_key,
            "kept": _empty_row_example(rows[0]),
            "removed": [_empty_row_example(row) for row in rows[1:]],
        }
        duplicate_groups.append(group)
        removed_rows.extend(rows[1:])

    retained_rows.sort(key=lambda row: (row["source_split"], row["source_row_index"], str(row["issue_id"])))
    duplicate_groups.sort(key=lambda group: group["title_body_sha256"])
    return retained_rows, {
        "duplicate_group_count": len(duplicate_groups),
        "candidate_rows_in_duplicate_groups": sum(1 + len(group["removed"]) for group in duplicate_groups),
        "duplicate_rows_removed": len(removed_rows),
        "duplicate_groups": duplicate_groups,
    }


def _normalized_output_row(row: dict[str, Any]) -> dict[str, Any]:
    """Select the public normalized dataset fields and provenance needed for review."""
    return {
        "issue_id": row["issue_id"],
        "repository": row["repository"],
        "title": row["title"],
        "body": row["body"],
        "raw_labels": row["raw_labels"],
        "target_category": row["target_category"],
        "atomic_labels": row["atomic_labels"],
        "mapping_reasons": row["mapping_reasons"],
        "source_split": row["source_split"],
        "source_row_index": row["source_row_index"],
    }


def write_normalized_dataset(rows: list[dict[str, Any]]) -> None:
    """Write the normalized retained rows as ignored JSONL, without split assignments."""
    NORMALIZED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED_DATASET_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(_normalized_output_row(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _class_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    """Count retained rows by final target category."""
    return Counter(row["target_category"] for row in rows)


def _retained_text_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report text conditions without filtering retained examples."""
    category_patterns = {
        "bug": r"\bbugs?\b",
        "feature": r"\b(?:feature|enhancement)\b",
        "documentation": r"\b(?:documentation|docs?)\b",
        "question_support": r"\b(?:question|support)\b",
    }
    category_word_counts = Counter()
    category_word_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        combined_text = f"{row['title']}\n{row['body']}"
        if len(row["body"]) == 0:
            body_condition = "empty_body"
        else:
            body_condition = None
        if body_condition:
            category_word_counts[body_condition] += 1
        if len(row["body"]) >= 10_000:
            category_word_counts["body_at_or_over_10000_characters"] += 1
        for category, pattern in category_patterns.items():
            if re.search(pattern, combined_text, flags=re.IGNORECASE):
                category_word_counts[f"category_word_in_text:{category}"] += 1
                _append_capped(category_word_examples[category], _empty_row_example(row), EXAMPLE_LIMIT)
    return {
        "empty_body_count": category_word_counts["empty_body"],
        "body_at_or_over_10000_characters_count": category_word_counts["body_at_or_over_10000_characters"],
        "category_word_in_title_or_body_counts": {
            category: category_word_counts[f"category_word_in_text:{category}"]
            for category in TARGET_CATEGORY_ORDER
        },
        "category_word_examples": dict(category_word_examples),
        "policy": "These conditions were reported only; no rows were removed for empty bodies, long bodies, or category words appearing in text.",
    }


def _repository_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build per-repository class counts for held-out split analysis."""
    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        repository = row["repository"]
        if repository not in counts:
            counts[repository] = {"total": 0, "classes": Counter()}
        counts[repository]["total"] += 1
        counts[repository]["classes"][row["target_category"]] += 1
    return counts


def _split_statistics(rows: list[dict[str, Any]], assignments: dict[Any, str], key_field: str, all_repositories: set[str]) -> dict[str, Any]:
    """Summarize issue, class, and repository coverage for a proposed split assignment."""
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_FRACTIONS}
    for row in rows:
        key = row["repository"] if key_field == "repository" else (row["source_row_index"], row["source_split"])
        split_rows[assignments[key]].append(row)

    output = {}
    for split, split_items in split_rows.items():
        class_counts = _class_counts(split_items)
        repository_names = sorted({row["repository"] for row in split_items})
        total = len(split_items)
        output[split] = {
            "issue_count": total,
            "class_counts": {category: class_counts[category] for category in TARGET_CATEGORY_ORDER},
            "class_proportions": {category: round(100 * class_counts[category] / total, 2) if total else 0 for category in TARGET_CATEGORY_ORDER},
            "repository_count": len(repository_names),
            "repositories": repository_names,
            "all_classes_present": all(class_counts[category] > 0 for category in TARGET_CATEGORY_ORDER),
        }
    output["all_classes_present_in_every_split"] = all(item["all_classes_present"] for item in output.values())
    output["repository_overlap"] = _repository_overlap(split_rows)
    output["repository_holdout"] = key_field == "repository"
    output["total_issue_count"] = len(rows)
    output["all_source_repositories"] = sorted(all_repositories)
    return output


def _repository_overlap(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Report repository overlap between issue-level split partitions."""
    repositories_by_split = {
        split: {row["repository"] for row in rows}
        for split, rows in split_rows.items()
    }
    overlap = {}
    split_names = list(SPLIT_FRACTIONS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            shared = sorted(repositories_by_split[left] & repositories_by_split[right])
            overlap[f"{left}_vs_{right}"] = {"shared_repository_count": len(shared), "shared_repositories": shared}
    return overlap


def _random_issue_assignments(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Assign issues deterministically by stable hash for a random-like benchmark."""
    assignments = {}
    for row in rows:
        digest = hashlib.sha256(f"{DATASET_ID}|{row['repository']}|{row['issue_id']}|{row['source_split']}|{row['source_row_index']}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < SPLIT_FRACTIONS["train"]:
            split = "train"
        elif value < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]:
            split = "validation"
        else:
            split = "test"
        assignments[row["source_row_index"], row["source_split"]] = split
    return assignments


def _repo_order(repository: str, seed: int, repository_counts: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    """Order repositories deterministically for one balance-search candidate."""
    digest = hashlib.sha256(f"repo-split|{seed}|{repository}".encode("utf-8")).hexdigest()
    counts = repository_counts[repository]["classes"]
    rare_class_weight = max((1 / max(counts[category], 1) for category in TARGET_CATEGORY_ORDER), default=0)
    return (-repository_counts[repository]["total"], -rare_class_weight, digest, repository)


def _repo_assignment_objective(assignment: dict[str, str], repository_counts: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    """Score only repository-count and class-distribution balance, never model performance."""
    split_totals = Counter()
    split_classes = {split: Counter() for split in SPLIT_FRACTIONS}
    for repository, split in assignment.items():
        split_totals[split] += repository_counts[repository]["total"]
        split_classes[split].update(repository_counts[repository]["classes"])
    total = sum(split_totals.values())
    global_classes = Counter()
    for info in repository_counts.values():
        global_classes.update(info["classes"])
    size_error = sum(abs(split_totals[split] / total - SPLIT_FRACTIONS[split]) for split in SPLIT_FRACTIONS)
    class_error = 0.0
    missing_classes = 0
    for split, fraction in SPLIT_FRACTIONS.items():
        for category in TARGET_CATEGORY_ORDER:
            target = global_classes[category] * fraction
            class_error += abs(split_classes[split][category] - target) / max(global_classes[category], 1)
            missing_classes += int(split_classes[split][category] == 0)
    return (float(missing_classes), class_error, size_error)


def _repair_repository_assignment(assignment: dict[str, str], repository_counts: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Repair missing classes with deterministic moves or swaps while preserving repository holdout."""
    for _ in range(20):
        split_classes = {split: Counter() for split in SPLIT_FRACTIONS}
        for repository, split in assignment.items():
            split_classes[split].update(repository_counts[repository]["classes"])
        missing = [(split, category) for split in SPLIT_FRACTIONS for category in TARGET_CATEGORY_ORDER if split_classes[split][category] == 0]
        if not missing:
            return assignment
        improved = False
        for target_split, category in missing:
            candidates = []
            for repository, source_split in assignment.items():
                if source_split == target_split:
                    continue
                if repository_counts[repository]["classes"][category] == 0:
                    continue
                if split_classes[source_split][category] <= repository_counts[repository]["classes"][category]:
                    continue
                candidate_assignment = dict(assignment)
                candidate_assignment[repository] = target_split
                candidates.append((_repo_assignment_objective(candidate_assignment, repository_counts), repository, candidate_assignment))
            if candidates:
                _, _, best_assignment = min(candidates, key=lambda item: (item[0], item[1]))
                assignment = best_assignment
                improved = True
                break
        if improved:
            continue
        return assignment
    return assignment


def _repository_split_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Search deterministic repository assignments and return the best balance candidates."""
    repository_counts = _repository_counts(rows)
    repositories = sorted(repository_counts)
    candidates = []
    for seed in REPO_SEARCH_SEEDS:
        assignment: dict[str, str] = {}
        split_totals = Counter()
        for repository in sorted(repositories, key=lambda item: _repo_order(item, seed, repository_counts)):
            options = []
            for split in SPLIT_FRACTIONS:
                tentative = dict(assignment)
                tentative[repository] = split
                options.append((_repo_assignment_objective(tentative, repository_counts), split, tentative))
            _, _, assignment = min(options, key=lambda item: (item[0], item[1]))
            split_totals[assignment[repository]] += repository_counts[repository]["total"]
        assignment = _repair_repository_assignment(assignment, repository_counts)
        objective = _repo_assignment_objective(assignment, repository_counts)
        stats = _split_statistics(rows, assignment, "repository", set(repositories))
        candidates.append({"seed": seed, "objective": objective, "repository_assignment": assignment, "statistics": stats})
    candidates.sort(key=lambda candidate: (candidate["objective"], candidate["seed"]))
    return candidates


def analyze_splits() -> dict[str, Any]:
    """Construct retained rows, write normalized JSONL, and analyze split strategies."""
    candidate_rows, construction = _load_candidate_rows()
    retained_rows, duplicate_report = _deduplicate_rows(candidate_rows)
    write_normalized_dataset(retained_rows)

    final_class_counts = _class_counts(retained_rows)
    final_repository_counts = defaultdict(set)
    for row in retained_rows:
        final_repository_counts[row["target_category"]].add(row["repository"])
    retained_total = len(retained_rows)
    final_classes = {
        category: {
            "issue_count": final_class_counts[category],
            "repository_count": len(final_repository_counts[category]),
            "percentage_of_retained_dataset": round(100 * final_class_counts[category] / retained_total, 2) if retained_total else 0,
        }
        for category in TARGET_CATEGORY_ORDER
    }

    random_assignments = _random_issue_assignments(retained_rows)
    random_stats = _split_statistics(retained_rows, random_assignments, "source_row_index", set(row["repository"] for row in retained_rows))
    repo_candidates = _repository_split_candidates(retained_rows)
    best_repo_candidate = repo_candidates[0] if repo_candidates else None

    repo_candidate_summaries = []
    for candidate in repo_candidates[:REPO_CANDIDATE_REPORT_LIMIT]:
        repo_candidate_summaries.append(
            {
                "search_seed": candidate["seed"],
                "balance_objective": candidate["objective"],
                "statistics": candidate["statistics"],
            }
        )

    return {
        "dataset_id": DATASET_ID,
        "construction_scope": {
            "target_categories": list(TARGET_CATEGORY_ORDER),
            "performance_included": False,
            "priority_and_severity_used": False,
            "mapping_rules_source": "Existing taxonomy_analysis.py high-confidence mappings only; no new lexical rules were added.",
            "raw_labels_preserved": True,
            "empty_bodies_removed": False,
            "long_bodies_removed": False,
            "category_words_in_text_removed": False,
            "dataset_splits_written": False,
            "model_downloaded": False,
            "tokenization_performed": False,
        },
        "normalized_dataset": {
            "path": str(NORMALIZED_DATASET_PATH),
            "format": "JSONL",
            "fields": ["issue_id", "repository", "title", "body", "raw_labels", "target_category", "atomic_labels", "mapping_reasons", "source_split", "source_row_index"],
            "retained_example_count": retained_total,
        },
        "source_and_exclusion_summary": construction,
        "duplicate_title_body_removal": duplicate_report,
        "final_class_counts": {
            "total_retained_examples": retained_total,
            "category_counts": final_classes,
        },
        "retained_text_conditions": _retained_text_conditions(retained_rows),
        "split_analysis": {
            "target_split_fractions": SPLIT_FRACTIONS,
            "random_issue_level": {
                "assignment_method": "Stable SHA-256 hash of dataset ID, repository, issue ID, and source location; thresholds 70/15/15. No model score was used.",
                "statistics": random_stats,
            },
            "repository_held_out": {
                "search_method": "64 deterministic greedy repository assignments with deterministic repair, scored only by missing-class count, class-count deviation, and issue-count deviation.",
                "candidate_count_searched": len(repo_candidates),
                "all_classes_in_every_candidate_count": sum(int(candidate["statistics"]["all_classes_present_in_every_split"]) for candidate in repo_candidates),
                "recommended_candidate": {
                    "search_seed": best_repo_candidate["seed"] if best_repo_candidate else None,
                    "balance_objective": best_repo_candidate["objective"] if best_repo_candidate else None,
                    "statistics": best_repo_candidate["statistics"] if best_repo_candidate else None,
                },
                "candidate_summaries": repo_candidate_summaries,
                "viability_assessment": {
                    "viable": bool(best_repo_candidate and best_repo_candidate["statistics"]["all_classes_present_in_every_split"]),
                    "reason": "Repository-held-out evaluation is viable when every split contains all four target classes; repository assignments are reported for review and no split files were written.",
                },
            },
        },
        "recommendation": {
            "primary_evaluation_design": "repository_held_out" if best_repo_candidate and best_repo_candidate["statistics"]["all_classes_present_in_every_split"] else "random_issue_level",
            "primary_reason": "A repository-held-out test measures cross-repository transfer and avoids repository-specific label conventions appearing in both train and test.",
            "secondary_random_benchmark": "Keep a deterministic random issue-level benchmark because it estimates within-repository performance and provides a useful comparison against the harder cross-repository evaluation.",
            "cautions": [
                "Do not choose repository assignments based on model scores; the candidate search used only class and size balance.",
                "The question_support and documentation classes are much smaller than bug and feature, so class proportions should be monitored in every split.",
                "No split files were finalized in this stage.",
            ],
        },
    }


def write_split_analysis_report(summary: dict[str, Any], path: Path = Path("results/split_analysis.json")) -> None:
    """Write split analysis metadata and candidate assignments as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_split_analysis_summary(summary: dict[str, Any]) -> None:
    """Print the final construction and split-analysis summary."""
    final_counts = summary["final_class_counts"]
    split_analysis = summary["split_analysis"]
    print("\nNormalized dataset and split analysis complete")
    print(f"Retained examples: {final_counts['total_retained_examples']}")
    for category, details in final_counts["category_counts"].items():
        print(f"  {category}: {details['issue_count']} issues across {details['repository_count']} repositories ({details['percentage_of_retained_dataset']}%)")
    print(f"Conflicting target rows removed: {summary['duplicate_title_body_removal']['duplicate_rows_removed']} duplicate rows removed after target filtering")
    print(f"Random issue-level split all classes present: {split_analysis['random_issue_level']['statistics']['all_classes_present_in_every_split']}")
    repo_stats = split_analysis["repository_held_out"]["recommended_candidate"]["statistics"]
    print(f"Repository-held-out split viable: {bool(repo_stats and repo_stats['all_classes_present_in_every_split'])}")
    print(f"Primary evaluation design: {summary['recommendation']['primary_evaluation_design']}")
    print(f"Normalized dataset: {summary['normalized_dataset']['path']}")
    print("Split analysis report: results\\split_analysis.json")
