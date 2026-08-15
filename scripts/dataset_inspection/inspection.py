"""Coordinate raw row inspection and assemble the machine-readable report."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .config import (
    DATASET_ID,
    LONG_BODY_LIMIT,
    SHORT_TEXT_BODY_LIMIT,
    SHORT_TEXT_TITLE_LIMIT,
    TOP_LABEL_LIMIT,
)
from .label_analysis import (
    counter_entries,
    label_appears_in_text,
    label_shape_signature,
    label_style_names,
    metadata_reasons,
)
from .loader import load_dataset_splits
from .raw_values import (
    choose_examples,
    display_value,
    example_payload,
    is_empty_text,
    is_null,
    json_safe,
    numeric_stats,
    raw_label_items,
    resolve_column,
    shorten_text,
    text_value,
    unique_label_items,
    value_key,
)


def _duplicate_groups(groups: dict[str, list[dict[str, Any]]], include_text: bool = False) -> dict[str, Any]:
    """Summarize duplicate groups while keeping the report bounded."""
    duplicates = [items for items in groups.values() if len(items) > 1]
    duplicates.sort(key=len, reverse=True)
    output = []
    for items in duplicates[:TOP_LABEL_LIMIT]:
        item = {"occurrences": len(items), "locations": items}
        if include_text:
            item["title"] = items[0]["title"]
            item["body_shortened"] = shorten_text(items[0]["body"])
            item["locations"] = [
                {key: value for key, value in location.items() if key not in {"title", "body"}}
                for location in items
            ]
        output.append(item)
    return {
        "duplicate_group_count": len(duplicates),
        "duplicate_row_count": sum(len(items) for items in duplicates),
        "groups_reported": output,
    }


def inspect_dataset() -> dict[str, Any]:
    """Load and inspect raw rows, preserving raw label values in the report."""
    dataset_splits = load_dataset_splits()
    split_names = list(dataset_splits.keys())
    all_columns = sorted({column for dataset in dataset_splits.values() for column in dataset.column_names})
    column_map = {
        "repository": resolve_column(all_columns, ("repository", "repo", "repository_name", "repo_name"), "repo"),
        "title": resolve_column(all_columns, ("title", "issue_title"), "title"),
        "body": resolve_column(all_columns, ("body", "issue_body", "description"), "body"),
        "labels": resolve_column(all_columns, ("labels", "label"), "label"),
        "issue_id": resolve_column(all_columns, ("issue_id", "id", "number", "issue_number"), "issue_id"),
    }

    records: list[dict[str, Any]] = []
    observed_types: dict[str, Counter[str]] = defaultdict(Counter)
    schema_by_split: dict[str, dict[str, Any]] = {}
    split_row_counts: dict[str, int] = {}
    storage_types: Counter[str] = Counter()
    serialized_storage_examples: dict[str, Any] = {}
    label_counter: Counter[str] = Counter()
    label_values: dict[str, Any] = {}
    repo_stats: dict[str, dict[str, Any]] = {}
    repo_label_counters: dict[str, Counter[str]] = defaultdict(Counter)
    global_style_counts: Counter[str] = Counter()
    repo_style_counts: dict[str, Counter[str]] = defaultdict(Counter)
    shape_values: dict[str, set[str]] = defaultdict(set)
    shape_repos: dict[str, set[str]] = defaultdict(set)
    shape_counts: dict[str, Counter[str]] = defaultdict(Counter)
    id_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    title_body_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leakage_counts: Counter[str] = Counter()
    leakage_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_counts: Counter[str] = Counter()
    metadata_reasons_by_key: dict[str, set[str]] = defaultdict(set)
    metadata_repos: dict[str, set[str]] = defaultdict(set)
    title_lengths: list[int] = []
    body_lengths: list[int] = []

    title_null_count = 0
    title_empty_count = 0
    body_empty_count = 0
    labels_empty_count = 0
    repository_missing_count = 0
    total_rows = 0

    for split_name, dataset in dataset_splits.items():
        split_row_counts[split_name] = len(dataset)
        schema_by_split[split_name] = {
            column: {
                "feature": repr(feature),
                "feature_type": type(feature).__name__,
            }
            for column, feature in dataset.features.items()
        }

        for row_index, row in enumerate(dataset):
            values = dict(row)
            total_rows += 1
            for column in all_columns:
                observed_types[column]["null" if is_null(values.get(column)) else type(values.get(column)).__name__] += 1

            title_value = values.get(column_map["title"]) if column_map["title"] else None
            body_value = values.get(column_map["body"]) if column_map["body"] else None
            repository_value = values.get(column_map["repository"]) if column_map["repository"] else None
            label_value = values.get(column_map["labels"]) if column_map["labels"] else None

            title_null_count += int(is_null(title_value))
            title_empty_count += int(is_empty_text(title_value))
            body_empty_count += int(is_empty_text(body_value))
            title_lengths.append(len(text_value(title_value)))
            body_lengths.append(len(text_value(body_value)))

            repository = text_value(repository_value).strip() or "<MISSING_REPOSITORY>"
            repository_missing_count += int(repository == "<MISSING_REPOSITORY>")
            if repository not in repo_stats:
                repo_stats[repository] = {
                    "issue_count": 0,
                    "labeled_issue_count": 0,
                    "zero_label_issues": 0,
                    "exactly_one_label_issues": 0,
                    "multi_label_issues": 0,
                    "raw_label_occurrence_count": 0,
                    "unique_raw_labels": 0,
                }
            repo_stats[repository]["issue_count"] += 1

            parsed_labels, storage_type = raw_label_items(label_value)
            storage_types[storage_type] += 1
            if storage_type in {"json_serialized_list", "python_literal_list"} and storage_type not in serialized_storage_examples:
                serialized_storage_examples[storage_type] = json_safe(label_value)
            unique_items = unique_label_items(parsed_labels)
            labels_empty_count += int(not unique_items)
            repo_stats[repository]["raw_label_occurrence_count"] += len(unique_items)
            if not unique_items:
                repo_stats[repository]["zero_label_issues"] += 1
            elif len(unique_items) == 1:
                repo_stats[repository]["labeled_issue_count"] += 1
                repo_stats[repository]["exactly_one_label_issues"] += 1
            else:
                repo_stats[repository]["labeled_issue_count"] += 1
                repo_stats[repository]["multi_label_issues"] += 1

            record = {
                "split": split_name,
                "row_index": row_index,
                "values": values,
                "raw_label_items": parsed_labels,
                "unique_label_items": unique_items,
                "repository": repository,
            }
            records.append(record)

            for raw_label in unique_items:
                key = value_key(raw_label)
                label_values.setdefault(key, raw_label)
                label_counter[key] += 1
                repo_label_counters[repository][key] += 1
                for style in label_style_names(raw_label):
                    global_style_counts[style] += 1
                    repo_style_counts[repository][style] += 1
                shape = label_shape_signature(raw_label)
                if shape:
                    raw_display = display_value(raw_label)
                    shape_values[shape].add(raw_display)
                    shape_repos[shape].add(repository)
                    shape_counts[shape][raw_display] += 1
                reasons = metadata_reasons(raw_label)
                if reasons:
                    metadata_counts[key] += 1
                    metadata_reasons_by_key[key].update(reasons)
                    metadata_repos[key].add(repository)
                if label_appears_in_text(raw_label, title_value, body_value, text_value):
                    leakage_counts[key] += 1
                    if len(leakage_examples[key]) < 3:
                        leakage_examples[key].append(example_payload(record, column_map["repository"], column_map["title"], column_map["body"]))

            repo_stats[repository]["unique_raw_labels"] = len(repo_label_counters[repository])

            if column_map["issue_id"]:
                issue_id = values.get(column_map["issue_id"])
                if not is_null(issue_id) and not is_empty_text(issue_id):
                    id_groups[value_key(issue_id)].append({"value": json_safe(issue_id), "split": split_name, "row_index": row_index, "repository": repository})

            title_body_key = value_key([text_value(title_value), text_value(body_value)])
            title_body_groups[title_body_key].append({"title": text_value(title_value), "body": text_value(body_value), "split": split_name, "row_index": row_index, "repository": repository})

    raw_label_frequency = counter_entries(label_counter, label_values)
    repo_output = {}
    for repository, stats in sorted(repo_stats.items(), key=lambda item: (-item[1]["issue_count"], item[0])):
        stats = dict(stats)
        stats["top_raw_labels"] = counter_entries(repo_label_counters[repository], label_values, 20)
        stats["label_naming_styles"] = dict(repo_style_counts[repository].most_common())
        repo_output[repository] = stats

    candidate_variant_groups = []
    for shape, raw_values in shape_values.items():
        if len(raw_values) < 2 or len(shape_repos[shape]) < 2:
            continue
        candidate_variant_groups.append(
            {
                "exploratory_signature": shape,
                "raw_label_variants": sorted(raw_values),
                "repositories": sorted(shape_repos[shape]),
                "counts_by_raw_label": dict(shape_counts[shape].most_common()),
            }
        )
    candidate_variant_groups.sort(key=lambda item: sum(item["counts_by_raw_label"].values()), reverse=True)

    metadata_candidates = []
    for key, count in metadata_counts.most_common(TOP_LABEL_LIMIT):
        metadata_candidates.append(
            {
                "raw_label": json_safe(label_values[key]),
                "display": display_value(label_values[key]),
                "issue_count": count,
                "reasons": sorted(metadata_reasons_by_key[key]),
                "repositories": sorted(metadata_repos[key]),
            }
        )

    leakage_candidates = []
    for key, count in leakage_counts.most_common(TOP_LABEL_LIMIT):
        leakage_candidates.append(
            {
                "raw_label": json_safe(label_values[key]),
                "display": display_value(label_values[key]),
                "issue_count_with_label": label_counter[key],
                "issue_count_with_name_overlap": count,
                "overlap_percentage": round(100 * count / label_counter[key], 2),
                "examples": leakage_examples[key],
            }
        )

    short_text_records = [
        record
        for record in records
        if len(text_value(record["values"].get(column_map["title"]))) <= SHORT_TEXT_TITLE_LIMIT
        or len(text_value(record["values"].get(column_map["body"]))) <= SHORT_TEXT_BODY_LIMIT
    ]
    long_body_records = [record for record in records if len(text_value(record["values"].get(column_map["body"]))) >= LONG_BODY_LIMIT]

    return {
        "dataset_id": DATASET_ID,
        "inspection_scope": {
            "raw_rows_only": True,
            "labels_normalized": False,
            "labels_remapped": False,
            "rows_removed": False,
            "splits_created": False,
            "priority_severity_columns_used": False,
            "ignored_untrusted_columns": [column for column in all_columns if column.casefold() in {"priority", "severity"}],
            "model_downloaded": False,
            "dataset_downloaded_for_inspection": True,
        },
        "basic_structure": {
            "total_rows": total_rows,
            "rows_per_split": split_row_counts,
            "split_names": split_names,
            "columns": all_columns,
            "column_mapping": column_map,
            "schema_by_split": schema_by_split,
            "observed_python_types": {column: dict(counts) for column, counts in observed_types.items()},
            "unique_repositories": len({record["repository"] for record in records if record["repository"] != "<MISSING_REPOSITORY>"}),
            "missing_repository_count": repository_missing_count,
            "title_missing_or_null_count": title_null_count,
            "title_empty_or_null_count": title_empty_count,
            "body_missing_null_or_empty_count": body_empty_count,
            "labels_missing_null_or_empty_count": labels_empty_count,
        },
        "labels": {
            "storage_types_by_row": dict(storage_types),
            "serialized_storage_examples": serialized_storage_examples,
            "raw_unique_label_count": len(label_counter),
            "top_100_raw_label_frequency": raw_label_frequency,
            "issue_label_cardinality": {
                "zero_labels": labels_empty_count,
                "zero_labels_percentage": round(100 * labels_empty_count / total_rows, 2) if total_rows else 0,
                "exactly_one_label": sum(1 for record in records if len(record["unique_label_items"]) == 1),
                "multiple_labels": sum(1 for record in records if len(record["unique_label_items"]) > 1),
            },
            "multi_label_examples": [example_payload(record, column_map["repository"], column_map["title"], column_map["body"]) for record in choose_examples(records, lambda record: len(record["unique_label_items"]) > 1)],
            "metadata_like_label_candidates": metadata_candidates,
            "label_name_text_overlap_candidates": leakage_candidates,
        },
        "repositories": {
            "issue_count_and_label_distribution": repo_output,
            "global_label_naming_style_counts": dict(global_style_counts.most_common()),
            "candidate_naming_variant_groups": candidate_variant_groups[:TOP_LABEL_LIMIT],
            "naming_convention_note": "Variant groups use a temporary case/separator signature for exploration only. Raw labels remain unchanged and no taxonomy is proposed.",
        },
        "text": {
            "title_length_characters": numeric_stats(title_lengths),
            "body_length_characters": numeric_stats(body_lengths),
            "representative_examples": [example_payload(record, column_map["repository"], column_map["title"], column_map["body"]) for record in choose_examples(records, lambda record: True)],
        },
        "data_quality": {
            "duplicate_issue_ids": _duplicate_groups(id_groups),
            "duplicate_title_body_pairs": _duplicate_groups(title_body_groups, include_text=True),
            "short_text_thresholds": {"title_characters_at_or_below": SHORT_TEXT_TITLE_LIMIT, "body_characters_at_or_below": SHORT_TEXT_BODY_LIMIT},
            "short_text_issue_count": len(short_text_records),
            "short_text_examples": [example_payload(record, column_map["repository"], column_map["title"], column_map["body"]) for record in choose_examples(short_text_records, lambda record: True)],
            "long_body_threshold_characters": LONG_BODY_LIMIT,
            "long_body_issue_count": len(long_body_records),
            "long_body_examples": [example_payload(record, column_map["repository"], column_map["title"], column_map["body"]) for record in choose_examples(long_body_records, lambda record: True)],
        },
    }
