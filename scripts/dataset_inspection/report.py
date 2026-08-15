"""Human-readable and machine-readable report output."""

from __future__ import annotations

import json
from typing import Any

from .config import OUTPUT_PATH


def write_report(summary: dict[str, Any]) -> None:
    """Write the compact JSON inspection report."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_human_summary(summary: dict[str, Any]) -> None:
    """Print the compact human-readable inspection result."""
    structure = summary["basic_structure"]
    labels = summary["labels"]
    repositories = summary["repositories"]
    text = summary["text"]
    quality = summary["data_quality"]
    cardinality = labels["issue_label_cardinality"]
    print("\nDataset inspection complete")
    print(f"Dataset: {summary['dataset_id']}")
    print(f"Rows: {structure['total_rows']} across splits {structure['rows_per_split']}")
    print(f"Columns: {', '.join(structure['columns'])}")
    print(f"Unique repositories: {structure['unique_repositories']}")
    print(f"Label storage: {labels['storage_types_by_row']}")
    print(f"Unique raw labels: {labels['raw_unique_label_count']}")
    print(
        "Label cardinality: "
        f"zero={cardinality['zero_labels']} ({cardinality['zero_labels_percentage']}%), "
        f"one={cardinality['exactly_one_label']}, multiple={cardinality['multiple_labels']}"
    )
    print("Top raw labels:")
    for entry in labels["top_100_raw_label_frequency"][:10]:
        print(f"  {entry['display']!r}: {entry['count']}")
    print(f"Potential duplicate issue-ID groups: {quality['duplicate_issue_ids']['duplicate_group_count']}")
    print(f"Duplicate title/body groups: {quality['duplicate_title_body_pairs']['duplicate_group_count']}")
    print(f"Short-text issues: {quality['short_text_issue_count']}; long-body issues: {quality['long_body_issue_count']}")
    print(f"Title lengths: {text['title_length_characters']}")
    print(f"Body lengths: {text['body_length_characters']}")
    print(f"Repositories with exploratory naming-variant evidence: {len(repositories['candidate_naming_variant_groups'])}")
    print(f"Metadata-like label candidates reported: {len(labels['metadata_like_label_candidates'])}")
    print(f"Label/text overlap candidates reported: {len(labels['label_name_text_overlap_candidates'])}")
    print(f"Machine-readable report: {OUTPUT_PATH}")
