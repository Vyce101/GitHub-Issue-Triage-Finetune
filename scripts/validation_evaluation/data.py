"""Load and verify the frozen validation rows and recorded few-shot examples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from baseline.data import read_jsonl

from .config import (
    CONFIG_PATH,
    FROZEN_MANIFEST_PATH,
    PROMPT_DEFINITION_PATH,
    PROMPT_SELECTION_PATH,
    TARGET_CATEGORIES,
    TRAIN_SPLIT_PATH,
    VALIDATION_ROW_COUNT,
    VALIDATION_SHA256,
    VALIDATION_SPLIT_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a frozen file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_row_identity_matches(row: dict[str, Any], reference: dict[str, Any], *, name: str) -> None:
    """Verify the recorded identity and target category for one selected row."""
    fields = ("issue_id", "repository", "source_split", "source_row_index", "target_category")
    mismatches = {
        field: (row.get(field), reference.get(field))
        for field in fields
        if row.get(field) != reference.get(field)
    }
    if mismatches:
        raise ValueError(f"{name} does not match the recorded selection: {mismatches}")


def load_frozen_validation_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only validation rows and enforce the frozen split boundary."""
    manifest = _read_json(FROZEN_MANIFEST_PATH)
    manifest_validation = manifest["splits"]["validation"]
    if manifest_validation["sha256"] != VALIDATION_SHA256:
        raise ValueError("Frozen manifest validation SHA-256 does not match the evaluation constant")
    if manifest_validation["row_count"] != VALIDATION_ROW_COUNT:
        raise ValueError("Frozen manifest validation row count does not match the evaluation constant")

    actual_sha256 = _sha256(VALIDATION_SPLIT_PATH)
    if actual_sha256 != VALIDATION_SHA256:
        raise ValueError(
            f"Validation split SHA-256 mismatch: expected {VALIDATION_SHA256}, got {actual_sha256}"
        )

    rows = read_jsonl(VALIDATION_SPLIT_PATH)
    if len(rows) != VALIDATION_ROW_COUNT:
        raise ValueError(f"Expected {VALIDATION_ROW_COUNT} validation rows, got {len(rows)}")

    expected_repositories = set(manifest_validation["repositories"])
    class_counts = Counter()
    for row in rows:
        if row.get("repository") not in expected_repositories:
            raise ValueError(f"Validation row has an unexpected repository: {row.get('repository')!r}")
        category = row.get("target_category")
        if category not in TARGET_CATEGORIES:
            raise ValueError(f"Validation row has an unexpected target category: {category!r}")
        class_counts[category] += 1

    if dict(class_counts) != manifest_validation["class_counts"]:
        raise ValueError(
            f"Validation class counts mismatch: expected {manifest_validation['class_counts']}, got {dict(class_counts)}"
        )

    return rows, {
        "path": str(VALIDATION_SPLIT_PATH.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/"),
        "sha256": actual_sha256,
        "row_count": len(rows),
        "class_counts": {category: class_counts[category] for category in TARGET_CATEGORIES},
        "repository_count": len({row["repository"] for row in rows}),
        "repositories": sorted({str(row["repository"]) for row in rows}),
        "test_split_accessed_by_evaluator": False,
    }


def load_recorded_few_shot_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the exact eight train-only demonstrations from recorded identities."""
    selection = _read_json(PROMPT_SELECTION_PATH)
    prompt_definition = _read_json(PROMPT_DEFINITION_PATH)
    train_rows = read_jsonl(TRAIN_SPLIT_PATH)
    train_by_key = {
        (row["source_split"], int(row["source_row_index"])): row
        for row in train_rows
    }

    selected_records = selection.get("few_shot_examples", [])
    prompt_records = prompt_definition.get("few_shot_examples", [])
    if len(selected_records) != 8 or len(prompt_records) != 8:
        raise ValueError("The recorded few-shot prompt must contain exactly eight examples")

    few_shot_rows = []
    for index, selected in enumerate(selected_records):
        if selected.get("source_split") != "train":
            raise ValueError("Recorded few-shot examples must all come from train")
        key = (selected["source_split"], int(selected["source_row_index"]))
        row = train_by_key.get(key)
        if row is None:
            raise ValueError(f"Recorded few-shot row is missing from train: {key}")
        _assert_row_identity_matches(row, selected, name=f"few-shot selection {index}")

        prompt_record = prompt_records[index]
        _assert_row_identity_matches(row, prompt_record, name=f"few-shot prompt definition {index}")
        for field in ("title", "body"):
            if row[field] != prompt_record.get(field):
                raise ValueError(f"Recorded few-shot prompt content differs for example {index}")
        few_shot_rows.append(row)

    class_counts = Counter(row["target_category"] for row in few_shot_rows)
    if {category: class_counts[category] for category in TARGET_CATEGORIES} != {
        category: 2 for category in TARGET_CATEGORIES
    }:
        raise ValueError("Recorded few-shot examples must contain two examples per category")

    return few_shot_rows, {
        "selection_path": str(PROMPT_SELECTION_PATH.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/"),
        "prompt_definition_path": str(PROMPT_DEFINITION_PATH.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/"),
        "source_split": "train",
        "example_count": len(few_shot_rows),
        "class_counts": {category: class_counts[category] for category in TARGET_CATEGORIES},
        "examples": [
            {
                "example_number": index,
                "issue_id": row["issue_id"],
                "repository": row["repository"],
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "target_category": row["target_category"],
            }
            for index, row in enumerate(few_shot_rows, start=1)
        ],
        "validation_or_test_examples_selected": False,
        "selection_recomputed": False,
    }
