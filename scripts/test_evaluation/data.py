"""Load the frozen TEST split and reconstruct the recorded TRAIN demonstrations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from baseline.data import read_jsonl
from validation_evaluation.data import load_recorded_few_shot_rows

from .config import (
    CONFIG_PATH,
    FROZEN_MANIFEST_PATH,
    TARGET_CATEGORIES,
    TEST_CLASS_COUNTS,
    TEST_ROW_COUNT,
    TEST_SHA256,
    TEST_SPLIT_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a frozen split file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_test_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify the frozen manifest, file hash, row count, repositories, and classes."""
    manifest = _read_json(FROZEN_MANIFEST_PATH)
    manifest_test = manifest["splits"]["test"]
    if manifest_test["sha256"] != TEST_SHA256:
        raise ValueError("Frozen manifest TEST SHA-256 does not match the evaluation constant")
    if manifest_test["row_count"] != TEST_ROW_COUNT:
        raise ValueError("Frozen manifest TEST row count does not match the evaluation constant")
    if manifest_test["class_counts"] != TEST_CLASS_COUNTS:
        raise ValueError("Frozen manifest TEST class counts do not match the evaluation constants")

    actual_sha256 = _sha256(TEST_SPLIT_PATH)
    if actual_sha256 != TEST_SHA256:
        raise ValueError(f"TEST split SHA-256 mismatch: expected {TEST_SHA256}, got {actual_sha256}")

    rows = read_jsonl(TEST_SPLIT_PATH)
    if len(rows) != TEST_ROW_COUNT:
        raise ValueError(f"Expected {TEST_ROW_COUNT} TEST rows, got {len(rows)}")

    expected_repositories = set(manifest_test["repositories"])
    class_counts = Counter()
    for row in rows:
        if not row.get("source_split"):
            raise ValueError("TEST row is missing its original source split provenance")
        if row.get("repository") not in expected_repositories:
            raise ValueError(f"TEST row has an unexpected repository: {row.get('repository')!r}")
        category = row.get("target_category")
        if category not in TARGET_CATEGORIES:
            raise ValueError(f"TEST row has an unexpected target category: {category!r}")
        class_counts[category] += 1

    normalized_counts = {category: class_counts[category] for category in TARGET_CATEGORIES}
    if normalized_counts != TEST_CLASS_COUNTS:
        raise ValueError(f"TEST class counts mismatch: expected {TEST_CLASS_COUNTS}, got {normalized_counts}")

    return rows, {
        "path": str(TEST_SPLIT_PATH.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/"),
        "sha256": actual_sha256,
        "expected_sha256": TEST_SHA256,
        "row_count": len(rows),
        "expected_row_count": TEST_ROW_COUNT,
        "class_counts": normalized_counts,
        "expected_class_counts": TEST_CLASS_COUNTS,
        "repository_count": len({row["repository"] for row in rows}),
        "repositories": sorted({str(row["repository"]) for row in rows}),
        "test_split_verified_for_final_evaluation": True,
    }


def load_frozen_prompt_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the unchanged eight-example TRAIN prompt and its reproducibility metadata."""
    return load_recorded_few_shot_rows()
