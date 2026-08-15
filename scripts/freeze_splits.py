"""Freeze the approved repository-held-out split and verify its invariants."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_SOURCE_PATH = PROJECT_ROOT / "data/processed/normalized_labeled_dataset.jsonl"
APPROVED_ANALYSIS_PATH = PROJECT_ROOT / "results/split_analysis.json"
SPLIT_DIRECTORY = PROJECT_ROOT / "data/processed/splits"
MANIFEST_PATH = PROJECT_ROOT / "results/final_split_manifest.json"
FREEZE_METHOD_VERSION = "freeze-splits-v1"
SPLIT_NAMES = ("train", "validation", "test")
CATEGORIES = ("bug", "feature", "documentation", "question_support")


def _stable_key(value: Any) -> str:
    """Serialize a value for stable identity comparisons."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _issue_id_sort_key(value: Any) -> tuple[int, Any]:
    """Sort numeric issue IDs numerically and other IDs lexically."""
    if isinstance(value, int) and not isinstance(value, bool):
        return 0, value
    return 1, str(value)


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Order records by repository, source issue ID, and provenance tie-breakers."""
    issue_id_kind, issue_id_value = _issue_id_sort_key(row["issue_id"])
    return (
        str(row["repository"]),
        issue_id_kind,
        issue_id_value,
        str(row["source_split"]),
        int(row["source_row_index"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records in source order."""
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact UTF-8 JSONL with deterministic LF line endings."""
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _approved_repository_assignment(analysis: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Read only the already-approved repository assignment from split analysis."""
    recommended = analysis["split_analysis"]["repository_held_out"]["recommended_candidate"]
    statistics = recommended["statistics"]
    assignment = {
        split: sorted(statistics[split]["repositories"])
        for split in SPLIT_NAMES
    }
    return assignment, recommended


def _split_rows_by_assignment(rows: list[dict[str, Any]], assignment: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Assign each normalized row exactly once using the approved repository lists."""
    repository_to_split = {}
    for split, repositories in assignment.items():
        for repository in repositories:
            if repository in repository_to_split:
                raise ValueError(f"Approved repository assignment overlaps: {repository}")
            repository_to_split[repository] = split

    split_rows = {split: [] for split in SPLIT_NAMES}
    for row in rows:
        repository = row["repository"]
        if repository not in repository_to_split:
            raise ValueError(f"Normalized row repository is missing from approved assignment: {repository}")
        split_rows[repository_to_split[repository]].append(row)
    for split in SPLIT_NAMES:
        split_rows[split].sort(key=_row_sort_key)
    return split_rows


def _class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return counts for all approved categories in a fixed order."""
    counts = Counter(row["target_category"] for row in rows)
    return {category: counts[category] for category in CATEGORIES}


def _repository_counts(rows: list[dict[str, Any]]) -> int:
    """Count distinct repositories in a split."""
    return len({row["repository"] for row in rows})


def _overlap_report(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Check source IDs, title/body pairs, and repositories across split pairs."""
    report = {}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            left_rows = split_rows[left]
            right_rows = split_rows[right]
            left_ids = {_stable_key(row["issue_id"]) for row in left_rows}
            right_ids = {_stable_key(row["issue_id"]) for row in right_rows}
            left_text = {_stable_key([row["title"], row["body"]]) for row in left_rows}
            right_text = {_stable_key([row["title"], row["body"]]) for row in right_rows}
            left_repositories = {row["repository"] for row in left_rows}
            right_repositories = {row["repository"] for row in right_rows}
            report[f"{left}_vs_{right}"] = {
                "source_issue_id_overlap_count": len(left_ids & right_ids),
                "title_body_overlap_count": len(left_text & right_text),
                "repository_overlap_count": len(left_repositories & right_repositories),
                "shared_repositories": sorted(left_repositories & right_repositories),
            }
    return report


def freeze_splits() -> dict[str, Any]:
    """Freeze, write, hash, and verify the approved repository-held-out split."""
    normalized_rows = _read_jsonl(NORMALIZED_SOURCE_PATH)
    analysis = json.loads(APPROVED_ANALYSIS_PATH.read_text(encoding="utf-8"))
    assignment, recommended_candidate = _approved_repository_assignment(analysis)
    split_rows = _split_rows_by_assignment(normalized_rows, assignment)

    source_keys = [_stable_key([row["source_split"], row["source_row_index"]]) for row in normalized_rows]
    split_source_keys = [
        _stable_key([row["source_split"], row["source_row_index"]])
        for split in SPLIT_NAMES
        for row in split_rows[split]
    ]
    source_ids = [_stable_key(row["issue_id"]) for row in normalized_rows]
    title_body_keys = [_stable_key([row["title"], row["body"]]) for row in normalized_rows]
    duplicate_source_keys = len(source_keys) - len(set(source_keys))
    duplicate_source_ids = len(source_ids) - len(set(source_ids))
    duplicate_title_body_pairs = len(title_body_keys) - len(set(title_body_keys))

    if len(split_source_keys) != len(source_keys) or Counter(split_source_keys) != Counter(source_keys):
        raise ValueError("Split source rows do not match the normalized source rows exactly")
    if duplicate_source_keys or duplicate_source_ids or duplicate_title_body_pairs:
        raise ValueError("Normalized source violates uniqueness invariants")

    expected_statistics = recommended_candidate["statistics"]
    expected_counts = {
        split: expected_statistics[split]["issue_count"]
        for split in SPLIT_NAMES
    }
    expected_class_counts = {
        split: expected_statistics[split]["class_counts"]
        for split in SPLIT_NAMES
    }
    expected_repository_counts = {
        split: expected_statistics[split]["repository_count"]
        for split in SPLIT_NAMES
    }
    actual_counts = {split: len(split_rows[split]) for split in SPLIT_NAMES}
    actual_class_counts = {split: _class_counts(split_rows[split]) for split in SPLIT_NAMES}
    actual_repository_counts = {split: _repository_counts(split_rows[split]) for split in SPLIT_NAMES}
    if actual_counts != expected_counts or actual_class_counts != expected_class_counts or actual_repository_counts != expected_repository_counts:
        raise ValueError("Generated split counts do not match the approved analysis")

    if not all(all(actual_class_counts[split][category] > 0 for category in CATEGORIES) for split in SPLIT_NAMES):
        raise ValueError("Every split must contain all approved categories")

    cross_split_overlap = _overlap_report(split_rows)
    if any(
        value["source_issue_id_overlap_count"]
        or value["title_body_overlap_count"]
        or value["repository_overlap_count"]
        for value in cross_split_overlap.values()
    ):
        raise ValueError(f"Cross-split leakage detected: {cross_split_overlap}")

    SPLIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split in SPLIT_NAMES:
        path = SPLIT_DIRECTORY / f"{split}.jsonl"
        _write_jsonl(path, split_rows[split])
        split_paths[split] = path

    split_hashes = {split: _sha256_file(path) for split, path in split_paths.items()}
    manifest = {
        "manifest_version": "1.0",
        "creation_method": FREEZE_METHOD_VERSION,
        "source": {
            "normalized_dataset_path": str(NORMALIZED_SOURCE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "normalized_dataset_sha256": _sha256_file(NORMALIZED_SOURCE_PATH),
            "normalized_dataset_row_count": len(normalized_rows),
            "approved_assignment_source": str(APPROVED_ANALYSIS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "approved_search_seed": recommended_candidate["search_seed"],
        },
        "target_categories": list(CATEGORIES),
        "repository_assignment": assignment,
        "ordering_rule": "UTF-8 JSONL with LF line endings; rows sorted by repository (lexical), numeric issue ID, source split, then source row index.",
        "splits": {
            split: {
                "path": str(split_paths[split].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": split_hashes[split],
                "row_count": actual_counts[split],
                "class_counts": actual_class_counts[split],
                "repository_count": actual_repository_counts[split],
                "repositories": assignment[split],
            }
            for split in SPLIT_NAMES
        },
        "verification": {
            "every_normalized_example_in_exactly_one_split": len(split_source_keys) == len(source_keys) and Counter(split_source_keys) == Counter(source_keys),
            "total_split_rows": sum(actual_counts.values()),
            "total_split_rows_match_normalized_source": sum(actual_counts.values()) == len(normalized_rows),
            "split_counts_match_approved_analysis": actual_counts == expected_counts,
            "class_counts_match_approved_analysis": actual_class_counts == expected_class_counts,
            "repository_counts_match_approved_analysis": actual_repository_counts == expected_repository_counts,
            "all_categories_in_all_splits": all(all(actual_class_counts[split][category] > 0 for category in CATEGORIES) for split in SPLIT_NAMES),
            "normalized_source_duplicate_issue_ids": duplicate_source_ids,
            "normalized_source_duplicate_title_body_pairs": duplicate_title_body_pairs,
            "cross_split_overlap": cross_split_overlap,
            "repository_overlap_zero": all(value["repository_overlap_count"] == 0 for value in cross_split_overlap.values()),
            "source_issue_id_overlap_zero": all(value["source_issue_id_overlap_count"] == 0 for value in cross_split_overlap.values()),
            "title_body_overlap_zero": all(value["title_body_overlap_count"] == 0 for value in cross_split_overlap.values()),
        },
        "freeze_statement": "The test split is frozen and must not be used for training, hyperparameter selection, prompt selection, or other development decisions.",
        "secondary_random_benchmark_created": False,
        "model_downloaded": False,
        "tokenization_performed": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Freeze the approved split and print its verification summary."""
    manifest = freeze_splits()
    print("Final repository-held-out split frozen")
    for split in SPLIT_NAMES:
        details = manifest["splits"][split]
        print(f"{split}: {details['row_count']} rows, {details['repository_count']} repositories, sha256={details['sha256']}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
