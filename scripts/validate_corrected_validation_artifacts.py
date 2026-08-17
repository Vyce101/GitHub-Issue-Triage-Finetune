"""Verify corrected validation artifacts preserve unaffected rows and full coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITIONS = ("base_zero_shot", "base_few_shot", "fine_tuned_zero_shot")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def validate_condition(
    condition: str,
    historical_path: Path,
    corrected_path: Path,
    expected_regenerated_count: int,
) -> dict[str, Any]:
    historical_rows = load_jsonl(historical_path)
    corrected_rows = load_jsonl(corrected_path)
    if len(historical_rows) != len(corrected_rows):
        raise ValueError(f"{condition} row count changed")

    regenerated_indices: list[int] = []
    unaffected_mismatches: list[int] = []
    for expected_index, (historical, corrected) in enumerate(zip(historical_rows, corrected_rows)):
        if corrected.get("evaluation_index") != expected_index:
            raise ValueError(f"{condition} is not sequential at index {expected_index}")
        was_truncated = bool(historical.get("input_truncated"))
        was_regenerated = bool(corrected.get("regenerated_under_prompt_preserving_truncation"))
        if was_truncated:
            if not was_regenerated:
                raise ValueError(f"{condition} missing regeneration marker at index {expected_index}")
            regenerated_indices.append(expected_index)
        else:
            if was_regenerated or corrected != historical:
                unaffected_mismatches.append(expected_index)

    if len(regenerated_indices) != expected_regenerated_count:
        raise ValueError(
            f"{condition} regenerated {len(regenerated_indices)} rows, "
            f"expected {expected_regenerated_count}"
        )
    if unaffected_mismatches:
        raise ValueError(f"{condition} changed unaffected rows: {unaffected_mismatches[:5]}")
    return {
        "total_rows": len(corrected_rows),
        "unique_sequential_indices": True,
        "regenerated_row_count": len(regenerated_indices),
        "regenerated_indices_first": regenerated_indices[:5],
        "regenerated_indices_last": regenerated_indices[-5:],
        "unaffected_rows_byte_equivalent_json_objects": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    expected_counts = {
        "base_zero_shot": 1280,
        "base_few_shot": 2556,
        "fine_tuned_zero_shot": 1280,
    }
    results = {
        condition: validate_condition(
            condition,
            root / "results" / "validation_evaluation" / f"{condition}.jsonl",
            root / "results" / "validation_evaluation_corrected" / f"{condition}.jsonl",
            expected_counts[condition],
        )
        for condition in CONDITIONS
    }
    print(json.dumps({"status": "passed", "conditions": results}, indent=2))


if __name__ == "__main__":
    main()
