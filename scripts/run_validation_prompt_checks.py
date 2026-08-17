"""Run tokenizer-only regression checks for the corrected validation prompt path."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from validation_evaluation.config import ADAPTER_PATH
from validation_evaluation.data import load_frozen_validation_rows, load_recorded_few_shot_rows
from validation_evaluation.prompt_checks import run_prompt_regression_checks


def _load_records(root: Path) -> dict[str, list[dict[str, object]]]:
    """Load the historical validation records without changing them."""
    result = {}
    for condition in ("base_zero_shot", "base_few_shot", "fine_tuned_zero_shot"):
        path = root / "results" / "validation_evaluation" / f"{condition}.jsonl"
        with path.open("r", encoding="utf-8") as source:
            result[condition] = [json.loads(line) for line in source if line.strip()]
    return result


def main() -> None:
    """Load only the tokenizer and run structural prompt checks."""
    root = Path(__file__).resolve().parents[1]
    validation_rows, _ = load_frozen_validation_rows()
    few_shot_rows, _ = load_recorded_few_shot_rows()
    historical_records = _load_records(root)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), local_files_only=True, use_fast=True)
    checks = run_prompt_regression_checks(
        tokenizer,
        validation_rows,
        few_shot_rows,
        historical_records,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
