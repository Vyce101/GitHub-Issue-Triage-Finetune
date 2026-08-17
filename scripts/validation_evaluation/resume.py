"""Validate and repair only the trailing write of a sequential JSONL artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SequentialPrefixResult:
    """Describe a validated artifact prefix and any recoverable trailing repair."""

    records: list[dict[str, Any]]
    discarded_trailing_bytes: int
    added_trailing_newline: bool


_IDENTITY_FIELDS = (
    "evaluation_index",
    "issue_id",
    "repository",
    "source_split",
    "source_row_index",
    "expected_category",
)
_REQUIRED_FIELDS = _IDENTITY_FIELDS + (
    "status",
    "schema_valid",
    "predicted_category",
    "input_token_count",
    "full_input_token_count",
    "input_truncated",
    "output_token_count",
    "inference_time_seconds",
)


def _validate_record(record: dict[str, Any], row: dict[str, Any], evaluation_index: int) -> None:
    """Require a complete record aligned to the corresponding frozen row."""
    missing_fields = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing_fields:
        raise RuntimeError(
            f"Existing result is incomplete at evaluation index {evaluation_index}: {missing_fields}"
        )
    expected_values = {
        "evaluation_index": evaluation_index,
        "issue_id": row["issue_id"],
        "repository": row["repository"],
        "source_split": row["source_split"],
        "source_row_index": row["source_row_index"],
        "expected_category": row["target_category"],
    }
    mismatches = {
        field: (record.get(field), expected_value)
        for field, expected_value in expected_values.items()
        if record.get(field) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"Existing result identity mismatch at evaluation index {evaluation_index}: {mismatches}"
        )
    if record["status"] not in {"ok", "inference_error"}:
        raise RuntimeError(f"Existing result has an unknown status at evaluation index {evaluation_index}")


def load_sequential_prefix(
    output_path: Path | None,
    rows: list[dict[str, Any]],
) -> SequentialPrefixResult:
    """Load a valid sequential prefix and discard only a corrupt final JSONL write."""
    if output_path is None or not output_path.exists():
        return SequentialPrefixResult(records=[], discarded_trailing_bytes=0, added_trailing_newline=False)

    raw_bytes = output_path.read_bytes()
    if not raw_bytes:
        return SequentialPrefixResult(records=[], discarded_trailing_bytes=0, added_trailing_newline=False)

    records: list[dict[str, Any]] = []
    cursor = 0
    discarded_trailing_bytes = 0
    lines = raw_bytes.splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        line_start = cursor
        cursor += len(line)
        if not line.strip():
            continue
        try:
            decoded = line.decode("utf-8")
            record = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            trailing_bytes = raw_bytes[cursor:]
            if trailing_bytes.strip():
                raise RuntimeError(
                    f"Malformed non-trailing existing result at {output_path}:{line_number}: {error}"
                ) from error
            output_path.write_bytes(raw_bytes[:line_start])
            discarded_trailing_bytes = len(raw_bytes) - line_start
            break
        if not isinstance(record, dict):
            raise RuntimeError(f"Existing result at {output_path}:{line_number} is not a JSON object")
        if len(records) >= len(rows):
            raise RuntimeError(f"Existing result has more rows than the frozen evaluation split")
        _validate_record(record, rows[len(records)], len(records))
        records.append(record)

    if discarded_trailing_bytes:
        return SequentialPrefixResult(
            records=records,
            discarded_trailing_bytes=discarded_trailing_bytes,
            added_trailing_newline=False,
        )

    if not records:
        raise RuntimeError(f"Existing result at {output_path} contains no complete JSONL records")

    added_trailing_newline = not raw_bytes.endswith((b"\n", b"\r"))
    if added_trailing_newline:
        with output_path.open("ab") as output_file:
            output_file.write(b"\n")

    return SequentialPrefixResult(
        records=records,
        discarded_trailing_bytes=0,
        added_trailing_newline=added_trailing_newline,
    )
