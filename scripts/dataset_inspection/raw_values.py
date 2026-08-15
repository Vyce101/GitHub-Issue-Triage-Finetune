"""Helpers for preserving and serializing raw dataset values."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Iterable

from .config import EXAMPLE_LIMIT, SHORT_BODY_LIMIT


def json_safe(value: Any) -> Any:
    """Convert common dataset values into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except Exception:
            pass
    return str(value)


def value_key(value: Any) -> str:
    """Create a stable key while preserving the raw value in the output."""
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)


def display_value(value: Any) -> str:
    """Create a compact human-readable representation of a raw value."""
    safe_value = json_safe(value)
    if isinstance(safe_value, str):
        return safe_value
    return json.dumps(safe_value, ensure_ascii=False, sort_keys=True)


def is_null(value: Any) -> bool:
    """Identify null-like scalar values without treating empty text as null."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def text_value(value: Any) -> str:
    """Return a field as text for length and empty-value checks."""
    if is_null(value):
        return ""
    return str(value)


def is_empty_text(value: Any) -> bool:
    """Identify missing or whitespace-only text."""
    return not text_value(value).strip()


def parse_serialized_label_list(value: str) -> tuple[list[Any] | None, str | None]:
    """Detect JSON or Python-literal list strings without changing raw labels."""
    stripped = value.strip()
    if not stripped or stripped[0] not in "[(":
        return None, None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed, "json_serialized_list"
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, (list, tuple)):
            return list(parsed), "python_literal_list"
    except (ValueError, SyntaxError):
        pass

    return None, None


def raw_label_items(value: Any) -> tuple[list[Any], str]:
    """Return raw label items and the observed top-level storage representation."""
    if is_null(value):
        return [], "null"
    if isinstance(value, list):
        return value, "list"
    if isinstance(value, tuple):
        return list(value), "tuple"
    if isinstance(value, set):
        return list(value), "set"
    if isinstance(value, str):
        if not value.strip():
            return [], "empty_string"
        serialized_items, serialized_type = parse_serialized_label_list(value)
        if serialized_items is not None:
            return serialized_items, serialized_type or "serialized_list"
        return [value], "string"
    return [value], type(value).__name__


def meaningful_label_items(items: Iterable[Any]) -> list[Any]:
    """Exclude only null or blank label entries from label counts."""
    return [item for item in items if not is_null(item) and not is_empty_text(item)]


def unique_label_items(items: Iterable[Any]) -> list[Any]:
    """Preserve raw label order while counting each raw label once per issue."""
    unique_items: list[Any] = []
    seen: set[str] = set()
    for item in meaningful_label_items(items):
        key = value_key(item)
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return unique_items


def resolve_column(columns: list[str], candidates: tuple[str, ...], keyword: str | None = None) -> str | None:
    """Resolve a field without assuming the dataset's exact column spelling."""
    lowercase_columns = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lowercase_columns:
            return lowercase_columns[candidate.casefold()]
    if keyword is not None:
        for column in columns:
            if keyword in column.casefold():
                return column
    return None


def shorten_text(value: Any, limit: int = SHORT_BODY_LIMIT) -> str:
    """Collapse whitespace and cap example text for compact output."""
    compact = re.sub(r"\s+", " ", text_value(value)).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def numeric_stats(values: list[int]) -> dict[str, int | float | None]:
    """Calculate reproducible character-length statistics."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None, "p95": None, "p99": None}

    ordered = sorted(values)

    def percentile(percent: float) -> int:
        index = round((len(ordered) - 1) * percent)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def choose_examples(records: list[dict[str, Any]], predicate, limit: int = EXAMPLE_LIMIT) -> list[dict[str, Any]]:
    """Choose deterministic evenly spaced examples from matching records."""
    candidates = [record for record in records if predicate(record)]
    if len(candidates) <= limit:
        return candidates
    positions = [round(index * (len(candidates) - 1) / (limit - 1)) for index in range(limit)]
    return [candidates[position] for position in positions]


def example_payload(record: dict[str, Any], repository_column: str | None, title_column: str | None, body_column: str | None) -> dict[str, Any]:
    """Serialize one issue example without exposing the full body."""
    return {
        "split": record["split"],
        "row_index": record["row_index"],
        "repository": json_safe(record["values"].get(repository_column)) if repository_column else None,
        "title": text_value(record["values"].get(title_column)) if title_column else "",
        "body_shortened": shorten_text(record["values"].get(body_column)) if body_column else "",
        "raw_labels": json_safe(record["raw_label_items"]),
    }
