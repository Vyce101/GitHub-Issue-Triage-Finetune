"""Parse model outputs under the canonical issue-classification JSON contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


TARGET_CATEGORIES = ("bug", "feature", "documentation", "question_support")
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"


@dataclass(frozen=True)
class ClassificationParseResult:
    """Describe the normalized output and the reason it was accepted or rejected."""

    schema_valid: bool
    predicted_category: str | None
    normalized_output: str
    empty_think_wrapper_stripped: bool
    nonempty_thinking_content: bool
    parse_error: str | None


class _DuplicateKeyError(ValueError):
    """Signal a JSON object containing the same key more than once."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise _DuplicateKeyError(f"duplicate key: {key!r}")
        parsed[key] = value
    return parsed


def _normalize_empty_think_wrapper(raw_output: str) -> tuple[str, bool, bool, str | None]:
    """Remove one empty leading Qwen wrapper and reject non-empty wrapper content."""
    if not isinstance(raw_output, str):
        return "", False, False, "type_error: model output must be text"

    normalized = raw_output.strip()
    if not normalized.startswith(THINKING_OPEN):
        return normalized, False, False, None

    closing_position = normalized.find(THINKING_CLOSE, len(THINKING_OPEN))
    if closing_position < 0:
        return normalized, False, False, "thinking_error: missing closing </think>"

    thought_content = normalized[len(THINKING_OPEN) : closing_position]
    if thought_content.strip():
        return normalized, False, True, "thinking_error: non-empty thinking content"

    remainder = normalized[closing_position + len(THINKING_CLOSE) :].strip()
    return remainder, True, False, None


def parse_classification_output(raw_output: str) -> ClassificationParseResult:
    """Apply the strict classification-output contract to one model response."""
    normalized_output, wrapper_stripped, nonempty_thinking, normalization_error = _normalize_empty_think_wrapper(raw_output)
    if normalization_error:
        return ClassificationParseResult(
            schema_valid=False,
            predicted_category=None,
            normalized_output=normalized_output,
            empty_think_wrapper_stripped=wrapper_stripped,
            nonempty_thinking_content=nonempty_thinking,
            parse_error=normalization_error,
        )

    try:
        parsed = json.loads(normalized_output, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKeyError as error:
        return ClassificationParseResult(
            schema_valid=False,
            predicted_category=None,
            normalized_output=normalized_output,
            empty_think_wrapper_stripped=wrapper_stripped,
            nonempty_thinking_content=nonempty_thinking,
            parse_error=f"json_schema_error: {error}",
        )
    except json.JSONDecodeError as error:
        return ClassificationParseResult(
            schema_valid=False,
            predicted_category=None,
            normalized_output=normalized_output,
            empty_think_wrapper_stripped=wrapper_stripped,
            nonempty_thinking_content=nonempty_thinking,
            parse_error=f"json_decode_error: {error.msg}",
        )

    if not isinstance(parsed, dict) or set(parsed) != {"type"}:
        return ClassificationParseResult(
            schema_valid=False,
            predicted_category=None,
            normalized_output=normalized_output,
            empty_think_wrapper_stripped=wrapper_stripped,
            nonempty_thinking_content=nonempty_thinking,
            parse_error="schema_error: expected exactly one type field",
        )

    predicted_category = parsed["type"]
    if predicted_category not in TARGET_CATEGORIES:
        return ClassificationParseResult(
            schema_valid=False,
            predicted_category=None,
            normalized_output=normalized_output,
            empty_think_wrapper_stripped=wrapper_stripped,
            nonempty_thinking_content=nonempty_thinking,
            parse_error="schema_error: type is not an approved category",
        )

    return ClassificationParseResult(
        schema_valid=True,
        predicted_category=predicted_category,
        normalized_output=normalized_output,
        empty_think_wrapper_stripped=wrapper_stripped,
        nonempty_thinking_content=nonempty_thinking,
        parse_error=None,
    )
