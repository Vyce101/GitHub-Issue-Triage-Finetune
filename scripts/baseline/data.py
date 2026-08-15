"""Load frozen split records, analyze tokenizer lengths, and select train-only examples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CONTEXT_LIMITS,
    EVALUATION_BATCH_SIZE,
    FEW_SHOT_EXAMPLES_PER_CLASS,
    OUTPUT_RESERVE_TOKENS,
    PROMPT_DEVELOPMENT_SIZE_PER_CLASS,
    PROMPT_DEVELOPMENT_MAX_ZERO_SHOT_INPUT_TOKENS,
    TARGET_CATEGORIES,
)
from .prompts import zero_shot_messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL split without modifying its records."""
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


def issue_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return a stable source identity for a normalized issue."""
    return str(row["source_split"]), int(row["source_row_index"])


def issue_id_sort_key(row: dict[str, Any]) -> tuple[int, Any]:
    """Sort numeric issue IDs numerically and other IDs lexically."""
    issue_id = row["issue_id"]
    if isinstance(issue_id, int) and not isinstance(issue_id, bool):
        return 0, issue_id
    return 1, str(issue_id)


def title_body_text(row: dict[str, Any]) -> str:
    """Represent title and body with the delimiter used by the prompts."""
    return f"{row['title']}\n\n{row['body']}"


def token_count(tokenizer: Any, text: str) -> int:
    """Count tokens without truncation or added special tokens."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return len(encoded["input_ids"])


def percentile_summary(values: list[int]) -> dict[str, int]:
    """Return the requested integer-rounded token length percentiles."""
    percentiles = np.percentile(values, [50, 75, 90, 95, 99])
    return {
        "median": int(round(percentiles[0])),
        "p75": int(round(percentiles[1])),
        "p90": int(round(percentiles[2])),
        "p95": int(round(percentiles[3])),
        "p99": int(round(percentiles[4])),
        "maximum": int(max(values)),
    }


def _prompt_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Count the exact chat-template input tokens reserved for a prompt."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    return len(encoded)


def token_length_analysis(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    zero_shot_messages_builder: Any,
) -> dict[str, Any]:
    """Analyze raw title/body lengths and exact zero-shot prompt limits."""
    content_lengths = [token_count(tokenizer, title_body_text(row)) for row in rows]
    prompt_lengths = [
        _prompt_token_count(tokenizer, zero_shot_messages_builder(row))
        for row in rows
    ]
    truncation_by_limit = {}
    for limit in CONTEXT_LIMITS:
        truncated = sum(
            prompt_length + OUTPUT_RESERVE_TOKENS > limit
            for prompt_length in prompt_lengths
        )
        truncation_by_limit[str(limit)] = {
            "issues_requiring_truncation": truncated,
            "percentage_requiring_truncation": round(100 * truncated / len(rows), 4),
            "definition": "Exact zero-shot chat-template input tokens plus a fixed 16-token output reserve exceed the candidate limit.",
        }
    overheads = [prompt - content for prompt, content in zip(prompt_lengths, content_lengths)]
    return {
        "row_count": len(rows),
        "title_body_token_lengths": percentile_summary(content_lengths),
        "zero_shot_prompt_input_token_lengths": percentile_summary(prompt_lengths),
        "zero_shot_instruction_and_template_overhead_tokens": percentile_summary(overheads),
        "candidate_total_input_limits": truncation_by_limit,
        "min_content_tokens": min(content_lengths),
        "max_content_tokens": max(content_lengths),
        "average_zero_shot_prompt_input_tokens": round(sum(prompt_lengths) / len(prompt_lengths), 4),
    }


def _round_robin_repository_selection(
    rows: list[dict[str, Any]],
    category: str,
    target_count: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Select a fixed count by cycling through repositories in lexical order."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["target_category"] == category
            and _prompt_token_count(tokenizer, zero_shot_messages(row))
            <= PROMPT_DEVELOPMENT_MAX_ZERO_SHOT_INPUT_TOKENS
        ):
            grouped[str(row["repository"])].append(row)
    for repository_rows in grouped.values():
        repository_rows.sort(key=lambda row: (issue_id_sort_key(row), issue_key(row)))
    selected = []
    repository_names = sorted(grouped)
    offset = 0
    while len(selected) < target_count:
        added_this_round = False
        for repository in repository_names:
            repository_rows = grouped[repository]
            if offset < len(repository_rows):
                selected.append(repository_rows[offset])
                added_this_round = True
                if len(selected) == target_count:
                    break
        if not added_this_round:
            raise ValueError(f"Not enough train rows for category {category!r}")
        offset += 1
    return selected


def select_prompt_development_rows(train_rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Select exactly 100 train-only examples per approved category."""
    selected = []
    for category in TARGET_CATEGORIES:
        selected.extend(
            _round_robin_repository_selection(
                train_rows,
                category,
                PROMPT_DEVELOPMENT_SIZE_PER_CLASS,
                tokenizer,
            )
        )
    return sorted(selected, key=lambda row: (row["target_category"], issue_key(row)))


def select_few_shot_rows(
    train_rows: list[dict[str, Any]],
    prompt_development_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Select short, cross-repository demonstrations outside the development set."""
    excluded = {issue_key(row) for row in prompt_development_rows}
    selected = []
    for category in TARGET_CATEGORIES:
        candidates = [
            row for row in train_rows
            if row["target_category"] == category and issue_key(row) not in excluded
        ]
        candidates.sort(
            key=lambda row: (
                token_count(tokenizer, title_body_text(row)),
                len(str(row["body"])),
                str(row["repository"]),
                issue_id_sort_key(row),
                issue_key(row),
            )
        )
        category_selected = []
        used_repositories = set()
        for row in candidates:
            if row["repository"] in used_repositories:
                continue
            category_selected.append(row)
            used_repositories.add(row["repository"])
            if len(category_selected) == FEW_SHOT_EXAMPLES_PER_CLASS:
                break
        if len(category_selected) < FEW_SHOT_EXAMPLES_PER_CLASS:
            for row in candidates:
                if issue_key(row) not in {issue_key(item) for item in category_selected}:
                    category_selected.append(row)
                if len(category_selected) == FEW_SHOT_EXAMPLES_PER_CLASS:
                    break
        if len(category_selected) != FEW_SHOT_EXAMPLES_PER_CLASS:
            raise ValueError(f"Not enough train rows for few-shot category {category!r}")
        selected.extend(category_selected)
    return selected


def selection_record(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep reproducibility metadata without copying original GitHub labels."""
    return [
        {
            "issue_id": row["issue_id"],
            "repository": row["repository"],
            "source_split": row["source_split"],
            "source_row_index": row["source_row_index"],
            "target_category": row["target_category"],
        }
        for row in rows
    ]
