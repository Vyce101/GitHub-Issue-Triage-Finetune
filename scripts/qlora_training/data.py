"""Select short train-only sanity examples and build prompt-completion records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from baseline.data import issue_id_sort_key, issue_key, read_jsonl
from baseline.prompts import zero_shot_messages


@dataclass(frozen=True)
class SanityExample:
    """One selected train example with the rendered token boundaries used for preflight checks."""

    row: dict[str, Any]
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    prompt_ids: list[int]
    full_ids: list[int]
    target_ids: list[int]

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_ids)

    @property
    def full_sequence_token_count(self) -> int:
        return len(self.full_ids)

    @property
    def target_token_count(self) -> int:
        return len(self.target_ids)

    @property
    def completion_text(self) -> str:
        return self.completion[0]["content"]


def _token_ids(encoded: Any) -> list[int]:
    """Normalize tokenizer output to one flat integer token sequence."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one tokenized conversation")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def apply_chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Render one conversation with the locked tokenizer and disabled thinking."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **chat_template_kwargs,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    return _token_ids(encoded)


def completion_text(category: str) -> str:
    """Return the exact normalized JSON completion used as the learning target."""
    return json.dumps({"type": category}, ensure_ascii=False, separators=(",", ":"))


def render_sanity_example(
    row: dict[str, Any],
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any],
) -> SanityExample:
    """Render a normalized row and preserve the prompt/completion token boundary."""
    prompt = zero_shot_messages(row)
    completion = [{"role": "assistant", "content": completion_text(row["target_category"])}]
    prompt_ids = apply_chat_template_ids(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_ids = apply_chat_template_ids(
        tokenizer,
        prompt + completion,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Tokenizer prompt prefix mismatch for {row['repository']} issue {row['issue_id']}"
        )
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError(f"Empty completion token sequence for {row['repository']} issue {row['issue_id']}")
    return SanityExample(
        row=row,
        prompt=prompt,
        completion=completion,
        prompt_ids=prompt_ids,
        full_ids=full_ids,
        target_ids=target_ids,
    )


def _candidate_sort_key(example: SanityExample) -> tuple[Any, ...]:
    """Sort shortest candidates deterministically with repository and source tie-breakers."""
    row = example.row
    return (
        example.full_sequence_token_count,
        example.prompt_token_count,
        len(str(row["body"])),
        str(row["repository"]),
        issue_id_sort_key(row),
        issue_key(row),
    )


def select_sanity_examples(
    train_path: Path,
    tokenizer: Any,
    *,
    categories: tuple[str, ...],
    examples_per_category: int,
    max_sequence_length: int,
    short_sequence_token_ceiling: int,
    chat_template_kwargs: dict[str, Any],
) -> list[SanityExample]:
    """Select exactly the shortest deterministic train-only examples per category."""
    train_rows = read_jsonl(train_path)
    candidates_by_category: dict[str, list[SanityExample]] = {category: [] for category in categories}
    for row in train_rows:
        category = row["target_category"]
        if category not in candidates_by_category:
            continue
        example = render_sanity_example(row, tokenizer, chat_template_kwargs)
        if example.full_sequence_token_count <= short_sequence_token_ceiling:
            candidates_by_category[category].append(example)

    selected: list[SanityExample] = []
    for category in categories:
        candidates = sorted(candidates_by_category[category], key=_candidate_sort_key)
        if len(candidates) < examples_per_category:
            raise ValueError(
                f"Only {len(candidates)} short train candidates available for {category}; "
                f"need {examples_per_category}"
            )
        category_examples = candidates[:examples_per_category]
        for example in category_examples:
            if example.full_sequence_token_count > max_sequence_length:
                raise ValueError("Selected sanity example exceeds the configured maximum sequence length")
        selected.extend(category_examples)

    if len(selected) != len(categories) * examples_per_category:
        raise AssertionError("Sanity example count is not category_count * examples_per_category")
    return selected


def sanity_dataset_records(
    examples: list[SanityExample],
    *,
    chat_template_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert selected examples to the prompt-completion schema consumed by TRL."""
    records = []
    for example in examples:
        row = example.row
        records.append(
            {
                "prompt": example.prompt,
                "completion": example.completion,
                "chat_template_kwargs": chat_template_kwargs,
                "issue_id": row["issue_id"],
                "repository": row["repository"],
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "target_category": row["target_category"],
            }
        )
    return records


def to_dataset(records: list[dict[str, Any]]) -> Dataset:
    """Build the in-memory 64-example dataset without writing processed data."""
    return Dataset.from_list(records)


def selection_report(examples: list[SanityExample]) -> dict[str, Any]:
    """Return compact reproducibility metadata for the selected examples."""
    return {
        "row_count": len(examples),
        "class_counts": {
            category: sum(example.row["target_category"] == category for example in examples)
            for category in ("bug", "feature", "documentation", "question_support")
        },
        "maximum_full_sequence_tokens": max(example.full_sequence_token_count for example in examples),
        "examples": [
            {
                "issue_id": example.row["issue_id"],
                "repository": example.row["repository"],
                "source_split": example.row["source_split"],
                "source_row_index": example.row["source_row_index"],
                "target_category": example.row["target_category"],
                "prompt_token_count": example.prompt_token_count,
                "target_token_count": example.target_token_count,
                "full_sequence_token_count": example.full_sequence_token_count,
            }
            for example in examples
        ],
    }
