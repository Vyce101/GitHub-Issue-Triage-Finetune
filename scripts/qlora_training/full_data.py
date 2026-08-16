"""Build the full frozen-split prompt-completion datasets for QLoRA training."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from baseline.data import read_jsonl
from baseline.prompts import zero_shot_messages

from .data import apply_chat_template_ids, completion_text


@dataclass(frozen=True)
class VerificationSample:
    """Keep one rendered row's boundaries for the final masking preflight."""

    dataset_index: int
    issue_id: Any
    repository: str
    source_split: str
    target_category: str
    original_prompt_token_count: int
    fed_prompt_token_count: int
    target_token_count: int
    original_full_sequence_token_count: int
    fed_sequence_token_count: int
    truncated: bool
    target_preserved: bool


@dataclass(frozen=True)
class SplitDatasetBuild:
    """Return a tokenized dataset and compact rendering statistics."""

    dataset: Dataset
    stats: dict[str, Any]
    verification_samples: list[VerificationSample]


def load_split_rows(path: Any, expected_source_split: str) -> list[dict[str, Any]]:
    """Load one frozen file and reject rows from another original source split."""
    rows = read_jsonl(path)
    for row_index, row in enumerate(rows):
        if row.get("source_split") != expected_source_split:
            raise ValueError(
                f"Row {row_index} in {path} has source_split={row.get('source_split')!r}; "
                f"expected {expected_source_split!r}"
            )
    return rows


def _decode_token_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token prefix without cleaning whitespace that belongs to the issue."""
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize issue text without adding model-level special tokens."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def _batch_token_ids(encoded: Any) -> list[list[int]]:
    """Normalize batched chat-template output to lists of integer token IDs."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], int):
        encoded = [encoded]
    return [[int(token_id) for token_id in sequence] for sequence in encoded]


def _render_rows_batch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    chat_template_kwargs: dict[str, Any],
) -> list[tuple[list[int], list[int], list[int]]]:
    """Render a bounded batch of rows with two tokenizer calls instead of one call per row."""
    prompts = [zero_shot_messages(row) for row in rows]
    full_conversations = [
        prompt + [{"role": "assistant", "content": completion_text(row["target_category"])}]
        for prompt, row in zip(prompts, rows)
    ]
    try:
        prompt_encoded = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        full_encoded = tokenizer.apply_chat_template(
            full_conversations,
            tokenize=True,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
    except TypeError:
        prompt_encoded = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True)
        full_encoded = tokenizer.apply_chat_template(full_conversations, tokenize=True, add_generation_prompt=False)
    prompt_rows = _batch_token_ids(prompt_encoded)
    full_rows = _batch_token_ids(full_encoded)
    if len(prompt_rows) != len(rows) or len(full_rows) != len(rows):
        raise ValueError("Batched tokenizer output count does not match the dataset batch")
    rendered = []
    for row, prompt_ids, full_ids in zip(rows, prompt_rows, full_rows):
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Tokenizer prompt prefix mismatch for {row['repository']} issue {row['issue_id']}"
            )
        target_ids = full_ids[len(prompt_ids) :]
        if not target_ids:
            raise ValueError(f"Empty assistant target for {row['repository']} issue {row['issue_id']}")
        rendered.append((prompt_ids, full_ids, target_ids))
    return rendered


def _render_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    chat_template_kwargs: dict[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    """Render the exact baseline conversation and return prompt/full/target IDs."""
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
        raise ValueError(f"Empty assistant target for {row['repository']} issue {row['issue_id']}")
    return prompt_ids, full_ids, target_ids


def _candidate_with_prefix(row: dict[str, Any], field: str, tokenizer: Any, prefix_length: int) -> dict[str, Any]:
    """Return a row with one issue-prompt field truncated from the right."""
    candidate = dict(row)
    source_text = str(row[field])
    source_ids = _text_token_ids(tokenizer, source_text)
    candidate[field] = _decode_token_prefix(tokenizer, source_ids[:prefix_length])
    return candidate


def _find_fitting_prefix(
    row: dict[str, Any],
    field: str,
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
    chat_template_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[int], list[int], list[int]] | None:
    """Find the longest right-truncated field that fits before the target."""
    source_ids = _text_token_ids(tokenizer, str(row[field]))
    low = 0
    high = len(source_ids)
    best: tuple[dict[str, Any], list[int], list[int], list[int]] | None = None
    while low <= high:
        prefix_length = (low + high) // 2
        candidate = _candidate_with_prefix(row, field, tokenizer, prefix_length)
        prompt_ids, full_ids, target_ids = _render_row(
            candidate,
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        if len(prompt_ids) <= max_prompt_tokens:
            best = (candidate, prompt_ids, full_ids, target_ids)
            low = prefix_length + 1
        else:
            high = prefix_length - 1
    return best


def _render_with_prompt_truncation(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    chat_template_kwargs: dict[str, Any],
    initial_render: tuple[list[int], list[int], list[int]] | None = None,
) -> tuple[list[int], list[int], list[int], int, int, int, int]:
    """Truncate only the issue prompt while preserving the complete target."""
    if initial_render is None:
        initial_render = _render_row(row, tokenizer, chat_template_kwargs=chat_template_kwargs)
    prompt_ids, full_ids, target_ids = initial_render
    if len(full_ids) <= max_length:
        return prompt_ids, full_ids, target_ids, len(prompt_ids), len(full_ids), len(prompt_ids), len(full_ids)

    prompt_budget = max_length - len(target_ids)
    if prompt_budget <= 0:
        raise ValueError(
            f"Assistant target for {row['repository']} issue {row['issue_id']} exceeds max_length={max_length}"
        )

    fitted = _find_fitting_prefix(
        row,
        "body",
        tokenizer,
        max_prompt_tokens=prompt_budget,
        chat_template_kwargs=chat_template_kwargs,
    )
    if fitted is None:
        body_empty = dict(row)
        body_empty["body"] = ""
        fitted = _find_fitting_prefix(
            body_empty,
            "title",
            tokenizer,
            max_prompt_tokens=prompt_budget,
            chat_template_kwargs=chat_template_kwargs,
        )
    if fitted is None:
        raise ValueError(
            f"Issue prompt cannot fit before the assistant target for {row['repository']} issue {row['issue_id']}"
        )

    _, fitted_prompt_ids, fitted_full_ids, fitted_target_ids = fitted
    if fitted_target_ids != target_ids:
        raise ValueError(
            f"Prompt truncation changed the assistant target for {row['repository']} issue {row['issue_id']}"
        )
    if len(fitted_full_ids) > max_length:
        raise AssertionError("Right-truncated issue prompt still exceeds max_length")
    if fitted_full_ids[len(fitted_prompt_ids) :] != target_ids:
        raise AssertionError("Target is not the final preserved region after prompt truncation")
    return (
        fitted_prompt_ids,
        fitted_full_ids,
        fitted_target_ids,
        len(fitted_prompt_ids),
        len(fitted_full_ids),
        len(prompt_ids),
        len(full_ids),
    )


def build_tokenized_split_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    expected_split: str,
    categories: tuple[str, ...],
    max_length: int,
    chat_template_kwargs: dict[str, Any],
) -> SplitDatasetBuild:
    """Render every row into pre-tokenized input IDs and an explicit completion mask."""
    if not rows:
        raise ValueError(f"The {expected_split} split is empty")

    input_ids_rows: list[list[int]] = []
    completion_masks: list[list[int]] = []
    class_counts = Counter()
    truncated_rows = 0
    original_token_total = 0
    fed_token_total = 0
    verification_samples: list[VerificationSample] = []

    batch_size = 512
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        initial_renders = _render_rows_batch(
            batch_rows,
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        for batch_offset, (row, initial_render) in enumerate(zip(batch_rows, initial_renders)):
            dataset_index = batch_start + batch_offset
            category = row.get("target_category")
            if category not in categories:
                raise ValueError(f"Unexpected target category {category!r} in {expected_split} row {dataset_index}")
            (
                prompt_ids,
                full_ids,
                target_ids,
                fed_prompt_length,
                fed_sequence_length,
                original_prompt_length,
                original_sequence_length,
            ) = _render_with_prompt_truncation(
                row,
                tokenizer,
                max_length=max_length,
                chat_template_kwargs=chat_template_kwargs,
                initial_render=initial_render,
            )
            if fed_sequence_length > max_length:
                raise AssertionError("A rendered sequence exceeds max_length")
            target_start = fed_prompt_length
            sequence_ids = full_ids
            completion_mask = [0] * target_start + [1] * len(target_ids)
            if len(completion_mask) != len(sequence_ids):
                raise AssertionError("Completion mask length does not match input IDs")
            target_preserved = sequence_ids[target_start:] == target_ids and bool(target_ids)
            if not target_preserved:
                raise AssertionError("Completion target was removed or changed by prompt truncation")

            was_truncated = fed_prompt_length < original_prompt_length
            truncated_rows += int(was_truncated)
            original_token_total += original_sequence_length
            fed_token_total += fed_sequence_length
            class_counts[category] += 1
            input_ids_rows.append(sequence_ids)
            completion_masks.append(completion_mask)

            if len(verification_samples) < 2 or (was_truncated and len(verification_samples) < 3):
                verification_samples.append(
                    VerificationSample(
                        dataset_index=dataset_index,
                        issue_id=row["issue_id"],
                        repository=str(row["repository"]),
                        source_split=str(row["source_split"]),
                        target_category=str(category),
                        original_prompt_token_count=original_prompt_length,
                        fed_prompt_token_count=fed_prompt_length,
                        target_token_count=len(target_ids),
                        original_full_sequence_token_count=original_sequence_length,
                        fed_sequence_token_count=fed_sequence_length,
                        truncated=was_truncated,
                        target_preserved=target_preserved,
                    )
                )

    dataset = Dataset.from_dict({"input_ids": input_ids_rows, "completion_mask": completion_masks})
    return SplitDatasetBuild(
        dataset=dataset,
        stats={
            "source_split": expected_split,
            "row_count": len(rows),
            "class_counts": {category: class_counts[category] for category in categories},
            "original_token_total": original_token_total,
            "fed_token_total": fed_token_total,
            "average_original_sequence_tokens": round(original_token_total / len(rows), 4),
            "average_fed_sequence_tokens": round(fed_token_total / len(rows), 4),
            "maximum_fed_sequence_tokens": max(len(row) for row in input_ids_rows),
            "truncated_row_count": truncated_rows,
            "truncated_row_percentage": round(100 * truncated_rows / len(rows), 6),
            "target_removed_count": 0,
            "target_preserved_for_every_row": True,
            "prompt_truncated_only": True,
        },
        verification_samples=verification_samples,
    )


def expected_optimizer_steps(row_count: int, micro_batch_size: int, gradient_accumulation_steps: int) -> int:
    """Compute optimizer updates for one epoch without dropping the final batch."""
    updates_per_batch = micro_batch_size * gradient_accumulation_steps
    return math.ceil(row_count / updates_per_batch)
