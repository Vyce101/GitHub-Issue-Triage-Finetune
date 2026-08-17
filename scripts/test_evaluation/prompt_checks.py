"""Run TEST-specific checks for the frozen prompt and generation boundary."""

from __future__ import annotations

from typing import Any

from qlora_training.data import apply_chat_template_ids

from validation_evaluation.config import (
    CONDITION_BASE_FEW_SHOT,
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    MAX_INPUT_TOKENS,
    TOTAL_CONTEXT_TOKENS,
)
from validation_evaluation.generation import _prepare_prompt, _prompt_messages


def _render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Render a complete chat prompt with the frozen disabled-thinking template."""
    return apply_chat_template_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )


def _check_case(
    tokenizer: Any,
    row: dict[str, Any],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove one TEST prompt preserves structure and reserves all output tokens."""
    original_messages = _prompt_messages(row, condition, few_shot_rows)
    original_prompt_ids = _render_ids(tokenizer, original_messages)
    prepared = _prepare_prompt(tokenizer, row, -1, condition, few_shot_rows)
    if prepared.full_input_token_count != len(original_prompt_ids):
        raise AssertionError("TEST prompt full token count changed during preparation")
    if len(prepared.input_ids) > MAX_INPUT_TOKENS:
        raise AssertionError("TEST prompt exceeds the frozen input token budget")
    if len(prepared.input_ids) + 16 > TOTAL_CONTEXT_TOKENS:
        raise AssertionError("TEST prompt does not reserve all 16 generation tokens")
    if not prepared.generation_prompt_preserved or not prepared.generation_prompt_suffix_text:
        raise AssertionError("TEST prompt lost the assistant generation boundary")

    if not prepared.input_truncated:
        if prepared.input_ids != original_prompt_ids:
            raise AssertionError("Non-truncated TEST prompt changed token IDs")
        if prepared.final_messages != original_messages:
            raise AssertionError("Non-truncated TEST messages changed")
    else:
        old_right_truncated = original_prompt_ids[:MAX_INPUT_TOKENS]
        if prepared.input_ids == old_right_truncated:
            raise AssertionError("TEST truncation still right-truncates the rendered chat structure")
        if prepared.final_messages[:-1] != original_messages[:-1]:
            raise AssertionError("TEST system or frozen demonstration messages changed")
        if prepared.final_messages[-1]["content"] == original_messages[-1]["content"]:
            raise AssertionError("TEST issue content was not shortened")
        if not prepared.final_messages[-1]["content"].startswith("Classify this new GitHub issue:"):
            raise AssertionError("TEST issue heading was not preserved")
        if prepared.truncation_strategy == "none":
            raise AssertionError("TEST truncated prompt has no truncation strategy")

    return {
        "condition": condition,
        "repository": row["repository"],
        "issue_id": row["issue_id"],
        "source_row_index": row["source_row_index"],
        "full_input_tokens": prepared.full_input_token_count,
        "fed_input_tokens": len(prepared.input_ids),
        "input_truncated": prepared.input_truncated,
        "truncation_strategy": prepared.truncation_strategy,
        "generation_prompt_suffix": prepared.generation_prompt_suffix_text,
        "generation_prompt_preserved": prepared.generation_prompt_preserved,
        "output_reserve_tokens": TOTAL_CONTEXT_TOKENS - len(prepared.input_ids),
    }


def run_test_prompt_regression_checks(
    tokenizer: Any,
    test_rows: list[dict[str, Any]],
    few_shot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check non-truncated identity, truncated structure, and title fallback on TEST-shaped inputs."""
    cases = []
    for condition in (
        CONDITION_BASE_ZERO_SHOT,
        CONDITION_BASE_FEW_SHOT,
        CONDITION_FINE_TUNED_ZERO_SHOT,
    ):
        short_row = None
        truncated_row = None
        for row in test_rows:
            prepared = _prepare_prompt(tokenizer, row, -1, condition, few_shot_rows)
            if prepared.input_truncated and truncated_row is None:
                truncated_row = row
            if not prepared.input_truncated and short_row is None:
                short_row = row
            if short_row is not None and truncated_row is not None:
                break
        if short_row is None or truncated_row is None:
            raise AssertionError(f"TEST does not provide both prompt-length regression cases for {condition}")
        cases.append(_check_case(tokenizer, short_row, condition, few_shot_rows))
        cases.append(_check_case(tokenizer, truncated_row, condition, few_shot_rows))

    synthetic_title_row = dict(test_rows[0])
    synthetic_title_row["title"] = "Very long title " * 2_000
    synthetic_title_row["body"] = ""
    title_fallback = _prepare_prompt(
        tokenizer,
        synthetic_title_row,
        -1,
        CONDITION_BASE_ZERO_SHOT,
        few_shot_rows,
    )
    if title_fallback.truncation_strategy != "body_right_then_title_right":
        raise AssertionError("TEST title fallback truncation path was not exercised")
    if title_fallback.title_token_count_after >= title_fallback.title_token_count_before:
        raise AssertionError("TEST title fallback did not shorten the title")
    if not title_fallback.generation_prompt_preserved:
        raise AssertionError("TEST title fallback lost the generation boundary")

    return {
        "passed": True,
        "implementation": "prompt_preserving_v2",
        "maximum_prompt_input_tokens": MAX_INPUT_TOKENS,
        "generation_output_reserve_tokens": 16,
        "cases": cases,
        "title_fallback": {
            "passed": True,
            "strategy": title_fallback.truncation_strategy,
            "title_tokens_before_after": [
                title_fallback.title_token_count_before,
                title_fallback.title_token_count_after,
            ],
            "fed_input_tokens": len(title_fallback.input_ids),
            "generation_prompt_preserved": title_fallback.generation_prompt_preserved,
        },
    }
