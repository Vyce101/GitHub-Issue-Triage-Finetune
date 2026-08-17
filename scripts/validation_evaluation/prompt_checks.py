"""Run tokenizer-only regression checks for prompt-preserving truncation."""

from __future__ import annotations

import json
from typing import Any

from classification.parser import parse_classification_output
from qlora_training.data import apply_chat_template_ids

from .config import (
    CONDITION_BASE_FEW_SHOT,
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    MAX_INPUT_TOKENS,
    TOTAL_CONTEXT_TOKENS,
)
from .generation import _prepare_prompt, _prompt_messages


def _parser_contract_check() -> dict[str, Any]:
    """Verify the canonical parser contract without changing parser behavior."""
    cases = {
        "plain_valid": ('{"type":"bug"}', True),
        "empty_think_wrapper_valid": ('<think></think>{"type":"bug"}', True),
        "nonempty_thinking_rejected": ('<think>reason</think>{"type":"bug"}', False),
        "extra_prose_rejected": ('Answer: {"type":"bug"}', False),
        "extra_key_rejected": ('{"type":"bug","extra":1}', False),
        "extra_object_rejected": ('{"type":"bug"}{"type":"feature"}', False),
        "unapproved_category_rejected": ('{"type":"other"}', False),
    }
    outcomes = {}
    for name, (raw_output, expected_valid) in cases.items():
        parsed = parse_classification_output(raw_output)
        if parsed.schema_valid != expected_valid:
            raise AssertionError(f"Parser regression check failed: {name}: {parsed}")
        outcomes[name] = {
            "schema_valid": parsed.schema_valid,
            "parse_error": parsed.parse_error,
        }
    return {"passed": True, "cases": outcomes}


def _check_case(
    tokenizer: Any,
    row: dict[str, Any],
    condition: str,
    few_shot_rows: list[dict[str, Any]],
    historical_record: dict[str, Any],
) -> dict[str, Any]:
    """Check one short or historically truncated prompt against the old path."""
    original_messages = _prompt_messages(row, condition, few_shot_rows)
    old_prompt_ids = apply_chat_template_ids(
        tokenizer,
        original_messages,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    prepared = _prepare_prompt(tokenizer, row, historical_record["evaluation_index"], condition, few_shot_rows)
    was_historically_truncated = bool(historical_record["input_truncated"])
    if len(old_prompt_ids) != prepared.full_input_token_count:
        raise AssertionError("Corrected prompt does not reproduce the old full rendered token count")
    if len(prepared.input_ids) > MAX_INPUT_TOKENS:
        raise AssertionError("Corrected prompt exceeds the input budget")
    if len(prepared.input_ids) + 16 > TOTAL_CONTEXT_TOKENS:
        raise AssertionError("Corrected prompt does not reserve all 16 generation tokens")
    if not prepared.generation_prompt_preserved or not prepared.generation_prompt_suffix_text:
        raise AssertionError("Corrected prompt did not preserve the generation boundary")

    if not was_historically_truncated:
        if prepared.input_ids != old_prompt_ids:
            raise AssertionError("A non-truncated prompt changed token IDs under the correction")
        if prepared.final_messages != original_messages:
            raise AssertionError("A non-truncated prompt changed rendered messages")
    else:
        old_right_truncated = old_prompt_ids[:MAX_INPUT_TOKENS]
        if prepared.input_ids == old_right_truncated:
            raise AssertionError("Corrected truncation still matches the old structural right-truncation")
        if prepared.final_messages[:-1] != original_messages[:-1]:
            raise AssertionError("System or frozen few-shot messages changed during truncation")
        if prepared.final_messages[-1]["content"] == original_messages[-1]["content"]:
            raise AssertionError("Historically truncated issue content was not shortened")
        if not prepared.final_messages[-1]["content"].startswith("Classify this new GitHub issue:"):
            raise AssertionError("Current issue prompt heading was not preserved")
        if prepared.truncation_strategy == "none":
            raise AssertionError("Historically truncated prompt has no truncation strategy")

    return {
        "condition": condition,
        "evaluation_index": historical_record["evaluation_index"],
        "issue_id": historical_record["issue_id"],
        "historically_truncated": was_historically_truncated,
        "old_full_prompt_tokens": len(old_prompt_ids),
        "old_right_truncated_prompt_tokens": min(len(old_prompt_ids), MAX_INPUT_TOKENS),
        "corrected_prompt_tokens": len(prepared.input_ids),
        "generation_prompt_suffix": prepared.generation_prompt_suffix_text,
        "truncation_strategy": prepared.truncation_strategy,
        "title_tokens_before_after": [
            prepared.title_token_count_before,
            prepared.title_token_count_after,
        ],
        "body_tokens_before_after": [
            prepared.body_token_count_before,
            prepared.body_token_count_after,
        ],
        "non_truncated_token_ids_identical_to_old_path": not was_historically_truncated,
        "structural_messages_preserved": True,
        "generation_prompt_preserved": True,
        "output_reserve_tokens": TOTAL_CONTEXT_TOKENS - len(prepared.input_ids),
    }


def run_prompt_regression_checks(
    tokenizer: Any,
    validation_rows: list[dict[str, Any]],
    few_shot_rows: list[dict[str, Any]],
    historical_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prove the corrected renderer preserves structure and unaffected token IDs."""
    cases = []
    for condition in (CONDITION_BASE_ZERO_SHOT, CONDITION_BASE_FEW_SHOT):
        records = historical_records[condition]
        short_record = next(record for record in records if not record["input_truncated"])
        truncated_record = next(record for record in records if record["input_truncated"])
        for record in (short_record, truncated_record):
            row = validation_rows[record["evaluation_index"]]
            cases.append(_check_case(tokenizer, row, condition, few_shot_rows, record))

    parser_contract = _parser_contract_check()
    synthetic_title_row = dict(validation_rows[0])
    synthetic_title_row["title"] = "Very long title " * 2_000
    synthetic_title_row["body"] = ""
    synthetic_title_prompt = _prepare_prompt(
        tokenizer,
        synthetic_title_row,
        -1,
        CONDITION_BASE_ZERO_SHOT,
        few_shot_rows,
    )
    if synthetic_title_prompt.truncation_strategy != "body_right_then_title_right":
        raise AssertionError("Title fallback truncation path was not exercised")
    if synthetic_title_prompt.title_token_count_after >= synthetic_title_prompt.title_token_count_before:
        raise AssertionError("Title fallback did not shorten the title")
    if len(synthetic_title_prompt.input_ids) > MAX_INPUT_TOKENS:
        raise AssertionError("Title fallback exceeded the input budget")
    return {
        "passed": True,
        "implementation": "prompt_preserving_v2",
        "maximum_prompt_input_tokens": MAX_INPUT_TOKENS,
        "generation_output_reserve_tokens": 16,
        "checks": {
            "short_zero_shot": cases[0],
            "truncated_zero_shot": cases[1],
            "short_few_shot": cases[2],
            "truncated_few_shot": cases[3],
        },
        "non_truncated_prompt_token_identity_verified": all(
            case["non_truncated_token_ids_identical_to_old_path"]
            for case in cases
            if not case["historically_truncated"]
        ),
        "parser_contract": parser_contract,
        "title_fallback": {
            "passed": True,
            "strategy": synthetic_title_prompt.truncation_strategy,
            "title_tokens_before_after": [
                synthetic_title_prompt.title_token_count_before,
                synthetic_title_prompt.title_token_count_after,
            ],
            "corrected_prompt_tokens": len(synthetic_title_prompt.input_ids),
            "generation_prompt_preserved": synthetic_title_prompt.generation_prompt_preserved,
        },
        "serialized_summary": json.dumps(cases, ensure_ascii=False, separators=(",", ":")),
    }
