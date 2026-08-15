"""Build the exact zero-shot and few-shot chat prompts without raw labels."""

from __future__ import annotations

from typing import Any

from .config import SYSTEM_INSTRUCTION


def _issue_content(row: dict[str, Any], heading: str) -> str:
    """Render only title and body for a model input."""
    return f"{heading}\nTitle: {row['title']}\nBody:\n{row['body']}"


def zero_shot_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the fixed zero-shot classification conversation."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": _issue_content(row, "Classify this new GitHub issue:")},
    ]


def _demonstration_user_message(row: dict[str, Any], number: int) -> str:
    """Render one few-shot user demonstration without source labels."""
    return _issue_content(row, f"Demonstration {number}:")


def few_shot_messages(
    row: dict[str, Any],
    few_shot_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the fixed few-shot conversation with eight train-only examples."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for number, example in enumerate(few_shot_rows, start=1):
        messages.append({"role": "user", "content": _demonstration_user_message(example, number)})
        messages.append({"role": "assistant", "content": f'{{"type":"{example["target_category"]}"}}'})
    messages.append({"role": "user", "content": _issue_content(row, "Classify this new GitHub issue:")})
    return messages


def prompt_definition(few_shot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return reproducible prompt definitions and demonstration content."""
    return {
        "output_schema": {"type": "bug"},
        "system_instruction": SYSTEM_INSTRUCTION,
        "zero_shot_message_roles": ["system", "user"],
        "few_shot_message_roles": ["system"] + [role for _ in few_shot_rows for role in ("user", "assistant")] + ["user"],
        "few_shot_examples": [
            {
                "example_number": number,
                "issue_id": row["issue_id"],
                "repository": row["repository"],
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "target_category": row["target_category"],
                "title": row["title"],
                "body": row["body"],
            }
            for number, row in enumerate(few_shot_rows, start=1)
        ],
        "raw_github_labels_in_model_inputs": False,
        "chain_of_thought_requested": False,
    }
