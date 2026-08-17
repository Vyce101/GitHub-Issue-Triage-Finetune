"""Preserve chat structure while truncating only the current issue content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from qlora_training.data import apply_chat_template_ids


@dataclass(frozen=True)
class PromptTruncationResult:
    """Describe the original and structure-preserving rendered prompt."""

    original_prompt_ids: list[int]
    prompt_ids: list[int]
    original_messages: list[dict[str, str]]
    final_messages: list[dict[str, str]]
    full_input_token_count: int
    input_truncated: bool
    title_token_count_before: int
    title_token_count_after: int
    body_token_count_before: int
    body_token_count_after: int
    generation_prompt_preserved: bool
    generation_prompt_suffix_text: str
    truncation_strategy: str


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize content without adding chat or tokenizer special tokens."""
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _decode_prefix(tokenizer: Any, token_ids: list[int], count: int) -> str:
    """Decode a token prefix without normalizing whitespace or punctuation."""
    if count == 0:
        return ""
    prefix = token_ids[:count]
    try:
        return tokenizer.decode(
            prefix,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(prefix, skip_special_tokens=False)


def _apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Render chat text with the same disabled-thinking template contract."""
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if not isinstance(rendered, str):
        raise ValueError("Tokenizer returned non-text chat-template output")
    return rendered


def generation_prompt_suffix(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Return the exact text appended by the tokenizer's generation prompt."""
    with_generation = _apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    without_generation = _apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    if not with_generation.startswith(without_generation):
        raise ValueError("Chat template generation prompt is not a suffix of the rendered conversation")
    suffix = with_generation[len(without_generation) :]
    if not suffix:
        raise ValueError("Chat template returned an empty generation prompt suffix")
    return suffix


def _render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Render one complete conversation with the assistant generation boundary."""
    return apply_chat_template_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )


def _max_fitting_prefix(
    tokenizer: Any,
    token_ids: list[int],
    other_text: str,
    *,
    prefix_is_title: bool,
    rebuild_messages: Callable[[str, str], list[dict[str, str]]],
    maximum_tokens: int,
) -> tuple[int, list[int], list[dict[str, str]]]:
    """Find the longest content-token prefix whose complete chat prompt fits."""
    low = 0
    high = len(token_ids)
    best_count = -1
    best_ids: list[int] | None = None
    best_messages: list[dict[str, str]] | None = None

    while low <= high:
        candidate_count = (low + high) // 2
        candidate_text = _decode_prefix(tokenizer, token_ids, candidate_count)
        title = candidate_text if prefix_is_title else other_text
        body = other_text if prefix_is_title else candidate_text
        candidate_messages = rebuild_messages(title, body)
        candidate_ids = _render_ids(tokenizer, candidate_messages)
        if len(candidate_ids) <= maximum_tokens:
            best_count = candidate_count
            best_ids = candidate_ids
            best_messages = candidate_messages
            low = candidate_count + 1
        else:
            high = candidate_count - 1

    if best_count < 0 or best_ids is None or best_messages is None:
        raise ValueError("Even an empty issue-content prefix does not fit the prompt budget")
    return best_count, best_ids, best_messages


def prepare_prompt_preserving_structure(
    tokenizer: Any,
    original_messages: list[dict[str, str]],
    *,
    title: str,
    body: str,
    rebuild_messages: Callable[[str, str], list[dict[str, str]]],
    maximum_tokens: int,
) -> PromptTruncationResult:
    """Render a complete prompt and shorten only body, then title, when required."""
    original_prompt_ids = _render_ids(tokenizer, original_messages)
    title_ids = _tokenize_text(tokenizer, title)
    body_ids = _tokenize_text(tokenizer, body)
    full_input_token_count = len(original_prompt_ids)
    suffix_text = generation_prompt_suffix(tokenizer, original_messages)

    if full_input_token_count <= maximum_tokens:
        final_messages = original_messages
        final_prompt_ids = original_prompt_ids
        strategy = "none"
        title_after = len(title_ids)
        body_after = len(body_ids)
    else:
        full_title = title
        empty_body = ""
        title_after = len(title_ids)
        full_title_empty_body_messages = rebuild_messages(full_title, empty_body)
        full_title_empty_body_ids = _render_ids(tokenizer, full_title_empty_body_messages)
        if len(full_title_empty_body_ids) <= maximum_tokens:
            body_after, final_prompt_ids, final_messages = _max_fitting_prefix(
                tokenizer,
                body_ids,
                full_title,
                prefix_is_title=False,
                rebuild_messages=rebuild_messages,
                maximum_tokens=maximum_tokens,
            )
            strategy = "body_right"
        else:
            title_after, final_prompt_ids, final_messages = _max_fitting_prefix(
                tokenizer,
                title_ids,
                empty_body,
                prefix_is_title=True,
                rebuild_messages=rebuild_messages,
                maximum_tokens=maximum_tokens,
            )
            body_after = 0
            strategy = "body_right_then_title_right"

    if len(final_prompt_ids) > maximum_tokens:
        raise ValueError("Prompt-preserving truncation exceeded the maximum input token budget")
    rendered_final = _apply_chat_template_text(
        tokenizer,
        final_messages,
        add_generation_prompt=True,
    )
    if not rendered_final.endswith(suffix_text):
        raise ValueError("Prompt-preserving truncation removed the assistant generation boundary")

    return PromptTruncationResult(
        original_prompt_ids=original_prompt_ids,
        prompt_ids=final_prompt_ids,
        original_messages=original_messages,
        final_messages=final_messages,
        full_input_token_count=full_input_token_count,
        input_truncated=full_input_token_count > maximum_tokens,
        title_token_count_before=len(title_ids),
        title_token_count_after=title_after,
        body_token_count_before=len(body_ids),
        body_token_count_after=body_after,
        generation_prompt_preserved=True,
        generation_prompt_suffix_text=suffix_text,
        truncation_strategy=strategy,
    )
