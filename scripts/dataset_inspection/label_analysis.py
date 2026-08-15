"""Exploratory analysis helpers for raw labels."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .config import METADATA_LABEL_PATTERNS, TOP_LABEL_LIMIT
from .raw_values import display_value, json_safe


def counter_entries(counter: Counter[str], values: dict[str, Any], limit: int = TOP_LABEL_LIMIT) -> list[dict[str, Any]]:
    """Serialize a counter keyed by raw-value fingerprints."""
    entries = []
    for key, count in counter.most_common(limit):
        raw_value = values[key]
        entries.append({"raw_label": json_safe(raw_value), "display": display_value(raw_value), "count": count})
    return entries


def label_style_names(value: Any) -> list[str]:
    """Describe raw label naming shapes without remapping their values."""
    label = display_value(value).strip()
    if not label:
        return ["empty"]

    styles = []
    if label == label.casefold():
        styles.append("lowercase")
    elif label == label.upper():
        styles.append("uppercase")
    elif label == label.title():
        styles.append("title_case")
    else:
        styles.append("mixed_case")
    if "-" in label:
        styles.append("hyphenated")
    if "_" in label:
        styles.append("underscored")
    if " " in label:
        styles.append("spaced")
    if ":" in label:
        styles.append("colon_prefixed_or_scoped")
    if re.search(r"\d", label):
        styles.append("contains_digits")
    return styles


def label_shape_signature(value: Any) -> str:
    """Build an exploratory separator/case signature, never a training label."""
    label = display_value(value).casefold().strip()
    label = re.sub(r"[-_:]+", " ", label)
    label = re.sub(r"\s+", " ", label)
    return re.sub(r"[^a-z0-9 ]", "", label)


def metadata_reasons(value: Any) -> list[str]:
    """Flag label names that may describe workflow metadata rather than issue type."""
    label = display_value(value).casefold().strip()
    reasons = []
    for reason, pattern in METADATA_LABEL_PATTERNS:
        if re.search(pattern, label):
            reasons.append(reason)
    return reasons


def label_appears_in_text(label: Any, title: Any, body: Any, text_value) -> bool:
    """Detect possible label-name leakage into issue text."""
    query = display_value(label).strip()
    if len(query) < 2 or not re.search(r"[A-Za-z]", query):
        return False
    combined_text = f"{text_value(title)}\n{text_value(body)}"
    if re.fullmatch(r"[A-Za-z0-9_ -]+", query):
        pattern = rf"(?<!\w){re.escape(query)}(?!\w)"
        return re.search(pattern, combined_text, flags=re.IGNORECASE) is not None
    return query.casefold() in combined_text.casefold()
