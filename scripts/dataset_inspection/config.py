"""Configuration and thresholds for the raw dataset inspection workflow."""

from pathlib import Path


DATASET_ID = "sharjeelyunus/github-issues-dataset"
OUTPUT_PATH = Path("results/dataset_inspection.json")
TOP_LABEL_LIMIT = 100
EXAMPLE_LIMIT = 5
SHORT_BODY_LIMIT = 500
SHORT_TEXT_TITLE_LIMIT = 5
SHORT_TEXT_BODY_LIMIT = 20
LONG_BODY_LIMIT = 10_000

METADATA_LABEL_PATTERNS = (
    ("priority-like", r"^(?:p[0-5]|priority)(?:[\s:_-].*)?$"),
    ("severity-like", r"^severity(?:[\s:_-].*)?$"),
    ("workflow/status-like", r"\b(?:status|state|workflow|triage|milestone)\b"),
    ("organization-like", r"\b(?:component|area|module|project|platform|os|language|version)\b"),
    ("effort-like", r"\b(?:size|effort|estimate)\b"),
)
