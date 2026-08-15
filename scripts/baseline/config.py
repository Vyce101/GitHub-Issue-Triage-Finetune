"""Configuration and paths for the prompting-only baseline experiment."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
MODEL_REVISION = "7744afa8566e264af1a92a806d8d9aae00cc7c78"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data/processed/splits/train.jsonl"
VALIDATION_SPLIT_PATH = PROJECT_ROOT / "data/processed/splits/validation.jsonl"
RESULTS_DIRECTORY = PROJECT_ROOT / "results/baselines"
PROMPT_DEVELOPMENT_SIZE_PER_CLASS = 100
PROMPT_DEVELOPMENT_MAX_ZERO_SHOT_INPUT_TOKENS = 512
FEW_SHOT_EXAMPLES_PER_CLASS = 2
CONTEXT_LIMITS = (1024, 1536, 2048, 3072, 4096)
OUTPUT_RESERVE_TOKENS = 16
GENERATION_MAX_NEW_TOKENS = 16
EVALUATION_BATCH_SIZE = 4
BASELINE_MODEL_LOAD_MAX_SEQUENCE_LENGTH = 2048
TARGET_CATEGORIES = ("bug", "feature", "documentation", "question_support")

CATEGORY_DEFINITIONS = {
    "bug": "An existing behavior is broken, incorrect, crashing, regressed, or producing an unintended result.",
    "feature": "A new capability, enhancement, requested behavior change, or improvement is being proposed.",
    "documentation": "Documentation, guides, examples, comments, wording, or explanatory material needs to be added or corrected.",
    "question_support": "The author is primarily asking for help, usage guidance, clarification, or support rather than reporting a defect or requesting a concrete feature.",
}

SYSTEM_INSTRUCTION = """You classify GitHub issues into exactly one issue type.

Use these category definitions:
- bug: an existing behavior is broken, incorrect, crashing, regressed, or producing an unintended result.
- feature: a new capability, enhancement, requested behavior change, or improvement is being proposed.
- documentation: documentation, guides, examples, comments, wording, or explanatory material needs to be added or corrected.
- question_support: the author is primarily asking for help, usage guidance, clarification, or support rather than reporting a defect or requesting a concrete feature.

Choose exactly one category. Do not provide reasoning. Return exactly one JSON object with this schema and no other text:
{"type":"bug"}
"""
