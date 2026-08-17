"""Hold paths and constants for the frozen validation evaluation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
MODEL_REVISION = "7744afa8566e264af1a92a806d8d9aae00cc7c78"
TARGET_CATEGORIES = ("bug", "feature", "documentation", "question_support")

CONFIG_PATH = PROJECT_ROOT / "configs/initial_qlora_config.json"
TRAINING_REPORT_PATH = PROJECT_ROOT / "results/initial_qlora_training.json"
FROZEN_MANIFEST_PATH = PROJECT_ROOT / "results/final_split_manifest.json"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data/processed/splits/train.jsonl"
VALIDATION_SPLIT_PATH = PROJECT_ROOT / "data/processed/splits/validation.jsonl"
PROMPT_SELECTION_PATH = PROJECT_ROOT / "results/baselines/prompt_development_selection.json"
PROMPT_DEFINITION_PATH = PROJECT_ROOT / "results/baselines/prompts.json"
ADAPTER_PATH = PROJECT_ROOT / "outputs/initial-real-qlora"

REPORT_PATH = PROJECT_ROOT / "results/validation_evaluation.json"
RAW_ARTIFACT_DIRECTORY = PROJECT_ROOT / "results/validation_evaluation"
CORRECTED_REPORT_PATH = PROJECT_ROOT / "results/validation_evaluation_corrected.json"
CORRECTED_RAW_ARTIFACT_DIRECTORY = PROJECT_ROOT / "results/validation_evaluation_corrected"

VALIDATION_SHA256 = "0cfb655d12bf66d83befb1a0be829ff3096139c48b5752c716a0b6489302eb5c"
VALIDATION_ROW_COUNT = 12748

TOTAL_CONTEXT_TOKENS = 1536
GENERATION_OUTPUT_RESERVE_TOKENS = 16
MAX_INPUT_TOKENS = TOTAL_CONTEXT_TOKENS - GENERATION_OUTPUT_RESERVE_TOKENS
GENERATION_MAX_NEW_TOKENS = 16
EVALUATION_BATCH_SIZE = 4
MAX_BATCH_INPUT_TOKENS = 4096
MODEL_LOAD_MAX_SEQUENCE_LENGTH = TOTAL_CONTEXT_TOKENS

CONDITION_BASE_ZERO_SHOT = "base_zero_shot"
CONDITION_BASE_FEW_SHOT = "base_few_shot"
CONDITION_FINE_TUNED_ZERO_SHOT = "fine_tuned_zero_shot"
CONDITIONS = (
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_BASE_FEW_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
)
