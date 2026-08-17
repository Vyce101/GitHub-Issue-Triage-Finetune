"""Hold paths and constants for the final frozen TEST evaluation."""

from pathlib import Path

from validation_evaluation.config import (
    ADAPTER_PATH,
    CONDITION_BASE_FEW_SHOT,
    CONDITION_BASE_ZERO_SHOT,
    CONDITION_FINE_TUNED_ZERO_SHOT,
    CONDITIONS,
    GENERATION_MAX_NEW_TOKENS,
    MAX_INPUT_TOKENS,
    MODEL_ID,
    MODEL_LOAD_MAX_SEQUENCE_LENGTH,
    MODEL_REVISION,
    TARGET_CATEGORIES,
    TOTAL_CONTEXT_TOKENS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/initial_qlora_config.json"
TRAINING_REPORT_PATH = PROJECT_ROOT / "results/initial_qlora_training.json"
FROZEN_MANIFEST_PATH = PROJECT_ROOT / "results/final_split_manifest.json"
TEST_SPLIT_PATH = PROJECT_ROOT / "data/processed/splits/test.jsonl"
PROMPT_DEFINITION_PATH = PROJECT_ROOT / "results/baselines/prompts.json"
TEST_REPORT_PATH = PROJECT_ROOT / "results/test_evaluation.json"
TEST_RAW_ARTIFACT_DIRECTORY = PROJECT_ROOT / "results/test_evaluation"

TEST_SHA256 = "7257588f3b21de561d7ac16cbc24bcc7887877ab20c99e3e45499635187a06bc"
TEST_ROW_COUNT = 9708
TEST_CLASS_COUNTS = {
    "bug": 5249,
    "feature": 4045,
    "documentation": 282,
    "question_support": 132,
}

STATISTICAL_SEED = 3407
BOOTSTRAP_REPLICATES = 10_000
STABLE_CLASS_SUPPORT_THRESHOLD = 30
LIMITED_CLASS_SUPPORT_THRESHOLD = 10
