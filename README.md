# GitHub Issue Triage Fine-Tune

This repository contains a reproducible machine-learning experiment for fine-tuning a small open-weight language model to classify GitHub issues into normalized categories.

The project is designed to preserve the complete experimental chain:

raw data → cleaned dataset → split methodology → base-model benchmark → training configuration → trained adapter → held-out evaluation → failure analysis → reproducible results

## Current status

Environment setup, raw dataset inspection, normalized dataset construction, the approved primary split, model smoke testing, and zero-shot/few-shot baseline evaluation are complete. The clean QLoRA context-feasibility benchmark is also complete; the initial real fine-tuning run has not started. The initial training configuration will use a maximum sequence length of 1536 based on the RTX 2060 SUPER benchmark.

The task predicts one of four normalized GitHub issue categories: `bug`, `feature`, `documentation`, or `question_support`. The locked starting model is `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at revision `7744afa8566e264af1a92a806d8d9aae00cc7c78`.

On the train-only prompt-development subset, the recorded macro-F1 was 0.5536 for zero-shot prompting and 0.6889 for few-shot prompting. Validation and test data have not been used for prompt development or fine-tuning.

## Primary benchmark split

The primary benchmark uses repository-held-out evaluation. Entire repositories are isolated between train, validation, and test. The frozen split assignment and integrity hashes are recorded in [results/final_split_manifest.json](results/final_split_manifest.json).

## Local environment

The project targets an isolated Python 3.11 environment and local QLoRA training on an 8 GB NVIDIA RTX 2060 SUPER. See [environment.md](environment.md) for the recorded setup, installed versions, and verification results.

Install the pinned base dependencies from PowerShell with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Repository conventions

Large model files, Hugging Face caches, local datasets, credentials, checkpoints, and generated processed JSONL datasets are intentionally excluded from version control. Reproducible source, configuration, documentation, frozen split metadata, baseline evidence, smoke-test results, and training-feasibility reports are tracked.
