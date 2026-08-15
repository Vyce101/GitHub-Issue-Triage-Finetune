# GitHub Issue Triage Fine-Tune

This repository contains a reproducible machine-learning experiment for fine-tuning a small open-weight language model to classify GitHub issues into normalized categories.

The project is designed to preserve the complete experimental chain:

raw data → cleaned dataset → split methodology → base-model benchmark → training configuration → trained adapter → held-out evaluation → failure analysis → reproducible results

## Current status

Environment setup and local GPU verification are complete. The training model, dataset, and fine-tuning pipeline have not yet been downloaded or created.

## Local environment

The project targets an isolated Python 3.11 environment and a CUDA-enabled PyTorch installation suitable for an 8 GB NVIDIA GPU. See [environment.md](environment.md) for the recorded setup, installed versions, and verification results.

Install the pinned base dependencies from PowerShell with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Repository conventions

Large model files, Hugging Face caches, local datasets, credentials, checkpoints, and generated experiment outputs are intentionally excluded from version control. The tracked repository should contain the code, configuration, documentation, and evidence needed to reproduce and evaluate the experiment.
