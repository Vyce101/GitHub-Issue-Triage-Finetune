# GitHub Issue Triage Fine-Tune

This repository contains a reproducible supervised fine-tuning experiment for classifying GitHub issues from their title and body into four normalized categories:

- `bug` — an existing behavior is broken, incorrect, crashing, or unintended
- `feature` — a new capability, enhancement, or requested behavior change
- `documentation` — documentation, examples, comments, or explanatory material
- `question_support` — a request for help, usage guidance, or clarification

The experiment used QLoRA on `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` and evaluated generalization across repositories rather than relying on a random-row split. The final result is negative and is reported plainly: the trained adapter did not outperform the untouched base model on the frozen held-out TEST repositories.

## Final result

Macro-F1 is the primary metric because the taxonomy is strongly imbalanced.

| Frozen TEST condition | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Base zero-shot | 0.827771 | 0.618810 |
| Base few-shot | 0.808096 | 0.575545 |
| Fine-tuned zero-shot | 0.819324 | 0.575087 |

Fine-tuned zero-shot minus base zero-shot was **-0.008447 accuracy** and **-0.043723 macro-F1**. The paired 95% bootstrap intervals were `[-0.014318, -0.002472]` for accuracy and `[-0.066798, -0.021391]` for macro-F1. The exact paired McNemar test was `p = 0.00690115`.

The fine-tuned model did not outperform the frozen base zero-shot baseline on held-out TEST repositories. Base zero-shot remained the strongest final condition. The complete evidence is in [results/test_evaluation.json](results/test_evaluation.json).

## Problem

GitHub issue labels are heterogeneous across projects. A random-row split can place repository-specific templates, vocabulary, workflow conventions, and label semantics in both training and evaluation. This project therefore treats repository-held-out classification as the primary generalization test: entire repositories are assigned to train, validation, or TEST, with zero repository overlap between splits.

## Dataset and taxonomy

The source is the [`sharjeelyunus/github-issues-dataset`](https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset). The raw source contained 114,073 issues. Normalization preserved the raw labels and mapped only high-confidence label evidence to the four approved issue types.

Before duplicate removal, 59,163 rows had no high-confidence target category and 577 had conflicting target categories; these were excluded. One duplicate title/body row was then removed. The final normalized dataset contains 54,332 examples:

| Split | Rows | bug | feature | documentation | question_support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 31,876 | 19,454 | 9,936 | 1,390 | 1,096 |
| Validation | 12,748 | 7,298 | 4,987 | 317 | 146 |
| TEST | 9,708 | 5,249 | 4,045 | 282 | 132 |

The final split assignment used 37 train repositories, 9 validation repositories, and 10 TEST repositories. Split hashes, repository assignments, class counts, and overlap checks are recorded in [results/final_split_manifest.json](results/final_split_manifest.json). The construction and normalization audit is in [results/split_analysis.json](results/split_analysis.json) and [results/taxonomy_reconciliation.json](results/taxonomy_reconciliation.json).

## Model and training

The locked base model was `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at revision `7744afa8566e264af1a92a806d8d9aae00cc7c78`.

- QLoRA with NF4 quantization, double quantization, and FP16 compute
- Approximately 33.0M trainable LoRA parameters on the 4.02B logical base model
- LoRA rank 16 and alpha 16, targeting the attention and MLP projection modules
- Maximum sequence length: 1536 tokens
- One full epoch over all 31,876 TRAIN examples
- 1,993 optimizer steps with effective batch size 16
- Completion-only loss on the assistant classification response
- No requested chain-of-thought target
- NVIDIA RTX 2060 SUPER with 8 GB VRAM
- Training runtime: approximately 9h44m

The approved settings are in [configs/initial_qlora_config.json](configs/initial_qlora_config.json), the executed-run audit is in [results/initial_qlora_training.json](results/initial_qlora_training.json), and the human-readable configuration notes are in [docs/initial_qlora_configuration.md](docs/initial_qlora_configuration.md).

## Evaluation methodology

Prompt development used TRAIN-derived evidence only. Corrected VALIDATION results were used for development and model-selection analysis; TEST remained frozen for the final comparison.

The final evaluation measured three paired conditions on the same TEST rows:

1. Base zero-shot with the locked model and no adapter.
2. Base few-shot with the same model and the eight frozen TRAIN-derived demonstrations.
3. Fine-tuned zero-shot with the locked base model plus the unmerged trained LoRA adapter.

All conditions used deterministic generation (`do_sample=false`, `max_new_tokens=16`), the canonical parser in [scripts/classification/parser.py](scripts/classification/parser.py), and the same structured JSON contract. The final report includes per-class metrics, confusion matrices, truncation strata, repository analysis, paired correctness counts, fixed-seed paired bootstrap resampling, and exact McNemar analysis. The statistical seed was 3407 with 10,000 bootstrap replicates.

An earlier read-only metadata/preflight command opened and SHA-256-hashed the frozen TEST file. This was documented as a process deviation, but TEST examples or labels were not used for training, prompt design, hyperparameter selection, model selection, or validation decisions. No post-TEST tuning or experiment was performed.

## Evaluation bug and correction

The historical validation evaluator right-truncated the already-rendered chat-template token sequence. On long prompts, that could remove the assistant generation boundary and cause the model to continue the issue text instead of generating classification JSON.

The defect was diagnosed and corrected before the final development decision. The corrected evaluator truncates only the current issue body from the right, then the title from the right if necessary, while preserving system instructions, frozen demonstrations, role structure, the assistant generation boundary, and space for 16 output tokens. Only affected validation rows were regenerated; unaffected historical rows were reused exactly, and the historical artifacts were retained.

The corrected validation report is [results/validation_evaluation_corrected.json](results/validation_evaluation_corrected.json). The failure analysis documents the defect, repair, regenerated-row counts, and remaining generalization patterns in [results/validation_failure_analysis_corrected.json](results/validation_failure_analysis_corrected.json). The historical report remains available at [results/validation_evaluation.json](results/validation_evaluation.json).

## What fine-tuning changed

On TEST, fine-tuning shifted behavior toward `bug` classification. Relative to base zero-shot, paired correctness changed by:

| True class | Net fine-tuned correctness change |
| --- | ---: |
| bug | +276 |
| feature | −306 |
| documentation | −39 |
| question_support | −13 |

The corresponding per-class F1 values moved from 0.862609 to 0.859037 for bug, 0.812863 to 0.788948 for feature, 0.552347 to 0.506667 for documentation, and 0.247423 to 0.145695 for question_support. The adapter therefore did not provide a useful final accuracy/class-balance tradeoff; base zero-shot remained the strongest frozen condition.

## Reproducibility

The project targets an isolated Python 3.11 environment. From the repository root in PowerShell:

```powershell
uv python install 3.11.13 --install-dir .python-runtime --no-bin --no-registry
$runtime = (Resolve-Path -LiteralPath '.python-runtime\cpython-3.11.13-windows-x86_64-none\python.exe').Path
& $runtime -m venv --copies .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The recorded hardware, package versions, CUDA verification, and environment notes are in [environment.md](environment.md).

For the source-to-split workflow, expose the script packages and run the relevant entry points from the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\scripts).Path
\.venv\Scripts\python.exe -c "from dataset_inspection.runner import main; main()"
\.venv\Scripts\python.exe -c "from dataset_inspection.split_analysis import analyze_splits, print_split_analysis_summary, write_split_analysis_report; summary=analyze_splits(); write_split_analysis_report(summary); print_split_analysis_summary(summary)"
```

The approved training runner supports a preflight gate and the recorded full run:

```powershell
\.venv\Scripts\python.exe -m qlora_training.run_full --preflight-only
\.venv\Scripts\python.exe -m qlora_training.run_full
```

Validation and final-evaluation tooling is organized under [scripts/validation_evaluation](scripts/validation_evaluation), [scripts/test_evaluation](scripts/test_evaluation), and [scripts/run_test_evaluation.py](scripts/run_test_evaluation.py):

```powershell
\.venv\Scripts\python.exe -m validation_evaluation.runner --smoke-only
\.venv\Scripts\python.exe -m validation_evaluation.runner
\.venv\Scripts\python.exe scripts\run_test_evaluation.py
```

The final TEST command is retained for audit/reproduction of the frozen evaluation, not for post-TEST model selection. The final report is tracked; model weights, adapters, datasets, caches, and row-level generated prediction JSONL files are local or ignored artifacts rather than repository contents.

## Repository structure

```text
configs/                     Frozen machine-readable experiment settings
docs/                        Human-readable training configuration notes
results/                     Tracked reports, manifests, metrics, and analysis
scripts/                     Dataset, baseline, training, validation, and TEST tooling
environment.md               Recreate-and-verify local environment record
```

Important tracked artifacts include:

- [results/test_evaluation.json](results/test_evaluation.json) — final frozen TEST report
- [results/final_split_manifest.json](results/final_split_manifest.json) — split assignments and hashes
- [results/initial_qlora_training.json](results/initial_qlora_training.json) — executed training audit
- [results/validation_evaluation_corrected.json](results/validation_evaluation_corrected.json) — corrected development evaluation
- [results/validation_failure_analysis_corrected.json](results/validation_failure_analysis_corrected.json) — truncation failure analysis
- [configs/initial_qlora_config.json](configs/initial_qlora_config.json) — approved training configuration

## Limitations

- Source GitHub labels are noisy and repository-dependent.
- The normalized taxonomy contains genuine ambiguity, especially at the boundaries between feature, documentation, and support questions.
- The class distribution is strongly imbalanced toward bug and feature.
- Repository/domain shift is the intended difficulty, but it also makes small repository-specific results unstable.
- Documentation and question_support are minority classes and are concentrated in relatively few repositories.
- This study uses one base model and one principal full-data QLoRA run.
- The experiment was constrained by a consumer RTX 2060 SUPER with 8 GB VRAM.
- The frozen TEST result is descriptive final evidence; it is not a basis for additional post-TEST tuning in this experiment.

## Key takeaway

Fine-tuning is not automatically superior to prompting. This experiment completed a successful QLoRA training pipeline, then rigorous repository-held-out evaluation showed that the intervention reduced cross-repository macro-F1 despite the successful training run.

## License

This project is released under the [MIT License](LICENSE). Copyright © 2026 Luke.
