# Initial real QLoRA configuration

This document explains the approved first-run configuration in [configs/initial_qlora_config.json](../configs/initial_qlora_config.json). It is a configuration artifact only: no full-data fine-tuning run or full-data adapter checkpoint has been created.

## Frozen inputs from the completed pipeline

- Base model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, revision `7744afa8566e264af1a92a806d8d9aae00cc7c78`.
- Task: classify each issue as `bug`, `feature`, `documentation`, or `question_support`.
- Training data: all 31,876 rows in the frozen repository-held-out train split.
- Validation data: 12,748 rows in the frozen validation split.
- Test data: 9,708 rows in the frozen test split; it remains unavailable for this stage.
- Prompt: the existing zero-shot system instruction and title/body user message. Raw GitHub labels are not included in model inputs.
- Target: one assistant JSON object, for example `{"type":"bug"}`; no reasoning target is requested.

The split hashes in the JSON configuration are copied from `results/final_split_manifest.json`. The train, validation, and test repositories are disjoint, and the manifest explicitly forbids using test data for development decisions.

## Approved first-run configuration

The first controlled full-data experiment uses one epoch. This is an initial measurement, not a claim that one epoch is ultimately optimal. The run uses QLoRA with 4-bit NF4 weights, double quantization, and FP16 compute. The RTX 2060 SUPER does not support BF16 in the verified Unsloth stack, so BF16 is disabled and FP16 is enabled.

The context length is 1536. The clean train-only benchmark passed at 1536 with micro-batch size 1, approximately 4.7 GiB peak allocated GPU memory, approximately 5.7 GiB peak reserved memory, and about 1.1 GiB free at the recorded peak. The 2048-token case also passed but left less headroom, while the 1536 case had better measured throughput in the clean benchmark. The training data path must truncate only the issue prompt and preserve the short assistant target.

The adapter configuration promotes the successfully sanity-tested LoRA shape: rank 16, alpha 16, no dropout, no bias parameters, and the seven attention/MLP projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`). The base model remains frozen.

The optimizer settings are micro-batch size 1, gradient accumulation 16, effective batch size 16, `adamw_8bit`, learning rate `2e-4`, linear decay, a 5% warmup ratio, weight decay `0.01`, and gradient clipping at 1.0. With 31,876 training rows, one epoch is expected to produce `ceil(31,876 / 16) = 1,993` optimizer steps. Natural frozen-train sampling is retained; no class weighting or oversampling is added to the first experiment so the result measures the effect of the adapter without a second intervention.

## Loss and evaluation boundaries

Only the final assistant response is trained (`completion_only_loss=true` and `train_on_assistant_response_only=true`). The Qwen chat template in the locked tokenizer does not contain TRL generation markers, so the configuration records Unsloth's response-only masking with the verified markers `<|im_start|>user\n` and `<|im_start|>assistant\n`. The current tokenizer renders an empty `<think>...</think>` control block before an assistant message even when `enable_thinking=false`; this is template syntax rather than a reasoning target. The final preflight must verify that prompt tokens are masked, the JSON target remains in the loss region, and the target survives truncation.

Training-time validation is limited to the frozen validation split. The initial one-epoch run has no competing checkpoints, so validation is reporting evidence rather than a tuning loop. The final model evaluation should use the same zero-shot prompt condition and deterministic generation contract as the baseline: `do_sample=false`, `max_new_tokens=16`, the canonical parser in `scripts/classification/parser.py`, macro-F1 as the primary metric, and accuracy, valid-output rate, per-class metrics, and confusion matrix as supporting metrics. The frozen test split is reserved for the final held-out comparison after the configuration and checkpoint-selection policy are approved.

## Historical draft choices superseded by this configuration

The earlier draft proposed rank 8, gradient accumulation 8, effective batch size 8, cosine scheduling, 120 warmup steps, and zero weight decay. Those values are retained here only as historical context and are not the settings for the approved first run. The promoted settings are rank 16, gradient accumulation 16, effective batch size 16, a linear scheduler, a 5% warmup ratio, and weight decay `0.01`, matching the completed sanity experiment.

## Trainable-parameter reporting

The completed 64-example sanity experiment trained exactly `33,030,144` LoRA parameters and reported `33,030,144 / 2,539,650,560 = 1.300578%`. That percentage uses the loaded quantized/library-reported denominator and is preserved in `results/training_sanity_check.json` as historical evidence.

The locked model smoke test recorded the logical base-model parameter count as `4,022,468,096`. Comparing the same `33,030,144` LoRA parameters with that logical count gives approximately `0.821%`. These denominators answer different questions and must not be conflated.

For README, resume, and other concise project descriptions, use unambiguous wording such as: approximately 33.0M LoRA parameters were trained on a 4.02B-parameter base model.

The final preflight and user review must be completed before the full-data run starts. No validation or test evaluation is performed by this configuration update.
