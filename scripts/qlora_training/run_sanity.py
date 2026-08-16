"""Run the 64-example train-only QLoRA overfit and masking sanity experiment."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data import (
    SanityExample,
    sanity_dataset_records,
    select_sanity_examples,
    selection_report,
    to_dataset,
)
from .evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/training_sanity_config.json"


def _read_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the sanity-only configuration boundary."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "sanity_only" or config["full_training_approved"]:
        raise ValueError("The training config must remain sanity_only and not approved for full training")
    if config["data"]["source_split"] != "train":
        raise ValueError("The sanity dataset must come from the train split")
    if config["data"]["total_examples"] != 64:
        raise ValueError("The sanity dataset must contain exactly 64 examples")
    return config


def _sha256_file(path: Path) -> str:
    """Hash the train split without opening any validation or test file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    """Record the pinned runtime versions used by the sanity experiment."""
    names = ("torch", "transformers", "trl", "peft", "bitsandbytes", "unsloth", "datasets")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["torch_cuda_runtime"] = torch.version.cuda
    return versions


def _gpu_memory_snapshot() -> dict[str, int | float]:
    """Capture current and peak CUDA memory in MiB."""
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "allocated_bytes": allocated,
        "allocated_mib": round(allocated / 2**20, 2),
        "reserved_bytes": reserved,
        "reserved_mib": round(reserved / 2**20, 2),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": round(peak_allocated / 2**20, 2),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": round(peak_reserved / 2**20, 2),
        "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 2),
    }


def _trainable_parameter_report(model: Any) -> dict[str, Any]:
    """Verify that PEFT left only LoRA parameters trainable."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    non_lora_trainable_names = [name for name in trainable_names if "lora_" not in name.lower()]
    base_parameters_frozen = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name.lower()
    )
    return {
        "total_parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "trainable_parameter_percentage": round(100 * trainable / total, 6),
        "trainable_parameter_name_count": len(trainable_names),
        "trainable_parameter_names_sample": trainable_names[:12],
        "only_lora_parameters_trainable": not non_lora_trainable_names,
        "base_parameters_frozen": base_parameters_frozen,
        "non_lora_trainable_names": non_lora_trainable_names,
    }


def _verify_trainer_labels(
    trainer: Any,
    examples: list[SanityExample],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    """Inspect the exact labels produced by TRL's completion-only collator."""
    prepared_dataset = trainer.train_dataset
    if len(prepared_dataset) != len(examples):
        raise AssertionError("Prepared trainer dataset changed the sanity example count")
    per_example = []
    prompt_loss_token_total = 0
    completion_loss_token_total = 0
    truncated_examples = []
    all_nonempty = True
    all_prompt_masked = True
    all_targets_preserved = True
    all_eos_present = True
    for index, example in enumerate(examples):
        prepared = prepared_dataset[index]
        batch = trainer.data_collator([prepared])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        if "attention_mask" in batch:
            actual_mask = batch["attention_mask"][0].bool()
        elif "position_ids" in batch:
            # Padding-free batches use position IDs; this one-example batch has no padding.
            actual_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            raise KeyError("The trainer collator returned neither attention_mask nor position_ids")
        actual_ids = input_ids[actual_mask].tolist()
        actual_labels = labels[actual_mask].tolist()
        prompt_length = example.prompt_token_count
        full_length = example.full_sequence_token_count
        prompt_loss_tokens = sum(
            label != -100 for label in actual_labels[: min(prompt_length, len(actual_labels))]
        )
        loss_positions = [position for position, label in enumerate(actual_labels) if label != -100]
        loss_token_ids = [actual_ids[position] for position in loss_positions]
        loss_text = tokenizer.decode(loss_token_ids, skip_special_tokens=False)
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None and tokenizer.eos_token is not None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        eos_marker = tokenizer.eos_token or "<|im_end|>"
        eos_positions = [
            position
            for position in range(prompt_length, len(actual_ids))
            if eos_token_id is not None and actual_ids[position] == eos_token_id
        ]
        eos_position = eos_positions[0] if eos_positions else None
        target_json_start = loss_text.find(example.completion_text)
        eos_marker_start = loss_text.find(eos_marker)
        target_json_preserved = (
            target_json_start >= 0
            and (eos_marker_start < 0 or target_json_start + len(example.completion_text) <= eos_marker_start)
        )
        eos_present = eos_position is not None
        eos_in_loss = eos_position is not None and eos_position in loss_positions
        truncated = len(actual_ids) != full_length or actual_ids[:full_length] != example.full_ids
        prompt_loss_token_total += prompt_loss_tokens
        completion_loss_token_total += len(loss_positions)
        all_nonempty = all_nonempty and bool(loss_positions)
        all_prompt_masked = all_prompt_masked and prompt_loss_tokens == 0
        all_targets_preserved = all_targets_preserved and target_json_preserved
        all_eos_present = all_eos_present and eos_present
        if truncated:
            truncated_examples.append(
                {
                    "issue_id": example.row["issue_id"],
                    "repository": example.row["repository"],
                    "expected_full_sequence_tokens": full_length,
                    "actual_sequence_tokens": len(actual_ids),
                }
            )
        per_example.append(
            {
                "issue_id": example.row["issue_id"],
                "repository": example.row["repository"],
                "target_category": example.row["target_category"],
                "prompt_token_count": prompt_length,
                "full_sequence_token_count": full_length,
                "actual_sequence_token_count": len(actual_ids),
                "completion_loss_token_count": len(loss_positions),
                "prompt_loss_token_count": prompt_loss_tokens,
                "target_json_preserved_before_eos": target_json_preserved,
                "eos_present_after_prompt": eos_present,
                "eos_in_completion_loss": eos_in_loss,
                "truncated": truncated,
                "loss_text_preview": loss_text[:160],
            }
        )
    return {
        "method": "TRL SFTTrainer prompt-completion completion_only_loss collator",
        "example_count": len(examples),
        "all_examples_have_nonempty_completion_loss_tokens": all_nonempty,
        "all_prompt_tokens_masked": all_prompt_masked,
        "all_target_json_preserved_before_eos": all_targets_preserved,
        "all_eos_tokens_present_after_prompt": all_eos_present,
        "eos_in_completion_loss_is_informational": True,
        "prompt_loss_token_total": prompt_loss_token_total,
        "completion_loss_token_total": completion_loss_token_total,
        "truncated_examples": truncated_examples,
        "all_checks_passed": all(
            (
                all_nonempty,
                all_prompt_masked,
                all_targets_preserved,
                all_eos_present,
                not truncated_examples,
            )
        ),
        "per_example": per_example,
    }


def _build_trainer(model: Any, tokenizer: Any, dataset: Any, config: dict[str, Any]) -> Any:
    """Build the TRL SFT trainer with the locked sanity-only optimization settings."""
    from trl import SFTConfig, SFTTrainer

    optimization = config["optimization"]
    artifacts = config["artifacts"]
    sequence = config["model"]
    loss = config["loss"]
    args = SFTConfig(
        output_dir=str(PROJECT_ROOT / artifacts["temporary_output_dir"]),
        per_device_train_batch_size=optimization["per_device_train_batch_size"],
        gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
        num_train_epochs=optimization["sanity_num_epochs"],
        max_steps=optimization["sanity_max_steps"],
        learning_rate=optimization["learning_rate"],
        lr_scheduler_type=optimization["lr_scheduler_type"],
        optim=optimization["optimizer"],
        weight_decay=optimization["weight_decay"],
        warmup_steps=optimization["sanity_warmup_steps"],
        max_grad_norm=optimization["max_grad_norm"],
        gradient_checkpointing=True,
        use_cache=optimization["use_cache"],
        fp16=optimization["fp16"],
        bf16=optimization["bf16"],
        tf32=optimization["tf32"],
        seed=optimization["seed"],
        data_seed=optimization["data_seed"],
        dataloader_num_workers=optimization["dataloader_num_workers"],
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        eval_strategy="no",
        save_strategy="no",
        do_train=True,
        do_eval=False,
        max_length=sequence["max_sequence_length"],
        packing=False,
        padding_free=optimization.get("padding_free", False),
        completion_only_loss=loss["completion_only_loss"],
        assistant_only_loss=loss["assistant_only_loss"],
        remove_unused_columns=True,
        run_name=config["experiment_name"],
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def _load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    """Load the pinned cached 4-bit model and attach the locked LoRA adapter."""
    from unsloth import FastLanguageModel

    model_config = config["model"]
    lora_config = config["lora"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["model_id"],
        revision=model_config["revision"],
        max_seq_length=model_config["max_sequence_length"],
        dtype=torch.float16,
        load_in_4bit=model_config["load_in_4bit"],
        load_in_8bit=model_config["load_in_8bit"],
        load_in_16bit=model_config["load_in_16bit"],
        device_map=model_config["device_map"],
        trust_remote_code=model_config["trust_remote_code"],
        use_gradient_checkpointing=False,
        local_files_only=model_config["local_files_only"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["rank"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing=lora_config["gradient_checkpointing"],
        random_state=lora_config["random_state"],
        use_rslora=lora_config["use_rslora"],
        loftq_config=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    return model, tokenizer


def _deduplicate_warnings(captured: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    """Keep warning categories and unique messages in the tracked report."""
    seen = set()
    output = []
    for warning in captured:
        item = {"category": warning.category.__name__, "message": str(warning.message)}
        key = (item["category"], item["message"])
        if key not in seen:
            output.append(item)
            seen.add(key)
    return output


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the compact tracked report without touching baseline artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compact_report(report: dict[str, Any]) -> None:
    """Keep aggregate evidence while removing bulky duplicate per-example traces."""
    for evaluation_key in ("pre_training_evaluation", "post_training_evaluation"):
        evaluation = report.get(evaluation_key)
        if not evaluation or "records" not in evaluation:
            continue
        records = evaluation.pop("records")
        evaluation["evaluated_record_count"] = len(records)
    label_report = report.get("label_mask_verification")
    if label_report and "per_example" in label_report:
        per_example = label_report.pop("per_example")
        label_report["verified_example_count"] = len(per_example)
        label_report["verification_sample"] = per_example[:4]


def run_sanity(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Run the complete preflight, 64-example overfit test, and tracked report."""
    config = _read_config(config_path)
    report_path = PROJECT_ROOT / config["artifacts"]["tracked_report_path"]
    train_path = PROJECT_ROOT / config["data"]["source_path"]
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "64-example train-only QLoRA overfit/sanity experiment; full training was not started.",
        "full_training_started": False,
        "validation_accessed": False,
        "test_accessed": False,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "configuration": config,
        "software_versions": _package_versions(),
        "errors": [],
    }
    model = None
    trainer = None
    tokenizer = None
    captured_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            captured_warnings = caught_warnings
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2060 SUPER":
                raise RuntimeError(f"Unexpected GPU detected: {torch.cuda.get_device_name(0)}")
            report["hardware"] = {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "cuda_runtime": torch.version.cuda,
            }
            expected_train_sha256 = "ac4642fb0adfeed9084e24fc35633477859fe0b59597c59cc7e7a5f3a539e133"
            actual_train_sha256 = _sha256_file(train_path)
            if actual_train_sha256 != expected_train_sha256:
                raise RuntimeError("Frozen train split hash does not match the approved manifest")
            report["data_boundary"] = {
                "train_path": str(train_path.relative_to(PROJECT_ROOT)),
                "train_sha256": actual_train_sha256,
                "validation_accessed": False,
                "test_accessed": False,
            }
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["model_id"],
                revision=config["model"]["revision"],
                trust_remote_code=config["model"]["trust_remote_code"],
                local_files_only=config["model"]["local_files_only"],
            )
            categories = tuple(config["data"]["target_categories"])
            examples = select_sanity_examples(
                train_path,
                tokenizer,
                categories=categories,
                examples_per_category=config["data"]["examples_per_category"],
                max_sequence_length=config["model"]["max_sequence_length"],
                short_sequence_token_ceiling=config["data"]["short_sequence_token_ceiling"],
                chat_template_kwargs=config["data"]["chat_template_kwargs"],
            )
            records = sanity_dataset_records(
                examples,
                chat_template_kwargs=config["data"]["chat_template_kwargs"],
            )
            counts = Counter(record["target_category"] for record in records)
            if counts != Counter({category: config["data"]["examples_per_category"] for category in categories}):
                raise AssertionError(f"Unexpected sanity class counts: {counts}")
            report["sanity_dataset"] = selection_report(examples)
            report["preflight"] = {
                "all_four_categories_have_exactly_16_examples": True,
                "all_selected_examples_are_train_split": all(
                    example.row["source_split"] == "train" for example in examples
                ),
                "max_sequence_length": config["model"]["max_sequence_length"],
                "short_sequence_ceiling": config["data"]["short_sequence_token_ceiling"],
            }
            dataset = to_dataset(records)
            model, tokenizer = _load_model_and_tokenizer(config)
            report["model"] = _trainable_parameter_report(model)
            report["model"]["parameter_devices"] = sorted({str(parameter.device) for parameter in model.parameters()})
            if report["model"]["parameter_devices"] != ["cuda:0"]:
                raise RuntimeError("Model parameters are not fully placed on cuda:0")
            if not report["model"]["only_lora_parameters_trainable"] or not report["model"]["base_parameters_frozen"]:
                raise RuntimeError("LoRA-only trainability preflight failed")
            FastLanguageModel.for_inference(model)
            pre_eval = evaluate_model(
                model,
                tokenizer,
                records,
                device=next(model.parameters()).device,
                categories=categories,
                max_length=config["model"]["max_sequence_length"],
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                chat_template_kwargs=config["data"]["chat_template_kwargs"],
            )
            report["pre_training_evaluation"] = pre_eval
            FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
            model.config.use_cache = False
            trainer = _build_trainer(model, tokenizer, dataset, config)
            label_report = _verify_trainer_labels(trainer, examples, tokenizer=tokenizer)
            report["label_mask_verification"] = label_report
            if not label_report["all_checks_passed"]:
                raise RuntimeError("Completion-only label/mask preflight failed")
            if not report["model"]["only_lora_parameters_trainable"]:
                raise RuntimeError("LoRA-only trainability check failed before training")
            torch.cuda.reset_peak_memory_stats()
            training_start = time.perf_counter()
            train_result = trainer.train()
            torch.cuda.synchronize()
            training_runtime = time.perf_counter() - training_start
            post_train_parameters = _trainable_parameter_report(trainer.model)
            FastLanguageModel.for_inference(trainer.model)
            post_eval = evaluate_model(
                trainer.model,
                tokenizer,
                records,
                device=next(trainer.model.parameters()).device,
                categories=categories,
                max_length=config["model"]["max_sequence_length"],
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                chat_template_kwargs=config["data"]["chat_template_kwargs"],
            )
            output_dir = PROJECT_ROOT / config["artifacts"]["temporary_output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            report["adapter_artifact"] = {
                "path": str(output_dir.relative_to(PROJECT_ROOT)),
                "saved": True,
                "temporary_and_git_ignored": True,
            }
            log_history = [
                {
                    key: value
                    for key, value in item.items()
                    if key in {"step", "epoch", "loss", "learning_rate"}
                }
                for item in trainer.state.log_history
                if "loss" in item
            ]
            report["training"] = {
                "train_result_metrics": train_result.metrics,
                "optimizer_steps": int(trainer.state.global_step),
                "configured_max_steps": config["optimization"]["sanity_max_steps"],
                "configured_sanity_num_epochs": config["optimization"]["sanity_num_epochs"],
                "runtime_seconds": round(training_runtime, 4),
                "peak_gpu_memory": _gpu_memory_snapshot(),
                "loss_progression": log_history,
                "post_training_parameter_report": post_train_parameters,
            }
            report["post_training_evaluation"] = post_eval
            pre_metrics = pre_eval["metrics"]
            post_metrics = post_eval["metrics"]
            report["overfit_assessment"] = {
                "success_criterion": "post accuracy >= 0.90, post macro-F1 >= 0.90, valid JSON >= 100%, and relative macro-F1 improvement >= 50%",
                "accuracy_delta": round(post_metrics["accuracy"] - pre_metrics["accuracy"], 6),
                "macro_f1_delta": round(post_metrics["macro_f1"] - pre_metrics["macro_f1"], 6),
                "macro_f1_relative_improvement": round(
                    (post_metrics["macro_f1"] - pre_metrics["macro_f1"]) / pre_metrics["macro_f1"],
                    6,
                ),
                "passed": bool(
                    post_metrics["accuracy"] >= 0.90
                    and post_metrics["macro_f1"] >= 0.90
                    and post_metrics["valid_output_percentage"] >= 100.0
                    and (
                        post_metrics["macro_f1"] - pre_metrics["macro_f1"]
                    ) / pre_metrics["macro_f1"] >= 0.50
                ),
            }
            report["status"] = "passed_sanity" if report["overfit_assessment"]["passed"] else "failed_overfit"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        report["warnings"] = _deduplicate_warnings(captured_warnings)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _compact_report(report)
        _write_report(report_path, report)
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main() -> None:
    """Run the train-only sanity experiment and print its final status."""
    report = run_sanity()
    print(json.dumps({"status": report["status"], "report_path": "results/training_sanity_check.json", "full_training_started": False}, indent=2))
    if report["status"] in {"failed", "failed_overfit"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
