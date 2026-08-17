"""Verify and load the locked base model and the completed adapter without merging it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import (
    ADAPTER_PATH,
    CONFIG_PATH,
    MODEL_ID,
    MODEL_LOAD_MAX_SEQUENCE_LENGTH,
    MODEL_REVISION,
    TARGET_CATEGORIES,
    TRAINING_REPORT_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one local JSON object."""
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _relative(path: Path) -> str:
    """Return a repository-relative path with stable separators."""
    return str(path.relative_to(CONFIG_PATH.parents[1])).replace("\\", "/")


def verify_adapter_artifact() -> dict[str, Any]:
    """Verify adapter metadata and PEFT settings against the completed training report."""
    approved_config = _read_json(CONFIG_PATH)
    training_report = _read_json(TRAINING_REPORT_PATH)
    metadata = _read_json(ADAPTER_PATH / "adapter_metadata.json")
    adapter_config = _read_json(ADAPTER_PATH / "adapter_config.json")

    expected_base = approved_config["base_model"]
    expected_lora = approved_config["lora"]
    expected_sequence = approved_config["sequence"]
    training_configuration = training_report["configuration"]
    files = sorted(path.name for path in ADAPTER_PATH.iterdir() if path.is_file())
    forbidden_full_model_weights = [
        name for name in files
        if name in {"pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"}
    ]

    checks = {
        "training_report_passed": training_report.get("status") == "passed",
        "full_training_started": training_report.get("full_training_started") is True,
        "training_report_did_not_access_test": training_report.get("test_accessed") is False,
        "adapter_directory_exists": ADAPTER_PATH.is_dir(),
        "adapter_weights_exist": (ADAPTER_PATH / "adapter_model.safetensors").is_file(),
        "adapter_is_not_a_standalone_full_model": not forbidden_full_model_weights,
        "metadata_base_model_matches": metadata.get("base_model_id") == MODEL_ID,
        "metadata_base_revision_matches": metadata.get("base_model_revision") == MODEL_REVISION,
        "metadata_lora_matches_approved_config": metadata.get("lora") == expected_lora,
        "metadata_sequence_matches_approved_config": metadata.get("max_sequence_length") == expected_sequence["max_length"],
        "metadata_training_rows_match": metadata.get("train_row_count") == approved_config["data"]["row_counts"]["train"],
        "adapter_base_model_matches": adapter_config.get("base_model_name_or_path") == MODEL_ID,
        "adapter_is_lora": adapter_config.get("peft_type") == "LORA",
        "adapter_rank_matches": adapter_config.get("r") == expected_lora["rank"],
        "adapter_alpha_matches": adapter_config.get("lora_alpha") == expected_lora["alpha"],
        "adapter_dropout_matches": adapter_config.get("lora_dropout") == expected_lora["dropout"],
        "adapter_bias_matches": adapter_config.get("bias") == expected_lora["bias"],
        "adapter_targets_match": set(adapter_config.get("target_modules", [])) == set(expected_lora["target_modules"]),
        "adapter_rslora_matches": adapter_config.get("use_rslora") == expected_lora["use_rslora"],
        "adapter_modules_to_save_match": adapter_config.get("modules_to_save") == expected_lora["modules_to_save"],
        "training_configuration_matches_approved_config": training_configuration == approved_config,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Adapter verification failed: {checks}")

    return {
        "adapter_path": _relative(ADAPTER_PATH),
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "adapter_config": {
            "peft_type": adapter_config["peft_type"],
            "r": adapter_config["r"],
            "lora_alpha": adapter_config["lora_alpha"],
            "lora_dropout": adapter_config["lora_dropout"],
            "bias": adapter_config["bias"],
            "target_modules": sorted(adapter_config["target_modules"]),
            "use_rslora": adapter_config["use_rslora"],
            "modules_to_save": adapter_config["modules_to_save"],
        },
        "checks": checks,
        "all_checks_passed": True,
        "standalone_full_model_weight_files": forbidden_full_model_weights,
        "training_report_path": _relative(TRAINING_REPORT_PATH),
        "approved_config_path": _relative(CONFIG_PATH),
        "target_categories": list(TARGET_CATEGORIES),
    }


def _assert_cuda_model(model: Any) -> None:
    """Ensure the quantized model is fully resident on the expected GPU."""
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if devices != ["cuda:0"]:
        raise RuntimeError(f"Model parameters are not fully placed on cuda:0: {devices}")


def load_base_model() -> tuple[Any, Any, dict[str, Any]]:
    """Load the locked 4-bit base checkpoint at the frozen revision."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=MODEL_LOAD_MAX_SEQUENCE_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        device_map="sequential",
        trust_remote_code=False,
        use_gradient_checkpointing=False,
    )
    FastLanguageModel.for_inference(model)
    _assert_cuda_model(model)
    return model, tokenizer, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "load_in_16bit": False,
        "quantization": {"quant_type": "nf4", "use_double_quant": True, "compute_dtype": "float16"},
        "device_map": "sequential",
        "max_sequence_length": MODEL_LOAD_MAX_SEQUENCE_LENGTH,
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "model_class": type(model).__name__,
    }


def load_adapter_model() -> tuple[Any, Any, dict[str, Any]]:
    """Load the locked base and attach the LoRA adapter as an unmerged PEFT model."""
    from peft import PeftModel

    from unsloth import FastLanguageModel

    adapter_verification = verify_adapter_artifact()
    base_model, tokenizer, base_load = load_base_model()
    adapter_model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), is_trainable=False)
    FastLanguageModel.for_inference(adapter_model)
    _assert_cuda_model(adapter_model)

    if not isinstance(adapter_model, PeftModel):
        raise RuntimeError("Adapter reload did not produce a PEFT model")
    adapter_module_names = [name for name, _module in adapter_model.named_modules() if "lora_" in name]
    if not adapter_module_names:
        raise RuntimeError("Reloaded adapter has no visible LoRA modules")

    return adapter_model, tokenizer, {
        "base_model_load": base_load,
        "adapter_verification": adapter_verification,
        "adapter_reload_succeeded": True,
        "peft_model_class": type(adapter_model).__name__,
        "adapter_module_count": len(adapter_module_names),
        "merged_into_standalone_model": False,
        "is_trainable": False,
        "parameter_devices": sorted({str(parameter.device) for parameter in adapter_model.parameters()}),
    }
