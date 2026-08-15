"""Download, load, and functionally smoke-test the locked starting model."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, scan_cache_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
REPORT_PATH = PROJECT_ROOT / "results/model_smoke_test.json"
SMOKE_TEST_MAX_SEQUENCE_LENGTH = 2048
GENERATION_MAX_NEW_TOKENS = 64
COMPUTE_DTYPE = torch.float16
PROMPTS = (
    "Explain why the sky appears blue in one concise paragraph.",
    "Write a short Python function that returns the sum of a list of integers, then explain it briefly.",
)


def _json_value(value: Any) -> Any:
    """Convert model metadata objects into JSON-compatible values."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_text(value: str) -> str:
    """Hash text metadata without storing the full chat template in the report."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _gpu_memory() -> dict[str, int]:
    """Return current CUDA allocation and device memory information in bytes."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _model_license(model_info: Any) -> Any:
    """Read the model card license when the Hub exposes it."""
    card_data = getattr(model_info, "card_data", None)
    card_data = _json_value(card_data)
    if isinstance(card_data, dict):
        return card_data.get("license")
    return None


def _reported_download_size(model_info: Any) -> int | None:
    """Sum file sizes reported by the Hub when available."""
    total = 0
    found_size = False
    for sibling in getattr(model_info, "siblings", []) or []:
        lfs = getattr(sibling, "lfs", None)
        size = getattr(lfs, "size", None) if lfs is not None else None
        if size is None:
            size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
            found_size = True
    return total if found_size else None


def _cache_report() -> dict[str, Any]:
    """Report the local Hub cache entry for this model when it can be located."""
    try:
        cache_info = scan_cache_dir()
        matching_repos = [repo for repo in cache_info.repos if repo.repo_id == MODEL_ID]
        if not matching_repos:
            return {"found": False}
        repo = matching_repos[0]
        revisions = []
        for revision in repo.revisions:
            revisions.append(
                {
                    "commit_hash": getattr(revision, "commit_hash", None),
                    "size_on_disk_bytes": getattr(revision, "size_on_disk", None),
                }
            )
        return {
            "found": True,
            "repo_cache_directory_name": repo.repo_path.name,
            "size_on_disk_bytes": getattr(repo, "size_on_disk", None),
            "revisions": revisions,
        }
    except Exception as error:  # Cache scanning is supplementary to the smoke test.
        return {"found": False, "error": f"{type(error).__name__}: {error}"}


def _tokenizer_report(tokenizer: Any) -> dict[str, Any]:
    """Collect tokenizer identity and chat-template information."""
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": _json_value(getattr(tokenizer, "model_max_length", None)),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "special_tokens_map": _json_value(tokenizer.special_tokens_map),
        "chat_template_present": bool(chat_template),
        "chat_template_sha256": _sha256_text(chat_template) if isinstance(chat_template, str) else None,
    }


def _architecture_report(model: Any) -> dict[str, Any]:
    """Collect architecture and parameter information from the loaded model."""
    config = model.config
    try:
        parameter_count = int(model.num_parameters())
    except Exception:
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    quantization_config = getattr(config, "quantization_config", None)
    quantization_config = _json_value(quantization_config)
    if not isinstance(quantization_config, dict):
        quantization_config = {"value": quantization_config}
    return {
        "model_class": type(model).__name__,
        "config_model_type": getattr(config, "model_type", None),
        "architectures": _json_value(getattr(config, "architectures", None)),
        "parameter_count": parameter_count,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "config_torch_dtype": _json_value(getattr(config, "torch_dtype", None)),
        "quantization_config": quantization_config,
    }


def _device_report(model: Any) -> dict[str, Any]:
    """Verify that model parameters are placed on the CUDA device."""
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    device_map = _json_value(getattr(model, "hf_device_map", None))
    device_map_values = list(device_map.values()) if isinstance(device_map, dict) else []
    device_map_cpu_entries = [str(value) for value in device_map_values if str(value).startswith("cpu")]
    cuda_parameter_devices = [device for device in parameter_devices if device.startswith("cuda")]
    if not cuda_parameter_devices or device_map_cpu_entries:
        raise RuntimeError(
            f"Model was not fully placed on CUDA: parameter_devices={parameter_devices}, "
            f"cpu_device_map_entries={device_map_cpu_entries}"
        )
    return {
        "parameter_devices": parameter_devices,
        "hf_device_map": device_map,
        "cpu_device_map_entries": device_map_cpu_entries,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_index": 0,
        "cuda_device_verified": True,
    }


def _generate(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    """Generate one neutral instruction response and inspect its numerical output."""
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        thinking_disabled = True
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        thinking_disabled = False
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    input_token_count = int(inputs["input_ids"].shape[-1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    start_time = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    sequences = generated.sequences
    response_tokens = sequences[:, input_token_count:]
    response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True).strip()
    nonfinite_score_values = 0
    for score in generated.scores or ():
        nonfinite_score_values += int((~torch.isfinite(score)).sum().item())
    generated_token_count = int(response_tokens.shape[-1])
    if not response_text:
        raise RuntimeError(f"Prompt generated an empty response: {prompt}")
    if nonfinite_score_values:
        raise RuntimeError(f"Generation produced {nonfinite_score_values} non-finite score values")
    return {
        "prompt": prompt,
        "thinking_disabled": thinking_disabled,
        "input_token_count": input_token_count,
        "generated_token_count": generated_token_count,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": generated_token_count / elapsed_seconds if elapsed_seconds else None,
        "response": response_text,
        "nonfinite_score_values": nonfinite_score_values,
    }


def run_smoke_test() -> dict[str, Any]:
    """Download, load, generate, release, and report on the locked model."""
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model_id": MODEL_ID,
        "smoke_test_max_sequence_length": SMOKE_TEST_MAX_SEQUENCE_LENGTH,
        "final_training_context_length": False,
        "requested_quantization": "4-bit bitsandbytes NF4 model weights; unchanged",
        "requested_compute_dtype": str(COMPUTE_DTYPE),
        "prompts_are_classification_prompts": False,
        "training_performed": False,
        "lora_adapter_created": False,
    }
    model = None
    tokenizer = None
    captured_warnings: list[warnings.WarningMessage] = []
    error: Exception | None = None
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            captured_warnings = caught_warnings
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            bf16_supported_before_unsloth_import = bool(torch.cuda.is_bf16_supported())
            from unsloth import FastLanguageModel

            bf16_supported_after_unsloth_import = bool(torch.cuda.is_bf16_supported())
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": bf16_supported_after_unsloth_import,
                "bf16_supported_before_unsloth_import": bf16_supported_before_unsloth_import,
                "bf16_supported_after_unsloth_import": bf16_supported_after_unsloth_import,
                "bf16_value_used_for_this_stack": bf16_supported_after_unsloth_import,
                "cuda_runtime": torch.version.cuda,
            }
            if report["gpu"]["name"] != "NVIDIA GeForce RTX 2060 SUPER":
                raise RuntimeError(f"Unexpected GPU detected: {report['gpu']['name']}")
            api = HfApi()
            model_info = api.model_info(MODEL_ID, revision="main")
            resolved_revision = model_info.sha
            if not resolved_revision:
                raise RuntimeError("Hugging Face did not return a resolved model commit SHA")
            report["huggingface"] = {
                "resolved_revision": resolved_revision,
                "license": _model_license(model_info),
                "reported_download_size_bytes": _reported_download_size(model_info),
            }
            report["gpu_memory_before_loading"] = _gpu_memory()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_ID,
                revision=resolved_revision,
                max_seq_length=SMOKE_TEST_MAX_SEQUENCE_LENGTH,
                dtype=COMPUTE_DTYPE,
                load_in_4bit=True,
                load_in_8bit=False,
                load_in_16bit=False,
                device_map="sequential",
                trust_remote_code=False,
                use_gradient_checkpointing=False,
            )
            torch.cuda.synchronize()
            report["model_architecture"] = _architecture_report(model)
            report["tokenizer"] = _tokenizer_report(tokenizer)
            report["device"] = _device_report(model)
            quantization_config = report["model_architecture"]["quantization_config"]
            report["quantization"] = {
                "requested_load_in_4bit": True,
                "requested_quant_type": "nf4",
                "config": quantization_config,
                "compute_dtype_actual": quantization_config.get("bnb_4bit_compute_dtype"),
            }
            report["gpu_memory_after_loading"] = _gpu_memory()
            report["cache"] = _cache_report()
            FastLanguageModel.for_inference(model)
            torch.cuda.reset_peak_memory_stats()
            report["generations"] = [
                _generate(model, tokenizer, prompt, next(model.parameters()).device)
                for prompt in PROMPTS
            ]
            report["peak_gpu_memory_during_generation"] = {
                "allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
            report["status"] = "passed"
    except Exception as caught_error:
        error = caught_error
        report["status"] = "failed"
        report["error"] = f"{type(caught_error).__name__}: {caught_error}"
    finally:
        had_loaded_objects = model is not None or tokenizer is not None
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            report["gpu_memory_after_release"] = _gpu_memory()
        report["model_released"] = had_loaded_objects
        report["warnings"] = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in captured_warnings
        ]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if error is not None:
        raise error
    return report


def main() -> None:
    """Run the model smoke test and print its compact outcome."""
    report = run_smoke_test()
    print(json.dumps({
        "status": report["status"],
        "requested_model_id": report["requested_model_id"],
        "resolved_revision": report["huggingface"]["resolved_revision"],
        "gpu": report["gpu"]["name"],
        "generations": report["generations"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
