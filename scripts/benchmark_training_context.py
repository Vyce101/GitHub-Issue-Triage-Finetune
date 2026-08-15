"""Benchmark train-only QLoRA memory and throughput at candidate context lengths."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import importlib.metadata
import json
import os
import re
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from baseline.config import MODEL_ID, MODEL_REVISION, TRAIN_SPLIT_PATH
from baseline.data import read_jsonl
from baseline.prompts import zero_shot_messages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "results/training_context_benchmark.json"
CONTEXT_LENGTHS = (512, 1024, 1536, 2048)
COMPUTE_DTYPE = torch.float16
WARMUP_STEPS = 1
MEASURED_STEPS = 3
BENCHMARK_LORA_RANK = 8
BENCHMARK_LORA_ALPHA = 16
BENCHMARK_LORA_DROPOUT = 0.0
BENCHMARK_LEARNING_RATE = 1e-4
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _parse_args() -> argparse.Namespace:
    """Parse optional controls for repeatable benchmark runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=list(CONTEXT_LENGTHS),
        choices=CONTEXT_LENGTHS,
        help="Candidate sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_STEPS,
        help="Unreported steps used to separate first-step compilation from steady state.",
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=MEASURED_STEPS,
        help="Reported steady-state optimizer steps per context length.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Output JSON path. Relative paths are resolved from the project root.",
    )
    return parser.parse_args()


def _ensure_positive_steps(args: argparse.Namespace) -> None:
    """Reject invalid step counts before loading the model."""
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.measured_steps < 1:
        raise ValueError("--measured-steps must be at least 1")


def _json_value(value: Any) -> Any:
    """Convert common torch and path values into JSON-compatible values."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _memory_snapshot() -> dict[str, int | float]:
    """Capture current CUDA memory and peak statistics in bytes and MiB."""
    allocated_bytes = int(torch.cuda.memory_allocated())
    reserved_bytes = int(torch.cuda.memory_reserved())
    max_allocated_bytes = int(torch.cuda.max_memory_allocated())
    max_reserved_bytes = int(torch.cuda.max_memory_reserved())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": allocated_bytes,
        "allocated_mib": round(allocated_bytes / 2**20, 2),
        "reserved_bytes": reserved_bytes,
        "reserved_mib": round(reserved_bytes / 2**20, 2),
        "peak_allocated_bytes": max_allocated_bytes,
        "peak_allocated_mib": round(max_allocated_bytes / 2**20, 2),
        "peak_reserved_bytes": max_reserved_bytes,
        "peak_reserved_mib": round(max_reserved_bytes / 2**20, 2),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "total_mib": round(int(total_bytes) / 2**20, 2),
    }


def _run_external_command(command: list[str]) -> dict[str, Any]:
    """Run a lightweight host diagnostic without affecting timed GPU steps."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _nvidia_smi_gpu_snapshot() -> dict[str, Any]:
    """Capture current total GPU memory usage and utilization from nvidia-smi."""
    query = _run_external_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    parsed = []
    for row in csv.reader(query["stdout"].splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        values = []
        for value in row[1:]:
            try:
                values.append(float(value))
            except ValueError:
                values.append(None)
        parsed.append(
            {
                "name": row[0].strip(),
                "memory_total_mib": values[0],
                "memory_used_mib": values[1],
                "memory_free_mib": values[2],
                "utilization_percent": values[3],
            }
        )
    compute_processes = _run_external_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "gpu_query": query,
        "gpus": parsed,
        "compute_process_query": compute_processes,
    }


class _MemoryStatusEx(ctypes.Structure):
    """Windows structure used to read system RAM without an extra dependency."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _system_ram_snapshot() -> dict[str, int | float | None]:
    """Capture total, used, and available physical RAM on Windows."""
    if os.name != "nt":
        return {"available": False, "reason": "Windows-only diagnostic"}
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"available": False, "reason": "GlobalMemoryStatusEx failed"}
    total_mib = status.ullTotalPhys / 2**20
    available_mib = status.ullAvailPhys / 2**20
    return {
        "available": True,
        "total_mib": round(total_mib, 2),
        "available_mib": round(available_mib, 2),
        "used_mib": round(total_mib - available_mib, 2),
        "used_percent": round(100 * (1 - available_mib / total_mib), 2),
    }


def _relevant_process_snapshot() -> dict[str, Any]:
    """List likely Python/model processes and their working-set sizes."""
    if os.name != "nt":
        return {"available": False, "reason": "Windows-only process diagnostic"}
    tasklist = _run_external_command(["tasklist", "/FO", "CSV", "/NH"])
    relevant_pattern = re.compile(
        r"(python|ollama|llama|comfy|text-generation|embedding|stable.?diffusion|pytorch|model)",
        re.IGNORECASE,
    )
    processes = []
    for row in csv.reader(tasklist["stdout"].splitlines()):
        if len(row) < 5 or not relevant_pattern.search(row[0]):
            continue
        memory_text = re.sub(r"[^0-9]", "", row[4])
        processes.append(
            {
                "image_name": row[0],
                "pid": row[1],
                "session_name": row[2],
                "session_number": row[3],
                "memory_working_set_mib": round(int(memory_text) / 1024, 2) if memory_text else None,
            }
        )
    return {
        "available": True,
        "tasklist_query": tasklist,
        "relevant_processes": processes,
    }


def _preflight_snapshot() -> dict[str, Any]:
    """Capture clean-system GPU, RAM, and relevant-process state before model loading."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": _nvidia_smi_gpu_snapshot(),
        "system_ram": _system_ram_snapshot(),
        "relevant_processes": _relevant_process_snapshot(),
    }


def _warning_records(captured: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    """Deduplicate warning text while retaining its warning category."""
    records = []
    seen = set()
    for warning in captured:
        record = {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        key = (record["category"], record["message"])
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _token_ids(encoded: Any) -> list[int]:
    """Extract one-dimensional token IDs from a tokenizer result."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one tokenized conversation")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    """Render one conversation with the locked tokenizer's supported template options."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    return _token_ids(encoded)


def _training_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the existing zero-shot classification prompt plus its normalized target."""
    return zero_shot_messages(row) + [
        {"role": "assistant", "content": f'{{"type":"{row["target_category"]}"}}'}
    ]


def _select_examples(train_rows: list[dict[str, Any]], tokenizer: Any) -> dict[int, dict[str, Any]]:
    """Select the shortest real train examples that reach each requested context length."""
    candidate_lengths: list[tuple[int, int]] = []
    for row_index, row in enumerate(train_rows):
        messages = _training_messages(row)
        prompt_ids = _apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
        full_ids = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("The tokenizer template did not preserve the prompt prefix")
        candidate_lengths.append((len(full_ids), row_index))

    selected: dict[int, dict[str, Any]] = {}
    for context_length in CONTEXT_LENGTHS:
        eligible = [candidate for candidate in candidate_lengths if candidate[0] >= context_length]
        if not eligible:
            raise ValueError(f"No train-only example reaches context length {context_length}")
        full_length, row_index = min(
            eligible,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        row = train_rows[row_index]
        messages = _training_messages(row)
        prompt_ids = _apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
        full_ids = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        prompt_length = len(prompt_ids)
        target_ids = full_ids[prompt_length:]
        prompt_budget = context_length - len(target_ids)
        if prompt_budget <= 0:
            raise ValueError(f"Target does not fit at context length {context_length}")
        sequence_ids = full_ids[:prompt_budget] + target_ids
        selected[context_length] = {
            "row": row,
            "row_index_in_train_file": row_index,
            "full_token_ids": full_ids,
            "prompt_token_count": prompt_length,
            "target_token_count": len(target_ids),
            "full_sequence_token_count": full_length,
            "fed_sequence_token_count": len(sequence_ids),
            "truncated_prompt_token_count": max(0, prompt_length - prompt_budget),
            "sequence_ids": sequence_ids,
            "label_start_index": min(prompt_length, prompt_budget),
        }
    return selected


def _trainable_parameter_count(model: Any) -> tuple[int, int]:
    """Return total and trainable parameter counts after attaching LoRA."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _device_placement(model: Any) -> dict[str, Any]:
    """Detect CPU or disk placement that would invalidate a pure-GPU comparison."""
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    device_map = _json_value(getattr(model, "hf_device_map", None))
    offload_entries = {}
    if isinstance(device_map, dict):
        offload_entries = {
            str(name): str(device)
            for name, device in device_map.items()
            if str(device).lower().startswith(("cpu", "disk"))
        }
    cpu_or_disk_devices = [
        device
        for device in parameter_devices
        if device.startswith("cpu") or device.startswith("meta")
    ]
    return {
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "cpu_buffers_present": any(device.startswith("cpu") for device in buffer_devices),
        "hf_device_map": device_map,
        "offload_device_map_entries": offload_entries,
        "cpu_or_disk_offload_detected": bool(cpu_or_disk_devices or offload_entries),
    }


def _load_trainable_model(context_length: int) -> tuple[Any, Any, dict[str, Any]]:
    """Load the locked 4-bit model and attach the benchmark-only LoRA adapter."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=context_length,
        dtype=COMPUTE_DTYPE,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        device_map="sequential",
        trust_remote_code=False,
        use_gradient_checkpointing=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=BENCHMARK_LORA_RANK,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=BENCHMARK_LORA_ALPHA,
        lora_dropout=BENCHMARK_LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    total_parameters, trainable_parameters = _trainable_parameter_count(model)
    placement = _device_placement(model)
    checkpoint_function = getattr(torch.utils.checkpoint, "checkpoint", None)
    checkpoint_function_name = getattr(checkpoint_function, "__name__", type(checkpoint_function).__name__)
    checkpoint_class = getattr(torch.utils.checkpoint, "CheckpointFunction", None)
    checkpoint_class_name = getattr(checkpoint_class, "__name__", type(checkpoint_class).__name__)
    return model, tokenizer, {
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "placement": placement,
        "gradient_checkpointing_requested": "unsloth",
        "gradient_checkpointing_enabled": bool(
            getattr(model, "is_gradient_checkpointing", False)
            or getattr(model, "gradient_checkpointing", False)
        ),
        "gradient_checkpointing_checkpoint_function": checkpoint_function_name,
        "gradient_checkpointing_checkpoint_class": checkpoint_class_name,
        "gradient_checkpointing_offloads_activations_to_cpu_ram": checkpoint_class_name
        == "UnslothCheckpointFunction",
    }


def _benchmark_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    sequence_ids: list[int],
    label_start_index: int,
    device: torch.device,
) -> float:
    """Run one FP16 forward, backward, and optimizer step on one real example."""
    input_ids = torch.tensor([sequence_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :label_start_index] = -100
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError(f"Training step produced a non-finite loss: {loss}")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return float(loss.detach().cpu())


def _cleanup_cuda_objects() -> None:
    """Release model, optimizer, and tokenizer references between context tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _benchmark_context(
    context_length: int,
    example: dict[str, Any],
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    """Run and report one context-length feasibility test, recovering after OOM."""
    result: dict[str, Any] = {
        "context_length": context_length,
        "status": "started",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": warmup_steps,
        "measured_steps_requested": measured_steps,
        "example": {
            "source_split": example["row"]["source_split"],
            "source_row_index": example["row"]["source_row_index"],
            "row_index_in_train_file": example["row_index_in_train_file"],
            "issue_id": example["row"]["issue_id"],
            "repository": example["row"]["repository"],
            "target_category": example["row"]["target_category"],
            "title": example["row"]["title"],
            "full_sequence_token_count": example["full_sequence_token_count"],
            "fed_sequence_token_count": example["fed_sequence_token_count"],
            "prompt_token_count": example["prompt_token_count"],
            "target_token_count": example["target_token_count"],
            "truncated_prompt_token_count": example["truncated_prompt_token_count"],
        },
        "cpu_or_ram_offloading": False,
        "warnings": [],
    }
    result["nvidia_smi_before_test"] = _nvidia_smi_gpu_snapshot()
    result["system_ram_before_test"] = _system_ram_snapshot()
    model = None
    tokenizer = None
    optimizer = None
    training_measurement_started = False
    captured_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            captured_warnings = caught_warnings
            torch.cuda.empty_cache()
            model, tokenizer, model_report = _load_trainable_model(context_length)
            result["model"] = model_report
            result["cpu_or_ram_offloading"] = bool(
                model_report["placement"]["cpu_or_disk_offload_detected"]
                or model_report["gradient_checkpointing_offloads_activations_to_cpu_ram"]
            )
            if (
                model_report["placement"]["cpu_or_disk_offload_detected"]
                or model_report["placement"]["parameter_devices"] != ["cuda:0"]
            ):
                raise RuntimeError(
                    "Model placement is not pure cuda:0; CPU/RAM offloading is not permitted for this benchmark"
                )
            result["gpu_memory_after_model_and_adapter_load"] = _memory_snapshot()
            result["nvidia_smi_after_model_load"] = _nvidia_smi_gpu_snapshot()
            result["system_ram_after_model_load"] = _system_ram_snapshot()
            torch.cuda.reset_peak_memory_stats()
            training_measurement_started = True
            device = next(model.parameters()).device
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable_parameters, lr=BENCHMARK_LEARNING_RATE)
            result["benchmark_optimizer"] = {
                "name": "AdamW",
                "learning_rate": BENCHMARK_LEARNING_RATE,
                "final_training_learning_rate": False,
            }

            warmup_start = time.perf_counter()
            warmup_losses = []
            for _ in range(warmup_steps):
                warmup_losses.append(
                    _benchmark_step(
                        model,
                        optimizer,
                        example["sequence_ids"],
                        example["label_start_index"],
                        device,
                    )
                )
            warmup_elapsed = time.perf_counter() - warmup_start
            result["warmup"] = {
                "elapsed_seconds": round(warmup_elapsed, 6),
                "losses": [round(loss, 6) for loss in warmup_losses],
                "compilation_or_first_step_time_accounted_separately": True,
            }

            steady_times = []
            steady_losses = []
            for _ in range(measured_steps):
                step_start = time.perf_counter()
                steady_losses.append(
                    _benchmark_step(
                        model,
                        optimizer,
                        example["sequence_ids"],
                        example["label_start_index"],
                        device,
                    )
                )
                steady_times.append(time.perf_counter() - step_start)
            steady_elapsed = sum(steady_times)
            processed_tokens = example["fed_sequence_token_count"] * measured_steps
            result["nvidia_smi_after_training"] = _nvidia_smi_gpu_snapshot()
            result["system_ram_after_training"] = _system_ram_snapshot()
            result["steady_state"] = {
                "completed_steps": measured_steps,
                "step_times_seconds": [round(value, 6) for value in steady_times],
                "seconds_per_step": round(steady_elapsed / measured_steps, 6),
                "losses": [round(loss, 6) for loss in steady_losses],
                "processed_tokens": processed_tokens,
                "tokens_per_second": round(processed_tokens / steady_elapsed, 4),
            }
            result["peak_gpu_memory_during_training"] = _memory_snapshot()
            result["status"] = "passed"
    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "cuda_out_of_memory": True,
        }
        if torch.cuda.is_available() and training_measurement_started:
            result["peak_gpu_memory_during_training"] = _memory_snapshot()
    except RuntimeError as error:
        error_text = str(error).lower()
        is_oom = "out of memory" in error_text or ("cuda error" in error_text and "memory" in error_text)
        result["status"] = "oom" if is_oom else "error"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "cuda_out_of_memory": is_oom,
        }
        if torch.cuda.is_available() and training_measurement_started:
            result["peak_gpu_memory_during_training"] = _memory_snapshot()
    except Exception as error:
        result["status"] = "error"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "cuda_out_of_memory": False,
        }
        if torch.cuda.is_available() and training_measurement_started:
            result["peak_gpu_memory_during_training"] = _memory_snapshot()
    finally:
        result["warnings"] = _warning_records(captured_warnings)
        del optimizer, model, tokenizer
        _cleanup_cuda_objects()
    return result


def _software_versions() -> dict[str, str | None]:
    """Record versions needed to reproduce the benchmark environment."""
    package_names = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "unsloth",
    )
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    versions["torch_cuda_runtime"] = torch.version.cuda
    return versions


def _percentage_difference(clean_value: float, preliminary_value: float) -> float | None:
    """Return signed clean-versus-preliminary percentage change."""
    if preliminary_value == 0:
        return None
    return round(100 * (clean_value - preliminary_value) / preliminary_value, 4)


def _compare_preliminary_results(clean_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare clean measurements with the immutable preliminary benchmark."""
    if not REPORT_PATH.exists():
        return {
            "available": False,
            "reason": f"Preliminary report not found at {REPORT_PATH}",
        }
    preliminary = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    preliminary_by_length = {
        int(result["context_length"]): result
        for result in preliminary.get("results", [])
    }
    comparisons = []
    for clean in clean_results:
        context_length = int(clean["context_length"])
        old = preliminary_by_length.get(context_length)
        if old is None or clean.get("status") != "passed" or old.get("status") != "passed":
            comparisons.append(
                {
                    "context_length": context_length,
                    "available": False,
                    "reason": "Both clean and preliminary results must be passed",
                }
            )
            continue
        clean_steady = clean["steady_state"]
        old_steady = old["steady_state"]
        clean_peak = clean["peak_gpu_memory_during_training"]
        old_peak = old["peak_gpu_memory_during_training"]
        comparisons.append(
            {
                "context_length": context_length,
                "available": True,
                "preliminary_seconds_per_step": old_steady["seconds_per_step"],
                "clean_seconds_per_step": clean_steady["seconds_per_step"],
                "seconds_per_step_difference_percent": _percentage_difference(
                    clean_steady["seconds_per_step"],
                    old_steady["seconds_per_step"],
                ),
                "preliminary_tokens_per_second": old_steady["tokens_per_second"],
                "clean_tokens_per_second": clean_steady["tokens_per_second"],
                "throughput_difference_percent": _percentage_difference(
                    clean_steady["tokens_per_second"],
                    old_steady["tokens_per_second"],
                ),
                "preliminary_peak_allocated_mib": old_peak["peak_allocated_mib"],
                "clean_peak_allocated_mib": clean_peak["peak_allocated_mib"],
                "peak_allocated_difference_mib": round(
                    clean_peak["peak_allocated_mib"] - old_peak["peak_allocated_mib"],
                    2,
                ),
                "preliminary_peak_reserved_mib": old_peak["peak_reserved_mib"],
                "clean_peak_reserved_mib": clean_peak["peak_reserved_mib"],
                "peak_reserved_difference_mib": round(
                    clean_peak["peak_reserved_mib"] - old_peak["peak_reserved_mib"],
                    2,
                ),
                "preliminary_cpu_or_ram_offloading": old.get("cpu_or_ram_offloading"),
                "clean_cpu_or_ram_offloading": clean.get("cpu_or_ram_offloading"),
            }
        )
    return {
        "available": True,
        "preliminary_report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "comparisons": comparisons,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print a concise table after all context tests finish."""
    print("context\tstatus\tpeak_allocated_mib\tpeak_reserved_mib\tseconds_per_step\ttokens_per_second\tcpu_or_ram_offloading")
    for result in results:
        peak = result.get("peak_gpu_memory_during_training", {})
        steady = result.get("steady_state", {})
        print(
            f"{result['context_length']}\t{result['status']}\t"
            f"{peak.get('peak_allocated_mib', 'n/a')}\t"
            f"{peak.get('peak_reserved_mib', 'n/a')}\t"
            f"{steady.get('seconds_per_step', 'n/a')}\t"
            f"{steady.get('tokens_per_second', 'n/a')}\t"
            f"{result.get('cpu_or_ram_offloading', False)}"
        )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run train-only example selection and all requested GPU feasibility tests."""
    _ensure_positive_steps(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2060 SUPER":
        raise RuntimeError(f"Unexpected GPU detected: {torch.cuda.get_device_name(0)}")
    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path = report_path.resolve()
    if report_path == REPORT_PATH.resolve() and REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite the preliminary report: {REPORT_PATH}")

    preflight = _preflight_snapshot()
    train_rows = read_jsonl(TRAIN_SPLIT_PATH)
    import unsloth  # noqa: F401  # Import before Transformers as required by Unsloth.

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    selected_examples = _select_examples(train_rows, tokenizer)
    del tokenizer
    gc.collect()

    context_lengths = tuple(dict.fromkeys(args.context_lengths))
    benchmark_results = [
        _benchmark_context(
            context_length,
            selected_examples[context_length],
            args.warmup_steps,
            args.measured_steps,
        )
        for context_length in context_lengths
    ]
    report = {
        "status": "passed" if all(result["status"] == "passed" for result in benchmark_results) else "completed_with_failures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "benchmark_scope": "Small train-only QLoRA memory and throughput feasibility benchmark; not the real fine-tuning run.",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 2),
            "compute_dtype": str(COMPUTE_DTYPE),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "cuda_runtime": torch.version.cuda,
        },
        "software_versions": _software_versions(),
        "preflight": preflight,
        "training_configuration": {
            "context_lengths_tested": list(context_lengths),
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "validation_enabled": False,
            "checkpoint_saving_enabled": False,
            "adapter_saved": False,
            "real_fine_tuning_run": False,
            "lora_configuration_is_benchmark_only": True,
            "lora_rank": BENCHMARK_LORA_RANK,
            "lora_alpha": BENCHMARK_LORA_ALPHA,
            "lora_dropout": BENCHMARK_LORA_DROPOUT,
            "lora_bias": "none",
            "lora_target_modules": list(LORA_TARGET_MODULES),
            "gradient_checkpointing": "unsloth",
            "cpu_ram_offloading_note": "Unsloth offloaded gradient checkpointing is reported when its checkpoint shim moves saved activations to CPU RAM; model parameters must remain on cuda:0.",
            "optimizer_learning_rate_is_benchmark_only": True,
        },
        "data_boundary": {
            "train_split_path": str(TRAIN_SPLIT_PATH.relative_to(PROJECT_ROOT)),
            "train_row_count": len(train_rows),
            "validation_accessed": False,
            "test_accessed": False,
            "selection_method": "Shortest rendered zero-shot train conversation reaching each context length; target tokens retained while only real issue-prompt tokens may be truncated to the requested length.",
            "prompt_format": "Existing baseline zero-shot system instruction and title/body prompt, followed by the normalized target JSON assistant response.",
            "raw_github_labels_in_model_input": False,
        },
        "comparison_to_preliminary": _compare_preliminary_results(benchmark_results),
        "results": benchmark_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(benchmark_results)
    print(f"Report written to {report_path}")
    return report


def main() -> None:
    """Run the reproducible context-length benchmark."""
    run_benchmark(_parse_args())


if __name__ == "__main__":
    main()
