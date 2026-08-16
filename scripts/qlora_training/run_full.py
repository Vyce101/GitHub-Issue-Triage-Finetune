"""Run the approved one-epoch full-data QLoRA experiment and write its audit report."""

from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .full_data import (
    SplitDatasetBuild,
    VerificationSample,
    build_tokenized_split_dataset,
    expected_optimizer_steps,
    load_split_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/initial_qlora_config.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results/initial_qlora_training.json"
EXPECTED_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
EXPECTED_MODEL_REVISION = "7744afa8566e264af1a92a806d8d9aae00cc7c78"
EXPECTED_TRAIN_SHA256 = "ac4642fb0adfeed9084e24fc35633477859fe0b59597c59cc7e7a5f3a539e133"
EXPECTED_TRAIN_ROWS = 31_876
EXPECTED_GPU = "NVIDIA GeForce RTX 2060 SUPER"
EXPECTED_TORCH_CUDA_RUNTIME = "13.0"
EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "unsloth": "2026.8.18",
    "unsloth-zoo": "2026.8.12",
    "transformers": "5.5.0",
    "datasets": "4.3.0",
    "trl": "0.24.0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "xformers": "0.0.35",
    "triton": None,
    "triton-windows": "3.7.1.post27",
}


def _relative_path(path: Path) -> str:
    """Represent repository paths without exposing machine-specific absolute paths."""
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def _read_config(config_path: Path) -> dict[str, Any]:
    """Load the approved configuration and reject a non-initial-run configuration."""
    if config_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError("The full runner only accepts configs/initial_qlora_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "approved_initial_run_pending_final_review":
        raise ValueError("The approved initial QLoRA configuration status is not pending final review")
    if config.get("training_performed") or config.get("full_training_started"):
        raise ValueError("The approved initial configuration is already marked as used")
    base_model = config["base_model"]
    if base_model["model_id"] != EXPECTED_MODEL_ID or base_model["revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError("The configured model ID or revision is not the locked approved checkpoint")
    if base_model["load_in_4bit"] is not True or base_model["load_in_8bit"] or base_model["load_in_16bit"]:
        raise ValueError("The approved run must use 4-bit loading only")
    if base_model["quantization"] != {
        "quant_type": "nf4",
        "use_double_quant": True,
        "compute_dtype": "float16",
    }:
        raise ValueError("The approved NF4 quantization settings were changed")
    if config["data"]["train_all_frozen_train_rows"] is not True:
        raise ValueError("The approved run must use every frozen train row")
    if config["data"]["class_sampling"] != "natural_frozen_train_distribution; no oversampling or class-weighted loss":
        raise ValueError("The approved class sampling policy was changed")
    if config["sequence"]["truncate_prompt_only_preserve_target"] is not True:
        raise ValueError("The approved run must preserve the target while truncating the issue prompt")
    if config["sequence"]["packing"] is not False:
        raise ValueError("Packing is not approved for the initial run")
    if config["loss"]["completion_only_loss"] is not True or config["loss"]["train_on_assistant_response_only"] is not True:
        raise ValueError("The approved run must train on the assistant completion only")
    if config["optimization"]["num_train_epochs"] != 1.0:
        raise ValueError("The approved run must be exactly one epoch")
    return config


def _sha256_file(path: Path) -> str:
    """Hash one split without touching any other split file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    """Read the installed project package versions without changing the environment."""
    versions: dict[str, str | None] = {}
    for package_name in EXPECTED_PACKAGE_VERSIONS:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _run_command(command: list[str]) -> dict[str, Any]:
    """Run a read-only diagnostic command and preserve its result."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _memory_status() -> dict[str, Any]:
    """Read Windows physical-memory usage without adding a runtime dependency."""
    class MemoryStatusEx(ctypes.Structure):
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

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"available": False}
    return {
        "available": True,
        "total_mib": round(status.ullTotalPhys / 2**20, 2),
        "available_mib": round(status.ullAvailPhys / 2**20, 2),
        "used_percent": round(status.dwMemoryLoad, 2),
    }


def _parse_tasklist(output: str) -> list[dict[str, str]]:
    """Parse tasklist CSV output into compact process records."""
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) >= 2:
            rows.append({"image_name": row[0], "pid": row[1], "memory": row[4] if len(row) > 4 else ""})
    return rows


def _resource_preflight() -> dict[str, Any]:
    """Capture GPU/RAM/process state and reject unrelated substantial model work."""
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    full_smi = _run_command(["nvidia-smi"])
    compute_apps = _run_command(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
    )
    tasklist = _run_command(["tasklist", "/FO", "CSV", "/NH"])
    relevant_processes = [
        process
        for process in _parse_tasklist(tasklist["stdout"])
        if any(
            marker in process["image_name"].lower()
            for marker in ("python", "ollama", "comfy", "stable", "jupyter", "torch", "cuda")
        )
    ]
    ollama = _run_command(["ollama", "ps"])
    ollama_lines = [line for line in ollama["stdout"].splitlines() if line.strip()][1:]
    numeric_gpu_workloads = []
    for line in compute_apps["stdout"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[2].replace(".", "", 1).isdigit():
            numeric_gpu_workloads.append(fields)
    substantial_unrelated_workload = bool(ollama_lines or numeric_gpu_workloads)
    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_query": gpu_query,
        "nvidia_smi": {
            "returncode": full_smi["returncode"],
            "first_lines": full_smi["stdout"].splitlines()[:24],
            "stderr": full_smi["stderr"],
        },
        "compute_apps": compute_apps,
        "system_ram": _memory_status(),
        "relevant_processes": relevant_processes,
        "ollama_ps": ollama,
        "substantial_unrelated_model_workload": substantial_unrelated_workload,
    }
    if gpu_query["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {gpu_query}")
    if substantial_unrelated_workload:
        raise RuntimeError(f"An unrelated model workload is active: {snapshot}")
    return snapshot


def _git_snapshot() -> dict[str, Any]:
    """Record the starting commit and reject unrelated or unsafe working-tree changes."""
    status_result = _run_command(["git", "status", "--short"])
    sha_result = _run_command(["git", "rev-parse", "HEAD"])
    check_result = _run_command(["git", "diff", "--check"])
    status_lines = [line for line in status_result["stdout"].splitlines() if line.strip()]
    unsafe_paths = []
    allowed_prefixes = ("scripts/qlora_training/", "scripts/run_initial_qlora_training.py")
    for line in status_lines:
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.replace("\\", "/")
        if not path.startswith(allowed_prefixes):
            unsafe_paths.append(path)
    snapshot = {
        "starting_commit_sha": sha_result["stdout"],
        "status_lines": status_lines,
        "diff_check": check_result,
        "acceptable_for_experiment": (
            status_result["returncode"] == 0
            and sha_result["returncode"] == 0
            and check_result["returncode"] == 0
            and not unsafe_paths
        ),
        "unsafe_paths": unsafe_paths,
    }
    if not snapshot["acceptable_for_experiment"]:
        raise RuntimeError(f"Git working tree is not acceptable for the run: {snapshot}")
    return snapshot


def _cuda_memory_snapshot() -> dict[str, int | float]:
    """Capture current and peak CUDA memory in bytes and MiB."""
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


def _parameter_report(model: Any) -> dict[str, Any]:
    """Verify that only the configured LoRA parameters are trainable."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_lora = [name for name in trainable_names if "lora_" not in name.lower()]
    target_modules = {
        name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1]
        for name in trainable_names
        if ".lora_" in name
    }
    base_frozen = all(
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
        "only_lora_parameters_trainable": not non_lora,
        "base_parameters_frozen": base_frozen,
        "non_lora_trainable_names": non_lora,
        "trainable_target_modules": sorted(target_modules),
    }


def _runtime_integrity() -> dict[str, Any]:
    """Verify the project interpreter, locked packages, CUDA, NF4, and Unsloth."""
    executable = Path(sys.executable).resolve()
    expected_executable = (PROJECT_ROOT / ".venv/Scripts/python.exe").resolve()
    versions = _package_versions()
    runtime = {
        "sys_executable": str(executable),
        "sys_version": sys.version,
        "executable_is_project_venv": executable == expected_executable,
        "python_version_is_expected": sys.version_info[:3] == (3, 11, 13),
        "package_versions": versions,
        "package_versions_match_recorded": versions == EXPECTED_PACKAGE_VERSIONS,
    }
    if not runtime["executable_is_project_venv"] or not runtime["python_version_is_expected"]:
        raise RuntimeError(f"The runner is not using the expected project Python: {runtime}")
    if not runtime["package_versions_match_recorded"]:
        raise RuntimeError(f"Project package versions changed: {runtime}")
    if not torch.cuda.is_available():
        raise RuntimeError("Project CUDA is unavailable")
    if torch.cuda.get_device_name(0) != EXPECTED_GPU or torch.version.cuda != EXPECTED_TORCH_CUDA_RUNTIME:
        raise RuntimeError("The project GPU or PyTorch CUDA runtime changed")
    runtime["cuda"] = {
        "is_available": True,
        "device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "bf16_supported_before_unsloth": bool(torch.cuda.is_bf16_supported()),
    }

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    runtime["cuda"]["tensor_operation"] = {"status": "PASS", "result": product.cpu().tolist()}

    import bitsandbytes as bnb
    from bitsandbytes import functional as bnb_functional
    from bitsandbytes.nn import Linear4bit, Params4bit

    bnb_report: dict[str, Any] = {"version": getattr(bnb, "__version__", None)}
    probe_weight = torch.randn(8, 64, dtype=torch.float16, device="cuda")
    packed, quant_state = bnb_functional.quantize_4bit(
        probe_weight,
        quant_type="nf4",
        compress_statistics=True,
    )
    restored = bnb_functional.dequantize_4bit(packed, quant_state)
    bnb_report["functional_nf4"] = {
        "status": "PASS",
        "packed_shape": list(packed.shape),
        "restored_shape": list(restored.shape),
    }
    layer = Linear4bit(64, 8, bias=False, compute_dtype=torch.float16).cuda()
    layer.weight = Params4bit(
        torch.randn(8, 64, dtype=torch.float16, device="cuda"),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    ).cuda()
    output = layer(torch.randn(4, 64, dtype=torch.float16, device="cuda"))
    torch.cuda.synchronize()
    bnb_report["linear4bit_nf4"] = {
        "status": "PASS",
        "input_shape": [4, 64],
        "output_shape": list(output.shape),
    }
    runtime["bitsandbytes_cuda_nf4"] = bnb_report
    del layer, output, packed, quant_state, restored, probe_weight, left, right, product
    gc.collect()
    torch.cuda.empty_cache()

    import unsloth
    import unsloth_zoo

    detection = {}
    for module in (unsloth, unsloth_zoo):
        for key in ("DEVICE_TYPE", "DEVICE_COUNT", "ALLOW_BITSANDBYTES", "CUDA_VERSION", "COMPUTE_CAPABILITY"):
            value = getattr(module, key, None)
            if value is not None:
                detection[f"{module.__name__}.{key}"] = value
    runtime["unsloth"] = {
        "version": versions["unsloth"],
        "unsloth_zoo_version": versions["unsloth-zoo"],
        "device_detection": detection,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported_after_unsloth": bool(torch.cuda.is_bf16_supported()),
        "initialization": "PASS",
    }
    if not runtime["unsloth"]["device_detection"].get("unsloth.DEVICE_TYPE") == "cuda":
        raise RuntimeError("Unsloth did not detect CUDA")
    if runtime["unsloth"]["bf16_supported_after_unsloth"]:
        raise RuntimeError("Unsloth/PyTorch BF16 behavior changed; approved FP16-only run is no longer valid")
    return runtime


def _data_boundary(config: dict[str, Any], tokenizer: Any) -> tuple[dict[str, Any], SplitDatasetBuild, SplitDatasetBuild]:
    """Load only the approved train/validation inputs and build separate datasets."""
    data_config = config["data"]
    train_path = PROJECT_ROOT / data_config["train_path"]
    validation_path = PROJECT_ROOT / data_config["validation_path"]
    test_path = PROJECT_ROOT / data_config["test_path"]
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("The approved train or validation split is missing")
    if train_path.resolve() in {validation_path.resolve(), test_path.resolve()}:
        raise ValueError("The training path overlaps another configured split")
    train_hash = _sha256_file(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256 or train_hash != data_config["train_sha256"]:
        raise RuntimeError(f"Frozen train split hash mismatch: {train_hash}")

    manifest_path = PROJECT_ROOT / data_config["frozen_split_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = manifest["splits"]["train"]
    if train_manifest["sha256"] != train_hash or train_manifest["row_count"] != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("The train split does not match the frozen manifest")
    categories = tuple(data_config["target_categories"])
    train_rows = load_split_rows(train_path, "train")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(train_rows) != data_config["row_counts"]["train"]:
        raise RuntimeError(f"Expected exactly {EXPECTED_TRAIN_ROWS} train rows, found {len(train_rows)}")
    train_repositories = set(manifest["repository_assignment"]["train"])
    train_repository_errors = [
        row["repository"] for row in train_rows if row.get("repository") not in train_repositories
    ]
    if train_repository_errors:
        raise RuntimeError(f"Train data contains a repository outside the frozen train assignment: {train_repository_errors[:3]}")
    if any(row.get("source_split") != "train" for row in train_rows):
        raise RuntimeError("The optimization dataset contains a non-train source_split")

    validation_rows = load_split_rows(validation_path, "train")
    if len(validation_rows) != data_config["row_counts"]["validation"]:
        raise RuntimeError("The validation row count does not match the approved configuration")
    validation_repositories = set(manifest["repository_assignment"]["validation"])
    if any(row.get("repository") not in validation_repositories for row in validation_rows):
        raise RuntimeError("The validation data contains a repository outside its frozen assignment")

    train_build = build_tokenized_split_dataset(
        train_rows,
        tokenizer,
        expected_split="train",
        categories=categories,
        max_length=config["sequence"]["max_length"],
        chat_template_kwargs={"enable_thinking": config["loss"]["chat_template_enable_thinking"]},
    )
    validation_build = build_tokenized_split_dataset(
        validation_rows,
        tokenizer,
        expected_split="validation",
        categories=categories,
        max_length=config["sequence"]["max_length"],
        chat_template_kwargs={"enable_thinking": config["loss"]["chat_template_enable_thinking"]},
    )
    boundary = {
        "train_path": _relative_path(train_path),
        "train_sha256": train_hash,
        "train_row_count": len(train_rows),
        "train_source_split_values": ["train"],
        "train_repository_count": len(train_repositories),
        "optimization_dataset_is_train_only": True,
        "validation_path": _relative_path(validation_path),
        "validation_sha256_from_config": data_config["validation_sha256"],
        "validation_row_count": len(validation_rows),
        "validation_used_for_epoch_reporting": True,
        "test_path": _relative_path(test_path),
        "validation_accessed": True,
        "test_accessed": False,
        "test_file_opened": False,
        "raw_github_labels_in_model_input": data_config["raw_github_labels_in_model_input"],
        "class_sampling": data_config["class_sampling"],
        "train_class_counts_match_manifest": train_build.stats["class_counts"] == train_manifest["class_counts"],
    }
    if not boundary["train_class_counts_match_manifest"]:
        raise RuntimeError("The rendered train class counts do not match the frozen manifest")
    del train_rows, validation_rows, manifest
    return boundary, train_build, validation_build


def _load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Load the local locked checkpoint and attach the approved LoRA adapter."""
    from huggingface_hub import snapshot_download
    from unsloth import FastLanguageModel

    base_model = config["base_model"]
    snapshot = snapshot_download(
        repo_id=base_model["model_id"],
        revision=base_model["revision"],
        local_files_only=True,
    )
    if Path(snapshot).name != base_model["revision"]:
        raise RuntimeError(f"The local model snapshot does not match revision {base_model['revision']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model["model_id"],
        revision=base_model["revision"],
        max_seq_length=config["sequence"]["max_length"],
        dtype=torch.float16,
        load_in_4bit=base_model["load_in_4bit"],
        load_in_8bit=base_model["load_in_8bit"],
        load_in_16bit=base_model["load_in_16bit"],
        device_map=base_model["device_map"],
        trust_remote_code=base_model["trust_remote_code"],
        use_gradient_checkpointing=False,
        local_files_only=True,
    )
    lora = config["lora"]
    optimization = config["optimization"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["rank"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing=optimization["gradient_checkpointing"],
        random_state=optimization["seed"],
        use_rslora=lora["use_rslora"],
        loftq_config=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = config["optimization"]["use_cache"]
    FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    return model, tokenizer, {
        "local_snapshot_revision": Path(snapshot).name,
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "parameter_report": _parameter_report(model),
    }


def _build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    config: dict[str, Any],
) -> Any:
    """Build the tested TRL SFT trainer from the approved machine-readable settings."""
    from trl import SFTConfig, SFTTrainer

    optimization = config["optimization"]
    trainer_config = config["trainer"]
    sequence = config["sequence"]
    loss = config["loss"]
    args = SFTConfig(
        output_dir=str(PROJECT_ROOT / trainer_config["output_dir"]),
        per_device_train_batch_size=optimization["per_device_train_batch_size"],
        per_device_eval_batch_size=trainer_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
        num_train_epochs=optimization["num_train_epochs"],
        learning_rate=optimization["learning_rate"],
        lr_scheduler_type=optimization["lr_scheduler_type"],
        optim=optimization["optimizer"],
        weight_decay=optimization["weight_decay"],
        warmup_ratio=optimization["warmup_ratio"],
        max_grad_norm=optimization["max_grad_norm"],
        gradient_checkpointing=True,
        use_cache=optimization["use_cache"],
        fp16=optimization["fp16"],
        bf16=optimization["bf16"],
        tf32=optimization["tf32"],
        seed=optimization["seed"],
        data_seed=optimization["data_seed"],
        dataloader_num_workers=optimization["dataloader_num_workers"],
        logging_strategy=trainer_config["logging_strategy"],
        logging_steps=trainer_config["logging_steps"],
        report_to=trainer_config["report_to"],
        run_name=config["experiment_name"],
        eval_strategy=trainer_config["eval_strategy"],
        prediction_loss_only=trainer_config["prediction_loss_only"],
        save_strategy=trainer_config["save_strategy"],
        save_total_limit=trainer_config["save_total_limit"],
        load_best_model_at_end=trainer_config["load_best_model_at_end"],
        save_only_model=trainer_config["save_only_model"],
        do_train=True,
        do_eval=True,
        max_length=sequence["max_length"],
        packing=sequence["packing"],
        padding_free=False,
        completion_only_loss=loss["completion_only_loss"],
        assistant_only_loss=False,
        remove_unused_columns=True,
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )


def _enum_value(value: Any) -> Any:
    """Normalize Transformers enum values for JSON and equality checks."""
    return getattr(value, "value", value)


def _resolved_hyperparameters(trainer: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Compare the trainer's resolved values with every approved run setting."""
    args = trainer.args
    optimization = config["optimization"]
    trainer_config = config["trainer"]
    sequence = config["sequence"]
    loss = config["loss"]
    resolved = {
        "max_length": args.max_length,
        "packing": args.packing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": _enum_value(args.lr_scheduler_type),
        "optimizer": _enum_value(args.optim),
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "gradient_checkpointing": "unsloth" if args.gradient_checkpointing else False,
        "use_cache": config["optimization"]["use_cache"],
        "fp16": args.fp16,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "eval_strategy": _enum_value(args.eval_strategy),
        "prediction_loss_only": args.prediction_loss_only,
        "save_strategy": _enum_value(args.save_strategy),
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "save_only_model": args.save_only_model,
        "logging_strategy": _enum_value(args.logging_strategy),
        "logging_steps": args.logging_steps,
        "report_to": "none" if not args.report_to else args.report_to,
        "padding_free": args.padding_free,
        "completion_only_loss": args.completion_only_loss,
        "assistant_only_loss": args.assistant_only_loss,
    }
    expected = {
        "max_length": sequence["max_length"],
        "packing": sequence["packing"],
        "per_device_train_batch_size": optimization["per_device_train_batch_size"],
        "per_device_eval_batch_size": trainer_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": optimization["gradient_accumulation_steps"],
        "effective_batch_size": optimization["effective_batch_size"],
        "num_train_epochs": optimization["num_train_epochs"],
        "learning_rate": optimization["learning_rate"],
        "lr_scheduler_type": optimization["lr_scheduler_type"],
        "optimizer": optimization["optimizer"],
        "weight_decay": optimization["weight_decay"],
        "warmup_ratio": optimization["warmup_ratio"],
        "max_grad_norm": optimization["max_grad_norm"],
        "gradient_checkpointing": optimization["gradient_checkpointing"],
        "use_cache": optimization["use_cache"],
        "fp16": optimization["fp16"],
        "bf16": optimization["bf16"],
        "tf32": optimization["tf32"],
        "seed": optimization["seed"],
        "data_seed": optimization["data_seed"],
        "dataloader_num_workers": optimization["dataloader_num_workers"],
        "eval_strategy": trainer_config["eval_strategy"],
        "prediction_loss_only": trainer_config["prediction_loss_only"],
        "save_strategy": trainer_config["save_strategy"],
        "save_total_limit": trainer_config["save_total_limit"],
        "load_best_model_at_end": trainer_config["load_best_model_at_end"],
        "save_only_model": trainer_config["save_only_model"],
        "logging_strategy": trainer_config["logging_strategy"],
        "logging_steps": trainer_config["logging_steps"],
        "report_to": trainer_config["report_to"],
        "padding_free": False,
        "completion_only_loss": loss["completion_only_loss"],
        "assistant_only_loss": False,
    }
    return {
        "resolved": resolved,
        "expected": expected,
        "matches_approved_configuration": resolved == expected,
    }


def _verify_collator_masking(trainer: Any, samples: list[VerificationSample]) -> dict[str, Any]:
    """Verify the actual trainer collator masks prompts and trains on complete targets."""
    sample_checks = []
    for sample in samples:
        prepared = trainer.train_dataset[sample.dataset_index]
        batch = trainer.data_collator([prepared])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        if "attention_mask" in batch:
            active = batch["attention_mask"][0].bool()
        elif "position_ids" in batch:
            active = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            raise KeyError("The trainer collator returned neither attention_mask nor position_ids")
        actual_ids = input_ids[active].tolist()
        actual_labels = labels[active].tolist()
        prompt_labels = actual_labels[: sample.fed_prompt_token_count]
        target_labels = actual_labels[sample.fed_prompt_token_count : sample.fed_sequence_token_count]
        prompt_masked = all(label == -100 for label in prompt_labels)
        target_receives_loss = bool(target_labels) and all(label != -100 for label in target_labels)
        sample_checks.append(
            {
                "dataset_index": sample.dataset_index,
                "issue_id": sample.issue_id,
                "repository": sample.repository,
                "truncated": sample.truncated,
                "actual_sequence_token_count": len(actual_ids),
                "prompt_tokens_do_not_receive_loss": prompt_masked,
                "completion_target_receives_loss": target_receives_loss,
                "target_preserved": sample.target_preserved,
            }
        )
    return {
        "method": "TRL SFTTrainer pre-tokenized completion_mask collator",
        "sample_checks": sample_checks,
        "all_prompt_tokens_masked": all(item["prompt_tokens_do_not_receive_loss"] for item in sample_checks),
        "all_completion_targets_receive_loss": all(item["completion_target_receives_loss"] for item in sample_checks),
        "all_targets_preserved": all(item["target_preserved"] for item in sample_checks),
    }


def _preflight_checks(
    config: dict[str, Any],
    runtime: dict[str, Any],
    boundary: dict[str, Any],
    train_build: SplitDatasetBuild,
    model_info: dict[str, Any],
    trainer: Any,
) -> dict[str, Any]:
    """Run every launch gate immediately before any training step."""
    optimization = config["optimization"]
    train_rows = train_build.stats["row_count"]
    expected_steps = expected_optimizer_steps(
        train_rows,
        optimization["per_device_train_batch_size"],
        optimization["gradient_accumulation_steps"],
    )
    masking = _verify_collator_masking(trainer, train_build.verification_samples)
    parameter_report = model_info["parameter_report"]
    resolved = _resolved_hyperparameters(trainer, config)
    checks = {
        "correct_model_id": config["base_model"]["model_id"] == EXPECTED_MODEL_ID,
        "correct_model_revision": config["base_model"]["revision"] == EXPECTED_MODEL_REVISION,
        "correct_train_sha256": boundary["train_sha256"] == EXPECTED_TRAIN_SHA256,
        "exactly_expected_train_rows": train_rows == EXPECTED_TRAIN_ROWS,
        "optimization_dataset_train_only": boundary["optimization_dataset_is_train_only"]
        and boundary["test_accessed"] is False,
        "only_lora_parameters_trainable": parameter_report["only_lora_parameters_trainable"]
        and parameter_report["base_parameters_frozen"],
        "completion_target_receives_loss": masking["all_completion_targets_receive_loss"],
        "prompt_tokens_do_not_receive_loss": masking["all_prompt_tokens_masked"],
        "target_cannot_be_removed_by_truncation": train_build.stats["target_removed_count"] == 0
        and train_build.stats["target_preserved_for_every_row"]
        and masking["all_targets_preserved"],
        "fp16_bf16_settings_match_environment": optimization["fp16"] is True
        and optimization["bf16"] is False
        and runtime["unsloth"]["bf16_supported_after_unsloth"] is False,
        "resolved_hyperparameters_match_configuration": resolved["matches_approved_configuration"],
        "expected_optimizer_steps": expected_steps == optimization["expected_optimizer_steps"] == 1993,
        "trainer_dataloader_row_count": len(trainer.get_train_dataloader()) == EXPECTED_TRAIN_ROWS,
    }
    result = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "expected_optimizer_steps": expected_steps,
        "parameter_report": parameter_report,
        "masking": masking,
        "resolved_hyperparameters": resolved,
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(f"Final launch verification failed: {result}")
    return result


def _monitor_callback() -> Any:
    """Create a callback that records loss, learning rate, gradient norm, and non-finite logs."""
    from transformers import TrainerCallback

    class TrainingMonitor(TrainerCallback):
        """Collect compact training telemetry from Trainer logging events."""

        def __init__(self) -> None:
            self.log_history: list[dict[str, Any]] = []
            self.nan_inf_events: list[dict[str, Any]] = []

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            record = {"step": int(state.global_step), "epoch": state.epoch}
            for key in (
                "loss",
                "learning_rate",
                "grad_norm",
                "num_input_tokens_seen",
                "eval_loss",
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            ):
                if key in logs:
                    value = logs[key]
                    record[key] = float(value) if isinstance(value, (int, float)) else value
                    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                        self.nan_inf_events.append({"step": state.global_step, "field": key, "value": repr(value)})
            self.log_history.append(record)

    return TrainingMonitor()


def _deduplicate_warnings(captured: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    """Keep unique warning category/message pairs for the report."""
    seen = set()
    result = []
    for warning in captured:
        item = {"category": warning.category.__name__, "message": str(warning.message)}
        key = (item["category"], item["message"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _adapter_metadata(config: dict[str, Any], git: dict[str, Any], train_build: SplitDatasetBuild) -> dict[str, Any]:
    """Return reload metadata written beside the ignored adapter."""
    return {
        "base_model_id": config["base_model"]["model_id"],
        "base_model_revision": config["base_model"]["revision"],
        "quantization": config["base_model"]["quantization"],
        "lora": config["lora"],
        "max_sequence_length": config["sequence"]["max_length"],
        "starting_git_commit_sha": git["starting_commit_sha"],
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "train_row_count": train_build.stats["row_count"],
        "experiment_name": config["experiment_name"],
    }


def _save_adapter(
    trainer: Any,
    tokenizer: Any,
    config: dict[str, Any],
    git: dict[str, Any],
    train_build: SplitDatasetBuild,
) -> dict[str, Any]:
    """Save only the PEFT adapter, tokenizer, and local reload metadata."""
    output_dir = PROJECT_ROOT / config["trainer"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata_path = output_dir / "adapter_metadata.json"
    metadata_path.write_text(
        json.dumps(_adapter_metadata(config, git, train_build), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    adapter_weights = [name for name in files if name.startswith("adapter_model.")]
    forbidden_full_model_weights = [
        name for name in files if name in {"pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"}
    ]
    if not adapter_weights or forbidden_full_model_weights:
        raise RuntimeError(
            f"Adapter output integrity failed: adapter_weights={adapter_weights}, "
            f"forbidden_full_model_weights={forbidden_full_model_weights}"
        )
    return {
        "path": _relative_path(output_dir),
        "saved": True,
        "adapter_only": True,
        "adapter_weight_files": adapter_weights,
        "metadata_file": _relative_path(metadata_path),
        "files": files,
        "git_ignored_output_policy": True,
    }


def run_full_training(config_path: Path = DEFAULT_CONFIG_PATH, *, preflight_only: bool = False) -> dict[str, Any]:
    """Run the final preflight and, unless requested otherwise, exactly one epoch."""
    config = _read_config(config_path)
    report: dict[str, Any] = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_training_started": False,
        "validation_accessed": False,
        "test_accessed": False,
        "preflight_only": preflight_only,
        "config_path": _relative_path(config_path),
        "configuration": config,
        "errors": [],
    }
    model = None
    tokenizer = None
    trainer = None
    captured_warnings: list[warnings.WarningMessage] = []
    monitor = None
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            captured_warnings = caught_warnings
            report["git"] = _git_snapshot()
            report["software_and_cuda_integrity"] = _runtime_integrity()
            report["resource_preflight"] = _resource_preflight()
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config["base_model"]["model_id"],
                revision=config["base_model"]["revision"],
                trust_remote_code=config["base_model"]["trust_remote_code"],
                local_files_only=True,
            )
            boundary, train_build, validation_build = _data_boundary(config, tokenizer)
            report["data_boundary"] = boundary
            report["validation_accessed"] = boundary["validation_accessed"]
            report["data_statistics"] = {
                "train": train_build.stats,
                "validation": validation_build.stats,
                "train_verification_samples": [sample.__dict__ for sample in train_build.verification_samples],
            }
            model, tokenizer, model_info = _load_model_and_tokenizer(config)
            report["model"] = model_info
            if model_info["parameter_devices"] != ["cuda:0"]:
                raise RuntimeError(f"Model parameters are not fully placed on cuda:0: {model_info}")
            trainer = _build_trainer(model, tokenizer, train_build.dataset, validation_build.dataset, config)
            report["launch_verification"] = _preflight_checks(
                config,
                report["software_and_cuda_integrity"],
                boundary,
                train_build,
                model_info,
                trainer,
            )
            if preflight_only:
                report["status"] = "preflight_passed"
            else:
                monitor = _monitor_callback()
                trainer.add_callback(monitor)
                torch.cuda.reset_peak_memory_stats()
                training_started_at = datetime.now(timezone.utc)
                training_timer_start = time.perf_counter()
                report["full_training_started"] = True
                train_result = trainer.train()
                torch.cuda.synchronize()
                training_finished_at = datetime.now(timezone.utc)
                runtime_seconds = time.perf_counter() - training_timer_start
                optimizer_steps = int(trainer.state.global_step)
                expected_steps = config["optimization"]["expected_optimizer_steps"]
                if optimizer_steps != expected_steps:
                    raise RuntimeError(f"Expected {expected_steps} optimizer steps, completed {optimizer_steps}")
                if monitor.nan_inf_events:
                    raise RuntimeError(f"NaN/Inf detected in training logs: {monitor.nan_inf_events}")
                report["training"] = {
                    "start_timestamp_utc": training_started_at.isoformat(),
                    "end_timestamp_utc": training_finished_at.isoformat(),
                    "runtime_seconds": round(runtime_seconds, 4),
                    "optimizer_steps_completed": optimizer_steps,
                    "expected_optimizer_steps": expected_steps,
                    "train_rows_processed": train_build.stats["row_count"],
                    "all_train_rows_processed_as_intended": train_build.stats["row_count"] == EXPECTED_TRAIN_ROWS,
                    "train_result_metrics": train_result.metrics,
                    "log_history": monitor.log_history,
                    "nan_inf_occurrence": bool(monitor.nan_inf_events),
                    "nan_inf_events": monitor.nan_inf_events,
                    "peak_gpu_memory": _cuda_memory_snapshot(),
                    "truncated_training_examples": {
                        "count": train_build.stats["truncated_row_count"],
                        "percentage": train_build.stats["truncated_row_percentage"],
                    },
                    "warnings": _deduplicate_warnings(captured_warnings),
                }
                evaluation_records = [record for record in monitor.log_history if "eval_loss" in record]
                if evaluation_records:
                    report["validation"] = evaluation_records[-1]
                report["adapter_artifact"] = _save_adapter(
                    trainer,
                    tokenizer,
                    config,
                    report["git"],
                    train_build,
                )
                report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "cuda_out_of_memory": isinstance(error, torch.OutOfMemoryError)
                or "out of memory" in str(error).lower(),
            }
        )
    finally:
        report["warnings"] = _deduplicate_warnings(captured_warnings)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["test_accessed"] = False
        if not preflight_only:
            DEFAULT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    """Run the final launch gate or the approved one-epoch experiment."""
    preflight_only = "--preflight-only" in sys.argv[1:]
    report = run_full_training(preflight_only=preflight_only)
    output = {
        "status": report["status"],
        "preflight_only": preflight_only,
        "full_training_started": report["full_training_started"],
        "report_path": None if preflight_only else _relative_path(DEFAULT_REPORT_PATH),
        "optimizer_steps": report.get("training", {}).get("optimizer_steps_completed"),
        "adapter_path": report.get("adapter_artifact", {}).get("path"),
        "errors": report.get("errors", []),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
