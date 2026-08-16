# Local environment record

This record covers environment setup, the completed locked-model smoke test, and Unsloth Core fine-tuning stack verification. The locked Qwen checkpoint was downloaded to the local Hugging Face cache by the completed model smoke test; no dataset was downloaded, Unsloth Studio was not installed, and no training pipeline was created. The checkpoint is local cache state and is not stored or committed in Git.

## Host detection

- OS: Windows
- GPU: NVIDIA GeForce RTX 2060 SUPER
- VRAM reported by `nvidia-smi`: 8192 MiB
- NVIDIA driver: 610.62
- CUDA UMD reported by `nvidia-smi`: 13.3

## Python isolation

- Project virtual environment: `.venv`
- Virtual-environment Python: CPython 3.11.13
- Virtual-environment pip: 26.2.1
- `uv` used to provision the project-local CPython runtime: 0.11.6
- Project-local runtime source: `.python-runtime/cpython-3.11.13-windows-x86_64-none`
- The registered `py -V:Astral/CPython3.11.13` entry was stale and pointed to a missing executable. The requested 3.11.13 runtime was therefore provisioned locally with `uv`; no Windows registry entry was added.
- Global Python 3.14.3 was not modified.
- StabilityMatrix environments were not modified.

## Installed packages

| Package | Version |
| --- | --- |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.1 |
| cut-cross-entropy | 25.1.1 |
| datasets | 4.3.0 |
| diffusers | 0.39.0 |
| dill | 0.4.0 |
| filelock | 3.32.3 |
| fsspec | 2025.9.0 |
| huggingface-hub | 1.27.0 |
| Jinja2 | 3.1.6 |
| MarkupSafe | 3.0.3 |
| multiprocess | 0.70.16 |
| mpmath | 1.3.0 |
| msgspec | 0.21.1 |
| networkx | 3.6.1 |
| numpy | 2.4.3 |
| pandas | 3.0.5 |
| peft | 0.20.0 |
| pyarrow | 25.0.1 |
| pillow | 12.3.0 |
| pip | 26.2.1 |
| protobuf | 7.35.1 |
| pydantic | 2.13.4 |
| safetensors | 0.8.0 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |
| sentencepiece | 0.2.2 |
| setuptools | 65.5.0 |
| torch | 2.11.0+cu130 |
| torchao | 0.18.0 |
| torchvision | 0.26.0+cu130 |
| tokenizers | 0.22.2 |
| sympy | 1.14.0 |
| transformers | 5.5.0 |
| triton-windows | 3.7.1.post27 |
| trl | 0.24.0 |
| typing_extensions | 4.16.0 |
| unsloth | 2026.8.18 |
| unsloth-zoo | 2026.8.12 |
| xformers | 0.0.35 |

PyTorch was installed from the CUDA 13.0 wheel index. Its bundled CUDA runtime reports `13.0`.

The requested Unsloth Core supervised fine-tuning stack is installed in `.venv`. Unsloth required the approved change from PyTorch `2.12.1+cu130` to `2.11.0+cu130`; CUDA support remained active. The locked Qwen checkpoint was downloaded to the local Hugging Face cache during the completed model smoke test; no dataset download was performed.

## Locked model cache

- Model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Resolved revision: `7744afa8566e264af1a92a806d8d9aae00cc7c78`
- Local Hugging Face cache entry: found during the smoke test under `models--unsloth--Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Cache size recorded by the smoke test: approximately 3.56 GB
- The cached checkpoint is ignored local state; model weights are not stored or committed in Git.

## Recreate the environment

From the project root in PowerShell:

```powershell
uv python install 3.11.13 --install-dir .python-runtime --no-bin --no-registry
$runtime = (Resolve-Path -LiteralPath '.python-runtime\cpython-3.11.13-windows-x86_64-none\python.exe').Path
& $runtime -m venv --copies .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verification results

Executed inside `.venv` on 2026-08-15:

- `torch` import: PASS
- `torch.cuda.is_available()`: `True`
- CUDA device count: `1`
- Detected GPU: `NVIDIA GeForce RTX 2060 SUPER`
- Compute capability: `7.5`
- PyTorch CUDA runtime: `13.0`
- PyTorch version before stack installation: `2.12.1+cu130`
- PyTorch version after stack installation: `2.11.0+cu130`
- CUDA tensor allocation: PASS
- CUDA tensor matrix operation and synchronization: PASS; result `[[5.0, 14.0, 23.0], [14.0, 50.0, 86.0], [23.0, 86.0, 149.0]]`
- `unsloth`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`, and `bitsandbytes` imports: PASS
- Unsloth initialization: PASS; the isolated import check initialized Unsloth and Unsloth Zoo before model loading.
- Unsloth version: `2026.8.18`
- Unsloth device detection: PASS; `DEVICE_TYPE=cuda`, `DEVICE_COUNT=1`, `ALLOW_BITSANDBYTES=True`, GPU `NVIDIA GeForce RTX 2060 SUPER`, CUDA `7.5`, toolkit `13.0`, approximately `8.0 GB`.
- `bitsandbytes` version: `0.50.1`
- bitsandbytes CUDA backend: PASS; its native library was present, CUDA symbols were available, and CUDA specs detected CUDA `13.0` with compute capability `(7, 5)`.
- bitsandbytes NF4 `Linear4bit` CUDA smoke test: PASS; input shape `(4, 64)`, output shape `(4, 8)`.
- bitsandbytes functional NF4 quantize/dequantize CUDA smoke test: PASS; packed shape `(1024, 1)`, restored shape `(32, 64)`.
- `torch.cuda.is_bf16_supported()`: `False`.
- `pip check`: `No broken requirements found.`

## Errors and warnings encountered

- The first virtual-environment attempt failed because the registered 3.11.13 launcher path did not exist. No `.venv` was created by that failed attempt; the local runtime provisioning above resolved it.
- An initial `nvidia-smi --query-gpu` diagnostic used unsupported field `cuda_version`; the corrected standard `nvidia-smi` output reported CUDA UMD 13.3.
- Before NumPy was installed, PyTorch emitted a non-fatal `No module named 'numpy'` warning during import. NumPy 2.4.3 was then installed, and the final import and CUDA verification completed without warnings or errors.
- Unsloth's dependency resolver required PyTorch `<2.12.0`, so the approved downgrade to `2.11.0+cu130` was applied. CUDA remained available after the change.
- PyTorch emitted a non-fatal Windows/macOS note that distributed multiprocessing redirects are not supported on those platforms.
- Importing Unsloth generated the local untracked `unsloth_compiled_cache/` runtime cache containing generated trainer source and bytecode. The completed model smoke test separately populated the ignored Hugging Face model cache; no dataset files were created.
