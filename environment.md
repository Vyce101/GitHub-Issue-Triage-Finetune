# Local environment record

This record covers environment setup and hardware verification only. No Hugging Face model, dataset, transformer stack, or training pipeline was installed or created.

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
| filelock | 3.32.3 |
| fsspec | 2026.7.0 |
| Jinja2 | 3.1.6 |
| MarkupSafe | 3.0.3 |
| mpmath | 1.3.0 |
| networkx | 3.6.1 |
| numpy | 2.4.3 |
| pip | 26.2.1 |
| setuptools | 65.5.0 |
| sympy | 1.14.0 |
| torch | 2.12.1+cu130 |
| typing_extensions | 4.16.0 |

PyTorch was installed from the stable CUDA 13.0 wheel index. Its bundled CUDA runtime reports `13.0`.

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
- CUDA tensor allocation: PASS
- CUDA tensor matrix operation and synchronization: PASS
- Operation result: `[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]`
- Device memory reported by PyTorch: `8589606912` bytes
- `pip check`: `No broken requirements found.`

## Errors and warnings encountered

- The first virtual-environment attempt failed because the registered 3.11.13 launcher path did not exist. No `.venv` was created by that failed attempt; the local runtime provisioning above resolved it.
- An initial `nvidia-smi --query-gpu` diagnostic used unsupported field `cuda_version`; the corrected standard `nvidia-smi` output reported CUDA UMD 13.3.
- Before NumPy was installed, PyTorch emitted a non-fatal `No module named 'numpy'` warning during import. NumPy 2.4.3 was then installed, and the final import and CUDA verification completed without warnings or errors.
