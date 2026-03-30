# Runtime Environment

Use this file as the authoritative environment contract for this run.
Do not re-guess Python paths unless a command from this file fails.

## Python
- Command: `/usr/bin/python3.12`
- Pip: `/usr/bin/python3.12 -m pip`
- Version: `3.12.12`

## Torch / ROCm
- Torch import OK: `True`
- Torch version: `2.9.1+git8907517`
- Torch path: `/usr/local/lib/python3.12/dist-packages/torch/__init__.py`
- HIP version: `7.0.51831-a3e329ad8`

## GPU
- CUDA/HIP available: `True`
- Device count: `1`
- Device name: ``

## Libraries
- Triton: `True`
- aiter: `True`
- composable_kernel: `False`

## Build Guidance
- Use the same Python that imports torch for any editable install or rebuild.
- Rebuild command pattern: `/usr/bin/python3.12 -m pip install -e . --no-build-isolation`
