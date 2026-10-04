# Installation

Run commands from the repository root. The archive did not include a complete
environment lockfile. Its VMamba header refers to Python 3.10, PyTorch 2.2 and
`timm==0.4.12`; these are historical clues, not a validated environment export.
The provided requirements describe dependencies found in the source.

## 1. PyTorch and Python packages

Create a clean Python 3.10 environment for the archived CUDA implementation.
Install a matching PyTorch/torchvision pair for your NVIDIA driver and CUDA
environment using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
Avoid installing an arbitrary recent Torch version over an existing compiled kernel.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-eval.txt
```

The evaluation requirements pin `pyiqa==0.1.12` as a packaging baseline for NIQE
and BRISQUE. Its [upstream requirements](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.12/requirements.txt)
allow the archived timm version; newer releases require a newer timm and conflict
with `timm==0.4.12`. The archive did not identify its original pyiqa version, and
this baseline still needs a runtime check. Record the installed version and
metric configuration with reported results.
`requirements-legacy.txt` is only for additional dependencies used by old entrypoints.

## 2. EfficientViT utilities

The bundled `future_process/efficientvit/` preserves the author's backbone, but
imports `efficientvit.models.nn` and `efficientvit.models.utils` from upstream.
Install the upstream package without its unrelated optional task dependencies:

```bash
python -m pip install --no-deps "git+https://github.com/mit-han-lab/efficientvit.git@bd2f02695c7c6da942a1a3177bdc89a417286fcf"
```

This pinned [upstream revision](https://github.com/mit-han-lab/efficientvit/commit/bd2f02695c7c6da942a1a3177bdc89a417286fcf)
is a documented packaging baseline containing the imported interfaces. The archive
did not identify the original EfficientViT revision, and this baseline has not
been validated with a GPU run on the preparation host.

## 3. VMamba selective-scan extension

Formal training/evaluation uses the bundled `VSSBlock` and requires the binary
module **`selective_scan_cuda_oflex`**. Installing only `mamba_ssm` does not supply
that named extension. The [VMamba selective-scan build script](https://github.com/MzeroMiko/VMamba/blob/main/kernels/selective_scan/setup.py)
builds it; the archived code calls its `fwd` and `bwd` functions.

On a Linux CUDA development environment with `nvcc` and a matching PyTorch build:

```bash
git clone https://github.com/MzeroMiko/VMamba.git vendor/VMamba
python -m pip install --no-build-isolation ./vendor/VMamba/kernels/selective_scan
git -C vendor/VMamba rev-parse HEAD
```

Record the printed revision and compiler/CUDA versions. Rebuild the extension
after changing Torch or CUDA. Triton accelerates cross-scan operations where
available; the archived file also contains a PyTorch cross-scan fallback.
The required selective-scan CUDA kernel remains necessary for the full model.
The preparation host is Windows without Torch/CUDA, so Windows GPU compatibility
and a complete CUDA setup have not been verified here.

## 4. Weights and runtime check

Merge `MSDSPDD_Model_Weights.zip` into this repository as described in
[checkpoints/README.md](../checkpoints/README.md), then run:

```bash
python tools/verify_weights.py
python tools/check_environment.py --require-cuda --require-weights --evaluation
python smoke_test.py
```

Inspect the smoke-test output: formal runs require `backbone fallback: False`
and `Mamba enabled: True`. The test must complete forward and backward passes.
CPU execution allows a fallback in the original smoke test and only checks the
fallback network's shape/loss wiring. It does not reproduce the released full model.

## Common failures

| Failure | Action |
| --- | --- |
| `No module named efficientvit` | Install the pinned upstream utilities above |
| `No module named selective_scan_cuda_oflex` | Build the VMamba kernel for the current Torch/CUDA environment |
| CUDA extension undefined symbol / ABI error | Rebuild against the active Torch environment |
| Missing `.pt` / `.pth` | Merge the weight ZIP and run checksum verification |
| Missing dataset image paths | Run `tools/prepare_splits.py --check-files` with your data root and prefix map |
