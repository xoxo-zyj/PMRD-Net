"""Preflight imports, CUDA and optional checkpoint integrity."""
import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--evaluation", action="store_true", help="Also check pyiqa.")
    args = parser.parse_args()
    failed = []
    print("Python:", sys.version.replace("\n", " "))
    names = ["torch", "torchvision", "numpy", "PIL", "tqdm", "einops", "timm",
             "fvcore", "pytorch_msssim", "skimage", "efficientvit.models.nn",
             "efficientvit.models.utils", "future_process.efficientvit.backbone"]
    if args.require_cuda:
        names += ["selective_scan_cuda_oflex", "future_process.vmamba"]
    if args.evaluation:
        names += ["pyiqa"]
    for name in names:
        try:
            module = importlib.import_module(name)
            if name == "future_process.vmamba" and not hasattr(module, "VSSBlock"):
                raise ImportError("VSSBlock unavailable")
            print("OK:", name, getattr(module, "__version__", ""))
        except Exception as error:
            failed.append(name)
            print(f"FAIL: {name}: {type(error).__name__}: {error}")
    try:
        torch = importlib.import_module("torch")
        print("Torch CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
        if args.require_cuda and not torch.cuda.is_available():
            failed.append("CUDA unavailable")
    except ImportError:
        pass
    try:
        triton = importlib.import_module("triton")
        print("Optional Triton:", getattr(triton, "__version__", "available"))
    except ImportError:
        print("Optional Triton unavailable; archived VMamba provides PyTorch cross-scan fallback.")
    if args.require_weights:
        from tools.verify_weights import verify
        if not verify(PROJECT_ROOT):
            failed.append("checkpoints")
    print("Preflight:", "FAILED" if failed else "PASSED")
    print("A successful import check still requires smoke_test.py forward/backward validation.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
