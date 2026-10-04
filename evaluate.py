# -*- coding: utf-8 -*-
"""MSDSPDD Complete V2 六数据集×八协议专项权重正式测试。

- I/O/NH-HAZE：只测微调时保留的 test_reserved 清单。
- SOTS-Indoor/Outdoor：测官方 SOTS 测试集。
- HazeRD：没有专项权重，默认加载通用 general_best.pth。
- 七种滑窗协议保持原始分辨率；resize256协议在256×256上统一评测。
- 默认只保存指标CSV；显式添加--save_images才保存各协议复原图。
- 指标：PSNR、SSIM、NIQE、BRISQUE。
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from future_process.model import MSDSPDDDehazer  # noqa: E402
from skimage.metrics import structural_similarity as skimage_ssim
try:
    import pyiqa
except Exception:
    pyiqa = None

DATASETS = ("I-HAZE", "O-HAZE", "NH-HAZE", "HazeRD", "SOTS-indoor", "SOTS-outdoor")
TASK = {
    "I-HAZE": "ihaze", "O-HAZE": "ohaze", "NH-HAZE": "nhhaze",
    "SOTS-indoor": "sots_indoor", "SOTS-outdoor": "sots_outdoor",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_FINETUNE = str(PROJECT_ROOT / "checkpoints" / "finetuned")
DEFAULT_GENERAL = str(PROJECT_ROOT / "checkpoints" / "general" / "general_best.pth")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "evaluation")


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def model_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict", "params", "params_ema"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                ckpt = value
                break
    if not isinstance(ckpt, dict) or not ckpt:
        raise TypeError("无法从 checkpoint 提取模型参数")
    if not all(torch.is_tensor(v) for v in ckpt.values()):
        raise TypeError("checkpoint 不是有效 state_dict")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def is_img(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_imgs(root: Path):
    return sorted(p for p in root.rglob("*") if is_img(p)) if root.exists() else []


def pick_dir(root: Path, names: Sequence[str]):
    for name in names:
        d = root / name
        if d.is_dir() and list_imgs(d):
            return d
    return None


def read_reserved_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"test_reserved 清单不存在：{path}")
    pairs = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            h = Path(row.get("hazy_path", "").strip())
            g = Path(row.get("gt_path", "").strip())
            if not h.is_file() or not g.is_file():
                raise FileNotFoundError(f"清单文件不存在：hazy={h} | gt={g}")
            pairs.append((h, g))
    if not pairs:
        raise RuntimeError(f"test_reserved 清单为空：{path}")
    return pairs


def name_keys(stem: str):
    s = stem.lower()
    keys = [s]
    for suffix in ("_hazy", "_haze", "_foggy", "_fog", "_input", "_gt", "_clear", "_clean", "_rgb"):
        if s.endswith(suffix):
            keys.append(s[:-len(suffix)])
    if "_" in s:
        parts = s.split("_")
        keys += [parts[0], "_".join(parts[:2])]
    out = []
    for k in keys:
        k = k.strip("_")
        if k and k not in out:
            out.append(k)
    return out


def gt_index(files):
    idx = defaultdict(list)
    for p in files:
        for key in name_keys(p.stem):
            idx[key].append(p)
    return idx


def match_gt(hazy: Path, idx, dataset: str):
    stem = hazy.stem.lower()
    if dataset == "HazeRD":
        parts = stem.split("_")
        if len(parts) >= 3 and idx.get("_".join(parts[:2])):
            return sorted(idx["_".join(parts[:2])], key=lambda p: len(str(p)))[0]
    if dataset.startswith("SOTS-"):
        key = stem.split("_")[0]
        if idx.get(key):
            return sorted(idx[key], key=lambda p: len(str(p)))[0]
    for key in name_keys(stem):
        if idx.get(key):
            return sorted(idx[key], key=lambda p: len(str(p)))[0]
    return None


def directory_pairs(dataset: str, root: Path):
    if not root.exists():
        raise FileNotFoundError(f"{dataset} 根目录不存在：{root}")
    if dataset == "HazeRD":
        h_names = ("haze", "hazy", "Haze", "Hazy", "HAZE", "HAZY")
        g_names = ("GT", "gt", "clear", "Clear", "clean", "Clean")
    else:
        h_names = ("hazy", "haze", "Hazy", "Haze")
        g_names = ("clear", "GT", "gt", "clean", "Clean")
    hdir, gdir = pick_dir(root, h_names), pick_dir(root, g_names)
    if hdir is None or gdir is None:
        raise FileNotFoundError(f"{dataset} 找不到 hazy/GT 目录：{root}")
    hazy_files, gt_files = list_imgs(hdir), list_imgs(gdir)
    idx = gt_index(gt_files)
    pairs, unmatched = [], []
    for h in hazy_files:
        g = match_gt(h, idx, dataset)
        (pairs if g else unmatched).append((h, g) if g else h)
    if not pairs:
        raise RuntimeError(f"{dataset} 没有有效配对")
    msg = f"root={root} | hazy={len(hazy_files)} | gt={len(gt_files)} | pairs={len(pairs)} | unmatched={len(unmatched)}"
    return pairs, msg


def dataset_pairs(dataset: str, args):
    if dataset in ("I-HAZE", "O-HAZE", "NH-HAZE"):
        manifest = Path(args.split_root) / TASK[dataset] / "split_manifest_test_reserved.csv"
        pairs = read_reserved_manifest(manifest)
        return pairs, f"reserved_manifest={manifest} | pairs={len(pairs)}"
    if dataset == "HazeRD":
        return directory_pairs(dataset, Path(args.hazerd_root))
    sub = "indoor" if dataset == "SOTS-indoor" else "outdoor"
    candidates = [Path(args.sots_root) / sub, Path(args.sots_root) / sub.capitalize(), Path(args.sots_root) / "SOTS" / sub]
    root = next((p for p in candidates if p.exists()), candidates[0])
    return directory_pairs(dataset, root)


def create_model(device, project_root: Path):
    weight = project_root / "future_process" / "efficientvit_b1_r256.pt"
    if not weight.is_file():
        raise FileNotFoundError(f"EfficientViT 权重不存在：{weight}")
    model = MSDSPDDDehazer(
        enable_end_to_end=True, weight_path=str(weight), route_temperature=1.0,
        use_mamba=True, enable_refiner=True, allow_backbone_fallback=False,
    ).to(device).eval()
    if getattr(model, "backbone_is_fallback", False):
        raise RuntimeError("检测到备用卷积主干，禁止正式测试")
    if not getattr(model, "mamba_enabled", True):
        raise RuntimeError("VMamba/VSSBlock 未启用，禁止正式测试")
    return model


def checkpoint_path(dataset: str, args):
    if dataset == "HazeRD":
        return Path(args.general_ckpt)
    return Path(args.finetune_root) / TASK[dataset] / "dehaze_best_model_only.pth"


def load_checkpoint(model, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"测试权重不存在：{path}")
    result = model.load_state_dict(model_state(torch_load(path)), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"权重不匹配：missing={result.missing_keys[:10]} unexpected={result.unexpected_keys[:10]}")
    model.eval()
    print(f"✅ 严格加载：{path}")


def parse_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("final_out", "J_final", "out", "output", "pred"):
            if torch.is_tensor(output.get(key)):
                return output[key]
    raise RuntimeError(f"无法解析模型输出：{type(output)}")


def pad_to_multiple(x, multiple=32, min_size=None):
    _, _, h, w = x.shape
    th = h if min_size is None else max(h, int(min_size))
    tw = w if min_size is None else max(w, int(min_size))
    th = ((th + multiple - 1) // multiple) * multiple
    tw = ((tw + multiple - 1) // multiple) * multiple
    ph, pw = th - h, tw - w
    if ph == 0 and pw == 0:
        return x.contiguous(), h, w
    mode = "reflect" if h > ph and w > pw else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode).contiguous(), h, w


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.inference_mode()
def forward_patch(model, x, amp=False):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and x.device.type == "cuda"):
        pred = parse_output(model(x.contiguous().float()))
    if pred.shape[-2:] != x.shape[-2:]:
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return pred.float().clamp(0, 1)


@torch.inference_mode()
def infer_full(model, x, tile, stride, amp=False):
    if tile <= 0:
        padded, oh, ow = pad_to_multiple(x, 32)
        return forward_patch(model, padded, amp)[..., :oh, :ow].contiguous()
    if stride <= 0 or stride > tile:
        raise ValueError("stride 必须满足 0 < stride <= tile")
    padded, oh, ow = pad_to_multiple(x, 32, min_size=tile)
    _, _, h, w = padded.shape
    out = torch.zeros((1, 3, h, w), device=x.device, dtype=torch.float32)
    weight = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    for yy in starts(h, tile, stride):
        for xx in starts(w, tile, stride):
            patch = padded[..., yy:yy + tile, xx:xx + tile].contiguous()
            pred = forward_patch(model, patch, amp)
            out[..., yy:yy + tile, xx:xx + tile] += pred
            weight[..., yy:yy + tile, xx:xx + tile] += 1.0
            del patch, pred
    return (out / weight.clamp_min(1.0))[..., :oh, :ow].contiguous().clamp(0, 1)


def crop_metric(x, shave):
    if shave <= 0:
        return x
    h, w = x.shape[-2:]
    return x if h <= 2 * shave or w <= 2 * shave else x[..., shave:-shave, shave:-shave]


@torch.inference_mode()
def calc_psnr(pred, gt, shave=0):
    pred, gt = crop_metric(pred.float().clamp(0, 1), shave), crop_metric(gt.float().clamp(0, 1), shave)
    mse = F.mse_loss(pred, gt).item()
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


@torch.inference_mode()
def calc_ssim(pred, gt, shave=0):
    pred, gt = crop_metric(pred.float().clamp(0, 1), shave), crop_metric(gt.float().clamp(0, 1), shave)
    p = pred.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    g = gt.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return float(skimage_ssim(p, g, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5))


def build_iqa(device):
    """只加载论文统一使用的 NIQE 和 BRISQUE。"""
    if pyiqa is None:
        raise RuntimeError(
            "缺少 pyiqa，无法计算 NIQE 和 BRISQUE。"
        )

    metrics = {}
    errors = {}

    for name in ("niqe", "brisque"):
        try:
            metrics[name] = pyiqa.create_metric(
                name,
                device=device,
                as_loss=False,
            ).eval()
            print(f"✅ 指标加载：{name.upper()}")
        except Exception as error:
            errors[name] = repr(error)

    if errors:
        raise RuntimeError(
            "IQA 指标加载失败：\n"
            + "\n".join(
                f"{name}: {message}"
                for name, message in errors.items()
            )
        )

    return metrics


@torch.inference_mode()
def calc_iqa(pred, metrics):
    """计算无参考 NIQE / BRISQUE；两项均为越低越好。"""
    output = {
        "NIQE": float("nan"),
        "BRISQUE": float("nan"),
    }

    for name, label in (
        ("niqe", "NIQE"),
        ("brisque", "BRISQUE"),
    ):
        metric = metrics[name]
        try:
            value = metric(pred)
            output[label] = float(
                value.detach().mean().cpu().item()
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                clear_cuda()
                raise RuntimeError(
                    f"{label} 在完整原图上计算时显存不足；"
                    "测试脚本没有对输出图进行 resize。"
                ) from error
            raise

    return output



# ============================================================
# 全部统一测试协议
# ============================================================
PROTOCOLS = {
    "tile1024_stride512": {
        "mode": "tile",
        "tile": 1024,
        "stride": 512,
    },
    "tile768_stride384": {
        "mode": "tile",
        "tile": 768,
        "stride": 384,
    },
    "tile640_stride320": {
        "mode": "tile",
        "tile": 640,
        "stride": 320,
    },
    "tile512_stride256": {
        "mode": "tile",
        "tile": 512,
        "stride": 256,
    },
    "tile384_stride192": {
        "mode": "tile",
        "tile": 384,
        "stride": 192,
    },
    "tile320_stride160": {
        "mode": "tile",
        "tile": 320,
        "stride": 160,
    },
    "tile256_stride128": {
        "mode": "tile",
        "tile": 256,
        "stride": 128,
    },
    "resize256": {
        "mode": "resize",
        "tile": None,
        "stride": None,
    },
}


@torch.inference_mode()
def infer_protocol(
    model: nn.Module,
    hazy: torch.Tensor,
    protocol_name: str,
    resize_size: int,
    amp: bool,
) -> torch.Tensor:
    config = PROTOCOLS[protocol_name]

    if config["mode"] == "tile":
        return infer_full(
            model,
            hazy,
            int(config["tile"]),
            int(config["stride"]),
            amp,
        )

    if config["mode"] == "resize":
        small = F.interpolate(
            hazy,
            size=(resize_size, resize_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)

        # 256可以被模型的多尺度结构整除，不再额外改变尺寸。
        prediction = forward_patch(
            model,
            small,
            amp,
        )
        del small
        return prediction.clamp(0.0, 1.0)

    raise ValueError(
        f"未知协议模式：{config['mode']}"
    )


@torch.inference_mode()
def calc_iqa_protocol(
    prediction: torch.Tensor,
    metrics,
    iqa_resize: int,
):
    """NIQE和BRISQUE统一在相同IQA尺寸上计算。"""
    output = {
        "NIQE": float("nan"),
        "BRISQUE": float("nan"),
    }

    iqa_input = prediction.clamp(0.0, 1.0)
    if iqa_resize > 0:
        iqa_input = F.interpolate(
            iqa_input,
            size=(iqa_resize, iqa_resize),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)

    for name, label in (
        ("niqe", "NIQE"),
        ("brisque", "BRISQUE"),
    ):
        try:
            result = metrics[name](iqa_input)
            output[label] = float(
                result.detach()
                .mean()
                .cpu()
                .item()
            )
        except Exception as error:
            clear_cuda()
            raise RuntimeError(
                f"{label}计算失败：{repr(error)}"
            ) from error

    del iqa_input
    return output


DETAIL_FIELDS = [
    "Dataset",
    "Protocol",
    "Mode",
    "Image",
    "HazyPath",
    "GTPath",
    "Checkpoint",
    "OriginalWidth",
    "OriginalHeight",
    "EvalWidth",
    "EvalHeight",
    "PSNR",
    "SSIM",
    "NIQE",
    "BRISQUE",
    "Seconds",
    "OutputPath",
]

SUMMARY_FIELDS = [
    "Dataset",
    "Protocol",
    "Mode",
    "Tile",
    "Stride",
    "ResizeSize",
    "Checkpoint",
    "N",
    "Fail",
    "Status",
    "PSNR",
    "SSIM",
    "NIQE",
    "BRISQUE",
    "AverageSeconds",
]

BEST_FIELDS = [
    "Dataset",
    "BestProtocol",
    "Checkpoint",
    "N",
    "Fail",
    "PSNR",
    "SSIM",
    "NIQE",
    "BRISQUE",
]

FAIL_FIELDS = [
    "Dataset",
    "Protocol",
    "Image",
    "HazyPath",
    "GTPath",
    "Error",
    "Traceback",
]


def read_csv_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def remove_combo_rows(rows, dataset, protocol):
    return [
        row
        for row in rows
        if not (
            row.get("Dataset") == dataset
            and row.get("Protocol") == protocol
        )
    ]


def save_prediction(
    prediction: torch.Tensor,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    TF.to_pil_image(
        prediction.detach()
        .cpu()
        .squeeze(0)
        .clamp(0.0, 1.0)
    ).save(
        path,
        format="PNG",
        compress_level=0,
    )


def evaluate_protocol(
    dataset: str,
    protocol_name: str,
    model: nn.Module,
    device: torch.device,
    iqa_metrics,
    pairs,
    checkpoint: Path,
    args,
    detail_rows,
    failed_rows,
):
    config = PROTOCOLS[protocol_name]

    print("\n" + "=" * 112)
    print(f"Dataset      : {dataset}")
    print(f"Protocol     : {protocol_name}")
    print(f"Mode         : {config['mode']}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Pairs        : {len(pairs)}")
    if config["mode"] == "tile":
        print(
            f"Inference     : original resolution | "
            f"tile={config['tile']} | "
            f"stride={config['stride']} | no resize"
        )
    else:
        print(
            f"Inference     : resize input and GT to "
            f"{args.resize_size}x{args.resize_size}"
        )
    print(
        f"IQA protocol : NIQE/BRISQUE resize "
        f"{args.iqa_resize}x{args.iqa_resize}"
        if args.iqa_resize > 0
        else "IQA protocol : NIQE/BRISQUE use prediction resolution"
    )
    print("=" * 112)

    values = {
        key: []
        for key in (
            "PSNR",
            "SSIM",
            "NIQE",
            "BRISQUE",
            "Seconds",
        )
    }
    local_fail = 0

    progress = tqdm(
        pairs,
        desc=f"{dataset} | {protocol_name}",
        dynamic_ncols=True,
    )

    for hazy_path, gt_path in progress:
        hazy = gt = gt_eval = prediction = None

        try:
            with Image.open(hazy_path) as image:
                hazy_image = image.convert("RGB")
            with Image.open(gt_path) as image:
                gt_image = image.convert("RGB")

            if hazy_image.size != gt_image.size:
                raise RuntimeError(
                    "hazy与GT原始尺寸不一致："
                    f"hazy={hazy_image.size}, gt={gt_image.size}"
                )

            original_width, original_height = hazy_image.size

            hazy = (
                TF.to_tensor(hazy_image)
                .unsqueeze(0)
                .to(device)
                .float()
                .contiguous()
            )
            gt = (
                TF.to_tensor(gt_image)
                .unsqueeze(0)
                .to(device)
                .float()
                .contiguous()
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            prediction = infer_protocol(
                model,
                hazy,
                protocol_name,
                args.resize_size,
                args.amp,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if config["mode"] == "resize":
                gt_eval = F.interpolate(
                    gt,
                    size=(
                        args.resize_size,
                        args.resize_size,
                    ),
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0.0, 1.0)
            else:
                gt_eval = gt

            if (
                prediction.shape[-2:]
                != gt_eval.shape[-2:]
            ):
                raise RuntimeError(
                    "输出和评测GT尺寸不一致："
                    f"pred={prediction.shape[-2:]}, "
                    f"gt_eval={gt_eval.shape[-2:]}"
                )

            psnr = calc_psnr(
                prediction,
                gt_eval,
                args.metric_shave,
            )
            ssim = calc_ssim(
                prediction,
                gt_eval,
                args.metric_shave,
            )
            iqa = calc_iqa_protocol(
                prediction,
                iqa_metrics,
                args.iqa_resize,
            )

            output_path = ""
            if args.save_images:
                output_file = (
                    Path(args.output_root)
                    / "restored"
                    / protocol_name
                    / dataset
                    / f"{hazy_path.stem}.png"
                )
                save_prediction(
                    prediction,
                    output_file,
                )
                output_path = str(output_file)

            eval_height, eval_width = (
                prediction.shape[-2:]
            )

            row = {
                "Dataset": dataset,
                "Protocol": protocol_name,
                "Mode": config["mode"],
                "Image": hazy_path.name,
                "HazyPath": str(hazy_path),
                "GTPath": str(gt_path),
                "Checkpoint": str(checkpoint),
                "OriginalWidth": original_width,
                "OriginalHeight": original_height,
                "EvalWidth": eval_width,
                "EvalHeight": eval_height,
                "PSNR": psnr,
                "SSIM": ssim,
                "NIQE": iqa["NIQE"],
                "BRISQUE": iqa["BRISQUE"],
                "Seconds": elapsed,
                "OutputPath": output_path,
            }
            detail_rows.append(row)

            for key in (
                "PSNR",
                "SSIM",
                "NIQE",
                "BRISQUE",
            ):
                value = float(row[key])
                if math.isfinite(value):
                    values[key].append(value)
            values["Seconds"].append(elapsed)

            progress.set_postfix(
                PSNR=f"{safe_mean(values['PSNR']):.3f}",
                SSIM=f"{safe_mean(values['SSIM']):.5f}",
                NIQE=f"{safe_mean(values['NIQE']):.3f}",
                BRISQUE=f"{safe_mean(values['BRISQUE']):.3f}",
            )

        except Exception as error:
            local_fail += 1
            failed_rows.append(
                {
                    "Dataset": dataset,
                    "Protocol": protocol_name,
                    "Image": hazy_path.name,
                    "HazyPath": str(hazy_path),
                    "GTPath": str(gt_path),
                    "Error": repr(error),
                    "Traceback": traceback.format_exc(),
                }
            )
            print(
                f"\n❌ {dataset}/{protocol_name}/"
                f"{hazy_path.name}：{repr(error)}"
            )

        finally:
            del hazy, gt, gt_eval, prediction
            clear_cuda()

    success_count = len(values["PSNR"])

    if success_count == 0:
        status = "all_failed"
        summary = {
            "Dataset": dataset,
            "Protocol": protocol_name,
            "Mode": config["mode"],
            "Tile": (
                config["tile"]
                if config["tile"] is not None
                else ""
            ),
            "Stride": (
                config["stride"]
                if config["stride"] is not None
                else ""
            ),
            "ResizeSize": (
                args.resize_size
                if config["mode"] == "resize"
                else ""
            ),
            "Checkpoint": str(checkpoint),
            "N": 0,
            "Fail": local_fail,
            "Status": status,
            "PSNR": float("nan"),
            "SSIM": float("nan"),
            "NIQE": float("nan"),
            "BRISQUE": float("nan"),
            "AverageSeconds": float("nan"),
        }
    else:
        status = (
            "completed"
            if local_fail == 0
            else "partial"
        )
        summary = {
            "Dataset": dataset,
            "Protocol": protocol_name,
            "Mode": config["mode"],
            "Tile": (
                config["tile"]
                if config["tile"] is not None
                else ""
            ),
            "Stride": (
                config["stride"]
                if config["stride"] is not None
                else ""
            ),
            "ResizeSize": (
                args.resize_size
                if config["mode"] == "resize"
                else ""
            ),
            "Checkpoint": str(checkpoint),
            "N": success_count,
            "Fail": local_fail,
            "Status": status,
            "PSNR": safe_mean(values["PSNR"]),
            "SSIM": safe_mean(values["SSIM"]),
            "NIQE": safe_mean(values["NIQE"]),
            "BRISQUE": safe_mean(values["BRISQUE"]),
            "AverageSeconds": safe_mean(
                values["Seconds"]
            ),
        }

    print(
        f"\n✅ {dataset} | {protocol_name} | "
        f"status={summary['Status']} | "
        f"N={summary['N']} Fail={summary['Fail']} | "
        f"PSNR={summary['PSNR']:.4f} | "
        f"SSIM={summary['SSIM']:.6f} | "
        f"NIQE={summary['NIQE']:.6f} | "
        f"BRISQUE={summary['BRISQUE']:.6f}"
    )
    return summary


def numeric_value(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def best_protocol_rows(summary_rows):
    output = []

    for dataset in DATASETS:
        candidates = [
            row
            for row in summary_rows
            if row.get("Dataset") == dataset
            and int(float(row.get("N", 0))) > 0
            and math.isfinite(
                numeric_value(row, "PSNR")
            )
        ]
        if not candidates:
            continue

        # PSNR最高优先；PSNR完全相同时SSIM更高优先。
        best = max(
            candidates,
            key=lambda row: (
                numeric_value(row, "PSNR"),
                numeric_value(row, "SSIM"),
            ),
        )
        output.append(
            {
                "Dataset": dataset,
                "BestProtocol": best["Protocol"],
                "Checkpoint": best["Checkpoint"],
                "N": best["N"],
                "Fail": best["Fail"],
                "PSNR": best["PSNR"],
                "SSIM": best["SSIM"],
                "NIQE": best["NIQE"],
                "BRISQUE": best["BRISQUE"],
            }
        )

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        "MSDSPDD V2 six-dataset all-protocol evaluation"
    )

    parser.add_argument(
        "--datasets",
        default="all",
        type=str,
        help="all或逗号分隔的数据集名称。",
    )
    parser.add_argument(
        "--protocols",
        default="all",
        type=str,
        help="all或逗号分隔的协议名称。",
    )
    parser.add_argument(
        "--project_root",
        default=str(PROJECT_ROOT),
        type=str,
    )
    parser.add_argument(
        "--split_root", type=str,
        default=str(PROJECT_ROOT / "data" / "local_splits"),
        help="I/O/NH-HAZE 的保留测试集清单根目录，与权重目录独立。",
    )
    parser.add_argument(
        "--finetune_root",
        default=DEFAULT_FINETUNE,
        type=str,
    )
    parser.add_argument(
        "--general_ckpt",
        default=DEFAULT_GENERAL,
        type=str,
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT,
        type=str,
    )
    parser.add_argument(
        "--hazerd_root",
        default=str(PROJECT_ROOT / "data" / "HazeRD"),
        type=str,
    )
    parser.add_argument(
        "--sots_root",
        default=str(PROJECT_ROOT / "data" / "SOTS"),
        type=str,
    )
    parser.add_argument(
        "--resize_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--iqa_resize",
        default=512,
        type=int,
        help=(
            "NIQE/BRISQUE统一计算尺寸；"
            "设为0表示保持协议输出尺寸。"
        ),
    )
    parser.add_argument(
        "--metric_shave",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--max_images",
        default=0,
        type=int,
        help="0表示每个数据集使用全部测试图。",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        type=str,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="默认FP32，不建议正式结果临时开启。",
    )
    parser.add_argument(
        "--clear_old_outputs",
        action="store_true",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help=(
            "默认不保存8协议复原图，避免占用大量空间。"
        ),
    )

    args = parser.parse_args()

    if args.resize_size <= 0:
        parser.error("--resize_size必须大于0")
    if args.iqa_resize < 0:
        parser.error("--iqa_resize不能小于0")
    if args.metric_shave < 0:
        parser.error("--metric_shave不能小于0")
    return args


def choose_datasets(text):
    if text.strip().lower() == "all":
        return list(DATASETS)

    selected = [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]
    unknown = [
        value
        for value in selected
        if value not in DATASETS
    ]
    if unknown:
        raise ValueError(
            f"未知数据集：{unknown}；可选：{DATASETS}"
        )
    return selected


def choose_protocols(text):
    if text.strip().lower() == "all":
        return list(PROTOCOLS.keys())

    selected = [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]
    unknown = [
        value
        for value in selected
        if value not in PROTOCOLS
    ]
    if unknown:
        raise ValueError(
            f"未知协议：{unknown}；"
            f"可选：{tuple(PROTOCOLS.keys())}"
        )
    return selected


def persist_outputs(
    output_root: Path,
    detail_rows,
    summary_rows,
    failed_rows,
):
    write_csv(
        output_root / "all_protocols_detail.csv",
        detail_rows,
        DETAIL_FIELDS,
    )
    write_csv(
        output_root / "all_protocols_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    write_csv(
        output_root / "skipped_or_failed_images.csv",
        failed_rows,
        FAIL_FIELDS,
    )
    write_csv(
        output_root / "best_protocol_by_dataset.csv",
        best_protocol_rows(summary_rows),
        BEST_FIELDS,
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    project_root = Path(args.project_root)
    if project_root.resolve() != PROJECT_ROOT.resolve():
        raise RuntimeError(
            "请把本脚本放在MSDSPDD_Final_Full工程根目录。"
            f"\n脚本目录={PROJECT_ROOT}"
            f"\n--project_root={project_root}"
        )

    output_root = Path(args.output_root)
    if (
        args.clear_old_outputs
        and output_root.exists()
    ):
        shutil.rmtree(output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_datasets = choose_datasets(
        args.datasets
    )
    selected_protocols = choose_protocols(
        args.protocols
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 112)
    print("MSDSPDD Complete V2：六数据集×八协议正式测试")
    print("=" * 112)
    print("device       :", device)
    print("datasets     :", selected_datasets)
    print("protocols    :", selected_protocols)
    print("finetune root:", args.finetune_root)
    print("general ckpt :", args.general_ckpt)
    print("output root  :", output_root)
    print("metrics      : PSNR / SSIM / NIQE / BRISQUE")
    print("resize256    :", args.resize_size)
    print("IQA resize   :", args.iqa_resize)
    print("save images  :", args.save_images)
    print("AMP          :", args.amp)
    print("=" * 112)

    model = create_model(
        device,
        project_root,
    )
    iqa_metrics = build_iqa(device)

    # 未清空输出目录时，保留已经完整完成的组合并自动跳过。
    detail_rows = read_csv_rows(
        output_root / "all_protocols_detail.csv"
    )
    summary_rows = read_csv_rows(
        output_root / "all_protocols_summary.csv"
    )
    failed_rows = read_csv_rows(
        output_root / "skipped_or_failed_images.csv"
    )

    completed_keys = {
        (
            row.get("Dataset"),
            row.get("Protocol"),
        )
        for row in summary_rows
        if row.get("Status") in {
            "completed",
            "partial",
            "all_failed",
        }
    }

    for dataset in selected_datasets:
        checkpoint = checkpoint_path(
            dataset,
            args,
        )
        load_checkpoint(
            model,
            checkpoint,
        )

        pairs, pair_info = dataset_pairs(
            dataset,
            args,
        )
        if args.max_images > 0:
            pairs = pairs[:args.max_images]

        print(
            f"\n📦 {dataset}配对：{pair_info}"
        )

        for protocol_name in selected_protocols:
            key = (dataset, protocol_name)

            if key in completed_keys:
                print(
                    f"⏭️ 已有完整记录，跳过："
                    f"{dataset} | {protocol_name}"
                )
                continue

            # 防止上次中断留下该组合的半截逐图记录。
            detail_rows = remove_combo_rows(
                detail_rows,
                dataset,
                protocol_name,
            )
            failed_rows = remove_combo_rows(
                failed_rows,
                dataset,
                protocol_name,
            )
            summary_rows = remove_combo_rows(
                summary_rows,
                dataset,
                protocol_name,
            )

            summary = evaluate_protocol(
                dataset=dataset,
                protocol_name=protocol_name,
                model=model,
                device=device,
                iqa_metrics=iqa_metrics,
                pairs=pairs,
                checkpoint=checkpoint,
                args=args,
                detail_rows=detail_rows,
                failed_rows=failed_rows,
            )
            summary_rows.append(summary)

            # 每完成一个数据集×协议组合立即保存。
            persist_outputs(
                output_root,
                detail_rows,
                summary_rows,
                failed_rows,
            )

    persist_outputs(
        output_root,
        detail_rows,
        summary_rows,
        failed_rows,
    )

    print("\n" + "=" * 112)
    print("ALL DATASETS × ALL PROTOCOLS")
    print("=" * 112)
    print(
        f"{'Dataset':14s} | "
        f"{'Protocol':24s} | "
        f"{'PSNR':>8s} | "
        f"{'SSIM':>9s} | "
        f"{'NIQE':>8s} | "
        f"{'BRISQUE':>9s} | "
        f"{'N':>4s} | "
        f"{'Fail':>4s} | Status"
    )
    print("-" * 112)

    selected_summary_rows = [
        row
        for row in summary_rows
        if row.get("Dataset") in selected_datasets
        and row.get("Protocol") in selected_protocols
    ]

    for row in selected_summary_rows:
        print(
            f"{row['Dataset']:14s} | "
            f"{row['Protocol']:24s} | "
            f"{numeric_value(row, 'PSNR'):8.4f} | "
            f"{numeric_value(row, 'SSIM'):9.6f} | "
            f"{numeric_value(row, 'NIQE'):8.4f} | "
            f"{numeric_value(row, 'BRISQUE'):9.4f} | "
            f"{str(row['N']):>4s} | "
            f"{str(row['Fail']):>4s} | "
            f"{row['Status']}"
        )

    print("\n" + "=" * 112)
    print("BEST PROTOCOL BY DATASET（PSNR最高，SSIM破同分）")
    print("=" * 112)
    best_rows = best_protocol_rows(
        selected_summary_rows
    )
    for row in best_rows:
        print(
            f"{row['Dataset']:14s} | "
            f"{row['BestProtocol']:24s} | "
            f"PSNR={numeric_value(row, 'PSNR'):.4f} | "
            f"SSIM={numeric_value(row, 'SSIM'):.6f} | "
            f"NIQE={numeric_value(row, 'NIQE'):.4f} | "
            f"BRISQUE={numeric_value(row, 'BRISQUE'):.4f}"
        )

    print("\n输出文件：")
    print(
        "summary :",
        output_root / "all_protocols_summary.csv",
    )
    print(
        "detail  :",
        output_root / "all_protocols_detail.csv",
    )
    print(
        "failed  :",
        output_root / "skipped_or_failed_images.csv",
    )
    print(
        "best    :",
        output_root / "best_protocol_by_dataset.csv",
    )
    if args.save_images:
        print(
            "images  :",
            output_root
            / "restored/<Protocol>/<Dataset>/",
        )

    del model
    clear_cuda()


if __name__ == "__main__":
    main()