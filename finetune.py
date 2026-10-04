# -*- coding: utf-8 -*-
"""
MSDSPDD Complete V2：五数据集独立专项微调
============================================================

用途
----
1. 从同一个新模型通用权重 general_best.pth 出发；
2. 分别独立微调：
   - I-HAZE
   - O-HAZE
   - NH-HAZE
   - SOTS-Indoor（只使用 ITS 训练/验证，绝不读取 SOTS 测试图训练）
   - SOTS-Outdoor（只使用 OTS 训练/验证，绝不读取 SOTS 测试图训练）
3. 每个任务都重新创建模型、优化器和调度器，不串联上一个任务的权重；
4. 复用此前 FDE_dataset_finetune 中已经核验过的无重叠 split manifest，
   只复用数据划分，不加载旧 FDE 权重；
5. 每个数据集独立保存：
   - dehaze_best.pth
   - dehaze_best_model_only.pth
   - dehaze_latest.pth
   - history.csv
   - split_manifest_train.csv
   - split_manifest_val.csv
   - split_manifest_test_reserved.csv
6. Step 0 先验证通用 base，并把它保存为初始 best；之后完全按 epoch 训练。
7. 默认最多 10000 epoch，每个 epoch 验证一次；连续 20 个 epoch 没有刷新则早停。
8. 使用三阶段解冻、epoch 级 warmup + Plateau 降学习率、PSNR 优先且 SSIM 参与的 best 规则。
9. 同时检查 hazy 路径和 GT/group_id，防止同一清晰图跨 train/val/test 泄漏。
10. 启动训练前先一次性检查全部所选数据集划分；只要任一任务有泄漏或缺文件，全部任务均不开始。
11. 数据检查通过后，默认删除旧 step 版输出目录中的专项 checkpoint（只删已知权重/日志，不碰 general_best 和旧 FDE 权重）。
12. 训练：paired random crop256 + paired hflip；验证：原分辨率重叠滑窗。

建议放置位置
------------
/root/autodl-tmp/MSDSPDD_Final_Full/finetune_msdspdd_v2_5datasets.py

默认输入
--------
通用新模型权重：
/root/autodl-tmp/MSDSPDD_Final_Full/checkpoints_msdspdd_v2_general/general_best.pth

复用划分：
/root/autodl-fs/FDE_dataset_finetune/<task>/split_manifest_*.csv

默认输出
--------
/root/autodl-fs/MSDSPDD_V2_finetune_5datasets_epoch_final/<task>/

注意
----
- 本脚本导入的是 future_process.model.MSDSPDDDehazer，属于当前 Complete V2，
  不是旧 FDE 模型。
- 默认 AMP 关闭。
- 不要对 SOTS 官方测试图进行训练或验证。
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import shutil
import sys
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from future_process.model import MSDSPDDDehazer
from future_process.losses import MSDSPDDCompositeLoss

try:
    from pytorch_msssim import ssim as msssim_ssim
    HAS_MSSSIM = True
except Exception:
    HAS_MSSSIM = False


TASK_ORDER = ("ihaze", "ohaze", "nhhaze", "sots_indoor", "sots_outdoor")

TASKS: Dict[str, Dict] = {
    "ihaze": {
        "display_name": "I-HAZE",
        "batch_size": 4,
        "max_epochs": 10000,
        "eval_every_epochs": 1,
        "val_max_images": None,
    },
    "ohaze": {
        "display_name": "O-HAZE",
        "batch_size": 4,
        "max_epochs": 10000,
        "eval_every_epochs": 1,
        "val_max_images": None,
    },
    "nhhaze": {
        "display_name": "NH-HAZE",
        "batch_size": 4,
        "max_epochs": 10000,
        "eval_every_epochs": 1,
        "val_max_images": None,
    },
    "sots_indoor": {
        "display_name": "SOTS-Indoor (fine-tune on ITS)",
        "batch_size": 8,
        "max_epochs": 10000,
        "eval_every_epochs": 1,
        "val_max_images": 100,
    },
    "sots_outdoor": {
        "display_name": "SOTS-Outdoor (fine-tune on OTS)",
        "batch_size": 8,
        "max_epochs": 10000,
        "eval_every_epochs": 1,
        "val_max_images": 100,
    },
}

DEFAULT_BASE_CKPT = str(PROJECT_ROOT / "checkpoints" / "general" / "general_best.pth")
DEFAULT_SPLIT_ROOT = str(PROJECT_ROOT / "data" / "local_splits")
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "runs" / "finetune")
DEFAULT_OLD_STEP_OUTPUT_ROOT = str(PROJECT_ROOT / "runs" / "legacy_step")

# 通用模型训练完成后，专项微调采用更低学习率。
LR_CONFIG = {
    "new_modules": 2e-5,
    "residual_refiner": 1e-5,
    "isp": 5e-6,
    "efficientvit_late": 2e-6,
    "prior_fusion": 1e-6,
}

LOSS_CONFIG = {
    "final_recon_weight": 1.00,
    "base_recon_weight": 0.30,
    "ssim_weight": 0.20,
    "perceptual_weight": 0.03,
    "edge_weight": 0.05,
    "color_weight": 0.02,
    "physics_weight": 0.005,
    "pgdsp_safety_weight": 0.005,
    "residual_weight": 0.001,
    "route_weight": 0.001,
    "minimum_route_usage": 0.03,
}

HISTORY_FIELDS = [
    "epoch",
    "stage",
    "train_loss",
    "val_psnr",
    "val_ssim",
    "best_psnr",
    "best_ssim",
    "early_stop_counter",
    "plateau_bad_epochs",
    "lr_reductions",
    "route_factor",
    "physics_weight",
    "is_best",
    "is_base_baseline",
    "lr_new_modules",
    "lr_residual_refiner",
    "lr_isp",
    "lr_efficientvit_late",
    "lr_prior_fusion",
]


@dataclass(frozen=True)
class PairRecord:
    hazy_path: str
    gt_path: str
    group_id: str = ""
    match_type: str = "manifest"


# ============================================================
# 基础工具
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def extract_model_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "params", "params_ema"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise TypeError("无法从 checkpoint 中提取模型参数。")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_manifest(path: Path, split_name: str, pairs: Sequence[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "hazy_path", "gt_path", "group_id", "match_type"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"split": split_name, **asdict(pair)})


def read_manifest(path: Path, allow_empty: bool = False) -> List[PairRecord]:
    if not path.exists():
        raise FileNotFoundError(f"划分清单不存在: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    pairs: List[PairRecord] = []
    missing_files: List[str] = []

    for row in rows:
        hazy = Path(row.get("hazy_path", "").strip())
        gt = Path(row.get("gt_path", "").strip())
        if not str(hazy) or not str(gt):
            continue
        if not hazy.is_file() or not gt.is_file():
            missing_files.append(f"hazy={hazy} | gt={gt}")
            continue
        pairs.append(
            PairRecord(
                hazy_path=str(hazy),
                gt_path=str(gt),
                group_id=row.get("group_id", "").strip() or str(gt.resolve()),
                match_type=row.get("match_type", "manifest").strip() or "manifest",
            )
        )

    if missing_files:
        preview = "\n".join("  " + item for item in missing_files[:10])
        raise FileNotFoundError(
            f"{path} 中有 {len(missing_files)} 个配对文件不存在，前10项：\n{preview}"
        )

    if not pairs and not allow_empty:
        raise RuntimeError(f"清单为空: {path}")
    return pairs


def resolved_hazy_set(pairs: Sequence[PairRecord]) -> set:
    return {str(Path(pair.hazy_path).resolve()) for pair in pairs}


def resolved_gt_set(pairs: Sequence[PairRecord]) -> set:
    return {str(Path(pair.gt_path).resolve()) for pair in pairs}


def normalized_group_set(pairs: Sequence[PairRecord]) -> set:
    groups = set()
    for pair in pairs:
        value = (pair.group_id or "").strip()
        if value:
            groups.add(value.lower())
        else:
            groups.add(str(Path(pair.gt_path).resolve()).lower())
    return groups


def split_overlap_report(
    train_pairs: Sequence[PairRecord],
    val_pairs: Sequence[PairRecord],
    test_pairs: Sequence[PairRecord],
) -> Dict[str, set]:
    train_hazy = resolved_hazy_set(train_pairs)
    val_hazy = resolved_hazy_set(val_pairs)
    test_hazy = resolved_hazy_set(test_pairs)

    train_gt = resolved_gt_set(train_pairs)
    val_gt = resolved_gt_set(val_pairs)
    test_gt = resolved_gt_set(test_pairs)

    train_group = normalized_group_set(train_pairs)
    val_group = normalized_group_set(val_pairs)
    test_group = normalized_group_set(test_pairs)

    return {
        "hazy_train_val": train_hazy & val_hazy,
        "hazy_train_test": train_hazy & test_hazy,
        "hazy_val_test": val_hazy & test_hazy,
        "gt_train_val": train_gt & val_gt,
        "gt_train_test": train_gt & test_gt,
        "gt_val_test": val_gt & test_gt,
        "group_train_val": train_group & val_group,
        "group_train_test": train_group & test_group,
        "group_val_test": val_group & test_group,
    }

def load_and_copy_splits(
    task_name: str,
    split_root: Path,
    output_dir: Path,
) -> Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]:
    source_dir = split_root / task_name
    train_pairs = read_manifest(source_dir / "split_manifest_train.csv")
    val_pairs = read_manifest(source_dir / "split_manifest_val.csv")
    test_pairs = read_manifest(
        source_dir / "split_manifest_test_reserved.csv",
        allow_empty=True,
    )

    overlaps = split_overlap_report(train_pairs, val_pairs, test_pairs)
    bad = {name: values for name, values in overlaps.items() if values}
    if bad:
        previews = []
        for name, values in bad.items():
            sample = list(sorted(values))[:5]
            previews.append(f"{name}: count={len(values)}, sample={sample}")
        raise RuntimeError(
            f"{task_name} 划分存在 hazy/GT/group 泄漏：\n  "
            + "\n  ".join(previews)
        )

    write_manifest(output_dir / "split_manifest_train.csv", "train", train_pairs)
    write_manifest(output_dir / "split_manifest_val.csv", "val", val_pairs)
    write_manifest(
        output_dir / "split_manifest_test_reserved.csv",
        "test_reserved",
        test_pairs,
    )

    print("\n" + "=" * 88)
    print(f"{TASKS[task_name]['display_name']} 划分检查通过")
    print(f"train         : {len(train_pairs)}")
    print(f"val           : {len(val_pairs)}")
    print(f"test_reserved : {len(test_pairs)}")
    print("hazy train ∩ val/test : 0")
    print("hazy val   ∩ test     : 0")
    print("GT   train ∩ val/test : 0")
    print("GT   val   ∩ test     : 0")
    print("group train/val/test  : 0")
    print(f"split source  : {source_dir}")
    print(f"split copy    : {output_dir}")
    print("=" * 88)

    return train_pairs, val_pairs, test_pairs


# ============================================================
# 数据集
# ============================================================

class PairedDehazeDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairRecord],
        crop_size: int,
        is_train: bool,
    ):
        self.pairs = list(pairs)
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def load_pair(pair: PairRecord) -> Tuple[Image.Image, Image.Image]:
        hazy = Image.open(pair.hazy_path).convert("RGB")
        gt = Image.open(pair.gt_path).convert("RGB")
        if hazy.size != gt.size:
            gt = TF.resize(
                gt,
                hazy.size[::-1],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
        return hazy, gt

    @staticmethod
    def ensure_min_size(
        hazy: Image.Image,
        gt: Image.Image,
        crop_size: int,
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = hazy.size
        if width >= crop_size and height >= crop_size:
            return hazy, gt

        scale = max(crop_size / max(width, 1), crop_size / max(height, 1))
        new_width = int(math.ceil(width * scale))
        new_height = int(math.ceil(height * scale))
        hazy = TF.resize(
            hazy,
            (new_height, new_width),
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        gt = TF.resize(
            gt,
            (new_height, new_width),
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        return hazy, gt

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        hazy, gt = self.load_pair(pair)

        if self.is_train:
            hazy, gt = self.ensure_min_size(hazy, gt, self.crop_size)
            width, height = hazy.size
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            hazy = TF.crop(hazy, top, left, self.crop_size, self.crop_size)
            gt = TF.crop(gt, top, left, self.crop_size, self.crop_size)

            if random.random() < 0.5:
                hazy = TF.hflip(hazy)
                gt = TF.hflip(gt)

        return (
            TF.to_tensor(hazy).float().clamp(0.0, 1.0),
            TF.to_tensor(gt).float().clamp(0.0, 1.0),
            pair.hazy_path,
        )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def uniform_subset_indices(length: int, max_items: Optional[int]) -> List[int]:
    if max_items is None or max_items <= 0 or length <= max_items:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=int(max_items))
    indices = sorted({int(round(value)) for value in raw})
    if len(indices) < int(max_items):
        for index in range(length):
            if index not in indices:
                indices.append(index)
            if len(indices) >= int(max_items):
                break
    return sorted(indices[: int(max_items)])


def create_loaders(
    train_pairs: Sequence[PairRecord],
    val_pairs: Sequence[PairRecord],
    crop_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_max_images: Optional[int],
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PairedDehazeDataset(train_pairs, crop_size, is_train=True)
    full_val_dataset = PairedDehazeDataset(val_pairs, crop_size, is_train=False)

    val_indices = uniform_subset_indices(len(full_val_dataset), val_max_images)
    if len(val_indices) < len(full_val_dataset):
        val_dataset = Subset(full_val_dataset, val_indices)
        print(
            f"固定均匀验证子集: {len(val_indices)} / {len(full_val_dataset)} 张，"
            "不会只取清单前100张。"
        )
    else:
        val_dataset = full_val_dataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# ============================================================
# 模型、参数分组与损失
# ============================================================

def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def unique_parameters(
    modules_or_params: Iterable,
    used_ids: set,
) -> List[nn.Parameter]:
    output: List[nn.Parameter] = []
    for item in modules_or_params:
        if isinstance(item, nn.Parameter):
            iterator = [item]
        elif isinstance(item, nn.Module):
            iterator = item.parameters()
        else:
            iterator = item
        for parameter in iterator:
            if id(parameter) in used_ids:
                continue
            used_ids.add(id(parameter))
            output.append(parameter)
    return output


def late_backbone_modules(model: MSDSPDDDehazer) -> List[nn.Module]:
    backbone = model.extractor.backbone
    modules: List[nn.Module] = []
    if hasattr(backbone, "stages"):
        stages = list(backbone.stages)
        if len(stages) >= 3:
            modules.extend(stages[1:3])
        elif len(stages) >= 2:
            modules.extend(stages[-2:])
    else:
        if hasattr(backbone, "stage2"):
            modules.append(backbone.stage2)
        if hasattr(backbone, "stage3"):
            modules.append(backbone.stage3)
    if not modules:
        raise RuntimeError("未找到 EfficientViT 后两级模块。")
    return modules


def new_reconstruction_modules(model: MSDSPDDDehazer) -> List[nn.Module]:
    return [
        model.extractor.response_encoder,
        model.extractor.route_adapters,
        model.fusion_s2,
        model.fusion_s4,
        model.fusion_s8,
        model.fusion_s16,
        model.local_detail_s2,
        model.local_detail_s4,
        model.mid_scale_s8,
        model.global_context_s16,
        model.decode_s8,
        model.decode_s4,
        model.decode_s2,
        model.decode_full,
        model.raw_detail_gate,
        model.output_head,
    ]


def create_model(device: torch.device) -> MSDSPDDDehazer:
    weight_path = PROJECT_ROOT / "future_process" / "efficientvit_b1_r256.pt"
    if not weight_path.is_file():
        raise FileNotFoundError(f"EfficientViT 权重不存在: {weight_path}")

    model = MSDSPDDDehazer(
        enable_end_to_end=True,
        weight_path=str(weight_path),
        route_temperature=1.0,
        use_mamba=True,
        enable_refiner=True,
        allow_backbone_fallback=False,
    ).to(device)

    if getattr(model, "backbone_is_fallback", False):
        raise RuntimeError("当前模型使用了备用卷积主干，禁止正式微调。")
    if not getattr(model, "mamba_enabled", True):
        raise RuntimeError("VMamba/VSSBlock 未启用，禁止正式微调。")
    return model


def finetune_stage_name(
    epoch_number: int,
    stage2_epoch: int,
    stage3_epoch: int,
) -> str:
    if epoch_number <= int(stage2_epoch):
        return "stage1_reconstruction_refiner"
    if epoch_number <= int(stage3_epoch):
        return "stage2_add_isp_prior_fusion"
    return "stage3_add_efficientvit_late"


def configure_finetune_stage(
    model: MSDSPDDDehazer,
    epoch_number: int,
    stage2_epoch: int,
    stage3_epoch: int,
    verbose: bool = True,
) -> str:
    """完全按 epoch 控制三阶段解冻。"""
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Stage 1: 保护通用能力，只适配重建与残差输出。
    for module in new_reconstruction_modules(model):
        set_module_trainable(module, True)
    set_module_trainable(model.residual_refiner, True)

    stage_name = finetune_stage_name(epoch_number, stage2_epoch, stage3_epoch)
    if stage_name == "stage2_add_isp_prior_fusion":
        # Stage 2: 再允许 ISP 和物理先验融合适配目标域。
        set_module_trainable(model.isp_module, True)
        set_module_trainable(model.prior_engine.fusion_conv, True)
    elif stage_name == "stage3_add_efficientvit_late":
        # Stage 3: 最后才以最低学习率微调 EfficientViT 后两级。
        set_module_trainable(model.isp_module, True)
        set_module_trainable(model.prior_engine.fusion_conv, True)
        for module in late_backbone_modules(model):
            set_module_trainable(module, True)

    if hasattr(model, "pg_dsp_module"):
        set_module_trainable(model.pg_dsp_module, False)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🔧 训练阶段: {stage_name}")
        print(f"   可训练参数: {trainable / 1e6:.3f} M / {total / 1e6:.3f} M")
    return stage_name


def configure_finetune_optimizer(
    model: MSDSPDDDehazer,
    weight_decay: float,
    lr_scale: float,
) -> torch.optim.Optimizer:
    if not hasattr(model, "prior_engine") or not hasattr(model.prior_engine, "fusion_conv"):
        raise AttributeError("模型缺少 prior_engine.fusion_conv。")

    used_ids: set = set()
    groups = [
        {
            "name": "new_modules",
            "params": unique_parameters(new_reconstruction_modules(model), used_ids),
            "lr": LR_CONFIG["new_modules"] * lr_scale,
        },
        {
            "name": "residual_refiner",
            "params": unique_parameters([model.residual_refiner], used_ids),
            "lr": LR_CONFIG["residual_refiner"] * lr_scale,
        },
        {
            "name": "isp",
            "params": unique_parameters([model.isp_module], used_ids),
            "lr": LR_CONFIG["isp"] * lr_scale,
        },
        {
            "name": "efficientvit_late",
            "params": unique_parameters(late_backbone_modules(model), used_ids),
            "lr": LR_CONFIG["efficientvit_late"] * lr_scale,
        },
        {
            "name": "prior_fusion",
            "params": unique_parameters([model.prior_engine.fusion_conv], used_ids),
            "lr": LR_CONFIG["prior_fusion"] * lr_scale,
        },
    ]
    groups = [group for group in groups if group["params"]]
    if not groups:
        raise RuntimeError("没有可放入优化器的参数。")

    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    print("优化器参数组（后续由epoch阶段控制requires_grad）：")
    for group in optimizer.param_groups:
        print(f"  {group['name']:<20s} base_lr={group['lr']:.2e} params={len(group['params'])}")
    return optimizer


def freeze_batch_norm(model: nn.Module) -> None:
    if hasattr(model, "freeze_batch_norm"):
        model.freeze_batch_norm(freeze_affine=False)
        return
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def create_criterion(device: torch.device) -> MSDSPDDCompositeLoss:
    criterion = MSDSPDDCompositeLoss(
        final_recon_weight=LOSS_CONFIG["final_recon_weight"],
        base_recon_weight=LOSS_CONFIG["base_recon_weight"],
        ssim_weight=LOSS_CONFIG["ssim_weight"],
        perceptual_weight=LOSS_CONFIG["perceptual_weight"],
        edge_weight=LOSS_CONFIG["edge_weight"],
        color_weight=LOSS_CONFIG["color_weight"],
        physics_weight=0.0,
        pgdsp_safety_weight=LOSS_CONFIG["pgdsp_safety_weight"],
        residual_weight=LOSS_CONFIG["residual_weight"],
        route_weight=LOSS_CONFIG["route_weight"],
        minimum_route_usage=LOSS_CONFIG["minimum_route_usage"],
    ).to(device)
    criterion.physics_weight = 0.0
    return criterion


def current_physics_weight(
    epoch_number: int,
    stage2_epoch: int,
    ramp_epochs: int,
) -> float:
    if epoch_number <= int(stage2_epoch):
        return 0.0
    progress = min(
        1.0,
        max(0.0, (epoch_number - int(stage2_epoch)) / max(int(ramp_epochs), 1)),
    )
    return float(LOSS_CONFIG["physics_weight"]) * progress


def current_route_factor(epoch_number: int, decay_epochs: int) -> float:
    return max(
        0.0,
        1.0 - (epoch_number - 1) / max(int(decay_epochs), 1),
    )


class WarmupPlateauByEpoch:
    """epoch级线性warmup，之后按验证PSNR停滞自动降学习率。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        plateau_patience: int,
        plateau_factor: float,
        plateau_threshold: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_factor = float(plateau_factor)
        if not 0.0 < self.plateau_factor < 1.0:
            raise ValueError("plateau_factor 必须在 0 和 1 之间。")
        self.plateau_threshold = max(0.0, float(plateau_threshold))
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.best_metric = -float("inf")
        self.bad_epochs = 0
        self.reductions = 0
        self.current_epoch = 0

    def begin_epoch(self, epoch_number: int) -> None:
        self.current_epoch = int(epoch_number)
        if self.warmup_epochs <= 0 or epoch_number > self.warmup_epochs:
            return
        progress = float(epoch_number) / float(self.warmup_epochs)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + (base_lr - self.min_lr) * progress

    def reset_plateau(self, metric: Optional[float] = None) -> None:
        self.bad_epochs = 0
        if metric is not None and np.isfinite(metric):
            self.best_metric = float(metric)

    def restore_groups_to_base(self, names: Sequence[str]) -> None:
        wanted = set(names)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if group.get("name") in wanted:
                group["lr"] = max(self.min_lr, float(base_lr))

    def step(self, metric: float, epoch_number: int) -> bool:
        metric = float(metric)
        if epoch_number <= self.warmup_epochs:
            if metric > self.best_metric:
                self.best_metric = metric
            self.bad_epochs = 0
            return False

        if metric > self.best_metric + self.plateau_threshold:
            self.best_metric = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs < self.plateau_patience:
            return False

        reduced = False
        for group in self.optimizer.param_groups:
            old_lr = float(group["lr"])
            new_lr = max(self.min_lr, old_lr * self.plateau_factor)
            group["lr"] = new_lr
            reduced = reduced or new_lr < old_lr - 1e-15
        self.bad_epochs = 0
        if reduced:
            self.reductions += 1
        return reduced

    def state_dict(self) -> Dict:
        return {
            "warmup_epochs": self.warmup_epochs,
            "plateau_patience": self.plateau_patience,
            "plateau_factor": self.plateau_factor,
            "plateau_threshold": self.plateau_threshold,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "reductions": self.reductions,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.warmup_epochs = int(state["warmup_epochs"])
        self.plateau_patience = int(state["plateau_patience"])
        self.plateau_factor = float(state["plateau_factor"])
        self.plateau_threshold = float(state["plateau_threshold"])
        self.min_lr = float(state["min_lr"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.best_metric = float(state.get("best_metric", -float("inf")))
        self.bad_epochs = int(state.get("bad_epochs", 0))
        self.reductions = int(state.get("reductions", 0))
        self.current_epoch = int(state.get("current_epoch", 0))



# ============================================================
# 推理与指标
# ============================================================

def parse_model_output(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("final_out", "J_final", "out", "output", "pred"):
            if key in output:
                return output[key]
    raise RuntimeError(f"无法解析模型输出: {type(output)}")


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 32,
    min_size: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    target_h = height if min_size is None else max(height, int(min_size))
    target_w = width if min_size is None else max(width, int(min_size))
    target_h = ((target_h + multiple - 1) // multiple) * multiple
    target_w = ((target_w + multiple - 1) // multiple) * multiple
    pad_h = target_h - height
    pad_w = target_w - width
    if pad_h == 0 and pad_w == 0:
        return x.contiguous(), height, width
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode).contiguous(), height, width


@torch.no_grad()
def model_forward_eval(
    model: nn.Module,
    x: torch.Tensor,
    amp_enabled: bool,
) -> torch.Tensor:
    x = x.contiguous().float()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled and x.device.type == "cuda",
    ):
        pred = parse_model_output(model(x))
    if pred.shape[-2:] != x.shape[-2:]:
        pred = F.interpolate(
            pred,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return pred.float().clamp(0.0, 1.0)


def sliding_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def tiled_forward(
    model: nn.Module,
    x: torch.Tensor,
    tile: int,
    stride: int,
    amp_enabled: bool,
) -> torch.Tensor:
    padded, original_h, original_w = pad_to_multiple(x, 32, min_size=tile)
    _, _, height, width = padded.shape
    output_sum = torch.zeros((1, 3, height, width), device=x.device, dtype=torch.float32)
    weight = torch.zeros((1, 1, height, width), device=x.device, dtype=torch.float32)

    for top in sliding_starts(height, tile, stride):
        for left in sliding_starts(width, tile, stride):
            patch = padded[..., top:top + tile, left:left + tile].contiguous()
            pred = model_forward_eval(model, patch, amp_enabled)
            output_sum[..., top:top + tile, left:left + tile] += pred
            weight[..., top:top + tile, left:left + tile] += 1.0

    pred = output_sum / torch.clamp(weight, min=1.0)
    return pred[..., :original_h, :original_w].contiguous().clamp(0.0, 1.0)


def fallback_ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, 1, pad)
    mu_y = F.avg_pool2d(gt, window_size, 1, pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * gt, window_size, 1, pad) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    value = (
        (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12)
    )
    return float(value.mean().item())


@torch.no_grad()
def calculate_psnr_ssim(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    pred = pred.float().clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)
    mse = F.mse_loss(pred, gt).item()
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)

    if HAS_MSSSIM:
        try:
            ssim = float(msssim_ssim(pred, gt, data_range=1.0, size_average=True).item())
        except Exception:
            ssim = fallback_ssim(pred, gt)
    else:
        ssim = fallback_ssim(pred, gt)
    return psnr, ssim


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tile: int,
    stride: int,
    max_images: Optional[int],
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.eval()
    psnrs: List[float] = []
    ssims: List[float] = []

    pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True)
    for index, (hazy, gt, _) in enumerate(pbar):
        if max_images is not None and index >= int(max_images):
            break

        hazy = hazy.to(device, non_blocking=True).float().contiguous()
        gt = gt.to(device, non_blocking=True).float().contiguous()
        pred = tiled_forward(model, hazy, tile, stride, amp_enabled)

        if pred.shape[-2:] != gt.shape[-2:]:
            raise RuntimeError(
                f"验证输出与GT尺寸不一致: pred={pred.shape[-2:]}, gt={gt.shape[-2:]}"
            )

        psnr, ssim = calculate_psnr_ssim(pred, gt)
        psnrs.append(psnr)
        ssims.append(ssim)
        pbar.set_postfix(
            PSNR=f"{np.mean(psnrs):.3f}",
            SSIM=f"{np.mean(ssims):.5f}",
        )
        if (index + 1) % 5 == 0:
            clear_cuda()

    if not psnrs:
        raise RuntimeError("验证集没有成功计算任何图片。")
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ============================================================
# 日志与checkpoint
# ============================================================

def append_history(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {
        group.get("name", f"group_{index}"): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def build_checkpoint(
    task_name: str,
    epoch: int,
    stage_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupPlateauByEpoch,
    scaler,
    best_psnr: float,
    best_ssim: float,
    current_psnr: float,
    current_ssim: float,
    early_stop_counter: int,
    route_factor: float,
    physics_weight: float,
    base_ckpt: Path,
    split_root: Path,
    task_cfg: Dict,
    args,
    is_base_baseline: bool,
    training_finished: bool,
) -> Dict:
    return {
        "model_name": "MSDSPDD Complete V2",
        "task": task_name,
        "display_name": task_cfg["display_name"],
        "epoch": int(epoch),
        "stage_name": stage_name,
        "training_mode": "epoch_staged_plateau_v3_split_audit_purge",
        "best_psnr": float(best_psnr),
        "best_ssim": float(best_ssim),
        "current_psnr": float(current_psnr),
        "current_ssim": float(current_ssim),
        "early_stop_counter": int(early_stop_counter),
        "route_factor": float(route_factor),
        "physics_weight": float(physics_weight),
        "is_base_baseline": bool(is_base_baseline),
        "training_finished": bool(training_finished),
        "base_checkpoint": str(base_ckpt),
        "split_root": str(split_root),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "task_config": task_cfg,
        "loss_config": LOSS_CONFIG,
        "lr_config": LR_CONFIG,
        "args": vars(args),
    }


def save_best(
    checkpoint: Dict,
    model: nn.Module,
    best_path: Path,
    best_model_only_path: Path,
) -> None:
    torch.save(checkpoint, best_path)
    torch.save(model.state_dict(), best_model_only_path)


# ============================================================
# 单个任务训练
# ============================================================

CHECKPOINT_ARTIFACT_NAMES = (
    "dehaze_best.pth",
    "dehaze_best_model_only.pth",
    "dehaze_latest.pth",
    "history.csv",
)


def remove_task_checkpoints(output_dir: Path) -> List[Path]:
    """删除一个任务目录中的训练权重/历史，不删除数据划分清单。"""
    removed: List[Path] = []
    for name in CHECKPOINT_ARTIFACT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def purge_old_step_outputs(
    selected_tasks: Sequence[str],
    old_step_output_root: Path,
) -> List[Path]:
    """
    删除旧 step 版专项微调产生的 checkpoint。

    安全边界：
    - 仅处理 DEFAULT_OLD_STEP_OUTPUT_ROOT（或命令行显式指定目录）下的所选任务；
    - 仅删除 CHECKPOINT_ARTIFACT_NAMES；
    - 不删除 split_manifest_*.csv；
    - 不接触 general_best.pth；
    - 不接触 /root/autodl-fs/FDE_dataset_finetune 中的旧 FDE 权重。
    """
    removed: List[Path] = []
    print("\n" + "=" * 100)
    print("清理旧 step 版专项微调输出")
    print(f"old step root : {old_step_output_root}")

    if not old_step_output_root.exists():
        print("旧 step 输出目录不存在，无需清理。")
        print("=" * 100)
        return removed

    for task_name in selected_tasks:
        task_dir = old_step_output_root / task_name
        task_removed = remove_task_checkpoints(task_dir)
        if task_removed:
            for path in task_removed:
                print(f"  已删除: {path}")
            removed.extend(task_removed)
        else:
            print(f"  {task_name}: 没有找到旧 step checkpoint")

    print(f"共删除旧 step 文件: {len(removed)}")
    print("注意：general_best、split manifest、旧 FDE 权重均未删除。")
    print("=" * 100)
    return removed


def preflight_selected_splits(
    selected_tasks: Sequence[str],
    split_root: Path,
    output_root: Path,
) -> Dict[str, Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]]:
    """训练前一次性检查全部所选任务；任一失败则整体中止。"""
    print("\n" + "=" * 100)
    print("阶段 0：全部数据集划分预检查（此阶段不训练，也不删除任何权重）")
    print("检查内容：文件存在性、hazy 重叠、GT 重叠、group_id 重叠")
    print("=" * 100)

    split_cache: Dict[
        str,
        Tuple[List[PairRecord], List[PairRecord], List[PairRecord]],
    ] = {}
    report_rows: List[Dict[str, object]] = []

    for task_name in selected_tasks:
        output_dir = output_root / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        train_pairs, val_pairs, test_pairs = load_and_copy_splits(
            task_name=task_name,
            split_root=split_root,
            output_dir=output_dir,
        )
        split_cache[task_name] = (train_pairs, val_pairs, test_pairs)
        report_rows.append({
            "task": task_name,
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test_reserved": len(test_pairs),
            "hazy_overlap": 0,
            "gt_overlap": 0,
            "group_overlap": 0,
            "status": "passed",
        })

    report_path = output_root / "split_audit_summary.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task", "train", "val", "test_reserved",
                "hazy_overlap", "gt_overlap", "group_overlap", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n✅ 全部所选数据集划分检查通过，训练/验证/测试无 hazy、GT、group_id 交叉。")
    print(f"检查报告: {report_path}")
    return split_cache


def train_one_task(
    task_name: str,
    base_ckpt: Path,
    split_root: Path,
    output_root: Path,
    args,
    prechecked_splits: Optional[
        Tuple[List[PairRecord], List[PairRecord], List[PairRecord]]
    ] = None,
) -> str:
    task_cfg = dict(TASKS[task_name])
    if args.batch_size > 0:
        task_cfg["batch_size"] = args.batch_size
    if args.max_epochs > 0:
        task_cfg["max_epochs"] = args.max_epochs
    if args.eval_every_epochs > 0:
        task_cfg["eval_every_epochs"] = args.eval_every_epochs

    output_dir = output_root / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if prechecked_splits is None:
        train_pairs, val_pairs, _ = load_and_copy_splits(
            task_name,
            split_root,
            output_dir,
        )
    else:
        train_pairs, val_pairs, _ = prechecked_splits

    latest_path = output_dir / "dehaze_latest.pth"
    best_path = output_dir / "dehaze_best.pth"
    best_model_only_path = output_dir / "dehaze_best_model_only.pth"
    history_path = output_dir / "history.csv"

    if args.force_restart:
        print(f"⚠️ {task_name}: 强制从 base 重新开始，删除本任务旧checkpoint。")
        removed_current = remove_task_checkpoints(output_dir)
        print(f"   删除当前epoch版文件: {len(removed_current)} 个")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_seed = args.seed + TASK_ORDER.index(task_name) * 1000
    set_seed(task_seed)

    train_loader, val_loader = create_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        crop_size=args.crop_size,
        batch_size=task_cfg["batch_size"],
        num_workers=args.num_workers,
        seed=task_seed,
        val_max_images=task_cfg.get("val_max_images"),
    )

    batches_per_epoch = max(1, len(train_loader))
    task_cfg["batches_per_epoch"] = batches_per_epoch
    task_cfg["validation_images"] = len(val_loader.dataset)

    model = create_model(device)
    optimizer = configure_finetune_optimizer(
        model,
        weight_decay=args.weight_decay,
        lr_scale=args.lr_scale,
    )
    scheduler = WarmupPlateauByEpoch(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        plateau_patience=args.lr_patience,
        plateau_factor=args.lr_factor,
        plateau_threshold=args.lr_threshold,
        min_lr=args.min_lr,
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.amp and torch.cuda.is_available()
        else None
    )
    criterion = create_criterion(device)

    start_epoch = 0
    best_psnr = -float("inf")
    best_ssim = -float("inf")
    current_psnr = float("nan")
    current_ssim = float("nan")
    early_stop_counter = 0
    route_factor = 1.0
    physics_weight = 0.0
    stage_name = configure_finetune_stage(
        model,
        epoch_number=1,
        stage2_epoch=args.stage2_epoch,
        stage3_epoch=args.stage3_epoch,
        verbose=True,
    )

    if latest_path.exists() and not args.force_restart:
        print(f"📂 恢复当前任务断点: {latest_path}")
        checkpoint = torch_load(latest_path, map_location=device)
        if checkpoint.get("task") not in {None, task_name}:
            raise RuntimeError(
                f"latest属于 {checkpoint.get('task')}，当前任务却是 {task_name}。"
            )
        if checkpoint.get("training_mode") != "epoch_staged_plateau_v3_split_audit_purge":
            raise RuntimeError(
                "检测到旧版断点，无法安全继承新的分阶段/Plateau状态。"
                "请加 --force_restart，从 general_best 重新开始本任务。"
            )

        model.load_state_dict(extract_model_state(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = int(checkpoint.get("epoch", 0))
        best_psnr = float(checkpoint.get("best_psnr", -float("inf")))
        best_ssim = float(checkpoint.get("best_ssim", -float("inf")))
        current_psnr = float(checkpoint.get("current_psnr", best_psnr))
        current_ssim = float(checkpoint.get("current_ssim", best_ssim))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        route_factor = float(checkpoint.get("route_factor", 1.0))
        physics_weight = float(checkpoint.get("physics_weight", 0.0))
        stage_name = configure_finetune_stage(
            model,
            epoch_number=max(1, start_epoch + 1),
            stage2_epoch=args.stage2_epoch,
            stage3_epoch=args.stage3_epoch,
            verbose=True,
        )

        if checkpoint.get("training_finished", False) and not args.no_skip_finished:
            print(
                f"⏭️ {task_cfg['display_name']} 已完成，自动跳过。"
                f" best={best_psnr:.4f}/{best_ssim:.6f}"
            )
            return "skipped_finished"

        print(
            f"✅ 从 epoch {start_epoch} 继续 | "
            f"best={best_psnr:.4f}/{best_ssim:.6f} | "
            f"early={early_stop_counter}/{args.patience}"
        )
    else:
        print(f"🔥 加载新模型通用 base: {base_ckpt}")
        base_checkpoint = torch_load(base_ckpt, map_location="cpu")
        state_dict = extract_model_state(base_checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=args.strict)
        if args.strict:
            missing, unexpected = [], []
        else:
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
        print(f"   missing keys   : {len(missing)}")
        print(f"   unexpected keys: {len(unexpected)}")
        if missing or unexpected:
            print("   ⚠️ 新V2通用权重建议使用 --strict 完全匹配。")

        print("\n🔎 Epoch 0：验证通用 base，建立初始 best")
        current_psnr, current_ssim = validate(
            model=model,
            loader=val_loader,
            device=device,
            tile=args.val_tile,
            stride=args.val_stride,
            max_images=None,
            amp_enabled=args.amp,
        )
        best_psnr = current_psnr
        best_ssim = current_ssim
        scheduler.reset_plateau(current_psnr)

        checkpoint = build_checkpoint(
            task_name=task_name,
            epoch=0,
            stage_name=stage_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_psnr=best_psnr,
            best_ssim=best_ssim,
            current_psnr=current_psnr,
            current_ssim=current_ssim,
            early_stop_counter=0,
            route_factor=1.0,
            physics_weight=0.0,
            base_ckpt=base_ckpt,
            split_root=split_root,
            task_cfg=task_cfg,
            args=args,
            is_base_baseline=True,
            training_finished=False,
        )
        save_best(checkpoint, model, best_path, best_model_only_path)
        torch.save(checkpoint, latest_path)
        append_history(
            history_path,
            {
                "epoch": 0,
                "stage": stage_name,
                "train_loss": "",
                "val_psnr": current_psnr,
                "val_ssim": current_ssim,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "early_stop_counter": 0,
                "plateau_bad_epochs": scheduler.bad_epochs,
                "lr_reductions": scheduler.reductions,
                "route_factor": 1.0,
                "physics_weight": 0.0,
                "is_best": 1,
                "is_base_baseline": 1,
                **{f"lr_{key}": value for key, value in optimizer_lrs(optimizer).items()},
            },
        )
        print(f"✅ Base baseline: PSNR={best_psnr:.4f}, SSIM={best_ssim:.6f}")

    print("\n" + "=" * 100)
    print(f"任务                : {task_cfg['display_name']}")
    print(f"新模型base          : {base_ckpt}")
    print(f"train / val         : {len(train_pairs)} / {len(val_loader.dataset)}")
    print(f"crop / batch        : {args.crop_size} / {task_cfg['batch_size']}")
    print(f"batches per epoch   : {batches_per_epoch}（仅显示批次数）")
    print(f"start / max epoch   : {start_epoch} / {task_cfg['max_epochs']}")
    print(f"eval every          : {task_cfg['eval_every_epochs']} epoch")
    print(f"early stop          : {args.patience} epochs")
    print(f"stage 1 / 2 / 3     : 1-{args.stage2_epoch} / "
          f"{args.stage2_epoch + 1}-{args.stage3_epoch} / {args.stage3_epoch + 1}+")
    print(f"best rule           : PSNR +{args.min_delta:.3f}dB，"
          f"或PSNR不下降且SSIM +{args.ssim_min_delta:.6f}")
    print(f"LR scheduler        : warmup {args.warmup_epochs} epoch + "
          f"plateau({args.lr_patience}, x{args.lr_factor})")
    print(f"val protocol        : tile{args.val_tile}_stride{args.val_stride}")
    print(f"AMP                 : {args.amp}")
    print(f"output              : {output_dir}")
    print("=" * 100)

    last_epoch = start_epoch
    stop_reason = "max_epochs"
    previous_stage = stage_name

    for epoch_index in range(start_epoch, int(task_cfg["max_epochs"])):
        epoch_number = epoch_index + 1
        next_stage = finetune_stage_name(
            epoch_number,
            args.stage2_epoch,
            args.stage3_epoch,
        )
        stage_changed = next_stage != previous_stage
        stage_name = configure_finetune_stage(
            model,
            epoch_number=epoch_number,
            stage2_epoch=args.stage2_epoch,
            stage3_epoch=args.stage3_epoch,
            verbose=stage_changed,
        )
        if stage_changed:
            print("🔄 进入新解冻阶段：早停与Plateau计数清零，给新模块充分适应机会。")
            if stage_name == "stage2_add_isp_prior_fusion":
                scheduler.restore_groups_to_base(["isp", "prior_fusion"])
            elif stage_name == "stage3_add_efficientvit_late":
                scheduler.restore_groups_to_base(["efficientvit_late"])
            early_stop_counter = 0
            scheduler.reset_plateau(current_psnr)
            previous_stage = stage_name

        scheduler.begin_epoch(epoch_number)
        route_factor = current_route_factor(epoch_number, args.route_decay_epochs)
        physics_weight = current_physics_weight(
            epoch_number,
            args.stage2_epoch,
            args.physics_ramp_epochs,
        )
        criterion.physics_weight = physics_weight
        model.set_route_temperature(1.0)

        print(
            f"\n🚀 Epoch [{epoch_number}/{task_cfg['max_epochs']}] | "
            f"stage={stage_name} | route={route_factor:.3f} | "
            f"physics={physics_weight:.6f}"
        )
        print(
            "🧠 LR: "
            + " | ".join(
                f"{group.get('name', 'group')}={group['lr']:.2e}"
                for group in optimizer.param_groups
            )
        )

        model.train()
        freeze_batch_norm(model)
        running_loss = 0.0
        running_count = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"{task_cfg['display_name']} Epoch {epoch_number}/{task_cfg['max_epochs']}",
            dynamic_ncols=True,
        )

        for hazy, gt, _ in train_pbar:
            hazy = hazy.to(device, non_blocking=True).float().contiguous()
            gt = gt.to(device, non_blocking=True).float().contiguous()
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(hazy)
                    loss, terms = criterion(
                        outputs,
                        hazy=hazy,
                        clean=gt,
                        model=model,
                        route_regularization_factor=route_factor,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(hazy)
                loss, terms = criterion(
                    outputs,
                    hazy=hazy,
                    clean=gt,
                    model=model,
                    route_regularization_factor=route_factor,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()

            running_loss += float(loss.item())
            running_count += 1
            term_total = terms.get("total", loss)
            if torch.is_tensor(term_total):
                term_total = float(term_total.detach().item())
            train_pbar.set_postfix(
                loss=f"{float(term_total):.4f}",
                bestP=f"{best_psnr:.3f}",
                early=f"{early_stop_counter}/{args.patience}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        last_epoch = epoch_number
        avg_train_loss = running_loss / max(running_count, 1)
        should_eval = (
            epoch_number % int(task_cfg["eval_every_epochs"]) == 0
            or epoch_number == int(task_cfg["max_epochs"])
        )

        if not should_eval:
            checkpoint = build_checkpoint(
                task_name=task_name,
                epoch=last_epoch,
                stage_name=stage_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_psnr=best_psnr,
                best_ssim=best_ssim,
                current_psnr=current_psnr,
                current_ssim=current_ssim,
                early_stop_counter=early_stop_counter,
                route_factor=route_factor,
                physics_weight=physics_weight,
                base_ckpt=base_ckpt,
                split_root=split_root,
                task_cfg=task_cfg,
                args=args,
                is_base_baseline=False,
                training_finished=False,
            )
            torch.save(checkpoint, latest_path)
            continue

        current_psnr, current_ssim = validate(
            model=model,
            loader=val_loader,
            device=device,
            tile=args.val_tile,
            stride=args.val_stride,
            max_images=None,
            amp_enabled=args.amp,
        )

        psnr_gain = current_psnr - best_psnr
        strong_psnr_improved = psnr_gain >= args.min_delta
        near_psnr_and_ssim_improved = (
            psnr_gain >= 0.0
            and psnr_gain < args.min_delta
            and current_ssim >= best_ssim + args.ssim_min_delta
        )
        improved = strong_psnr_improved or near_psnr_and_ssim_improved

        if improved:
            best_psnr = current_psnr
            best_ssim = current_ssim
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        lr_reduced = scheduler.step(current_psnr, epoch_number)
        if lr_reduced:
            print("📉 验证PSNR持续停滞，所有参数组学习率按比例下降。")

        checkpoint = build_checkpoint(
            task_name=task_name,
            epoch=last_epoch,
            stage_name=stage_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_psnr=best_psnr,
            best_ssim=best_ssim,
            current_psnr=current_psnr,
            current_ssim=current_ssim,
            early_stop_counter=early_stop_counter,
            route_factor=route_factor,
            physics_weight=physics_weight,
            base_ckpt=base_ckpt,
            split_root=split_root,
            task_cfg=task_cfg,
            args=args,
            is_base_baseline=False,
            training_finished=False,
        )
        torch.save(checkpoint, latest_path)
        if improved:
            save_best(checkpoint, model, best_path, best_model_only_path)
            print(
                f"\n🏆 新Best：epoch={last_epoch}, "
                f"PSNR={best_psnr:.4f}, SSIM={best_ssim:.6f}"
            )

        lr_values = optimizer_lrs(optimizer)
        append_history(
            history_path,
            {
                "epoch": last_epoch,
                "stage": stage_name,
                "train_loss": avg_train_loss,
                "val_psnr": current_psnr,
                "val_ssim": current_ssim,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "early_stop_counter": early_stop_counter,
                "plateau_bad_epochs": scheduler.bad_epochs,
                "lr_reductions": scheduler.reductions,
                "route_factor": route_factor,
                "physics_weight": physics_weight,
                "is_best": int(improved),
                "is_base_baseline": 0,
                **{f"lr_{key}": value for key, value in lr_values.items()},
            },
        )

        print(
            f"\n[Eval:{task_cfg['display_name']}] epoch={last_epoch} | "
            f"PSNR={current_psnr:.4f} | SSIM={current_ssim:.6f} | "
            f"best={best_psnr:.4f}/{best_ssim:.6f} | "
            f"early={early_stop_counter}/{args.patience} | "
            f"lr_bad={scheduler.bad_epochs}/{args.lr_patience}"
        )
        clear_cuda()

        if early_stop_counter >= args.patience:
            stop_reason = "early_stop"
            break

    final_checkpoint = build_checkpoint(
        task_name=task_name,
        epoch=last_epoch,
        stage_name=stage_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
        current_psnr=current_psnr,
        current_ssim=current_ssim,
        early_stop_counter=early_stop_counter,
        route_factor=route_factor,
        physics_weight=physics_weight,
        base_ckpt=base_ckpt,
        split_root=split_root,
        task_cfg=task_cfg,
        args=args,
        is_base_baseline=False,
        training_finished=True,
    )
    final_checkpoint["stop_reason"] = stop_reason
    torch.save(final_checkpoint, latest_path)

    print("\n" + "#" * 100)
    print(f"✅ {task_cfg['display_name']} 专项微调完成")
    print(f"   stop reason : {stop_reason}")
    print(f"   final epoch : {last_epoch}")
    print(f"   best        : {best_psnr:.4f} / {best_ssim:.6f}")
    print(f"   best ckpt   : {best_path}")
    print(f"   model only  : {best_model_only_path}")
    print(f"   latest      : {latest_path}")
    print("#" * 100)

    del model, optimizer, scheduler, scaler, criterion, train_loader, val_loader
    clear_cuda()
    return "completed"


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MSDSPDD Complete V2 五数据集独立专项微调（epoch最终版）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + list(TASK_ORDER),
    )
    parser.add_argument("--base_ckpt", type=str, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--split_root", type=str, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--old_step_output_root",
        type=str,
        default=DEFAULT_OLD_STEP_OUTPUT_ROOT,
        help="旧 step 版专项微调输出目录；仅在 --purge_old_step_outputs 时清理所选任务的旧权重。",
    )
    parser.add_argument(
        "--keep_old_step_outputs",
        action="store_true",
        default=True,
        help="保留旧 step 版权重（发布版默认行为）。",
    )
    parser.add_argument(
        "--purge_old_step_outputs", action="store_false",
        dest="keep_old_step_outputs",
        help="显式清理 --old_step_output_root 中所选任务的旧权重。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4 if sys.platform != "win32" else 0)

    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--eval_every_epochs", type=int, default=0)

    # 完全按epoch控制。
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.02)
    parser.add_argument("--ssim_min_delta", type=float, default=1e-4)

    # 三阶段解冻：1-10 / 11-20 / 21+。
    parser.add_argument("--stage2_epoch", type=int, default=10)
    parser.add_argument("--stage3_epoch", type=int, default=20)

    # epoch级warmup + Plateau。
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_threshold", type=float, default=0.01)
    parser.add_argument("--min_lr", type=float, default=1e-7)

    parser.add_argument("--route_decay_epochs", type=int, default=40)
    parser.add_argument("--physics_ramp_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_scale", type=float, default=1.0)

    parser.add_argument("--val_tile", type=int, default=512)
    parser.add_argument("--val_stride", type=int, default=256)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no_strict", action="store_false", dest="strict")
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--no_skip_finished", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_ckpt = Path(args.base_ckpt)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    old_step_output_root = Path(args.old_step_output_root)

    if not base_ckpt.is_file():
        raise FileNotFoundError(f"新模型通用权重不存在: {base_ckpt}")
    if not split_root.is_dir():
        raise FileNotFoundError(f"复用划分根目录不存在: {split_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = list(TASK_ORDER) if args.dataset == "all" else [args.dataset]

    print("\n" + "=" * 100)
    print("MSDSPDD Complete V2 五数据集独立专项微调（epoch最终版）")
    print(f"project root : {PROJECT_ROOT}")
    print(f"base ckpt    : {base_ckpt}")
    print(f"split root   : {split_root}")
    print(f"output root  : {output_root}")
    print(f"tasks        : {selected}")
    print(f"max epochs   : 各任务默认10000（可用 --max_epochs 覆盖）")
    print(f"early stop   : 连续 {args.patience} 个epoch无有效提升")
    print(f"stages       : 1-{args.stage2_epoch} / {args.stage2_epoch + 1}-{args.stage3_epoch} / {args.stage3_epoch + 1}+")
    print(f"LR schedule  : epoch warmup + plateau")
    print(f"AMP          : {args.amp}")
    print(f"old step root: {old_step_output_root}")
    print(f"purge old    : {not args.keep_old_step_outputs}")
    print("=" * 100)

    # 先检查全部数据划分；任何一项失败都不会开始训练或删除旧权重。
    split_cache = preflight_selected_splits(
        selected_tasks=selected,
        split_root=split_root,
        output_root=output_root,
    )

    if args.check_only:
        print("\n✅ --check_only 完成：只检查了数据划分，未删除旧 step 权重，也未开始训练。")
        return

    # 数据检查通过后，按用户要求自动清理旧 step 版专项权重。
    if args.keep_old_step_outputs:
        print("\n保留旧 step 版权重：已指定 --keep_old_step_outputs。")
    else:
        purge_old_step_outputs(
            selected_tasks=selected,
            old_step_output_root=old_step_output_root,
        )

    completed: List[str] = []
    failed: List[str] = []

    for task_name in selected:
        try:
            status = train_one_task(
                task_name=task_name,
                base_ckpt=base_ckpt,
                split_root=split_root,
                output_root=output_root,
                args=args,
                prechecked_splits=split_cache[task_name],
            )
            completed.append(f"{task_name}:{status}")
        except Exception as exc:
            failed.append(task_name)
            print("\n" + "!" * 100)
            print(f"❌ {task_name} 失败: {exc!r}")
            traceback.print_exc()
            print("!" * 100)
            clear_cuda()

    print("\n" + "=" * 100)
    print("全部任务结束")
    print("completed:", completed)
    print("failed   :", failed)
    print("output   :", output_root)
    print("=" * 100)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()