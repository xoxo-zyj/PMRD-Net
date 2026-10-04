# -*- coding: utf-8 -*-
"""
MSDSPDD Complete V2 - General Training
======================================

This script is adapted from the user's existing crop256 training code.

Training stages:
    Stage 1: warm up newly added reconstruction modules while freezing
             PriorEngine, ISP and the shared EfficientViT backbone.
    Stage 2: jointly fine-tune ISP, the last two EfficientViT stages and
             the learnable prior fusion layer with separate learning rates.

Expected project layout:
    MSDSPDD_Final_Full/
    ├── train_msdspdd_final_full.py
    ├── future_process/
    │   ├── model.py
    │   ├── losses.py
    │   ├── efficientvit/
    │   ├── vmamba.py
    │   └── efficientvit_b1_r256.pt
    ├── generate_prior/
    └── isp/

The model keeps the legacy seven-output forward interface, but the complete
loss also reads the model's auxiliary tensors:
    refined t1, local A1, route weights, Raw Detail Gate, PG-DSP maps,
    physics error and Residual Refiner outputs.
"""

import os
import sys
import csv
import random
import warnings
import math
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None

from skimage.metrics import structural_similarity as calculate_ssim

try:
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("⚠️ 未检测到 piq，NIQE/BRISQUE 将显示为 999。")
    print("   如需启用，请执行: pip install piq")

# Project root is resolved from this script, so the script can be launched
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Split-package model actually present in MSDSPDD_Final_Full:
# generate_prior -> ISP -> five routes -> reconstruction -> refinement.
from future_process.model import MSDSPDDDehazer
from future_process.losses import MSDSPDDCompositeLoss

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


TRAIN_CONFIG = {
    # Five images share one EfficientViT, but activations are approximately
    # five-route scale. Start from 8; on a 96 GB GPU you may try 12.
    "batch_size": 24,
    "num_workers": 8 if sys.platform != "win32" else 0,
    "max_epochs": 10000,
    "patience": 20,

    # Stage 1: only new modules are trained.
    "warmup_new_modules_epochs": 10,

    # Separate learning rates used from the beginning. Frozen groups do not
    # update until they are unfrozen in Stage 2.
    "lr_new_modules": 1e-4,
    "lr_residual_refiner": 5e-5,
    "lr_isp": 2e-5,
    "lr_efficientvit_late": 1e-5,
    "lr_prior_fusion": 3e-6,

    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,

    # Keep False until the current PyTorch/CUDA build is confirmed to support
    # the installed GPU architecture.
    "amp_enabled": False,

    # Route temperature: smooth early routing -> clearer later routing.
    "route_temperature_start": 1.5,
    "route_temperature_end": 1.0,
    "route_temperature_epochs": 40,

    # Route survival regularization is only active early.
    "route_regularization_epochs": 40,

    # Physics loss is disabled in warmup and gradually enabled in Stage 2.
    "physics_ramp_epochs": 10,
    "physics_target_weight": 0.005,

    "save_freq": 0,
    "cache_clear_freq": 100,
    "no_reference_eval_every": 10,
}


DATA_CONFIG = {
    "crop_size": 256,
    "val_crop_seed": 2026,

    "train_hazy_root": r"/root/autodl-tmp/train/train",
    "val_hazy_root": r"/root/autodl-tmp/train/val",
    "gt_root": r"/root/autodl-tmp/remade_clear",

    "v1_pretrained_ckpt": (
        r"/root/autodl-tmp/U-net/FDE_model/"
        r"checkpoints_crop256_finetune/dehaze_best.pth"
    ),

    "efficientvit_weight": str(
        PROJECT_ROOT / "future_process" / "efficientvit_b1_r256.pt"
    ),

    "write_pair_reports": True,

    "pair_report_dir": str(
        PROJECT_ROOT
        / "checkpoints_msdspdd_v2_general"
        / "pair_reports"
    ),
}


LOSS_CONFIG = {
    # Final RGB output is the main target.
    "final_recon_weight": 1.00,

    # J_original must also be supervised so the Residual Refiner cannot carry
    # all reconstruction responsibility.
    "base_recon_weight": 0.30,

    "ssim_weight": 0.20,
    "perceptual_weight": 0.03,
    "edge_weight": 0.05,
    "color_weight": 0.02,

    # This value is ramped from 0 to physics_target_weight after warmup.
    "physics_weight": 0.005,

    # PG-DSP controls where residual correction is safe.
    "pgdsp_safety_weight": 0.005,

    # Keep final correction low-amplitude.
    "residual_weight": 0.001,

    # Prevent route death early; it is gradually removed.
    "route_weight": 0.001,
    "minimum_route_usage": 0.03,

    "best_score_ssim_weight": 10.0,
}


try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("⚠️ 未检测到 torchmetrics，将使用 skimage 计算 SSIM。")


def get_gpu_memory() -> float:
    if torch.cuda.is_available() and GPUtil is not None:
        gpus = GPUtil.getGPUs()
        return gpus[0].memoryUsed if gpus else 0.0
    return 0.0

# ==========================================
# 📊 2. 数据集加载器：只改这里，适配当前 8w source 子文件夹数据
# ==========================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_img(path: Path):
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def list_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    return sorted([p for p in root.rglob("*") if is_img(p)])


class SimpleDehazeDataset(Dataset):
    """
    crop256 微调版数据集：
    读取原图成对 hazy / GT。
    训练阶段：random crop 256 + hflip。
    验证阶段：fixed random crop 256，不 hflip，保证每轮验证位置固定。

    支持：
        ITS_v2:
            hazy: 1_1_0.90179.png      -> gt: 1.png

        OTS_BETA:
            hazy: 0025_0.8_0.04.png    -> gt: 0025.png

        RW2AH:
            hazy: 1-0001-03.png        -> gt: 1-0001-03.png

        NTIRE19:
            hazy: 01_hazy.png          -> gt: 01_GT.png
    """

    def __init__(self, hazy_root, gt_root, is_train=True, split_name="train"):
        self.hazy_root = Path(hazy_root)
        self.gt_root = Path(gt_root)
        self.is_train = is_train
        self.split_name = split_name

        self.hazy_files = list_images(self.hazy_root)

        self.gt_by_source_stem, self.gt_by_source_name = self._build_gt_index()
        self.pairs, self.unmatched_rows, self.ambiguous_rows = self._build_pairs()

        print(f"\n[{self.split_name}] 数据集加载完成")
        print(f"  hazy_root         : {self.hazy_root}")
        print(f"  gt_root           : {self.gt_root}")
        print(f"  hazy 文件数        : {len(self.hazy_files)}")
        print(f"  valid pairs       : {len(self.pairs)}")
        print(f"  unmatched         : {len(self.unmatched_rows)}")
        print(f"  ambiguous skipped : {len(self.ambiguous_rows)}")
        print(f"  is_train          : {self.is_train}")
        if self.is_train:
            print(f"  train transform   : random crop {DATA_CONFIG['crop_size']} x {DATA_CONFIG['crop_size']} + hflip")
        else:
            print(f"  val transform     : fixed random crop {DATA_CONFIG['crop_size']} x {DATA_CONFIG['crop_size']}")

        if DATA_CONFIG.get("write_pair_reports", False):
            self._write_pair_reports(Path(DATA_CONFIG["pair_report_dir"]))

        if len(self.pairs) == 0:
            raise RuntimeError(f"[{self.split_name}] 没有有效配对，请检查路径和文件名规则。")

    def _build_gt_index(self):
        gt_files = list_images(self.gt_root)

        by_stem = defaultdict(lambda: defaultdict(list))
        by_name = defaultdict(dict)

        for gp in gt_files:
            rel = gp.relative_to(self.gt_root)
            source = rel.parts[0] if len(rel.parts) >= 2 else "__root__"

            by_stem[source][gp.stem.lower()].append(gp)
            by_name[source].setdefault(gp.name.lower(), gp)

        return by_stem, by_name

    def _source_of_hazy(self, hp: Path):
        rel = hp.relative_to(self.hazy_root)
        source = rel.parts[0] if len(rel.parts) >= 2 else "__root__"
        return source, rel

    def _candidate_stems(self, source: str, hazy_stem: str):
        cands = []

        # 1. 原 stem
        cands.append(hazy_stem)

        # 2. NTIRE19 特殊规则：01_hazy -> 01_GT
        if source.upper() == "NTIRE19":
            if hazy_stem.lower().endswith("_hazy"):
                base = hazy_stem[:-len("_hazy")]
                cands.append(base + "_GT")
                cands.append(base)

        # 3. 通用多雾图规则：
        # ITS_v2: 1_1_0.90179 -> 1
        # OTS_BETA: 0025_0.8_0.04 -> 0025
        if "_" in hazy_stem:
            cands.append(hazy_stem.split("_")[0])

        # 去重，保持优先级
        out = []
        seen = set()

        for c in cands:
            key = c.lower().strip()
            if key and key not in seen:
                out.append(c)
                seen.add(key)

        return out

    def _find_gt(self, hp: Path):
        source, _ = self._source_of_hazy(hp)

        # 1. 完全同名
        name_key = hp.name.lower()
        if source in self.gt_by_source_name and name_key in self.gt_by_source_name[source]:
            return self.gt_by_source_name[source][name_key], source, "exact_filename"

        # 2. 按 stem 规则匹配
        for stem in self._candidate_stems(source, hp.stem):
            hits = self.gt_by_source_stem[source].get(stem.lower(), [])

            if len(hits) == 1:
                if stem == hp.stem:
                    return hits[0], source, "same_stem"

                if source.upper() == "NTIRE19" and stem.lower().endswith("_gt"):
                    return hits[0], source, "ntire_hazy_to_gt"

                return hits[0], source, "first_underscore_stem"

            if len(hits) > 1:
                return None, source, f"ambiguous_gt_stem:{stem}"

        return None, source, "unmatched"

    def _build_pairs(self):
        pairs = []
        unmatched = []
        ambiguous = []

        for hp in self.hazy_files:
            gp, source, match_type = self._find_gt(hp)

            row = {
                "split": self.split_name,
                "source": source,
                "hazy_path": str(hp),
                "hazy_name": hp.name,
                "gt_path": "" if gp is None else str(gp),
                "gt_name": "" if gp is None else gp.name,
                "match_type": match_type,
            }

            if gp is None:
                if match_type.startswith("ambiguous"):
                    ambiguous.append(row)
                else:
                    unmatched.append(row)
                continue

            pairs.append((hp, gp, source, match_type))

        return pairs, unmatched, ambiguous

    def _write_pair_reports(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "split",
            "source",
            "hazy_path",
            "hazy_name",
            "gt_path",
            "gt_name",
            "match_type",
        ]

        pair_rows = []
        for hp, gp, source, match_type in self.pairs:
            pair_rows.append({
                "split": self.split_name,
                "source": source,
                "hazy_path": str(hp),
                "hazy_name": hp.name,
                "gt_path": str(gp),
                "gt_name": gp.name,
                "match_type": match_type,
            })

        def write_csv(name, rows):
            path = out_dir / name
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        write_csv(f"{self.split_name}_pairs.csv", pair_rows)
        write_csv(f"{self.split_name}_unmatched.csv", self.unmatched_rows)
        write_csv(f"{self.split_name}_ambiguous.csv", self.ambiguous_rows)

        print(f"[{self.split_name}] 配对报告已保存: {out_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path, source, match_type = self.pairs[idx]

        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        # 正常成对数据应尺寸一致；如果不一致，先把 GT 对齐到 hazy 尺寸。
        # PIL.size 是 (W, H)，TF.resize 需要 (H, W)。
        if hazy_img.size != gt_img.size:
            gt_img = TF.resize(
                gt_img,
                hazy_img.size[::-1],
                interpolation=TF.InterpolationMode.BICUBIC
            )

        crop_size = DATA_CONFIG["crop_size"]
        w, h = hazy_img.size

        # 若图像小于 crop_size，先等比例放大到可以裁 crop_size。
        # 这里只处理极小图；正常大图不会触发。
        if w < crop_size or h < crop_size:
            scale = max(crop_size / w, crop_size / h)
            new_w = int(math.ceil(w * scale))
            new_h = int(math.ceil(h * scale))

            hazy_img = TF.resize(
                hazy_img,
                (new_h, new_w),
                interpolation=TF.InterpolationMode.BICUBIC
            )
            gt_img = TF.resize(
                gt_img,
                (new_h, new_w),
                interpolation=TF.InterpolationMode.BICUBIC
            )

            w, h = hazy_img.size

        if self.is_train:
            # ✅ 训练：每次随机 crop256，hazy 和 GT 必须同位置裁剪
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)

            hazy_img = TF.crop(hazy_img, top, left, crop_size, crop_size)
            gt_img = TF.crop(gt_img, top, left, crop_size, crop_size)

            # ✅ 训练：随机水平翻转，hazy 和 GT 必须一起翻转
            if random.random() < 0.5:
                hazy_img = TF.hflip(hazy_img)
                gt_img = TF.hflip(gt_img)

        else:
            # ✅ 验证：fixed random crop256
            # 每张验证图固定一个随机 crop 位置，每一轮验证都一致。
            rng = random.Random(DATA_CONFIG["val_crop_seed"] + idx)

            top = rng.randint(0, h - crop_size)
            left = rng.randint(0, w - crop_size)

            hazy_img = TF.crop(hazy_img, top, left, crop_size, crop_size)
            gt_img = TF.crop(gt_img, top, left, crop_size, crop_size)

        hazy = TF.to_tensor(hazy_img)
        gt = TF.to_tensor(gt_img)

        return hazy, gt


# ==========================================
# 📈 3. 指标计算
# ==========================================
class AcademicMetrics:
    def __init__(self, device):
        self.device = device

        if HAS_TORCHMETRICS:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def calculate(self, pred, gt, shave=4):
        pred = torch.clamp(pred, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        if shave > 0:
            pred = pred[..., shave:-shave, shave:-shave]
            gt = gt[..., shave:-shave, shave:-shave]

        mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        if HAS_TORCHMETRICS:
            ssim = self.ssim_metric(pred, gt)
        else:
            pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)

            ssim = torch.tensor(
                [
                    calculate_ssim(
                        p,
                        g,
                        channel_axis=2,
                        data_range=1.0,
                        gaussian_weights=True,
                        sigma=1.5
                    )
                    for p, g in zip(pred_np, gt_np)
                ]
            )

        return psnr.mean().item(), ssim.mean().item()

    def calculate_no_reference(self, pred):
        """
        NIQE / BRISQUE:
        越低越好。

        注意：
        1. 这里只做验证监控，不参与 loss。
        2. 如果 piq 没装，或者某张图计算失败，返回 999，避免训练中断。
        """
        if not HAS_PIQ:
            return 999.0, 999.0

        pred = torch.clamp(pred, 0.0, 1.0)

        niqe_value = 999.0
        brisque_value = 999.0

        try:
            if hasattr(piq, "niqe"):
                niqe_value = piq.niqe(
                    pred,
                    data_range=1.0,
                    reduction="mean"
                ).detach().item()
        except Exception as e:
            print("[NIQE ERROR]", repr(e))
            niqe_value = 999.0

        try:
            if hasattr(piq, "brisque"):
                brisque_value = piq.brisque(
                    pred,
                    data_range=1.0,
                    reduction="mean"
                ).detach().item()
        except Exception as e:
            brisque_value = 999.0

        return niqe_value, brisque_value



# ==========================================
# 📉 7. 余弦退火重启调度器
# ==========================================
class CosineAnnealingWithRestartsDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        gamma=0.8,
        eta_min=1e-7,
        warmup_epochs=5,
        last_epoch=-1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.gamma = gamma
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            return [
                base_lr * (epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]

        epoch = epoch - self.warmup_epochs
        n = 0
        current_restart_epoch = 0
        T_i = self.T_0

        while epoch >= current_restart_epoch + T_i:
            current_restart_epoch += T_i
            T_i = int(T_i * self.T_mult)
            n += 1

        T_cur = epoch - current_restart_epoch

        peak_lrs = [
            base_lr * (self.gamma ** n)
            for base_lr in self.base_lrs
        ]

        return [
            self.eta_min
            + (peak_lr - self.eta_min)
            * (1 + math.cos(math.pi * T_cur / T_i))
            / 2
            for peak_lr in peak_lrs
        ]




# =============================================================================
# Parameter grouping and staged training
# =============================================================================

def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def unique_trainable_parameters(
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
    """
    EfficientViT mapping in the complete model:
        stem       -> H/2
        stages[0]  -> H/4
        stages[1]  -> H/8
        stages[2]  -> H/16

    Only H/8 and H/16 are jointly fine-tuned.
    """
    backbone = model.extractor.backbone
    modules: List[nn.Module] = []

    if hasattr(backbone, "stages"):
        stages = list(backbone.stages)
        if len(stages) >= 3:
            modules.extend(stages[1:3])
        elif len(stages) >= 2:
            modules.extend(stages[-2:])
    else:
        # Formal training must not use fallback. This branch is kept defensive.
        if hasattr(backbone, "stage2"):
            modules.append(backbone.stage2)
        if hasattr(backbone, "stage3"):
            modules.append(backbone.stage3)

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


def build_optimizer(model: MSDSPDDDehazer) -> torch.optim.Optimizer:
    used_ids: set = set()

    new_params = unique_trainable_parameters(
        new_reconstruction_modules(model),
        used_ids,
    )
    refiner_params = unique_trainable_parameters(
        [model.residual_refiner],
        used_ids,
    )
    isp_params = unique_trainable_parameters(
        [model.isp_module],
        used_ids,
    )
    backbone_late_params = unique_trainable_parameters(
        late_backbone_modules(model),
        used_ids,
    )

    prior_fusion_params: List[nn.Parameter] = []
    if hasattr(model, "prior_engine") and hasattr(model.prior_engine, "fusion_conv"):
        prior_fusion_params = unique_trainable_parameters(
            [model.prior_engine.fusion_conv],
            used_ids,
        )

    groups = [
        {
            "name": "new_modules",
            "params": new_params,
            "lr": TRAIN_CONFIG["lr_new_modules"],
        },
        {
            "name": "residual_refiner",
            "params": refiner_params,
            "lr": TRAIN_CONFIG["lr_residual_refiner"],
        },
        {
            "name": "isp",
            "params": isp_params,
            "lr": TRAIN_CONFIG["lr_isp"],
        },
        {
            "name": "efficientvit_late",
            "params": backbone_late_params,
            "lr": TRAIN_CONFIG["lr_efficientvit_late"],
        },
        {
            "name": "prior_fusion",
            "params": prior_fusion_params,
            "lr": TRAIN_CONFIG["lr_prior_fusion"],
        },
    ]
    groups = [group for group in groups if len(group["params"]) > 0]

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    return optimizer


def configure_training_stage(
    model: MSDSPDDDehazer,
    epoch: int,
    verbose: bool = True,
) -> str:
    warmup_epochs = TRAIN_CONFIG["warmup_new_modules_epochs"]

    # New modules are always trainable.
    for module in new_reconstruction_modules(model):
        set_module_trainable(module, True)
    set_module_trainable(model.residual_refiner, True)

    # Traditional PG-DSP is an analytic controller and is not separately
    # optimized. Its outputs still guide the differentiable refiner loss.
    set_module_trainable(model.pg_dsp_module, False)

    if epoch < warmup_epochs:
        stage_name = "stage1_new_modules_warmup"

        if hasattr(model, "prior_engine"):
            set_module_trainable(model.prior_engine, False)
        set_module_trainable(model.isp_module, False)
        set_module_trainable(model.extractor.backbone, False)

    else:
        stage_name = "stage2_joint_finetune"

        # Prior physical operations remain fixed; only the learnable dark/bright
        # fusion convolution is unfrozen.
        if hasattr(model, "prior_engine"):
            set_module_trainable(model.prior_engine, False)
            if hasattr(model.prior_engine, "fusion_conv"):
                set_module_trainable(model.prior_engine.fusion_conv, True)

        # ISP refinement and operator controls become trainable.
        set_module_trainable(model.isp_module, True)

        # Freeze the whole shared backbone first, then unfreeze only H/8/H/16.
        set_module_trainable(model.extractor.backbone, False)
        for module in late_backbone_modules(model):
            set_module_trainable(module, True)

    if verbose:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"🔧 训练阶段: {stage_name}")
        print(f"   可训练参数: {trainable / 1e6:.3f} M / {total / 1e6:.3f} M")

    return stage_name


def route_regularization_factor(epoch: int) -> float:
    total = max(int(TRAIN_CONFIG["route_regularization_epochs"]), 1)
    return max(0.0, 1.0 - epoch / total)


def current_route_temperature(epoch: int) -> float:
    start = float(TRAIN_CONFIG["route_temperature_start"])
    end = float(TRAIN_CONFIG["route_temperature_end"])
    total = max(int(TRAIN_CONFIG["route_temperature_epochs"]), 1)
    progress = min(max(epoch / total, 0.0), 1.0)
    return start + (end - start) * progress


def current_physics_weight(epoch: int) -> float:
    warmup = int(TRAIN_CONFIG["warmup_new_modules_epochs"])
    if epoch < warmup:
        return 0.0

    ramp = max(int(TRAIN_CONFIG["physics_ramp_epochs"]), 1)
    progress = min(max((epoch - warmup + 1) / ramp, 0.0), 1.0)
    return float(TRAIN_CONFIG["physics_target_weight"]) * progress


def print_optimizer_lrs(optimizer: torch.optim.Optimizer) -> None:
    parts = []
    for group in optimizer.param_groups:
        parts.append(f"{group.get('name', 'group')}={group['lr']:.2e}")
    print("🧠 LR: " + " | ".join(parts))


def freeze_batch_norm(model: nn.Module) -> None:
    # The complete model already exposes this helper.
    if hasattr(model, "freeze_batch_norm"):
        model.freeze_batch_norm(freeze_affine=False)
        return

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


def scalar_terms(terms: Dict[str, torch.Tensor]) -> Dict[str, float]:
    result = {}
    for key, value in terms.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().item())
        else:
            result[key] = float(value)
    return result



def validate_environment_paths() -> None:
    """Fail early when the split-package project or data paths are incomplete."""
    required_files = {
        "model.py": PROJECT_ROOT / "future_process" / "model.py",
        "losses.py": PROJECT_ROOT / "future_process" / "losses.py",
        "EfficientViT weight": Path(DATA_CONFIG["efficientvit_weight"]),
    }
    required_dirs = {
        "future_process": PROJECT_ROOT / "future_process",
        "generate_prior": PROJECT_ROOT / "generate_prior",
        "isp": PROJECT_ROOT / "isp",
        "train_hazy_root": Path(DATA_CONFIG["train_hazy_root"]),
        "val_hazy_root": Path(DATA_CONFIG["val_hazy_root"]),
        "gt_root": Path(DATA_CONFIG["gt_root"]),
    }

    missing = []
    for label, path in required_files.items():
        if not path.is_file():
            missing.append(f"{label}: {path}")
    for label, path in required_dirs.items():
        if not path.is_dir():
            missing.append(f"{label}: {path}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "训练前检查失败，以下文件或目录不存在：\n" + details
        )

    v1_path = Path(DATA_CONFIG.get("v1_pretrained_ckpt", ""))
    if not v1_path.is_file():
        print(
            "⚠️ 未找到旧版V1最佳权重，将只加载EfficientViT预训练权重：\n"
            f"   {v1_path}"
        )

    print("✅ 工程、数据和EfficientViT权重路径检查通过")
    print(f"   工程根目录: {PROJECT_ROOT}")
    print(f"   训练集: {DATA_CONFIG['train_hazy_root']}")
    print(f"   验证集: {DATA_CONFIG['val_hazy_root']}")
    print(f"   GT目录: {DATA_CONFIG['gt_root']}")
    print(f"   输出目录: {PROJECT_ROOT / 'checkpoints_msdspdd_v2_general'}")


# =============================================================================
# Main training
# =============================================================================

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device} | 初始显存: {get_gpu_memory():.2f} MB")

    validate_environment_paths()

    model = MSDSPDDDehazer(
        enable_end_to_end=True,
        weight_path=DATA_CONFIG["efficientvit_weight"],
        route_temperature=TRAIN_CONFIG["route_temperature_start"],
        use_mamba=True,
        enable_refiner=True,
        allow_backbone_fallback=False,
    ).to(device)

    # Formal training must use the real modules.
    if model.backbone_is_fallback:
        raise RuntimeError(
            "当前正在使用备用卷积主干，禁止正式训练。请修复 EfficientViT 依赖。"
        )
    if not model.mamba_enabled:
        raise RuntimeError(
            "VMamba/VSSBlock 未启用，禁止正式训练。请先修复 VMamba 环境。"
        )

    # Configure Stage 1 before optimizer creation. Frozen parameters are still
    # placed in optimizer groups and start updating after Stage 2 unfreezes them.
    configure_training_stage(model, epoch=0, verbose=True)
    optimizer = build_optimizer(model)

    scheduler = CosineAnnealingWithRestartsDecay(
        optimizer,
        T_0=50,
        T_mult=2,
        gamma=0.7,
        eta_min=1e-7,
        warmup_epochs=5,
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if TRAIN_CONFIG["amp_enabled"]
        else None
    )

    criterion = MSDSPDDCompositeLoss(
        final_recon_weight=LOSS_CONFIG["final_recon_weight"],
        base_recon_weight=LOSS_CONFIG["base_recon_weight"],
        ssim_weight=LOSS_CONFIG["ssim_weight"],
        perceptual_weight=LOSS_CONFIG["perceptual_weight"],
        edge_weight=LOSS_CONFIG["edge_weight"],
        color_weight=LOSS_CONFIG["color_weight"],
        physics_weight=0.0,  # ramped every epoch
        pgdsp_safety_weight=LOSS_CONFIG["pgdsp_safety_weight"],
        residual_weight=LOSS_CONFIG["residual_weight"],
        route_weight=LOSS_CONFIG["route_weight"],
        minimum_route_usage=LOSS_CONFIG["minimum_route_usage"],
    ).to(device)

    metrics_engine = AcademicMetrics(device)

    train_dataset = SimpleDehazeDataset(
        DATA_CONFIG["train_hazy_root"],
        DATA_CONFIG["gt_root"],
        is_train=True,
        split_name="train",
    )
    val_dataset = SimpleDehazeDataset(
        DATA_CONFIG["val_hazy_root"],
        DATA_CONFIG["gt_root"],
        is_train=False,
        split_name="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=TRAIN_CONFIG["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("\n🔍 验证 GT 值域:")
    _, gt_test = next(iter(train_loader))
    print(
        f"   GT最大值: {gt_test.max().item():.4f} | "
        f"GT最小值: {gt_test.min().item():.4f}"
    )
    if gt_test.max() > 1.1:
        raise RuntimeError("GT未归一化到0~1。")
    if gt_test.max() < 0.9:
        print("⚠️ GT最大值偏低，请检查数据。")

    checkpoint_dir = PROJECT_ROOT / "checkpoints_msdspdd_v2_general"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / "general_latest.pth"
    best_path = checkpoint_dir / "general_best.pth"

    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_score = 0.0
    best_niqe = 999.0
    best_brisque = 999.0
    lowest_niqe = 999.0
    lowest_brisque = 999.0
    early_stop_counter = 0

    if resume_path.exists():
        print(f"📂 恢复V2完整断点: {resume_path}")
        # This is a trusted checkpoint produced by this training script.
        # PyTorch 2.6 otherwise defaults to weights_only=True and may reject
        # optimizer/scheduler state or NumPy scalar metadata.
        try:
            checkpoint = torch.load(
                resume_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", 0.0))
        best_ssim = float(checkpoint.get("best_ssim", 0.0))
        best_score = float(checkpoint.get("best_score", 0.0))
        best_niqe = float(checkpoint.get("best_niqe", 999.0))
        best_brisque = float(checkpoint.get("best_brisque", 999.0))
        lowest_niqe = float(checkpoint.get("lowest_niqe", 999.0))
        lowest_brisque = float(checkpoint.get("lowest_brisque", 999.0))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))

        print(f"✅ 从第 {start_epoch + 1} 轮继续")

    else:
        v1_path = DATA_CONFIG.get("v1_pretrained_ckpt", "")
        if v1_path and os.path.exists(v1_path):
            print(f"🔥 使用V1 best进行形状安全迁移: {v1_path}")
            report = model.load_v1_checkpoint(v1_path, map_location="cpu")
            print(f"   成功加载: {len(report['loaded'])}")
            print(f"   跳过旧结构: {len(report['skipped'])}")
            print(f"   V2新参数: {len(report['missing'])}")
        else:
            print("⚠️ 未找到V1 best；EfficientViT仍会加载自己的预训练权重。")

    previous_stage = None

    for epoch in range(start_epoch, TRAIN_CONFIG["max_epochs"]):
        print(f"\n{'=' * 72}")
        print(f"🚀 Epoch [{epoch + 1}/{TRAIN_CONFIG['max_epochs']}]")

        stage_name = configure_training_stage(
            model,
            epoch=epoch,
            verbose=(previous_stage != (
                "stage1_new_modules_warmup"
                if epoch < TRAIN_CONFIG["warmup_new_modules_epochs"]
                else "stage2_joint_finetune"
            )),
        )
        previous_stage = stage_name

        temperature = current_route_temperature(epoch)
        model.set_route_temperature(temperature)

        route_factor = route_regularization_factor(epoch)
        criterion.physics_weight = current_physics_weight(epoch)

        print_optimizer_lrs(optimizer)
        print(
            f"🌡️ Route T={temperature:.3f} | "
            f"Route factor={route_factor:.3f} | "
            f"Physics weight={criterion.physics_weight:.6f}"
        )

        model.train()
        # Re-apply after every model.train(), especially for small-batch stages.
        freeze_batch_norm(model)

        train_loss_sum = 0.0
        running_terms: Dict[str, float] = defaultdict(float)

        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

        for batch_idx, (hazy, gt) in enumerate(pbar):
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if TRAIN_CONFIG["amp_enabled"]:
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
                    [p for p in model.parameters() if p.requires_grad],
                    TRAIN_CONFIG["grad_clip_norm"],
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
                    [p for p in model.parameters() if p.requires_grad],
                    TRAIN_CONFIG["grad_clip_norm"],
                )
                optimizer.step()

            term_values = scalar_terms(terms)
            train_loss_sum += float(loss.item())
            for key, value in term_values.items():
                running_terms[key] += value

            pbar.set_postfix(
                Total=f"{term_values['total']:.4f}",
                Final=f"{term_values['final_recon']:.4f}",
                Base=f"{term_values['base_recon']:.4f}",
                Phys=f"{term_values['physics']:.4f}",
                Route=f"{term_values['route_survival']:.5f}",
                GPU=f"{get_gpu_memory():.0f}MB",
            )

            if (
                (batch_idx + 1) % TRAIN_CONFIG["cache_clear_freq"] == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        batches = max(len(train_loader), 1)
        avg_train_loss = train_loss_sum / batches
        avg_terms = {
            key: value / batches
            for key, value in running_terms.items()
        }

        print(f"\n📈 训练平均总损失: {avg_train_loss:.6f}")
        print(
            "   "
            + " | ".join(
                f"{key}={value:.5f}"
                for key, value in avg_terms.items()
                if key in {
                    "final_recon",
                    "base_recon",
                    "ssim",
                    "perceptual",
                    "edge",
                    "color",
                    "physics",
                    "pgdsp_safety",
                    "residual",
                    "route_survival",
                }
            )
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        model.eval()
        val_psnr: List[float] = []
        val_ssim: List[float] = []
        val_niqe: List[float] = []
        val_brisque: List[float] = []

        route_usage_sum: Dict[str, torch.Tensor] = {}
        route_usage_count: Dict[str, int] = defaultdict(int)

        calculate_no_ref = (
            epoch % max(TRAIN_CONFIG["no_reference_eval_every"], 1) == 0
        )

        with torch.no_grad():
            for hazy, gt in tqdm(val_loader, desc="Evaluating", leave=False):
                try:
                    hazy = hazy.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)

                    outputs = model(hazy)
                    final_out = outputs[0]

                    if final_out.shape[-2:] != gt.shape[-2:]:
                        final_out = F.interpolate(
                            final_out,
                            size=gt.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    psnr, ssim = metrics_engine.calculate(final_out, gt)
                    val_psnr.append(psnr)
                    val_ssim.append(ssim)

                    auxiliary = model.get_auxiliary_outputs(detach=True)
                    for scale, weights in auxiliary["route_weights"].items():
                        if weights is None:
                            continue
                        usage = weights.mean(dim=(0, 2, 3)).cpu()
                        if scale not in route_usage_sum:
                            route_usage_sum[scale] = torch.zeros_like(usage)
                        route_usage_sum[scale] += usage
                        route_usage_count[scale] += 1

                    if calculate_no_ref:
                        niqe, brisque = metrics_engine.calculate_no_reference(final_out)
                        if np.isfinite(niqe):
                            val_niqe.append(niqe)
                        if np.isfinite(brisque):
                            val_brisque.append(brisque)

                except Exception as exc:
                    tqdm.write(f"⚠️ 验证异常跳过: {exc!r}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_psnr = float(np.mean(val_psnr)) if val_psnr else 0.0
        avg_ssim = float(np.mean(val_ssim)) if val_ssim else 0.0

        if calculate_no_ref:
            avg_niqe = float(np.mean(val_niqe)) if val_niqe else 999.0
            avg_brisque = (
                float(np.mean(val_brisque))
                if val_brisque
                else 999.0
            )
            lowest_niqe = min(lowest_niqe, avg_niqe)
            lowest_brisque = min(lowest_brisque, avg_brisque)
        else:
            avg_niqe = 999.0
            avg_brisque = 999.0

        current_score = (
            avg_psnr
            + LOSS_CONFIG["best_score_ssim_weight"] * avg_ssim
        )

        print("\n🧪 验证结果:")
        print(f"   PSNR : {avg_psnr:.4f} | Best: {best_psnr:.4f}")
        print(f"   SSIM : {avg_ssim:.6f} | Best: {best_ssim:.6f}")
        if calculate_no_ref:
            print(
                f"   NIQE : {avg_niqe:.4f} | "
                f"BRISQUE: {avg_brisque:.4f}"
            )
        else:
            print("   NIQE/BRISQUE: 本轮跳过，减少验证耗时")
        print(f"   Score: {current_score:.6f} | Best: {best_score:.6f}")

        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            if calculate_no_ref:
                best_niqe = avg_niqe
                best_brisque = avg_brisque
            early_stop_counter = 0
            print("🏆 新版通用模型刷新最佳记录")
        else:
            early_stop_counter += 1
            print(
                f"💬 未刷新，早停剩余 "
                f"{TRAIN_CONFIG['patience'] - early_stop_counter} 轮"
            )

        route_usage = {
            scale: (
                route_usage_sum[scale]
                / max(route_usage_count[scale], 1)
            ).tolist()
            for scale in sorted(route_usage_sum)
        }

        if route_usage:
            route_names = list(model.ROUTE_NAMES)
            print("🧭 验证集五路平均使用率:")
            for scale, values in route_usage.items():
                text = " | ".join(
                    f"{name}={value:.4f}"
                    for name, value in zip(route_names, values)
                )
                print(f"   {scale}: {text}")

        checkpoint = {
            "epoch": epoch,
            "stage_name": stage_name,
            "route_temperature": temperature,
            "route_regularization_factor": route_factor,
            "physics_weight": criterion.physics_weight,

            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_score": best_score,
            "best_niqe": best_niqe,
            "best_brisque": best_brisque,
            "lowest_niqe": lowest_niqe,
            "lowest_brisque": lowest_brisque,
            "early_stop_counter": early_stop_counter,

            "current_psnr": avg_psnr,
            "current_ssim": avg_ssim,
            "current_score": current_score,
            "current_niqe": avg_niqe,
            "current_brisque": avg_brisque,

            "route_usage": route_usage,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,

            "loss_config": LOSS_CONFIG,
            "train_config": TRAIN_CONFIG,
            "data_config": DATA_CONFIG,
        }

        torch.save(checkpoint, resume_path)
        if is_best:
            torch.save(checkpoint, best_path)

        if (
            TRAIN_CONFIG["save_freq"]
            and (epoch + 1) % TRAIN_CONFIG["save_freq"] == 0
        ):
            torch.save(
                checkpoint,
                checkpoint_dir / f"general_epoch_{epoch + 1}.pth",
            )

        # Save route usage separately for quick inspection.
        with open(
            checkpoint_dir / "latest_route_usage.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(route_usage, file, ensure_ascii=False, indent=2)

        if early_stop_counter >= TRAIN_CONFIG["patience"]:
            print("\n🛑 早停触发，训练结束")
            break

        scheduler.step()

    print("\n🎉 新版通用训练完成")
    print(f"   Best PSNR: {best_psnr:.4f}")
    print(f"   Best SSIM: {best_ssim:.6f}")
    print(f"   Best Score: {best_score:.6f}")
    print(f"   Best checkpoint: {best_path}")


if __name__ == "__main__":
    train()