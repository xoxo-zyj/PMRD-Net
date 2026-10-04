# -*- coding: utf-8 -*-
"""
MSDSPDD Complete Model: Prior -> ISP -> Five Routes -> Reconstruction -> Refinement
==================================================================================

This is the complete, single-file model-specific implementation requested by
its author. It contains every custom stage from the physical prior to J_final:

1. PriorEngineV3: initial transmission t0 + global atmospheric light A0.
2. DynamicPixelwiseISP: refined t1 + spatial local atmospheric-light A1 and
   Dehaze / Denoise / White-Balance / Sharpen / Raw images.
3. Five-route shared EfficientViT-B1 multi-scale feature extraction.
4. Route adapters + explicit operator-response features.
5. t1/A1-guided fusion at H/2, H/4, H/8 and H/16.
6. Local detail attention at H/2 and H/4; VMamba at H/16.
7. Multi-scale U-Net decoder.
8. Raw Detail Gate before J_original.
9. PG-DSP indicator after J_original.
10. Physics-error-guided Residual Refiner producing J_final.
11. Full composite training loss including reconstruction, SSIM, visual,
    edge, color, physical consistency, PG-DSP safety and route survival.

Only the generic EfficientViT and VMamba implementations remain in the bundled
``future_process`` dependency directory; all dehazing-specific modules are in
this one file.

Legacy-compatible forward output:
    final_out, refined_t, J_original, indicator_safe,
    five_route_images, initial_t, initial_A0 = model(hazy)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from future_process.efficientvit.backbone import efficientvit_backbone_b1
    _EFFICIENTVIT_IMPORT_ERROR = None
except Exception as exc:
    efficientvit_backbone_b1 = None
    _EFFICIENTVIT_IMPORT_ERROR = exc

try:
    from future_process.vmamba import VSSBlock
    _VMAMBA_IMPORT_ERROR = None
except Exception as exc:
    VSSBlock = None
    _VMAMBA_IMPORT_ERROR = exc

__all__ = [
    "PriorEngineV3",
    "SpatialOptimizedFilters",
    "DynamicPixelwiseISP",
    "RawDetailGate",
    "PGDSPSubmodule",
    "ResidualRefiner",
    "MSDSPDDDehazer",
    "MSDSPDDCompositeLoss",
]


# =============================================================================
# 1. Complete physical prior module
# =============================================================================

DEFAULT_PRIOR_CONFIG = {
    "dark_pool_size": 15,
    "gf_radius": 5,
    "gf_eps": 1e-3,
    "laplacian_weight": 0.02,
    "top_k_ratio": 0.001,
}


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_size = 2 * int(radius) + 1
    return F.avg_pool2d(x, kernel_size, stride=1, padding=int(radius))


def guided_filter(
    guide: torch.Tensor,
    source: torch.Tensor,
    radius: Optional[int] = None,
    eps: float = 1e-3,
    r: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable guided filter supporting 1- or 3-channel guidance."""
    if radius is None:
        radius = r
    if radius is None:
        raise ValueError("guided_filter requires radius or r.")
    if guide.ndim != 4 or source.ndim != 4:
        raise ValueError("guide and source must be [B,C,H,W].")
    if guide.shape[0] != source.shape[0] or guide.shape[-2:] != source.shape[-2:]:
        raise ValueError("guide and source batch/spatial sizes must match.")
    if guide.shape[1] == 3:
        guide_gray = (
            0.299 * guide[:, 0:1]
            + 0.587 * guide[:, 1:2]
            + 0.114 * guide[:, 2:3]
        )
    elif guide.shape[1] == 1:
        guide_gray = guide
    else:
        raise ValueError("guide must have one or three channels.")

    mean_i = box_filter(guide_gray, radius)
    mean_p = box_filter(source, radius)
    mean_ip = box_filter(guide_gray * source, radius)
    covariance = mean_ip - mean_i * mean_p
    mean_ii = box_filter(guide_gray * guide_gray, radius)
    variance = mean_ii - mean_i * mean_i
    a = covariance / (variance + eps)
    b = mean_p - a * mean_i
    return box_filter(a, radius) * guide_gray + box_filter(b, radius)


def estimate_atmospheric_light(
    image: torch.Tensor,
    dark_channel: torch.Tensor,
    top_k_ratio: float = 0.001,
) -> torch.Tensor:
    """Top-k physical A0 estimate. Top-k selection is intentionally detached."""
    batch, channels, height, width = image.shape
    k = max(1, int(height * width * float(top_k_ratio)))
    _, indices = torch.topk(dark_channel.reshape(batch, -1), k, dim=1)
    values = []
    for batch_index in range(batch):
        pixels = image[batch_index].reshape(channels, -1)
        values.append(pixels[:, indices[batch_index]].mean(dim=-1))
    atmospheric_light = torch.stack(values, dim=0).view(batch, 3, 1, 1)
    return atmospheric_light.clamp(0.6, 1.0)


class PriorEngineV3(nn.Module):
    """
    Complete dual-channel physical prior.

    Outputs:
        t0: [B,1,H,W] initial transmission map.
        A0: [B,3,1,1] initial global atmospheric light.
    """

    def __init__(self, config: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.config = dict(DEFAULT_PRIOR_CONFIG)
        if config is not None:
            self.config.update(config)

        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.fusion_conv.weight.copy_(
                torch.tensor([0.7, 0.3], dtype=torch.float32).view(1, 2, 1, 1)
            )

        laplacian = torch.tensor(
            [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
        ) * 0.5
        self.register_buffer("laplacian_kernel", laplacian)

    def laplacian_smooth(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(x, self.laplacian_kernel, padding=1)
        return x + float(self.config["laplacian_weight"]) * residual

    def forward(self, raw_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3:
            raise ValueError("raw_rgb must be [B,3,H,W].")
        raw_rgb = raw_rgb.clamp(0.0, 1.0)
        pool_size = int(self.config["dark_pool_size"])

        minimum_rgb = raw_rgb.amin(dim=1, keepdim=True)
        dark_channel = -F.max_pool2d(
            -minimum_rgb,
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
        )
        maximum_rgb = raw_rgb.amax(dim=1, keepdim=True)
        bright_channel = F.max_pool2d(
            maximum_rgb,
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
        )

        t_dark = 1.0 - dark_channel
        t_bright = 1.0 - bright_channel
        t_raw = self.fusion_conv(torch.cat([t_dark, t_bright], dim=1))
        t_guided = guided_filter(
            raw_rgb,
            t_raw,
            int(self.config["gf_radius"]),
            float(self.config["gf_eps"]),
        )
        t0 = self.laplacian_smooth(t_guided).clamp(0.0, 1.0)
        A0 = estimate_atmospheric_light(
            raw_rgb,
            dark_channel.detach(),
            float(self.config["top_k_ratio"]),
        )
        return t0, A0


# =============================================================================
# 2. ISP refinement and five dynamic operator-response images
# =============================================================================


class SpatialOptimizedFilters(nn.Module):
    """
    空间自适应物理算子池 (适配新先验 τ_asm + 全局大气光 A)
    输入要求：
        - 所有图像必须是 0~1 范围的 float32 张量
        - tau_asm_map: [B,H,W] 或 [B,1,H,W]，数值范围 0~max_tau
        - A: [B,3,1,1]（全局大气光）或 [B,3,H,W]（空间变化大气光），数值范围 0~1
        - routing_weights: 支持两种格式：
            - [B,3,H,W] 空间权重图（推荐，真正的空间自适应）
            - [B,3] 全局权重（兼容旧代码，自动广播）
    """

    def __init__(
            self,
            denoise_radii=(2, 4, 8),
            denoise_eps=(1e-4, 1e-3, 1e-2),
            denoise_max_strength: float = 0.8,
            sharpen_kernels=(3, 5, 7),
            sharpen_sigmas=(1.0, 1.5, 2.0),
            sharpen_max_strength: float = 1.5,
            wb_max_strength: float = 1.0,
            min_transmission: float = 0.1,
            max_tau: float = 1.0
    ):
        super().__init__()
        self.eps = 1e-4
        self.denoise_radii = denoise_radii
        self.denoise_eps = denoise_eps
        self.denoise_max_strength = denoise_max_strength
        self.sharpen_kernels = sharpen_kernels
        self.sharpen_sigmas = sharpen_sigmas
        self.sharpen_max_strength = sharpen_max_strength
        self.wb_max_strength = wb_max_strength
        self.min_transmission = min_transmission
        self.max_tau = max_tau

        # 预初始化高斯锐化核（仅初始化一次，提升推理速度）
        for i, (k, s) in enumerate(zip(self.sharpen_kernels, self.sharpen_sigmas)):
            kernel_1d = torch.linspace(-(k // 2), k // 2, k)
            kernel_1d = torch.exp(-0.5 * (kernel_1d ** 2) / (s ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d / kernel_2d.sum()
            kernel_2d = kernel_2d.view(1, 1, k, k)
            self.register_buffer(f"sharpen_kernel_{i}", kernel_2d)

    def _check_image(self, x, name="image"):
        """图像输入校验：补batch维度 + 数值范围检查 + 裁剪"""
        if x.ndim == 3:
            x = x.unsqueeze(0)
            warnings.warn(f"{name} 缺少batch维度，已自动补全为 [1,{x.shape[1]},{x.shape[2]},{x.shape[3]}]")

        if x.ndim != 4:
            raise ValueError(f"{name} 必须是3维或4维张量，当前形状: {x.shape}")

        if x.shape[1] != 3:
            raise ValueError(f"{name} 必须是3通道RGB图像，当前通道数: {x.shape[1]}")

        # 数值范围警告（帮助快速定位预处理错误）
        if x.min() < -0.1 or x.max() > 1.1:
            warnings.warn(f"{name} 数值范围超出 0~1，当前范围: [{x.min():.2f}, {x.max():.2f}]，将被裁剪到 0~1")

        return torch.clamp(x, 0.0, 1.0)

    def _process_tau(self, tau_asm_map):
        """统一处理τ_asm维度 + 形状校验"""
        if tau_asm_map.ndim == 3:
            # [B,H,W] → [B,1,H,W]
            return tau_asm_map.unsqueeze(1)
        elif tau_asm_map.ndim == 4 and tau_asm_map.shape[1] == 1:
            # [B,1,H,W] → 直接返回
            return tau_asm_map
        else:
            raise ValueError(
                f"tau_asm_map 形状必须是 [B,H,W] 或 [B,1,H,W]，当前形状: {tau_asm_map.shape}\n"
                f"注意：τ_asm必须是单通道光学厚度图"
            )

    def _check_A(self, A, image_shape):
        """大气光A的形状校验 + 自动广播"""
        if A.ndim != 4:
            raise ValueError(f"大气光A必须是4维张量 [B,3,1,1] 或 [B,3,H,W]，当前形状: {A.shape}")

        if A.shape[1] != 3:
            raise ValueError(f"大气光A必须是3通道，当前通道数: {A.shape[1]}")

        if A.shape[0] != image_shape[0]:
            raise ValueError(f"大气光A的batch数必须与图像一致，A: {A.shape[0]}, 图像: {image_shape[0]}")

        B, _, H, W = image_shape

        if A.shape[-2:] == (1, 1):
            # 全局大气光，自动广播到空间尺寸（不复制数据，内存高效）
            return A.expand(-1, -1, H, W)
        elif A.shape[-2:] == (H, W):
            # 空间变化大气光，直接返回
            return A
        else:
            raise ValueError(
                f"大气光A的空间尺寸必须是 (1,1) 或与图像一致 ({H},{W})，当前尺寸: {A.shape[-2:]}"
            )

    def _process_routing_weights(self, routing_weights, image_shape):
        """统一处理路由权重：支持全局权重[B,3]和空间权重[B,3,H,W]"""
        B, _, H, W = image_shape

        if routing_weights.ndim == 4 and routing_weights.shape == (B, 3, H, W):
            # 空间权重图（推荐），直接使用
            return routing_weights
        elif routing_weights.ndim == 2 and routing_weights.shape == (B, 3):
            # 全局权重（兼容旧代码），自动广播
            warnings.warn(
                "使用全局路由权重 [B,3]，将自动广播为空间图。\n"
                "为获得更好的空间自适应效果，建议使用 [B,3,H,W] 空间权重图。"
            )
            return routing_weights[:, :, None, None].expand(-1, -1, H, W)
        else:
            raise ValueError(
                f"routing_weights 形状不合法，当前形状: {routing_weights.shape}\n"
                f"支持的格式：\n"
                f"  - [B,3,H,W] 空间权重图（推荐）\n"
                f"  - [B,3] 全局权重（兼容）"
            )

    def dehaze_spatial(self, image, tau_asm_map, A):
        """
        空间自适应去雾
        tau_asm_map: 透射率图 [B,H,W] 或 [B,1,H,W]
        A: 大气光，支持两种形状：
            - [B,3,1,1] 全局大气光（来自PriorEngineV3）
            - [B,3,H,W] 空间变化大气光
        """
        image = self._check_image(image, "dehaze输入图像")
        tau_tensor = self._process_tau(tau_asm_map)
        A_spatial = self._check_A(A, image.shape)

        t = torch.clamp(tau_tensor, self.min_transmission + self.eps, 1.0)
        J = A_spatial + (image - A_spatial) / t
        return torch.clamp(J, 0.0, 1.0)

    def denoise_spatial(self, image, tau_asm_map, routing_weights):
        """空间自适应降噪"""
        image = self._check_image(image, "denoise输入图像")
        tau_tensor = self._process_tau(tau_asm_map)
        routing_weights = self._process_routing_weights(routing_weights, image.shape)

        strength_map = (1.0 - tau_tensor) * self.denoise_max_strength

        # 提前转灰度引导图，避免3次重复计算
        gray_guide = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        blurred_1 = guided_filter(gray_guide, image, r=self.denoise_radii[0], eps=self.denoise_eps[0])
        blurred_2 = guided_filter(gray_guide, image, r=self.denoise_radii[1], eps=self.denoise_eps[1])
        blurred_3 = guided_filter(gray_guide, image, r=self.denoise_radii[2], eps=self.denoise_eps[2])

        w1, w2, w3 = routing_weights[:, 0:1], routing_weights[:, 1:2], routing_weights[:, 2:3]
        fused_blurred = w1 * blurred_1 + w2 * blurred_2 + w3 * blurred_3

        lum = gray_guide
        adaptive_strength = strength_map * (1.0 - lum + self.eps)

        return image * (1 - adaptive_strength) + fused_blurred * adaptive_strength

    def sharpen_spatial(self, image, tau_asm_map, routing_weights):
        """
        Reliability-aware spatial sharpening.

        Unlike the old rule ``strength=(1-t)*max_strength``, this version avoids
        applying the strongest sharpening in the densest haze. It combines:
        learned kernel routing, edge confidence, transmission confidence and
        a local-noise suppression factor.
        """
        image = self._check_image(image, "sharpen输入图像")
        tau_tensor = self._process_tau(tau_asm_map)
        routing_weights = self._process_routing_weights(routing_weights, image.shape)

        blurred_1 = self._gaussian_blur(image, kernel_idx=0)
        blurred_2 = self._gaussian_blur(image, kernel_idx=1)
        blurred_3 = self._gaussian_blur(image, kernel_idx=2)

        w1, w2, w3 = routing_weights[:, 0:1], routing_weights[:, 1:2], routing_weights[:, 2:3]
        fused_blurred = w1 * blurred_1 + w2 * blurred_2 + w3 * blurred_3

        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + self.eps)
        edge_max = edge.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        edge_confidence = (edge / edge_max).clamp(0.0, 1.0)

        local_mean = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        noise = (image - local_mean).abs().mean(dim=1, keepdim=True)
        noise_max = noise.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        noise_normalized = (noise / noise_max).clamp(0.0, 1.0)
        noise_suppression = torch.exp(-4.0 * noise_normalized)

        transmission_confidence = 0.20 + 0.80 * tau_tensor.clamp(0.0, 1.0)
        strength_map = (
            self.sharpen_max_strength
            * edge_confidence
            * noise_suppression
            * transmission_confidence
        )
        detail = image - fused_blurred
        return torch.clamp(image + strength_map * detail, 0.0, 1.0)

    def white_balance_spatial(self, image, tau_asm_map):
        """空间自适应白平衡"""
        image = self._check_image(image, "白平衡输入图像")
        tau_tensor = self._process_tau(tau_asm_map)

        mean_rgb = torch.mean(image, dim=(2, 3), keepdim=True)  # [B,3,1,1]
        gain = 0.5 / (mean_rgb + self.eps)
        gain = torch.clamp(gain, 0.4, 2.5)

        strength_map = (1.0 - tau_tensor) * self.wb_max_strength

        wb_image = image * gain
        return torch.clamp(image * (1 - strength_map) + wb_image * strength_map, 0.0, 1.0)

    def _gaussian_blur(self, x, kernel_idx):
        """高斯模糊（复用预初始化核，分组卷积提升效率）"""
        B, C, H, W = x.shape
        kernel = getattr(self, f"sharpen_kernel_{kernel_idx}")
        kernel = kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=C)


# ====================== ISP 参数自适应微调模块 ======================
class DynamicPixelwiseISP(nn.Module):
    def __init__(self, num_finetune_params: int = 10, hidden_channels: int = 16, kernel_size: int = 3,
                 max_tau: float = 1.0, **filter_kwargs):
        super().__init__()
        self.max_tau = max_tau

        wb_strength = filter_kwargs.pop('wb_max_strength', 1.0)
        _ = filter_kwargs.pop('contrast_max_factor', None)

        self.tools = SpatialOptimizedFilters(
            max_tau=max_tau,
            wb_max_strength=wb_strength,
            **filter_kwargs
        )

        self.finetune_net = nn.Sequential(
            nn.Conv2d(7, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, num_finetune_params, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        nn.init.zeros_(self.finetune_net[-1].weight)
        nn.init.zeros_(self.finetune_net[-1].bias)

    def forward(self, raw_img, tau_asm, soft_A):
        B, C, H, W = raw_img.shape

        tau_tensor = tau_asm if tau_asm.ndim == 4 else tau_asm.unsqueeze(1)
        if tau_tensor.shape != (B, 1, H, W):
            raise ValueError(f"tau_asm 形状不合法: {tau_tensor.shape}")

        if soft_A.shape == (B, 3, 1, 1):
            A_tensor = soft_A.expand(-1, -1, H, W)
        elif soft_A.shape == (B, 3, H, W):
            A_tensor = soft_A
        else:
            raise ValueError(f"soft_A 形状不合法: {soft_A.shape}")

        tau_normalized = torch.clamp(tau_tensor, 0.0, 1.0)
        x = torch.cat([raw_img, tau_normalized, A_tensor], dim=1)
        commands = self.finetune_net(x)

        delta_tau = torch.tanh(commands[:, 0:1, :, :]) * 0.1
        delta_A = torch.tanh(commands[:, 1:4, :, :]) * 0.1
        
        w_denoise = F.softmax(commands[:, 4:7, :, :], dim=1)
        w_sharpen = F.softmax(commands[:, 7:10, :, :], dim=1)

        # ✅ 修复：对齐下界限制，采用 tools 设定的 min_transmission 并加上 eps，防止梯度截断丢失
        finetuned_tau_tensor = torch.clamp(
            tau_tensor + delta_tau, 
            min=self.tools.min_transmission + self.tools.eps, 
            max=self.max_tau
        )
        finetuned_A_tensor = torch.clamp(A_tensor + delta_A, min=0.0, max=1.0)
        
        # ✅ 优化：去除多余的 squeeze 操作，直接传递 4 维张量，减少无意义的维度变换
        dehaze_img = self.tools.dehaze_spatial(raw_img, finetuned_tau_tensor, finetuned_A_tensor)
        dn_img = self.tools.denoise_spatial(raw_img, finetuned_tau_tensor, w_denoise)
        wb_img = self.tools.white_balance_spatial(raw_img, finetuned_tau_tensor)
        sharp_img = self.tools.sharpen_spatial(raw_img, finetuned_tau_tensor, w_sharpen)

        return [dehaze_img, dn_img, wb_img, sharp_img, raw_img], finetuned_tau_tensor, finetuned_A_tensor

# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------

def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDWBlock(nn.Module):
    """Depthwise-local residual block, stable for batch size 1."""

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden,
            bias=False,
        )
        self.norm_hidden = nn.GroupNorm(_group_count(hidden), hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.act(self.pw1(y))
        y = self.act(self.norm_hidden(self.dw(y)))
        y = self.pw2(y)
        return x + self.res_scale * y


class ECALayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class LocalDetailEnhancementBlock(nn.Module):
    """
    Local detail module used only at H/2 and H/4:
    DWConv + pointwise mixing + ECA + local spatial gate + residual.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.eca = ECALayer(channels)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.output = ResidualDWBlock(channels)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.eca(self.local(x))
        spatial = self.spatial_gate(
            torch.cat(
                [local.mean(dim=1, keepdim=True), local.amax(dim=1, keepdim=True)],
                dim=1,
            )
        )
        return self.output(x + self.scale * local * spatial)


def _icnr_init(weight: torch.Tensor, upscale_factor: int = 2) -> None:
    out_channels, in_channels, kh, kw = weight.shape
    reduced_out = out_channels // (upscale_factor ** 2)
    kernel = torch.empty(
        reduced_out,
        in_channels,
        kh,
        kw,
        dtype=weight.dtype,
        device=weight.device,
    )
    nn.init.kaiming_normal_(kernel)
    kernel = kernel.repeat_interleave(upscale_factor ** 2, dim=0)
    with torch.no_grad():
        weight.copy_(kernel)


# -----------------------------------------------------------------------------
# EfficientViT shared five-route encoder
# -----------------------------------------------------------------------------

class FallbackPyramidBackbone(nn.Module):
    """Smoke-test fallback. Formal experiments should use EfficientViT-B1."""

    def __init__(self) -> None:
        super().__init__()
        self.stage0 = nn.Sequential(
            ConvGNAct(3, 16, kernel_size=3, stride=2),
            ResidualDWBlock(16),
        )
        self.stage1 = nn.Sequential(
            ConvGNAct(16, 32, kernel_size=3, stride=2),
            ResidualDWBlock(32),
        )
        self.stage2 = nn.Sequential(
            ConvGNAct(32, 64, kernel_size=3, stride=2),
            ResidualDWBlock(64),
        )
        self.stage3 = nn.Sequential(
            ConvGNAct(64, 128, kernel_size=3, stride=2),
            ResidualDWBlock(128),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s2 = self.stage0(x)
        s4 = self.stage1(s2)
        s8 = self.stage2(s4)
        s16 = self.stage3(s8)
        return {
            "stage0": s2,
            "stage1": s4,
            "stage2": s8,
            "stage3": s16,
            "stage_final": s16,
        }


class ResponsePyramidEncoder(nn.Module):
    """Shared encoder for explicit operator response Ri = Xi - Raw."""

    def __init__(self, channels: Sequence[int] = (16, 32, 64, 128)) -> None:
        super().__init__()
        c2, c4, c8, c16 = [int(v) for v in channels]
        self.s2 = nn.Sequential(
            ConvGNAct(3, c2, kernel_size=3, stride=2),
            ResidualDWBlock(c2),
        )
        self.s4 = nn.Sequential(
            ConvGNAct(c2, c4, kernel_size=3, stride=2),
            ResidualDWBlock(c4),
        )
        self.s8 = nn.Sequential(
            ConvGNAct(c4, c8, kernel_size=3, stride=2),
            ResidualDWBlock(c8),
        )
        self.s16 = nn.Sequential(
            ConvGNAct(c8, c16, kernel_size=3, stride=2),
            ResidualDWBlock(c16),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s2 = self.s2(x)
        s4 = self.s4(s2)
        s8 = self.s8(s4)
        s16 = self.s16(s8)
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16}


class RouteAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ResidualDWBlock(channels),
            ConvGNAct(channels, channels, kernel_size=1),
            ResidualDWBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FiveRouteEfficientViTEncoder(nn.Module):
    ROUTE_NAMES = ("dehaze", "denoise", "white_balance", "sharpen", "raw")

    def __init__(
        self,
        weight_path: str,
        channels: Sequence[int] = (16, 32, 64, 128),
        allow_backbone_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        self.num_routes = 5
        self.backbone_is_fallback = False

        if efficientvit_backbone_b1 is None:
            if not allow_backbone_fallback:
                raise ImportError(
                    "EfficientViT-B1 could not be imported. Install the original "
                    "project dependency before formal training. Original error: "
                    f"{_EFFICIENTVIT_IMPORT_ERROR!r}"
                )
            warnings.warn(
                "EfficientViT dependency is unavailable; smoke-test fallback is active. "
                "Do not use fallback weights for formal experiments.",
                RuntimeWarning,
            )
            self.backbone = FallbackPyramidBackbone()
            self.backbone_is_fallback = True
        else:
            self.backbone = efficientvit_backbone_b1(pretrained=False)
            self._load_backbone_weight(weight_path)
            # stages[0:3] correspond to H/4, H/8 and H/16. Drop H/32.
            if hasattr(self.backbone, "stages") and len(self.backbone.stages) > 3:
                del self.backbone.stages[3:]

        self.response_encoder = ResponsePyramidEncoder(self.channels)
        self.response_scales = nn.ParameterDict(
            {
                "s2": nn.Parameter(torch.tensor(0.1)),
                "s4": nn.Parameter(torch.tensor(0.1)),
                "s8": nn.Parameter(torch.tensor(0.1)),
                "s16": nn.Parameter(torch.tensor(0.1)),
            }
        )

        self.route_adapters = nn.ModuleDict()
        for scale_name, channels_at_scale in zip(
            ("s2", "s4", "s8", "s16"), self.channels
        ):
            self.route_adapters[scale_name] = nn.ModuleList(
                [RouteAdapter(channels_at_scale) for _ in range(self.num_routes)]
            )

    def _load_backbone_weight(self, weight_path: str) -> None:
        if not weight_path or not os.path.exists(weight_path):
            warnings.warn(
                f"EfficientViT weight not found: {weight_path}. Backbone will start "
                "from random initialization.",
                RuntimeWarning,
            )
            return

        try:
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(weight_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            for candidate in ("state_dict", "model", "params", "params_ema"):
                if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
                    checkpoint = checkpoint[candidate]
                    break

        cleaned = {}
        for key, value in checkpoint.items():
            for prefix in ("module.", "backbone.", "model.backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value

        result = self.backbone.load_state_dict(cleaned, strict=False)
        if len(result.missing_keys) > 0:
            warnings.warn(
                f"EfficientViT loaded with {len(result.missing_keys)} missing keys and "
                f"{len(result.unexpected_keys)} unexpected keys.",
                RuntimeWarning,
            )

    def forward(
        self,
        route_images: Sequence[torch.Tensor],
        raw_image: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        if len(route_images) != self.num_routes:
            raise ValueError(f"Expected five route images, got {len(route_images)}.")

        batch, _, height, width = raw_image.shape
        aligned_routes = []
        for image in route_images:
            if image.shape[-2:] != (height, width):
                image = F.interpolate(
                    image,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_routes.append(image)

        route_stack = torch.stack(aligned_routes, dim=1)  # [B,5,3,H,W]
        route_batch = route_stack.reshape(batch * self.num_routes, 3, height, width)
        response_batch = (
            route_stack - raw_image.unsqueeze(1)
        ).reshape(batch * self.num_routes, 3, height, width)

        image_features = self.backbone(route_batch)
        response_features = self.response_encoder(response_batch)

        stage_mapping = {
            "s2": "stage0",
            "s4": "stage1",
            "s8": "stage2",
            "s16": "stage3",
        }
        output: Dict[str, List[torch.Tensor]] = {}

        for scale_name, stage_name in stage_mapping.items():
            image_feature = image_features[stage_name]
            response_feature = response_features[scale_name]
            _, channels, hs, ws = image_feature.shape

            image_feature = image_feature.reshape(
                batch, self.num_routes, channels, hs, ws
            )
            response_feature = response_feature.reshape(
                batch, self.num_routes, channels, hs, ws
            )

            route_list: List[torch.Tensor] = []
            response_scale = self.response_scales[scale_name]
            for route_index in range(self.num_routes):
                combined = (
                    image_feature[:, route_index]
                    + response_scale * response_feature[:, route_index]
                )
                route_list.append(
                    self.route_adapters[scale_name][route_index](combined)
                )
            output[scale_name] = route_list

        return output


# -----------------------------------------------------------------------------
# Prior-guided route fusion
# -----------------------------------------------------------------------------

class PriorGuidedFiveRouteFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        prior_channels: int = 4,
        num_routes: int = 5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_routes = num_routes

        self.prior_encoder = nn.Sequential(
            ConvGNAct(prior_channels, channels, kernel_size=3),
            ResidualDWBlock(channels),
        )
        self.route_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(channels, channels, kernel_size=1),
                    ResidualDWBlock(channels),
                )
                for _ in range(num_routes)
            ]
        )

        spatial_hidden = max(16, channels // 2)
        channel_hidden = max(8, channels // 4)
        self.spatial_heads = nn.ModuleList()
        self.channel_heads = nn.ModuleList()
        for _ in range(num_routes):
            spatial_head = nn.Sequential(
                ConvGNAct(channels * 2, spatial_hidden, kernel_size=3),
                nn.Conv2d(spatial_hidden, 1, kernel_size=1, bias=True),
            )
            channel_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channel_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(channel_hidden, channels, kernel_size=1),
            )
            nn.init.zeros_(spatial_head[-1].weight)
            nn.init.zeros_(spatial_head[-1].bias)
            nn.init.zeros_(channel_head[-1].weight)
            nn.init.zeros_(channel_head[-1].bias)
            self.spatial_heads.append(spatial_head)
            self.channel_heads.append(channel_head)

        self.output_refine = nn.Sequential(
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )

    def forward(
        self,
        route_features: Sequence[torch.Tensor],
        refined_prior: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(route_features) != self.num_routes:
            raise ValueError(f"Expected {self.num_routes} routes.")
        temperature = max(float(temperature), 1e-3)

        target_size = route_features[0].shape[-2:]
        prior = F.interpolate(
            refined_prior,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        prior_feature = self.prior_encoder(prior)

        calibrated_features: List[torch.Tensor] = []
        spatial_logits: List[torch.Tensor] = []
        channel_gates: List[torch.Tensor] = []

        for route_index, feature in enumerate(route_features):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            feature = self.route_projectors[route_index](feature)
            joint = torch.cat([feature, prior_feature], dim=1)
            spatial_logits.append(self.spatial_heads[route_index](joint))
            # 2*sigmoid(0)=1: neutral channel calibration at initialization.
            channel_gate = 2.0 * torch.sigmoid(self.channel_heads[route_index](joint))
            channel_gates.append(channel_gate)
            calibrated_features.append(feature * channel_gate)

        logits = torch.cat(spatial_logits, dim=1)
        route_weights = torch.softmax(logits / temperature, dim=1)

        fused = torch.zeros_like(calibrated_features[0])
        for route_index, feature in enumerate(calibrated_features):
            fused = fused + feature * route_weights[:, route_index:route_index + 1]

        fused = self.output_refine(fused)
        channel_gate_tensor = torch.stack(channel_gates, dim=1)
        return fused, route_weights, channel_gate_tensor


# -----------------------------------------------------------------------------
# Global context, decoder and three explicitly retained final modules
# -----------------------------------------------------------------------------

class GlobalContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ssm_d_state: int,
        use_mamba: bool,
    ) -> None:
        super().__init__()
        self.mamba_enabled = bool(use_mamba and VSSBlock is not None)
        self.pre = ResidualDWBlock(channels)

        if self.mamba_enabled:
            self.context = nn.Sequential(
                nn.GroupNorm(_group_count(channels), channels),
                VSSBlock(
                    hidden_dim=channels,
                    drop_path=0.05,
                    channel_first=True,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=2.0,
                    ssm_conv=3,
                    mlp_ratio=2.0,
                ),
                nn.GroupNorm(_group_count(channels), channels),
            )
        else:
            if use_mamba and VSSBlock is None:
                warnings.warn(
                    "VMamba is unavailable; dilated-convolution fallback is active. "
                    f"Original import error: {_VMAMBA_IMPORT_ERROR!r}",
                    RuntimeWarning,
                )
            self.context = nn.Sequential(
                ResidualDWBlock(channels, dilation=1),
                ResidualDWBlock(channels, dilation=2),
                ResidualDWBlock(channels, dilation=3),
            )
        self.post = ResidualDWBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = x + self.context(x)
        return self.post(x)


class PixelShuffleDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_conv = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        _icnr_init(self.up_conv.weight, upscale_factor=2)
        self.up = nn.Sequential(
            self.up_conv,
            nn.PixelShuffle(2),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(out_channels + skip_channels, out_channels, kernel_size=3),
            ResidualDWBlock(out_channels),
            ResidualDWBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        _icnr_init(conv.weight, upscale_factor=2)
        self.body = nn.Sequential(
            conv,
            nn.PixelShuffle(2),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            ResidualDWBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RawDetailGate(nn.Module):
    """
    Retained module 1/3.

    Operates before J_original. It controls which full-resolution raw textures
    can enter the final decoder while suppressing raw haze in low-transmission
    regions. The gate is initialized to approximately the refined transmission.
    """

    def __init__(self, decoder_channels: int = 24, raw_channels: int = 24) -> None:
        super().__init__()
        self.raw_encoder = nn.Sequential(
            ConvGNAct(3, raw_channels, kernel_size=3),
            ResidualDWBlock(raw_channels),
        )
        hidden = max(16, decoder_channels)
        self.gate_net = nn.Sequential(
            ConvGNAct(decoder_channels + raw_channels + 1, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        self.fuse = nn.Sequential(
            ConvGNAct(decoder_channels + raw_channels, decoder_channels, kernel_size=3),
            ResidualDWBlock(decoder_channels),
            ResidualDWBlock(decoder_channels),
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        raw_image: torch.Tensor,
        refined_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_feature = self.raw_encoder(raw_image)
        if decoder_feature.shape[-2:] != raw_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=raw_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if refined_t.shape[-2:] != raw_feature.shape[-2:]:
            refined_t = F.interpolate(
                refined_t,
                size=raw_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gate_residual = self.gate_net(
            torch.cat([decoder_feature, raw_feature, refined_t], dim=1)
        )
        t_safe = refined_t.clamp(0.05, 0.95)
        prior_logit = torch.log(t_safe) - torch.log1p(-t_safe)
        detail_gate = torch.sigmoid(gate_residual + prior_logit)
        gated_raw = raw_feature * detail_gate
        fused = self.fuse(torch.cat([decoder_feature, gated_raw], dim=1))
        return fused, detail_gate


class PGDSPSubmodule(nn.Module):
    """
    Retained module 2/3.

    PG-DSP does not directly alter RGB. It creates a safe residual-correction
    indicator: high in flat, haze-heavy regions and lower near strong edges.
    """

    def __init__(
        self,
        gamma_edge: float = 3.0,
        haze_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("gamma_edge", torch.tensor(float(gamma_edge)))
        self.haze_power = float(haze_power)
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.last_structure_map: Optional[torch.Tensor] = None
        self.last_haze_need_map: Optional[torch.Tensor] = None

    def _gradient(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3]
        )
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(
        self,
        restored: torch.Tensor,
        refined_t: torch.Tensor,
    ) -> torch.Tensor:
        gradient = self._gradient(restored)
        grad_max = gradient.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        structure = (gradient / grad_max).clamp(0.0, 1.0)
        haze_need = (1.0 - refined_t.clamp(0.0, 1.0)).pow(self.haze_power)
        edge_protection = torch.exp(-self.gamma_edge * structure)
        indicator_safe = (haze_need * edge_protection).clamp(0.0, 1.0)
        self.last_structure_map = structure
        self.last_haze_need_map = haze_need
        return indicator_safe


class ResidualRefiner(nn.Module):
    """
    Retained module 3/3.

    The only one of the three retained modules that directly changes RGB.
    It is zero-initialized and low-amplitude, and its gate is explicitly guided
    by the PG-DSP indicator.
    """

    def __init__(
        self,
        hidden_channels: int = 48,
        max_residual: float = 0.10,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        # base(3)+raw(3)+t(1)+A(3)+indicator(1)+physics_error(3)=14
        self.stem = ConvGNAct(14, hidden_channels, kernel_size=3)
        self.body = nn.Sequential(
            ResidualDWBlock(hidden_channels),
            LocalDetailEnhancementBlock(hidden_channels),
            ResidualDWBlock(hidden_channels),
        )
        self.delta_head = nn.Conv2d(
            hidden_channels, 3, kernel_size=3, padding=1, bias=True
        )
        self.gate_head = nn.Conv2d(
            hidden_channels, 1, kernel_size=3, padding=1, bias=True
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        base: torch.Tensor,
        raw: torch.Tensor,
        refined_t: torch.Tensor,
        local_A: torch.Tensor,
        indicator_safe: torch.Tensor,
        physics_error: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat(
            [base, raw, refined_t, local_A, indicator_safe, physics_error],
            dim=1,
        )
        feature = self.body(self.stem(x))
        delta = self.max_residual * torch.tanh(self.delta_head(feature))
        learned_gate = torch.sigmoid(self.gate_head(feature))
        pgdsp_gate = 0.1 + 0.9 * indicator_safe
        gate = learned_gate * pgdsp_gate
        output = torch.clamp(base + gate * delta, 0.0, 1.0)
        return output, delta, gate


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------

class MSDSPDDDehazer(nn.Module):
    ROUTE_NAMES = FiveRouteEfficientViTEncoder.ROUTE_NAMES

    def __init__(
        self,
        prior_channels: int = 4,
        ssm_d_state: int = 16,
        max_tau: float = 1.0,
        gamma: float = 3.0,
        weight_path: str = "./future_process/efficientvit_b1_r256.pt",
        enable_end_to_end: bool = True,
        route_temperature: float = 1.5,
        use_mamba: bool = True,
        enable_refiner: bool = True,
        allow_backbone_fallback: bool = False,
    ) -> None:
        super().__init__()
        if prior_channels != 4:
            raise ValueError(
                "The refined prior must contain t(1) + local atmospheric light(3)."
            )

        self.max_tau = float(max_tau)
        self.enable_end_to_end = bool(enable_end_to_end)
        self.enable_refiner = bool(enable_refiner)
        self.register_buffer(
            "route_temperature",
            torch.tensor(float(route_temperature), dtype=torch.float32),
        )

        if self.enable_end_to_end:
            self.prior_engine = PriorEngineV3()

        # Keep the original physical module name for checkpoint compatibility.
        self.isp_module = DynamicPixelwiseISP(max_tau=max_tau)

        self.extractor = FiveRouteEfficientViTEncoder(
            weight_path=weight_path,
            channels=(16, 32, 64, 128),
            allow_backbone_fallback=allow_backbone_fallback,
        )

        self.fusion_s2 = PriorGuidedFiveRouteFusion(16, prior_channels)
        self.fusion_s4 = PriorGuidedFiveRouteFusion(32, prior_channels)
        self.fusion_s8 = PriorGuidedFiveRouteFusion(64, prior_channels)
        self.fusion_s16 = PriorGuidedFiveRouteFusion(128, prior_channels)

        self.local_detail_s2 = LocalDetailEnhancementBlock(16)
        self.local_detail_s4 = LocalDetailEnhancementBlock(32)
        self.mid_scale_s8 = nn.Sequential(
            ResidualDWBlock(64),
            ResidualDWBlock(64),
        )
        self.global_context_s16 = GlobalContextBlock(
            128,
            ssm_d_state=ssm_d_state,
            use_mamba=use_mamba,
        )

        self.decode_s8 = PixelShuffleDecoderBlock(128, 64, 96)
        self.decode_s4 = PixelShuffleDecoderBlock(96, 32, 64)
        self.decode_s2 = PixelShuffleDecoderBlock(64, 16, 32)
        self.decode_full = PixelShuffleUp(32, 24)

        # Explicitly retained final modules.
        self.raw_detail_gate = RawDetailGate(decoder_channels=24, raw_channels=24)
        self.output_head = nn.Sequential(
            ResidualDWBlock(24),
            nn.Conv2d(24, 3, kernel_size=3, padding=1, bias=True),
        )
        self.pg_dsp_module = PGDSPSubmodule(gamma_edge=gamma, haze_power=1.0)
        self.residual_refiner = ResidualRefiner(
            hidden_channels=48,
            max_residual=0.10,
        )

        # Aux tensors are deliberately kept with gradients for the composite loss.
        self.last_route_weights: Dict[str, torch.Tensor] = {}
        self.last_channel_gates: Dict[str, torch.Tensor] = {}
        self.last_refined_transmission: Optional[torch.Tensor] = None
        self.last_local_atmospheric_light: Optional[torch.Tensor] = None
        self.last_raw_detail_gate: Optional[torch.Tensor] = None
        self.last_physics_error: Optional[torch.Tensor] = None
        self.last_refine_delta: Optional[torch.Tensor] = None
        self.last_refine_gate: Optional[torch.Tensor] = None

    @property
    def mamba_enabled(self) -> bool:
        return bool(self.global_context_s16.mamba_enabled)

    @property
    def backbone_is_fallback(self) -> bool:
        return bool(self.extractor.backbone_is_fallback)

    def set_route_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("Route temperature must be positive.")
        self.route_temperature.fill_(float(temperature))

    @staticmethod
    def _spatial_atmospheric_light(
        atmospheric_light: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if atmospheric_light.ndim != 4 or atmospheric_light.shape[1] != 3:
            raise ValueError(
                "Atmospheric light must be [B,3,1,1] or [B,3,H,W]."
            )
        if atmospheric_light.shape[-2:] == (1, 1):
            return atmospheric_light.expand(-1, -1, height, width)
        if atmospheric_light.shape[-2:] == (height, width):
            return atmospheric_light
        return F.interpolate(
            atmospheric_light,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def rehaze(
        clear_image: torch.Tensor,
        transmission: torch.Tensor,
        local_atmospheric_light: torch.Tensor,
    ) -> torch.Tensor:
        return (
            clear_image * transmission
            + local_atmospheric_light * (1.0 - transmission)
        )

    def forward(
        self,
        raw_img: torch.Tensor,
        tau_asm: Optional[torch.Tensor] = None,
        soft_A: Optional[torch.Tensor] = None,
    ):
        if raw_img.ndim != 4 or raw_img.shape[1] != 3:
            raise ValueError("raw_img must have shape [B,3,H,W].")
        raw_img = raw_img.clamp(0.0, 1.0)
        batch, _, height, width = raw_img.shape

        if self.enable_end_to_end:
            tau_asm, soft_A = self.prior_engine(raw_img)
        elif tau_asm is None or soft_A is None:
            raise ValueError(
                "tau_asm and soft_A are required when enable_end_to_end=False."
            )

        initial_t = tau_asm if tau_asm.ndim == 4 else tau_asm.unsqueeze(1)
        if initial_t.shape[-2:] != (height, width):
            initial_t = F.interpolate(
                initial_t,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # ISP jointly refines physical variables and creates five images.
        ops_imgs_5, refined_t, local_A = self.isp_module(
            raw_img,
            initial_t,
            soft_A,
        )
        refined_t = refined_t.clamp(1e-4, self.max_tau)
        local_A = self._spatial_atmospheric_light(local_A, height, width).clamp(0.0, 1.0)
        refined_prior = torch.cat([refined_t, local_A], dim=1)

        route_features = self.extractor(ops_imgs_5, raw_img)
        temperature = float(self.route_temperature.item())

        f2, w2, c2 = self.fusion_s2(route_features["s2"], refined_prior, temperature)
        f4, w4, c4 = self.fusion_s4(route_features["s4"], refined_prior, temperature)
        f8, w8, c8 = self.fusion_s8(route_features["s8"], refined_prior, temperature)
        f16, w16, c16 = self.fusion_s16(route_features["s16"], refined_prior, temperature)

        f2 = self.local_detail_s2(f2)
        f4 = self.local_detail_s4(f4)
        f8 = self.mid_scale_s8(f8)
        f16 = self.global_context_s16(f16)

        d8 = self.decode_s8(f16, f8)
        d4 = self.decode_s4(d8, f4)
        d2 = self.decode_s2(d4, f2)
        full_decoder = self.decode_full(d2)
        gated_feature, raw_detail_gate = self.raw_detail_gate(
            full_decoder,
            raw_img,
            refined_t,
        )

        logits = self.output_head(gated_feature)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        J_original = torch.sigmoid(logits)

        # PG-DSP judges where residual correction is safe/useful.
        indicator_safe = self.pg_dsp_module(J_original, refined_t)
        re_hazy = self.rehaze(J_original, refined_t, local_A)
        physics_error = raw_img - re_hazy

        if self.enable_refiner:
            final_out, refine_delta, refine_gate = self.residual_refiner(
                J_original,
                raw_img,
                refined_t,
                local_A,
                indicator_safe,
                physics_error,
            )
        else:
            final_out = J_original.clamp(0.0, 1.0)
            refine_delta = torch.zeros_like(final_out)
            refine_gate = torch.zeros_like(final_out[:, 0:1])

        self.last_route_weights = {
            "s2": w2,
            "s4": w4,
            "s8": w8,
            "s16": w16,
        }
        self.last_channel_gates = {
            "s2": c2,
            "s4": c4,
            "s8": c8,
            "s16": c16,
        }
        self.last_refined_transmission = refined_t
        self.last_local_atmospheric_light = local_A
        self.last_raw_detail_gate = raw_detail_gate
        self.last_physics_error = physics_error
        self.last_refine_delta = refine_delta
        self.last_refine_gate = refine_gate

        # Keep the old return tuple. tau_ode now correctly uses refined t1.
        tau_ode = refined_t
        if self.enable_end_to_end:
            return (
                final_out,
                tau_ode,
                J_original,
                indicator_safe,
                ops_imgs_5,
                tau_asm,
                soft_A,
            )
        return final_out, tau_ode, J_original, indicator_safe, ops_imgs_5

    def get_auxiliary_outputs(self, detach: bool = False) -> Dict[str, object]:
        aux: Dict[str, object] = {
            "route_weights": self.last_route_weights,
            "channel_gates": self.last_channel_gates,
            "refined_transmission": self.last_refined_transmission,
            "local_atmospheric_light": self.last_local_atmospheric_light,
            "raw_detail_gate": self.last_raw_detail_gate,
            "physics_error": self.last_physics_error,
            "refine_delta": self.last_refine_delta,
            "refine_gate": self.last_refine_gate,
            "pgdsp_structure": self.pg_dsp_module.last_structure_map,
            "pgdsp_haze_need": self.pg_dsp_module.last_haze_need_map,
        }
        if not detach:
            return aux

        def _detach(value):
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, dict):
                return {k: _detach(v) for k, v in value.items()}
            return value

        return _detach(aux)

    def load_v1_checkpoint(
        self,
        checkpoint: Union[str, os.PathLike, Dict[str, torch.Tensor]],
        map_location: str = "cpu",
    ) -> Dict[str, List[str]]:
        """Shape-safe partial warm start from a V1 checkpoint."""
        if isinstance(checkpoint, (str, os.PathLike)):
            try:
                checkpoint = torch.load(
                    checkpoint,
                    map_location=map_location,
                    weights_only=True,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint, map_location=map_location)

        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must be a path or a state-dict-like object.")

        state = checkpoint
        for candidate in ("model", "state_dict", "params", "params_ema"):
            if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
                state = checkpoint[candidate]
                break

        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value

        current = self.state_dict()
        compatible = {
            key: value
            for key, value in cleaned.items()
            if key in current and current[key].shape == value.shape
        }
        result = self.load_state_dict(compatible, strict=False)
        skipped = [key for key in cleaned if key not in compatible]
        return {
            "loaded": sorted(compatible.keys()),
            "skipped": sorted(skipped),
            "missing": sorted(result.missing_keys),
            "unexpected": sorted(result.unexpected_keys),
        }

    def freeze_batch_norm(self, freeze_affine: bool = True) -> None:
        """Call after model.train() for tiny-batch dataset-specific fine-tuning."""
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if freeze_affine:
                    for parameter in module.parameters():
                        parameter.requires_grad = False


# =============================================================================
# 11. Complete composite training loss
# =============================================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).square() + self.eps ** 2).mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    def _window(self, channels: int, device, dtype) -> torch.Tensor:
        coords = torch.arange(self.window_size, device=device, dtype=dtype)
        coords = coords - (self.window_size - 1) / 2
        kernel_1d = torch.exp(-(coords.square()) / (2 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.view(1, 1, self.window_size, self.window_size).repeat(
            channels, 1, 1, 1
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        window = self._window(channels, pred.device, pred.dtype)
        padding = self.window_size // 2
        mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_y = F.conv2d(target, window, padding=padding, groups=channels)
        sigma_x = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x.square()
        sigma_y = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y.square()
        sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_x * mu_y
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - ssim.mean()


class GradientLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _gradient(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = image.shape[1]
        kx = self.kx.repeat(channels, 1, 1, 1)
        ky = self.ky.repeat(channels, 1, 1, 1)
        gx = F.conv2d(image, kx, padding=1, groups=channels)
        gy = F.conv2d(image, ky, padding=1, groups=channels)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        px, py = self._gradient(pred)
        tx, ty = self._gradient(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class ColorConsistencyLoss(nn.Module):
    """Matches global RGB statistics and opponent-color chroma."""

    @staticmethod
    def _opponent(image: torch.Tensor) -> torch.Tensor:
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = b - y
        v = r - y
        return torch.cat([y, u, v], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3), unbiased=False)
        target_std = target.std(dim=(2, 3), unbiased=False)
        stats = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
        chroma = F.l1_loss(self._opponent(pred), self._opponent(target))
        return stats + 0.5 * chroma


class MultiScaleVisualPerceptualLoss(nn.Module):
    """
    Dependency-free perceptual/visual loss.

    It compares luminance/chroma and gradients over several image scales. This
    is safer than silently using an untrained VGG when pretrained weights are
    unavailable. A VGG/LPIPS loss can still be added externally if desired.
    """

    def __init__(
        self,
        scales: Sequence[int] = (1, 2, 4),
        scale_weights: Sequence[float] = (1.0, 0.5, 0.25),
    ) -> None:
        super().__init__()
        if len(scales) != len(scale_weights):
            raise ValueError("scales and scale_weights must have equal length.")
        self.scales = tuple(int(v) for v in scales)
        self.scale_weights = tuple(float(v) for v in scale_weights)
        self.gradient = GradientLoss()
        self.color = ColorConsistencyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_tensor(0.0)
        normalizer = sum(self.scale_weights)
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1:
                p, t = pred, target
            else:
                p = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                t = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            total = total + weight * (
                F.l1_loss(p, t)
                + 0.25 * self.gradient(p, t)
                + 0.25 * self.color(p, t)
            )
        return total / max(normalizer, 1e-8)


class MSDSPDDCompositeLoss(nn.Module):
    """
    Full loss requested by the user:
    reconstruction + structure + visual perception + edge + color + physics
    + PG-DSP safety + residual regularization + early route survival.
    """

    def __init__(
        self,
        final_recon_weight: float = 1.0,
        base_recon_weight: float = 0.30,
        ssim_weight: float = 0.20,
        perceptual_weight: float = 0.03,
        edge_weight: float = 0.05,
        color_weight: float = 0.02,
        physics_weight: float = 0.005,
        pgdsp_safety_weight: float = 0.005,
        residual_weight: float = 0.001,
        route_weight: float = 0.001,
        minimum_route_usage: float = 0.03,
    ) -> None:
        super().__init__()
        self.final_recon_weight = float(final_recon_weight)
        self.base_recon_weight = float(base_recon_weight)
        self.ssim_weight = float(ssim_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.edge_weight = float(edge_weight)
        self.color_weight = float(color_weight)
        self.physics_weight = float(physics_weight)
        self.pgdsp_safety_weight = float(pgdsp_safety_weight)
        self.residual_weight = float(residual_weight)
        self.route_weight = float(route_weight)
        self.minimum_route_usage = float(minimum_route_usage)

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.perceptual = MultiScaleVisualPerceptualLoss()
        self.edge = GradientLoss()
        self.color = ColorConsistencyLoss()

    def _route_survival_loss(self, route_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not route_weights:
            raise RuntimeError("Route weights are unavailable; run model forward first.")
        usages = []
        for weights in route_weights.values():
            # [B,5,H,W] -> [5]
            usages.append(weights.mean(dim=(0, 2, 3)))
        usage = torch.stack(usages, dim=0).mean(dim=0)
        return F.relu(self.minimum_route_usage - usage).square().mean()

    def forward(
        self,
        model_outputs,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        model,
        route_regularization_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(model_outputs) < 4:
            raise ValueError("Unexpected model output tuple.")
        final_out = model_outputs[0]
        J_original = model_outputs[2]
        indicator_safe = model_outputs[3]

        refined_t = model.last_refined_transmission
        local_A = model.last_local_atmospheric_light
        route_weights = model.last_route_weights
        if refined_t is None or local_A is None:
            raise RuntimeError("Model auxiliary physical tensors are unavailable.")

        final_recon = self.charbonnier(final_out, clean)
        base_recon = self.charbonnier(J_original, clean)
        ssim_loss = self.ssim(final_out, clean)
        perceptual_loss = self.perceptual(final_out, clean)
        edge_loss = self.edge(final_out, clean)
        color_loss = self.color(final_out, clean)

        re_hazy_final = final_out * refined_t + local_A * (1.0 - refined_t)
        physics_loss = self.charbonnier(re_hazy_final, hazy)

        correction = (final_out - J_original).abs()
        pgdsp_safety_loss = (correction * (1.0 - indicator_safe)).mean()
        residual_loss = correction.mean()
        route_loss = self._route_survival_loss(route_weights)

        route_factor = max(float(route_regularization_factor), 0.0)
        total = (
            self.final_recon_weight * final_recon
            + self.base_recon_weight * base_recon
            + self.ssim_weight * ssim_loss
            + self.perceptual_weight * perceptual_loss
            + self.edge_weight * edge_loss
            + self.color_weight * color_loss
            + self.physics_weight * physics_loss
            + self.pgdsp_safety_weight * pgdsp_safety_loss
            + self.residual_weight * residual_loss
            + self.route_weight * route_factor * route_loss
        )

        terms = {
            "total": total.detach(),
            "final_recon": final_recon.detach(),
            "base_recon": base_recon.detach(),
            "ssim": ssim_loss.detach(),
            "perceptual": perceptual_loss.detach(),
            "edge": edge_loss.detach(),
            "color": color_loss.detach(),
            "physics": physics_loss.detach(),
            "pgdsp_safety": pgdsp_safety_loss.detach(),
            "residual": residual_loss.detach(),
            "route_survival": route_loss.detach(),
        }
        return total, terms
