# -*- coding: utf-8 -*-
"""
MSDSPDDDehazer Final
====================

Final architecture requested by the user:

1. PriorEngineV3 estimates initial transmission t0 and global atmospheric light A0.
2. DynamicPixelwiseISP refines them to t1 and a spatially adaptive local
   atmospheric-light map A1, and produces five operator-response images.
3. The five images share one EfficientViT-B1 encoder and independently produce
   H/2, H/4, H/8 and H/16 features.
4. Route-specific adapters and explicit operator-response features preserve the
   functional differences among Dehaze / Denoise / WB / Sharpen / Raw.
5. t1/A1 guide five-route fusion at every scale.
6. H/2 and H/4 use local-detail attention; H/8 uses residual local modeling;
   H/16 uses one VMamba/VSSBlock for global haze-field modeling.
7. A multi-scale U-Net decoder reconstructs J_original.
8. A transmission-guided Raw Detail Gate injects full-resolution raw texture
   before J_original is produced.
9. PG-DSP produces a residual-haze / structural-safety indicator.
10. A PG-DSP-guided low-amplitude Residual Refiner produces J_final.

The default forward contract is kept compatible with the old training code:

    final_out, tau_ode, J_original, indicator_safe,
    ops_imgs_5, tau_asm, soft_A = model(raw_img)

Important semantic note:
The legacy variable name ``tau`` is retained for compatibility, but the code
uses it as a transmission map t in the atmospheric-scattering equation.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_prior import PriorEngineV3
from isp import DynamicPixelwiseISP

try:
    from .efficientvit.backbone import efficientvit_backbone_b1
    _EFFICIENTVIT_IMPORT_ERROR = None
except Exception as exc:  # allows syntax/smoke testing without the external package
    efficientvit_backbone_b1 = None
    _EFFICIENTVIT_IMPORT_ERROR = exc

try:
    from .vmamba import VSSBlock
    _VMAMBA_IMPORT_ERROR = None
except Exception as exc:
    VSSBlock = None
    _VMAMBA_IMPORT_ERROR = exc


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
            # V1 checkpoint is created by this project and contains training
            # metadata (including NumPy scalar objects). PyTorch 2.6 defaults
            # torch.load() to weights_only=True, which rejects those objects.
            # Use weights_only=False only for a checkpoint you created/trust.
            try:
                checkpoint = torch.load(
                    checkpoint,
                    map_location=map_location,
                    weights_only=False,
                )
            except TypeError:
                # Compatibility with older PyTorch releases that do not expose
                # the weights_only argument.
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
