# -*- coding: utf-8 -*-
"""Composite training losses for MSDSPDDDehazer Final."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
