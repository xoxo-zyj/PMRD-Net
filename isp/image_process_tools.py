import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


# ====================== 可微导向滤波（与先验模块100%对齐）======================
def box_filter(x, r):
    """盒式滤波（可微，用平均池化实现）"""
    kernel_size = 2 * r + 1
    return F.avg_pool2d(x, kernel_size, stride=1, padding=r)


def guided_filter(I, p, r, eps=1e-3):
    """
    可微导向滤波
    I: 引导图 [B,1,H,W] 或 [B,3,H,W]
    p: 输入图 [B,C,H,W]
    r: 半径
    eps: 正则化系数
    返回: q [B,C,H,W]
    """
    B, C, H, W = p.shape

    # 输入形状校验
    if I.ndim != 4:
        raise ValueError(f"引导图I必须是4维张量 [B,C,H,W]，当前形状: {I.shape}")
    if I.shape[0] != B or I.shape[2:] != (H, W):
        raise ValueError(f"引导图I的batch和空间尺寸必须与输入p一致，I: {I.shape}, p: {p.shape}")

    # 引导图转灰度（统一处理，避免重复计算）
    if I.shape[1] == 3:
        I_gray = 0.299 * I[:, 0:1] + 0.587 * I[:, 1:2] + 0.114 * I[:, 2:3]
    elif I.shape[1] == 1:
        I_gray = I
    else:
        raise ValueError(f"引导图I只能是1通道或3通道，当前通道数: {I.shape[1]}")

    mean_I = box_filter(I_gray, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I_gray * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I_gray * I_gray, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I_gray + mean_b
    return q


# ====================== 空间自适应物理算子池 ======================
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