import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# ================= 1. 配置 =================
CONFIG = {
    "input_haze_dir": r"E:\model_learning\shujuji+fenge\remade_shujuji\val_standard_v5_fixed\images_foggy",
    "output_root": r"E:\model_learning\shujuji+fenge\remade_shujuji\xianyanxinxi_val",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dark_pool_size": 15,
    "gf_radius": 5,
    "gf_eps": 1e-3,
    "laplacian_weight": 0.02,
    "top_k_ratio": 0.001,
}

# ================= 2. 可微导向滤波（与ISP模块100%对齐）=================
def box_filter(x, r):
    """盒式滤波（可微，用平均池化实现）"""
    kernel_size = 2 * r + 1
    return F.avg_pool2d(x, kernel_size, stride=1, padding=r)

def guided_filter(I, p, r, eps):
    """
    可微导向滤波
    I: 引导图 [B,1,H,W] 或 [B,3,H,W]
    p: 输入图 [B,1,H,W]
    返回: q [B,1,H,W]
    """
    if I.shape[1] == 3:
        I_gray = 0.299 * I[:, 0:1] + 0.587 * I[:, 1:2] + 0.114 * I[:, 2:3]
    else:
        I_gray = I

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

# ================= 3. 大气光估计（不可导，用 detach 切断梯度）=================
def estimate_atmospheric_light(img, dark_channel, top_k_ratio=0.001):
    """
    img: [B,3,H,W] 0~1
    dark_channel: [B,1,H,W] (建议传入已 detach 的)
    返回: A [B,3,1,1]
    """
    B, C, H, W = img.shape
    num_pixels = H * W
    k = max(1, int(num_pixels * top_k_ratio))

    dark_flat = dark_channel.view(B, -1)
    _, indices = torch.topk(dark_flat, k, dim=1)

    A_list = []
    for b in range(B):
        img_b = img[b].view(C, -1)
        brightest = img_b[:, indices[b]]  # [C, k]
        A_list.append(brightest.mean(dim=-1))
    A = torch.stack(A_list).view(B, 3, 1, 1)
    
    return torch.clamp(A, 0.6, 1.0)

# ================= 4. 主模块 =================
class PriorEngineV3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else CONFIG
        self.device = self.config["device"]

        # ---------- 可学习融合卷积 ----------
        # ✅ 修复1：通道数匹配 -> 输入2个特征图（暗、亮），输出1个融合透射率图
        self.fusion_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
        
        # ✅ 修复2：权重形状严格匹配 [out_channels, in_channels, 1, 1] 即 [1, 2, 1, 1]
        # 使用 .view() 保证维度绝对安全，并赋予初始值 70% 和 30%
        self.fusion_conv.weight.data = torch.tensor([0.7, 0.3]).view(1, 2, 1, 1)

        # ---------- 固定拉普拉斯核 ----------
        laplacian_kernel = torch.tensor(
            [[[[0., 1., 0.],
               [1., -4., 1.],
               [0., 1., 0.]]]],
            dtype=torch.float32
        ) * 0.5
        self.register_buffer('laplacian_kernel', laplacian_kernel)

        # ToTensor 转换
        self.transform = T.ToTensor()

    def laplacian_smooth(self, x, weight=None):
        """物理扩散平滑，x: [B,1,H,W]"""
        if weight is None:
            weight = self.config["laplacian_weight"]
        smooth = F.conv2d(x, self.laplacian_kernel, padding=1)
        return x + weight * smooth

    def forward(self, raw_rgb):
        """
        输入：raw_rgb [B,3,H,W] 值域 0~1 的雾图
        输出：tau_asm [B,1,H,W], A [B,3,1,1]
        """
        if raw_rgb.ndim != 4:
            raise ValueError(f"输入图像必须是4维张量 [B,3,H,W]，当前形状: {raw_rgb.shape}")
        if raw_rgb.shape[1] != 3:
            raise ValueError(f"输入图像必须是3通道RGB，当前通道数: {raw_rgb.shape[1]}")
        if raw_rgb.min() < -0.1 or raw_rgb.max() > 1.1:
            import warnings
            warnings.warn(f"输入图像数值范围超出 0~1，当前范围: [{raw_rgb.min():.2f}, {raw_rgb.max():.2f}]")
            raw_rgb = torch.clamp(raw_rgb, 0.0, 1.0)

        # ---- 1.1 暗通道 ----
        min_c, _ = torch.min(raw_rgb, dim=1, keepdim=True)  # [B,1,H,W]
        dark_channel = -F.max_pool2d(
            -min_c,
            kernel_size=self.config["dark_pool_size"],
            stride=1,
            padding=self.config["dark_pool_size"] // 2
        )

        # ---- 1.2 亮通道 ----
        max_c, _ = torch.max(raw_rgb, dim=1, keepdim=True)
        bright_channel = F.max_pool2d(
            max_c,
            kernel_size=self.config["dark_pool_size"],
            stride=1,
            padding=self.config["dark_pool_size"] // 2
        )

        # ---- 1.3 自适应融合 ----
        # 暗通道的值 = 1 - 透射率tau，所以先转换成tau再融合
        tau_dark = 1.0 - dark_channel
        # ✅ 修复3：物理规律修正。亮通道值越高代表雾越浓，透射率应该越低。
        tau_bright = 1.0 - bright_channel
        
        tau_cat = torch.cat([tau_dark, tau_bright], dim=1)  # 维度 [B,2,H,W]
        tau_raw = self.fusion_conv(tau_cat)                 # 经过修正后的1x1卷积，输出 [B,1,H,W]

        # ---- 1.4 导向滤波 ----
        # ✅ 修复4：直接传入 raw_rgb 即可，guided_filter 内部已有灰度转换逻辑，避免重复计算
        tau_smooth = guided_filter(raw_rgb, tau_raw, self.config["gf_radius"], self.config["gf_eps"])

        # ---- 1.5 CAP 正则 ----
        tau_asm = self.laplacian_smooth(tau_smooth)

        # 限制透射率范围，保证数值稳定
        tau_asm = torch.clamp(tau_asm, 0.0, 1.0)

        # ---- 1.6 大气光（切断梯度，因为 topk 不可导） ----
        A = estimate_atmospheric_light(raw_rgb, dark_channel.detach(), self.config["top_k_ratio"])

        return tau_asm, A

    # 单张图像预处理+保存功能（无梯度模式）
    def process_and_save(self, img_path, base_name):
        """读取单张图像，计算并保存 tau_asm 和 A"""
        image = Image.open(img_path).convert('RGB')
        raw_rgb = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tau_asm, A = self.forward(raw_rgb)
        tau_tensor = tau_asm.squeeze().cpu()
        A_np = A.squeeze().cpu().numpy()

        os.makedirs(os.path.join(self.config["output_root"], "tau_asm"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_root"], "A"), exist_ok=True)
        torch.save(tau_tensor, os.path.join(self.config["output_root"], "tau_asm", f"{base_name}.pt"))
        with open(os.path.join(self.config["output_root"], "A", f"{base_name}.json"), 'w') as f:
            json.dump({"A": A_np.tolist()}, f, indent=2)
        return tau_tensor, A_np

    # 批量处理功能
    def process_batch(self):
        """批量处理输入目录下的所有雾图"""
        valid_exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        img_names = [f for f in os.listdir(self.config["input_haze_dir"]) if any(f.endswith(ext) for ext in valid_exts)]
        
        print(f"✅ 开始批量处理 {len(img_names)} 张图像...")
        for img_name in tqdm(img_names):
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.config["input_haze_dir"], img_name)
            self.process_and_save(img_path, base_name)
        
        print(f"🎉 批量处理完成！结果已保存至: {self.config['output_root']}")


if __name__ == "__main__":
    # 批量处理示例
    prior_engine = PriorEngineV3()
    # 将模型移动到配置好的设备上
    prior_engine.to(prior_engine.device) 
    prior_engine.process_batch()