# future_process/generate_priop/__init__.py
__all__ = [
    "PriorEngineV3",    # V3.4 先验提取引擎（支持像素级大气光）
    "CONFIG"            # 先验提取全局配置
]

from .generate_prior import PriorEngineV3, CONFIG