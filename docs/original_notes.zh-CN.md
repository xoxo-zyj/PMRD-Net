# MSDSPDD Final 完整模型

## 最终保留的三部分

- `RawDetailGate`：位于 `J_original` 之前，只控制原图浅层特征进入解码器。
- `PGDSPSubmodule`：位于 `J_original` 之后，生成残雾/结构安全指示图，不直接改RGB。
- `ResidualRefiner`：最后一步，使用 PG-DSP 指示与物理误差执行低幅度RGB残差修正。

三者已经全部写入 `future_process/model.py`。

## 其他已经完成的结构

- 初始 `t0/A0` 由完整 `PriorEngineV3` 获得。
- ISP 输出修正透射率 `t1` 和空间自适应局部大气光图 `A1`。
- 五路固定为 Dehaze / Denoise / White Balance / Sharpen / Raw。
- 五路共享一套 EfficientViT-B1，分别提取 H/2、H/4、H/8、H/16 特征。
- Route Adapter 与 `Xi-Raw` 算子响应特征均已加入。
- `t1/A1` 在四个尺度引导五路软路由融合。
- H/2、H/4 使用局部细节注意力；H/16 使用一个 VMamba。
- 多尺度 U-Net 使用 F16、F8、F4、F2 全部跳跃特征。
- 锐化分支已改成“边缘+噪声+透射率+学习路由”联合控制，不再是雾越浓锐化越强。

## 损失

`future_process/losses.py` 中提供 `MSDSPDDCompositeLoss`，包括：

- Final Charbonnier 主重建损失；
- J_original 基础恢复监督；
- SSIM；
- 多尺度视觉感知损失；
- Edge/Gradient；
- Color consistency；
- 物理重雾一致性；
- PG-DSP安全约束；
- 残差幅度约束；
- 训练前期的路径存活约束。

## 替换

建议先备份原工程，再用本文件夹覆盖对应文件。核心改动文件：

```text
future_process/model.py
future_process/losses.py
future_process/__init__.py
isp/image_process_tools.py
```

## 前向兼容

默认返回顺序未改变：

```python
final_out, tau_ode, J_original, indicator_safe, \
ops_imgs_5, tau_asm, soft_A = model(hazy)
```

其中 `tau_ode` 现在使用 ISP 修正后的透射率 `t1`。

## 正式训练前检查

```bash
cd MSDSPDD_Final_Full
python smoke_test_final.py
```

正式环境必须确认：

```text
backbone fallback: False
Mamba enabled: True
```

如果 backbone fallback 为 True，说明 EfficientViT 依赖没有正确安装；备用主干只能用于形状测试，不能用于论文实验。

## 两阶段训练

第一阶段冻结 PriorEngine、ISP、EfficientViT，训练新编码适配、融合、局部模块、解码器和精修器；第二阶段解冻 ISP、EfficientViT 后两级和先验融合层，以更低学习率端到端联合微调。
