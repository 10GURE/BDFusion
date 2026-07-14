#!/usr/bin/env python3
"""Evaluator使用示例 - 简化版"""
import numpy as np
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from utils.Evaluator import Evaluator

print("=" * 60)
print("Evaluator.py 输入输出演示")
print("=" * 60)
print()

# 创建模拟图像数据
print("【输入数据】")
print("-" * 60)
fusion_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
rgb_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
ir_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

print(f"融合图像: shape={fusion_img.shape}, dtype={fusion_img.dtype}")
print(f"RGB源图:  shape={rgb_img.shape}, dtype={rgb_img.dtype}")
print(f"IR源图:   shape={ir_img.shape}, dtype={ir_img.dtype}")
print()

# 单图像指标
print("【单图像指标】(无需参考图像)")
print("-" * 60)
en = Evaluator.EN(fusion_img)
sd = Evaluator.SD(fusion_img)
sf = Evaluator.SF(fusion_img)
ag = Evaluator.AG(fusion_img)
print(f"EN (熵):       {en:8.4f} - 信息丰富度")
print(f"SD (标准差):   {sd:8.4f} - 对比度")
print(f"SF (空间频率): {sf:8.4f} - 清晰度")
print(f"AG (平均梯度): {ag:8.4f} - 边缘质量")
print()

# 融合指标
print("【图像融合指标】(需要三张图像)")
print("-" * 60)
mi = Evaluator.MI(fusion_img, rgb_img, ir_img)
mse = Evaluator.MSE(fusion_img, rgb_img, ir_img)
cc = Evaluator.CC(fusion_img, rgb_img, ir_img)
psnr = Evaluator.PSNR(fusion_img, rgb_img, ir_img)
ssim = Evaluator.SSIM(fusion_img, rgb_img, ir_img, data_range=255.0)

print(f"MI (互信息):      {mi:8.4f} - 信息保留")
print(f"MSE (均方误差):   {mse:8.4f} - 融合误差")
print(f"CC (相关系数):    {cc:8.4f} - 相关性")
print(f"PSNR (峰值信噪比): {psnr:8.4f} dB - 信号质量")
print(f"SSIM (结构相似性): {ssim:8.4f} - 结构保留")
print()

print("【输出说明】")
print("-" * 60)
print(f"• 所有方法返回类型: {type(en)}")
print(f"• 值越大越好: EN, SD, SF, AG, MI, CC, PSNR, SSIM")
print(f"• 值越小越好: MSE")
print()

print("=" * 60)
print("✓ 演示完成！详细文档请查看 EVALUATOR_README.md")
print("=" * 60)
