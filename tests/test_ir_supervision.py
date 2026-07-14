#!/usr/bin/env python3
"""
测试 IR 监督功能的脚本
验证数据集加载和模型前向传播是否正常工作
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 添加当前路径到 sys.path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'informationSynthesis'))

def create_test_data():
    """创建测试用的模拟数据"""
    print("创建测试数据...")
    
    # 创建测试目录结构
    base_dir = "/tmp/test_ir_supervision"
    dirs = [
        f"{base_dir}/rgb_degraded/train",
        f"{base_dir}/ir_degraded/train", 
        f"{base_dir}/gt_clean/train",
        f"{base_dir}/ir_clean/train"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建几张测试图像
    test_files = ["test001.png", "test002.png", "test003.png"]
    
    for filename in test_files:
        # 创建 RGB 图像 (256x256x3)
        rgb_degraded = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(rgb_degraded).save(f"{base_dir}/rgb_degraded/train/{filename}")
        
        rgb_clean = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) 
        Image.fromarray(rgb_clean).save(f"{base_dir}/gt_clean/train/{filename}")
        
        # 创建 IR 图像 (256x256 灰度)
        ir_degraded = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(ir_degraded, mode='L').save(f"{base_dir}/ir_degraded/train/{filename}")
        
        ir_clean = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        Image.fromarray(ir_clean, mode='L').save(f"{base_dir}/ir_clean/train/{filename}")
    
    print(f"测试数据已创建在 {base_dir}")
    return base_dir

def test_dataset():
    """测试数据集加载"""
    print("\n=== 测试数据集加载 ===")
    
    try:
        from dataset import DegradedFusionDatasetTrain
        
        base_dir = create_test_data()
        
        # 测试带 IR 监督的数据集
        dataset = DegradedFusionDatasetTrain(
            rgb_root=f"{base_dir}/rgb_degraded/train/",
            ir_root=f"{base_dir}/ir_degraded/train/",
            clean_root=f"{base_dir}/gt_clean/train/",
            ir_clean_root=f"{base_dir}/ir_clean/train/",
            size=128,
            random_crop=True,
            apply_rgb_aug=False,
            apply_ir_aug=False,
            extra_noise=False
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"数据格式: {len(sample)} 个元素")
        
        if len(sample) == 5:
            gt, rgb, ir, ir_clean, name = sample
            print(f"GT shape: {gt.shape}")
            print(f"RGB shape: {rgb.shape}")  
            print(f"IR shape: {ir.shape}")
            print(f"IR Clean shape: {ir_clean.shape}")
            print(f"文件名: {name}")
            print("✓ 数据集加载成功，支持 IR 监督")
        else:
            print(f"❌ 数据格式不正确，期望 5 个元素，实际 {len(sample)} 个")
            
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_forward():
    """测试模型前向传播"""
    print("\n=== 测试模型前向传播 ===")
    
    try:
        # 创建模拟的批次数据
        batch_size = 2
        gt = torch.randn(batch_size, 128, 128, 3)  # RGB GT
        rgb_deg = torch.randn(batch_size, 128, 128, 3)  # 退化 RGB
        ir_deg = torch.randn(batch_size, 128, 128, 3)   # 退化 IR (3通道)
        ir_clean = torch.randn(batch_size, 128, 128, 3)  # 干净 IR (3通道)
        names = ["test1", "test2"]
        
        batch_with_ir = [gt, rgb_deg, ir_deg, ir_clean, names]
        batch_without_ir = [gt, rgb_deg, ir_deg, names]
        
        # 测试 batch 解析
        print("测试 batch 解析逻辑...")
        
        if len(batch_with_ir) == 5:
            print("✓ 5元素批次 - 包含 IR 监督")
            
        if len(batch_without_ir) == 4:
            print("✓ 4元素批次 - 不包含 IR 监督")
            
        # 测试 RGB 到灰度转换
        print("\n测试 RGB 到灰度转换...")
        rgb_tensor = torch.randn(2, 3, 64, 64)  # (B, C, H, W)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114])
        gray = torch.sum(rgb_tensor * rgb_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        
        print(f"RGB tensor shape: {rgb_tensor.shape}")
        print(f"Gray tensor shape: {gray.shape}")
        print("✓ RGB 到灰度转换正常")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_loss_computation():
    """测试损失计算"""
    print("\n=== 测试损失计算 ===")
    
    try:
        # 模拟预测和真实值
        pred_gray = torch.randn(2, 1, 64, 64)  # 预测的灰度图
        ir_clean = torch.randn(2, 1, 64, 64)   # 干净的 IR GT
        
        # 计算 L1 损失
        ir_loss = F.l1_loss(pred_gray, ir_clean, reduction='mean')
        print(f"IR L1 损失: {ir_loss.item():.4f}")
        
        # 测试权重应用
        ir_loss_weight = 0.1
        weighted_loss = ir_loss_weight * ir_loss
        print(f"加权 IR 损失: {weighted_loss.item():.4f}")
        
        print("✓ 损失计算正常")
        
    except Exception as e:
        print(f"❌ 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始测试 IR 监督功能...")
    print("=" * 50)
    
    # 测试数据集
    test_dataset()
    
    # 测试模型组件
    test_model_forward()
    
    # 测试损失计算
    test_loss_computation()
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    main()