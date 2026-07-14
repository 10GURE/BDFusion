#!/usr/bin/env python3
"""
VIIS Lite 微调脚本
从原始 checkpoint 初始化轻量模型，微调恢复质量。
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# ── 常量 ──
LITE_CONFIG = "./informationSynthesis/Image2Image_Lite.yaml"
TEACHER_CKPT = "./checkpoint/last.ckpt"
OUTPUT_DIR = "./checkpoint/lite"
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1.0e-6
BATCH_SIZE = 4  # 3090 24GB 保守值


def load_teacher_weights(student, teacher_ckpt_path):
    """只从旧 checkpoint 复制 VAE 和 cond_stage（冻结模块），UNet/encoder 随机初始化。"""
    state = torch.load(teacher_ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    student_state = student.state_dict()
    loaded = 0
    
    # 只复制 first_stage_model (VAE) 和 cond_stage_model
    for key, param in student_state.items():
        for prefix in ["first_stage_model.", "cond_stage_model."]:
            if key.startswith(prefix):
                src_key = key  # cond_stage_model is first_stage_model (same)
                if src_key in state and state[src_key].shape == param.shape:
                    param.copy_(state[src_key])
                    loaded += 1
                    break
                # also try model. prefix
                src_key2 = "model." + key
                if src_key2 in state and state[src_key2].shape == param.shape:
                    param.copy_(state[src_key2])
                    loaded += 1
                    break

    skipped = len(student_state) - loaded
    print(f"[初始化] 复制了 {loaded} 个参数（VAE/cond_stage），{skipped} 个随机初始化。")
    return student


def run_finetune():
    print("=" * 60)
    print("  VIIS Lite 微调")
    print(f"  配置: {LITE_CONFIG}")
    print(f"  教师 ckpt: {TEACHER_CKPT}")
    print(f"  Epochs: {FINETUNE_EPOCHS}  LR: {FINETUNE_LR}")
    print("=" * 60)

    seed_everything(42)

    # 1. 创建 Lite 模型
    config = OmegaConf.load(LITE_CONFIG)
    model = instantiate_from_config(config.model)
    
    # 2. 从旧 checkpoint 加载可匹配的权重
    model = load_teacher_weights(model, TEACHER_CKPT)

    # 3. 确保 VAE 和 cond_stage 冻结（从 teacher 继承的完整权重）
    if hasattr(model, 'first_stage_model'):
        for p in model.first_stage_model.parameters():
            p.requires_grad = False
    if hasattr(model, 'cond_stage_model'):
        for p in model.cond_stage_model.parameters():
            p.requires_grad = False

    # 4. 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[参数] 可训练: {trainable/1e6:.1f}M / 总计: {total/1e6:.1f}M")

    # 5. 设置学习率
    model.learning_rate = FINETUNE_LR

    # 6. 数据模块
    data_module = instantiate_from_config(config.data)

    # 7. Checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="lite-{epoch:02d}-{val/loss_simple_ema:.4f}",
        save_top_k=3,
        monitor="val/loss_simple_ema",
        mode="min",
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 8. 训练
    trainer = Trainer(
        max_epochs=FINETUNE_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision=32,
        benchmark=True,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        default_root_dir=OUTPUT_DIR,
    )

    print(f"[训练] 开始 {FINETUNE_EPOCHS} epoch 微调...")
    trainer.fit(model, datamodule=data_module)

    print(f"[完成] 最优模型: {checkpoint_callback.best_model_path}")
    print(f"[完成] 最优 val_loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    run_finetune()
