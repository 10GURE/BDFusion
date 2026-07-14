#!/usr/bin/env python3
"""
VIIS Progressive Distillation: 教 Lite UNet 用 N 步模仿 teacher 的 2N 步。

阶段 1 (TARGET_STEPS   = 2): student 2-step ≈ teacher 5-step
阶段 2 (TARGET_STEPS   = 2): student 1-step ≈ student 2-step

每阶段 30 epoch。改 TARGET_STEPS 常量即可切换。
"""

import os, sys, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import extract_into_tensor

# ── 配置 ──
STUDENT_CONFIG = "./informationSynthesis/Image2Image_Lite.yaml"
TEACHER_CONFIG = "./informationSynthesis/Image2Image.yaml"
TEACHER_CKPT   = "./checkpoint/last.ckpt"
OUTPUT_DIR     = "./checkpoint/distill"
DISTILL_EPOCHS = 30
DISTILL_LR     = 2.0e-6
DISTILL_LAMBDA = 0.5
BATCH_SIZE     = 4
TARGET_STEPS   = 2       # student 用几步模仿 teacher


def _ddim_step(model, x_t, t_start, t_end, c1, c2):
    """DDIM deterministic step: x_t → x_{t_end} (eta=0)."""
    acp = model.alphas_cumprod.to(x_t.device)
    b, device = x_t.shape[0], x_t.device
    ts = torch.full((b,), t_start, device=device, dtype=torch.long)
    eps = model.apply_model(x_t, ts, c1, c2)
    abar = extract_into_tensor(acp, ts, x_t.shape)
    abar_end = extract_into_tensor(acp, torch.full((b,), t_end, device=device, dtype=torch.long), x_t.shape)
    x0 = (x_t - (1.0 - abar).sqrt() * eps) / abar.sqrt().clamp(min=1e-10)
    return abar_end.sqrt() * x0 + (1.0 - abar_end).sqrt() * eps


class ProgressiveDistillModel(pl.LightningModule):
    def __init__(self, student, teacher, target_steps=2, teacher_steps=5, distill_lambda=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.target_steps = target_steps
        self.teacher_steps = teacher_steps
        self.distill_lambda = distill_lambda
        self.first_stage_key = student.first_stage_key
        self.num_timesteps = student.num_timesteps
        self.use_ema = student.use_ema

    def training_step(self, batch, batch_idx):
        inputs = self.student.get_input(batch, self.first_stage_key)
        if len(inputs) == 3:
            x, c1, c2 = inputs
        elif len(inputs) == 4:
            x, c1, c2, _ = inputs
        else:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        b, dev, T = x.shape[0], self.device, self.num_timesteps
        step_span = T // self.teacher_steps
        t_start = torch.randint(self.target_steps * step_span, T, (b,), device=dev).long()
        noise = torch.randn_like(x)
        x_t = self.student.q_sample(x_start=x, t=t_start, noise=noise)

        # Build timestep sequences
        # 为每个 batch item 独立生成 timestep 序列
        ts_student = torch.stack([
            torch.linspace(t_start[i], 0, self.target_steps + 1, device=dev)
            for i in range(b)]).long()
        ts_teacher = torch.stack([
            torch.linspace(t_start[i], 0, self.teacher_steps + 1, device=dev)
            for i in range(b)]).long()

        # Teacher: run teacher_steps DDIM
        with torch.no_grad():
            xt = x_t.clone()
            for i in range(self.teacher_steps):
                xt = _ddim_step(self.teacher, xt, ts_teacher[0, i].item(),
                                ts_teacher[0, i + 1].item(), c1, c2)

        # Student: run target_steps DDIM → should match teacher's output
        xs = x_t.clone()
        for i in range(self.target_steps):
            xs = _ddim_step(self.student, xs, ts_student[0, i].item(),
                            ts_student[0, i + 1].item(), c1, c2)

        loss_distill = F.mse_loss(xs, xt)
        eps_pred = self.student.apply_model(x_t, t_start, c1, c2)
        loss_denoise = F.mse_loss(eps_pred, noise)
        loss = loss_denoise + self.distill_lambda * loss_distill

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/loss_distill", loss_distill, prog_bar=True, on_step=True)
        self.log("global_step", float(self.global_step), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.student.get_input(batch, self.first_stage_key)
        if len(inputs) == 3: x, c1, c2 = inputs
        elif len(inputs) == 4: x, c1, c2, _ = inputs
        else: return
        b, dev, T = x.shape[0], self.device, self.num_timesteps
        step_span = T // self.teacher_steps
        t_start = torch.randint(self.target_steps * step_span, T, (b,), device=dev).long()
        noise = torch.randn_like(x)
        x_t = self.student.q_sample(x_start=x, t=t_start, noise=noise)
        ts_s = torch.stack([
            torch.linspace(t_start[i], 0, self.target_steps + 1, device=dev)
            for i in range(b)]).long()
        ts_t = torch.stack([
            torch.linspace(t_start[i], 0, self.teacher_steps + 1, device=dev)
            for i in range(b)]).long()
        with torch.no_grad():
            xt = x_t.clone()
            for i in range(self.teacher_steps):
                xt = _ddim_step(self.teacher, xt, ts_t[0, i].item(), ts_t[0, i + 1].item(), c1, c2)
            xs = x_t.clone()
            for i in range(self.target_steps):
                xs = _ddim_step(self.student, xs, ts_s[0, i].item(), ts_s[0, i + 1].item(), c1, c2)
        self.log("val/loss", F.mse_loss(xs, xt), prog_bar=True, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.student.use_ema:
            self.student.model_ema(self.student.model)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.student.model.parameters(), lr=DISTILL_LR)
        return opt


# ── 加载 ──
def load_teacher():
    config = OmegaConf.load(TEACHER_CONFIG)
    teacher = instantiate_from_config(config.model)
    state = torch.load(TEACHER_CKPT, map_location="cpu")
    if "state_dict" in state: state = state["state_dict"]
    remap = {}
    for k, v in list(state.items()):
        if '.to_k.' in k:
            nk = k.replace('.to_k.', '.to_v.')
            if nk not in state: remap[nk] = v.clone()
    state.update(remap)
    teacher.load_state_dict(state, strict=False)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()
    return teacher


def load_student():
    config = OmegaConf.load(STUDENT_CONFIG)
    student = instantiate_from_config(config.model)
    t_state = torch.load(TEACHER_CKPT, map_location="cpu")
    if "state_dict" in t_state: t_state = t_state["state_dict"]
    remap = {}
    for k, v in list(t_state.items()):
        if '.to_k.' in k:
            nk = k.replace('.to_k.', '.to_v.')
            if nk not in t_state: remap[nk] = v.clone()
    t_state.update(remap)
    for k, p in student.state_dict().items():
        if k.startswith('first_stage_model.') and k in t_state and t_state[k].shape == p.shape:
            p.copy_(t_state[k])
    for n, p in student.named_parameters():
        if n.startswith('first_stage_model.') or n.startswith('cond_stage_model.'):
            p.requires_grad = False
    return student


def run_distill():
    print(f"=== Progressive Distill: student {TARGET_STEPS}-step, teacher 5-step ===")
    seed_everything(42)
    teacher = load_teacher().cuda()
    student = load_student().cuda()
    distill_model = ProgressiveDistillModel(
        student, teacher, target_steps=TARGET_STEPS,
        teacher_steps=5, distill_lambda=DISTILL_LAMBDA
    )
    data_config = OmegaConf.load(STUDENT_CONFIG).data
    data_module = instantiate_from_config(data_config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename=f"prog-{TARGET_STEPS}step-" + "{epoch:02d}-{val/loss:.4f}",
        save_top_k=3, monitor="val/loss", mode="min", every_n_epochs=1,
    )
    trainer = Trainer(
        max_epochs=DISTILL_EPOCHS, accelerator="gpu", devices=1,
        precision=32, benchmark=True,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        check_val_every_n_epoch=1, log_every_n_steps=50,
        default_root_dir=OUTPUT_DIR,
    )
    trainer.fit(distill_model, datamodule=data_module)
    print(f"[完成] 最优: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    run_distill()
