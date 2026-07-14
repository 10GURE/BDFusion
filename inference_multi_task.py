#!/usr/bin/env python3
"""
多任务类型推理脚本
自动扫描测试集目录下的所有任务类型，分别运行推理并将结果按任务分组输出。

数据目录结构:
    <test_dir>/<任务类型>/ir/      原始红外图像
    <test_dir>/<任务类型>/vi/      原始可见光图像（退化RGB）
    <test_dir>/<任务类型>/gt/      参考GT清晰图像

输出目录结构:
    <output_dir>/<任务类型>/       各任务的融合结果
"""

import argparse
import gc
import os
import sys
import time
import random
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm

from ldm.util import instantiate_from_config


# ==================== 图像处理工具 ====================

def crop_to_size(img, crop):
    h, w = img.shape[:2]
    img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    return img


def format_process(img):
    img = (img / 127.5 - 1.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    return torch.from_numpy(img)


def get_img(img_path, device=None):
    img = Image.open(img_path)
    img = np.array(img).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    min_size = min(img.shape[0], img.shape[1])
    img = crop_to_size(img, min_size)
    img = cv2.resize(img, (256, 256))
    img_t = format_process(img)
    img_t = img_t.to(device=device)
    return img_t


def make_batch(rgb_path, infrared_path, device):
    return {"rgb": get_img(rgb_path, device), "infrared": get_img(infrared_path, device)}


def post_process(img):
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) / 2.0)[0]
    img = img.permute(1, 2, 0).numpy() * 255
    return img.astype(np.uint8)


# ==================== 推理模型 ====================

class Inferen(nn.Module):
    def __init__(self, model, steps):
        super(Inferen, self).__init__()
        self.model = model
        self.steps = steps

    def forward(self, batch):
        c1 = self.model.get_learned_conditioning(batch["rgb"])
        c2 = self.model.get_learned_conditioning(batch["infrared"])
        samples_ddim, _ = self.model.sample_log(
            cond1=c1, cond2=c2, batch_size=1, ddim=True,
            ddim_steps=self.steps, eta=1.0
        )
        return samples_ddim


# ==================== 数据目录扫描 ====================

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def _collect_image_paths(folder: str) -> List[str]:
    """收集目录中的所有有效图像路径"""
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f.lower())[1] in VALID_EXTS]
    return sorted(files)


def _pair_images(vi_dir: str, ir_dir: str) -> Tuple[List[str], List[str]]:
    """按文件名配对 VI 和 IR 图像"""
    vi_files = _collect_image_paths(vi_dir)
    ir_files = _collect_image_paths(ir_dir)
    if not vi_files:
        raise RuntimeError(f"VI 目录无有效图片: {vi_dir}")
    if not ir_files:
        raise RuntimeError(f"IR 目录无有效图片: {ir_dir}")

    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    vi_map = {stem(p): p for p in vi_files}
    ir_map = {stem(p): p for p in ir_files}
    common = sorted(set(vi_map) & set(ir_map))
    if not common:
        raise RuntimeError(f"未找到任何同名文件对 (vi: {vi_dir}, ir: {ir_dir})")
    return [vi_map[k] for k in common], [ir_map[k] for k in common]


def discover_tasks(test_dir: str) -> Dict[str, dict]:
    """
    自动发现测试集目录下的所有任务类型。

    返回: {任务名: {"vi_dir": ..., "ir_dir": ..., "gt_dir": ...}, ...}
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    tasks = {}
    for entry in sorted(os.listdir(test_dir)):
        task_path = os.path.join(test_dir, entry)
        if not os.path.isdir(task_path):
            continue

        vi_dir = os.path.join(task_path, 'vi')
        ir_dir = os.path.join(task_path, 'ir')
        gt_dir = os.path.join(task_path, 'gt')

        # 至少需要 vi 和 ir 两个子目录
        if os.path.isdir(vi_dir) and os.path.isdir(ir_dir):
            tasks[entry] = {
                "vi_dir": vi_dir,
                "ir_dir": ir_dir,
                "gt_dir": gt_dir if os.path.isdir(gt_dir) else None,
            }

    if not tasks:
        raise RuntimeError(f"未在 {test_dir} 下发现任何有效的任务目录 "
                           f"(需要包含 ir/ 和 vi/ 子目录)")
    return tasks


# ==================== 主流程 ====================

def load_model(config_path: str, ckpt_path: str, device: torch.device):
    """加载模型（只加载一次，所有任务共用）"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    # 直接加载到 GPU 以避免 CPU 内存峰值（checkpoint 8.6GB，系统 RAM 仅 15GB）
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.to(device)
    model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return model


def run_task_inference(model, inference_model, task_name: str, vi_paths: List[str],
                       ir_paths: List[str], out_dir: str, device: torch.device):
    """对单个任务类型执行推理"""
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        with model.ema_scope():
            for vi_p, ir_p in tqdm(zip(vi_paths, ir_paths), total=len(vi_paths),
                                   desc=f"  [{task_name}] 推理"):
                batch = make_batch(vi_p, ir_p, device=device)
                samples_ddim = inference_model(batch)
                decoded = model.decode_first_stage(samples_ddim)
                output_img = post_process(decoded)

                out_name = os.path.basename(vi_p)
                outpath = os.path.join(out_dir, out_name)
                Image.fromarray(output_img).save(outpath)
                count += 1

                # 释放中间张量，防止显存累积
                del batch, samples_ddim, decoded

            # 每个任务结束后清理一次显存
            torch.cuda.empty_cache()

    return count


def main():
    parser = argparse.ArgumentParser(
        description="多任务类型推理脚本 — 自动扫描任务目录，分组保存融合结果"
    )
    parser.add_argument("--test_dir", default="./data/test_imgs", type=str,
                        help="测试集根目录，包含各任务类型子目录 (默认: ./data/test_imgs)")
    parser.add_argument("--output_dir", default="./experiments/output", type=str,
                        help="输出根目录，结果按任务类型分组 (默认: ./experiments/output)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_PROJ_ROOT, "informationSynthesis", "Image2Image_AllInOne.yaml"),
                        help="模型配置文件")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(_PROJ_ROOT, "checkpoint", "last.ckpt"),
                        help="模型权重路径")
    parser.add_argument("--steps", type=int, default=200,
                        help="DDIM 采样步数 (默认: 200)")
    parser.add_argument("--seed", type=int, default=4,
                        help="随机种子 (默认: 4)")
    parser.add_argument("--tasks", type=str, nargs='*', default=None,
                        help="指定要推理的任务类型名称，不指定则处理所有任务")

    opt = parser.parse_args()

    # ── 固定随机种子 ──
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # ── 发现任务 ──
    print("=" * 80)
    print("多任务类型推理系统")
    print("=" * 80)
    print(f"测试集目录: {os.path.abspath(opt.test_dir)}")
    print(f"输出目录:   {os.path.abspath(opt.output_dir)}")
    print()

    all_tasks = discover_tasks(opt.test_dir)

    # 如果用户指定了任务子集，则过滤
    if opt.tasks:
        filtered = {}
        for t in opt.tasks:
            if t in all_tasks:
                filtered[t] = all_tasks[t]
            else:
                print(f"[警告] 指定的任务 '{t}' 不存在，已跳过。可用任务: {list(all_tasks.keys())}")
        all_tasks = filtered
        if not all_tasks:
            print("[错误] 没有有效的任务可执行。")
            sys.exit(1)

    print(f"发现 {len(all_tasks)} 个任务类型:")
    for name, info in all_tasks.items():
        vi_count = len(_collect_image_paths(info["vi_dir"]))
        gt_flag = "有GT" if info["gt_dir"] else "无GT"
        print(f"  - {name:<20s}  {vi_count:>4d} 张图像  ({gt_flag})")
    print()

    # ── 加载模型（只加载一次）──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 80}")
    print(f"设备信息:")
    print(f"  使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU型号:  {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'=' * 80}\n")

    print("加载模型...")
    model = load_model(opt.config, opt.checkpoint, device)
    inference_model = Inferen(model, opt.steps)
    print("模型加载完成。\n")

    # ── 逐任务推理 ──
    start_time = time.time()
    total_images = 0
    summary = []

    for task_name, task_info in all_tasks.items():
        print("-" * 60)
        print(f"任务: {task_name}")
        print("-" * 60)

        try:
            vi_paths, ir_paths = _pair_images(task_info["vi_dir"], task_info["ir_dir"])
        except RuntimeError as e:
            print(f"  [跳过] {e}")
            summary.append((task_name, 0, "跳过（配对失败）"))
            continue

        print(f"  配对成功: {len(vi_paths)} 组图像")

        out_dir = os.path.join(opt.output_dir, task_name)
        count = run_task_inference(model, inference_model, task_name,
                                   vi_paths, ir_paths, out_dir, device)

        print(f"  ✓ 完成，结果保存至: {out_dir}")
        total_images += count
        summary.append((task_name, count, "完成"))
        gc.collect()
        torch.cuda.empty_cache()
        print()

    # ── 汇总 ──
    elapsed = time.time() - start_time
    print("=" * 80)
    print("推理汇总")
    print("=" * 80)
    print(f"{'任务类型':<20s}  {'图像数':>6s}  {'状态':<12s}")
    print("-" * 50)
    for name, cnt, status in summary:
        print(f"{name:<20s}  {cnt:>6d}  {status:<12s}")
    print("-" * 50)
    print(f"{'合计':<20s}  {total_images:>6d}")
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    if total_images > 0:
        print(f"平均每张: {elapsed / total_images:.2f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
