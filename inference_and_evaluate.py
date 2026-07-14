#!/usr/bin/env python3
"""
推理+评估一体化脚本
功能：
1. 运行图像融合推理
2. 自动评估所有生成的融合图像
3. 输出详细的评估报告
"""

import argparse
import os
import sys
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from ldm.util import instantiate_from_config
import random
import cv2
from typing import List
import csv
import time

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# 导入评估器
sys.path.insert(0, _PROJ_ROOT)
from utils.Evaluator import Evaluator, image_read_cv2

# 推理相关函数保持与 inference.py 一致
def crop_to_size(img, crop):
    h, w, = img.shape[:2]
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


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def create_batch_grids(rgb_paths, fusion_images, batch_size=4):
    """
    将RGB输入和Fusion结果分别拼接成4x1网格
    返回: [(rgb_grid, fusion_grid), ...]
    """
    grids = []
    num_batches = (len(rgb_paths) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(rgb_paths))
        
        # 当前批次的图像
        batch_rgb_paths = rgb_paths[start_idx:end_idx]
        batch_fusion_imgs = fusion_images[start_idx:end_idx]
        
        # 读取RGB图像并统一尺寸
        rgb_imgs = []
        target_h, target_w = 256, 256  # 默认尺寸
        
        for rgb_path in batch_rgb_paths:
            img = cv2.imread(rgb_path)
            if img is not None:
                img = cv2.resize(img, (target_w, target_h))
                # OpenCV读取的是BGR，转换为RGB以匹配fusion图像
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_imgs.append(img)
        
        # 处理fusion图像，统一尺寸和通道
        # 注意：fusion_img已经是RGB格式（来自post_process）
        fusion_imgs = []
        for fusion_img in batch_fusion_imgs:
            if len(fusion_img.shape) == 2:
                fusion_img = cv2.cvtColor(fusion_img, cv2.COLOR_GRAY2RGB)
            fusion_img = cv2.resize(fusion_img, (target_w, target_h))
            fusion_imgs.append(fusion_img)
        
        # 如果不足4张，用黑色图像填充
        while len(rgb_imgs) < batch_size:
            rgb_imgs.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
            fusion_imgs.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        
        # 水平拼接成4x1
        rgb_grid = np.hstack(rgb_imgs)
        fusion_grid = np.hstack(fusion_imgs)
        
        grids.append((rgb_grid, fusion_grid))
    
    return grids

from contextlib import nullcontext

# ── Fast samplers ──
try:
    from informationSynthesis.fast_samplers import DPMSolverSampler, OptimizedDDIMSampler
    _FAST_SAMPLERS = True
except ImportError:
    _FAST_SAMPLERS = False

_COMPILE_AVAILABLE = hasattr(torch, 'compile')


def _amp_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class Inferen(nn.Module):
    def __init__(self, model, steps, eta=1.0, batch_size=1, use_autocast=False, profile=False,
                 sampler_name="ddim", plms_sampler=None, use_compile=False):
        super(Inferen, self).__init__()
        self.model = model
        self.steps = steps
        self.eta = eta
        self.batch_size = batch_size
        self.use_autocast = use_autocast
        self.profile = profile
        self.sampler_name = sampler_name
        self.plms = plms_sampler
        self.use_compile = use_compile
        self._autocast_fallback_warned = False
        self._init_sampler()

    def _init_sampler(self):
        if not _FAST_SAMPLERS:
            return
        if self.sampler_name in ("dpmsolver", "dpmsolver-2S", "dpmsolver-1S"):
            self._fast_sampler = DPMSolverSampler(self.model)
            self._fast_sampler._order = 1 if self.sampler_name == "dpmsolver-1S" else 2
        elif self.sampler_name == "ddim-opt":
            self._fast_sampler = OptimizedDDIMSampler(self.model)
        else:
            self._fast_sampler = None

    def forward(self, batch):
        _sync_cuda()
        t0 = time.perf_counter()
        try:
            with _amp_context(self.use_autocast):
                c1 = self.model.get_learned_conditioning(batch["rgb"])
                c2 = self.model.get_learned_conditioning(batch["infrared"])
                _sync_cuda()
                t1 = time.perf_counter()
                samples_ddim = self._run_sampler(c1, c2)
        except RuntimeError as err:
            if self.use_autocast:
                self.use_autocast = False
                if not self._autocast_fallback_warned:
                    print("[提示] 自动回退为 FP32 推理。")
                    self._autocast_fallback_warned = True
                c1 = self.model.get_learned_conditioning(batch["rgb"])
                c2 = self.model.get_learned_conditioning(batch["infrared"])
                _sync_cuda()
                t1 = time.perf_counter()
                samples_ddim = self._run_sampler(c1, c2)
            else:
                raise
        _sync_cuda()
        t2 = time.perf_counter()
        self.last_timing = {
            "conditioning_s": t1 - t0,
            "sampling_s": t2 - t1,
            "inference_s": t2 - t0,
        }
        return samples_ddim

    def _run_sampler(self, c1, c2):
        shape = c1.shape[1:]
        if self.sampler_name in ("dpmsolver", "dpmsolver-2S", "dpmsolver-1S") and _FAST_SAMPLERS:
            order = 1 if self.sampler_name == "dpmsolver-1S" else 2
            samples, _ = self._fast_sampler.sample(
                cond1=c1, cond2=c2, batch_size=self.batch_size,
                shape=shape, steps=self.steps, order=order, eta=self.eta, verbose=False,
            )
            return samples
        elif self.sampler_name == "ddim-opt" and _FAST_SAMPLERS:
            samples, _ = self._fast_sampler.sample(
                cond1=c1, cond2=c2, batch_size=self.batch_size,
                shape=shape, steps=self.steps, eta=self.eta, verbose=False,
            )
            return samples
        elif self.sampler_name == "plms" and self.plms is not None:
            samples_ddim, _ = self.plms.sample(
                S=self.steps, batch_size=self.batch_size,
                shape=shape, conditioning=(c1, c2), eta=self.eta, verbose=False,
            )
            return samples_ddim
        else:
            samples_ddim, _ = self.model.sample_log(
                cond1=c1, cond2=c2, batch_size=self.batch_size,
                ddim=True, ddim_steps=self.steps, eta=self.eta,
            )
            return samples_ddim

def _collect_image_paths(folder: str) -> List[str]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"输入目录不存在: {folder}")
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f.lower())[1] in exts]
    return sorted(files)

def _pair_rgb_ir(rgb_dir: str, ir_dir: str):
    rgb_files = _collect_image_paths(rgb_dir)
    ir_files = _collect_image_paths(ir_dir)
    if not rgb_files:
        raise RuntimeError(f"RGB 目录无有效图片: {rgb_dir}")
    if not ir_files:
        raise RuntimeError(f"IR 目录无有效图片: {ir_dir}")

    def key(p):
        return os.path.splitext(os.path.basename(p))[0]
    rgb_map = {key(p): p for p in rgb_files}
    ir_map = {key(p): p for p in ir_files}
    common = sorted(set(rgb_map) & set(ir_map))
    if not common:
        raise RuntimeError("未找到任何同名文件对")
    return [rgb_map[k] for k in common], [ir_map[k] for k in common]

# 评估相关函数
def evaluate_single_image(fusion_path, rgb_path, ir_path, gt_path=None, ir_clean_path=None):
    """
    评估融合图像质量（针对去退化复原任务优化）
    
    参数:
        fusion_path: 融合结果路径
        rgb_path: 退化RGB输入路径（仅用于调试/可视化）
        ir_path: 退化IR输入路径（仅用于调试/可视化）
        gt_path: GT清晰RGB图像路径（用于复原质量评估）
        ir_clean_path: 清晰IR图像路径（用于融合质量评估）
    
    评估逻辑:
        - 复原质量指标 (PSNR, SSIM, MSE): fusion vs GT
        - 融合质量指标 (MI, SCD, VIFF, Qabf, CC): fusion vs (GT + IR_clean)
        - 单图质量指标 (EN, SD, SF, AG): 仅fusion
    """
    try:
        fusion = image_read_cv2(fusion_path, mode='GRAY')
        
        # 确保所有图像尺寸一致（与推理时的处理保持一致）
        target_shape = fusion.shape
        
        # 读取GT清晰图像
        gt_img = None
        if gt_path and os.path.exists(gt_path):
            gt_img = image_read_cv2(gt_path, mode='GRAY')
            if gt_img.shape != target_shape:
                min_size = min(gt_img.shape[0], gt_img.shape[1])
                gt_img = crop_to_size(gt_img, min_size)
                gt_img = cv2.resize(gt_img, (target_shape[1], target_shape[0]))
        
        # 读取清晰IR图像
        ir_clean_img = None
        if ir_clean_path and os.path.exists(ir_clean_path):
            ir_clean_img = image_read_cv2(ir_clean_path, mode='GRAY')
            if ir_clean_img.shape != target_shape:
                min_size = min(ir_clean_img.shape[0], ir_clean_img.shape[1])
                ir_clean_img = crop_to_size(ir_clean_img, min_size)
                ir_clean_img = cv2.resize(ir_clean_img, (target_shape[1], target_shape[0]))
        
        # 单图质量指标（不需要参照）
        result = {
            'filename': os.path.basename(fusion_path),
            'EN': Evaluator.EN(fusion),
            'SD': Evaluator.SD(fusion),
            'SF': Evaluator.SF(fusion),
            'AG': Evaluator.AG(fusion),
        }
        
        from skimage.metrics import structural_similarity as ssim_func
        
        # 复原质量指标：fusion vs GT
        if gt_img is not None:
            mse_val = np.mean((fusion.astype(np.float64) - gt_img.astype(np.float64)) ** 2)
            result['MSE'] = mse_val
            result['PSNR'] = 10 * np.log10(255.0 ** 2 / mse_val) if mse_val > 0 else 100.0
            result['SSIM'] = ssim_func(fusion, gt_img, data_range=255.0)
        else:
            print(f"  [警告] GT文件不存在: {gt_path}")
            result['MSE'] = 0.0
            result['PSNR'] = 0.0
            result['SSIM'] = 0.0
        
        # 融合质量指标：fusion vs (GT_clean + IR_clean)
        # 这才是有意义的融合评估！衡量融合图保留了多少清晰源图像的信息
        if gt_img is not None and ir_clean_img is not None:
            result['MI'] = Evaluator.MI(fusion, gt_img, ir_clean_img)
            result['CC'] = Evaluator.CC(fusion, gt_img, ir_clean_img)
            result['SCD'] = Evaluator.SCD(fusion, gt_img, ir_clean_img)
            result['VIFF'] = Evaluator.VIFF(fusion, gt_img, ir_clean_img)
            result['Qabf'] = Evaluator.Qabf(fusion, gt_img, ir_clean_img)
        elif gt_img is not None:
            # 只有GT，没有IR_clean，退化使用退化IR（不推荐）
            ir_src = image_read_cv2(ir_path, mode='GRAY')
            if ir_src.shape != target_shape:
                min_size = min(ir_src.shape[0], ir_src.shape[1])
                ir_src = crop_to_size(ir_src, min_size)
                ir_src = cv2.resize(ir_src, (target_shape[1], target_shape[0]))
            print(f"  [警告] 无IR_clean，融合指标使用退化IR: {ir_path}")
            result['MI'] = Evaluator.MI(fusion, gt_img, ir_src)
            result['CC'] = Evaluator.CC(fusion, gt_img, ir_src)
            result['SCD'] = Evaluator.SCD(fusion, gt_img, ir_src)
            result['VIFF'] = Evaluator.VIFF(fusion, gt_img, ir_src)
            result['Qabf'] = Evaluator.Qabf(fusion, gt_img, ir_src)
        else:
            # 都没有，使用退化输入（传统融合评估，对复原任务无意义）
            rgb_src = image_read_cv2(rgb_path, mode='GRAY')
            ir_src = image_read_cv2(ir_path, mode='GRAY')
            if rgb_src.shape != target_shape:
                min_size = min(rgb_src.shape[0], rgb_src.shape[1])
                rgb_src = crop_to_size(rgb_src, min_size)
                rgb_src = cv2.resize(rgb_src, (target_shape[1], target_shape[0]))
            if ir_src.shape != target_shape:
                min_size = min(ir_src.shape[0], ir_src.shape[1])
                ir_src = crop_to_size(ir_src, min_size)
                ir_src = cv2.resize(ir_src, (target_shape[1], target_shape[0]))
            print(f"  [警告] 无GT和IR_clean，融合指标使用退化输入（对复原任务无意义）")
            result['MI'] = Evaluator.MI(fusion, rgb_src, ir_src)
            result['CC'] = Evaluator.CC(fusion, rgb_src, ir_src)
            result['SCD'] = Evaluator.SCD(fusion, rgb_src, ir_src)
            result['VIFF'] = Evaluator.VIFF(fusion, rgb_src, ir_src)
            result['Qabf'] = Evaluator.Qabf(fusion, rgb_src, ir_src)
        
        return result
    except Exception as e:
        print(f"评估失败 {os.path.basename(fusion_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_all_results(output_dir, rgb_dir, ir_dir, gt_dir=None, ir_clean_dir=None):
    """
    评估所有融合结果（针对去退化复原任务优化）
    
    参数:
        output_dir: 融合结果目录
        rgb_dir: 退化RGB输入目录
        ir_dir: 退化IR输入目录
        gt_dir: GT清晰RGB图像目录（用于复原质量评估）
        ir_clean_dir: 清晰IR图像目录（用于融合质量评估）
    """
    print("\n" + "=" * 80)
    print("开始评估融合图像质量...")
    print("=" * 80)
    
    fusion_images = sorted([f for f in os.listdir(output_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not fusion_images:
        print("未找到融合图像！")
        return []
    
    print(f"找到 {len(fusion_images)} 张融合图像\n")
    
    # 检查参照图像目录
    has_gt = gt_dir is not None and os.path.isdir(gt_dir)
    has_ir_clean = ir_clean_dir is not None and os.path.isdir(ir_clean_dir)
    
    print("评估配置:")
    print(f"  - 复原质量指标 (PSNR/SSIM/MSE): fusion vs GT {'✓' if has_gt else '✗ 未提供GT'}")
    if has_gt and has_ir_clean:
        print(f"  - 融合质量指标 (MI/SCD/VIFF/Qabf/CC): fusion vs (GT + IR_clean) ✓")
    elif has_gt:
        print(f"  - 融合质量指标 (MI/SCD/VIFF/Qabf/CC): fusion vs (GT + 退化IR) ⚠️")
    else:
        print(f"  - 融合质量指标 (MI/SCD/VIFF/Qabf/CC): fusion vs 退化输入 ⚠️ (对复原任务无意义)")
    print(f"  - 单图质量指标 (EN/SD/SF/AG): 仅评估fusion ✓")
    
    if has_gt:
        print(f"  - GT目录: {gt_dir}")
    if has_ir_clean:
        print(f"  - IR_clean目录: {ir_clean_dir}")
    print()
    
    all_results = []
    for filename in tqdm(fusion_images, desc="评估进度"):
        fusion_path = os.path.join(output_dir, filename)
        original_name = filename.replace('enh_', '') if filename.startswith('enh_') else filename
        rgb_path = os.path.join(rgb_dir, original_name)
        ir_path = os.path.join(ir_dir, original_name)
        gt_path = os.path.join(gt_dir, original_name) if has_gt else None
        ir_clean_path = os.path.join(ir_clean_dir, original_name) if has_ir_clean else None
        
        if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
            continue
        
        result = evaluate_single_image(fusion_path, rgb_path, ir_path, gt_path, ir_clean_path)
        if result:
            all_results.append(result)
    
    return all_results

def print_evaluation_results(results):
    if not results:
        return None
    
    print("\n" + "=" * 100)
    print("评估结果汇总")
    print("=" * 100)
    print(f"{'文件名':<25} {'EN':>7} {'SD':>7} {'MI':>7} {'PSNR':>8} {'SSIM':>7} {'Qabf':>7}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['filename']:<25} {r['EN']:>7.3f} {r['SD']:>7.2f} {r['MI']:>7.3f} "
              f"{r['PSNR']:>8.2f} {r['SSIM']:>7.4f} {r['Qabf']:>7.4f}")
    
    print("-" * 100)
    metrics = ['EN', 'SD', 'MI', 'PSNR', 'SSIM', 'Qabf']
    avg = {k: sum(r[k] for r in results) / len(results) for k in metrics}
    print(f"{'平均值':<25} {avg['EN']:>7.3f} {avg['SD']:>7.2f} {avg['MI']:>7.3f} "
          f"{avg['PSNR']:>8.2f} {avg['SSIM']:>7.4f} {avg['Qabf']:>7.4f}")
    print("=" * 100)
    return avg

def save_evaluation_results(results, output_dir):
    if not results:
        return
    
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ 详细结果已保存到: {csv_path}")
    
    txt_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("图像融合质量评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"评估图像数量: {len(results)}\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        metrics = ['EN', 'SD', 'SF', 'AG', 'MI', 'MSE', 'CC', 'PSNR', 'SCD', 'VIFF', 'Qabf', 'SSIM']
        f.write("指标统计:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<10} {'平均值':>12} {'最小值':>12} {'最大值':>12} {'标准差':>12}\n")
        f.write("-" * 80 + "\n")
        
        for metric in metrics:
            values = [r[metric] for r in results]
            f.write(f"{metric:<10} {np.mean(values):>12.4f} {np.min(values):>12.4f} "
                   f"{np.max(values):>12.4f} {np.std(values):>12.4f}\n")
    
    print(f"✓ 评估摘要已保存到: {txt_path}")

def run_inference(opt):
    print("\n" + "=" * 80)
    print("步骤 1/2: 运行图像融合推理")
    print("=" * 80)
    
    rgbs, infrareds = _pair_rgb_ir(opt.rgb_path, opt.ir_path)
    print(f"共找到 {len(rgbs)} 对匹配的输入图像。")
    
    # 创建批量网格输出目录
    if opt.save_batch_grids:
        batch_grid_dir = os.path.join(os.path.dirname(opt.outdir), 'batch_grids')
        os.makedirs(batch_grid_dir, exist_ok=True)
        print(f"批量网格将保存到: {batch_grid_dir}")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    state = torch.load(opt.checkpoint, map_location='cpu')
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\n{'='*80}")
    print(f"设备信息:")
    print(f"  - 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  - GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  - 当前已用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"{'='*80}\n")
    
    model = model.to(device)
    inference_model = Inferen(model, opt.steps, eta=opt.eta, batch_size=opt.batch_size)
    os.makedirs(opt.outdir, exist_ok=True)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    print("开始推理...")
    fusion_images = []  # 存储所有融合结果
    
    stats = {
        "preprocess_s": 0.0,
        "conditioning_s": 0.0,
        "sampling_s": 0.0,
        "decode_s": 0.0,
        "post_s": 0.0,
        "total_s": 0.0,
    }
    with torch.no_grad():
        with model.ema_scope():
            _sync_cuda()
            overall_start = time.perf_counter()
            for rgb, infrare in tqdm(zip(rgbs, infrareds), total=len(rgbs), desc="推理进度"):
                _sync_cuda()
                t0 = time.perf_counter()
                batch = make_batch(rgb, infrare, device=device)
                _sync_cuda()
                t1 = time.perf_counter()
                samples_ddim = inference_model(batch)
                timing = getattr(inference_model, "last_timing", {})
                _sync_cuda()
                t2 = time.perf_counter()
                decoded = model.decode_first_stage(samples_ddim)
                _sync_cuda()
                t3 = time.perf_counter()
                output_img = post_process(decoded)
                t4 = time.perf_counter()
                outpath = os.path.join(opt.outdir, f"enh_{os.path.basename(rgb)}")
                Image.fromarray(output_img.astype(np.uint8)).save(outpath)
                t5 = time.perf_counter()

                stats["preprocess_s"] += t1 - t0
                stats["conditioning_s"] += timing.get("conditioning_s", 0.0)
                stats["sampling_s"] += timing.get("sampling_s", 0.0)
                stats["decode_s"] += t3 - t2
                stats["post_s"] += t4 - t3
                stats["total_s"] += t5 - t0
                
                # 收集融合图像用于批量网格
                if opt.save_batch_grids:
                    fusion_images.append(output_img)
    
    wall_clock = time.perf_counter() - overall_start
    if len(rgbs) > 0:
        print("\n推理性能统计:")
        print(f"  - 平均预处理: {stats['preprocess_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 平均条件编码: {stats['conditioning_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 平均采样: {stats['sampling_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 平均解码: {stats['decode_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 平均后处理: {stats['post_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 平均总耗时: {stats['total_s'] / len(rgbs) * 1000:.2f} ms/张")
        print(f"  - 整体吞吐: {len(rgbs) / wall_clock:.2f} 张/秒")

    print(f"✓ 推理完成，结果已保存到 {opt.outdir}")
    
    # 生成批量网格图
    if opt.save_batch_grids and fusion_images:
        print("\n生成批量网格图...")
        grids = create_batch_grids(rgbs, fusion_images, batch_size=4)
        
        for idx, (rgb_grid, fusion_grid) in enumerate(grids):
            # 保存RGB网格（转回BGR供OpenCV保存）
            rgb_grid_path = os.path.join(batch_grid_dir, f'rgb_grid_batch_{idx+1:02d}.png')
            cv2.imwrite(rgb_grid_path, cv2.cvtColor(rgb_grid, cv2.COLOR_RGB2BGR))
            
            # 保存Fusion网格（转回BGR供OpenCV保存）
            fusion_grid_path = os.path.join(batch_grid_dir, f'fusion_grid_batch_{idx+1:02d}.png')
            cv2.imwrite(fusion_grid_path, cv2.cvtColor(fusion_grid, cv2.COLOR_RGB2BGR))
        
        print(f"✓ 批量网格已保存到 {batch_grid_dir}")
        print(f"  - 共生成 {len(grids)} 组网格图（每组包含RGB和Fusion各1张）")
    
    return len(rgbs)

def main():
    parser = argparse.ArgumentParser(description="图像融合推理+评估一体化脚本 (支持 Blind All-in-One 去退化复原)")
    parser.add_argument("--rgb_path", default="./experiments/instance/rgb", type=str, help="退化RGB输入目录")
    parser.add_argument("--ir_path", default="./experiments/instance/ir", type=str, help="退化IR输入目录")
    parser.add_argument("--outdir", default='./experiments/instance/output', type=str, help="输出目录")
    parser.add_argument("--gt_dir", default="./experiments/instance/gt_clean", type=str, 
                       help="GT清晰RGB图像目录（用于PSNR/SSIM/MSE复原质量评估）")
    parser.add_argument("--ir_clean_dir", default="./experiments/instance/ir_clean", type=str,
                       help="清晰IR图像目录（用于MI/SCD/VIFF等融合质量评估）")
    parser.add_argument("--config", default=os.path.join(_PROJ_ROOT, 'informationSynthesis', 'Image2Image_AllInOne.yaml'), type=str,
                       help="模型配置文件 (默认使用 Blind All-in-One 配置)")
    parser.add_argument("--checkpoint", default=os.path.join(_PROJ_ROOT, 'checkpoint', 'last.ckpt'), type=str,
                       help="模型权重路径")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--save_batch_grids", action="store_true", default=True,
                       help="保存批量网格图（每4张一组，默认启用）")
    parser.add_argument("--no_batch_grids", dest="save_batch_grids", action="store_false",
                       help="禁用批量网格图生成")
    
    opt = parser.parse_args()
    
    print("=" * 80)
    print("图像融合推理+评估一体化系统")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        if not opt.skip_inference:
            num_images = run_inference(opt)
        else:
            print("\n跳过推理步骤")
        
        if not opt.skip_evaluation:
            print("\n" + "=" * 80)
            print("步骤 2/2: 评估融合图像质量")
            print("=" * 80)
            results = evaluate_all_results(opt.outdir, opt.rgb_path, opt.ir_path, 
                                          opt.gt_dir, opt.ir_clean_dir)
            avg_metrics = print_evaluation_results(results)
            save_evaluation_results(results, opt.outdir)
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        print(f"总耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
