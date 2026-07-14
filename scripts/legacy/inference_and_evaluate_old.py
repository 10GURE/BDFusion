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

class Inferen(nn.Module):
    def __init__(self, model, steps):
        super(Inferen, self).__init__()
        self.model = model
        self.steps = steps

    def forward(self, batch):
        c1 = self.model.get_learned_conditioning(batch["rgb"])
        c2 = self.model.get_learned_conditioning(batch["infrared"])
        samples_ddim, _ = self.model.sample_log(cond1=c1, cond2=c2, batch_size=1, ddim=True,
                                           ddim_steps=self.steps, eta=1.0)
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
def evaluate_single_image(fusion_path, rgb_path, ir_path, gt_path=None):
    try:
        fusion = image_read_cv2(fusion_path, mode='GRAY')
        rgb_src = image_read_cv2(rgb_path, mode='GRAY')
        ir_src = image_read_cv2(ir_path, mode='GRAY')
        
        # 如果提供了GT，也读取GT图像
        gt_img = None
        if gt_path and os.path.exists(gt_path):
            gt_img = image_read_cv2(gt_path, mode='GRAY')
        
        # 确保所有图像尺寸一致（resize到融合图像的尺寸）
        target_shape = fusion.shape
        if rgb_src.shape != target_shape:
            rgb_src = cv2.resize(rgb_src, (target_shape[1], target_shape[0]))
        if ir_src.shape != target_shape:
            ir_src = cv2.resize(ir_src, (target_shape[1], target_shape[0]))
        if gt_img is not None and gt_img.shape != target_shape:
            gt_img = cv2.resize(gt_img, (target_shape[1], target_shape[0]))
        
        result = {
            'filename': os.path.basename(fusion_path),
            'EN': Evaluator.EN(fusion),
            'SD': Evaluator.SD(fusion),
            'SF': Evaluator.SF(fusion),
            'AG': Evaluator.AG(fusion),
            'MI': Evaluator.MI(fusion, rgb_src, ir_src),
            'MSE': Evaluator.MSE(fusion, rgb_src, ir_src),
            'CC': Evaluator.CC(fusion, rgb_src, ir_src),
            'PSNR': Evaluator.PSNR(fusion, rgb_src, ir_src),
            'SCD': Evaluator.SCD(fusion, rgb_src, ir_src),
            'VIFF': Evaluator.VIFF(fusion, rgb_src, ir_src),
            'Qabf': Evaluator.Qabf(fusion, rgb_src, ir_src),
            'SSIM': Evaluator.SSIM(fusion, rgb_src, ir_src, data_range=255.0),
        }
        
        # 如果有GT，计算与GT的对比指标
        if gt_img is not None:
            result['PSNR_GT'] = 10 * np.log10(np.max(fusion) ** 2 / np.mean((fusion - gt_img) ** 2))
            from skimage.metrics import structural_similarity as ssim
            result['SSIM_GT'] = ssim(fusion, gt_img, data_range=255.0)
            result['MSE_GT'] = np.mean((fusion - gt_img) ** 2)
        
        return result
    except Exception as e:
        print(f"评估失败 {os.path.basename(fusion_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_all_results(output_dir, rgb_dir, ir_dir, gt_dir=None):
    print("\n" + "=" * 80)
    print("开始评估融合图像质量...")
    print("=" * 80)
    
    fusion_images = sorted([f for f in os.listdir(output_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not fusion_images:
        print("未找到融合图像！")
        return []
    
    print(f"找到 {len(fusion_images)} 张融合图像\n")
    
    all_results = []
    for filename in tqdm(fusion_images, desc="评估进度"):
        fusion_path = os.path.join(output_dir, filename)
        original_name = filename.replace('enh_', '') if filename.startswith('enh_') else filename
        rgb_path = os.path.join(rgb_dir, original_name)
        ir_path = os.path.join(ir_dir, original_name)
        gt_path = os.path.join(gt_dir, original_name) if has_gt else None
        
        if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
            continue
        
        result = evaluate_single_image(fusion_path, rgb_path, ir_path, gt_path)
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

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    state = torch.load(opt.checkpoint, map_location='cpu')
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    inference_model = Inferen(model, opt.steps)
    os.makedirs(opt.outdir, exist_ok=True)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    print("开始推理...")
    with torch.no_grad():
        with model.ema_scope():
            for rgb, infrare in tqdm(zip(rgbs, infrareds), total=len(rgbs), desc="推理进度"):
                batch = make_batch(rgb, infrare, device=device)
                samples_ddim = inference_model(batch)
                decoded = model.decode_first_stage(samples_ddim)
                output_img = post_process(decoded)
                outpath = os.path.join(opt.outdir, f"enh_{os.path.basename(rgb)}")
                Image.fromarray(output_img.astype(np.uint8)).save(outpath)
    
    print(f"✓ 推理完成，结果已保存到 {opt.outdir}")
    return len(rgbs)

def main():
    parser = argparse.ArgumentParser(description="图像融合推理+评估一体化脚本")
    parser.add_argument("--rgb_path", default="./experiments/instance/rgb", type=str)
    parser.add_argument("--ir_path", default="./experiments/instance/ir", type=str)
    parser.add_argument("--outdir", default='./experiments/instance/output', type=str)
    parser.add_argument("--gt_dir", default="./experiments/instance/inference", type=str, help="GT清晰图像目录")
    parser.add_argument("--config", default=os.path.join(_PROJ_ROOT, 'informationSynthesis', 'Image2Image.yaml'), type=str)
    parser.add_argument("--checkpoint", default=os.path.join(_PROJ_ROOT, 'checkpoint', 'last.ckpt'), type=str)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    
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
            results = evaluate_all_results(opt.outdir, opt.rgb_path, opt.ir_path, opt.gt_dir)
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
