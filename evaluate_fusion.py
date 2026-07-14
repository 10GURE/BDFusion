#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用融合结果量化评估脚本
功能：对多任务类型的融合结果进行量化评估

输入结构（融合结果）：
    <result_dir>/<任务类型>/        - 融合结果图像

参考数据结构（源图像 + GT）：
    <test_dir>/<任务类型>/ir/       - 红外图像
    <test_dir>/<任务类型>/vi/       - 可见光图像
    <test_dir>/<任务类型>/gt/       - 参考图像（可选）

输出（评估报告）：
    <result_dir>/<任务类型>.csv     - 每个任务类型的评估结果

用法示例：
    # 评估 SeAFusion 的结果
    python evaluate_fusion.py --result_dir ./output --test_dir ./test_imgs

    # 评估其他方法的结果
    python evaluate_fusion.py --result_dir ./output_methodB --test_dir ./test_imgs
"""

import argparse
import os
import time
import csv
import math
import numpy as np
import cv2
from tqdm import tqdm
from natsort import natsorted
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim


# ==================== 评估器类 ====================
def image_read_cv2(path, mode='RGB'):
    """读取图像并转换为指定格式"""
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def _center_crop_to_aspect(img, target_h, target_w):
    """中心裁剪到目标长宽比"""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        start_w = (w - new_w) // 2
        return img[:, start_w:start_w + new_w]
    else:
        new_h = int(round(w / target_ratio))
        start_h = (h - new_h) // 2
        return img[start_h:start_h + new_h, :]


def _crop_and_resize_to(img, target_shape):
    """先中心裁剪，再缩放到目标尺寸"""
    target_h, target_w = target_shape
    img = _center_crop_to_aspect(img, target_h, target_w)
    return cv2.resize(img, (target_w, target_h))


class Evaluator:
    """图像融合质量评估器"""

    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):
        """计算图像的信息熵"""
        cls.input_check(img)
        hist = np.histogram(img, bins=256, range=[0, 255])[0]
        hist = hist / float(sum(hist))
        entropy = -sum([p * np.log2(p) for p in hist if p != 0])
        return entropy

    @classmethod
    def SD(cls, img):
        """计算图像的标准差"""
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        """计算图像的空间频率"""
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) +
                       np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):
        """计算图像的平均梯度"""
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        """计算互信息"""
        cls.input_check(image_F, image_A, image_B)
        from sklearn.metrics import mutual_info_score
        f = np.clip(np.rint(image_F), 0, 255).astype(np.uint8).flatten()
        a = np.clip(np.rint(image_A), 0, 255).astype(np.uint8).flatten()
        b = np.clip(np.rint(image_B), 0, 255).astype(np.uint8).flatten()
        return mutual_info_score(f, a) + mutual_info_score(f, b)

    @classmethod
    def MSE(cls, image_F, image_A, image_B):
        """计算均方误差"""
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        """计算相关系数"""
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        """计算峰值信噪比"""
        cls.input_check(image_F, image_A, image_B)
        mse = cls.MSE(image_F, image_A, image_B)
        if mse == 0:
            return 100.0
        return 10 * np.log10(255.0 ** 2 / mse)

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        """计算和-相关差"""
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        """计算视觉信息保真度"""
        cls.input_check(image_F, image_A, image_B)
        return cls._compare_viff(image_A, image_F) + cls._compare_viff(image_B, image_F)

    @classmethod
    def _compare_viff(cls, ref, dist):
        """计算单对图像的VIF"""
        sigma_nsq = 2
        eps = 1e-10
        num = 0.0
        den = 0.0

        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den if den != 0 else 1.0
        return 1.0 if np.isnan(vifp) else vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        """计算基于梯度的融合质量指标 Qabf"""
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls._Qabf_getArray(image_A)
        gB, aB = cls._Qabf_getArray(image_B)
        gF, aF = cls._Qabf_getArray(image_F)
        QAF = cls._Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls._Qabf_getQabf(aB, gB, aF, gF)

        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno if deno != 0 else 0.0

    @classmethod
    def _Qabf_getArray(cls, img):
        """计算图像的梯度幅值和方向"""
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def _Qabf_getQabf(cls, aA, gA, aF, gF):
        """计算 Qabf 分量"""
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8

        GAF = np.zeros_like(aA)
        GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B, data_range=255.0):
        """计算结构相似度"""
        cls.input_check(image_F, image_A, image_B)
        return (ssim(image_F, image_A, data_range=data_range) +
                ssim(image_F, image_B, data_range=data_range))


# ==================== 评估函数 ====================
def evaluate_single_image(fusion_path, vi_path, ir_path, gt_path=None):
    """评估单张融合图像质量"""
    try:
        fusion = image_read_cv2(fusion_path, mode='GRAY')
        target_shape = fusion.shape

        vi_img = image_read_cv2(vi_path, mode='GRAY')
        ir_img = image_read_cv2(ir_path, mode='GRAY')

        if vi_img.shape != target_shape:
            vi_img = _crop_and_resize_to(vi_img, target_shape)
        if ir_img.shape != target_shape:
            ir_img = _crop_and_resize_to(ir_img, target_shape)

        gt_img = None
        if gt_path and os.path.exists(gt_path):
            gt_img = image_read_cv2(gt_path, mode='GRAY')
            if gt_img.shape != target_shape:
                gt_img = _crop_and_resize_to(gt_img, target_shape)

        result = {
            'filename': os.path.basename(fusion_path),
        }

        # 自评价指标
        result['EN'] = Evaluator.EN(fusion)
        result['SD'] = Evaluator.SD(fusion)
        result['SF'] = Evaluator.SF(fusion)
        result['AG'] = Evaluator.AG(fusion)

        # 对比评价指标：fusion vs (vi + ir)
        result['MI'] = Evaluator.MI(fusion, vi_img, ir_img)
        result['CC'] = Evaluator.CC(fusion, vi_img, ir_img)
        result['SCD'] = Evaluator.SCD(fusion, vi_img, ir_img)
        result['VIFF'] = Evaluator.VIFF(fusion, vi_img, ir_img)
        result['Qabf'] = Evaluator.Qabf(fusion, vi_img, ir_img)
        result['SSIM'] = Evaluator.SSIM(fusion, vi_img, ir_img)

        # GT 对比指标
        if gt_img is not None:
            mse_gt = np.mean((fusion.astype(np.float64) - gt_img.astype(np.float64)) ** 2)
            result['MSE_gt'] = mse_gt
            result['PSNR_gt'] = 10 * np.log10(255.0 ** 2 / mse_gt) if mse_gt > 0 else 100.0
            result['SSIM_gt'] = ssim(fusion, gt_img, data_range=255.0)
        else:
            result['MSE_gt'] = 0.0
            result['PSNR_gt'] = 0.0
            result['SSIM_gt'] = 0.0

        return result

    except Exception as e:
        print(f"    [错误] 评估 {os.path.basename(fusion_path)} 失败: {e}")
        return None


def evaluate_task_results(result_dir, vi_dir, ir_dir, gt_dir=None):
    """评估某个任务类型的所有融合结果"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    fusion_images = natsorted([f for f in os.listdir(result_dir)
                               if os.path.splitext(f.lower())[1] in exts])

    if not fusion_images:
        print(f"    [警告] {result_dir} 中未找到融合图像")
        return []

    all_results = []
    for filename in tqdm(fusion_images, desc="    评估进度"):
        fusion_path = os.path.join(result_dir, filename)
        vi_path = os.path.join(vi_dir, filename)
        ir_path = os.path.join(ir_dir, filename)
        gt_path = os.path.join(gt_dir, filename) if gt_dir else None

        if not os.path.exists(vi_path) or not os.path.exists(ir_path):
            continue

        result = evaluate_single_image(fusion_path, vi_path, ir_path, gt_path)
        if result:
            all_results.append(result)

    return all_results


def save_evaluation_results(results, csv_path):
    """保存评估结果到 CSV"""
    if not results:
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

        f.write('\n')
        avg_row = {'filename': 'AVERAGE'}
        metrics = [k for k in results[0].keys() if k != 'filename']
        for metric in metrics:
            values = [r[metric] for r in results]
            avg_row[metric] = np.mean(values)
        writer.writerow(avg_row)

    print(f"    ✓ 评估结果已保存到: {csv_path}")


def print_task_summary(task_name, results):
    """打印单个任务的评估摘要"""
    if not results:
        return

    metrics = ['EN', 'SD', 'SF', 'AG', 'MI', 'VIFF', 'Qabf', 'SSIM']
    if results[0].get('PSNR_gt', 0) > 0:
        metrics.extend(['PSNR_gt', 'SSIM_gt'])

    print(f"\n    [{task_name}] 评估摘要 ({len(results)} 张图像):")
    print("    " + "-" * 70)

    avg_values = {}
    for metric in metrics:
        values = [r[metric] for r in results]
        avg_values[metric] = np.mean(values)

    line1_metrics = metrics[:4]
    line2_metrics = metrics[4:8]
    line3_metrics = metrics[8:] if len(metrics) > 8 else []

    print("    " + "  ".join([f"{m}:{avg_values[m]:>7.3f}" for m in line1_metrics]))
    print("    " + "  ".join([f"{m}:{avg_values[m]:>7.3f}" for m in line2_metrics]))
    if line3_metrics:
        print("    " + "  ".join([f"{m}:{avg_values[m]:>7.3f}" for m in line3_metrics]))


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='通用融合结果量化评估脚本')
    parser.add_argument('--result_dir', type=str, default='./experiments/output',
                        help='融合结果根目录，结构: <result_dir>/<任务类型>/<图像>')
    parser.add_argument('--test_dir', type=str, default='./test_imgs',
                        help='测试数据根目录，结构: <test_dir>/<任务类型>/ir|vi|gt/')

    args = parser.parse_args()

    print("=" * 80)
    print("通用融合结果量化评估")
    print("=" * 80)
    print(f"融合结果目录: {args.result_dir}")
    print(f"测试数据目录: {args.test_dir}")

    start_time = time.time()

    # 获取所有任务类型（以 result_dir 中的子目录为准）
    task_types = [d for d in os.listdir(args.result_dir)
                  if os.path.isdir(os.path.join(args.result_dir, d))]
    task_types = natsorted(task_types)

    print(f"\n发现 {len(task_types)} 个任务类型: {task_types}")

    all_task_results = {}

    for task_name in task_types:
        print(f"\n{'=' * 80}")
        print(f"评估任务: {task_name}")
        print("=" * 80)

        task_result_dir = os.path.join(args.result_dir, task_name)
        ir_dir = os.path.join(args.test_dir, task_name, 'ir')
        vi_dir = os.path.join(args.test_dir, task_name, 'vi')
        gt_dir = os.path.join(args.test_dir, task_name, 'gt')

        if not os.path.isdir(ir_dir) or not os.path.isdir(vi_dir):
            print(f"  [跳过] 在 {args.test_dir}/{task_name}/ 中未找到 ir 或 vi 目录")
            continue

        has_gt = os.path.isdir(gt_dir)
        if has_gt:
            print(f"  发现 GT 目录，将计算与 GT 的对比指标")
        else:
            print(f"  未发现 GT 目录，仅计算自评价和源图像对比指标")

        results = evaluate_task_results(
            task_result_dir, vi_dir, ir_dir,
            gt_dir if has_gt else None
        )

        if results:
            csv_path = os.path.join(args.result_dir, f"{task_name}.csv")
            save_evaluation_results(results, csv_path)
            print_task_summary(task_name, results)
            all_task_results[task_name] = results

    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("评估完成!")
    print("=" * 80)
    print(f"总耗时: {elapsed_time:.2f} 秒")

    if all_task_results:
        print(f"\n{'=' * 80}")
        print("所有任务平均指标对比:")
        print("=" * 80)

        metrics = ['EN', 'SD', 'SF', 'AG', 'MI', 'VIFF', 'Qabf']
        header = f"{'任务类型':<20}" + "".join([f"{m:>10}" for m in metrics])
        print(header)
        print("-" * len(header))

        for task_name, results in all_task_results.items():
            row = f"{task_name:<20}"
            for metric in metrics:
                values = [r[metric] for r in results]
                row += f"{np.mean(values):>10.3f}"
            print(row)


if __name__ == "__main__":
    main()
