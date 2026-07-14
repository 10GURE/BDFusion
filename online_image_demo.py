#!/usr/bin/env python3
"""
BDFusion 逐图片文件在线处理 demo。

功能：
1. 监听两个目录中的 RGB / IR 图片文件
2. 发现同名文件对后立即做推理
3. 输出到结果目录，并可选实时预览

这个脚本的目标是做最小可演示版本，不追求复杂调度。
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

from inference import get_img
from stream_process import load_model


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _collect_image_paths(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"输入目录不存在: {folder}")
    files = [os.path.join(folder, name) for name in os.listdir(folder)
             if os.path.splitext(name.lower())[1] in VALID_EXTS]
    return sorted(files)


def _pair_existing(rgb_dir: str, ir_dir: str) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    rgb_files = _collect_image_paths(rgb_dir)
    ir_files = _collect_image_paths(ir_dir)
    rgb_map = {_stem(path): path for path in rgb_files}
    ir_map = {_stem(path): path for path in ir_files}
    ready = sorted(set(rgb_map) & set(ir_map))
    return rgb_map, ir_map, ready


def _load_display_image(path: str, size: int = 256) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    h, w = img.shape[:2]
    crop = min(h, w)
    img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    img = cv2.resize(img, (size, size))
    return img


def _build_preview(rgb_path: str, result_rgb: np.ndarray) -> np.ndarray:
    rgb_img = _load_display_image(rgb_path)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    panel = np.hstack([rgb_img, result_bgr])
    canvas = cv2.copyMakeBorder(panel, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    cv2.putText(canvas, "RGB input", (12, panel.shape[0] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(canvas, "BDFusion output", (panel.shape[1] // 2 + 12, panel.shape[0] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return canvas


def _process_one(stream, device, rgb_path: str, ir_path: str) -> np.ndarray:
    rgb = get_img(rgb_path).to(device=device).contiguous(memory_format=torch.channels_last)
    ir = get_img(ir_path).to(device=device).contiguous(memory_format=torch.channels_last)
    result = stream.process(rgb, ir)
    return result[0] if result.ndim == 4 else result


def main():
    parser = argparse.ArgumentParser(description="BDFusion 逐图片文件在线处理 demo")
    parser.add_argument("--rgb_dir", type=str, default="./experiments/instance/rgb", help="RGB 图片目录")
    parser.add_argument("--ir_dir", type=str, default="./experiments/instance/ir", help="IR 图片目录")
    parser.add_argument("--out_dir", type=str, default="./experiments/demo_output", help="输出目录")
    parser.add_argument("--config", type=str, default=os.path.join(_PROJ_ROOT, "informationSynthesis", "Image2Image.yaml"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_PROJ_ROOT, "checkpoint", "last.ckpt"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sampler", type=str, default="dpmsolver",
                        choices=["ddim", "ddim-opt", "dpmsolver", "dpmsolver-1S", "dpmsolver-2S", "dpmsolver-graph", "plms"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--precision", type=str, default="autocast", choices=["fp32", "autocast"])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--preview", action="store_true", help="实时预览窗口")
    parser.add_argument("--poll_interval", type=float, default=0.5, help="轮询间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只处理当前已存在文件，处理完退出")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少对图片，0 表示不限")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = args.precision == "autocast" and device.type == "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"加载模型: {args.checkpoint}")
    model, inferen, stream = load_model(
        args.config,
        args.checkpoint,
        device,
        sampler=args.sampler,
        steps=args.steps,
        eta=args.eta,
        batch_size=1,
        use_autocast=use_autocast,
    )

    print(f"预热 {args.warmup} 次...")
    stream.warmup(steps=args.warmup)

    # 预热会消耗随机数，重置种子保证后续结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.preview:
        cv2.namedWindow("BDFusion Online Demo", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEP_RATIO)

    processed = set()
    total_done = 0
    t_start = time.perf_counter()

    print("开始在线监听目录...")
    print(f"  RGB: {args.rgb_dir}")
    print(f"  IR : {args.ir_dir}")
    print(f"  OUT: {args.out_dir}")
    print("  按 q 可退出预览")

    try:
        while True:
            rgb_map, ir_map, ready = _pair_existing(args.rgb_dir, args.ir_dir)
            pending = []
            for stem in ready:
                if stem in processed:
                    continue
                out_path = os.path.join(args.out_dir, f"enh_{os.path.basename(rgb_map[stem])}")
                if not args.overwrite and os.path.exists(out_path):
                    processed.add(stem)
                    continue
                pending.append(stem)

            if not pending:
                if args.once and total_done > 0:
                    break
                if args.preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                time.sleep(args.poll_interval)
                continue

            for stem in pending:
                rgb_path = rgb_map[stem]
                ir_path = ir_map[stem]

                result = _process_one(stream, device, rgb_path, ir_path)
                out_path = os.path.join(args.out_dir, f"enh_{os.path.basename(rgb_path)}")
                Image.fromarray(result.astype(np.uint8)).save(out_path)

                if args.preview:
                    preview = _build_preview(rgb_path, result)
                    cv2.imshow("BDFusion Online Demo", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        processed.add(stem)
                        total_done += 1
                        break

                processed.add(stem)
                total_done += 1

                elapsed = time.perf_counter() - t_start
                fps = total_done / elapsed if elapsed > 0 else 0.0
                print(f"[{total_done}] {os.path.basename(rgb_path)} -> {os.path.basename(out_path)} | {fps:.2f} fps")

                if args.limit > 0 and total_done >= args.limit:
                    raise KeyboardInterrupt

            if args.once and total_done > 0:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if args.preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    if total_done > 0:
        print(f"\n完成: {total_done} 对图片")
        print(f"总耗时: {elapsed:.1f}s")
        print(f"平均吞吐: {total_done / elapsed:.2f} fps")
        print(f"输出目录: {args.out_dir}")
    else:
        print("未发现可处理的图片对。")


if __name__ == "__main__":
    main()