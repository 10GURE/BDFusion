#!/usr/bin/env python3
"""
BDFusion 增强视频处理：集成全文进管线（共享噪声+时域中值+LAB修正）。

使用: python process_video_enhanced.py --rgb_video ... --ir_video ... --out_video ...
"""

import argparse, os, sys, time
import cv2, queue, threading
import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

from inference import (
    Inferen, StreamInferen, get_img, _amp_context,
    _enable_pytorch_deform_attn, post_process,
)


def load_model(config_path, ckpt_path, device, sampler="dpmsolver", steps=5,
               eta=0.0, batch_size=1, use_autocast=True):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith('student.') for k in state):
        state = {k[8:] if k.startswith('student.') else k: v for k, v in state.items()}
    remapped = 0
    keys_to_add = {}
    for k, v in list(state.items()):
        if '.to_k.' in k:
            new_k = k.replace('.to_k.', '.to_v.')
            if new_k not in state:
                keys_to_add[new_k] = v.clone()
                remapped += 1
    if remapped:
        state.update(keys_to_add)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    _enable_pytorch_deform_attn(model)
    inferen = Inferen(model, steps, eta=eta, batch_size=batch_size,
                      use_autocast=use_autocast, sampler_name=sampler, use_compile=False)
    stream = StreamInferen(inferen, model, device, enable_stats=True)
    return model, inferen, stream


def preprocess_frame(frame_bgr, size=256):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    crop = min(h, w)
    rgb = rgb[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    rgb = cv2.resize(rgb, (size, size))
    img = (rgb / 127.5 - 1.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    return torch.from_numpy(img)


def lab_correct(input_rgb: np.ndarray, output_bgr: np.ndarray,
                alpha_L=0.6, alpha_AB=0.25) -> np.ndarray:
    """LAB 空间亮度+色彩修正。"""
    vi = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    vi_lab = cv2.cvtColor(vi, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab[:, :, 0] = alpha_L * out_lab[:, :, 0] + (1 - alpha_L) * vi_lab[:, :, 0]
    out_lab[:, :, 1] = alpha_AB * out_lab[:, :, 1] + (1 - alpha_AB) * vi_lab[:, :, 1]
    out_lab[:, :, 2] = alpha_AB * out_lab[:, :, 2] + (1 - alpha_AB) * vi_lab[:, :, 2]
    return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def pixel_median(*frames):
    """逐像素中值聚合。"""
    stack = np.stack([f.astype(np.float32) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def process_video_enhanced(
    stream: StreamInferen,
    model,
    rgb_video: str,
    ir_video: str,
    out_video: str,
    device: torch.device,
    batch_size: int = 1,
    start_frame: int = 0,
    end_frame: int = None,
    frame_skip: int = 1,
):
    """增强视频处理：共享噪声 + 3帧像素中值 + LAB 修正。"""
    rgb_cap = cv2.VideoCapture(rgb_video)
    ir_cap = cv2.VideoCapture(ir_video)
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(rgb_total, ir_total)
    if end_frame is not None:
        total = min(total, end_frame)
    print(f"增强管线: {total} 帧 @ {rgb_fps:.1f}fps (共享噪声+中值+LAB)")

    # 读取队列
    read_queue = queue.Queue(maxsize=4)

    def reader_thread():
        frame_idx = 0
        emitted = 0
        while emitted < total:
            if not rgb_cap.grab() or not ir_cap.grab():
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if (frame_idx - start_frame) % frame_skip != 0:
                frame_idx += 1
                continue
            _, rgb_f = rgb_cap.retrieve()
            _, ir_f = ir_cap.retrieve()
            if rgb_f is None or ir_f is None:
                break
            # 对输入也 resize 到 256 并保留 LAB 参考
            rgb_ref = cv2.cvtColor(rgb_f, cv2.COLOR_BGR2RGB)
            h, w = rgb_ref.shape[:2]
            crop = min(h, w)
            rgb_ref = rgb_ref[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            rgb_ref = cv2.resize(rgb_ref, (256, 256))
            read_queue.put((emitted, rgb_f, ir_f, rgb_ref))
            emitted += 1
            frame_idx += 1
        rgb_cap.release()
        ir_cap.release()
        read_queue.put(None)

    threading.Thread(target=reader_thread, daemon=True).start()

    # 共享噪声
    shared_z = torch.randn(1, 3, 64, 64, device=device)

    # 输出缓冲：存储最近 3 帧的 decoded output + 对应的 LAB 参考
    out_buf = []  # [(frame_idx, result_bgr, rgb_ref), ...]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, rgb_fps, (256, 256))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {out_video}")

    stream.reset_stats()
    t_start = time.perf_counter()
    processed = 0
    written = 0

    with torch.inference_mode(), nullcontext():
        while True:
            item = read_queue.get()
            if item is None:
                break

            frame_idx, rgb_frame, ir_frame, rgb_ref = item

            # 预处理 → GPU
            rgb_t = preprocess_frame(rgb_frame).to(device).contiguous(memory_format=torch.channels_last)
            ir_t = preprocess_frame(ir_frame).to(device).contiguous(memory_format=torch.channels_last)

            result = stream.process(rgb_t, ir_t, z_init=shared_z)

            # 转换为 BGR
            result_bgr = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)

            # 存入输出缓冲
            out_buf.append((frame_idx, result_bgr, rgb_ref))
            processed += 1

            # 当缓冲 ≥ 3 帧时，输出中间帧的中值聚合结果
            if len(out_buf) >= 3:
                idx0, bgr0, ref0 = out_buf[0]
                idx1, bgr1, ref1 = out_buf[1]
                idx2, bgr2, ref2 = out_buf[2]

                # 3帧像素中值
                median_bgr = pixel_median(bgr0, bgr1, bgr2)

                # LAB 亮度+色彩修正 (使用中间帧的参考)
                corrected = lab_correct(ref1, median_bgr)

                writer.write(corrected)
                written += 1

                # 弹出最早帧
                out_buf.pop(0)

            sys.stdout.write(f"\r[{processed}/{total}]")
            sys.stdout.flush()
            read_queue.task_done()

    # 排空缓冲区中的剩余帧
    while len(out_buf) >= 1:
        if len(out_buf) == 1:
            _, bgr, ref = out_buf[0]
            corrected = lab_correct(ref, bgr)
            writer.write(corrected)
            written += 1
            out_buf.pop(0)
        elif len(out_buf) == 2:
            _, bgr0, ref0 = out_buf[0]
            _, bgr1, ref1 = out_buf[1]
            # 2帧平均
            avg_bgr = ((bgr0.astype(np.float32) + bgr1.astype(np.float32)) / 2).astype(np.uint8)
            corrected = lab_correct(ref0, avg_bgr)
            writer.write(corrected)
            writer.write(corrected)  # 写两帧
            written += 2
            out_buf.clear()

    writer.release()
    wall_clock = time.perf_counter() - t_start

    print(f"\n增强视频完成: {processed} 帧处理, {written} 帧输出")
    print(f"  总耗时: {wall_clock:.1f}s")
    print(f"  吞吐:   {processed / wall_clock:.2f} fps")
    print(f"  输出: {out_video}")


def main():
    parser = argparse.ArgumentParser(description="BDFusion 增强视频处理")
    parser.add_argument("--rgb_video", required=True)
    parser.add_argument("--ir_video", required=True)
    parser.add_argument("--out_video", default="experiments/demo_output_video/enhanced_output.mp4")
    parser.add_argument("--config", default=os.path.join(_PROJ_ROOT, "informationSynthesis", "Image2Image.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(_PROJ_ROOT, "checkpoint", "last.ckpt"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sampler", default="dpmsolver")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--precision", default="autocast")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = args.precision == "autocast" and device.type == "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, inferen, stream = load_model(
        args.config, args.checkpoint, device,
        sampler=args.sampler, steps=args.steps, eta=args.eta,
        batch_size=1, use_autocast=use_autocast,
    )
    stream.warmup(steps=args.warmup)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    process_video_enhanced(
        stream, model,
        rgb_video=args.rgb_video,
        ir_video=args.ir_video,
        out_video=args.out_video,
        device=device,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
