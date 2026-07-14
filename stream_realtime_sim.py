#!/usr/bin/env python3
"""
BDFusion 实时推理速度模拟：按真实时钟推进源帧，输出推理速度帧率的视频。

原理：
  处理帧 i 时，源视频时钟在走。处理完后，跳转到源视频当前时刻对应的帧号继续处理。
  最后输出一个帧率 = 推理吞吐的视频，模拟真实系统的输出节奏。

使用:
  python stream_realtime_sim.py \
      --rgb_video data/test_videos/rgb_0_clip/smoke_0_40-80_rgb.mp4 \
      --ir_video data/test_videos/ir_0_clip/smoke_0_40-80_thermal.mp4 \
      --out_video experiments/demo_output_video/ntu4d_realtime.mp4
"""

import argparse, os, time
import cv2, threading
import numpy as np
import torch
from contextlib import nullcontext

from inference import (
    Inferen, StreamInferen, _amp_context,
    _enable_pytorch_deform_attn, post_process,
)

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════
#  模型加载（复用 stream_process 逻辑）
# ═══════════════════════════════════════════════

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


# ═══════════════════════════════════════════════
#  预处理 & 后处理
# ═══════════════════════════════════════════════

def preprocess_frame(frame_bgr, size=256):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    crop = min(h, w)
    rgb = rgb[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    rgb = cv2.resize(rgb, (size, size))
    img = (rgb / 127.5 - 1.0).astype(np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)[None])


def lab_correct(input_rgb: np.ndarray, output_bgr: np.ndarray,
                alpha_L=0.6, alpha_AB=0.25) -> np.ndarray:
    vi = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    vi_lab = cv2.cvtColor(vi, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab[:, :, 0] = alpha_L * out_lab[:, :, 0] + (1 - alpha_L) * vi_lab[:, :, 0]
    out_lab[:, :, 1] = alpha_AB * out_lab[:, :, 1] + (1 - alpha_AB) * vi_lab[:, :, 1]
    out_lab[:, :, 2] = alpha_AB * out_lab[:, :, 2] + (1 - alpha_AB) * vi_lab[:, :, 2]
    return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def read_frame_at(cap, frame_idx):
    """读取指定帧号（不改变后续读取位置的最佳方式）。"""
    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if cur != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return ret, frame


# ═══════════════════════════════════════════════
#  实时模拟主逻辑
# ═══════════════════════════════════════════════

def run_realtime_simulation(stream, model, device, rgb_video, ir_video,
                            out_video, shared_z, total_frames):
    """按真实时钟推进源帧，模拟实时推理效果。"""
    rgb_cap = cv2.VideoCapture(rgb_video)
    ir_cap = cv2.VideoCapture(ir_video)
    src_fps = rgb_cap.get(cv2.CAP_PROP_FPS)

    # 输出缓冲区
    out_frames = []
    frame_latencies = []

    sim_time = 0.0          # 模拟时钟（秒）
    src_frame_idx = 0       # 源视频当前帧号
    TARGET_FPS = 0          # 将由实际吞吐计算

    print(f"源视频: {src_fps:.1f} fps, 共 {total_frames} 帧")
    print(f"实时模拟开始: 按推理实际速度推进源帧...")
    print(f"{'帧号':>6s} {'源帧':>6s} {'耗时':>8s} {'累计':>8s} {'等效fps':>8s}")
    print("-" * 45)

    with torch.inference_mode(), nullcontext():
        while src_frame_idx < total_frames:
            # 读取源视频当前帧
            ok_rgb, rgb_frame = read_frame_at(rgb_cap, src_frame_idx)
            ok_ir, ir_frame = read_frame_at(ir_cap, src_frame_idx)
            if not ok_rgb or not ok_ir:
                break

            # 预处理
            rgb_ref = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_ref.shape[:2]
            crop = min(h, w)
            rgb_ref = rgb_ref[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            rgb_ref = cv2.resize(rgb_ref, (256, 256))

            t0 = time.perf_counter()

            # GPU 推理
            rgb_t = preprocess_frame(rgb_frame).to(device).contiguous(memory_format=torch.channels_last)
            ir_t = preprocess_frame(ir_frame).to(device).contiguous(memory_format=torch.channels_last)
            result = stream.process(rgb_t, ir_t, z_init=shared_z)
            result_bgr = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)

            # LAB 修正
            corrected = lab_correct(rgb_ref, result_bgr)

            t1 = time.perf_counter()
            proc_time = t1 - t0

            # 保存
            out_frames.append(corrected)
            frame_latencies.append(proc_time)

            # 推进模拟时钟
            sim_time += proc_time
            src_frame_idx = int(sim_time * src_fps)

            avg_fps = len(out_frames) / sim_time if sim_time > 0 else 0
            print(f"{len(out_frames):6d} {src_frame_idx:6d} {proc_time*1000:7.1f}ms {sim_time:7.1f}s {avg_fps:7.1f}")

    rgb_cap.release()
    ir_cap.release()

    # ── 输出视频（帧率 = 实际推理吞吐） ──
    n_frames = len(out_frames)
    total_time = sum(frame_latencies)
    effective_fps = n_frames / total_time if total_time > 0 else 1.0

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, effective_fps, (256, 256))
    for frame in out_frames:
        writer.write(frame)
    writer.release()

    # ── 统计 ──
    latencies_ms = [l * 1000 for l in frame_latencies]
    print(f"\n{'='*45}")
    print(f"实时模拟完成")
    print(f"  源视频: {src_fps:.1f} fps, {total_frames} 帧, {total_frames/src_fps:.1f}s")
    print(f"  推理输出: {n_frames} 帧, {total_time:.1f}s, {effective_fps:.2f} fps")
    print(f"  丢帧率: {(1 - n_frames/total_frames)*100:.1f}%")
    print(f"  推理耗时: min={min(latencies_ms):.0f}ms, avg={np.mean(latencies_ms):.0f}ms, "
          f"max={max(latencies_ms):.0f}ms, p99={np.percentile(latencies_ms,99):.0f}ms")
    print(f"  输出视频: {out_video} ({256}x{256}, {effective_fps:.2f} fps)")

    return effective_fps, n_frames


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BDFusion 实时推理速度模拟")
    parser.add_argument("--rgb_video", required=True)
    parser.add_argument("--ir_video", required=True)
    parser.add_argument("--out_video", default="experiments/demo_output_video/ntu4d_realtime.mp4")
    parser.add_argument("--config", default=os.path.join(_PROJ_ROOT, "informationSynthesis", "Image2Image.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(_PROJ_ROOT, "checkpoint", "last.ckpt"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sampler", default="dpmsolver")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, inferen, stream = load_model(
        args.config, args.checkpoint, device,
        sampler=args.sampler, steps=args.steps, eta=args.eta,
        batch_size=1, use_autocast=True,
    )
    stream.warmup(steps=args.warmup)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 共享噪声
    shared_z = torch.randn(1, 3, 64, 64, device=device)

    # 获取源视频信息
    cap = cv2.VideoCapture(args.rgb_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    run_realtime_simulation(
        stream, model, device,
        rgb_video=args.rgb_video,
        ir_video=args.ir_video,
        out_video=args.out_video,
        shared_z=shared_z,
        total_frames=total_frames,
    )


if __name__ == "__main__":
    main()
