#!/usr/bin/env python3
"""
BDFusion 流式处理管线：帧序列批处理 + 视频流式 I/O + 实时显示。

模式:
    frames  — 从 RGB+IR 帧序列文件夹批量处理，输出帧序列
    video   — 从双路视频文件读取，流式处理并编码输出
    camera  — 从双摄像头实时采集，处理并显示（可选录制）

设计:
    生产者-消费者流水线，GPU 推理与磁盘 I/O / 采集完全解耦。
    复用 inference.py 的 Inferen + StreamInferen 推理核心。
"""

import argparse
import os
import sys
import time
import threading
import queue
import collections
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 复用 inference.py 核心 ──
from ldm.util import instantiate_from_config
from inference import (
    Inferen, StreamInferen, get_img, format_process,
    _amp_context, _sync_cuda, _enable_pytorch_deform_attn,
    _pair_rgb_ir, post_process,
)

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


# ═══════════════════════════════════════════════
#  帧预处理工具
# ═══════════════════════════════════════════════

def preprocess_frame_cv2(frame_bgr: np.ndarray, size: int = 256) -> torch.Tensor:
    """
    cv2 BGR 帧 → 归一化 tensor [1, 3, H, W] 在 CPU。
    调用方负责 .to(device)。
    """
    if frame_bgr is None:
        raise ValueError("空帧")
    # BGR → RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # 居中裁剪为正方形
    h, w = rgb.shape[:2]
    crop = min(h, w)
    rgb = rgb[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    rgb = cv2.resize(rgb, (size, size))
    img = (rgb / 127.5 - 1.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    return torch.from_numpy(img)


def preprocess_batch_frames(rgb_frames: List[torch.Tensor],
                            ir_frames: List[torch.Tensor],
                            device: torch.device) -> Dict[str, torch.Tensor]:
    """将多帧拼接为 batch tensor 并传到 GPU。"""
    rgb_batch = torch.cat(rgb_frames, dim=0).to(device=device).contiguous(memory_format=torch.channels_last)
    ir_batch = torch.cat(ir_frames, dim=0).to(device=device).contiguous(memory_format=torch.channels_last)
    return {"rgb": rgb_batch, "infrared": ir_batch}


# ═══════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════

def load_model(config_path: str, ckpt_path: str, device: torch.device,
               sampler: str = "dpmsolver", steps: int = 5, eta: float = 0.0,
               batch_size: int = 1, use_autocast: bool = True,
               profile: bool = False):
    """加载模型并构建 Inferen 推理引擎。"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in state:
        state = state["state_dict"]

    # 蒸馏 checkpoint: 剥离 student. 前缀
    if any(k.startswith('student.') for k in state):
        state = {k[8:] if k.startswith('student.') else k: v for k, v in state.items()}
        print("[权重] 蒸馏 checkpoint，已剥离 student. 前缀。")

    # to_v 权重重映射
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
        print(f"[权重转换] {remapped} 个 to_v 权重从 to_k 复制")

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

    inference_model = Inferen(
        model, steps, eta=eta, batch_size=batch_size,
        use_autocast=use_autocast, profile=profile,
        sampler_name=sampler, use_compile=False,
    )
    stream = StreamInferen(inference_model, model, device, enable_stats=True)
    return model, inference_model, stream


# ═══════════════════════════════════════════════
#  管线 Stage 基类
# ═══════════════════════════════════════════════

class PipelineStage:
    """流水线阶段基类：独立线程 + 输入/输出队列。"""

    def __init__(self, name: str):
        self.name = name
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self):
        raise NotImplementedError


# ═══════════════════════════════════════════════
#  模式1: 帧序列流式处理
# ═══════════════════════════════════════════════

class FrameReaderStage(PipelineStage):
    """从帧序列文件夹预加载帧对到队列。"""

    def __init__(self, rgb_paths: List[str], ir_paths: List[str],
                 out_queue: queue.Queue, device: torch.device,
                 batch_size: int = 1, max_prefetch: int = 8):
        super().__init__("FrameReader")
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.out_queue = out_queue
        self.device = device
        self.batch_size = batch_size
        self.max_prefetch = max_prefetch
        self.total = len(rgb_paths)
        self._idx = 0

    def _run(self):
        try:
            while self._idx < self.total and not self._stop_event.is_set():
                # 组 batch
                end = min(self._idx + self.batch_size, self.total)
                batch_rgb = []
                batch_ir = []
                for i in range(self._idx, end):
                    batch_rgb.append(get_img(self.rgb_paths[i]).to(device=self.device).contiguous(memory_format=torch.channels_last))
                    batch_ir.append(get_img(self.ir_paths[i]).to(device=self.device).contiguous(memory_format=torch.channels_last))

                # 背压: 队列满则等
                while not self._stop_event.is_set():
                    try:
                        self.out_queue.put((self._idx, end, batch_rgb, batch_ir), timeout=0.5)
                        break
                    except queue.Full:
                        continue

                self._idx = end
        finally:
            self.out_queue.put(None)  # 哨兵


class FrameWriterStage(PipelineStage):
    """将处理结果写入磁盘。"""

    def __init__(self, in_queue: queue.Queue, rgb_paths: List[str],
                 out_dir: str, total: int):
        super().__init__("FrameWriter")
        self.in_queue = in_queue
        self.rgb_paths = rgb_paths
        self.out_dir = out_dir
        self.total = total
        self.written = 0

    def _run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        while not self._stop_event.is_set():
            try:
                item = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            start_idx, end_idx, results = item
            for j in range(start_idx, end_idx):
                outpath = os.path.join(self.out_dir, f"enh_{os.path.basename(self.rgb_paths[j])}")
                Image.fromarray(results[j - start_idx].astype(np.uint8)).save(outpath)
                self.written += 1

            self.in_queue.task_done()


def process_frames_streaming(
    stream: StreamInferen,
    model: nn.Module,
    rgb_dir: str,
    ir_dir: str,
    out_dir: str,
    device: torch.device,
    batch_size: int = 1,
    profile: bool = False,
):
    """
    帧序列流式批处理主循环（GPU 线程 + 异步 I/O）。
    """
    rgbs, irs = _pair_rgb_ir(rgb_dir, ir_dir)
    total = len(rgbs)
    print(f"共 {total} 对帧，batch_size={batch_size}，共 { (total+batch_size-1)//batch_size } 批")

    # 队列
    read_queue = queue.Queue(maxsize=4)
    write_queue = queue.Queue(maxsize=4)

    reader = FrameReaderStage(rgbs, irs, read_queue, device, batch_size=batch_size)
    writer = FrameWriterStage(write_queue, rgbs, out_dir, total)

    reader.start()
    writer.start()

    # ── GPU 推理主循环 ──
    stream.reset_stats()
    t_start = time.perf_counter()
    processed = 0
    batches = 0

    with torch.inference_mode(), nullcontext():
        while True:
            item = read_queue.get()
            if item is None:
                write_queue.put(None)
                break

            start_idx, end_idx, batch_rgb, batch_ir = item
            batch = preprocess_batch_frames(batch_rgb, batch_ir, device)

            t0 = time.perf_counter()
            samples = stream.inferen(batch)
            timing = getattr(stream.inferen, "last_timing", {})

            with _amp_context(stream.inferen.use_autocast):
                decoded = model.decode_first_stage(samples)
            t1 = time.perf_counter()

            results = post_process(decoded)

            batch_count = end_idx - start_idx
            processed += batch_count
            batches += 1

            write_queue.put((start_idx, end_idx, results))
            read_queue.task_done()

    reader.join()
    writer.join()
    wall_clock = time.perf_counter() - t_start

    # ── 统计 ──
    print(f"\n流式处理完成: {processed} 帧 / {batches} 批")
    print(f"  总耗时: {wall_clock:.1f}s")
    print(f"  吞吐:   {processed / wall_clock:.2f} fps")
    infer_stats = stream.stats()
    if infer_stats.get("frames", 0) > 0:
        print(f"  推理耗时: {infer_stats['total_s']:.1f}s "
              f"({infer_stats.get('avg_latency_ms', 0):.0f}ms/帧)")
    print(f"  输出: {out_dir}/")


# ═══════════════════════════════════════════════
#  模式2: 视频文件流式处理
# ═══════════════════════════════════════════════

class VideoReaderStage(PipelineStage):
    """从双路视频文件读取帧到队列。"""

    def __init__(self, rgb_video: str, ir_video: str,
                 out_queue: queue.Queue, max_prefetch: int = 4,
                 start_frame: int = 0, end_frame: Optional[int] = None,
                 frame_skip: int = 1):
        super().__init__("VideoReader")
        self.rgb_cap = cv2.VideoCapture(rgb_video)
        self.ir_cap = cv2.VideoCapture(ir_video)
        self.out_queue = out_queue
        self.max_prefetch = max_prefetch
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = max(frame_skip, 1)

        # 验证
        if not self.rgb_cap.isOpened():
            raise RuntimeError(f"无法打开 RGB 视频: {rgb_video}")
        if not self.ir_cap.isOpened():
            raise RuntimeError(f"无法打开 IR 视频: {ir_video}")

        self.rgb_fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        self.ir_fps = self.ir_cap.get(cv2.CAP_PROP_FPS)
        self.rgb_total = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ir_total = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total = min(self.rgb_total, self.ir_total)

        if self.end_frame is not None:
            self.total = min(self.total, self.end_frame)
        self.total = max(0, self.total - self.start_frame)
        self.total = (self.total + self.frame_skip - 1) // self.frame_skip

        print(f"视频信息: RGB {self.rgb_total}帧 @{self.rgb_fps:.1f}fps, "
              f"IR {self.ir_total}帧 @{self.ir_fps:.1f}fps, "
              f"将处理 {self.total} 帧")

    def _run(self):
        try:
            # Seek 到起始帧
            frame_idx = 0
            while frame_idx < self.start_frame:
                self.rgb_cap.grab()
                self.ir_cap.grab()
                frame_idx += 1

            emitted = 0
            while emitted < self.total and not self._stop_event.is_set():
                ret_rgb = self.rgb_cap.grab()
                ret_ir = self.ir_cap.grab()

                if not ret_rgb or not ret_ir:
                    break

                if (frame_idx - self.start_frame) % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                _, rgb_frame = self.rgb_cap.retrieve()
                _, ir_frame = self.ir_cap.retrieve()

                if rgb_frame is None or ir_frame is None:
                    break

                # 背压
                while not self._stop_event.is_set():
                    try:
                        self.out_queue.put((emitted, rgb_frame, ir_frame), timeout=0.5)
                        break
                    except queue.Full:
                        continue

                emitted += 1
                frame_idx += 1
        finally:
            self.rgb_cap.release()
            self.ir_cap.release()
            self.out_queue.put(None)  # 哨兵


class VideoWriterStage(PipelineStage):
    """编码帧为输出视频文件。"""

    def __init__(self, in_queue: queue.Queue, out_path: str,
                 fps: float, size: Tuple[int, int] = (256, 256)):
        super().__init__("VideoWriter")
        self.in_queue = in_queue
        self.out_path = out_path
        self.fps = fps
        self.size = size
        self.written = 0

    def _run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, self.size)
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {self.out_path}")

        try:
            while not self._stop_event.is_set():
                try:
                    item = self.in_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    break

                frame_idx, result = item
                # post_process 输出 [H, W, 3] uint8
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                writer.write(result_bgr)
                self.written += 1
                self.in_queue.task_done()
        finally:
            writer.release()


def process_video_streaming(
    stream: StreamInferen,
    model: nn.Module,
    rgb_video: str,
    ir_video: str,
    out_video: str,
    device: torch.device,
    batch_size: int = 1,
    preview: bool = False,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_skip: int = 1,
    profile: bool = False,
):
    """
    视频流式处理：读帧 → GPU推理 → 编码输出（+ 可选实时预览）。
    """
    # 队列
    read_queue = queue.Queue(maxsize=4)
    write_queue = queue.Queue(maxsize=4)

    reader = VideoReaderStage(rgb_video, ir_video, read_queue,
                              start_frame=start_frame, end_frame=end_frame,
                              frame_skip=frame_skip)
    writer = VideoWriterStage(write_queue, out_video, fps=reader.rgb_fps)

    reader.start()
    writer.start()

    # ── GPU 推理 + 预览 主循环 ──
    stream.reset_stats()
    t_start = time.perf_counter()
    processed = 0
    window_name = "BDFusion Stream" if preview else None
    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEP_RATIO)

    rgb_buf: List[torch.Tensor] = []
    ir_buf: List[torch.Tensor] = []
    idx_buf: List[int] = []
    rgb_orig_buf: List[np.ndarray] = []

    def _flush_batch():
        nonlocal processed
        if not rgb_buf:
            return
        batch = preprocess_batch_frames(rgb_buf, ir_buf, device)
        samples = stream.inferen(batch)
        with _amp_context(stream.inferen.use_autocast):
            decoded = model.decode_first_stage(samples)
        results = post_process(decoded)

        for j, ridx in enumerate(idx_buf):
            write_queue.put((ridx, results[j]))
            if preview:
                display = np.hstack([rgb_orig_buf[j], results[j]])
                cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        processed += len(idx_buf)
        rgb_buf.clear()
        ir_buf.clear()
        idx_buf.clear()
        rgb_orig_buf.clear()

    try:
        with torch.inference_mode(), nullcontext():
            while True:
                item = read_queue.get()
                if item is None:
                    _flush_batch()
                    write_queue.put(None)
                    break

                frame_idx, rgb_frame, ir_frame = item
                rgb_t = preprocess_frame_cv2(rgb_frame)
                ir_t = preprocess_frame_cv2(ir_frame)

                rgb_buf.append(rgb_t)
                ir_buf.append(ir_t)
                idx_buf.append(frame_idx)
                if preview:
                    rgb_orig_buf.append(preprocess_frame_cv2(rgb_frame))

                if len(rgb_buf) >= batch_size:
                    _flush_batch()

                read_queue.task_done()

                if preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[用户中断]")
                    reader.stop()
                    break
    finally:
        if preview:
            cv2.destroyAllWindows()

    reader.join()
    writer.join()
    wall_clock = time.perf_counter() - t_start

    print(f"\n视频处理完成: {processed} 帧")
    print(f"  总耗时: {wall_clock:.1f}s")
    print(f"  吞吐:   {processed / wall_clock:.2f} fps")
    print(f"  输出: {out_video}")


# ═══════════════════════════════════════════════
#  模式3: 实时摄像头流式处理
# ═══════════════════════════════════════════════

class CameraCapture:
    """双摄像头采集封装。"""

    def __init__(self, cam_rgb: int = 0, cam_ir: int = 1,
                 width: int = 640, height: int = 480):
        self.rgb_cap = cv2.VideoCapture(cam_rgb)
        self.ir_cap = cv2.VideoCapture(cam_ir)
        for cap in (self.rgb_cap, self.ir_cap):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.rgb_ok = self.rgb_cap.isOpened()
        self.ir_ok = self.ir_cap.isOpened()
        if not self.rgb_ok:
            raise RuntimeError(f"无法打开 RGB 摄像头 #{cam_rgb}")
        if not self.ir_ok:
            raise RuntimeError(f"无法打开 IR 摄像头 #{cam_ir}")

        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_ir: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0
        self._frames = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="Camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.rgb_cap.release()
        self.ir_cap.release()

    def _capture_loop(self):
        t0 = time.perf_counter()
        while self._running:
            ret_rgb = self.rgb_cap.grab()
            ret_ir = self.ir_cap.grab()
            if not ret_rgb or not ret_ir:
                time.sleep(0.01)
                continue

            _, rgb = self.rgb_cap.retrieve()
            _, ir = self.ir_cap.retrieve()
            if rgb is None or ir is None:
                continue

            with self._lock:
                self._latest_rgb = rgb
                self._latest_ir = ir
                self._frames += 1

            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                self._fps = self._frames / elapsed
                self._frames = 0
                t0 = time.perf_counter()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._latest_rgb, self._latest_ir

    @property
    def capture_fps(self) -> float:
        return self._fps


def process_camera_streaming(
    stream: StreamInferen,
    model: nn.Module,
    device: torch.device,
    cam_rgb: int = 0,
    cam_ir: int = 1,
    record_path: Optional[str] = None,
    profile: bool = False,
):
    """
    实时摄像头处理：采集 → 推理 → 显示 + 可选录制。
    按 'q' 退出，按 's' 截图，按 'r' 切换录制。
    """
    camera = CameraCapture(cam_rgb, cam_ir)
    camera.start()

    recorder: Optional[cv2.VideoWriter] = None
    recording = False
    if record_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorder = cv2.VideoWriter(record_path, fourcc, 15.0, (256, 256))
        recording = True
        print(f"[录制] 开始录制 → {record_path}")

    window_name = "BDFusion Live | q=退出 s=截图 r=录制"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEP_RATIO)

    stream.reset_stats()
    t_start = time.perf_counter()
    frames = 0
    screenshot_dir = os.path.join(os.path.dirname(record_path) if record_path else ".", "screenshots")
    screenshot_idx = 0

    print(f"\n实时处理中... ({window_name})")
    print(f"  推理 @ {stream.inferen.steps}步 | 采集 fps: 实时")
    print("-" * 40)

    try:
        with torch.inference_mode(), nullcontext():
            while True:
                rgb, ir = camera.get_latest()
                if rgb is None or ir is None:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                # 预处理 + 推理
                rgb_t = preprocess_frame_cv2(rgb).to(device=device).contiguous(memory_format=torch.channels_last)
                ir_t = preprocess_frame_cv2(ir).to(device=device).contiguous(memory_format=torch.channels_last)

                result = stream.process(rgb_t, ir_t)
                frames += 1

                # 构建显示画面: 原始RGB | 处理结果
                rgb_disp = cv2.resize(rgb, (256, 256))
                disp_top = np.hstack([rgb_disp, cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)])

                # 信息叠加
                elapsed = time.perf_counter() - t_start
                infer_fps = frames / elapsed if elapsed > 0 else 0
                info_line = (f"Infer: {infer_fps:.1f} fps | "
                             f"Capture: {camera.capture_fps:.1f} fps | "
                             f"Frames: {frames} | "
                             f"REC" if recording else "")
                disp = cv2.copyMakeBorder(disp_top, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                cv2.putText(disp, info_line, (10, disp_top.shape[0] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow(window_name, disp)

                if recording and recorder:
                    result_bgr = cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)
                    recorder.write(result_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    os.makedirs(screenshot_dir, exist_ok=True)
                    spath = os.path.join(screenshot_dir, f"screenshot_{screenshot_idx:04d}.png")
                    cv2.imwrite(spath, cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR))
                    print(f"[截图] {spath}")
                    screenshot_idx += 1
                elif key == ord('r'):
                    recording = not recording
                    if recording and recorder is None:
                        rec_path = record_path or f"recording_{int(time.time())}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        recorder = cv2.VideoWriter(rec_path, fourcc, 15.0, (256, 256))
                        print(f"[录制] 开始 → {rec_path}")
                    elif not recording and recorder:
                        recorder.release()
                        recorder = None
                        print("[录制] 停止")

    finally:
        camera.stop()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()

    wall_clock = time.perf_counter() - t_start
    print(f"\n实时处理结束: {frames} 帧")
    print(f"  总耗时: {wall_clock:.1f}s")
    if frames > 0:
        print(f"  平均吞吐: {frames / wall_clock:.2f} fps")


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BDFusion 流式处理管线: 帧序列 / 视频 / 实时摄像头",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 帧序列模式
  python stream_process.py --mode frames \\
      --rgb_dir experiments/instance/rgb --ir_dir experiments/instance/ir --outdir experiments/output_stream

  # 视频模式
  python stream_process.py --mode video \\
      --rgb_video input_rgb.mp4 --ir_video input_ir.mp4 --out_video output.mp4 --preview

  # 摄像头模式
  python stream_process.py --mode camera --cam_rgb 0 --cam_ir 1 --record
        """,
    )

    # 模式
    parser.add_argument("--mode", type=str, default="frames",
                        choices=["frames", "video", "camera"],
                        help="处理模式 (默认: frames)")

    # 模型
    parser.add_argument("--config", type=str,
                        default=os.path.join(_PROJ_ROOT, 'informationSynthesis', 'Image2Image.yaml'),
                        help="模型配置文件")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(_PROJ_ROOT, 'checkpoint', 'last.ckpt'),
                        help="模型权重路径")
    parser.add_argument("--steps", type=int, default=5,
                        help="采样步数 (5 推荐)")
    parser.add_argument("--sampler", type=str, default="dpmsolver",
                        choices=["ddim", "ddim-opt", "dpmsolver", "dpmsolver-1S",
                                 "dpmsolver-2S", "dpmsolver-graph", "plms"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理批大小 (视频/摄像头模式下建议=1)")
    parser.add_argument("--precision", type=str, default="autocast",
                        choices=["fp32", "autocast"])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--profile", action="store_true",
                        help="详细计时")

    # 帧序列
    parser.add_argument("--rgb_dir", type=str, default="./experiments/instance/rgb")
    parser.add_argument("--ir_dir", type=str, default="./experiments/instance/ir")
    parser.add_argument("--outdir", type=str, default="./experiments/output_stream")

    # 视频
    parser.add_argument("--rgb_video", type=str, default=None)
    parser.add_argument("--ir_video", type=str, default=None)
    parser.add_argument("--out_video", type=str, default="./experiments/output_stream.mp4")
    parser.add_argument("--preview", action="store_true",
                        help="实时显示预览窗口")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--frame_skip", type=int, default=1)

    # 摄像头
    parser.add_argument("--cam_rgb", type=int, default=0)
    parser.add_argument("--cam_ir", type=int, default=1)
    parser.add_argument("--record", action="store_true",
                        help="自动开始录制 (按 'r' 切换)")

    opt = parser.parse_args()

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = opt.precision == "autocast" and device.type == "cuda"

    # ── 随机种子 ──
    import random
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(opt.seed)

    # ── 加载模型 ──
    print(f"加载模型: {opt.checkpoint}")
    model, inferen, stream = load_model(
        opt.config, opt.checkpoint, device,
        sampler=opt.sampler, steps=opt.steps, eta=opt.eta,
        batch_size=opt.batch_size, use_autocast=use_autocast,
        profile=opt.profile,
    )

    # ── 预热 ──
    print("预热 CUDA kernel cache...")
    stream.warmup(steps=3)
    # 预热后重置随机种子，保证推理噪声与 inference.py 一致
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(opt.seed)

    # ── 分发模式 ──
    if opt.mode == "frames":
        process_frames_streaming(
            stream, model,
            rgb_dir=opt.rgb_dir, ir_dir=opt.ir_dir,
            out_dir=opt.outdir, device=device,
            batch_size=opt.batch_size, profile=opt.profile,
        )

    elif opt.mode == "video":
        if not opt.rgb_video or not opt.ir_video:
            parser.error("视频模式需要 --rgb_video 和 --ir_video")
        if not os.path.isfile(opt.rgb_video):
            parser.error(f"RGB 视频不存在: {opt.rgb_video}")
        if not os.path.isfile(opt.ir_video):
            parser.error(f"IR 视频不存在: {opt.ir_video}")

        process_video_streaming(
            stream, model,
            rgb_video=opt.rgb_video, ir_video=opt.ir_video,
            out_video=opt.out_video, device=device,
            batch_size=opt.batch_size, preview=opt.preview,
            start_frame=opt.start_frame, end_frame=opt.end_frame,
            frame_skip=opt.frame_skip, profile=opt.profile,
        )

    elif opt.mode == "camera":
        record_path = f"recording_{int(time.time())}.mp4" if opt.record else None
        process_camera_streaming(
            stream, model, device,
            cam_rgb=opt.cam_rgb, cam_ir=opt.cam_ir,
            record_path=record_path, profile=opt.profile,
        )


if __name__ == "__main__":
    main()
