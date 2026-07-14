import argparse, os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from ldm.util import instantiate_from_config
import random
import cv2
from typing import List, Optional, Dict, Tuple, Union
import time
from contextlib import nullcontext

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Fast samplers ──
from informationSynthesis.fast_samplers import DPMSolverSampler, OptimizedDDIMSampler
from informationSynthesis.I2Iddim import DDIMSampler

# ── Torch compile helpers ──
_compiled_fns: Dict[str, callable] = {}
_COMPILE_AVAILABLE = hasattr(torch, 'compile')


def _cached_compile(fn, key: str, **compile_kwargs):
    """Compile a function once and cache the result."""
    if key not in _compiled_fns:
        if _COMPILE_AVAILABLE:
            _compiled_fns[key] = torch.compile(fn, **compile_kwargs)
        else:
            _compiled_fns[key] = fn
    return _compiled_fns[key]


# ── Image I/O helpers ──

def crop_to_size(img, crop):
    h, w, = img.shape[:2]
    img = img[(h - crop) // 2:(h + crop) // 2,
         (w - crop) // 2:(w + crop) // 2]
    return img


def format_process(img):
    img = (img / 127.5 - 1.0).astype(np.float32)
    img = img.transpose(2, 0, 1)[None]
    return torch.from_numpy(img)


def get_img(img_path):
    img = Image.open(img_path)
    img = np.array(img).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    min_size = min(img.shape[0], img.shape[1])
    img = crop_to_size(img, min_size)
    img = cv2.resize(img, (256, 256))
    return format_process(img)


def make_batch(rgb_path, infrared_path, device):
    return {
        "rgb": get_img(rgb_path).to(device=device).contiguous(memory_format=torch.channels_last),
        "infrared": get_img(infrared_path).to(device=device).contiguous(memory_format=torch.channels_last)
    }


def post_process(img):
    img = img.float().detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) * 127.5
    img = img.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
    return img


def _sync_cuda(enabled=False):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _amp_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _has_ms_deform_attn(model) -> bool:
    for m in model.modules():
        if m.__class__.__name__ == "MSDeformAttn":
            return True
    return False


def _enable_pytorch_deform_attn(model) -> int:
    """Switch all MSDeformAttn to pure PyTorch (FP16/compatible, torch.compile ready)."""
    count = 0
    for m in model.modules():
        if m.__class__.__name__ == "MSDeformAttn":
            m.use_pytorch_impl = True
            count += 1
    return count


def _convert_result_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Fast tensor-to-numpy with no intermediate copies."""
    x = tensor.clamp(-1.0, 1.0)
    x = (x + 1.0) * 127.5
    x = x.permute(0, 2, 3, 1).to(torch.uint8)
    return x.cpu().numpy()


def make_batch_from_pairs(rgb_paths, infrared_paths, device):
    rgb_batch = torch.cat([get_img(path) for path in rgb_paths], dim=0).to(device=device).contiguous(memory_format=torch.channels_last)
    infrared_batch = torch.cat([get_img(path) for path in infrared_paths], dim=0).to(device=device).contiguous(memory_format=torch.channels_last)
    return {"rgb": rgb_batch, "infrared": infrared_batch}


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
        raise RuntimeError("未找到任何同名文件对，请确保两个目录下文件名一致。")
    miss_rgb = sorted(set(ir_map) - set(rgb_map))
    miss_ir = sorted(set(rgb_map) - set(ir_map))
    if miss_rgb or miss_ir:
        lines = []
        if miss_rgb:
            lines.append(f"缺少 RGB 文件: {miss_rgb[:5]}{' ...' if len(miss_rgb)>5 else ''}")
        if miss_ir:
            lines.append(f"缺少 IR 文件: {miss_ir[:5]}{' ...' if len(miss_ir)>5 else ''}")
        raise RuntimeError("RGB/IR 文件名不完全匹配:\n" + "\n".join(lines))
    return [rgb_map[k] for k in common], [ir_map[k] for k in common]


# ═══════════════════════════════════════════════
#  Inferen – 优化推理引擎
# ═══════════════════════════════════════════════

class Inferen(nn.Module):
    """
    Unified inference engine supporting multiple samplers, precision modes,
    and optional torch.compile for the diffusion UNet.
    
    Samplers:
        - ddim: Classic DDIM with configurable steps and eta
        - ddim-opt: Optimized DDIM with fused operations
        - dpmsolver: DPM-Solver++ 2S (best quality/speed tradeoff)
        - dpmsolver-1S: DPM-Solver++ 1S (fastest, slightly lower quality)
    """
    
    def __init__(self, model, steps, eta=0.0, batch_size=1, 
                 use_autocast=True, profile=False, sampler_name="ddim",
                 plms_sampler=None, use_compile=False):
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
        self._compiled_apply = None
        self._init_sampler()
        
    def _init_sampler(self):
        """Initialize sampler based on name."""
        if self.sampler_name in ("dpmsolver", "dpmsolver-2S", "dpmsolver-1S", "dpmsolver-graph"):
            order = 1 if self.sampler_name in ("dpmsolver-1S", "dpmsolver-graph") else 2
            self._fast_sampler = DPMSolverSampler(self.model)
            self._fast_sampler._order = order
            self._use_cuda_graph = (self.sampler_name == "dpmsolver-graph")
        elif self.sampler_name == "ddim-opt":
            self._fast_sampler = OptimizedDDIMSampler(self.model)
            self._use_cuda_graph = False
        else:
            self._fast_sampler = None
            self._use_cuda_graph = False

    def _is_half_unsupported_error(self, err: RuntimeError) -> bool:
        msg = str(err)
        return "ms_deform_attn_forward_cuda" in msg and "Half" in msg

    def _get_compiled_apply_model(self):
        """Get or create a torch.compiled version of apply_model for the diffusion backbone."""
        if self._compiled_apply is not None:
            return self._compiled_apply
        if self.use_compile and _COMPILE_AVAILABLE:
            fn = self.model.apply_model
            self._compiled_apply = torch.compile(fn, mode="reduce-overhead", fullgraph=False)
        else:
            self._compiled_apply = self.model.apply_model
        return self._compiled_apply

    def forward(self, batch, z_init=None):
        """Forward pass with optional shared initial noise for temporal consistency."""
        _sync_cuda(self.profile)
        t0 = time.perf_counter()
        try:
            with _amp_context(self.use_autocast):
                cond_inputs = torch.cat([batch["rgb"], batch["infrared"]], dim=0)
                cond_outputs = self.model.get_learned_conditioning(cond_inputs)
                c1, c2 = cond_outputs.chunk(2, dim=0)
                _sync_cuda(self.profile)
                t1 = time.perf_counter()

                samples_ddim = self._run_sampler(c1, c2, z_init=z_init)

        except RuntimeError as err:
            if self.use_autocast and self._is_half_unsupported_error(err):
                self.use_autocast = False
                if not self._autocast_fallback_warned:
                    print("[提示] 检测到 ms_deform_attn 不支持 FP16，已自动回退为 FP32 推理。")
                    self._autocast_fallback_warned = True
                self._compiled_apply = None  # invalidate compiled cache

                cond_inputs = torch.cat([batch["rgb"], batch["infrared"]], dim=0)
                cond_outputs = self.model.get_learned_conditioning(cond_inputs)
                c1, c2 = cond_outputs.chunk(2, dim=0)
                _sync_cuda(self.profile)
                t1 = time.perf_counter()
                samples_ddim = self._run_sampler(c1, c2, z_init=z_init)
            else:
                raise

        _sync_cuda(self.profile)
        t2 = time.perf_counter()
        self.last_timing = {
            "conditioning_s": t1 - t0,
            "sampling_s": t2 - t1,
            "inference_s": t2 - t0,
        }
        return samples_ddim

    def _run_sampler(self, c1, c2, z_init=None):
        """Run the configured sampler with optional shared initial noise."""
        shape = c1.shape[1:]
        
        # Pre-compute degradation task tokens ONCE for all steps
        task_tokens = self.model.get_degradation_tokens(c1, c2)

        # Fast samplers (DPM-Solver / Optimized DDIM)
        if self.sampler_name in ("dpmsolver", "dpmsolver-2S", "dpmsolver-1S", "dpmsolver-graph"):
            order = 1 if self.sampler_name in ("dpmsolver-1S", "dpmsolver-graph") else 2
            if self.use_compile and self._compiled_apply is None:
                self._get_compiled_apply_model()
            samples, _ = self._fast_sampler.sample(
                cond1=c1, cond2=c2,
                batch_size=self.batch_size,
                shape=shape, steps=self.steps,
                order=order, eta=self.eta, verbose=False,
                task_tokens=task_tokens,
                use_cuda_graph=self._use_cuda_graph,
                z_init=z_init,
            )
            return samples

        elif self.sampler_name == "ddim-opt":
            samples, _ = self._fast_sampler.sample(
                cond1=c1, cond2=c2,
                batch_size=self.batch_size,
                shape=shape, steps=self.steps,
                eta=self.eta, verbose=False,
                task_tokens=task_tokens,
                z_init=z_init,
            )
            return samples

        elif self.sampler_name == "plms" and self.plms is not None:
            samples_ddim, _ = self.plms.sample(
                S=self.steps,
                batch_size=self.batch_size,
                shape=shape,
                conditioning=(c1, c2),
                eta=self.eta,
                verbose=False,
            )
            return samples_ddim

        else:  # default DDIM
            samples_ddim, _ = self.model.sample_log(
                cond1=c1, cond2=c2,
                batch_size=self.batch_size,
                ddim=True, ddim_steps=self.steps,
                eta=self.eta,
            )
            return samples_ddim


# ═══════════════════════════════════════════════
#  StreamInferen – 流式推理接口
# ═══════════════════════════════════════════════

class StreamInferen:
    """
    Streaming inference API for continuous frame processing.
    
    Usage::
        stream = StreamInferen(model_wrapper, device)
        stream.warmup()
        for rgb, ir in frame_generator:
            result = stream.process(rgb, ir)  # single-frame or batch
        
        stats = stream.stats()
    """
    
    def __init__(self, model_wrapper: Inferen, model: nn.Module, device: torch.device,
                 enable_stats: bool = True):
        self.inferen = model_wrapper
        self.model = model
        self.device = device
        self.enable_stats = enable_stats
        
        # Stats
        self._n_frames = 0
        self._total_time = 0.0
        self._latencies = []
        
        # Warmup flag
        self._warmed_up = False
    
    def warmup(self, steps: int = 3):
        """Run a few forward passes to warm up CUDA kernel cache and torch.compile."""
        if self._warmed_up:
            return
        
        bs = self.inferen.batch_size
        print(f"[StreamInferen] 预热中 (batch_size={bs})...")
        dummy_rgb = torch.randn(bs, 3, 256, 256, device=self.device).contiguous(memory_format=torch.channels_last)
        dummy_ir = torch.randn(bs, 3, 256, 256, device=self.device).contiguous(memory_format=torch.channels_last)
        batch = {"rgb": dummy_rgb, "infrared": dummy_ir}
        
        for _ in range(steps):
            self.inferen(batch)
            torch.cuda.synchronize()
        
        self._warmed_up = True
        print("[StreamInferen] 预热完成。")
    
    def process(self, rgb_tensor: torch.Tensor, ir_tensor: torch.Tensor,
                z_init: torch.Tensor = None) -> np.ndarray:
        """
        Process a single frame pair.
        
        Args:
            rgb_tensor: [1, 3, H, W] or [B, 3, H, W] on device
            ir_tensor:  [1, 3, H, W] or [B, 3, H, W] on device
            z_init:     Optional [1, C, H_latent, W_latent] shared initial noise
                        for temporal noise consistency across frames
        Returns:
            np.ndarray [B, H, W, 3] uint8
        """
        batch = {"rgb": rgb_tensor.contiguous(memory_format=torch.channels_last),
                  "infrared": ir_tensor.contiguous(memory_format=torch.channels_last)}
        
        t0 = time.perf_counter()
        samples = self.inferen(batch, z_init=z_init)
        
        with _amp_context(self.inferen.use_autocast):
            decoded = self.model.decode_first_stage(samples)
        
        result = _convert_result_to_rgb(decoded)
        elapsed = time.perf_counter() - t0
        
        if self.enable_stats:
            self._n_frames += rgb_tensor.shape[0]
            self._total_time += elapsed
            self._latencies.append(elapsed)
        
        return result
    
    def stats(self) -> dict:
        if self._n_frames == 0:
            return {"frames": 0}
        return {
            "frames": self._n_frames,
            "total_s": self._total_time,
            "fps": self._n_frames / self._total_time if self._total_time > 0 else 0,
            "avg_latency_ms": (self._total_time / self._n_frames) * 1000,
            "p99_latency_ms": np.percentile(self._latencies, 99) * 1000 if self._latencies else 0,
        }
    
    def reset_stats(self):
        self._n_frames = 0
        self._total_time = 0.0
        self._latencies.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", default="./experiments/instance/rgb", type=str, nargs="?", help="dir containing rgb images")
    parser.add_argument("--ir_path", default="./experiments/instance/ir", type=str, nargs="?", help="dir containing infrared images")
    parser.add_argument("--outdir", default='./experiments/instance/output', type=str, nargs="?", help="dir to write results to")
    parser.add_argument("--config", default=os.path.join(_PROJ_ROOT, 'informationSynthesis', 'Image2Image.yaml'), type=str, nargs="?", help="config file")
    parser.add_argument("--checkpoint", default=os.path.join(_PROJ_ROOT, 'checkpoint', 'last.ckpt'), type=str, nargs="?", help="checkpoint path")
    parser.add_argument("--steps", type=int, default=5, help="number of sampling steps (5 recommended for quality/speed balance)")
    parser.add_argument("--sampler", type=str, default="dpmsolver",
                        choices=["ddim", "ddim-opt", "dpmsolver", "dpmsolver-1S", "dpmsolver-2S", "dpmsolver-graph", "plms"],
                        help="sampler (default: dpmsolver=DPM-Solver++ 2S)")
    parser.add_argument("--eta", type=float, default=0.0, help="sampler stochasticity (0=deterministic)")
    parser.add_argument("--batch_size", type=int, default=1, help="inference batch size")
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--precision", type=str, default="autocast",
                        choices=["fp32", "autocast"], help="precision; autocast=FP16 on CUDA")
    parser.add_argument("--profile", action="store_true", help="detailed stage timing")
    parser.add_argument("--compile", action="store_true", help="use torch.compile on UNet (PyTorch>=2.0)")
    parser.add_argument("--warmup", type=int, default=0, help="number of warmup iterations before timing")
    opt = parser.parse_args()

    rgb_dir, ir_dir = opt.rgb_path, opt.ir_path
    rgbs, infrareds = _pair_rgb_ir(rgb_dir, ir_dir)
    print(f"共找到 {len(rgbs)} 对匹配的输入图像。")

    # ── Load model ──
    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    ckpt_path = opt.checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in state:
        state = state["state_dict"]
    # Handle distill checkpoint: strip 'student.' prefix
    if any(k.startswith('student.') for k in state):
        state = {k[8:] if k.startswith('student.') else k: v for k, v in state.items()}
        print("[权重] 蒸馏 checkpoint，已剥离 student. 前缀。")
    # Remap to_v
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
        print(f"[权重转换] {remapped} 个 to_v 权重从 to_k 复制（Deformable→Standard CrossAttn）")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[警告] 缺少 {len(missing)} 个权重键（示例）: {missing[:5]}")
    if unexpected:
        print(f"[警告] 未使用 {len(unexpected)} 个权重键（示例）: {unexpected[:5]}")

    # ── Device & precision setup ──
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    use_autocast = opt.precision == "autocast" and device.type == "cuda"
    n_deform = _enable_pytorch_deform_attn(model)
    if n_deform > 0:
        print(f"[优化] {n_deform} 个 MSDeformAttn 已切换为纯 PyTorch (冗余，attention已替换)")

    # ── Sampler setup ──
    sampler_name = opt.sampler
    plms_sampler = None
    if sampler_name == "plms":
        try:
            from ldm.models.diffusion.plms import PLMSSampler
            plms_sampler = PLMSSampler(model)
        except Exception as e:
            print(f"[警告] PLMS 不可用: {e}; 回退为 DDIM")
            sampler_name = "ddim"

    use_compile = False  # VAE/UNet compile unstable in this codebase; skip for now

    inference_model = Inferen(
        model, opt.steps, eta=opt.eta, batch_size=opt.batch_size,
        use_autocast=use_autocast, profile=opt.profile,
        sampler_name=sampler_name, plms_sampler=plms_sampler,
        use_compile=use_compile,
    )

    os.makedirs(opt.outdir, exist_ok=True)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # ── Stream inferen wrapper ──
    stream = StreamInferen(inference_model, model, device, enable_stats=True)

    # Warmup
    if opt.warmup > 0 or use_compile:
        stream.warmup(steps=max(opt.warmup, 3))

    stats = {
        "preprocess_s": 0.0, "conditioning_s": 0.0,
        "sampling_s": 0.0, "decode_s": 0.0, "post_s": 0.0, "total_s": 0.0,
    }
    num_images = len(rgbs)

    with torch.inference_mode():
        # Skip EMA scope when using standard CrossAttention (model_ema missing to_v keys)
        with nullcontext():
            overall_start = time.perf_counter()
            for start in tqdm(range(0, len(rgbs), opt.batch_size), total=(len(rgbs) + opt.batch_size - 1) // opt.batch_size):
                end = min(start + opt.batch_size, len(rgbs))
                batch_rgbs, batch_infrareds = rgbs[start:end], infrareds[start:end]

                _sync_cuda(opt.profile)
                t0 = time.perf_counter()
                batch = make_batch_from_pairs(batch_rgbs, batch_infrareds, device=device)
                _sync_cuda(opt.profile)
                t1 = time.perf_counter()

                samples_ddim = inference_model(batch)
                timing = getattr(inference_model, "last_timing", {})

                _sync_cuda(opt.profile)
                t2 = time.perf_counter()
                with _amp_context(inference_model.use_autocast):
                    decoded = model.decode_first_stage(samples_ddim)
                _sync_cuda(opt.profile)
                t3 = time.perf_counter()

                output_img = post_process(decoded)
                for idx, rgb in enumerate(batch_rgbs):
                    outpath = os.path.join(opt.outdir, f"enh_{os.path.basename(rgb)}")
                    Image.fromarray(output_img[idx].astype(np.uint8)).save(outpath)
                t4 = time.perf_counter()

                batch_count = len(batch_rgbs)
                if opt.profile:
                    stats["preprocess_s"] += (t1 - t0) / batch_count
                    stats["conditioning_s"] += timing.get("conditioning_s", 0.0) / batch_count
                    stats["sampling_s"] += timing.get("sampling_s", 0.0) / batch_count
                    stats["decode_s"] += (t3 - t2) / batch_count
                    stats["post_s"] += (t4 - t3) / batch_count
                stats["total_s"] += (t4 - t0) / batch_count

    wall_clock = time.perf_counter() - overall_start

    if num_images > 0:
        print("\n推理性能统计:")
        print(f"  - 采样器: {sampler_name.upper()}")
        print(f"  - 步数: {opt.steps}")
        print(f"  - 精度: {opt.precision}")
        print(f"  - torch.compile: {'ON' if use_compile else 'OFF'}")
        print(f"  - 计时: {'详细阶段计时' if opt.profile else '仅总耗时'}")
        if opt.profile:
            print(f"  - 平均预处理:   {stats['preprocess_s'] / num_images * 1000:.2f} ms/张")
            print(f"  - 平均条件编码: {stats['conditioning_s'] / num_images * 1000:.2f} ms/张")
            print(f"  - 平均采样:     {stats['sampling_s'] / num_images * 1000:.2f} ms/张")
            print(f"  - 平均解码:     {stats['decode_s'] / num_images * 1000:.2f} ms/张")
            print(f"  - 平均后处理:   {stats['post_s'] / num_images * 1000:.2f} ms/张")
        print(f"  - 平均总耗时:   {stats['total_s'] / num_images * 1000:.2f} ms/张")
        print(f"  - 整体吞吐:     {num_images / wall_clock:.2f} 张/秒")
    print(f"推理完成，结果已保存到 {opt.outdir}")

