# BDFusion: Blind Degradation-Aware Visible-Infrared Fusion

Official implementation of **BDFusion** — a blind all-in-one diffusion model for severe low-light image enhancement using RGB + Infrared (IR) fusion with degradation-aware conditioning.

## Features

- **Blind Degradation Sensing** — automatically extracts degradation patterns from input images; no task-ID labels needed
- **RGB+IR Fusion** — dual-stream architecture fuses visible and infrared modalities via cross-attention
- **Fast Inference** — DPM-Solver++ at 5 steps achieves ~258 ms/image (3.9 FPS) on RTX 3090
- **All-in-One** — single model handles low-light, fog, rain, snow, and mixed degradations

## Project Structure

```
├── inference.py              ← Main inference entry point
├── train_model.py            ← Training entry point
├── inference_and_evaluate.py ← Inference + automatic evaluation
├── inference_multi_task.py   ← Multi-task batch inference
├── process_video_enhanced.py ← Enhanced video processing
├── stream_process.py         ← Streaming pipeline (frames/video/camera)
├── stream_realtime_sim.py    ← Real-time inference simulation
├── online_image_demo.py      ← Online image-by-image demo
├── finetune_lite.py          ← Lite model fine-tuning
├── distill_model.py          ← Progressive distillation
├── evaluate_fusion.py        ← Standalone evaluation script
├── run_inference_evaluate.sh ← Quick launch script
├── informationSynthesis/     ← Core model: UNet, diffusion, degradation encoder, attention
├── ldm/                      ← Latent Diffusion Model library (modified from CompVis)
├── utils/                    ← Evaluator, loss functions, fusion utilities
├── scripts/                  ← Reference scripts from original VIIS project
├── tests/                    ← Unit/functional tests
├── docs/diagrams/            ← Architecture diagrams (.drawio)
├── experiments/              ← Experimental outputs (gitignored)
├── data/                     ← Datasets (gitignored)
├── checkpoint/               ← Model weights (gitignored)
└── models/                   ← Pretrained first-stage models (gitignored)
```

## Quick Start

### 1. Environment

```bash
# Create conda environment (based on CompVis latent-diffusion)
conda env create -f environment.yaml
conda activate ldm

# Install deformable attention (from Deformable-DETR)
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops && sh make.sh
```

### 2. Download Resources

The following files are **not** included in this repository due to size limits.
Download them and place at the specified paths.

| Resource | Size | Download | Target Path |
|----------|------|----------|-------------|
| BDFusion main model | 8.6 GB | [Google Drive](https://drive.google.com/file/d/1_G79pvtnfq_RKLzh3sJ42bsPocnlxICP/view) | `./checkpoint/last.ckpt` |
| BDFusion distilled model | 5.6 GB | [Google Drive](https://drive.google.com/file/d/1gfAJd5wqgUV_0qYrNC9313gByFELQFhw/view) | `./checkpoint/bdfusion-distill.ckpt` |
| VQ-F4 first-stage | 722 MB | [CompVis](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) | `./models/first_stage_models/vq-f4/model.ckpt` |
| MSRS-BD-F dataset | — | [Google Drive](https://drive.google.com/file/d/1Jne11xdmFLGiAo2qJRteOm6recoevxJ8/view) | `./data/` (extract here) |

After downloading, verify the file structure:

```
checkpoint/
├── last.ckpt                  ← main model (8.6G)
└── bdfusion-distill.ckpt      ← distilled model (5.6G)
models/
└── first_stage_models/
    └── vq-f4/
        └── model.ckpt         ← VQ-F4 (722M)
data/
├── msrs_bd-f/                 ← training dataset
├── test_imgs/                 ← test images (multi-scene)
└── test_videos/               ← test video clips (RGB + IR pairs)
```

### 3. Run Inference

```bash
# Default: DPM-Solver++ 5-step, ~258 ms/image
python inference.py

# With custom paths
python inference.py \
    --rgb_path ./experiments/instance/rgb \
    --ir_path  ./experiments/instance/ir \
    --outdir   ./experiments/instance/output \
    --steps 5 --sampler dpmsolver
```

## Performance

| Configuration | Latency | FPS | Speedup |
|--------------|---------|-----|---------|
| DDIM 20-step (baseline) | 1315 ms | 0.76 | 1.0× |
| DPM-Solver++ 5-step | 425 ms | 2.35 | 3.1× |
| **5-step + Standard CrossAttn** | **258 ms** | **3.87** | **5.1×** |
| batch=4 (effective) | 33 ms | 7.5 | ~40× |

*Benchmarked on NVIDIA RTX 3090, PyTorch 2.4, FP16 autocast.*

Key optimizations:
- **DPM-Solver++** replaces DDIM with 5-step high-quality sampling
- **Standard CrossAttention** replaces DeformableCrossAttention in UNet (1.4× UNet speedup)
- **Encoder sharing** — RGB+IR batched encoding saves 15 ms
- **DSDE caching** — degradation encoder runs once, reused across denoising steps

## Evaluation

```python
from utils.Evaluator import image_read_cv2, EN, PSNR, SSIM, MI, VIFF

gt     = image_read_cv2('./path/to/gt_clean.png')
result = image_read_cv2('./path/to/output.png')

print(f"PSNR: {PSNR(gt, result):.2f} dB")
print(f"SSIM: {SSIM(gt, result):.4f}")
print(f"EN:   {EN(result):.4f}")
```

Available metrics: `EN`, `SD`, `AG`, `SF` (no-reference), `PSNR`, `SSIM`, `MI` (with GT), `VIFF`, `Qabf` (fusion quality).

Run full evaluation pipeline:
```bash
bash run_inference_evaluate.sh
```

## Architecture

```
Degraded RGB ─┐
              ├─→ DegradationEncoder ─→ Task Tokens ─┐
Degraded IR  ─┘                                      │
                                                     ▼
                                          UNet Cross-Attention
                                                     │
              ┌──────────────────────────────────────┘
              ▼
       Latent Diffusion (DPM-Solver++) ─→ Enhanced RGB Output
```

The model uses **blind degradation encoding**: a CNN/ResNet encoder extracts degradation features directly from image content, producing task tokens that condition the UNet via cross-attention — no explicit task labels required.

See `docs/diagrams/BDFusion_Architecture.drawio` for the full architecture diagram.

## Citation

If you use this work, please cite:

```bibtex
@article{bdfusion2025,
  title={BDFusion: Visible and Infrared Information Synthesis for Severe Low-light Image Enhancement},
  author={},
  journal={},
  year={2025}
}
```

## Acknowledgments

This project builds upon [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), and the original VIIS framework.

---

### 5. Download MSRS Dataset

Download the MSRS dataset from the [PIAFusion repository](https://github.com/Linfeng-Tang/PIAFusion)  
and place it in the `./data/` directory.
