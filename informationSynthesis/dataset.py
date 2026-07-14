import random
import torch
import os
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
# 修正导入：原代码引用 ZC_image2image.Data_Augment，不存在该包；本仓库文件就在本目录。
from informationSynthesis.Data_Augment import (
    brightness_aug,
    poisson_noise,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
)
from informationSynthesis.task_embedding import (
    DegradationType,
    DEGRADATION_NAMES,
    DEGRADATION_NAME_TO_ID
)
from skimage import transform
import cv2
from typing import Optional, List, Tuple, Dict
from glob import glob


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

def get_transform(method=Image.BICUBIC):
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

class I2IDatasetBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 filter_suffix: str = 'D.png',  # 兼容原逻辑；设为 None 或空串表示使用全部图片
                 require_ir: bool = True       # 若对应红外不存在是否跳过
                 ):
        self.data_root = data_root
        self.size = size
        self.filter_suffix = filter_suffix if filter_suffix else None
        self.require_ir = require_ir

        # 收集所有候选图像
        all_files = [f for f in os.listdir(self.data_root)
                     if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
        if self.filter_suffix:
            all_files = [f for f in all_files if f.endswith(self.filter_suffix)]

        image_paths = []
        missing_ir = 0
        for f in sorted(all_files):
            rgb_path = os.path.join(self.data_root, f)
            ir_path = rgb_path.replace('rgbs', 'infrareds')
            if self.require_ir and not os.path.exists(ir_path):
                missing_ir += 1
                continue
            image_paths.append(rgb_path)

        if missing_ir > 0:
            print(f"[I2IDatasetTrain] 跳过 {missing_ir} 张无对应红外图的样本。保留 {len(image_paths)} 张。")
        self.image_paths = image_paths
        self._length = len(self.image_paths)

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.Norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_name = os.path.basename(self.image_paths[i])

        # --- Load and augment RGB input image ---
        image_rgb = Image.open(image_path)
        contrast_factor = random.uniform(0.1, 1.0)
        image_rgb = ImageEnhance.Contrast(image_rgb).enhance(contrast_factor)
        image_rgb = np.array(image_rgb).astype(np.uint8)

        crop_size = min(image_rgb.shape[:2])
        x1 = np.random.randint(0, image_rgb.shape[1] - crop_size + 1)
        image_rgb = image_rgb[:, x1:x1 + crop_size, :]

        # Apply custom augmentations
        image_rgb = brightness_aug(image_rgb, 3 + random.random() * 7)
        image_rgb = poisson_noise(image_rgb)
        image_rgb = add_gaussian_noise(image_rgb)

        # Resize and normalize to [-1, 1]
        image_rgb = cv2.resize(image_rgb, (self.size, self.size))
        image_rgb = (image_rgb / 127.5 - 1.0).astype(np.float32)

        # --- Load infrared image ---
        infrared_path = image_path.replace("rgbs", "infrareds")
        image_ir = Image.open(infrared_path).convert("L").convert("RGB")
        image_ir = np.array(image_ir).astype(np.uint8)
        image_ir = image_ir[:, x1:x1 + crop_size, :]
        image_ir = cv2.resize(image_ir, (self.size, self.size))
        image_ir = (image_ir / 127.5 - 1.0).astype(np.float32)

        # --- Load supervised ground truth (RGB) image ---
        image_gt = Image.open(image_path)
        image_gt = np.array(image_gt)[:, x1:x1 + crop_size, :].astype(np.uint8)
        image_gt = cv2.resize(image_gt, (self.size, self.size))
        image_gt = (image_gt / 127.5 - 1.0).astype(np.float32)

        return image_gt, image_rgb, image_ir, image_name


class DegradedFusionDatasetTrain(Dataset):
    """支持四个独立目录：退化RGB、退化IR、干净RGB GT、干净IR GT。

    使用场景：你拥有成对的 (rgb_degraded, ir_degraded, gt_clean_rgb, gt_clean_ir)。希望模型以退化模态作为条件，预测干净 RGB GT，同时输出灰度与干净 IR GT 对齐。

    设计原则：
    1. 不破坏现有 I2IDataset* 用法；新增独立类。
    2. 目录对齐基于同名文件（文件名完全一致）。
    3. 可选随机裁剪（保持四张图同步区域）+ resize。
    4. augment 仅应用在退化输入 (RGB/IR) 上；GT 只做裁剪+resize+归一化。
    5. IR 若为单通道会自动转 RGB（保持与模型 in_channels 兼容）。
    6. 支持可选再人为添加额外噪声/亮度扰动（如果原始退化已包含，这些可关闭）。
    7. 新增 ir_clean_root 支持 IR 监督损失。
    """
    def __init__(self,
                 rgb_root: str,
                 ir_root: str,
                 clean_root: str,
                 ir_clean_root: Optional[str] = None,
                 size: int = 256,
                 random_crop: bool = True,
                 center_crop_val: bool = True,
                 apply_rgb_aug: bool = True,
                 apply_ir_aug: bool = False,
                 extra_noise: bool = False,
                 match_suffix: Optional[str] = None,
                 verbose: bool = True):
        super().__init__()
        self.rgb_root = rgb_root
        self.ir_root = ir_root
        self.clean_root = clean_root
        self.ir_clean_root = ir_clean_root
        self.use_ir_supervision = ir_clean_root is not None
        self.size = size
        self.random_crop = random_crop
        self.center_crop_val = center_crop_val
        self.apply_rgb_aug = apply_rgb_aug
        self.apply_ir_aug = apply_ir_aug
        self.extra_noise = extra_noise
        self.match_suffix = match_suffix  # 若仅匹配特定后缀（例如 .png）

        def _list(p):
            return [f for f in os.listdir(p)
                    if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]

        rgb_files = set(_list(rgb_root))
        ir_files = set(_list(ir_root))
        gt_files = set(_list(clean_root))
        if self.use_ir_supervision:
            ir_clean_files = set(_list(ir_clean_root))
        else:
            ir_clean_files = set()
            
        if match_suffix:
            rgb_files = {f for f in rgb_files if f.endswith(match_suffix)}
            ir_files = {f for f in ir_files if f.endswith(match_suffix)}
            gt_files = {f for f in gt_files if f.endswith(match_suffix)}
            if self.use_ir_supervision:
                ir_clean_files = {f for f in ir_clean_files if f.endswith(match_suffix)}

        if self.use_ir_supervision:
            common = sorted(list(rgb_files & ir_files & gt_files & ir_clean_files))
            missing_ir_clean = (rgb_files & ir_files & gt_files) - ir_clean_files
        else:
            common = sorted(list(rgb_files & ir_files & gt_files))
            missing_ir_clean = set()
            
        missing_rgb = gt_files - rgb_files
        missing_ir = gt_files - ir_files
        missing_gt = (rgb_files & ir_files) - gt_files
        
        if verbose:
            base_msg = f"[DegradedFusionDatasetTrain] 样本数: {len(common)} | 缺少RGB:{len(missing_rgb)} 缺少IR:{len(missing_ir)} 缺少GT:{len(missing_gt)}"
            if self.use_ir_supervision:
                base_msg += f" 缺少IR_GT:{len(missing_ir_clean)}"
            print(base_msg)
        self.files = common

    def __len__(self):
        return len(self.files)

    def _open_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        return np.array(img).astype(np.uint8)

    def _open_ir(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode != 'RGB':
            # 转成 L 再复制到3通道
            img = img.convert('L').convert('RGB')
        return np.array(img).astype(np.uint8)

    def _sync_crop(self, arrs: List[np.ndarray]) -> List[np.ndarray]:
        # arr: H,W,C
        h, w = arrs[0].shape[:2]
        side = min(h, w)
        if self.random_crop:
            x0 = 0 if w == side else random.randint(0, w - side)
            y0 = 0 if h == side else random.randint(0, h - side)
        else:
            x0 = (w - side) // 2
            y0 = (h - side) // 2
        cropped = [a[y0:y0+side, x0:x0+side, :] for a in arrs]
        return cropped

    def _open_ir_clean(self, path: str) -> np.ndarray:
        """加载干净 IR GT，保持单通道再转 RGB（或直接 RGB）"""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('L').convert('RGB')
        return np.array(img).astype(np.uint8)

    def _resize_and_norm(self, arr: np.ndarray) -> np.ndarray:
        arr = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return (arr / 127.5 - 1.0).astype(np.float32)

    def _maybe_aug_rgb(self, arr: np.ndarray) -> np.ndarray:
        if not self.apply_rgb_aug:
            return arr
        # 对比度
        pil = Image.fromarray(arr)
        factor = random.uniform(0.5, 1.5)
        pil = ImageEnhance.Contrast(pil).enhance(factor)
        arr = np.array(pil)
        # 亮度/噪声（可选）
        arr = brightness_aug(arr, 2 + random.random() * 6)
        if self.extra_noise:
            arr = poisson_noise(arr)
            arr = add_gaussian_noise(arr)
        return arr

    def _maybe_aug_ir(self, arr: np.ndarray) -> np.ndarray:
        if not self.apply_ir_aug:
            return arr
        # IR 通常不做强对比增强，保守处理
        if self.extra_noise:
            arr = add_gaussian_noise(arr)
        return arr

    def __getitem__(self, idx: int):
        name = self.files[idx]
        rgb_path = os.path.join(self.rgb_root, name)
        ir_path = os.path.join(self.ir_root, name)
        gt_path = os.path.join(self.clean_root, name)

        rgb = self._open_rgb(rgb_path)
        ir = self._open_ir(ir_path)
        gt = self._open_rgb(gt_path)

        if self.use_ir_supervision:
            ir_clean_path = os.path.join(self.ir_clean_root, name)
            ir_clean = self._open_ir_clean(ir_clean_path)
            # 同步随机/中心裁剪（四张图）
            rgb, ir, gt, ir_clean = self._sync_crop([rgb, ir, gt, ir_clean])
        else:
            ir_clean = None
            # 同步随机/中心裁剪（三张图）
            rgb, ir, gt = self._sync_crop([rgb, ir, gt])

        # 退化增强（仅输入）
        rgb = self._maybe_aug_rgb(rgb)
        ir = self._maybe_aug_ir(ir)
        # gt 和 ir_clean 不做增强

        # 尺寸与归一化
        rgb = self._resize_and_norm(rgb)
        ir = self._resize_and_norm(ir)
        gt = self._resize_and_norm(gt)
        if ir_clean is not None:
            ir_clean = self._resize_and_norm(ir_clean)

        if self.use_ir_supervision:
            return gt, rgb, ir, ir_clean, name
        else:
            return gt, rgb, ir, name


class DegradedFusionDatasetVal(DegradedFusionDatasetTrain):
    """验证版：默认使用中心裁剪，关闭随机增强。"""
    def __init__(self, **kwargs):
        kwargs.setdefault('random_crop', False)
        kwargs.setdefault('apply_rgb_aug', False)
        kwargs.setdefault('apply_ir_aug', False)
        kwargs.setdefault('extra_noise', False)
        super().__init__(**kwargs)

class I2IDatasetVal(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.image_paths = [data_root + '/' + os.path.basename(f) for f in os.listdir(data_root)]
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.Norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def _load_and_preprocess(self, path):
        """Loads, center-crops, resizes, and normalizes an image."""
        image = Image.open(path).convert("RGB")
        image = np.array(image).astype(np.uint8)

        h, w = image.shape[:2]
        crop_squ = min(h, w)
        image = image[(h - crop_squ) // 2:(h + crop_squ) // 2,
                      (w - crop_squ) // 2:(w + crop_squ) // 2]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_name = os.path.basename(self.image_paths[i])

        rgb_image = self._load_and_preprocess(image_path)
        infrared_path = image_path.replace('rgbs', 'infrareds')
        infrared_image= self._load_and_preprocess(infrared_path)
        
        return rgb_image, rgb_image, infrared_image, image_name

class I2IDatasetTrain(I2IDatasetBase):
    def __init__(self, **kwargs):
        # 允许在 YAML 中设置 filter_suffix: null 来使用全部图片
        super().__init__(**kwargs)

class I2IDatasetValidation(I2IDatasetVal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ============================================================================
# All-in-One Dataset with Task Conditioning
# ============================================================================

class AllInOneDegradedDataset(Dataset):
    """
    BLIND All-in-One Dataset for Multiple Degradation Types.
    
    This dataset supports training a unified model on multiple degradation types
    (rain, fog, low-light, noise, blur, etc.) WITHOUT explicit task type labels.
    
    Key Design: BLIND Degradation Encoding (AirNet-style)
    - No task_id is returned to the model
    - DegradationEncoder extracts task information from image content
    - Model learns to identify degradation patterns automatically
    
    Directory Structure Expected:
    root/
    ├── rain/
    │   ├── degraded/
    │   │   ├── rgb/
    │   │   └── ir/
    │   └── clean/
    │       ├── rgb/
    │       └── ir/
    ├── fog/
    │   ├── degraded/
    │   │   ├── rgb/
    │   │   └── ir/
    │   └── clean/
    │       ├── rgb/
    │       └── ir/
    └── ...
    
    Or Flat Structure:
    root/
    ├── rgb_degraded/
    │   ├── rain_001.png
    │   ├── fog_002.png
    │   └── ...
    ├── ir_degraded/
    ├── rgb_clean/
    └── ir_clean/
    
    Returns:
        (rgb_clean, rgb_degraded, ir_degraded, name) - 4 elements
        (rgb_clean, rgb_degraded, ir_degraded, ir_clean, name) - 5 elements with IR GT
        
    Note: task_id is NOT returned! DegradationEncoder handles task detection.
    
    Args:
        root_dir: Root directory containing degradation folders
        degradation_folders: Dict mapping degradation names to folder paths
        size: Output image size
        use_flat_structure: If True, use flat directory structure
        verbose: Print dataset statistics
    """
    
    def __init__(
        self,
        root_dir: str,
        degradation_folders: Optional[Dict[str, str]] = None,
        size: int = 256,
        random_crop: bool = True,
        apply_aug: bool = True,
        use_flat_structure: bool = False,
        task_prefix_in_filename: bool = True,
        use_ir_clean: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.size = size
        self.random_crop = random_crop
        self.apply_aug = apply_aug
        self.use_ir_clean = use_ir_clean
        
        self.samples: List[Dict] = []
        
        if use_flat_structure:
            self._load_flat_structure(root_dir, task_prefix_in_filename)
        elif degradation_folders is not None:
            self._load_from_degradation_folders(degradation_folders)
        else:
            self._load_hierarchical_structure(root_dir)
        
        if verbose:
            self._print_statistics()
    
    def _load_flat_structure(self, root_dir: str, task_prefix: bool):
        """Load from flat directory structure with task prefix in filenames."""
        rgb_deg_dir = os.path.join(root_dir, 'rgb_degraded')
        ir_deg_dir = os.path.join(root_dir, 'ir_degraded')
        rgb_clean_dir = os.path.join(root_dir, 'rgb_clean')
        ir_clean_dir = os.path.join(root_dir, 'ir_clean') if self.use_ir_clean else None
        
        if not os.path.exists(rgb_deg_dir):
            raise ValueError(f"RGB degraded directory not found: {rgb_deg_dir}")
        
        for fname in os.listdir(rgb_deg_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Extract task type from filename prefix (e.g., "rain_001.png" -> "rain")
            if task_prefix:
                parts = fname.split('_')
                task_name = parts[0].lower() if len(parts) > 1 else 'mixed'
            else:
                task_name = 'mixed'
            
            task_id = DEGRADATION_NAME_TO_ID.get(task_name, DegradationType.MIXED)
            
            sample = {
                'rgb_degraded': os.path.join(rgb_deg_dir, fname),
                'ir_degraded': os.path.join(ir_deg_dir, fname),
                'rgb_clean': os.path.join(rgb_clean_dir, fname),
                'ir_clean': os.path.join(ir_clean_dir, fname) if ir_clean_dir else None,
                'task_id': task_id,
                'task_name': task_name,
                'name': fname
            }
            
            # Verify files exist
            if os.path.exists(sample['rgb_degraded']) and os.path.exists(sample['rgb_clean']):
                self.samples.append(sample)
    
    def _load_hierarchical_structure(self, root_dir: str):
        """Load from hierarchical directory structure."""
        for task_name in os.listdir(root_dir):
            task_dir = os.path.join(root_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            
            task_id = DEGRADATION_NAME_TO_ID.get(task_name.lower(), DegradationType.MIXED)
            
            rgb_deg_dir = os.path.join(task_dir, 'degraded', 'rgb')
            ir_deg_dir = os.path.join(task_dir, 'degraded', 'ir')
            rgb_clean_dir = os.path.join(task_dir, 'clean', 'rgb')
            ir_clean_dir = os.path.join(task_dir, 'clean', 'ir') if self.use_ir_clean else None
            
            if not os.path.exists(rgb_deg_dir):
                continue
            
            for fname in os.listdir(rgb_deg_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                sample = {
                    'rgb_degraded': os.path.join(rgb_deg_dir, fname),
                    'ir_degraded': os.path.join(ir_deg_dir, fname),
                    'rgb_clean': os.path.join(rgb_clean_dir, fname),
                    'ir_clean': os.path.join(ir_clean_dir, fname) if ir_clean_dir else None,
                    'task_id': task_id,
                    'task_name': task_name,
                    'name': fname
                }
                
                if os.path.exists(sample['rgb_degraded']) and os.path.exists(sample['rgb_clean']):
                    self.samples.append(sample)
    
    def _load_from_degradation_folders(self, folders: Dict[str, str]):
        """Load from explicitly specified degradation folders."""
        for task_name, folder_path in folders.items():
            task_id = DEGRADATION_NAME_TO_ID.get(task_name.lower(), DegradationType.MIXED)
            
            rgb_deg_dir = os.path.join(folder_path, 'rgb_degraded')
            ir_deg_dir = os.path.join(folder_path, 'ir_degraded')
            rgb_clean_dir = os.path.join(folder_path, 'rgb_clean')
            ir_clean_dir = os.path.join(folder_path, 'ir_clean') if self.use_ir_clean else None
            
            if not os.path.exists(rgb_deg_dir):
                print(f"Warning: RGB degraded dir not found for {task_name}: {rgb_deg_dir}")
                continue
            
            for fname in os.listdir(rgb_deg_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                sample = {
                    'rgb_degraded': os.path.join(rgb_deg_dir, fname),
                    'ir_degraded': os.path.join(ir_deg_dir, fname),
                    'rgb_clean': os.path.join(rgb_clean_dir, fname),
                    'ir_clean': os.path.join(ir_clean_dir, fname) if ir_clean_dir else None,
                    'task_id': task_id,
                    'task_name': task_name,
                    'name': fname
                }
                
                if os.path.exists(sample['rgb_degraded']) and os.path.exists(sample['rgb_clean']):
                    self.samples.append(sample)
    
    def _print_statistics(self):
        """Print dataset statistics."""
        task_counts = {}
        for sample in self.samples:
            task_name = sample['task_name']
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
        
        print(f"\n[AllInOneDegradedDataset] Total samples: {len(self.samples)}")
        print("Per-task distribution:")
        for task_name, count in sorted(task_counts.items()):
            print(f"  - {task_name}: {count}")
        print()
    
    def __len__(self):
        return len(self.samples)
    
    def _open_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        return np.array(img).astype(np.uint8)
    
    def _open_ir(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('L').convert('RGB')
        return np.array(img).astype(np.uint8)
    
    def _sync_crop(self, arrs: List[np.ndarray]) -> List[np.ndarray]:
        h, w = arrs[0].shape[:2]
        side = min(h, w)
        if self.random_crop:
            x0 = 0 if w == side else random.randint(0, w - side)
            y0 = 0 if h == side else random.randint(0, h - side)
        else:
            x0 = (w - side) // 2
            y0 = (h - side) // 2
        cropped = [a[y0:y0+side, x0:x0+side, :] for a in arrs]
        return cropped
    
    def _resize_and_norm(self, arr: np.ndarray) -> np.ndarray:
        arr = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return (arr / 127.5 - 1.0).astype(np.float32)
    
    def _maybe_aug(self, arr: np.ndarray, is_ir: bool = False) -> np.ndarray:
        if not self.apply_aug:
            return arr
        
        if not is_ir:
            # RGB augmentation
            pil = Image.fromarray(arr)
            factor = random.uniform(0.5, 1.5)
            pil = ImageEnhance.Contrast(pil).enhance(factor)
            arr = np.array(pil)
            arr = brightness_aug(arr, 2 + random.random() * 6)
        
        return arr
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        # Load images
        rgb_deg = self._open_rgb(sample['rgb_degraded'])
        rgb_clean = self._open_rgb(sample['rgb_clean'])
        
        # Load IR if exists
        if os.path.exists(sample['ir_degraded']):
            ir_deg = self._open_ir(sample['ir_degraded'])
        else:
            ir_deg = np.zeros_like(rgb_deg)
        
        # Load IR clean if needed
        if self.use_ir_clean and sample['ir_clean'] and os.path.exists(sample['ir_clean']):
            ir_clean = self._open_ir(sample['ir_clean'])
            images = [rgb_deg, ir_deg, rgb_clean, ir_clean]
        else:
            ir_clean = None
            images = [rgb_deg, ir_deg, rgb_clean]
        
        # Synchronized crop
        images = self._sync_crop(images)
        
        if ir_clean is not None:
            rgb_deg, ir_deg, rgb_clean, ir_clean = images
        else:
            rgb_deg, ir_deg, rgb_clean = images
        
        # Augmentation (only on degraded inputs)
        rgb_deg = self._maybe_aug(rgb_deg, is_ir=False)
        ir_deg = self._maybe_aug(ir_deg, is_ir=True)
        
        # Resize and normalize
        rgb_deg = self._resize_and_norm(rgb_deg)
        ir_deg = self._resize_and_norm(ir_deg)
        rgb_clean = self._resize_and_norm(rgb_clean)
        
        # Note: task_id is NOT returned for blind degradation encoding!
        # The DegradationEncoder will extract task information from image content.
        
        if ir_clean is not None:
            ir_clean = self._resize_and_norm(ir_clean)
            return rgb_clean, rgb_deg, ir_deg, ir_clean, sample['name']
        else:
            return rgb_clean, rgb_deg, ir_deg, sample['name']


class AllInOneDegradedDatasetVal(AllInOneDegradedDataset):
    """Validation version with center crop and no augmentation."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('random_crop', False)
        kwargs.setdefault('apply_aug', False)
        super().__init__(**kwargs)
