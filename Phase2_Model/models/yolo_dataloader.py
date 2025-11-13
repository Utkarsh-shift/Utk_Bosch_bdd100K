"""
======================================================================
Advanced YOLO DataLoader for BDD100K with Data-Driven Normalization and Augmentation
======================================================================

Author: Utkarsh
Date: 2025-11-10

This script provides an advanced, configurable PyTorch DataLoader
for BDD100K datasets in YOLO format, integrating:

1. Multiple normalization methods (Min-Max, Z-score, Per-Channel)
2. Class-specific augmentation control
3. Data-informed augmentation (guided by brightness, contrast, saturation stats)
4. Parallel data loading, shuffling, and batching
5. Detailed console-level feedback and reporting

======================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import List, Dict, Optional


# ================================================================
# 1. Class Mappings
# ================================================================
CLASS_NAMES = [
    'bike', 'bus', 'car', 'motor', 'person',
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


# ================================================================
# 2. Normalization Methods
# ================================================================
def normalize_minmax(img_tensor):
    """Min-Max normalization to [0, 1]."""
    return (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)


def normalize_zscore(img_tensor):
    """Z-score normalization (global mean/std)."""
    mean = img_tensor.mean()
    std = img_tensor.std()
    return (img_tensor - mean) / (std + 1e-8)


def normalize_per_channel(img_tensor):
    """Per-channel normalization."""
    for c in range(img_tensor.size(0)):
        mean = img_tensor[c].mean()
        std = img_tensor[c].std()
        img_tensor[c] = (img_tensor[c] - mean) / (std + 1e-8)
    return img_tensor

def normalize_robust(img_tensor):
    """Robust normalization using median and IQR."""
    median = img_tensor.median()
    q1 = torch.quantile(img_tensor, 0.25)
    q3 = torch.quantile(img_tensor, 0.75)
    iqr = q3 - q1 + 1e-8
    return (img_tensor - median) / iqr


def normalize_dynamic_range(img_tensor, low_percentile=2, high_percentile=98):
    """Contrast stretch based on percentiles."""
    low = torch.quantile(img_tensor, low_percentile / 100)
    high = torch.quantile(img_tensor, high_percentile / 100)
    img_tensor = torch.clamp((img_tensor - low) / (high - low + 1e-8), 0, 1)
    return img_tensor


def normalize_log(img_tensor):
    """Logarithmic normalization for dynamic range compression."""
    return torch.log1p(img_tensor) / torch.log1p(img_tensor.max())

def normalize_gamma(img_tensor, gamma=0.8):
    """Gamma correction-based normalization."""
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor ** gamma

import torch.nn.functional as F

def normalize_local_contrast(img_tensor, kernel_size=15):
    """Local contrast normalization using mean filter."""
    mean = F.avg_pool2d(img_tensor, kernel_size, stride=1, padding=kernel_size // 2)
    diff = img_tensor - mean
    std = torch.sqrt(F.avg_pool2d(diff ** 2, kernel_size, stride=1, padding=kernel_size // 2) + 1e-8)
    return diff / (std + 1e-8)


def normalize_hybrid(img_tensor):
    img_tensor = normalize_dynamic_range(img_tensor)
    img_tensor = normalize_zscore(img_tensor)
    return img_tensor


def normalize_adaptive(img_tensor, class_name, class_stats):
    """Adaptive normalization per class based on brightness/contrast statistics."""
    stats = class_stats.get(class_name, {})
    brightness_mean = stats.get("exposure_mean", 75.0)
    contrast_mean = stats.get("contrast_mean", 50.0)

    # Scale intensity based on dataset stats
    img_tensor = (img_tensor / 255.0) * (brightness_mean / 100.0)
    img_tensor = (img_tensor - 0.5) * (contrast_mean / 50.0) + 0.5
    return torch.clamp(img_tensor, 0, 1)


NORMALIZATION_METHODS = {
    "minmax": normalize_minmax,
    "zscore": normalize_zscore,
    "per_channel": normalize_per_channel,
    "robust": normalize_robust,
    "dynamic_range": normalize_dynamic_range,
    "log": normalize_log,
    "gamma": normalize_gamma,
    "local_contrast": normalize_local_contrast,
    "hybrid": normalize_hybrid
}


# ================================================================
# 3. Augmentation Utilities
# ================================================================
# ================================================================
# 3. Augmentation Utilities (Comprehensive)
# ================================================================
from PIL import ImageEnhance, ImageFilter
import torchvision.transforms.functional as F
import numpy as np
import torch
import random


# ---------------------------------------------------------------
# 1. Brightness / Contrast / Color Adjustments
# ---------------------------------------------------------------
def adjust_brightness_contrast(img, brightness_factor=1.0, contrast_factor=1.0):
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def adjust_color_balance(img, saturation_factor=1.0, hue_factor=0.0):
    img = ImageEnhance.Color(img).enhance(saturation_factor)
    if abs(hue_factor) > 1e-6:
        img = F.adjust_hue(img, hue_factor)
    return img


def apply_gamma_correction(img, gamma=0.8):
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = np.power(img_arr, gamma)
    img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


# ---------------------------------------------------------------
# 2. Geometric Transformations
# ---------------------------------------------------------------
def random_horizontal_flip(img, p=0.5):
    return F.hflip(img) if random.random() < p else img


def random_rotation(img, degrees=(-10, 10)):
    return F.rotate(img, angle=random.uniform(*degrees))


def random_crop(img, scale=(0.8, 1.0)):
    w, h = img.size
    scale_factor = random.uniform(*scale)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return F.crop(img, top, left, new_h, new_w).resize((w, h))


def random_perspective(img, distortion_scale=0.3, p=0.5):
    if random.random() < p:
        return F.perspective(img, startpoints=None, endpoints=None, distortion_scale=distortion_scale)
    return img


def random_affine(img, degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.9, 1.1)):
    angle = random.uniform(*degrees)
    translate_vals = (random.uniform(-translate[0], translate[0]) * img.size[0],
                      random.uniform(-translate[1], translate[1]) * img.size[1])
    scale_factor = random.uniform(*scale)
    return F.affine(img, angle=angle, translate=translate_vals, scale=scale_factor, shear=0)


# ---------------------------------------------------------------
# 3. Noise and Blur
# ---------------------------------------------------------------
def add_gaussian_noise(img_tensor, std=0.02):
    noise = torch.randn_like(img_tensor) * std
    return torch.clamp(img_tensor + noise, 0, 1)


def apply_motion_blur(img, radius=3):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, radius)))


def apply_jpeg_artifacts(img, quality_range=(50, 95)):
    from io import BytesIO
    buffer = BytesIO()
    quality = random.randint(*quality_range)
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


# ---------------------------------------------------------------
# 4. Spatial Cutout / Erasing / Mixing
# ---------------------------------------------------------------
def random_erasing(img_tensor, p=0.5, scale=(0.02, 0.2)):
    if random.random() < p:
        c, h, w = img_tensor.shape
        erase_w = int(random.uniform(scale[0], scale[1]) * w)
        erase_h = int(random.uniform(scale[0], scale[1]) * h)
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        img_tensor[:, y:y + erase_h, x:x + erase_w] = 0.0
    return img_tensor


def random_cutout(img_tensor, holes=1, size=0.2):
    c, h, w = img_tensor.shape
    for _ in range(holes):
        mask_w = int(w * size * random.uniform(0.5, 1.5))
        mask_h = int(h * size * random.uniform(0.5, 1.5))
        x = random.randint(0, w - mask_w)
        y = random.randint(0, h - mask_h)
        img_tensor[:, y:y + mask_h, x:x + mask_w] = torch.rand((c, mask_h, mask_w)) * 0.2
    return img_tensor


def random_mixup(img_tensor1, img_tensor2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return lam * img_tensor1 + (1 - lam) * img_tensor2


# ---------------------------------------------------------------
# 5. Domain-Specific Augmentations (for BDD100K)
# ---------------------------------------------------------------
def simulate_fog(img, fog_intensity=0.4):
    arr = np.array(img).astype(np.float32) / 255.0
    fog = np.ones_like(arr) * random.uniform(0.6, 1.0)
    arr = arr * (1 - fog_intensity) + fog * fog_intensity
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def simulate_rain(img, drop_count=200, drop_size=(1, 3)):
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape
    for _ in range(drop_count):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        thickness = random.randint(*drop_size)
        arr[max(0, y - thickness):min(h, y + thickness), x:x + thickness] = 255
    return Image.fromarray(arr.astype(np.uint8))


def simulate_shadow(img, intensity=0.5):
    arr = np.array(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    x1, x2 = random.randint(0, w // 2), random.randint(w // 2, w)
    y1, y2 = random.randint(0, h // 2), random.randint(h // 2, h)
    arr[y1:y2, x1:x2] *= random.uniform(1 - intensity, 1)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

AUGMENTATION_METHODS = {
    "brightness_contrast": adjust_brightness_contrast,
    "color_balance": adjust_color_balance,
    "gamma": apply_gamma_correction,
    "horizontal_flip": random_horizontal_flip,
    "rotation": random_rotation,
    "crop": random_crop,
    "perspective": random_perspective,
    "affine": random_affine,
    "gaussian_noise": add_gaussian_noise,
    "blur": apply_motion_blur,
    "jpeg_artifacts": apply_jpeg_artifacts,
    "erasing": random_erasing,
    "cutout": random_cutout,
    "mixup": random_mixup,
    "fog": simulate_fog,
    "rain": simulate_rain,
    "shadow": simulate_shadow
}

# ================================================================
# 4. YOLO Dataset Class
# ================================================================
class BDDYOLODataset(Dataset):
    """
    YOLO-format Dataset for BDD100K with configurable augmentation and normalization.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 640,
        normalization: str = "minmax",
        augment: bool = True,
        augment_classes: Optional[List[str]] = None,
        class_quality_stats: Optional[Dict] = None
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.img_size = img_size
        self.normalization = normalization
        self.augment = augment
        self.augment_classes = augment_classes or []
        self.class_quality_stats = class_quality_stats or {}

        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        print("Initializing BDDYOLODataset")
        print(f"  Image Directory   : {self.images_dir}")
        print(f"  Label Directory   : {self.labels_dir}")
        print(f"  Total Images      : {len(self.img_files)}")
        print(f"  Image Size        : {self.img_size}")
        print(f"  Normalization     : {self.normalization}")
        print(f"  Augment Enabled   : {self.augment}")
        print(f"  Augment Classes   : {self.augment_classes}")
        print("-" * 70)

    def __len__(self):
        return len(self.img_files)

    def _should_augment(self, labels):
        for label in labels:
            cls = int(label[0])
            if CLASS_NAMES[cls] in self.augment_classes:
                return True
        return False

    def _apply_augmentations(self, img, labels):
        """
        Apply a randomized, multi-stage augmentation pipeline.

        Features:
        - Random subset of augmentations per sample
        - Per-augmentation probability
        - Per-class augmentation configuration (optional)
        - Intensity randomness (for photometric aug)
        """

        # Example configuration
        AUGMENTATION_PIPELINE = {
            "brightness_contrast": 0.7,   # 70% chance
            "color_balance": 0.4,         # 40% chance
            "rotation": 0.5,
            "horizontal_flip": 0.8,
            "fog": 0.2,
            "shadow": 0.2,
            "gamma": 0.5
        }

        # Optional: Per-class augmentation customization
        PER_CLASS_AUGS = {
            "train": ["fog", "shadow"],
            "bus": ["fog"],
            "car": ["rotation", "brightness_contrast"],
            "person": ["color_balance", "rotation"],
        }

        # Determine if we should apply class-specific augmentations
        active_classes = [CLASS_NAMES[int(lbl[0])] for lbl in labels] if len(labels) else []
        selected_augs = set()

        for cls_name in active_classes:
            if cls_name in PER_CLASS_AUGS:
                selected_augs.update(PER_CLASS_AUGS[cls_name])

        # Combine selected per-class augmentations with global pipeline
        if not selected_augs:
            selected_augs = set(AUGMENTATION_PIPELINE.keys())

        # Randomly apply augmentations with probabilities
        for aug_name, prob in AUGMENTATION_PIPELINE.items():
            if aug_name in selected_augs and random.random() < prob:
                aug_func = AUGMENTATION_METHODS.get(aug_name)
                if aug_func:
                    try:
                        img = aug_func(img)
                        # print(f"[AUG] Applied: {aug_name}")
                    except Exception as e:
                        print(f"[WARN] Augmentation '{aug_name}' failed: {e}")

        return img


    def __getitem__(self, idx):
        random.seed(idx)

        img_path = self.img_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))

        # Load labels
        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    labels.append([cls, x, y, w, h])
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)


        # Apply augmentations
        if self.augment:
            img = self._apply_augmentations(img, labels)

        # Convert to tensor
        img_tensor = F.to_tensor(img)

        # Optional advanced augmentations on tensor
        if self.augment:
            img_tensor = add_gaussian_noise(img_tensor, std=0.01)
            img_tensor = random_erasing(img_tensor, p=0.4, scale=(0.05, 0.15))

        # Apply chosen normalization
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        if self.normalization in NORMALIZATION_METHODS:
            img_tensor = NORMALIZATION_METHODS[self.normalization](img_tensor)

        return img_tensor, labels


# ================================================================
# 5. Collate Function
# ================================================================
def yolo_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets


# ================================================================
# 6. DataLoader Builder
# ================================================================
def build_dataloader(
    dataset_root: str,
    split: str = "train",
    batch_size: int = 8,
    img_size: int = 640,
    normalization: str = "minmax",
    augment: bool = True,
    augment_classes: Optional[List[str]] = None,
    class_quality_stats: Optional[Dict] = None,
    num_workers: int = 4
):
    images_dir = Path(dataset_root) / "images" / split
    labels_dir = Path(dataset_root) / "labels" / split

    dataset = BDDYOLODataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=img_size,
        normalization=normalization,
        augment=augment,
        augment_classes=augment_classes,
        class_quality_stats=class_quality_stats
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=yolo_collate_fn
    )

    print("YOLO DataLoader successfully initialized.")
    print(f"  Split          : {split}")
    print(f"  Batch Size     : {batch_size}")
    print(f"  Workers        : {num_workers}")
    print(f"  Normalization  : {normalization}")
    print(f"  Augment        : {augment}")
    print(f"  Augment Classes: {augment_classes}")
    print("-" * 70)
    return dataloader


# ================================================================
# 7. Self-Test & Visualization Generator

if __name__ == "__main__":
    import os
    from pprint import pprint
    from torchvision.utils import save_image
    from itertools import product

    # ================================================================
    # Class-Level Quality Statistics (Extracted from Dataset Summary)
    # ================================================================
    CLASS_STATS_SAMPLE = {
        "bike": {"exposure_mean": 85.42, "contrast_mean": 55.66},
        "bus": {"exposure_mean": 86.79, "contrast_mean": 56.18},
        "car": {"exposure_mean": 73.04, "contrast_mean": 49.60},
        "motor": {"exposure_mean": 85.80, "contrast_mean": 54.99},
        "person": {"exposure_mean": 82.99, "contrast_mean": 54.91},
        "rider": {"exposure_mean": 88.67, "contrast_mean": 57.43},
        "traffic light": {"exposure_mean": 65.78, "contrast_mean": 47.66},
        "traffic sign": {"exposure_mean": 73.98, "contrast_mean": 49.34},
        "train": {"exposure_mean": 87.78, "contrast_mean": 56.53},
        "truck": {"exposure_mean": 87.10, "contrast_mean": 55.19},
        "lane": {"exposure_mean": 72.96, "contrast_mean": 49.38},
        "drivable area": {"exposure_mean": 72.90, "contrast_mean": 49.01}
    }

    # ================================================================
    # Paths
    # ================================================================
    dataset_root = "./bdd100k_yolo_subset"
    output_dir = Path("./bdd100k_yolo_subset_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Choose augmentations and normalizations to visualize
    # ================================================================
    normalization_methods = list(NORMALIZATION_METHODS.keys())
    augmentation_methods = list(AUGMENTATION_METHODS.keys())

    print(f"\n Generating visual samples for:")
    print(f"  Classes        : {list(CLASS_STATS_SAMPLE.keys())}")
    print(f"  Normalizations : {normalization_methods}")
    print(f"  Augmentations  : {augmentation_methods}")
    print("=" * 80)

    # ================================================================
    # Build base dataloader
    # ================================================================
    dataloader = build_dataloader(
        dataset_root=dataset_root,
        split="train",
        batch_size=4,  # <<<<<< Changed batch size to 4
        img_size=640,
        normalization="minmax",
        augment=False,
        augment_classes=list(CLASS_STATS_SAMPLE.keys()),
        class_quality_stats=CLASS_STATS_SAMPLE,
        num_workers=0
    )

    # Fetch one sample per class (representative)
    samples = []
    for img_tensors, labels_batch in dataloader:
        for i in range(len(img_tensors)):
            samples.append((img_tensors[i], labels_batch[i]))
            if len(samples) >= len(CLASS_STATS_SAMPLE):
                break
        if len(samples) >= len(CLASS_STATS_SAMPLE):
            break

 # ================================================================
# Generate and Save Visualizations
# ================================================================

    import torch.nn.functional as F_torch  # for avg_pool2d
    import torchvision.transforms.functional as TF 
    for class_name in CLASS_STATS_SAMPLE.keys():
        class_dir = output_dir / f"class_{class_name.replace(' ', '_')}"
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n Generating samples for class: {class_name}")

        # Pick one random image as base
        base_img_tensor, labels = random.choice(samples)
        base_img = F.to_pil_image(base_img_tensor)

        # Iterate over combinations of normalization × augmentation
        for norm_name, aug_name in product(normalization_methods, augmentation_methods):
            img_aug = base_img.copy()
            aug_func = AUGMENTATION_METHODS.get(aug_name)

            try:
                # --- Handle PIL-based augmentations ---
                if aug_name in [
                    "brightness_contrast", "color_balance", "gamma", "horizontal_flip",
                    "rotation", "crop", "affine", "blur", "jpeg_artifacts",
                    "fog", "rain", "shadow"
                ]:
                    img_aug = aug_func(img_aug)

                # --- Handle tensor-based augmentations ---
                elif aug_name in ["gaussian_noise", "erasing", "cutout", "mixup"]:
                    try:
                        # --- Convert safely ---
                        if not isinstance(img_aug, Image.Image):
                            img_aug = TF.to_pil_image(img_aug)
                        img_tensor_tmp = TF.to_tensor(img_aug)

                        if aug_name == "mixup":
                            # Get a random second tensor image
                            img2, _ = random.choice(samples)
                            if not isinstance(img2, torch.Tensor):
                                img2 = TF.to_tensor(img2)

                            # Apply MixUp
                            mixed_tensor = aug_func(img_tensor_tmp.clone(), img2.clone())
                            mixed_tensor = torch.clamp(mixed_tensor, 0, 1)

                            # Convert back to PIL for next steps
                            img_aug = TF.to_pil_image(mixed_tensor)

                            # Optional visualization panel
                            try:
                                from torchvision.utils import save_image
                                panel = torch.cat([img_tensor_tmp, mixed_tensor, img2], dim=2)
                                panel_path = class_dir / f"mixup_panel_{class_name.replace(' ', '_')}.jpg"
                                save_image(panel, panel_path)
                                print(f"[MIXUP] Saved visual panel for {class_name} → {panel_path}")
                            except Exception as viz_err:
                                print(f"[WARN] MixUp visualization failed for {class_name}: {viz_err}")

                        else:
                            # Tensor-based augmentations
                            aug_tensor = aug_func(img_tensor_tmp.clone())
                            img_aug = TF.to_pil_image(torch.clamp(aug_tensor, 0, 1))

                    except Exception as e:
                        print(f"[WARN] Augmentation '{aug_name}' failed for {class_name}: {e}")
                        continue

                # --- Handle perspective safely ---
                elif aug_name == "perspective":
                    img_aug = F.perspective(
                        img_aug,
                        startpoints=[(0, 0), (640, 0), (640, 640), (0, 640)],
                        endpoints=[
                            (random.randint(0, 10), random.randint(0, 10)),
                            (630 + random.randint(0, 10), random.randint(0, 10)),
                            (630 + random.randint(0, 10), 630 + random.randint(0, 10)),
                            (random.randint(0, 10), 630 + random.randint(0, 10))
                        ]
                    )

                else:
                    continue

            except Exception as e:
                print(f"[WARN] Augmentation '{aug_name}' failed for {class_name}: {e}")
                continue

            # --- Normalization phase ---
            try:
                img_tensor = F.to_tensor(img_aug)
                norm_func = NORMALIZATION_METHODS.get(norm_name)

                if norm_name == "local_contrast":
                    # Fix: use torch.nn.functional.avg_pool2d
                    mean = F_torch.avg_pool2d(img_tensor, 15, stride=1, padding=7)
                    diff = img_tensor - mean
                    std = torch.sqrt(
                        F_torch.avg_pool2d(diff ** 2, 15, stride=1, padding=7) + 1e-8
                    )
                    img_tensor = diff / (std + 1e-8)
                elif norm_func:
                    img_tensor = norm_func(img_tensor)

            except Exception as e:
                print(f"[WARN] Normalization '{norm_name}' failed: {e}")
                continue

            # Save output
            out_path = class_dir / f"norm_{norm_name}_aug_{aug_name}.jpg"
            save_image(torch.clamp(img_tensor, 0, 1), out_path)

        print(f"  Saved samples for {class_name} → {class_dir}")

    print("\n  Visualization generation complete.")
    print(f"All samples saved under: {output_dir.resolve()}")

    # ================================================================
# 8. Data Loading Performance Demonstration
# ================================================================
    import time
    from tqdm import tqdm

    print("\n Measuring DataLoader performance...")
    start_time = time.time()
    num_batches = 0
    num_images = 0

    for imgs, labels in tqdm(dataloader, total=len(dataloader), desc="Loading Batches"):
        num_batches += 1
        num_images += imgs.size(0)
        # Ensure CHW correctness
        assert imgs.ndim == 4 and imgs.shape[1] in [1, 3], "Image batch not in CHW format!"

    elapsed = time.time() - start_time
    avg_batch_time = elapsed / num_batches
    img_per_sec = num_images / elapsed

    print("\n Data Loading Benchmark")
    print(f"  Total Batches     : {num_batches}")
    print(f"  Total Images      : {num_images}")
    print(f"  Avg Time/Batch    : {avg_batch_time:.4f} sec")
    print(f"  Throughput        : {img_per_sec:.2f} images/sec")
    print("   Data loading pipeline test complete.")

