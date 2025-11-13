"""
Per-Class Image Data Quality Analysis for BDD100K.
Computes sharpness, exposure (brightness), contrast, colorfulness, resolution consistency, and corruption stats.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict
from ..utils.cache_utils import cache_get, cache_set
from ..utils.data_loader import load_annotations


# ========== Metric Functions ==========

def variance_of_laplacian(image: np.ndarray) -> float:
    """Sharpness score (variance of Laplacian). Low → blurry image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_exposure(image: np.ndarray) -> float:
    """Exposure score (mean brightness)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


def compute_contrast(image: np.ndarray) -> float:
    """Contrast score (standard deviation of grayscale intensities)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def compute_colorfulness(image: np.ndarray) -> float:
    """Colorfulness (Hasler & Süsstrunk metric)."""
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + (0.3 * np.sqrt(mean_rg**2 + mean_yb**2)))


# ========== Main Function ==========

def compute_data_quality_metrics_per_class(
    image_dir: Path, ann_json: Path, sample_size: int = 300
):
    """
    Perform per-class data quality analysis:
    - Sharpness
    - Exposure (Brightness)
    - Contrast
    - Colorfulness
    - Resolution
    - Corruption check
    """

    cache_key = "data_quality_metrics_per_class"
    cached = cache_get(cache_key, image_dir)
    if cached:
        print("[CACHE] Loaded per-class data quality metrics.")
        return cached

    import json

    with open(ann_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"[INFO] Loaded {len(annotations)} annotated images.")

    # Create class → image mapping
    class_to_images = defaultdict(list)
    for ann in annotations:
        img_name = ann["name"]
        labels = [lbl["category"] for lbl in ann.get("labels", [])]
        for cls in labels:
            class_to_images[cls].append(img_name)

    # Limit sample size per class
    for cls in class_to_images:
        class_to_images[cls] = list(set(class_to_images[cls]))[:sample_size]

    # Results container
    per_class_results = {}

    for cls, image_names in class_to_images.items():
        print(f"[{cls}] Processing {len(image_names)} samples...")
        metrics = {
            "exposure": [],
            "contrast": [],
            "sharpness": [],
            "colorfulness": [],
            "widths": [],
            "heights": [],
            "corrupt": 0,
        }

        for img_name in tqdm(image_names, desc=f"[QUALITY] {cls}", leave=False):
            img_path = Path(image_dir) / img_name
            if not img_path.exists():
                metrics["corrupt"] += 1
                continue

            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                h, w = img.shape[:2]

                # Compute metrics
                metrics["exposure"].append(compute_exposure(img))
                metrics["contrast"].append(compute_contrast(img))
                metrics["sharpness"].append(variance_of_laplacian(img))
                metrics["colorfulness"].append(compute_colorfulness(img))
                metrics["widths"].append(w)
                metrics["heights"].append(h)

            except Exception:
                metrics["corrupt"] += 1
                continue

        # Skip empty classes
        if len(metrics["exposure"]) == 0:
            continue

        # Summary stats per class
        per_class_results[cls] = {
            "exposure_mean": float(np.mean(metrics["exposure"])),
            "exposure_std": float(np.std(metrics["exposure"])),
            "contrast_mean": float(np.mean(metrics["contrast"])),
            "contrast_std": float(np.std(metrics["contrast"])),
            "sharpness_mean": float(np.mean(metrics["sharpness"])),
            "colorfulness_mean": float(np.mean(metrics["colorfulness"])),
            "avg_resolution": [float(np.mean(metrics["widths"])), float(np.mean(metrics["heights"]))],
            "corrupt_count": metrics["corrupt"],
            "num_samples": len(metrics["exposure"]),
        }

    summary = {
        "subset": "train",
        "sample_size_per_class": sample_size,
        "per_class_quality": per_class_results,
        "Last Computed": str(datetime.now()),
    }

    cache_set(cache_key, image_dir, summary)
    print("[CACHE] Stored per-class data quality metrics.")
    return summary
