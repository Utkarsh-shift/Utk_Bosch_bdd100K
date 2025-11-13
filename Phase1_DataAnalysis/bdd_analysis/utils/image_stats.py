"""
image_stats.py

Compute brightness, contrast, and saturation distributions per class
for the BDD100K dataset with dataset-aware caching.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
from collections import defaultdict
from datetime import datetime
from ..utils.cache_utils import cache_get, cache_set
from ..utils.data_loader import load_annotations

load_dotenv()


# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------


def _get_image_paths(subset: str) -> list[Path]:
    """Return list of image paths for train or val set."""
    root = Path(os.getenv("BDD100K_ROOT", "data"))
    if subset == "train":
        img_dir = Path(
            os.getenv(
                "BDD100K_TRAIN_IMAGES",
                root / "bdd100k_images_100k/bdd100k/images/100k/train",
            )
        )
    else:
        img_dir = Path(
            os.getenv(
                "BDD100K_VAL_IMAGES",
                root / "bdd100k_images_100k/bdd100k/images/100k/val",
            )
        )

    extensions = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in extensions:
        paths.extend(img_dir.rglob(ext))

    print(f"[INFO] {subset.upper()} | Found {len(paths)} images in {img_dir}")
    return paths


def _compute_image_metrics(image_path: Path) -> dict:
    """
    Compute brightness, contrast, and saturation for a given image.
    Brightness: mean grayscale intensity (0–255)
    Contrast: RMS contrast (std of grayscale)
    Saturation: mean S channel in HSV (0–255)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32)

        # Convert to grayscale for brightness/contrast
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))  # RMS contrast

        # Convert to HSV for saturation
        hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))

        return {"brightness": brightness, "contrast": contrast, "saturation": saturation}
    except Exception as e:
        print(f"[WARN] Failed on {image_path.name}: {e}")
        return None


# --------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------


def compute_visual_distributions(
    subset: str = "train", sample_size: int = 20000, recompute: bool = False
) -> dict:
    """
    Compute per-class brightness, contrast, and saturation distributions.
    Uses dataset-aware caching with checksum-based invalidation.
    """
    dataset_dir = Path(os.getenv("BDD100K_ROOT", "data"))
    cache_key = f"visual_stats_{subset}_bcs"

    #  Try to load from cache
    if not recompute:
        cached = cache_get(cache_key, dataset_dir)
        if cached:
            cached["meta"]["cache_status"] = "hit"
            print(f"[CACHE] Loaded visual stats for {subset}")
            return cached

    # Load data
    img_paths = _get_image_paths(subset)
    annotations = load_annotations(subset)

    if not img_paths or not annotations:
        print(f"[WARN] No data found for {subset}.")
        return {}

    np.random.seed(42)
    sampled_paths = np.random.choice(
        img_paths, min(len(img_paths), sample_size), replace=False
    )

    # Containers
    brightness_per_class = defaultdict(list)
    contrast_per_class = defaultdict(list)
    saturation_per_class = defaultdict(list)

    # Process images
    for img_path in tqdm(sampled_paths, desc=f"[{subset}] Computing B/C/S"):
        metrics = _compute_image_metrics(img_path)
        if not metrics:
            continue

        image_id = img_path.stem
        labels = annotations.get(image_id, [])
        for label in labels:
            cat = label.get("category")
            brightness_per_class[cat].append(metrics["brightness"])
            contrast_per_class[cat].append(metrics["contrast"])
            saturation_per_class[cat].append(metrics["saturation"])

    # Compute histograms (20 bins)
    brightness_dist_per_class = {}
    contrast_dist_per_class = {}
    saturation_dist_per_class = {}
    mean_brightness_per_class = {}
    mean_contrast_per_class = {}
    mean_saturation_per_class = {}

    for cls in brightness_per_class.keys():
        b_vals = brightness_per_class[cls]
        c_vals = contrast_per_class[cls]
        s_vals = saturation_per_class[cls]

        if not b_vals:
            continue

        b_hist, _ = np.histogram(b_vals, bins=20, range=(0, 255))
        c_hist, _ = np.histogram(c_vals, bins=20, range=(0, 128))
        s_hist, _ = np.histogram(s_vals, bins=20, range=(0, 255))

        brightness_dist_per_class[cls] = b_hist.tolist()
        contrast_dist_per_class[cls] = c_hist.tolist()
        saturation_dist_per_class[cls] = s_hist.tolist()

        mean_brightness_per_class[cls] = float(np.mean(b_vals))
        mean_contrast_per_class[cls] = float(np.mean(c_vals))
        mean_saturation_per_class[cls] = float(np.mean(s_vals))

    # Final result
    result = {
        "subset": subset,
        "brightness_distribution_per_class": brightness_dist_per_class,
        "contrast_distribution_per_class": contrast_dist_per_class,
        "saturation_distribution_per_class": saturation_dist_per_class,
        "mean_brightness_per_class": mean_brightness_per_class,
        "mean_contrast_per_class": mean_contrast_per_class,
        "mean_saturation_per_class": mean_saturation_per_class,
        "sample_size": len(sampled_paths),
        "meta": {
            "Computation Time (s)": 0.0,
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss",
        },
    }

    # Store in cache
    cache_set(cache_key, dataset_dir, result)
    print(f"[CACHE] Stored visual stats for {subset}")
    return result
