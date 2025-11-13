import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from ..utils.cache_utils import cache_get, cache_set


def compute_object_density(ann_path: Path, dataset_dir: Path, sample_size: int = 5000):
    """
    Computes object density metrics (per image, per class, and global).
    """
    cached = cache_get("object_density", dataset_dir)
    if cached:
        print(f"[CACHE] Loaded object density for {dataset_dir.stem}")
        return cached

    print(f"[INFO] Loading annotations from {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    annotations = annotations[:sample_size]
    print(f"[INFO] Loaded {len(annotations)} annotations")

    total_objects = 0
    total_images = len(annotations)
    per_class_counts = {}
    objects_per_image = []

    for ann in tqdm(annotations, desc=f"[{dataset_dir.stem}] Counting objects"):
        labels = ann.get("labels", [])
        total_objects += len(labels)
        objects_per_image.append(len(labels))

        for label in labels:
            cat = label.get("category", "unknown")
            per_class_counts[cat] = per_class_counts.get(cat, 0) + 1

    # Compute per-class object density
    per_class_density = {
        c: count / total_images for c, count in per_class_counts.items()
    }

    result = {
        "subset": dataset_dir.stem,
        "total_images": total_images,
        "total_objects": total_objects,
        "objects_per_image_mean": float(np.mean(objects_per_image)),
        "objects_per_image_std": float(np.std(objects_per_image)),
        "objects_per_image_histogram": np.histogram(objects_per_image, bins=20)[0].tolist(),
        "per_class_density": per_class_density,
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss"
        }
    }

    cache_set("object_density", dataset_dir, result)
    print(f"[CACHE] Stored object density for {dataset_dir.stem}")
    return result
