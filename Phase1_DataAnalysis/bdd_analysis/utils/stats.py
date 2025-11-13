"""Statistical computations for BDD100K dataset analysis."""
import pandas as pd
import os
from .cache_utils import cache_get, cache_set
"""Dataset summary computation with Django caching."""

import json
from pathlib import Path
from datetime import datetime
from django.core.cache import cache
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def _count_annotations(json_path: Path) -> int:
    """Count total annotations (objects) in a BDD100K label file."""
    if not json_path.exists():
        return 0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sum(len(item.get("labels", [])) for item in data)


def compute_dataset_summary(recompute: bool = False) -> dict:
    """
    Compute dataset summary using Django cache.

    If `recompute=True`, clear cache and force recalculation.
    Cached results persist across reloads (file-based cache).
    """
    cache_key = "dataset_summary_v1"

    # Force recompute if requested
    if recompute:
        cache.delete(cache_key)

    # Return cached version if available
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    # Load paths from .env
    from pathlib import Path
    import os

    # Get current working directory
    cwd = Path.cwd()

    # Compute dataset root dynamically
    # If running inside Docker, data lives in /app/data/bdd100k
    # If local, it lives in ../Data/assignment_data_bdd
    if (cwd / "data" / "bdd100k").exists():
        dataset_root = cwd / "data" / "bdd100k"
    else:
        dataset_root = cwd.parent / "Data" / "assignment_data_bdd"

    # Allow override from env if set
    dataset_root = Path("/app/data/bdd100k") 
    print(f"Resolved dataset root**********: {dataset_root}")
    # Construct JSON paths
    train_json = dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
    val_json = dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    # Debug print
    print(f"Using dataset root: {dataset_root}")
    print(f"Train JSON path: {train_json}")
    print(f"Val JSON path: {val_json}")

    # Check if files exist
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            f" One or more dataset JSON files are missing.\n"
            f"Train: {train_json}\n"
            f"Val: {val_json}\n"
            f"Working Dir: {cwd}"
        )

    # Compute summary
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_images = len(train_data)
    val_images = len(val_data)
    train_annotations = _count_annotations(train_json)
    avg_objects_per_image = (
        train_annotations / train_images if train_images > 0 else 0
    )

    summary = {
        "Training Images": train_images,
        "Validation Images": val_images,
        "Training Annotations": train_annotations,
        "Avg Objects/Image": round(avg_objects_per_image, 2),
        "Dataset Root": str(dataset_root.resolve()),
        "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to cache for future requests
    cache.set(cache_key, summary)
    return summary
"""Compute train vs validation class distribution for BDD100K."""

import pandas as pd
from django.core.cache import cache


def _load_labels(json_path: Path) -> pd.DataFrame:
    """Helper: load a BDD100K label JSON into a flat DataFrame."""
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        image_name = item.get("name")
        for obj in item.get("labels", []):
            category = obj.get("category", "unknown")
            records.append({"image": image_name, "category": category})
    return pd.DataFrame(records)
    

def compute_class_distribution(recompute: bool = False) -> dict:
    """
    Compute class distribution for both TRAIN and VALIDATION sets.

    Returns:
        {
            "train": {
                "class_counts": {label: count, ...},
                "class_percent": {label: pct, ...},
                "total_images": int,
                "total_objects": int
            },
            "val": {...},
            "Last Computed": "YYYY-MM-DD HH:MM:SS"
        }
    """

    cache_key = "class_distribution_v2"

    if recompute:
        cache.delete(cache_key)

    cached = cache.get(cache_key)
    if cached:
        return cached

    # === Load dataset paths ===
    dataset_root = Path("/app/data/bdd100k") 
    print(f"Resolved dataset root**********: {dataset_root}")
    # Construct JSON paths
    train_json = dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
    val_json = dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    # Debug print
    print(f"Using dataset root: {dataset_root}")
    print(f"Train JSON path: {train_json}")
    print(f"Val JSON path: {val_json}")

    # Check if files exist
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            f" One or more dataset JSON files are missing.\n"
            f"Train: {train_json}\n"
            f"Val: {val_json}\n"
            f"Working Dir: {cwd}"
        )
    # === Load dataframes ===
    df_train = _load_labels(train_json)
    df_val = _load_labels(val_json)

    # === Compute distributions ===
    def _compute_stats(df: pd.DataFrame) -> dict:
        class_counts = df["category"].value_counts().sort_values(ascending=False)
        total_objects = int(class_counts.sum())
        class_percent = (class_counts / total_objects * 100).round(2).to_dict()
        return {
            "class_counts": class_counts.to_dict(),
            "class_percent": class_percent,
            "total_images": int(df["image"].nunique()),
            "total_objects": total_objects,
        }

    train_stats = _compute_stats(df_train)
    val_stats = _compute_stats(df_val)

    # === Prepare final result ===
    result = {
        "train": train_stats,
        "val": val_stats,
        "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


    result["train"]["log_class_counts"] = {
        cls: float(np.log10(count + 1)) for cls, count in result["train"]["class_counts"].items()
    }
    result["val"]["log_class_counts"] = {
        cls: float(np.log10(count + 1)) for cls, count in result["val"]["class_counts"].items()
    }



    cache.set(cache_key, result)
    return result
