"""
Enhanced Cross-platform Sample Visualization for BDD100K.
Includes bounding box overlays, co-occurrence analysis,
and consistent sample selection across OS.
"""

import os
import cv2
import json
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

from .cache_utils import cache_get, cache_set
from .data_loader import load_annotations


# ==============================================================
# Helper Functions
# ==============================================================

def safe_path_join(base_dir, filename):
    """Safely join paths across Windows/Linux."""
    filename = os.path.basename(str(filename))
    return Path(base_dir) / filename


def draw_bounding_boxes(image_path, labels, save_path):
    """Draw bounding boxes and class names on an image."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] Could not load {image_path}")
            return

        for lbl in labels:
            if "box2d" not in lbl:
                continue  # skip lane markings, polygons, etc.
            box = lbl["box2d"]
            cls = lbl.get("category", "")
            try:
                x1, y1 = int(box["x1"]), int(box["y1"])
                x2, y2 = int(box["x2"]), int(box["y2"])
            except (TypeError, KeyError):
                continue

            color = (0, 255, 0) if cls == "car" else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls[:12], (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(str(save_path), img)
    except Exception as e:
        print(f"[WARN] Could not draw boxes for {image_path}: {e}")


def compute_cooccurrence_patterns(annotations, top_k=12):
    """Compute co-occurrence pairs of object classes."""
    pairs = Counter()
    for ann in annotations:
        classes = sorted(set(l.get("category") for l in ann.get("labels", []) if l.get("category")))
        for a, b in itertools.combinations(classes, 2):
            pairs[(a, b)] += 1

    if not pairs:
        return [{"pair": ["N/A", "N/A"], "count": 0}]

    filtered = [{"pair": [a, b], "count": c} for (a, b), c in pairs.most_common(top_k) if c > 2]
    return filtered or [{"pair": ["N/A", "N/A"], "count": 0}]


# ==============================================================
# Sample Selection
# ==============================================================

def select_basic_samples(annotations, n=15):
    densities = [(a["name"], len(a.get("labels", []))) for a in annotations if "name" in a]
    if not densities:
        return []
    counts = sorted(densities, key=lambda x: x[1])
    mid = len(counts) // 2
    return [name for name, _ in counts[max(0, mid - n // 2): mid + n // 2]]


def select_extreme_density_samples(annotations, n=10):
    densities = [(a["name"], len(a.get("labels", []))) for a in annotations if "name" in a]
    if not densities:
        return {"low_density": [], "high_density": []}
    sorted_imgs = sorted(densities, key=lambda x: x[1])
    return {
        "low_density": [name for name, _ in sorted_imgs[:n // 2]],
        "high_density": [name for name, _ in sorted_imgs[-n // 2:]]
    }


def select_bbox_size_extremes(annotations, n=10):
    areas = []
    for ann in annotations:
        if not isinstance(ann, dict) or "labels" not in ann or "name" not in ann:
            continue
        total_area = 0
        for lbl in ann.get("labels", []):
            box = lbl.get("box2d")
            if not box:
                continue
            w, h = max(1, box["x2"] - box["x1"]), max(1, box["y2"] - box["y1"])
            total_area += w * h
        areas.append((ann["name"], total_area))

    sorted_imgs = sorted(areas, key=lambda x: x[1])
    return {
        "smallest_boxes": [name for name, _ in sorted_imgs[:n // 2]],
        "largest_boxes": [name for name, _ in sorted_imgs[-n // 2:]]
    }


def select_class_representatives(annotations, n=10):
    class_to_images = defaultdict(list)
    for ann in annotations:
        for lbl in ann.get("labels", []):
            cls = lbl.get("category")
            if cls:
                class_to_images[cls].append(ann["name"])

    sampled = {}
    for cls, imgs in class_to_images.items():
        if imgs and len(sampled) < n:
            sampled[cls] = imgs[0]
    return sampled


def select_diversity_samples(annotations, n=5):
    diversity = [(a["name"], len(set(l["category"] for l in a.get("labels", [])))) for a in annotations]
    top = sorted(diversity, key=lambda x: x[1], reverse=True)[:n]
    return [name for name, _ in top]


def select_occlusion_samples(annotations, n=5, threshold=0.3):
    def iou(b1, b2):
        xA, yA = max(b1["x1"], b2["x1"]), max(b1["y1"], b2["y1"])
        xB, yB = min(b1["x2"], b2["x2"]), min(b1["y2"], b2["y2"])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
        area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
        return inter / (area1 + area2 - inter + 1e-6)

    overlaps = []
    for ann in annotations:
        boxes = [l["box2d"] for l in ann.get("labels", []) if "box2d" in l]
        if len(boxes) < 2:
            continue
        overlap_count = sum(iou(b1, b2) > threshold for i, b1 in enumerate(boxes) for b2 in boxes[i + 1:])
        overlaps.append((ann["name"], overlap_count))

    top = sorted(overlaps, key=lambda x: x[1], reverse=True)[:n]
    return [name for name, _ in top]


# ==============================================================
# Main Function
# ==============================================================
def compute_sample_visualizations(
    image_dir: Path,
    ann_json: Path,
    output_dir="output_samples/organized_samples",
    copy_files=True
):
    """
    Compute curated visualization samples with bounding box overlays.
    Safe for malformed, nested, or partial BDD100K annotation files.
    """

    cache_key = "sample_visualizations"
    cached = cache_get(cache_key, image_dir)
    if cached:
        print("[CACHE] Loaded sample visualizations.")
        return cached

    # --- Load JSON safely ---
    try:
        with open(ann_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load {ann_json}: {e}")

    # --- Normalize structure ---
    annotations = []
    if isinstance(raw, dict):
        # Single annotation dict
        annotations = [raw]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                # Some BDD100K files nest a list of dicts under "frames"
                if "frames" in item and isinstance(item["frames"], list):
                    annotations.extend(f for f in item["frames"] if isinstance(f, dict))
                else:
                    annotations.append(item)
    elif isinstance(raw, str):
        # If stringified JSON
        try:
            annotations = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string in {ann_json}")
    else:
        raise TypeError(f"Unexpected JSON type: {type(raw)}")

    # --- Filter out invalid entries ---
    annotations = [a for a in annotations if isinstance(a, dict) and "name" in a]

    if not annotations:
        raise ValueError(f"No valid annotations found in {ann_json}")

    print(f"[INFO] Loaded {len(annotations)} valid annotation entries for visualization.")

    # --- Ensure output dir exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Compute result structure ---
    result = {
        "1_basic_samples": select_basic_samples(annotations),
        "2_extreme_density": select_extreme_density_samples(annotations),
        "3_bbox_size_extremes": select_bbox_size_extremes(annotations),
        "4_class_representatives": select_class_representatives(annotations),
        "5_diversity_samples": select_diversity_samples(annotations),
        "6_occlusion_samples": select_occlusion_samples(annotations),
        "7_cooccurrence_patterns": compute_cooccurrence_patterns(annotations),
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss"
        }
    }

    # --- Save annotated images ---
    if copy_files:
        used_images = set()
        for key, group in result.items():
            if key.startswith("meta") or key == "7_cooccurrence_patterns":
                continue

            subdir = Path(output_dir) / key
            subdir.mkdir(parents=True, exist_ok=True)

            imgs = []
            if isinstance(group, dict):
                for val in group.values():
                    imgs.extend(val if isinstance(val, list) else [val])
            elif isinstance(group, list):
                imgs.extend(group)

            for img_name in imgs:
                if not isinstance(img_name, str) or img_name in used_images:
                    continue
                used_images.add(img_name)

                src = safe_path_join(image_dir, img_name)
                if not src.exists():
                    alt = list(image_dir.glob(f"{Path(img_name).stem}.*"))
                    if alt:
                        src = alt[0]
                    else:
                        print(f"[WARN] Missing {img_name}")
                        continue

                ann = next((a for a in annotations if a.get("name") == img_name), None)
                dst = subdir / Path(img_name).name
                if ann:
                    draw_bounding_boxes(src, ann.get("labels", []), dst)
                else:
                    shutil.copy(str(src), str(dst))

    # --- Cache and return ---
    cache_set(cache_key, image_dir, result)
    print("[CACHE] Stored sample visualizations.")
    return result
