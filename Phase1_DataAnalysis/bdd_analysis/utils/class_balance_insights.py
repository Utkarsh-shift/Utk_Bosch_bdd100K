"""
Compute advanced class balance and dataset bias metrics for BDD100K.
Includes entropy, Gini, drift analysis, and resampling weights with caching.
"""

import os
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from .data_loader import load_annotations
from .cache_utils import cache_get, cache_set

# ===========================================================
# Utility Metrics
# ===========================================================

def gini_index(probs):
    """Compute Gini index: G = 1 - sum(p_i^2)."""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return float(1 - np.sum(probs ** 2))

def shannon_entropy(probs):
    """Compute Shannon entropy: H = -sum(p_i * log2(p_i))."""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def effective_num_classes(entropy):
    """Effective number of uniform classes."""
    return round(2 ** entropy, 2)

def normalized_difference(train_percent, val_percent):
    """Per-class normalized drift."""
    return round(train_percent - val_percent, 3)

def compute_sampling_suggestions(class_percents):
    """Generate sampling strategy hints and reweighting vector."""
    weights = {}
    total_inv = sum(1.0 / p if p > 0 else 0 for p in class_percents.values())
    for cls, p in class_percents.items():
        if p < 1:
            suggestion = "🔼 Oversample"
        elif p > 25:
            suggestion = "🔽 Undersample"
        else:
            suggestion = "✅ Balanced"
        weights[cls] = {
            "sampling_action": suggestion,
            "weight": round((1.0 / p) / total_inv, 4) if p > 0 else 0
        }
    return weights


# ===========================================================
# Main Function
# ===========================================================

def compute_class_balance_insights(train_json_path, val_json_path):
    """
    Compute class distribution, imbalance ratios, Gini index, entropy,
    and sampling suggestions. Uses caching.
    """

    cache_key = "class_balance_insights"
    dataset_dir = Path(train_json_path).parent
    cached = cache_get(cache_key, dataset_dir)
    if cached:
        print("[CACHE] Loaded class balance insights.")
        return cached

    # ---- Load JSON safely ----
    def safe_load(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle both list or dict structures
            if isinstance(data, dict):
                data = [data]
            return [x for x in data if isinstance(x, dict)]
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return []

    train_data = safe_load(train_json_path)
    val_data = safe_load(val_json_path)

    # ---- Count class occurrences ----
    def count_classes(data):
        counts = {}
        for entry in data:
            for label in entry.get("labels", []):
                cls = label.get("category")
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1
        return counts

    train_counts = count_classes(train_data)    
    val_counts = count_classes(val_data)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())

    # ---- Percentages ----
    train_percent = {k: (v / total_train) * 100 for k, v in train_counts.items()}
    val_percent = {k: (v / total_val) * 100 for k, v in val_counts.items()}

    # ---- Metrics ----
    probs = np.array(list(train_counts.values())) / total_train
    gini = gini_index(probs)
    entropy = shannon_entropy(probs)
    eff_classes = effective_num_classes(entropy)

    max_class = max(train_counts, key=train_counts.get)
    min_class = min(train_counts, key=train_counts.get)
    imbalance_ratio = train_counts[max_class] / max(1, train_counts[min_class])

    drift = {
        cls: normalized_difference(train_percent.get(cls, 0), val_percent.get(cls, 0))
        for cls in sorted(set(train_percent) | set(val_percent))
    }
    mean_abs_drift = round(np.mean(np.abs(list(drift.values()))), 3)

    sampling_weights = compute_sampling_suggestions(train_percent)

    # ---- Cumulative Distribution ----
    sorted_classes = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
    cumulative, running = [], 0
    for cls, count in sorted_classes:
        running += count
        cumulative.append({"class": cls, "cumulative_percent": round((running / total_train) * 100, 2)})





    # ---- Class Coverage Stats ----
    coverage_over_1 = sum(1 for p in train_percent.values() if p >= 1)
    coverage_under_1 = len(train_percent) - coverage_over_1
    coverage_ratio = round((coverage_over_1 / len(train_percent)) * 100, 2) if len(train_percent) > 0 else 0

    coverage_stats = {
        "classes_over_1_percent": coverage_over_1,
        "classes_under_1_percent": coverage_under_1,
        "coverage_ratio": f"{coverage_ratio}%",
    }







    # ---- Final Output ----
    insights = {
        "summary": {
            "total_train_objects": total_train,
            "total_val_objects": total_val,
            "most_common_class": max_class,
            "most_common_count": train_counts[max_class],
            "least_common_class": min_class,
            "least_common_count": train_counts[min_class],
            "class_imbalance_ratio": f"{imbalance_ratio:,.1f}:1",
            "gini_index": round(gini, 4),
            "shannon_entropy": round(entropy, 4),
            "effective_classes": eff_classes,
            "train_val_drift": mean_abs_drift,
        },
        "coverage_stats": coverage_stats,   
        "train_distribution": train_percent,
        "val_distribution": val_percent,
        "drift_analysis": drift,
        "cumulative_distribution": cumulative,
        "sampling_suggestions": sampling_weights,
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss"
        }
    }

    cache_set(cache_key, dataset_dir, insights)
    print("[CACHE] Stored class balance insights.")
    return insights
