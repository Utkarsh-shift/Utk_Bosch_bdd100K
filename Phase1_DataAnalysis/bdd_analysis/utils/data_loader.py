"""
data_loader.py

Utility functions for loading BDD100K annotation data.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_annotations(subset: str = "train") -> dict:
    """
    Load BDD100K JSON annotations (train/val) and return a mapping:
        {image_id: [ {category, bbox, etc.}, ... ] }
    """
    # root = Path(os.getenv("BDD100K_ROOT", "data"))
    root = Path("/app/data/bdd100k/bdd100k_labels_release/bdd100k")  # Ensure consistent path in Docker
    if subset == "train":
        json_path = Path( root / "labels/bdd100k_labels_images_train.json")
    else:
        json_path = Path( root / "labels/bdd100k_labels_images_val.json")

    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    print(f"[INFO] Loading {subset} annotations from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_to_labels = {}

    # Loop through each annotated image entry
    for entry in annotations:
        image_name = Path(entry["name"]).stem  # e.g., 'b1c12d_0'
        labels = entry.get("labels", [])
        image_to_labels[image_name] = labels

    print(f"[INFO] Loaded {len(image_to_labels)} annotated images for {subset}")
    return image_to_labels
