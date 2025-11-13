"""
======================================================================
Convert BDD100K annotations from JSON to YOLO format (.txt)
======================================================================

Author: Utkarsh
Usage Example:
    python scripts/convert_bdd_to_yolo.py \
        --data-root /app/data/bdd100k \
        --output-root /app/data/bdd100k_yolo

----------------------------------------------------------------------
About:
----------------------------------------------------------------------
This script converts the official BDD100K dataset annotations (in JSON)
to YOLO format. The YOLO format represents each bounding box in the
following structure per line:

    class_id x_center y_center width height

All values are normalized to [0,1] by image width and height.

The script:
    1️ Reads the original BDD100K JSON annotation files.
    2️ Extracts bounding boxes and object categories.
    3️ Converts them into YOLO-compatible text files.
    4️ Generates one .txt file per image, containing all its objects.
    5️ Organizes outputs into train/val directories.

----------------------------------------------------------------------
Dataset Info:
----------------------------------------------------------------------
BDD100K contains 100K driving images (1280x720) annotated for
object detection tasks. There are 10 main classes used for detection:

    0. bike
    1. bus
    2. car
    3. motor
    4. person
    5. rider
    6. traffic light
    7. traffic sign
    8. train
    9. truck

======================================================================
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse


# =====================================================
# 10-Class Mapping for BDD100K Detection
# =====================================================
CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person',
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]
class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}


# =====================================================
# Core Conversion Function
# =====================================================
def convert_bdd_to_yolo(
    json_path: str,
    output_dir: str,
    split: str = 'train'
):
    """
    Convert BDD100K JSON labels to YOLO format.

    Args:
        json_path: Path to BDD100K JSON label file
        output_dir: Directory to save YOLO format labels
        split: Dataset split name (train/val)
    """

    print(f"\n{'='*70}")
    print(f" Converting BDD100K {split} labels to YOLO format")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading JSON annotations from: {json_path}")
    with open(json_path, 'r') as f:
        labels_data = json.load(f)

    print(f"Total images found: {len(labels_data)}")
    print(f"Saving YOLO labels to: {output_dir}\n")

    converted_count = 0
    skipped_count = 0
    total_boxes = 0

    # Iterate through all images in the JSON file
    for item in tqdm(labels_data, desc=f"Converting {split}"):
        img_name = item['name']
        attributes = item.get("attributes", {})

        # Print some meta attributes (optional — for analysis reporting)
        print(f"\nImage: {img_name}")
        print(f"  → Weather: {attributes.get('weather', 'N/A')}")
        print(f"  → Scene: {attributes.get('scene', 'N/A')}")
        print(f"  → Time of Day: {attributes.get('timeofday', 'N/A')}")

        # Fixed BDD100K image resolution
        img_width, img_height = 1280, 720

        # Parse bounding boxes
        yolo_lines = []

        for label in item.get('labels', []):
            if 'box2d' not in label:
                continue

            category = label['category']

            # Skip irrelevant categories
            if category not in class_to_idx:
                continue

            box2d = label['box2d']
            x1, y1 = box2d['x1'], box2d['y1']
            x2, y2 = box2d['x2'], box2d['y2']

            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                continue

            # Convert to YOLO normalized coordinates
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Clip to [0,1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            class_id = class_to_idx[category]

            # Optional print — show attributes of this object
            obj_attr = label.get("attributes", {})
            print(f"    Detected: {category}")
            print(f"      occluded: {obj_attr.get('occluded', False)}")
            print(f"      truncated: {obj_attr.get('truncated', False)}")
            print(f"      trafficLightColor: {obj_attr.get('trafficLightColor', 'none')}")
            print(f"      box2d: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

            # YOLO line format
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
            total_boxes += 1

        # Write .txt file if there are valid annotations
        if yolo_lines:
            txt_name = Path(img_name).stem + '.txt'
            txt_path = output_path / txt_name

            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            converted_count += 1
        else:
            skipped_count += 1

    print(f"\n{'='*70}")
    print(" Conversion Summary:")
    print(f"  → Split: {split}")
    print(f"  → Converted images: {converted_count}")
    print(f"  → Skipped (no valid annotations): {skipped_count}")
    print(f"  → Total bounding boxes: {total_boxes}")
    print(f"  → Average boxes per image: {total_boxes / max(converted_count, 1):.2f}")
    print(f"{'='*70}\n")


# =====================================================
# Entry Point
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLO format")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../../Data/assignment_data_bdd",  # base dataset directory inside Docker
        help="Root directory containing BDD100K dataset",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="../bdd100k_yolo",  # output directory for YOLO-format labels
        help="Root directory for YOLO-format labels",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # ==========================
    # Input dataset paths (from .env layout)
    # ==========================
    train_json = data_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
    val_json = data_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    train_images = data_root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "train"
    val_images = data_root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "val"

    # ==========================
    # Output directories
    # ==========================
    train_output = output_root / "labels" / "train"
    val_output = output_root / "labels" / "val"

    print("\n Using dataset structure:")
    print(f"  Train JSON : {train_json}")
    print(f"  Val JSON   : {val_json}")
    print(f"  Train imgs : {train_images}")
    print(f"  Val imgs   : {val_images}")
    print(f"  Output dir : {output_root}\n")

    # ==========================
    # Convert train split
    # ==========================
    if train_json.exists():
        convert_bdd_to_yolo(
            json_path=str(train_json),
            output_dir=str(train_output),
            split="train"
        )
    else:
        print(f" Train JSON not found at {train_json}")

    # ==========================
    # Convert val split
    # ==========================
    if val_json.exists():
        convert_bdd_to_yolo(
            json_path=str(val_json),
            output_dir=str(val_output),
            split="val"
        )
    else:
        print(f" Val JSON not found at {val_json}")

    print("\n All conversions completed successfully!")
    print(f"YOLO labels saved under: {output_root}/labels/")
    print("\nNext steps:")
    print("  1️ Update configs/bdd100k.yaml with correct paths")
    print("  2️ Run training: python models/yolov8_custom_train.py")


# =====================================================
# Script Entry
# =====================================================
if __name__ == "__main__":
    main()
