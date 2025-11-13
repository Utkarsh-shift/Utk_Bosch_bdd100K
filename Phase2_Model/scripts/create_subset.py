"""
======================================================================
Create a YOLO-Formatted Subset of the BDD100K Dataset
======================================================================

Author: Utkarsh

Description:
------------
This script creates a smaller YOLO-compatible subset of the BDD100K dataset.
It selects a random subset of images and their corresponding YOLO label files.
The resulting dataset can be used directly for YOLOv8/YOLOv10 training.

Output Folder Structure:
------------------------
bdd100k_yolo_subset/
├── images/train/
├── labels/train/
├── images/val/
└── labels/val/

Usage Example:
--------------
python scripts/create_yolo_subset.py \
    --images-root /bdd_model/bdd100k_images_100k/bdd100k/images/100k \
    --labels-root /bdd_model/bdd100k_yolo/labels \
    --output-root /bdd_model/bdd100k_yolo_subset \
    --split train \
    --num-samples 2000 \
    --seed 42
======================================================================
"""

import shutil
import random
from pathlib import Path
import argparse


def create_yolo_subset(
    images_root: Path,
    labels_root: Path,
    output_root: Path,
    split: str = "train",
    num_samples: int = 1000,
    seed: int = 42
) -> None:
    """
    Create a YOLO-formatted subset of BDD100K.

    Args:
        images_root (Path): Path to BDD100K original images directory.
        labels_root (Path): Path to YOLO label directory.
        output_root (Path): Path where subset will be saved.
        split (str): Dataset split ('train' or 'val').
        num_samples (int): Number of samples to include in subset.
        seed (int): Random seed for reproducibility.
    """
    print("\n" + "=" * 70)
    print(f" Creating YOLO Subset for BDD100K ({split.upper()}) Split")
    print("=" * 70)
    print(f" Images Root : {images_root}")
    print(f" Labels Root : {labels_root}")
    print(f" Output Path : {output_root}")
    print(f" Samples     : {num_samples}")
    print(f" Random Seed : {seed}")
    print("=" * 70 + "\n")

    # Define source directories
    src_img_dir = images_root / split
    src_lbl_dir = labels_root / split

    # Define destination directories
    dst_img_dir = output_root / "images" / split
    dst_lbl_dir = output_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f" Source images : {src_img_dir}")
    print(f" Source labels : {src_lbl_dir}")
    print(f" Output images : {dst_img_dir}")
    print(f" Output labels : {dst_lbl_dir}\n")

    # Collect available images
    all_images = sorted(list(src_img_dir.glob("*.jpg")))
    if not all_images:
        print(f" No image files found in {src_img_dir}")
        return

    total_images = len(all_images)
    print(f" Found {total_images} total images in source.")

    # Random sampling
    random.seed(seed)
    subset_images = random.sample(all_images, min(num_samples, total_images))

    copied = 0
    missing = 0

    print(" Copying image-label pairs...\n")
    for img_path in subset_images:  
        # print(" Processing:", img_path)
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        # print(" Corresponding label:", label_path)
        if not label_path.exists():
            # print(f" Skipping {img_path.name} (no matching label found)")
            missing += 1
            continue

        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(label_path, dst_lbl_dir / label_path.name)
        copied += 1

        if copied % 100 == 0:
            print(f" Copied {copied} pairs so far...")

    print("\n" + "=" * 70)
    print(f" Subset Creation Completed for {split.upper()}")
    print("=" * 70)
    print(f" Total Images Available : {total_images}")
    print(f" Total Copied Pairs     : {copied}")
    print(f" Missing Labels Skipped  : {missing}")
    print(f" Images saved in         : {dst_img_dir}")
    print(f" Labels saved in         : {dst_lbl_dir}")

    # Integrity Check
    final_imgs = len(list(dst_img_dir.glob("*.jpg")))
    final_lbls = len(list(dst_lbl_dir.glob("*.txt")))
    print("\n Integrity Check Summary:")
    print(f"  → Images in subset: {final_imgs}")
    print(f"  → Labels in subset: {final_lbls}")
    print(f"  → Mismatch count  : {abs(final_imgs - final_lbls)}")
    print("=" * 70 + "\n")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Create a YOLO subset of BDD100K dataset.")
    parser.add_argument(
        "--images-root",
        type=str,
        default="../../Data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k",
    )

    parser.add_argument(
        "--labels-root",
        type=str,
        default="../bdd100k_yolo/labels",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="../bdd100k_yolo_subset",
    )

    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Dataset split to subset (train/val)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to include in the subset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=144,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    create_yolo_subset(
        images_root=Path(args.images_root),
        labels_root=Path(args.labels_root),
        output_root=Path(args.output_root),
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
