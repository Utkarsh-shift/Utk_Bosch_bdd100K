"""
======================================================================
End-to-End Custom YOLOv8m Training Pipeline for BDD100K
======================================================================

Author: Utkarsh

Pipeline Stages:
----------------
1️ Convert BDD100K annotations (JSON → YOLO format)
2️ Create subset of YOLO dataset (train & val)
3️ Train YOLOv8m using your custom DataLoader

======================================================================
"""
import sys
from pathlib import Path

# ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import os
import json
import random
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn 

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import copy
import time
import os

from models.yolo_dataloader import build_dataloader

import torch.nn.functional as F
from ultralytics.nn.modules import Detect
# Custom modules


from ultralytics import YOLO
from models.yolo_dataloader import build_dataloader

import yaml
from pathlib import Path

def load_config(config_path="./scripts/config.yaml"):
    """Load configuration YAML file."""
    config_path = Path(config_path)
    print("The config path is:::::::::::::::::", config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"\n Loaded config from {config_path}")
    return config

CONFIG = load_config()

# ======================================================
# BDD100K Object Detection Classes
# ======================================================
CLASS_NAMES = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]

# Create ID mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

print(f" Loaded {len(CLASS_NAMES)} BDD100K classes:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {i:2d}: {cls}")


# Load paths and training params from config
# ---------------------------
# Resolve config paths relative to project root (ROOT_DIR)
# ---------------------------
# ROOT_DIR points to Phase2_Model (scripts/..)/..
# We resolve relative paths in config to absolute paths here.
DATA_ROOT = (ROOT_DIR / Path(CONFIG["paths"]["data_root"])).resolve()
IMAGES_ROOT = (ROOT_DIR / Path(CONFIG["paths"]["images_root"])).resolve()
OUTPUT_ROOT = (ROOT_DIR / Path(CONFIG["paths"]["output_root"])).resolve()
OUTPUT_SUBSET = (ROOT_DIR / Path(CONFIG["paths"]["subset_root"])).resolve()

# Training / hardware settings
DEVICE = CONFIG["hardware"]["device"]
EPOCHS = CONFIG["training"]["epochs"]
BATCH_SIZE = CONFIG["training"]["batch_size"]
IMG_SIZE = CONFIG["training"]["img_size"]
LR = CONFIG["training"]["lr"]
CLASS_NAMES = CONFIG["dataset"]["class_names"]
NUM_CLASSES = CONFIG["dataset"]["num_classes"]

class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}



# =====================================================
# Core Conversion Function
# =====================================================

import random
from pathlib import Path
from tqdm import tqdm
import json
print(f"\n[CONFIG DEBUG] Using device: {DEVICE}, epochs: {EPOCHS}, classes: {NUM_CLASSES}")
print(f"[CONFIG DEBUG] Output dirs → {OUTPUT_ROOT}, subset → {OUTPUT_SUBSET}")

# =======================================================
# Run Conversion (both train and val)
# =======================================================
def run_conversion(data_root: Path, output_root: Path):
    """
    Executes the full conversion pipeline for both train and val splits.
    """


class YOLOWithAdapter(nn.Module):
    def __init__(self, base_model, num_new_classes=CONFIG["dataset"]["num_classes"]):
        super().__init__()
        self.base = base_model.model  # raw YOLOv8 model
        self.adapter = nn.Conv2d(144, 64 + num_new_classes, 1)  # project from 144 → 74 channels

    def forward(self, x):
        y = self.base(x)
        if isinstance(y, (list, tuple)):
            y = y[0]  # some YOLO heads return list
        y = self.adapter(y)
        return y


def rebuild_detect_head(model, num_classes, class_names=None):
    """
    Rebuild YOLOv8 Detect head for a new number of classes.
    Works for all YOLOv8 architectures (nano → x).
    """
    detect_layer = None
    for m in model.model.modules():
        if isinstance(m, Detect):
            detect_layer = m
            break

    if detect_layer is None:
        raise RuntimeError("Detect layer not found in YOLO model. Ensure this is a YOLOv8 architecture.")

    old_nc = detect_layer.nc
    old_no = detect_layer.no
    reg_max = detect_layer.reg_max
    anchors = len(detect_layer.stride)

    print(f"\n[INFO] Rebuilding Detect Head:")
    print(f"   → Old classes: {old_nc}")
    print(f"   → New classes: {num_classes}")
    print(f"   → reg_max: {reg_max}, Anchors: {anchors}")

    # Calculate new output dimension
    detect_layer.nc = num_classes
    detect_layer.no = num_classes + reg_max * 4
    detect_layer.names = class_names or [f"class_{i}" for i in range(num_classes)]

    # rebuild conv heads
    detect_layer.cv2 = nn.ModuleList([
        nn.Conv2d(x[0].in_channels if isinstance(x, nn.Sequential) else x.in_channels,
                  detect_layer.no, 1) for x in detect_layer.cv2
    ])
    detect_layer.cv3 = nn.ModuleList([
        nn.Conv2d(x[0].in_channels if isinstance(x, nn.Sequential) else x.in_channels,
                  detect_layer.no, 1) for x in detect_layer.cv3
    ])

    print(f"[INFO] Detect head rebuilt → output channels: {detect_layer.no}")
    return model











def convert_bdd_to_yolo(
    json_path: str,
    output_dir: str,
    split: str = 'train',
    sample_size: int = 2000
):
    """
    Convert BDD100K JSON labels to YOLO format (sampled subset).

    Args:
        json_path: Path to BDD100K JSON label file
        output_dir: Directory to save YOLO format labels
        split: Dataset split name (train/val)
        sample_size: Number of samples to randomly convert
    """

    print(f"\n{'='*70}")
    print(f" Converting BDD100K {split.upper()} labels to YOLO format (sampled subset)")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------
    print(f" Loading JSON annotations from: {json_path}")
    with open(json_path, 'r') as f:
        labels_data = json.load(f)

    total_images = len(labels_data)
    print(f"  Total images in original {split}: {total_images}")

    # ------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------
    random.seed(42)
    sample_size = min(sample_size, total_images)
    sampled_data = random.sample(labels_data, sample_size)
    print(f" Sampling {sample_size} images for conversion.\n")

    # ------------------------------------------------------------
    # Conversion Loop
    # ------------------------------------------------------------
    converted_count = 0
    skipped_count = 0
    total_boxes = 0

    for item in tqdm(sampled_data, desc=f"Converting {split} subset"):
        img_name = item['name']
        img_width, img_height = 1280, 720  # fixed BDD100K resolution
        yolo_lines = []

        for label in item.get('labels', []):
            if 'box2d' not in label:
                continue

            category = label['category']
            if category not in class_to_idx:
                continue

            x1, y1 = label['box2d']['x1'], label['box2d']['y1']
            x2, y2 = label['box2d']['x2'], label['box2d']['y2']

            if x2 <= x1 or y2 <= y1:
                continue

            # YOLO normalized coords
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Clip
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            cls_id = class_to_idx[category]
            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            total_boxes += 1

        # Save
        if yolo_lines:
            txt_file = output_path / f"{Path(img_name).stem}.txt"
            with open(txt_file, "w") as f:
                f.write("\n".join(yolo_lines))
            converted_count += 1
        else:
            skipped_count += 1

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f" Conversion Summary for {split.upper()} Subset")
    print(f"{'='*70}")
    print(f"  → Total Available Images   : {total_images}")
    print(f"  → Sampled & Converted      : {converted_count}")
    print(f"  → Skipped (no valid boxes) : {skipped_count}")
    print(f"  → Total Bounding Boxes     : {total_boxes}")
    print(f"  → Avg Boxes per Image      : {total_boxes / max(converted_count, 1):.2f}")
    print(f"{'='*70}\n")

    print(f" YOLO labels written to: {output_path}\n")


# =======================================================
# Verify YOLO Label Integrity
# =======================================================
import re

def verify_yolo_labels(images_dir: Path, labels_dir: Path, num_classes: int = len(CLASS_NAMES)):
    """
    Verify YOLO label file integrity and consistency with images.

    Args:
        images_dir (Path): Path to images (train/val).
        labels_dir (Path): Path to YOLO .txt labels.
        num_classes (int): Number of classes in the dataset.

    Returns:
        summary (dict): Summary of verification statistics.
    """
    print("\n" + "=" * 70)
    print(f" Verifying YOLO Label Integrity for {images_dir.name.upper()} Split")
    print("=" * 70)

    if not images_dir.exists():
        print(f"[ERROR] Image directory not found: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"[ERROR] Label directory not found: {labels_dir}")
        return

    image_files = sorted(list(images_dir.glob("*.jpg")))
    label_files = sorted(list(labels_dir.glob("*.txt")))

    print(f"  Found {len(image_files)} images")
    print(f" Found {len(label_files)} label files\n")

    # Regex to validate YOLO format line: class_id x y w h
    pattern = re.compile(r"^\d+\s+0\.\d+\s+0\.\d+\s+0\.\d+\s+0\.\d+$")

    missing_labels = []
    extra_labels = []
    invalid_entries = []
    total_boxes = 0
    valid_boxes = 0

    # Check image-label pair existence
    image_stems = {img.stem for img in image_files}
    label_stems = {lbl.stem for lbl in label_files}

    # Missing or extra
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    if missing_labels:
        print(f"  Missing label files for {len(missing_labels)} images.")
    if extra_labels:
        print(f"  Extra label files without matching images: {len(extra_labels)}")

    # Validate label contents
    for lbl_file in tqdm(label_files, desc="Validating label files"):
        with open(lbl_file, "r") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split()
                if len(parts) != 5:
                    invalid_entries.append((lbl_file.name, line_no, "Incorrect column count"))
                    continue

                cls, x, y, w, h = map(float, parts)
                cls = int(cls)

                if cls < 0 or cls >= num_classes:
                    invalid_entries.append((lbl_file.name, line_no, f"Invalid class ID {cls}"))
                    continue

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    invalid_entries.append((lbl_file.name, line_no, "Coords out of range [0,1]"))
                    continue

                valid_boxes += 1
                total_boxes += 1

            except Exception as e:
                invalid_entries.append((lbl_file.name, line_no, str(e)))
                continue

    # Print summary
    print("\n" + "-" * 70)
    print(" YOLO Label Verification Summary")
    print("-" * 70)
    print(f"  • Total images checked     : {len(image_files)}")
    print(f"  • Total label files checked: {len(label_files)}")
    print(f"  • Valid bounding boxes     : {valid_boxes}")
    print(f"  • Invalid label entries    : {len(invalid_entries)}")
    print(f"  • Missing label files      : {len(missing_labels)}")
    print(f"  • Extra label files        : {len(extra_labels)}")
    print("-" * 70)

    if invalid_entries:
        print("   Invalid entries found (showing first 10):")
        for f, line_no, reason in invalid_entries[:10]:
            print(f"     → {f} [line {line_no}]: {reason}")
    else:
        print("  All label files passed format validation!")

    print("=" * 70 + "\n")

    return {
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "valid_boxes": valid_boxes,
        "invalid_entries": len(invalid_entries),
        "missing_labels": len(missing_labels),
        "extra_labels": len(extra_labels)
    }




# ============================================================
# STEP 2 — Create Subset
# ============================================================
import random
import shutil
from pathlib import Path

def create_yolo_subset(
    images_root: Path,
    labels_root: Path,
    output_root: Path,
    split: str = "train",
    num_samples: int = 2000,
    seed: int = 42
) -> None:
    """
    Create a YOLO-formatted subset of BDD100K containing only valid image-label pairs.
    Randomly samples from images but ensures matching YOLO labels exist.

    Args:
        images_root (Path): Path to BDD100K original images directory.
        labels_root (Path): Path to YOLO label directory.
        output_root (Path): Path where subset will be saved.
        split (str): Dataset split ('train' or 'val').
        num_samples (int): Number of valid pairs to include in subset.
        seed (int): Random seed for reproducibility.
    """
    print("\n" + "=" * 70)
    print(f" Creating YOLO Subset for BDD100K ({split.upper()}) Split")
    print("=" * 70)
    print(f" Images Root : {images_root}")
    print(f" Labels Root : {labels_root}")
    print(f" Output Path : {output_root}")
    print(f" Target Samples: {num_samples}")
    print(f" Random Seed : {seed}")
    print("=" * 70 + "\n")

    # Define directories
    src_img_dir = images_root / split
    print("The source image directory is:::::::::::::::", src_img_dir)
    src_lbl_dir = labels_root / split
    print("The source label directory is:::::::::::::::", src_lbl_dir)
    dst_img_dir = output_root / "images" / split
    dst_lbl_dir = output_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect available images
    all_images = sorted(list(src_img_dir.glob("*.jpg")))
    total_images = len(all_images)
    print(f" Found {total_images} total images in source.\n")

    # Filter only valid pairs (image + label exist)
    valid_images = [
        img for img in all_images
        if (src_lbl_dir / f"{img.stem}.txt").exists()
    ]
    total_valid = len(valid_images)

    if total_valid == 0:
        print(" No valid image-label pairs found!")
        return

    print(f"  Found {total_valid} valid image-label pairs.")
    num_samples = min(num_samples, total_valid)
    print(f"  Sampling {num_samples} random pairs...\n")

    # Randomly sample valid pairs
    random.seed(seed)
    subset_images = random.sample(valid_images, num_samples)

    # Copy sampled pairs
    copied = 0
    for img_path in subset_images:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(label_path, dst_lbl_dir / label_path.name)
        copied += 1

        if copied % 200 == 0:
            print(f" Copied {copied} pairs so far...")

    # Final integrity summary
    print("\n" + "=" * 70)
    print(f"  Subset Creation Completed for {split.upper()}")
    print("=" * 70)
    print(f" Total Valid Pairs Found  : {total_valid}")
    print(f" Total Pairs Copied       : {copied}")
    print(f" Subset Size (requested)  : {num_samples}")
    print(f" Output Image Dir         : {dst_img_dir}")
    print(f" Output Label Dir         : {dst_lbl_dir}")
    print("=" * 70 + "\n")


# ================================================================
# STEP 2 — Subset Creation Runner
# ================================================================
def run_subset_creation():
    """
    Automatically create smaller YOLO-format subsets (train & val)
    from the fully converted YOLO dataset for faster training.
    """
    print("\n STEP 2: Creating YOLO subset...\n")

    # Phase2 root
    PHASE2 = ROOT_DIR

    # Dataset root (Data/assignment_data_bdd inside repo root)
    DATA_ROOT = (PHASE2 / ".." / "Data" / "assignment_data_bdd").resolve()
    print("DATA_ROOT =", DATA_ROOT)

    # Full YOLO output produced by conversion (inside Phase2)
    OUTPUT_YOLO = PHASE2 / "bdd100k_yolo"

    # Subset output (inside Phase2)
    OUTPUT_SUBSET = PHASE2 / "bdd100k_yolo_subset"

    # Image root (official BDD100K layout inside Data)
    img_root = DATA_ROOT / "bdd100k_images_100k" / "bdd100k" / "images" / "100k"

    # Label root (converted labels)
    lbl_root = OUTPUT_YOLO / "labels"

    # sanity prints
    print(f" Images root: {img_root}")
    print(f" Labels root: {lbl_root}")
    print(f" Subset output: {OUTPUT_SUBSET}\n")

    # Create train and val subsets
    create_yolo_subset(
        images_root=img_root,
        labels_root=lbl_root,
        output_root=OUTPUT_SUBSET,
        split="train",
        num_samples=CONFIG["dataset"]["sample_size_train"],
        seed=CONFIG.get("seed", 42)
    )

    create_yolo_subset(
        images_root=img_root,
        labels_root=lbl_root,
        output_root=OUTPUT_SUBSET,
        split="val",
        num_samples=CONFIG["dataset"]["sample_size_val"],
        seed=CONFIG.get("seed", 42)
    )

    print("Subset creation complete.")
    print(f"Subset stored at: {OUTPUT_SUBSET}")
    print("=" * 70 + "\n")


# ============================================================
# STEP 4 — Train YOLOv8m with Custom DataLoader
# ============================================================






def bbox_iou(box1, box2, eps=1e-7):
    """IoU for boxes in xywh normalized format."""
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) + \
            ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) - inter
    return inter / (union + eps)


def manual_yolo_loss(preds, targets, num_classes=CONFIG["dataset"]["num_classes"], device="cuda"):
    """
    Manual YOLO loss computation matching the current model configuration.
    Includes diagnostic prints for prediction/label dimensional consistency.
    """
    import torch
    import torch.nn.functional as F

    # Handle list/tuple output from YOLO (multi-scale)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]


    B, C, H, W = preds.shape
    reg_max = 16  # default YOLOv8 distribution focal loss setting
    head_dim = reg_max * 4 + num_classes



    # Validate channel divisibility
    assert C % head_dim == 0, (
        f"Cannot reshape {C} channels into (reg_max*4 + num_classes)={head_dim}-dim head. "
        f"Got num_classes={num_classes} → check Detect head rebuild."
    )

    # Reshape predictions
    A = C // head_dim  # anchors per grid cell
    preds = preds.view(B, head_dim, A, H * W)

    pred_distri = preds[:, :reg_max * 4, :, :]
    pred_cls = preds[:, reg_max * 4:, :, :]

    # Mean over anchors and spatial dims
    pred_cls_avg = pred_cls.mean(dim=2).mean(dim=2)  # [B, num_classes]

    gt_cls = torch.zeros_like(pred_cls_avg)

    for lbl in targets:
        # ensure lbl is at least 2D
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(0)

        if lbl.shape[0] == 0:
            continue

        for row in lbl:
            # Make sure it's 1D (in case of a scalar tensor)
            if row.dim() == 0:
                row = row.unsqueeze(0)

            # Each row should be [img_idx, cls, x, y, w, h]
            if row.numel() < 2:
                continue  # malformed entry, skip

            cls_id = int(row[1].item())  # class ID is at index 1
            if 0 <= cls_id < num_classes:
                gt_cls[:, cls_id] = 1.0

 

    # --- Compute simplified YOLO-like losses ---
    box_loss = pred_distri.abs().mean() * 0.01
    cls_loss = F.binary_cross_entropy_with_logits(pred_cls_avg, gt_cls)

    dfl_loss = (pred_distri ** 2).mean() * 0.001

    total_loss = box_loss + cls_loss + dfl_loss
    return total_loss, (box_loss, cls_loss, dfl_loss)




from ultralytics import YOLO
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy, time, os
from pathlib import Path

def train_with_custom_dataloader():
    print("\n STEP 4: Training YOLOv8m using Custom DataLoader...\n")

    # ============================================
    # 1️ Paths & Config
    # ============================================
       # dataset_root expected relative in config; resolve against ROOT_DIR
    dataset_root = (ROOT_DIR / Path(CONFIG["paths"]["subset_root"])).resolve()
   
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure checkpoints/logs paths are resolved and created
    checkpoints_dir = (ROOT_DIR / Path(CONFIG["paths"]["checkpoints"])).resolve()
    logs_dir = (ROOT_DIR / Path(CONFIG["paths"]["logs"])).resolve()
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    writer = SummaryWriter(logs_dir.as_posix())

    num_epochs = CONFIG["training"]["epochs"]
    base_lr = CONFIG["training"]["lr"]
    accum_steps = CONFIG["training"]["accum_steps"]
    grad_clip_norm = CONFIG["training"]["grad_clip_norm"]
    ema_decay = CONFIG["training"]["ema_decay"]

    # ============================================
    # 2️ Data Loaders
    # ============================================
    train_loader = build_dataloader(
        dataset_root=dataset_root,
        split="train",
        batch_size=CONFIG["training"]["batch_size"],
        img_size=CONFIG["training"]["img_size"],
        normalization="zscore",
        augment=True,
        augment_classes=[
            "bike", "bus", "car", "motor", "person",
            "rider", "traffic light", "traffic sign", "train", "truck"
        ],
        num_workers=2
    )

    val_loader = build_dataloader(
        dataset_root=dataset_root,
        split="val",
        batch_size=8,
        img_size=640,
        normalization="zscore",
        augment=False,
        num_workers=2
    )

    print(" Custom DataLoaders initialized.\n")

    # ============================================
    # 3️ Model, Optimizer, Scheduler
    # ============================================
    base_model = YOLO("yolov8m.pt")
    wrapped = YOLOWithAdapter(base_model, num_new_classes=10).to(device)

    print("*************************************************")
    print("Adapter-based model created. Training only adapter layer.\n")

    for p in wrapped.base.parameters():
        p.requires_grad = False

    optimizer = AdamW(wrapped.adapter.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    ema_model = copy.deepcopy(wrapped)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ============================================
    # 4️ Training Loop
    # ============================================
    print(f" Starting Custom Training ({num_epochs} epochs)...\n")
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n======================== Epoch {epoch + 1}/{num_epochs} ========================")
        epoch_loss, epoch_box, epoch_cls, epoch_dfl = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        wrapped.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)

        for batch_idx, (imgs, labels) in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if global_step < 100:
                lr_scale = min(1.0, (global_step + 1) / 100)
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * lr_scale

            # Prepare labels
            batch_targets = []
            for i, label in enumerate(labels):
                if label.numel() == 0:
                    continue
                img_idx = torch.full((label.size(0), 1), i, dtype=torch.float32)
                formatted = torch.cat((img_idx, label), dim=1)
                batch_targets.append(formatted)
            if not batch_targets:
                continue
            batch_labels = torch.cat(batch_targets, dim=0).to(device)

            # Forward + Loss
            with torch.amp.autocast('cuda', enabled=True):
                preds = wrapped(imgs)
                loss, (box_loss, cls_loss, dfl_loss) = manual_yolo_loss(preds, batch_labels)

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped.adapter.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # EMA
            # EMA update — only on adapter weights
            with torch.no_grad():
                for ema_param, param in zip(ema_model.adapter.parameters(), wrapped.adapter.parameters()):
                    if ema_param.data.shape == param.data.shape:
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)


            epoch_loss += loss.item()
            epoch_box += box_loss.item()
            epoch_cls += cls_loss.item()
            epoch_dfl += dfl_loss.item()
            global_step += 1

            # Progress bar & logging
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = epoch_loss / (batch_idx + 1)
            writer.add_scalar("train/total_loss", loss.item(), global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

            progress_bar.set_description_str(
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Box: {box_loss.item():.4f} | Cls: {cls_loss.item():.4f} | DFL: {dfl_loss.item():.4f}"
            )

        # ================= End of Epoch =================
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Avg Total Loss: {epoch_loss / len(train_loader):.6f}")
        print(f"   Box Loss:       {epoch_box / len(train_loader):.6f}")
        print(f"   Cls Loss:       {epoch_cls / len(train_loader):.6f}")
        print(f"   DFL Loss:       {epoch_dfl / len(train_loader):.6f}")
        print(f"    Duration:       {epoch_time:.2f}s\n")

        # ================= Validation =================
        print(f" Running validation after epoch {epoch+1}...")
        # =============================================
# Create YOLO Dataset YAML for Validation
# =============================================
        dataset_yaml_path = checkpoints_dir / "bdd100k_dataset.yaml"
        dataset_yaml_content = {
            "path": str((ROOT_DIR / Path(CONFIG['paths']['subset_root'])).resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(CONFIG["dataset"]["class_names"]),
            "names": CONFIG["dataset"]["class_names"]
        }

        import yaml
        with open(dataset_yaml_path, "w") as f:
            yaml.dump(dataset_yaml_content, f)

        # =============================================
        # Run Validation
        # =============================================
        val_results = base_model.val(
            data=str(dataset_yaml_path),
            imgsz=640,
            batch=8,
            device=device,
            split="val",
            save_json=False,
            conf=0.25,
            iou=0.5,
            verbose=True,
        )

       

        print("\n Validation Results:")
        print(f"  mAP@0.5:      {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"  Precision:    {val_results.box.mp:.4f}")
        print(f"  Recall:       {val_results.box.mr:.4f}\n")

        writer.add_scalar("val/mAP50", val_results.box.map50, epoch)
        writer.add_scalar("val/mAP5095", val_results.box.map, epoch)
        writer.add_scalar("val/precision", val_results.box.mp, epoch)
        writer.add_scalar("val/recall", val_results.box.mr, epoch)

    # ============================================
    # 5 Save Final Model
    # ============================================
    final_path = checkpoints_dir / "yolov8m_adapter_final.pt"

    torch.save(wrapped.state_dict(), final_path)
    print(f"\n Final model saved → {final_path}")

    writer.close()
    print(" Training complete. Logs stored in TensorBoard path:", CONFIG["paths"]["logs"])

# ============================================================
# MAIN PIPELINE
# ============================================================

import json
from pathlib import Path
from tqdm import tqdm
import argparse

# =======================================================
# Entry Point
# =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BDD100K JSON annotations into YOLO format with full logging."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config.yaml",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=CONFIG["paths"]["data_root"],
        help="Path to directory containing BDD100K JSON label files "
             "(bdd100k_labels_images_train.json and bdd100k_labels_images_val.json)."
    )

    parser.add_argument(
        "--images-root",
        type=str,
        default=CONFIG["paths"]["images_root"],
        help="Path to directory containing original BDD100K images (train/val/test folders)."
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=CONFIG["paths"]["output_root"],
        help="Path to store YOLOv8m-formatted labels and subsets."
    )

    args = parser.parse_args()

    # Convert to Path objects
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    print("\n Dataset Source Layout:")
    print(f"  → JSON Labels Path : {data_root}")
    print(f"  → Images Root Path : {args.images_root}")
    print(f"  → Output Directory : {output_root}\n")

    # Update JSON paths manually (since we're inside labels folder)
    train_json = data_root / "bdd100k_labels_images_train.json"
    val_json = data_root / "bdd100k_labels_images_val.json"

    # Output label directories (YOLOv8m alias)
    train_output = output_root / "labels" / "train"
    val_output = output_root / "labels" / "val"

    if train_json.exists():
        convert_bdd_to_yolo(
            train_json, train_output, split="train",
            sample_size=CONFIG["dataset"]["sample_size_train"]
        )
    else:
        print(f"[WARN] Train JSON not found at {train_json}")

    if val_json.exists():
        convert_bdd_to_yolo(
            val_json, val_output, split="val",
            sample_size=CONFIG["dataset"]["sample_size_val"]
        )
    else:
        print(f"[WARN] Val JSON not found at {val_json}")

    print("\n Conversion completed successfully!")
    print(f"YOLOv8m labels saved under: {output_root}/labels/")
    print("\nNext steps:")
    print("  1️ Verify YOLO label integrity.")

        # ======================================
    # Step 2: Verify Label Integrity
    # ======================================
    print("\n Verifying YOLO label integrity...")

    train_images_dir = Path(args.images_root) / "train"
    val_images_dir = Path(args.images_root) / "val"

    verify_yolo_labels(train_images_dir, train_output)
    verify_yolo_labels(val_images_dir, val_output)

    print("\n YOLOv8m conversion and verification complete!")
    print(f"Final YOLO dataset ready at: {output_root}")

    print("  2️ Run training using your custom DataLoader or YOLOv8m config.")
    # ======================================
    # Step 3: Create Subset
    # ======================================
    run_subset_creation()

    print("\n YOLOv8m conversion, verification, and subset creation complete!")
    print(f"Subset ready at: {(ROOT_DIR / Path(CONFIG['paths']['subset_root'])).resolve()}")

    print("\nNext steps:")
    train_with_custom_dataloader()

    print("  → Train YOLOv8m using your custom DataLoader or config.")