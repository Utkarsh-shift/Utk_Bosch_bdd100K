"""
==============================================================
YOLOv8 Training Script (Using Built-in DataLoader)
==============================================================

Author: Utkarsh
Description:
    Trains YOLOv8 (e.g., yolov8m) directly on the YOLO-formatted
    BDD100K subset without a custom DataLoader.

    This script:
    - Reads training paths and hyperparameters from YAML config
    - Creates YOLO-compatible dataset YAML
    - Calls Ultralytics' built-in .train() pipeline

Usage:
    python scripts/train_yolo_builtin.py --config ./configs/bdd100k_subset.yaml
==============================================================
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# 1. Load Config
# ============================================================
def load_config(config_path: str):
    """Load YAML configuration file."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f" Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f" Loaded configuration from {cfg_path}")
    return cfg


# ============================================================
# 2. Create YOLO Dataset YAML
# ============================================================
def create_dataset_yaml(cfg, save_path="bdd100k_dataset.yaml"):
    """Create a YOLO-compatible dataset YAML file."""
    dataset_yaml = {
        "path": str(cfg["paths"]["subset_root"]),
        "train": "images/train",
        "val": "images/val",
        "nc": cfg["dataset"]["num_classes"],
        "names": cfg["dataset"]["class_names"]
    }

    save_path = Path(save_path)
    with open(save_path, "w") as f:
        yaml.dump(dataset_yaml, f)
    print(f" Created YOLO dataset YAML at: {save_path}")
    return save_path


# ============================================================
# 3. Train YOLOv8 (Built-in)
# ============================================================
def train_yolo_builtin(cfg):
    """Train YOLOv8 using Ultralytics’ built-in DataLoader."""

    # Prepare dataset YAML for YOLO
    dataset_yaml = create_dataset_yaml(cfg)

    # Load pretrained model (default: yolov8m)
    weights = cfg["training"].get("weights", "yolov8m.pt")
    model = YOLO(weights)

    # Select device
    requested_device = cfg["hardware"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("\n[WARNING] CUDA requested but NOT available. Falling back to CPU.\n")
        device = "cpu"
    else:
        device = requested_device

    # Create project/checkpoint dirs
    project_dir = Path(cfg["paths"]["output_root"])
    project_dir.mkdir(parents=True, exist_ok=True)

    print("\n Starting YOLOv8 training (built-in mode)...\n")
    print("===================================================")
    print(f"Model Weights   : {weights}")
    print(f"Device          : {device}")
    print(f"Train Dataset   : {cfg['paths']['subset_root']}/images/train")
    print(f"Val Dataset     : {cfg['paths']['subset_root']}/images/val")
    print(f"Epochs          : {cfg['training']['epochs']}")
    print(f"Batch Size      : {cfg['training']['batch_size']}")
    print(f"Image Size      : {cfg['training']['img_size']}")
    print(f"Learning Rate   : {cfg['training']['lr']}")
    print("===================================================\n")

    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        imgsz=cfg["training"]["img_size"],
        lr0=cfg["training"]["lr"],
        device=device,
        optimizer="AdamW",
        project=str(project_dir),
        name="yolov8m_builtin",
        exist_ok=True,
        pretrained=True,
        val=True,
        save_period=5,
        workers=cfg["training"].get("num_workers", 2),
        plots=True,
        verbose=True,
    )

    print("\n Training complete!")
    print(f"Results saved to: {project_dir}/yolov8m_builtin/")
    return results


# ============================================================
# 4. Validate YOLOv8 Model (Standalone)
# ============================================================
def validate_yolo_model(cfg, model_path=None):
    """
    Validate a trained YOLOv8 model or a pretrained checkpoint.

    Args:
        cfg (dict): Configuration dictionary loaded from YAML.
        model_path (str): Path to model weights (.pt). If None, uses cfg's training weights.
    """
    dataset_yaml = create_dataset_yaml(cfg)
    model_path = model_path or cfg["training"].get("weights", "yolov8m.pt")

    print("\n Starting YOLOv8 Validation...\n")
    print("===================================================")
    print(f"Model Path   : {model_path}")
    print(f"Dataset YAML : {dataset_yaml}")
    print(f"Validation Set: {cfg['paths']['subset_root']}/images/val")
    print("===================================================\n")

    model = YOLO(model_path)
    requested_device = cfg["hardware"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("\n[WARNING] CUDA requested but NOT available. Falling back to CPU.\n")
        device = "cpu"
    else:
     device = requested_device

    results = model.val(
        data=str(dataset_yaml),
        imgsz=cfg["training"]["img_size"],
        batch=cfg["training"]["batch_size"],
        device=device,
        split="val",
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.6,
        half=True,
        verbose=True,
    )

    # Print summary metrics
    print("\n Validation Summary:")
    print(f"   mAP@0.5:       {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95:  {results.box.map:.4f}")
    print(f"   Precision:     {results.box.mp:.4f}")
    print(f"   Recall:        {results.box.mr:.4f}")
    print("===================================================\n")

    print(" Validation complete. Results and plots saved.")
    return results


# ============================================================
# 5. Main Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate YOLOv8 using built-in DataLoader")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--validate-only", action="store_true", help="Run only validation on given model weights")
    parser.add_argument("--model-path", type=str, default=None, help="Path to custom model weights for validation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.validate_only:
        validate_yolo_model(cfg, args.model_path)
    else:
        results = train_yolo_builtin(cfg)

        # Run validation on the trained model
        validate_yolo_model(cfg, args.model_path)
