"""
BDD100K Evaluation Pipeline for YOLOv8

Author: Utkarsh | Bosch Assignment - Phase 3
Date: November 2025

Features:
- Loads BDD100K JSON labels and converts to YOLO format
- Runs YOLOv8 evaluation automatically
- Exports logs, plots, and CSV metrics
"""
import cv2
import cv2
import numpy as np
import random
import shutil
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import yaml

import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
import time
from thop import profile




import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
width = 1280
height = 720
def add_gaussian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_random_occlusion(img, num_patches=3):
    h, w, _ = img.shape
    for _ in range(num_patches):
        x1 = random.randint(0, w - 50)
        y1 = random.randint(0, h - 50)
        x2 = min(w, x1 + random.randint(50, 200))
        y2 = min(h, y1 + random.randint(50, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return img
class BBoxMatcher:
    """
    Match predicted and ground-truth bounding boxes using IoU or Hungarian algorithm.
    Computes L1 and L2 regression errors for matched pairs.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(boxA, boxB):
        """Compute IoU between two boxes (x1, y1, x2, y2)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter_area / (boxA_area + boxB_area - inter_area + 1e-6)

    def match_boxes(self, preds, gts):
        """
        Match predicted and GT boxes using Hungarian algorithm on IoU matrix.
        Args:
            preds (ndarray): Predicted boxes [N, 4]
            gts (ndarray): Ground-truth boxes [M, 4]
        Returns:
            matches: List of (pred_idx, gt_idx, iou)
        """
        if len(preds) == 0 or len(gts) == 0:
            return []

        # Build IoU matrix
        iou_matrix = np.zeros((len(preds), len(gts)))
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                iou_matrix[i, j] = self.compute_iou(p, g)

        # Hungarian matching (maximize IoU)
        pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
        matches = [
            (pi, gi, iou_matrix[pi, gi])
            for pi, gi in zip(pred_idx, gt_idx)
            if iou_matrix[pi, gi] >= self.iou_threshold
        ]
        return matches

    def compute_bbox_errors(self, preds, gts):
        """Compute L1 and L2 distance for matched boxes."""
        matches = self.match_boxes(preds, gts)
        if not matches:
            return {"L1": None, "L2": None, "mean_iou": 0.0}

        l1_list, l2_list, iou_list = [], [], []
        for pi, gi, iou in matches:
            diff = np.abs(preds[pi] - gts[gi])
            l1_list.append(np.sum(diff))
            l2_list.append(np.sqrt(np.sum(diff**2)))
            iou_list.append(iou)

        return {
            "L1": np.mean(l1_list),
            "L2": np.mean(l2_list),
            "mean_iou": np.mean(iou_list),
        }




class BDD100KConverter:
    """
    Converts BDD100K JSON annotations into YOLOv8 .txt label format.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        output_label_dir: str,
        class_list: Optional[List[str]] = None,
        image_width: int = 1280,
        image_height: int = 720,
    ):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.output_label_dir = Path(output_label_dir)
        self.image_width = image_width
        self.image_height = image_height
        self.class_list = (
            class_list
            if class_list
            else [
                "person",
                "rider",
                "car",
                "bus",
                "truck",
                "traffic light",
                "traffic sign",
                "train",
                "motor",
                "bicycle",
            ]
        )
        self.class_to_id = {c: i for i, c in enumerate(self.class_list)}

    def convert(self) -> int:
        """
        Converts BDD JSON annotations into YOLOv8 labels.

        Returns:
            int: number of files converted
        """
        self.output_label_dir.mkdir(parents=True, exist_ok=True)
        print(f" Converting BDD100K JSON → YOLO format...")
        with open(self.json_path, "r") as f:
            data = json.load(f)

        count = 0
        for item in data:
            img_name = item["name"]
            labels = []
            for obj in item.get("labels", []):
                if "box2d" not in obj:
                    continue
                category = obj.get("category")
                if category not in self.class_to_id:
                    continue

                cid = self.class_to_id[category]
                x1, y1, x2, y2 = (
                    obj["box2d"]["x1"],
                    obj["box2d"]["y1"],
                    obj["box2d"]["x2"],
                    obj["box2d"]["y2"],
                )

                # Normalize to YOLO format
                xc = (x1 + x2) / 2 / self.image_width
                yc = (y1 + y2) / 2 / self.image_height
                w = (x2 - x1) / self.image_width
                h = (y2 - y1) / self.image_height

                labels.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if labels:
                label_path = self.output_label_dir / (Path(img_name).stem + ".txt")
                with open(label_path, "w") as f:
                    f.writelines(labels)
                count += 1

        print(f" Converted {count} JSON annotations to YOLO labels.")
        return count

import numpy as np
import json
import os
from shapely.geometry import box
from scipy.optimize import linear_sum_assignment
from pathlib import Path

class BDD100KEvaluator:
    """
    Handles evaluation of YOLOv8 models on BDD100K dataset.
    """

    def __init__(
        self,
        model_path: str,
        image_dir: str,
        label_dir: str,
        output_dir: str = "bdd_eval_results",
        data_yaml_path: str = "bdd_eval_data.yaml",
    ):
        self.model_path = Path(model_path)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.data_yaml_path = Path(data_yaml_path)
        self.class_names = [
            "person",
            "rider",
            "car",
            "bus",
            "truck",
            "traffic light",
            "traffic sign",
            "train",
            "motor",
            "bicycle",
        ]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._create_data_yaml()




        
    def compute_precision_recall_from_json(self) -> dict:
        """
        Compute Precision, Recall, and F-beta using predictions.json and YOLO ground truth labels.
        """
        print(" Computing Precision–Recall from predictions.json and GT labels...")

        # Locate predictions.json
        run_folders = sorted(self.output_dir.glob("val_run_*"), key=os.path.getmtime)
        if not run_folders:
            raise FileNotFoundError(" No val_run_* folder found in output directory.")
        latest_run = run_folders[-1]
        pred_json = latest_run / "predictions.json"
        if not pred_json.exists():
            raise FileNotFoundError(f" predictions.json not found in {latest_run}")

        print(f" Found predictions.json: {pred_json}")

        with open(pred_json, "r") as f:
            preds = json.load(f)

        # Group predictions by image_id
        pred_by_img = {}
        for p in preds:
            img_id = Path(p["file_name"]).stem
            bbox = p["bbox"]  # [x, y, w, h] in pixels
            x1, y1, w, h = bbox
            pred_box = [x1, y1, x1 + w, y1 + h]
            pred_by_img.setdefault(img_id, []).append(pred_box)

        TP, FP, FN = 0, 0, 0
        iou_threshold = 0.5
        img = cv2.imread(str(list(Path(self.image_dir).glob("*.jpg"))[0]))
        height, width = img.shape[:2]
 # BDD100K standard

        # Iterate over GT labels
        label_files = list((self.label_dir).glob("*.txt"))

        if not label_files:
             raise FileNotFoundError(f" No GT labels found in {self.label_dir}")

        for label_path in label_files:
            img_id = label_path.stem
            if img_id not in pred_by_img:
                gt_boxes = np.loadtxt(label_path).reshape(-1, 5)[:, 1:]
                FN += len(gt_boxes)
                continue

            gt_boxes = np.loadtxt(label_path).reshape(-1, 5)[:, 1:]
            # Convert YOLO normalized (0–1) → pixel coordinates
            gt_boxes_px = gt_boxes.copy()
            gt_boxes_px[:, [0, 2]] *= width
            gt_boxes_px[:, [1, 3]] *= height

            gt_boxes_xyxy = np.column_stack((
                gt_boxes_px[:, 0] - gt_boxes_px[:, 2] / 2,
                gt_boxes_px[:, 1] - gt_boxes_px[:, 3] / 2,
                gt_boxes_px[:, 0] + gt_boxes_px[:, 2] / 2,
                gt_boxes_px[:, 1] + gt_boxes_px[:, 3] / 2
            ))

            preds = np.array(pred_by_img[img_id])
            matched_gt = np.zeros(len(gt_boxes_xyxy), dtype=bool)

            for pb in preds:
                ious = []
                for gb in gt_boxes_xyxy:
                    inter_x1 = max(pb[0], gb[0])
                    inter_y1 = max(pb[1], gb[1])
                    inter_x2 = min(pb[2], gb[2])
                    inter_y2 = min(pb[3], gb[3])
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    iou = inter_area / (area_p + area_g - inter_area + 1e-6)
                    ious.append(iou)

                if ious and max(ious) >= iou_threshold:
                    max_idx = np.argmax(ious)
                    if not matched_gt[max_idx]:
                        TP += 1
                        matched_gt[max_idx] = True
                    else:
                        FP += 1
                else:
                    FP += 1

            FN += np.sum(~matched_gt)

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        beta = 2
        fbeta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall + 1e-8)

        print(f"\n Precision–Recall Summary:")
        print(f"  TP={TP}, FP={FP}, FN={FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F-beta (β=2): {fbeta:.4f}")

        return {
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "F-beta (β=2)": round(float(fbeta), 4),
            "TP": TP,
            "FP": FP,
            "FN": FN
        }




    def _create_data_yaml(self):
        """Create YOLOv8-compatible dataset YAML."""

        # dataset root = .../bdd100k
        dataset_root = str(Path(self.image_dir).parents[2])

        data_yaml = {
            "path": dataset_root,
            "train": "images/100k/train",
            "val":   "images/100k/val",
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        with open(self.data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        print(f"Created data.yaml at {self.data_yaml_path}")




    def compute_efficiency_metrics(self, sample_img_path: Optional[str] = None) -> Dict:
        """
        Compute computational efficiency metrics for the model.

        Metrics:
            - FPS (frames per second)
            - Latency per image (ms)
            - FLOPs (GFLOPs)
            - Model size (MB)
            - GPU memory footprint (MB, optional)
        """
        print(" Computing efficiency metrics...")

        # Load model
        model = YOLO(str(self.model_path))
        torch_model = model.model.eval()

        # Pick a sample image
        if sample_img_path is None:
            image_files = list(self.image_dir.rglob("*.jpg"))
            if not image_files:
                raise FileNotFoundError(f"No images found in {self.image_dir}")
            sample_img_path = str(image_files[0])

        # Read image
        img = cv2.imread(sample_img_path)
        if img is None:
            raise ValueError(f"Could not read sample image: {sample_img_path}")
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Measure latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        _ = torch_model(img_tensor.to(next(torch_model.parameters()).device))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency = (time.time() - start_time) * 1000  # ms
        fps = 1000.0 / latency

        # Compute FLOPs
        flops, params = profile(torch_model, inputs=(img_tensor.to(next(torch_model.parameters()).device),), verbose=False)
        gflops = flops / 1e9

        # Model size
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        # Optional: GPU memory
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            gpu_memory_mb = None

        metrics = {
            "FPS": round(fps, 2),
            "Latency (ms)": round(latency, 2),
            "FLOPs (GFLOPs)": round(gflops, 2),
            "Parameters (M)": round(params / 1e6, 2),
            "Model Size (MB)": round(model_size_mb, 2),
            "GPU Memory (MB)": round(gpu_memory_mb, 2) if gpu_memory_mb else "N/A",
        }

        print("\n Efficiency Metrics Summary:")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v}")

        return metrics

    

    # ----------------------------------------------------------------
    #  Localization Metrics
    # ----------------------------------------------------------------






    def compute_localization_metrics(self, results) -> dict:
        """
        Compute localization-related metrics:
            - Bounding Box Regression Error (L1, L2)
            - Mean IoU
            - IoU Variance
            - IoU Stability per Class
        Uses predictions.json and GT YOLO labels.
        """
        print(" Computing Localization Metrics...")
        metrics = {}

        # 1️ Locate latest predictions.json
        run_folders = sorted(self.output_dir.glob("*/"), key=os.path.getmtime)

        if not run_folders:
            raise FileNotFoundError(" No val_run_* folder found in output directory.")
        latest_run = run_folders[-1]
        pred_json = latest_run / "predictions.json"
        if not pred_json.exists():
            raise FileNotFoundError(f" predictions.json not found in {latest_run}")
        print(f" Using predictions.json: {pred_json}")

        with open(pred_json, "r") as f:
            preds = json.load(f)

        # 2️ Group predictions by image and class
        pred_by_img = {}
        for p in preds:
            img_id = Path(p["file_name"]).stem
            cat = p["category_id"]
            bbox = p["bbox"]  # [x, y, w, h]
            x1, y1, w, h = bbox
            pred_box = [x1, y1, x1 + w, y1 + h]
            pred_by_img.setdefault(img_id, []).append((cat, pred_box))

        img = cv2.imread(str(list(Path(self.image_dir).glob("*.jpg"))[0]))
        height, width = img.shape[:2]
 # image size
        iou_values = []
        l1_diffs, l2_diffs = [], []

        iou_by_class = {cls_id: [] for cls_id in range(len(self.class_names))}

        # 3️ Loop over GT labels
        label_files = list(Path(self.label_dir).glob("*.txt"))

        if not label_files:
            raise FileNotFoundError(f" No GT labels found in {self.label_dir}")

        for label_path in label_files:
            img_id = label_path.stem
            if img_id not in pred_by_img:
                continue

            gt_data = np.loadtxt(label_path).reshape(-1, 5)
            gt_cls = gt_data[:, 0].astype(int)
            gt_boxes = gt_data[:, 1:]

            # Convert YOLO normalized → pixel coordinates
            gt_boxes_px = gt_boxes.copy()
            gt_boxes_px[:, [0, 2]] *= width
            gt_boxes_px[:, [1, 3]] *= height

            gt_boxes_xyxy = np.column_stack((
                gt_boxes_px[:, 0] - gt_boxes_px[:, 2] / 2,
                gt_boxes_px[:, 1] - gt_boxes_px[:, 3] / 2,
                gt_boxes_px[:, 0] + gt_boxes_px[:, 2] / 2,
                gt_boxes_px[:, 1] + gt_boxes_px[:, 3] / 2
            ))

            preds = pred_by_img[img_id]
            pred_boxes = np.array([p[1] for p in preds])
            pred_classes = np.array([p[0] for p in preds])

            # Build IoU matrix
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes_xyxy)))
            for i, pb in enumerate(pred_boxes):
                for j, gb in enumerate(gt_boxes_xyxy):
                    xA = max(pb[0], gb[0])
                    yA = max(pb[1], gb[1])
                    xB = min(pb[2], gb[2])
                    yB = min(pb[3], gb[3])
                    inter_area = max(0, xB - xA) * max(0, yB - yA)
                    area_pb = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    area_gb = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    union = area_pb + area_gb - inter_area + 1e-6
                    iou_matrix[i, j] = inter_area / union

            # Match boxes with Hungarian assignment
            from scipy.optimize import linear_sum_assignment
            pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)

            for pi, gi in zip(pred_idx, gt_idx):
                iou = iou_matrix[pi, gi]
                iou_values.append(iou)

                diff = np.abs(np.array(pred_boxes[pi]) - np.array(gt_boxes_xyxy[gi]))
                l1_diffs.append(np.sum(diff))
                l2_diffs.append(np.sqrt(np.sum(diff**2)))

                cls_id = gt_cls[gi]
                if cls_id in iou_by_class:
                    iou_by_class[cls_id].append(iou)

        # 4️ Aggregate Metrics
        if len(l1_diffs):
            metrics["BBox Regression L1 Error"] = round(float(np.mean(l1_diffs)), 4)
            metrics["BBox Regression L2 Error"] = round(float(np.mean(l2_diffs)), 4)
        else:
            metrics["BBox Regression L1 Error"] = "N/A"
            metrics["BBox Regression L2 Error"] = "N/A"

        if len(iou_values):
            metrics["Mean IoU"] = round(float(np.mean(iou_values)), 4)
            metrics["IoU Variance"] = round(float(np.var(iou_values)), 4)
        else:
            metrics["Mean IoU"] = 0.0
            metrics["IoU Variance"] = 0.0

        # Compute IoU stability (std per class)
        stability = {}
        for cid, vals in iou_by_class.items():
            if len(vals):
                stability[self.class_names[cid]] = round(float(np.std(vals)), 4)
        metrics["IoU Stability per Class"] = stability

        # 5️ Print Summary
        print("\n Localization Metrics Summary:")
        for k, v in metrics.items():
            print(f"  {k:<35}: {v}")

        return metrics




    def compute_additional_metrics(self, results) -> dict:
        """
        Compute additional evaluation metrics:
            - Average Recall (AR)
            - AP_S, AP_M, AP_L (object-scale based)
            - F-beta score (β=2)
            - IoU Variance
        """
        print(" Computing additional metrics...")
        metrics = {}

        # ------------------------------------------------------------
        # 1️ Average Recall (AR)
        # ------------------------------------------------------------
        maps = getattr(results, "maps", None)
        if maps is not None and len(maps):
            metrics["Average Recall (AR)"] = float(np.mean(maps))
        else:
            raise ValueError(" 'results.maps' missing or empty — cannot compute AR.")

        # ------------------------------------------------------------
        # 2️ AP_S, AP_M, AP_L (based on object area)
        # ------------------------------------------------------------
        json_path = Path(results.save_dir) / "predictions.json"
        if not json_path.exists():
            raise FileNotFoundError(f" predictions.json not found in {results.save_dir}")

        with open(json_path, "r") as f:
            preds = json.load(f)

        if not isinstance(preds, list) or len(preds) == 0:
            raise ValueError(" predictions.json is empty or not a list.")

        areas = np.array([p["bbox"][2] * p["bbox"][3] for p in preds])
        scores = np.array([p["score"] for p in preds])

        # Define area-based group masks
        small_mask = areas < 32**2
        medium_mask = (areas >= 32**2) & (areas < 96**2)
        large_mask = areas >= 96**2

        from sklearn.metrics import precision_recall_curve, auc

        def compute_ap(mask, label):
            if not np.any(mask):
                print(f" No objects found in {label} group.")
                return None
            y_true = np.ones(mask.sum())
            y_scores = scores[mask]
            if len(np.unique(y_scores)) <= 1:
                print(f" Degenerate scores in {label} group (all same).")
                return None
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            return float(auc(recall, precision))

        metrics["AP_S (Small)"] = compute_ap(small_mask, "Small")
        metrics["AP_M (Medium)"] = compute_ap(medium_mask, "Medium")
        metrics["AP_L (Large)"] = compute_ap(large_mask, "Large")

        # ------------------------------------------------------------
        # 3️ F-beta Score (β=2)
        # ------------------------------------------------------------
        beta = 2.0
        precision = getattr(results, "mp", None)
        recall = getattr(results, "mr", None)

        if precision is None or recall is None:
            print(" YOLO results missing precision/recall — computing manually...")

            #  Get precision and recall using predictions + GT
            pr_metrics = self.compute_precision_recall_from_json()
            precision = pr_metrics["Precision"]
            recall = pr_metrics["Recall"]

        if precision is None or recall is None:
            raise ValueError(" Precision or Recall could not be computed.")

        fbeta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall + 1e-8)
        metrics["F-beta (β=2)"] = round(float(fbeta), 4)

        # ------------------------------------------------------------
        # 4️ IoU Variance
        # ------------------------------------------------------------
        if hasattr(results, "ious"):
            iou_values = np.array(results.ious)
        elif hasattr(results, "metrics") and isinstance(results.metrics, dict) and "iou" in results.metrics:
            iou_values = np.array(results.metrics["iou"])
        else:
            # fallback: simulate from AP scores
            iou_values = np.clip(scores / scores.max(), 0, 1)
            print(" Using normalized scores as proxy for IoU values.")

        metrics["IoU Variance"] = round(float(np.var(iou_values)), 4)

        # ------------------------------------------------------------
        #  Print Summary
        # -----------------------------------------------------------
        metrics["mAP40"] = round(float(self.compute_custom_mAP(results, 0.40)), 4)
        metrics["mAP75"] = round(float(results.box.map75), 4)  # YOLOv8 builtin

        print("\n Additional Metrics Summary:")
        for k, v in metrics.items():
            print(f"  {k:<25}: {v}")

        return metrics


    def compute_custom_mAP(self, results, iou_threshold: float):
        """
        Compute AP at a custom IoU threshold (e.g., 0.40) using precision-recall curves.

        YOLOv8 returns predictions.json with per-detection scores.
        We compute custom mAP using IoU-based matching.
        """
        pred_json = Path(results.save_dir) / "predictions.json"
        if not pred_json.exists():
            raise FileNotFoundError(f"predictions.json missing: {pred_json}")

        with open(pred_json, "r") as f:
            preds = json.load(f)

        # Load GT labels
        label_files = list(Path(self.label_dir).glob("*.txt"))
        if not label_files:
            raise FileNotFoundError("No GT labels found.")

        iou_thr = iou_threshold

        TP, FP, total_gt = [], [], 0

        # Convert YOLO loader GT to dict
        gt_dict = {}
        for lbl in label_files:
            img_id = lbl.stem
            g = np.loadtxt(lbl).reshape(-1, 5)
            boxes = g[:, 1:]
            classes = g[:, 0].astype(int)
            gt_dict[img_id] = boxes
            total_gt += len(boxes)

        # Build prediction list (score, TP/FP)
        records = []

        img = cv2.imread(str(list(Path(self.image_dir).glob("*.jpg"))[0]))
        height, width = img.shape[:2]


        for p in preds:
            img_id = Path(p["file_name"]).stem
            if img_id not in gt_dict:
                records.append((p["score"], 0))
                continue

            gt_boxes = gt_dict[img_id]

            # Pred box in xyxy
            x1, y1, w, h = p["bbox"]
            pred_box = np.array([x1, y1, x1 + w, y1 + h])

            # GT boxes in pixel coords
            gt_b = gt_boxes.copy()
            gt_b[:, [0, 2]] *= width
            gt_b[:, [1, 3]] *= height

            gt_xyxy = np.column_stack([
                gt_b[:, 0] - gt_b[:, 2] / 2,
                gt_b[:, 1] - gt_b[:, 3] / 2,
                gt_b[:, 0] + gt_b[:, 2] / 2,
                gt_b[:, 1] + gt_b[:, 3] / 2,
            ])

            # IoU with every GT
            ious = []
            for gt in gt_xyxy:
                inter_x1 = max(pred_box[0], gt[0])
                inter_y1 = max(pred_box[1], gt[1])
                inter_x2 = min(pred_box[2], gt[2])
                inter_y2 = min(pred_box[3], gt[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                area_p = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                area_g = (gt[2] - gt[0]) * (gt[3] - gt[1])
                iou = inter_area / (area_p + area_g - inter_area + 1e-6)
                ious.append(iou)

            if ious and max(ious) >= iou_thr:
                records.append((p["score"], 1))
            else:
                records.append((p["score"], 0))

        # Sort by score
        records.sort(key=lambda x: -x[0])

        tps = np.array([r[1] for r in records])
        fps = 1 - tps

        TP_cum = np.cumsum(tps)
        FP_cum = np.cumsum(fps)

        precision = TP_cum / (TP_cum + FP_cum + 1e-9)
        recall = TP_cum / (total_gt + 1e-9)

        # AP = area under PR curve
        ap = np.trapz(precision, recall)
        return ap



    

    # def create_robustness_dataset(self, corruption_type: str, severity: float = 1.0) -> Path:
    #     """
    #     Create a YOLO-compatible corrupted validation dataset:
    #     - Adds Gaussian noise or occlusion to images
    #     - Copies labels
    #     - Ensures proper /images and /labels folder structure
    #     """
    #     print(f" Generating corrupted validation dataset ({corruption_type}, severity={severity})...")
    #     corrupted_root = self.output_dir / f"val_corrupted_{corruption_type}"
    #     img_dir = corrupted_root / "images"
    #     lbl_dir = corrupted_root / "labels"

    #     img_dir.mkdir(parents=True, exist_ok=True)
    #     lbl_dir.mkdir(parents=True, exist_ok=True)

    #     img_paths = list(self.image_dir.glob("*.jpg"))
    #     if not img_paths:
    #         raise FileNotFoundError(f"No images found in {self.image_dir}")

    #     for img_path in img_paths:
    #         img = cv2.imread(str(img_path))
    #         if img is None:
    #             continue

    #         # Apply corruption
    #         if corruption_type == "noise":
    #             img_corrupt = add_gaussian_noise(img, std=15 * severity)
    #         elif corruption_type == "occlusion":
    #             img_corrupt = add_random_occlusion(img, num_patches=int(3 * severity))
    #         else:
    #             raise ValueError(" Unknown corruption type. Use 'noise' or 'occlusion'.")

    #         # Save corrupted image
    #         out_img_path = img_dir / img_path.name
    #         cv2.imwrite(str(out_img_path), img_corrupt)

    #         # Copy corresponding label if exists
    #         label_path = Path(self.label_dir) / f"{img_path.stem}.txt"
    #         if label_path.exists():
    #             shutil.copy(str(label_path), str(lbl_dir / label_path.name))

    #     print(f" Corrupted dataset saved at: {corrupted_root}")
    #     return corrupted_root


    # def compute_robustness_metrics(self, corruption_type="noise", severity=1.0) -> dict:
    #     """
    #     Re-run YOLOv8 model on corrupted validation dataset to compute robustness metrics.
    #     """
    #     print(f"\n Evaluating robustness under '{corruption_type}' (severity={severity})...")

    #     # 1️ Generate corrupted dataset (with /images + /labels)
    #     corrupted_root = self.create_robustness_dataset(corruption_type, severity)
    #     corrupted_images_dir = corrupted_root / "images"

    #     # 2️ Create temporary YOLO dataset YAML
    #     temp_yaml = self.output_dir / f"robustness_{corruption_type}.yaml"
    #     data_yaml = {
    #         "path": "/workspace",  # Root directory for YOLO
    #         "train": str(self.image_dir),  # Keep same training path
    #         "val": str(corrupted_images_dir),  #  Corrupted validation images path
    #         "names": {i: name for i, name in enumerate(self.class_names)},
    #     }

    #     with open(temp_yaml, "w", encoding="utf-8") as f:
    #         yaml.dump(data_yaml, f)
    #     print(f" Created temporary robustness YAML: {temp_yaml}")
    #     print(f"   → Validation path: {corrupted_images_dir}")

    #     # 3️ Run YOLO evaluation
    #     model = YOLO(str(self.model_path))
    #     results = model.val(
    #         data=str(temp_yaml),
    #         split="val",
    #         imgsz=640,
    #         save_json=False,
    #         verbose=False,
    #         project=str(self.output_dir),
    #         name=f"robustness_{corruption_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #         exist_ok=True,
    #     )

    #     # 4️ Extract key metrics
    #     metrics = {}
    #     if hasattr(results, "metrics") and isinstance(results.metrics, dict):
    #         metrics["mAP50"] = float(results.metrics.get("mAP50", 0.0))
    #         metrics["mAP50-95"] = float(results.metrics.get("mAP50-95", 0.0))
    #     if hasattr(results, "mp"):
    #         metrics["Precision"] = float(results.mp)
    #     if hasattr(results, "mr"):
    #         metrics["Recall"] = float(results.mr)

    #     metrics["corruption_type"] = corruption_type
    #     metrics["severity"] = severity

    #     print(f" Robustness metrics ({corruption_type}) → {metrics}")
    #     return metrics












    def evaluate(self):
        """Run YOLOv8 evaluation on the dataset and compute all metrics."""
        print(" Running YOLOv8 Evaluation...")

        # ------------------------------------------------------------
        # 1️ Run YOLOv8 Validation
        # ------------------------------------------------------------
        model = YOLO(str(self.model_path))
        results = model.val(
            data=str(self.data_yaml_path),
            split="val",
            imgsz=640,
            save_json=True,
            verbose=True,
            save_txt=True,
            project=str(self.output_dir),
            name=f"val_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True,
        )

        print(f" Evaluation complete — Results saved at: {results.save_dir}")

        # ------------------------------------------------------------
        # 2️ Compute Efficiency Metrics
        # ------------------------------------------------------------
        print("\n Computing Efficiency Metrics...")
        eff_metrics = self.compute_efficiency_metrics()
        eff_path = self.output_dir / "efficiency_metrics.json"
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(eff_metrics, f, indent=2)
        print(f" Efficiency metrics saved → {eff_path}")

        # ------------------------------------------------------------
        # 3️ Compute Additional Metrics (AR, APs, APm, APl, F-beta, IoU Var)
        # ------------------------------------------------------------
        print("\n Computing Additional Metrics...")
        extra_metrics = self.compute_additional_metrics(results)
        extra_path = self.output_dir / "additional_metrics.json"
        with open(extra_path, "w", encoding="utf-8") as f:
            json.dump(extra_metrics, f, indent=2)
        print(f" Additional metrics saved → {extra_path}")

        # ------------------------------------------------------------
        # 4️ Compute Localization Metrics (L1, L2, IoU Stats)
        # ------------------------------------------------------------
        print("\n Computing Localization Metrics...")
        localization_metrics = self.compute_localization_metrics(results)
        loc_path = self.output_dir / "localization_metrics.json"
        with open(loc_path, "w", encoding="utf-8") as f:
            json.dump(localization_metrics, f, indent=2)
        print(f" Localization metrics saved → {loc_path}")

        # ------------------------------------------------------------
        # 5️ Compute Precision–Recall (GT vs Predictions)
        # ------------------------------------------------------------
        print("\n Computing Precision–Recall Metrics...")
        pr_metrics = self.compute_precision_recall_from_json()

        # Convert NumPy int64/float64 to native Python types for JSON serialization
        pr_metrics_clean = {
            k: (float(v) if isinstance(v, np.floating)
                else int(v) if isinstance(v, np.integer)
                else v)
            for k, v in pr_metrics.items()
        }

        pr_metrics_path = self.output_dir / "precision_recall_metrics.json"
        with open(pr_metrics_path, "w", encoding="utf-8") as f:
            json.dump(pr_metrics_clean, f, indent=2)
        print(f" Precision–Recall metrics saved → {pr_metrics_path}")

        # ------------------------------------------------------------
        # 6️ Robustness Evaluation (Noise & Occlusion)
        # ------------------------------------------------------------
        # print("\n Evaluating Robustness (Noise & Occlusion)...")
        # noise_metrics = self.compute_robustness_metrics("noise", severity=1.0)
        # occlusion_metrics = self.compute_robustness_metrics("occlusion", severity=1.0)

        # robustness_metrics = {
        #     "noise_performance": noise_metrics,
        #     "occlusion_performance": occlusion_metrics,
        # }
        # robustness_path = self.output_dir / "robustness_metrics.json"
        # with open(robustness_path, "w", encoding="utf-8") as f:
        #     json.dump(robustness_metrics, f, indent=2)
        # print(f" Robustness metrics saved → {robustness_path}")

        # ------------------------------------------------------------
        # 7️ Merge All Metrics Together (Unified Final Report)
        # ------------------------------------------------------------
        print("\n Compiling Final Evaluation Report...")
        all_metrics = {
            "Efficiency Metrics": eff_metrics,
            "Additional Metrics": extra_metrics,
            "Localization Metrics": localization_metrics,
            "Precision–Recall Metrics": pr_metrics_clean,
            # "Robustness Metrics": robustness_metrics,
        }

        all_metrics_path = self.output_dir / "all_metrics.json"
        with open(all_metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n All metrics saved → {all_metrics_path}")

        # Nicely formatted console summary
        print("\n Final Evaluation Summary:")
        for section, data in all_metrics.items():
            print(f"\n {section}")
            for k, v in data.items():
                print(f"  {k:<35}: {v}")

        return results



    


def main():
    """Main entrypoint for BDD100K evaluation pipeline."""

    ROOT = Path(__file__).resolve().parent.parent   # → Utk_Bosch_bdd100K

    DATA_ROOT = ROOT / "Data" / "assignment_data_bdd"

    # JSON labels
    json_path = DATA_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    # Image directory
    image_dir = DATA_ROOT / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "val"

    # YOLO ground-truth labels generated in Phase2_Model
    label_dir = ROOT / "Phase3_eval_viz" / "gt_labels"


    # Model
    model_path = ROOT / "Phase2_Model" / "yolov8m.pt"

    # Step 1 — Convert JSON → YOLO labels (overwrite existing labels)
    converter = BDD100KConverter(
        json_path=json_path,
        image_dir=image_dir,
        output_label_dir=label_dir
    )
    converter.convert()

    # Step 2 — Evaluate YOLOv8 model
    evaluator = BDD100KEvaluator(
        model_path=str(model_path),
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        output_dir = ROOT / "Phase3_eval_viz" / "evaluation" / "results",
        data_yaml_path = ROOT / "Phase3_eval_viz" / "bdd_eval_data.yaml"

    )
    evaluator.evaluate()

    

if __name__ == "__main__":
    main()
