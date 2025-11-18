
---

# YOLOv8 Training Script (Built-In Ultralytics DataLoader)

This script provides a clean, configuration-driven training workflow for YOLOv8 models on the BDD100K dataset, using Ultralytics’ built-in DataLoader instead of a custom dataloader.

The script performs:

* Loading hyperparameters from a YAML config file
* Automatic creation of a YOLO-compatible dataset YAML
* YOLOv8 training using Ultralytics’ `.train()` API
* Optional validation using YOLOv8’s `.val()` API

It is intended for the "Phase 1 / Phase 2" part of your repository.

---

# Key Features

### 1. Configuration-Driven Workflow

All training settings are read from a YAML config, including:

* Dataset paths
* Class names and number of classes
* Model weights
* Learning rate, batch size, image size, epochs
* Hardware/Device selection (CPU/GPU)

### 2. Automatic Dataset YAML Generation

The script generates a YOLOv8 dataset YAML file automatically:

```
path: <dataset_root>
train: images/train
val: images/val
nc: <num_classes>
names: <class_names>
```

### 3. Built-In YOLOv8 Training Pipeline

Uses:

* Pretrained weights (default: yolov8m.pt)
* AdamW optimizer
* Automatic checkpoints
* Built-in validation and plots
* Built-in logging

### 4. Hardware Auto-Selection

If CUDA is requested but unavailable, the script safely falls back to CPU.

### 5. Validation Pipeline

In addition to training, the script supports:

```
--validate-only
--model-path <custom.pt>
```

for running standalone validation.

---

# File Structure

```
scripts/
│   train_yolo_builtin.py
configs/
│   bdd100k_subset.yaml
dataset_root/
│   images/
│       train/
│       val/
│   labels/
│       train/
│       val/
```

---

# Configuration Example

Below is a minimal configuration file expected by the script:

```yaml
paths:
  subset_root: ./bdd100k_subset
  output_root: ./runs_yolo

dataset:
  num_classes: 10
  class_names: ["bike","bus","car","motor","person","rider","traffic light","traffic sign","train","truck"]

training:
  epochs: 50
  batch_size: 8
  img_size: 640
  lr: 0.001
  weights: yolov8m.pt
  num_workers: 4

hardware:
  device: cuda
```

---

# Running the Training Script

## Train YOLOv8

```
python scripts/train_yolo_builtin.py --config ./configs/bdd100k_subset.yaml
```

## Validate Only

```
python scripts/train_yolo_builtin.py \
    --config ./configs/bdd100k_subset.yaml \
    --validate-only \
    --model-path ./runs_yolo/yolov8m_builtin/weights/best.pt
```

---

# Metrics Explained

This section defines every metric computed by YOLOv8 (built-in) and the extended metrics from your Phase 3 evaluation pipeline.
You can reuse this block across Phase 2 and Phase 3 READMEs.

---

# Classification and Detection Metrics

### Precision

Precision = TP / (TP + FP)
Represents "how many predicted detections were correct."
High precision means few false positives.

### Recall

Recall = TP / (TP + FN)
Represents "how many actual objects were successfully detected."
High recall means few missed objects.

### F-Beta Score (β=2)

Weighted combination of Precision and Recall that emphasizes Recall:

```
Fβ = (1 + β²) * (P * R) / (β² * P + R)
```

With β=2, recall is weighted 4× more than precision.
Useful for safety-critical applications where missing an object is worse than a false detection.

---

# IoU-Based Metrics

### IoU (Intersection over Union)

Measures how well predicted boxes overlap with ground truth:

```
IoU = Area(Intersection) / Area(Union)
```

### mAP@0.5

Mean Average Precision at IoU threshold of 0.50.
Detection is considered correct if IoU ≥ 0.50.

### mAP@0.5:0.95

Average mAP over 10 IoU thresholds (0.50 to 0.95).
This is the COCO standard and is harder to optimize.

### Custom AP at IoUThreshold (mAP@0.40, mAP@0.75)

Your script also computes:

* AP@0.40 (more lenient)
* AP@0.75 (more strict)

These give a clearer understanding of model behavior at different IoU demands.

### Mean IoU

Average IoU over all matched predictions.
Measures bounding box alignment quality.

### IoU Variance

Variance of IoU values:

* Low variance = stable detection performance.
* High variance = inconsistent bounding box quality.

### IoU Stability per Class

Standard deviation of IoU values per class.
Shows which classes are less consistent (e.g., bus vs. traffic light).

---

# Object-Scale Metrics

BDD100K and COCO define object sizes using bounding-box area.

### AP_S (Small Objects)

Average Precision on objects with area < 32×32 px.
Measures small object detection ability (e.g., traffic lights).

### AP_M (Medium Objects)

Average Precision on objects with area between 32×32 and 96×96.

### AP_L (Large Objects)

Average Precision on objects ≥ 96×96.
Usually highest AP due to easier visibility.

### Average Recall (AR)

Average recall across IoU thresholds and object sizes.
Captures how completely the model finds objects.

---

# Localization Regression Metrics

These metrics quantify how accurate bounding boxes are positioned.

### L1 Error (Manhattan Distance)

L1 = Σ |pred_i − gt_i|
Measures absolute differences between predicted and GT box coordinates.

### L2 Error (Euclidean Distance)

L2 = sqrt(Σ (pred_i − gt_i)²)
More sensitive to large deviations than L1.

Low L1/L2 errors indicate accurate localization.

---

# PR-Based Detection Metrics

### True Positives (TP)

Predicted box correctly matched to ground truth with IoU ≥ threshold.

### False Positives (FP)

Predictions that do not match any ground truth.

### False Negatives (FN)

Ground truth boxes that were not detected.

Your script saves TP/FP/FN in JSON as part of the PR metrics.

---

# Model Efficiency Metrics

These metrics measure compute cost and runtime performance.

### Latency (ms)

Time taken for one forward pass of the model.

### FPS (Frames per Second)

Number of images the model can process per second.

```
FPS = 1000 / latency(ms)
```

### FLOPs (GFLOPs)

Floating Point Operations required for one inference pass.
Measured using the THOP library.

### Parameter Count (Millions)

Number of trainable weights in the model.

### Model Size (MB)

Size of the weight file (.pt) on disk.

### GPU Memory Usage (MB)

VRAM required during inference.

These are essential metrics when deploying on embedded devices like Bosch ECUs, automotive chips, TPUs, Jetson, etc.

---

# Internal Architecture (High-Level)

```
train_yolo_builtin.py
│
├── load_config()              # loads YAML config
├── create_dataset_yaml()      # auto-generates YOLO dataset YAML
├── train_yolo_builtin()       # runs model.train()
├── validate_yolo_model()      # runs model.val()
└── __main__                   # argument parser
```

---

# Summary

This training script provides a clean, extensible training pipeline for YOLOv8, supporting:

* Fully YAML-driven configuration
* Automated dataset preparation
* Built-in Ultralytics training and validation
* GPU/CPU auto-handling
* Easy extensibility for your Phase 3 evaluation framework

