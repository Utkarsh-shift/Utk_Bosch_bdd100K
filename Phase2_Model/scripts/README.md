
---

# **YOLOv8 Training Script (Built-In DataLoader Mode)**

### *Train YOLOv8 using Ultralytics’ Native Pipeline on BDD100K YOLO-Formatted Dataset*

This repository provides a **simple, reliable, and production-ready training pipeline** for YOLOv8 using Ultralytics’ built-in DataLoader (instead of a custom PyTorch loader).

The script:

* Loads training configurations from a YAML file
* Automatically generates a **YOLO dataset YAML**
* Trains a YOLOv8 model using Ultralytics API
* Supports GPU/CPU selection
* Saves checkpoints, logs, plots, and metrics
* Provides a complete validation pipeline

---

#  **Key Features**

### **1. Config-Driven Pipeline**

All parameters are loaded from a supplied YAML config:

✔ Dataset paths
✔ Image size, batch size, epochs
✔ Learning rate
✔ Model weights
✔ Device selection (CPU/GPU)
✔ Class names & class count

---

### **2. Automatic Dataset YAML Builder**

The script auto-generates a YOLO-compatible dataset YAML:

```yaml
path: ./bdd100k_subset
train: images/train
val: images/val
nc: 10
names: ["car","bus",...]
```

No manual creation required.

---

### **3. Built-In YOLOv8 Training**

Uses:

```python
model = YOLO("yolov8m.pt")
model.train(...)
```

Including:

* Optimizer: **AdamW**
* Checkpoint saving every **5 epochs**
* Validation during training
* Automatic logging & plots

---

### **4. Device Auto-Selection**

If config requests `"cuda"` but GPU is unavailable, the script automatically falls back to CPU with a warning.

---

### **5. Structured Output Directory**

All training logs, checkpoints, and plots are saved under:

```
<output_root>/yolov8m_builtin/
```

---

### **6. Full Validation Pipeline**

Also supports standalone validation:

✔ mAP@0.5
✔ mAP@0.5:0.95
✔ Precision
✔ Recall
✔ JSON output for COCO evaluation
✔ Plots enabled

---

#  **Project Structure**

```
.
├── scripts/
│   └── train_yolo_builtin.py      # Main training script
├── configs/
│   └── bdd100k_subset.yaml        # User-defined config
├── bdd100k_subset/
│   ├── images/train/              # YOLO images
│   ├── images/val/
│   ├── labels/train/              # YOLO labels
│   └── labels/val/
└── README.md
```

---

#  **Configuration File Example (`bdd100k_subset.yaml`)**

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

# **Usage**

## **1. Train YOLOv8**

```
python scripts/train_yolo_builtin.py --config ./configs/bdd100k_subset.yaml
```

Runs:

* Dataset YAML generation
* Model initialization
* Training
* Validation after training

---

## **2. Validation Only**

```
python scripts/train_yolo_builtin.py \
    --config ./configs/bdd100k_subset.yaml \
    --validate-only \
    --model-path ./runs_yolo/yolov8m_builtin/weights/best.pt
```

---

# **Training Logs Printed to Console**

* Model weights
* Device
* Train/val image directories
* Epochs, batch size, learning rate
* Built-in YOLO metrics during training

---

#  **Validation Metrics**

Displayed after validation:

```
mAP@0.5:       0.8421
mAP@0.5:0.95:  0.5634
Precision:     0.8123
Recall:        0.7551
```

Plots and JSON results are saved automatically.

---

#  **Script Architecture Overview**

```
train_yolo_builtin.py
|
├── load_config()          # Load YAML config
├── create_dataset_yaml()  # Auto-generate YOLO dataset file
├── train_yolo_builtin()   # Main training function
├── validate_yolo_model()  # Validation function
└── __main__               # CLI entry point
```

---

#  **Requirements**

```
Python 3.8+
PyTorch
Ultralytics 8.x
PyYAML
```

Install Ultralytics:

```
pip install ultralytics
```

