# **Utk_Bosch_bdd100K**

A complete, end-to-end workflow built on the **BDD100K Object Detection Dataset (ODD)**, demonstrating full ML lifecycle expertise — from **data exploration** to **model training**, **evaluation**, **visualization**, and **containerized deployment**.

This repository is divided into **three well-structured phases**, each with its own environment, documentation, and reproducible workflow.

---

# ** Repository Structure**

```
Utk_Bosch_bdd100K/
│
├── Phase1_DataAnalysis/        # Data ingestion, cleaning, EDA, dashboard UI
├── Phase2_Model/               # YOLOv8 training, configs, logging
├── Phase3_eval_viz/            # Evaluation, advanced metrics, visualization
│
├── Data/                       # (Git-ignored) BDD100K dataset
├── checkpoints/                # Model checkpoints (tracked via Git LFS)
├── logs/                       # Training logs
└── README.md                   # Root documentation
```

---

# ** Phase Documentation**

Use the links below to navigate to detailed guides for each phase:

| Phase       | Documentation                                                    |
| ----------- | ---------------------------------------------------------------- |
| **Phase 1** |  **[Data Analysis & Insights](Phase1_DataAnalysis/README.md)** |
| **Phase 2** |  **[YOLOv8 Training](Phase2_Model/README.md)**                 |
| **Phase 3** |  **[Evaluation & Visualization](Phase3_eval_viz/README.md)**   |

---

# ** Project Overview**

This repository replicates a real-world computer vision workflow, including:

### ** Phase 1 — Data Analysis & Visualization (Django + Streamlit)**

* Dataset ingestion & cleaning
* Object distribution & density heatmaps
* Bounding-box size analysis
* Occlusion & diversity sampling
* REST API + caching layer
* Streamlit dashboard

➡ Full guide at: `Phase1_DataAnalysis/README.md`

---

### * Phase 2 — Model Training (YOLOv8)**

Supports **two complete pipelines**:

#### 1️ Built-in YOLO Training

* `train_yolo_withoutcustomdataloader.py`
* Config-driven (YAML)
* Automatic logs, plots, checkpoints

#### 2️ Custom PyTorch DataLoader Training

* `train_customdataloader_yolo.py`
* Robust subset loader
* Hard-coded consistency checks

Also includes:

* JSON → YOLO label conversion
* Auto subset creation
* Training monitoring & visualization

➡ Full guide at: `Phase2_Model/README.md`

---

### ** Phase 3 — Evaluation & Visualization**

A full evaluation framework that computes and saves:

#### **Core YOLO Metrics**

✔ mAP50
✔ mAP50–95
✔ Precision / Recall / F1
✔ Confusion Matrix
✔ PR / F1 / P / R curves

#### **Advanced Metrics**

✔ **Average Recall (AR)**
✔ **AP_S / AP_M / AP_L** (scale-based AP)
✔ **Custom mAP@0.40**
✔ **F-beta (β = 2)**
✔ **IoU Variance**

#### **Localization Metrics**

✔ Box-based Hungarian matching
✔ L1 / L2 regression error
✔ IoU stability per class

#### **Efficiency Metrics**

✔ FPS
✔ Latency
✔ FLOPs
✔ Model size
✔ Parameter count

➡ Full guide at: `Phase3_eval_viz/README.md`

---

# ** Docker Support (Full Reproducibility)**

Each phase contains its own Dockerfile + docker-compose.

### Run Everything Reproducibly:

```
docker-compose up --build
```

Advantages:

* Identical environment on any machine
* No dependency conflicts
* Easy distribution for evaluation

---

# ** Tech Stack**

### **Computer Vision & ML**

* YOLOv8 (Ultralytics)
* PyTorch
* THOP (FLOPs)
* SciPy (Hungarian matching)
* OpenCV
* NumPy, Pandas

### **Backend**

* Django 5.x
* Django REST Framework
* Redis + Celery (optional caching)

### **Frontend**

* Streamlit
* Tailwind-like UI

### **DevOps**

* Docker & Docker Compose
* Git LFS (for `.pt`, `.jpg`, `.mp4`)

---

# **🛠 Installation & Setup**

### 1 Clone the Repo

```
git clone https://github.com/Utkarsh-shift/Utk_Bosch_bdd100K.git
cd Utk_Bosch_bdd100K
```

### 2 Install Git LFS

```
git lfs install
git lfs pull
```

### 3️ Run Phase Workflows

| Phase       | Command                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| **Phase 1** | `cd Phase1_DataAnalysis && docker-compose up --build`                                                  |
| **Phase 2** | `cd Phase2_Model && python scripts/train_yolo_withoutcustomdataloader.py --config scripts/config.yaml` |
| **Phase 3** | `cd Phase3_eval_viz && python bdd_evaluator.py`                                                        |

---

# ** Results & Reports**

All evaluation assets are exported to:

```
Phase3_eval_viz/evaluation/results/
```

Includes:

* `predictions.json`
* PR, F1, Precision, Recall curves
* confusion matrices
* `efficiency_metrics.json`
* `localization_metrics.json`
* `additional_metrics.json`
* **all_metrics.json (final unified report)**

---

# ** License**

Designed for educational and assessment purposes only.
Dataset follows BDD100K license.

---

# ** Author**

**Utkarsh Suryaman**
AI Researcher | Computer Vision | Deep Learning
IIT Ropar | Thapar | Bosch Assessment

