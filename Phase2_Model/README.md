# Phase 3 — YOLOv8m Training Pipeline & Custom DataLoader

**Repository section:** `Phase3` (or root where `configs/`, `models/`, `runs/`, `scripts/` live)
**Author:** Utkarsh Suryaman — End-to-end YOLOv8m training for BDD100K (conversion → subset → custom dataloader → training → validation → visualization)

---

> Place this file as `Phase3/README.md` (or at repo root under `Phase3_Training_README.md`). It explains everything — file layout, what each script/module does, how to run every stage (conversion, verification, subset creation, training), configuration options, expected outputs, common errors and fixes, evaluation & visualization notes, and recommended next steps.

---

# Table of contents

1. [Overview](#overview)
2. [Directory tree & key files](#directory-tree--key-files)
3. [Design & high-level flow](#design--high-level-flow)
4. [Detailed description of important modules & scripts](#detailed-description-of-important-modules--scripts)
5. [Configuration (config.yaml) — expected keys & examples](#configuration-configyaml----expected-keys--examples)
6. [Runbook — commands for each stage](#runbook----commands-for-each-stage)
7. [Model training details & internals (what to watch for)](#model-training-details--internals-what-to-watch-for)
8. [Validation, evaluation and visualization outputs](#validation-evaluation-and-visualization-outputs)
9. [Performance tips & debugging common errors](#performance-tips--debugging-common-errors)
10. [Artifacts, checkpoints & reproducibility](#artifacts-checkpoints--reproducibility)
11. [Git / Large-file handling & recommended storage practices](#git--large-file-handling--recommended-storage-practices)
12. [Recommended improvements & next steps](#recommended-improvements--next-steps)
13. [Contact / credits](#contact--credits)

---

# Overview

Phase 3 implements a production-ready pipeline that:

* converts BDD100K JSON annotations → YOLO format (sampled subset option),
* verifies YOLO label integrity,
* creates train/val subsets containing only valid image–label pairs,
* provides a configurable, feature-rich `BDDYOLODataset` + `build_dataloader()` in `models/` (includes many normalization and augmentation methods),
* trains YOLOv8m using a small “adapter” head strategy (freeze base model, train adapter conv layer),
* supports a manual YOLO-style loss computation (diagnostic + simplified),
* validates using Ultralytics `YOLO.val()` and writes results/logs,
* saves checkpoints and TensorBoard logs under `runs/` / `logs/` / `models/`,
* contains utilities for visualization & sample generation.

This README documents how each piece fits together and how to run it.

---

# Directory tree & key files

Your provided tree (trimmed) — put in repo:

```
Phase3/
├── configs/                # config.yaml and possibly training/experiment configs
├── logs/                   # TensorBoard logs, training logs
│   └── yolo_training
├── models/                 # custom dataloader & augmentation code, model helpers
│   └── yolo_dataloader.py  # contains BDDYOLODataset and build_dataloader()
├── runs/                   # ultralytics/own run outputs (val visualizations, plots)
│   └── detect/
│       ├── val/
│       ├── val2/
│       └── val3/
├── scripts/                # main pipeline runner script (convert → subset → train)
│   └── train_pipeline.py   # (your long pipeline script)
├── configs/config.yaml     # central configuration file (paths, training params)
└── README.md               # <-- you will create (this file)
```

**Important internals you have:**

* `models/yolo_dataloader.py` — contains:

  * class/idx maps `CLASS_NAMES`, `CLASS_TO_IDX`
  * MANY normalization functions (minmax, zscore, per_channel, robust, dynamic_range, log, gamma, local contrast, adaptive)
  * AUGMENTATION_METHODS: photometric, geometric, noise/blur, cutout/mixup, domain-specific (fog/rain/shadow)
  * `BDDYOLODataset` with `_apply_augmentations`, `__getitem__` returns `(img_tensor, labels)`
  * `yolo_collate_fn` and `build_dataloader()`
  * Self-test block that generates sample visualizations and measures DataLoader throughput
* `scripts/*` — contains full end-to-end pipeline:

  * `load_config()` reads `scripts/config.yaml`
  * `convert_bdd_to_yolo()` → converts BDD JSON to YOLO `.txt` files (with sampling)
  * `verify_yolo_labels()` → checks label format, missing labels, invalid entries
  * `create_yolo_subset()` → sample valid pairs and copy to `bdd100k_yolo_subset` structure
  * `run_subset_creation()` orchestrator
  * `YOLOWithAdapter`, `rebuild_detect_head()` to adapt YOLOv8 Detect head / adapter conv
  * `manual_yolo_loss()` as a diagnostic/simple loss
  * `train_with_custom_dataloader()` — training loop using Ultralytics trunk model with adapter layer, AMP, EMA, scheduler, writer, and per epoch validation via `base_model.val()`
  * CLI `if __name__ == "__main__":` to run conversion, verification, subset creation and then `train_with_custom_dataloader()`

---

# Design & high-level flow

1. **Convert** raw BDD100K annotation JSONs → YOLO text labels (per-image `.txt` with `cls x_center y_center w h` normalized). You can sample size (e.g., 2k images) for faster iterations.
2. **Verify** each converted label for format sanity (class ids, normalized coordinates, missing/extra label files).
3. **Create subset**: copy matched image/label pairs into `bdd100k_yolo_subset/images/{train,val}` and `bdd100k_yolo_subset/labels/{train,val}`.
4. **Train**:

   * Use `build_dataloader()` to create DataLoader(s) with augmentation, normalization strategies and collate function.
   * Load Ultralytics YOLOv8 base (`YOLO("yolov8m.pt")`) and wrap with `YOLOWithAdapter` (adapter conv layer).
   * Freeze base weights; train only adapter (fine-tuning minimal capacity).
   * Use manual loss wrapper `manual_yolo_loss` for diagnostics (and you call `base_model.val()` for official metrics).
   * Save checkpoints and TensorBoard metrics.
5. **Validate & Visualize**:

   * The training loop builds a `bdd100k_dataset.yaml` for Ultralytics val, runs `base_model.val(...)` and logs mAP/precision/recall.
   * Ultralytics `val` writes results that can be rendered in `runs/detect/...` folders — use existing visualization modules to extract PR curves, confusion matrices, JSON predictions.

---

# Detailed description of important modules & scripts

Below are the important code pieces and what they do — use this to maintain or extend them.

## `models/yolo_dataloader.py` (key points)

### Class mapping

* `CLASS_NAMES` and `CLASS_TO_IDX` are the canonical mapping. Keep consistent with `CONFIG["dataset"]["class_names"]`.

### Normalization functions

* Many normalization methods are implemented. They accept PyTorch `img_tensor` in CHW float format (0..1 or 0..255 depending on caller). Example functions:

  * `normalize_minmax`, `normalize_zscore`, `normalize_per_channel`
  * `normalize_robust` (median + IQR)
  * `normalize_dynamic_range` (contrast stretch)
  * `normalize_local_contrast` (local mean/std using `avg_pool2d`)
  * `normalize_adaptive` uses `class_quality_stats` to scale brightness/contrast per class.
* `NORMALIZATION_METHODS` maps names → callables used by `BDDYOLODataset`.

**Important**: many functions assume `img_tensor` is `torch.tensor` in 0..1 range. Confirm input scaling before applying.

### Augmentations

* Photometric: brightness/contrast, color balance, gamma.
* Geometric: flip, rotation, crop, affine, perspective.
* Noise & blur: gaussian noise, motion blur (via PIL GaussianBlur), JPEG artifacts.
* Cutout/mixup/erasing: implemented for tensor-level or PIL-level as appropriate.
* Domain augmentations: `simulate_fog`, `simulate_rain`, `simulate_shadow`.

**Pipeline behavior**:

* `_apply_augmentations()` defines `AUGMENTATION_PIPELINE` + optional `PER_CLASS_AUGS` to apply class-specific augmentations.
* Final tensor augmentations (noise, erasing) applied after PIL→tensor conversion.

### `BDDYOLODataset`

* `__init__`: expects `images_dir` and `labels_dir` with `.jpg` and `.txt` files.
* `__getitem__`: loads image, reads labels (YOLO-style lines), applies augmentations and normalization, returns `(img_tensor, labels)` where `labels` is `(N,5)` float tensor or empty `(0,5)`.
* Uses `random.seed(idx)` to make per-index augment deterministic across epochs if needed.
* Prints helpful initialization info.

### Collate & builder

* `yolo_collate_fn(batch)` stacks images into `(B,C,H,W)` and returns targets as list of tensors. This is required because number of boxes vary per image.
* `build_dataloader()` returns a `DataLoader` configured with `pin_memory=True`, collate_fn, `shuffle=True` for train.

### Self-test block

* When running file directly, it builds a small dataloader, samples images, runs combinations of normalization × augmentations, saves images under `bdd100k_yolo_subset_samples/` and measures throughput.

---

## `scripts/train_pipeline.py` (or your big script)

### `load_config(config_path)`

* Loads YAML config. Script caches `CONFIG`. All other paths are resolved relative to `ROOT_DIR` (project root) so the script is portable.

### Conversion: `convert_bdd_to_yolo(json_path, output_dir, split, sample_size)`

* Reads BDD JSON list `labels_data` (each item a dict with `'name'` and `'labels'`).
* Samples `sample_size` items (seeded) and converts `box2d` coordinates to normalized YOLO format (x_center,y_center,w,h).
* Writes `.txt` label files into `output_dir`.
* Prints conversion summary (converted/skipped/avg boxes per image).

### Verify: `verify_yolo_labels(images_dir, labels_dir)`

* Validates `.txt` content with regex and numeric checks.
* Reports missing labels, extra labels, invalid entries and returns a summary dict.

### Subset creation: `create_yolo_subset(images_root, labels_root, output_root, split, num_samples)`

* Finds valid image-label pairs (image has label file), samples `num_samples`, copies pairs into `output_root/images/{split}` and `output_root/labels/{split}`.

### Adapter model + detect head helpers

* `YOLOWithAdapter(base_model, num_new_classes)` — wraps `base_model.model` and adds `self.adapter` conv to map channel dims to a shape compatible with the new head.
* `rebuild_detect_head(model, num_classes, class_names)` — finds `Detect` layer in Ultralytics model and rebuilds its `cv2`/`cv3` conv heads with new `no` (num outputs per anchor) matching `reg_max` assumption. Use this when adapting pre-trained weights to new `nc`.

**Caveat**: Rebuilding Detect head must match internal model architecture and channel counts — incorrect `in_channels` assumptions will raise shape errors. See troubleshooting below.

### Manual loss: `manual_yolo_loss(preds, targets, num_classes, device)`

* Flatten/reshape predicted channels into `[B, head_dim, A, H*W]` and splits into distribution (bbox regression) and classification parts. Creates a simplified `gt_cls` and computes BCE on averaged class logits + small DFL / box losses. This is primarily diagnostic and not guaranteed to match official YOLO training loss exactly.

### `train_with_custom_dataloader()`

* Loads train and val DataLoaders with `build_dataloader`.
* Loads Ultralytics base `YOLO("yolov8m.pt")`, wraps with `YOLOWithAdapter`, freezes base, trains adapter with `AdamW`.
* Uses `GradScaler` for AMP; uses Cosine LR scheduler; EMA update on adapter params.
* Writes TensorBoard scalars and progress bar.
* After each epoch, builds a `bdd100k_dataset.yaml` file and runs `base_model.val()` for official evaluation metrics (mAP, precision, recall). Logs metrics to TensorBoard.
* Saves final state dict `yolov8m_adapter_final.pt` into `checkpoints_dir`.

---

# Configuration (`scripts/config.yaml` expected keys)

Your script expects `scripts/config.yaml`. Typical keys used in script:

```yaml
paths:
  data_root: "Data/assignment_data_bdd"         # where BDD JSONs are
  images_root: "Data/bdd100k_images_100k/bdd100k/images/100k"
  output_root: "Phase2/outputs"                 # where conversion writes
  subset_root: "Phase2/bdd100k_yolo_subset"     # subset used for training
  checkpoints: "models/checkpoints"
  logs: "logs/yolo_training"

dataset:
  class_names: ["bike","bus","car","motor","person","rider","traffic light","traffic sign","train","truck"]
  num_classes: 10
  sample_size_train: 2000
  sample_size_val: 500
  # optionally other fields

training:
  epochs: 20
  batch_size: 8
  img_size: 640
  lr: 1e-4
  accum_steps: 1
  grad_clip_norm: 1.0
  ema_decay: 0.999

hardware:
  device: "cuda"   # or "cpu"

seed: 42
```

**Tips:**

* Keep `images_root` and `data_root` absolute or resolvable (script resolves relative to ROOT_DIR).
* `subset_root` must be consistent with the `build_dataloader()` call later.

---

# Runbook — commands for each stage

> Run these from the Phase3 project root (where `scripts/`, `models/`, `configs/` exist).

## 0. Prepare environment

```bash
# create venv (recommended)
python -m venv .venv
source .venv/bin/activate        # linux/mac
pip install -r requirements.txt
# requirements should include: torch, torchvision, ultralytics, tqdm, pyyaml, tensorboard, Pillow, numpy
```

## 1. Convert JSON → YOLO labels (sampled)

```bash
python scripts/train_pipeline.py --config scripts/config.yaml
# the script will run conversion + verification + subset creation by default before training
```

If you prefer to call functions separately, you can open an interactive shell or add CLI flags; the script's main block runs conversion if JSONs exist.

## 2. Verify labels manually (quick check)

```python
# from project root (python REPL)
from scripts.train_pipeline import verify_yolo_labels
verify_yolo_labels(Path("Data/bdd100k_images_100k/.../train"), Path("output_root/labels/train"))
```

## 3. Create subset (if you want to run only subset creation)

```bash
# inside scripts/train_pipeline.py there is run_subset_creation() — you can call it from a small runner
python -c "from scripts.train_pipeline import run_subset_creation; run_subset_creation()"
```

## 4. Train with custom dataloader (if you only want to train)

```bash
python -c "from scripts.train_pipeline import train_with_custom_dataloader; train_with_custom_dataloader()"
# or run the entire script as in step 1; it'll call train_with_custom_dataloader() at end
```

## 5. TensorBoard / logs

```bash
tensorboard --logdir logs/yolo_training --port 6006
# open http://localhost:6006
```

## 6. After training: validate / run ultralytics detect on saved model

* You can use the saved `yolov8m_adapter_final.pt`. If you saved only adapter weights, ensure you have code that loads base weights + adapter weights mapping.

---

# Model training details & internals (what to watch for)

## Adapter strategy

* You load pretrained `yolov8m.pt` and add `YOLOWithAdapter` (adapter conv). The script freezes base parameters and trains only adapter.
* Pros: far fewer params to update → faster convergence and reduced overfitting on small subsets.
* Cons: if adapter shape mismatch occurs or Detect head requires different channel shapes, you must either rebuild detect head correctly or fine-tune more layers.

## Rebuilding Detect head

* `rebuild_detect_head()` looks for `Detect` module in `model.model.modules()` and resets its `nc` and `no`.
* It rebuilds `cv2`/`cv3` conv lists; ensure `in_channels` inference is valid for your YOLO version — mismatches are the most common source of errors ("Cannot reshape C channels into head_dim").

## `manual_yolo_loss` caveats

* This is a simplified loss intended for diagnostics. It reshapes predicted channels by assuming `preds` shape and `reg_max` constant (16).
* If your model's internal reg_max or output layout differs, manual loss will assert or produce incorrect gradients. Use it for debugging, but rely on Ultralytics training loss for production.

## DataLoader & augmentations

* The `BDDYOLODataset` returns image tensors and variable-length label tensors. The collate function returns `images, targets` where targets is a tuple/list of label tensors per sample (this script formats them into concatenated batch targets before loss).
* Deterministic per-index seeding is used (`random.seed(idx)`) — this makes augment deterministic per index if desired.

---

# Validation, evaluation and visualization outputs

* Validation calls `base_model.val(data=dataset_yaml_path, imgsz=640, batch=8, ...)` and returns `val_results` with `val_results.box.map50`, `val_results.box.map` (mAP@.5:.95), `val_results.box.mp` (precision), `val_results.box.mr` (recall).
* Ultralytics' `val()` can optionally save predictions to JSON (set `save_json=True`) into `runs/detect/exp*/predictions.json` — heavy files (predict outputs) may exceed GitHub limits; store them offline if big.
* `runs/detect/...` will contain:

  * `confusion_matrix.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, etc.
  * `predictions.json` (contains per-image boxes + scores)
* You already have `Phase3_eval_viz/` in repo with these plots; use those to populate README examples.

---

# Performance tips & debugging common errors

### 1. `Cannot reshape C channels into (reg_max*4 + num_classes)` or assertion errors in `manual_yolo_loss`

* Cause: channel count `C` does not divide by `head_dim` (reg_max*4 + num_classes). This usually means:

  * The Detect head was not reconstructed properly to match new `nc`, or
  * The adapter conv produced unexpected channel dims, or
  * Ultralytics changed internal layout/reg_max.
* Fixes:

  * Print `preds.shape` right after forward, then compute `head_dim` expected and check `C % head_dim`.
  * Use `rebuild_detect_head()` to set `detect_layer.nc = num_classes` and recalc `detect_layer.no`. Then reload model or re-init detect layers.
  * Ensure `reg_max` used in `manual_yolo_loss` matches model's `detect_layer.reg_max`.

### 2. `RuntimeError: expected CPU tensor but got CUDA tensor` / device mismatch

* Ensure `batch_labels` and `preds` are both on same device. In training loop you do `batch_labels = batch_labels.to(device)`, ensure `wrapped` is also on `device` and images are `.to(device, non_blocking=True)`.

### 3. Slow DataLoader / low throughput

* Use `num_workers` > 0 (2–8 depending on CPU). On Windows, `num_workers` > 0 requires `if __name__ == "__main__":` guard.
* Use `pin_memory=True` and `prefetch_factor` in DataLoader.
* Cache expensive pre-processing or precompute heavy features if repeated.

### 4. Validation results unexpectedly low / training loss not decreasing

* Check label correctness: run `verify_yolo_labels()` — fix out-of-range coords, missing labels or label-image mismatches.
* Visualize a few training pairs to ensure augmentations did not destroy boxes/labels alignment.
* Try disabling aggressive augmentations (mixup, heavy crop) to verify base performance.

### 5. Ultralytics `val()` outputs vs manual loss mismatch

* `base_model.val()` uses internal loss & NMS/metrics; `manual_yolo_loss` is simplified. Use `val()` metrics as authoritative.

---

# Artifacts, checkpoints & reproducibility

* Checkpoints: saved to `checkpoints_dir` (configured in `config.yaml`) e.g. `checkpoints/yolov8m_adapter_final.pt`.
* TensorBoard logs under `logs/yolo_training`.
* `runs/detect` contains validation visualizations and saved predictions (large).
* To reproduce training exactly:

  * Pin `torch` and `ultralytics` versions in `requirements.txt`.
  * Pin random seeds: `torch.manual_seed(seed)`, `random.seed(seed)`, `np.random.seed(seed)` at the top of `train_with_custom_dataloader()`.

---

# Git / Large-file handling & recommended storage practices

* Avoid committing large files (>100MB) to GitHub (you earlier hit `predictions.json ~293MB`).
* Use **Git LFS** for binaries (model weights `.pt`, large `.mp4`, `.jpg` datasets). Install & configure:

```bash
# install git-lfs (system-specific) then:
git lfs install
git lfs track "*.pt"
git lfs track "*.mp4"
git lfs track "Phase3_eval_viz/**/predictions.json"   # if you must version predictions
git add .gitattributes
git commit -m "Enable LFS for large artifacts"
```

* Better: store heavy artifacts off-repo (S3, GDrive, internal artifact repo) and add download scripts / instructions in README to fetch them.

---

# Recommended improvements & next steps

1. **Modularize the big script**: split `scripts/train_pipeline.py` into modules: `convert.py`, `verify.py`, `subset.py`, `train.py` for clarity, unit tests and reusability.
2. **Integrate job queue for recompute/long ops**: use Celery/RQ for long preprocessing to avoid blocking.
3. **Make `manual_yolo_loss` consistent with Ultralytics** or rely on ultralytics training API for loss/optimizer integration — manual loss is mostly diagnostic.
4. **Add unit tests** for:

   * conversion numeric properties (box area > 0),
   * `verify_yolo_labels()` behaviours,
   * `build_dataloader()` shapes & throughput.
5. **Provide sample config presets** (`config_debug.yaml`, `config_full.yaml`) for small/fast experiments and full training.
6. **Add a small `docker-compose`** for reproducible training (GPU passthrough via `nvidia-docker` when available).
7. **Add REST endpoint** or a small CLI to export `runs/detect/*` into `Phase3_eval_viz/` for consistent plotting & archiving (avoid committing the big JSONs).

---

# Example trouble-shoot checklist (quick)

* Labels off / empty results:

  * run `python -c "from scripts.train_pipeline import verify_yolo_labels; verify_yolo_labels(Path('...images/train'), Path('...labels/train'))"`
* `C % head_dim != 0` assertion:

  * print `preds.shape` and `head_dim` inside `manual_yolo_loss` and confirm `reg_max`.
* GPU OOM:

  * reduce `batch_size` or `img_size` or use gradient accumulation (`accum_steps`) as provided.
* Validation slow / saving huge prediction JSON:

  * run `base_model.val(save_json=False)` or redirect predictions to external storage.

---

# Final notes

* The code already contains many helpful prints and warnings. Keep them — they make debugging much faster.
* Before large training runs, run the **self-test block** in `models/yolo_dataloader.py` to ensure augmentations & normalizations behave and that the dataset loads correctly.
* Keep `configs/config.yaml` under version control but **do not** store absolute local paths; use relative paths and document where the raw BDD JSON/images live.

---

# Contact / credits

Author: **Utkarsh Suryaman**
Project: YOLOv8m — BDD100K detection pipeline
