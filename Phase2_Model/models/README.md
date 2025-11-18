
---

# **Advanced BDD100K YOLO DataLoader**

### *A Fully Configurable PyTorch DataLoader with Data-Driven Augmentation & Normalization*

This repository provides an **advanced, production-grade DataLoader pipeline** for the BDD100K dataset in **YOLO format**.
It is designed to be modular, highly configurable, class-aware, and optimized for large-scale training.

---

##  **Key Features**

### **1. Multiple Normalization Methods**

Supports a variety of normalization strategies, including:

* **Min-Max Normalization**
* **Z-Score Normalization**
* **Per-Channel Normalization**
* **Robust Normalization (Median + IQR)**
* **Dynamic Range Stretching (percentile-based)**
* **Logarithmic Normalization**
* **Gamma Normalization**
* **Local Contrast Normalization**
* **Hybrid Normalization (Dynamic Range + Z-Score)**
* **Adaptive Class-Specific Normalization**
  Uses class-level brightness/contrast statistics.

---

##  **2. Comprehensive Augmentation Pipeline**

Includes a modular, multi-stage augmentation system:

### **Photometric Augmentations**

* Brightness / Contrast adjustment
* Color balance & hue shift
* Gamma correction
* JPEG compression artifacts
* Gaussian noise

### **Geometric Augmentations**

* Horizontal flip
* Random rotation
* Random crop & rescale
* Perspective distortion
* Random affine transforms

### **Noise & Blur**

* Motion blur
* Gaussian noise injection

### **Spatial Masking Augmentations**

* Random Erasing
* Cutout
* MixUp (with visualization panel generator)

### **Domain-Specific Augmentations (BDD100K-style)**

* Fog simulation
* Rain simulation
* Shadow simulation

---

##  **3. Class-Aware Augmentation**

A flexible system that:

* Applies different augmentations for different classes
* Supports optional `augment_classes` filtering
* Includes sample defaults:

  * Cars → rotation + brightness
  * Trains → fog + shadow
  * Persons → hue + rotation

---

##  **4. Data-Driven Normalization**

Uses per-class image quality statistics:

* Exposure (brightness mean)
* Contrast mean
* Per-class adaptive scaling

This ensures consistent image appearance across classes.

---

##  **5. Efficient Data Loading**

Optimized DataLoader with:

* Parallel workers (`num_workers`)
* Batch collation for YOLO format
* Memory pinning
* Deterministic seeding for reproducibility

---

##  **6. Automatic Visualization Generator**

The system can:

* Generate visual samples for every class
* Apply every augmentation × normalization combination
* Save normalized & augmented outputs
* Produce mixup visualization panels
* Organize outputs into structured folders

Outputs saved in:

```
bdd100k_yolo_subset_samples/
    class_car/
    class_person/
    ...
```

---

##  **7. Self-Test & Benchmarking**

Built-in benchmarking:

* Loads entire dataset
* Measures images/sec
* Measures time per batch
* Validates tensor dimensions (CHW)

Example output:

```
Total Batches: 128
Total Images: 1024
Avg Time/Batch: 0.0312 sec
Throughput: 102.8 images/sec
```

---

##  **8. Easy-to-Use Dataloader Builder**

```python
dataloader = build_dataloader(
    dataset_root="./bdd100k_yolo_subset",
    split="train",
    batch_size=8,
    img_size=640,
    normalization="minmax",
    augment=True,
    augment_classes=["car", "person"],
    class_quality_stats=CLASS_STATS_SAMPLE,
    num_workers=4
)
```

---

##  **9. Clean Dataset Class API**

```python
class BDDYOLODataset(Dataset):
    def __getitem__(self, idx):
        ...
        return img_tensor, labels
```

Supports:

* Automatic label loading in YOLO format
* On-the-fly augmentations
* Configurable normalization
* Tensor & PIL augmentations

---

##  **10. Fully Modular & Extendable Code Structure**

The script is organized in clear functional blocks:

1. Class mappings
2. Normalization functions
3. Augmentation utilities
4. Dataset class
5. Collate function
6. Builder function
7. Visualization generator
8. Performance benchmark

Each part is fully replaceable and extendable.

---

##  **Project Structure**

```
/
├── dataloader.py                # This script
├── README.md                    # Documentation
├── bdd100k_yolo_subset/         # Dataset (images & labels)
└── bdd100k_yolo_subset_samples/ # Visualization outputs
```

---

##  **Requirements**

```
Python 3.8+
PyTorch
Torchvision
Pillow
NumPy
tqdm
```



Sure! Here is a **highly detailed architecture diagram** describing every major internal component of your Advanced BDD100K YOLO DataLoader pipeline.
This is designed in **ASCII-flowchart style**, perfect for a README or documentation.

If you want, I can also produce a **mermaid.js diagram**, **draw.io XML**, or **PNG image**.

---

# ** Advanced BDD100K YOLO DataLoader – Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ADVANCED BDD100K YOLO DATALOADER PIPELINE                │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │   1. User Config / Entry Point      │
                        │─────────────────────────────────────│
                        │ build_dataloader(...) → parameters  │
                        │  • dataset_root                     │
                        │  • batch_size                       │
                        │  • img_size                         │
                        │  • normalization                    │
                        │  • augment                          │
                        │  • augment_classes                  │
                        │  • class_quality_stats              │
                        └─────────────────────────────────────┘
                                        │
                                        ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. Dataset Initialization                            │
└─────────────────────────────────────────────────────────────────────────────┘

       ┌──────────────────────────────┐
       │   BDDYOLODataset.__init__   │
       │──────────────────────────────│
       │ • Load image & label paths   │
       │ • Validate files             │
       │ • Store augmentation configs │
       │ • Print initialization info  │
       └──────────────────────────────┘
                        │
                        ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                           3. __getitem__ Flow                               │
└─────────────────────────────────────────────────────────────────────────────┘

For index `i`:

┌─────────────────────────────┐
│ (A) Image Loading           │
│─────────────────────────────│
│ • Open PIL image            │
│ • Convert → RGB             │
│ • Resize to (img_size,img)  │
└─────────────────────────────┘
                │
                ▼
┌─────────────────────────────┐
│ (B) Label Loading           │
│─────────────────────────────│
│ • Load YOLO txt file        │
│ • Parse: class, x,y,w,h     │
│ • Convert to Tensor         │
└─────────────────────────────┘
                │
                │ IF augment==True AND class in augment_classes
                ▼
┌───────────────────────────────────────────────────────────────┐
│ (C) AUGMENTATION PIPELINE                                     │
│───────────────────────────────────────────────────────────────│
│ 1. Determine active classes from label list                    │
│ 2. Identify class-specific augmentations (PER_CLASS_AUGS)      │
│ 3. Combine with global augmentation pipeline                   │
│ 4. Randomly apply:                                             │
│     • brightness/contrast                                      │
│     • color balance                                            │
│     • gamma correction                                         │
│     • flips, rotation, affine                                  │
│     • fog / rain / shadow                                      │
│     • gaussian noise                                           │
│     • JPEG artifacts                                           │
└───────────────────────────────────────────────────────────────┘
                │
                ▼

┌─────────────────────────────┐
│ (D) Convert to Tensor       │
│─────────────────────────────│
│ • PIL → torch.Tensor        │
│ • Apply tensor-level augs:  │
│      - gaussian noise        │
│      - random erasing        │
│ • Clamp values to [0,1]     │
└─────────────────────────────┘
                │
                ▼

┌───────────────────────────────────────────────────────────────┐
│ (E) NORMALIZATION MODULE                                       │
│───────────────────────────────────────────────────────────────│
│ Depending on user-selected method:                             │
│                                                                │
│  • Min-Max                                                     │
│  • Z-score                                                     │
│  • Per-channel normalization                                   │
│  • Robust median+IQR normalization                             │
│  • Dynamic range stretch                                       │
│  • Logarithmic normalization                                   │
│  • Gamma normalization                                         │
│  • Local contrast normalization                                │
│  • Hybrid normalization                                        │
│                                                                │
│ Or **class-aware adaptive normalization**:                     │
│  uses exposure_mean + contrast_mean from class_stats           │
└───────────────────────────────────────────────────────────────┘
                │
                ▼

┌─────────────────────────────┐
│ Output: (img_tensor, labels)│
└─────────────────────────────┘



┌─────────────────────────────────────────────────────────────────────────────┐
│                        4. YOLO Collate Function                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

Images stacked → (B, C, H, W)
Labels stored as list of tensors per image.

```
collate_fn:
    images = torch.stack(batch_images)
    targets = list of variable-length label tensors
```

---

# ** 5. DataLoader Runtime Pipeline**

```
Dataloader() →
    • Shuffling
    • Parallel workers
    • Prefetching
    • Pin memory
    • Collation
    ↓
Batches Yielded:
    (batch_tensor, list_of_labels)
```

---

# ** 6. Visualization Generator Architecture**

```
┌───────────────────────────────────────────────┐
│ Visualization Engine                           │
│───────────────────────────────────────────────│
│ • Picks 1 sample per class                     │
│ • For every normalization × augmentation combo │
│ • Applies transformations                      │
│ • Saves images under class-specific folders    │
│ • Generates MixUp visualization panels         │
└───────────────────────────────────────────────┘
```

---

# ** 7. Performance Benchmark Module**

```
Benchmarking:
  • Iterate through full dataloader
  • Measure throughput (img/sec)
  • Time per batch
  • Validate CHW format
  • Report statistics
```



---

##  Support

If you want, I can also create:

✔ A **badge-style** header
✔ A shorter or longer version
✔ A GitHub-ready version with code blocks, examples, gifs, and screenshots
✔ A version for your portfolio

---

