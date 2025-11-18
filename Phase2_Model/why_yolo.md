---

# **Justification for Using YOLOv8m for BDD100K Object Detection**

Object detection on the BDD100K dataset presents several challenges: large-scale urban scenes, heavy class imbalance, diverse lighting/weather conditions, and a need for real-time or near-real-time inference. Selecting an appropriate detection architecture is crucial for achieving a balance between **accuracy, speed, and deployability**.

After evaluating multiple model families and variants, **YOLOv8m (medium)** was selected as the optimal backbone for this project.
The choice is supported by technical, practical, and deployment-driven considerations outlined below.

---

# **1. Balance Between Accuracy and Speed**

YOLOv8m achieves an ideal middle ground:

* **Higher accuracy** than YOLOv8n (nano) and YOLOv8s (small)
* **Much lower inference cost** than YOLOv8l / YOLOv8x
* Suitable for real-time or embedded applications

### Model Comparison (Ultralytics official benchmarks)

| Model       | mAP50-95 | Params       | FLOPs        | Speed (ms)  |
| ----------- | -------- | ------------ | ------------ | ----------- |
| YOLOv8n     | Low      | Very small   | Very low     | Fastest     |
| YOLOv8s     | Medium   | Small        | Low          | Fast        |
| **YOLOv8m** | **High** | **Moderate** | **Moderate** | **Optimal** |
| YOLOv8l     | Higher   | Large        | High         | Slower      |
| YOLOv8x     | Highest  | Very large   | Very high    | Slowest     |

YOLOv8m offers **significant accuracy gains over small models** without the computational overhead of large ones—ideal for the BDD100K domain.

---

# **2. Best Fit for BDD100K's Complexity**

BDD100K scenes include:

* Dense traffic
* Small objects (traffic lights, traffic signs)
* Occlusions & clutter
* Night, fog, snow, rain conditions
* High-resolution images

YOLOv8m has:

* **Deeper backbone** (C2f blocks)
* **Larger feature maps**
* **Better receptive field**
* **Stronger representation power**

These allow it to detect **small and distant objects** more accurately, which is critical for BDD100K.

---

# **3. Superior Small-Object Detection**

BDD100K contains small objects like:

* Traffic lights
* Traffic signs
* Pedestrians far away
* Motorcycles and riders

YOLOv8m’s architecture provides:

* Richer multi-scale feature fusion
* More capacity to learn tiny objects
* Stronger detection head (Decoupled head + Task-specific convs)

In comparison, YOLOv8n and YOLOv8s tend to **miss small objects** more often.

---

# **4. Better Generalization with Moderate Dataset Size**

Your subset (e.g., ~2000 train + 400 val) is relatively small.

YOLOv8m:

* Has enough capacity to generalize better than lightweight models
* Avoids **underfitting**, which is common in smaller models
* Still avoids **overfitting**, thanks to strong regularization and modern architecture (C2f, DFL, EMA training)

Thus it gives the best trade-off for the subset size.

---

# **5. Deployment Flexibility (Edge + Cloud)**

YOLOv8m runs efficiently on:

* RTX-series GPUs
* LG ThinQ chipset (if quantized)
* NVIDIA Jetson devices
* Embedded automotive hardware (TensorRT-ready)

Compared to YOLOv8l/x, YOLOv8m:

* Has lower VRAM requirements
* Fits in memory even on mid-range GPUs
* Is easy to convert to ONNX / TensorRT / OpenVINO

This makes it suitable for **real-time ADAS, traffic analytics, or on-device inference**.

---

# **6. Strong Training Stability + Modern Loss Functions**

YOLOv8m uses:

* **DFL (Distribution Focal Loss)** for bounding boxes
* **BCEWithLogits** for classification
* **Decoupled head** improves gradient flow
* **Deep C2f blocks** for stable representation learning
* **Faster convergence** on COCO-like datasets

BDD100K is similar to COCO (multi-class, dense objects), so YOLOv8m benefits fully from this architecture.

---

# **7. Practical Considerations (Your Project Constraints)**

You had additional constraints:

### ✔ Limited training time

YOLOv8m trains significantly faster than YOLOv8l/x.

### ✔ Limited GPU memory on the interviewer's machine

YOLOv8m fits comfortably into 12–16 GB VRAM.

### ✔ Need a single robust model

YOLOv8m can handle:

* Day / night
* Low-quality images
* Motion blur
* Class imbalance
* Diverse traffic densities

A smaller model fails in these conditions; a larger model is too slow and heavy.

---

# **8. Strong Baseline for Further Improvements**

YOLOv8m provides enough capacity for:

* Fine-tuning
* Knowledge distillation
* Adapter layers
* Quantization
* Custom loss functions
* Augmentations

This flexibility allows future performance improvements without changing architecture.

---

# **9. Industry Standard for Autonomous Driving**

YOLOv8m is widely used in:

* Autonomous driving research
* Traffic monitoring systems
* Smart city analytics
* Road safety applications

Its accuracy on:

* Cars
* Pedestrians
* Trucks
* Riders
* Traffic lights
* Traffic signs

matches real-world ADAS requirements.

---

# **10. Proven Performance on COCO-like Datasets**

Because BDD100K is similar to COCO (urban scenes, multi-class), YOLOv8m achieves **excellent transfer learning performance**.

YOLOv8m pretrained on COCO offers:

* Strong initialization
* Faster convergence
* Higher mAP
* Better recall
* More stable gradients

than smaller models.

---

# **Conclusion: Why YOLOv8m Is the Best Choice**

### ✔ Best balance of accuracy vs speed

### ✔ Perfect for BDD100K’s complexity

### ✔ Strong small-object detection

### ✔ Efficient for training and deployment

### ✔ Good fit for subset size

### ✔ Architecture supports modern loss and strong generalization

### ✔ Industry-proven model for traffic datasets

### ✔ Fits GPU memory constraints

### ✔ Strong performance without heavy compute cost

YOLOv8m was selected because it delivers **the highest practical accuracy**, **training stability**, and **deployment flexibility**, making it the most appropriate choice for this ADAS-style detection task.

---

#  **Comparison: YOLOv8m vs Faster R-CNN, DETR, Swin Transformer, SSD**

### *Objective: Justify why YOLOv8m is the most balanced and practical choice for BDD100K.*

---

##  **High-Level Comparison Table**

| Model                         | Architecture Type              | Strengths                                                                      | Weaknesses                                            | Performance on Small Objects | Inference Speed           | Training Complexity | Best For                                   |
| ----------------------------- | ------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------- | ------------------------- | ------------------- | ------------------------------------------ |
| **YOLOv8m**                   | One-stage, anchor-free         | Fast, high accuracy, optimized for small objects, lightweight, stable training | Slightly lower AP than DETR/Swin-L on huge datasets   | **⭐⭐⭐⭐⭐ Excellent**          | **⭐⭐⭐⭐ Fast (real-time)** | **⭐⭐ Easy**         | Real-time detection, autonomous driving    |
| **Faster R-CNN**              | Two-stage                      | Very accurate, proven, robust                                                  | Slow, high compute, heavy backbone                    | ⭐⭐⭐ Medium                   | ⭐ Slow (not real-time)    | ⭐⭐⭐ Complex         | Research, high-precision offline detection |
| **DETR**                      | Transformer-based (end-to-end) | Simple pipeline, strong global reasoning                                       | Extremely slow training, struggles with small objects | ⭐ Poor                       | ⭐⭐ Very slow              | ⭐⭐⭐⭐ Very complex   | High-end GPUs, research                    |
| **Swin Transformer Detector** | Hierarchical Transformer       | Very high accuracy, strong small-object performance                            | Heavy, slow, requires large GPU memory                | ⭐⭐⭐⭐ Good                    | ⭐⭐ Slow                   | ⭐⭐⭐ Complex         | Large-scale offline training               |
| **SSD300 / SSD512**           | Classic one-stage              | Fast, lightweight                                                              | Low accuracy, poor for tiny objects                   | ⭐ Very poor                  | ⭐⭐⭐⭐ Fast                 | ⭐⭐ Simple           | Low-end devices, outdated                  |

---

#  **Detailed Explanation — Why YOLOv8m Wins**

---

## 1️ **YOLOv8m vs Faster R-CNN**

| Aspect     | YOLOv8m              | Faster R-CNN |
| ---------- | -------------------- | ------------ |
| Type       | One-stage            | Two-stage    |
| Speed      | **~30–80 FPS**       | **3–7 FPS**  |
| Accuracy   | High                 | High         |
| Latency    | Low                  | High         |
| Complexity | Low                  | Very high    |
| Deployment | Easy (ONNX/TensorRT) | Hard         |

**Why YOLOv8m wins:**
Faster R-CNN is accurate but heavy. It cannot run in real-time for autonomous-driving scenarios. BDD100K requires fast response (traffic lights, pedestrians, etc.) → YOLOv8m is 10–20× faster with similar accuracy.

---

## 2️ **YOLOv8m vs DETR (Facebook)**

| Aspect           | YOLOv8m       | DETR                                           |
| ---------------- | ------------- | ---------------------------------------------- |
| Detection type   | Convolutional | Transformer + Hungarian matching               |
| Convergence time | 50–100 epochs | **300–500 epochs**                             |
| Small objects    | **Strong**    | Weak (Transformers struggle with tiny objects) |
| GPU memory       | Low           | Very high                                      |
| Real-time        | Yes           | No                                             |

**Why YOLOv8m wins:**
DETR is elegant but slow. It requires enormous datasets + long training. Its small-object detection is inferior — a big issue for **traffic lights / signs**.

---

## 3️ **YOLOv8m vs Swin Transformer Detector**

| Aspect     | YOLOv8m  | Swin Detector |
| ---------- | -------- | ------------- |
| Accuracy   | High     | Very high     |
| Speed      | **Fast** | Slow          |
| Params     | 25M      | 48M–300M      |
| Memory     | 1–4 GB   | 8–24 GB       |
| Deployment | Easy     | Hard          |

**Why YOLOv8m wins:**
Swin Transformer is accurate but unsuitable for deployment due to:

* Gigantic memory requirement
* Slow inference (even on A100)
* Hard to optimize for edge devices

YOLOv8m is **far more practical**.

---

## 4️ **YOLOv8m vs SSD (Single Shot Detector)**

| Aspect         | YOLOv8m   | SSD        |
| -------------- | --------- | ---------- |
| Performance    | **SOTA**  | Old (2016) |
| Small objects  | Excellent | Very poor  |
| Speed          | Fast      | Fast       |
| Accuracy       | Very high | Low        |
| Generalization | Excellent | Poor       |

**Why YOLOv8m wins:**
SSD is extremely outdated and performs badly on dense scenes like BDD100K.

---

#  **Conclusion: YOLOv8m Is the Most Balanced Choice**

### ✔ Best trade-off between speed and accuracy

### ✔ Excellent performance on small objects (critical for traffic signals)

### ✔ Real-time inference capability

### ✔ Easy training, stable gradients

### ✔ Lightweight enough for deployment

### ✔ Modern architecture (C2f + DFL + decoupled head)

### ✔ Fits BDD100K’s environmental variations (night, rain, fog, low-contrast)

YOLOv8m provides **the optimal compromise** between accuracy, speed, and deployability — making it the most sensible choice for BDD100K autonomous-driving object detection.





---
