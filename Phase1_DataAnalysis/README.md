# Phase 1 — Data Analysis & Visual Analytics (BDD100K)

**Repository:** `Phase1_DataAnalysis/`
**Author:** Utkarsh Suryaman — BDD100K Visual Analytics Suite (Robert Bosch)

---

This README documents **everything** in Phase 1: what’s included, why it exists, how it works, how to run it (Docker and local), data formats, API endpoints, the Streamlit frontend (detailed per-page behavior and UI controls), the precompute / caching model, and troubleshooting (including large-file / git-lfs guidance given your recent push issues).

Use this file as the canonical blueprint for Phase 1. Copy it to:

```
Phase1_DataAnalysis/README.md
```

---

# Table of contents

1. [Project overview](#project-overview)
2. [High-level features](#high-level-features)
3. [Repository structure (Phase 1)](#repository-structure-phase-1)
4. [Data sources & formats used](#data-sources--formats-used)
5. [How it works — pipeline & flow](#how-it-works---pipeline--flow)
6. [API endpoints (Django backend)](#api-endpoints-django-backend)
7. [Streamlit frontend — pages & controls (detailed)](#streamlit-frontend---pages--controls-detailed)
8. [Precompute & caching (recommended workflow)](#precompute--caching-recommended-workflow)
9. [Run locally (non-Docker)](#run-locally-non-docker)
10. [Run with Docker / docker-compose (recommended)](#run-with-docker--docker-compose-recommended)
11. [Deployment notes & production considerations](#deployment-notes--production-considerations)
12. [Troubleshooting & Git / large files (git-lfs)](#troubleshooting--git--large-files-git-lfs)
13. [Testing & validation](#testing--validation)
14. [Contributing](#contributing)
15. [Contact & Credits](#contact--credits)

---

# Project overview

Phase 1 is a production-oriented data analysis and visualization module for the BDD100K dataset. It:

* extracts and caches dataset statistics (class distribution, object density, visual metrics such as brightness/contrast/saturation, data-quality metrics like sharpness/exposure),
* generates curated sample categories (basic, dense, occlusion, bbox extremes, class representatives, diversity),
* exposes a Django REST API for these precomputed JSONs,
* runs an interactive Streamlit frontend that visualizes all analytics: charts, tables, boxplots, heatmaps, and sample image galleries,
* is Dockerized for reproducibility.

The core design decision: **compute heavy statistics once (precompute)** and serve cached JSONs for a snappy UI. The UI provides a “Recompute Summary” control to force re-computation when required.

---

# High-level features

* Automated conversion of BDD100K annotation JSON → convenient internal formats (YOLO-ish labels used elsewhere).
* Precompute script that saves JSON outputs to `precomputed_backend/` for instant frontend load.
* REST API with endpoints that serve class distribution, object density, visual distributions, data-quality-per-class, sample lists, and combined summaries.
* Streamlit dashboard with these pages:

  * Dataset Summary (with Lottie loader & recompute option)
  * Class Distribution (train vs validation, Count / Log / Percentage modes)
  * Visual Distributions (Brightness / Contrast / Saturation histograms & boxplots)
  * Data Quality per Class (exposure / contrast / sharpness / colorfulness + heatmaps)
  * Object Density
  * Class Balance Insights (Gini, entropy, drift analysis and sampling recommendations)
  * Sample Visualizations (2 images per curated category)
* Polished UI with CSS/glassmorphism, Plotly/Matplotlib/Seaborn visualizations.

---

# Repository structure (Phase 1)

```
Phase1_DataAnalysis/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── precompute_backend_data.py         # main precompute script (generate JSONs)
├── precomputed_backend/               # cached JSONs served by API
│   ├── dataset_summary.json
│   ├── class_distribution.json
│   ├── visual-distributions.json
│   └── ...
├── bdd100k_utk/                       # Django project
│   ├── manage.py
│   └── ...
├── bdd_analysis/                      # Django app with API endpoints
│   ├── views.py
│   ├── serializers.py
│   └── ...
├── streamlit_app/
│   └── app.py                         # the front-end you provided (UI pages)
├── samples/                           # curated sample images (6 categories)
│   ├── 1_basic_samples
│   ├── 2_extreme_density
│   └── ...
├── docs/
│   └── ... (optional)
└── README.md (this file)
```

---

# Data sources & formats used

* Source dataset: **BDD100K** annotation JSONs (e.g., `bdd100k_labels_images_val.json`) and corresponding images.
* Backend precomputed JSONs (placed under `precomputed_backend/`), example keys:

  * `dataset_summary.json` — high-level counts, last computed timestamp, `meta.cache_status`
  * `class_distribution.json` — `train` & `val` dictionaries with `class_counts`, `class_percent`, `log_class_counts`, etc.
  * `visual-distributions.json` — brightness/contrast/saturation per-class histograms and mean values.
  * `data_quality_per_class.json` — exposure_mean/contrast_mean/sharpness_mean/colorfulness_mean etc.
  * `object_density.json` — objects per image histograms, per-class density averages.
  * `class_balance_insghts1.json` — sampling suggestions, drift, coverage, Gini, entropy, etc.

**Note:** The Streamlit UI expects certain JSON structure/keys exactly as used in code — if you change the precompute output keys, update both backend serializers and the frontend `fetch_api` usage.

---

# How it works — pipeline & flow

1. `precompute_backend_data.py` (or equivalent) reads BDD100K labels & images, computes:

   * counts & distributions,
   * per-class histograms,
   * data-quality metrics using OpenCV (sharpness, contrast, exposure),
   * curated sample selection logic.
2. Script writes JSON files under `precomputed_backend/`.
3. Django backend serves these JSONs through REST endpoints.
4. Streamlit frontend fetches JSONs (or reads from cache) and renders:

   * charts (Plotly/Matplotlib/Seaborn),
   * tables (pandas),
   * sample images (PIL),
   * Lottie loader for recompute operations.
5. UI offers a "Recompute" button (Dataset Summary) which calls API endpoint with `?recompute=1`. Backend should implement this to re-run the precompute logic (or queue it) and then return the regenerated JSON when done.

**Important behavior**:

* By default the frontend loads **cached precomputed** JSONs (fast).
* If the user presses **Recompute Summary**, the frontend calls the backend API with `?recompute=1`. The backend must then perform re-computation and replace JSON files in `precomputed_backend/` (or return a report when complete). The frontend displays a Lottie animation and blocks until completion. (This is already implemented in your `fetch_dataset_summary` call in the frontend.)

---

# API endpoints (Django backend)

Implement these endpoints (names used in the front-end — match them exactly):

```
GET  /api/analysis/dataset-summary/           -> dataset_summary.json
GET  /api/analysis/class-distribution/        -> class_distribution.json
GET  /api/analysis/visual-distributions/      -> visual-distributions.json
GET  /api/analysis/data-quality-per-class/    -> data_quality_per_class.json
GET  /api/analysis/object-density/            -> object_density.json
GET  /api/analysis/class-balance-insights/    -> class_balance_insghts1.json
GET  /api/analysis/samples/                   -> list of sample image names and categories
```

* All endpoints should return `200` + JSON or meaningful HTTP error.
* Support `?recompute=1` where recomputation is heavy: the endpoint should either:

  * perform the recompute in-process and return the regenerated JSON (this is simple but blocks the request until computation finishes), **or**
  * trigger a background job (recommended for production) and return job status (but then frontend needs polling). Your current Streamlit front-end expects synchronous behavior (it waits), so the simplest path is to run recompute synchronously in the endpoint for now.

**Security note:** If recompute runs long or is expensive, add authentication or restrict access.

---

# Streamlit frontend — pages & controls (detailed)

Your `streamlit_app/app.py` (the code you provided) contains the following pages and exact behaviors:

### 1. Dataset Summary

* Shows a Lottie animation (loader) when recomputing.
* `Recompute Summary` button calls `fetch_dataset_summary(recompute=True)` which does `GET /api/analysis/dataset-summary/?recompute=1`.
* Shows cards: Training Images, Validation Images, Training Annotations, Avg Objects/Image, Last Computed.
* Displays a meta info area: `Last Computed` and cache status from `data.meta.cache_status`.
* Design notes: glassmorphism CSS, gradient text, `st.toast` on success.

### 2. Class Distribution

* Buttons for Count / Log Count / Percentage (implemented via `st.button`; uses `st.session_state["mode"]`).
* Tabs for `Training Set` and `Validation Set`.
* Fetches `GET /api/analysis/class-distribution/` and expects `train` and `val` dicts with:

  * `class_counts`, `log_class_counts`, `class_percent`.
* Plotly bar charts (beautiful styling, hover templates, bargap).

### 3. Visual Distributions

* Fetches `GET /api/analysis/visual-distributions/`.
* Two views: Histogram Distribution and Boxplot (via `st.radio`).
* Histogram: draws train & val distributions for brightness/contrast/saturation using Matplotlib & Seaborn.
* Boxplot: reconstructs values from histogram bins (helper `expand_histogram_to_values`) and renders per-class boxplots.
* Shows a styled `df_mean` table comparing train vs val mean metrics with Δ columns.

### 4. Data Quality per Class

* Loads `precomputed_backend/data_quality_per_class.json` (or calls `GET /api/analysis/data-quality-per-class/` fallback).
* Visualizes exposure/contrast/sharpness/colorfulness via compact bar plots, correlation heatmap, variance error bars, and a scatter `Sharpness vs Contrast`.
* CSS glass cards, Lottie animation for startup.

### 5. Object Density

* Loads `precomputed_backend/object_density.json` (or calls API).
* Presents summary table with `Total Images`, `Total Objects`, `Mean Objects/Image`, `Std Dev`.
* Histogram of "objects per image", and per-class object density (horizontal bars).

### 6. Class Balance Insights

* Uses `precomputed_backend/class_balance_insghts1.json` (or `GET /api/analysis/class-balance-insights/`).
* Shows summary cards: total train objects, total val objects, most/least common classes.
* Shows Gini index, Shannon entropy, class-coverage, drift analysis table, and recommended sampling weights/actions.
* Displays DataFrame of sampling recommendations.

### 7. Sample Visualizations

* Reads `samples/` folder (6 categories).
* For each category picks up to 2 images randomly and displays them with captions.
* If images missing, shows warnings (so ensure samples are present).

### Footer & Sidebar

* Footer copyright + contact.
* Sidebar contains a cache clearing input: `Enter cache name to clear:` — uses `clear_cache(cache_target)` to remove cached JSONs.

---

# Precompute & caching (recommended workflow)

**Why precompute?** Heavy image-level metrics (sharpness, histograms, object counts) are slow. Precompute once and serve JSONs for immediate front-end responses.

**Script:** `precompute_backend_data.py` — should output JSON files to `precomputed_backend/`. Example filenames:

* `precomputed_backend/dataset_summary.json`
* `precomputed_backend/class_distribution.json`
* `precomputed_backend/visual-distributions.json`
* `precomputed_backend/data_quality_per_class.json`
* `precomputed_backend/object_density.json`
* `precomputed_backend/class_balance_insghts1.json`
* `precomputed_backend/samples.json` (sample lists)

**Recommended run:**

```bash
# from Phase1_DataAnalysis/
python precompute_backend_data.py
# verify
ls precomputed_backend/
```

**Cache policy:**

* The frontend reads `precomputed_backend/` if present.
* Recompute button calls backend with `?recompute=1`. Backend should:

  * re-run the precompute script (overwrite JSONs),
  * return new `dataset_summary.json` (with `meta.cache_status = "fresh"` and `Last Computed` timestamp).

**Clear cache:**

* You can delete specific JSON from `precomputed_backend/` to force recompute next time:

```bash
rm precomputed_backend/dataset_summary.json
```

---

# Run locally (non-Docker)

1. Create virtual environment & install dependencies:

```bash
cd Phase1_DataAnalysis
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

2. Precompute (strongly recommended before launching UI):

```bash
python precompute_backend_data.py
```

3. Run Django backend (assumes `bdd100k_utk/manage.py` present and configured):

```bash
cd bdd100k_utk
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

4. Run Streamlit frontend:

```bash
cd ../streamlit_app
streamlit run app.py
# UI will open at http://localhost:8501
```

**Notes:**

* Backend `BASE_URL` defaults to `http://bdd_backend:8000` in the front-end code. For local runs, either edit the `BASE_URL` in `app.py` to `http://localhost:8000` or set an env var / config override.
* The front-end code uses `timeout=None` for requests — be mindful of blocking behavior on recompute.

---

# Run with Docker / docker-compose (recommended)

There are `Dockerfile.backend`, `Dockerfile.frontend`, and a `docker-compose.yml` that orchestrates:

```bash
cd Phase1_DataAnalysis
docker-compose up --build
```

* Backend service name is `bdd_backend` and runs on port `8000` inside the compose network, matching the frontend `BASE_URL="http://bdd_backend:8000"`.
* Streamlit frontend service runs on `8501` mapped to host (check `docker-compose.yml`).
* Precompute can be run inside backend container (recommended) so cached JSONs are available at container volume shared by backend & frontend.

---

# Deployment notes & production considerations

* For production, do not run recompute synchronously on HTTP request — convert recompute to an asynchronous job (Celery, RQ) and return job status; frontend polls job status and shows progress. For your current Streamlit synchronous UI, keep recompute synchronous for simplicity.
* Protect recompute endpoint with authentication / token.
* Store precomputed JSONs on durable storage (S3 / shared NFS) for multi-instance setups.

---

# Troubleshooting & Git / large files (git-lfs)

You previously hit GitHub's 100MB limit on `predictions.json` (~293MB) — a few notes & safe steps:

## 1) Don’t push heavy artifacts to repo

* Save large artifacts (predictions.json, model weights, large images/videos) to a release artifact storage (S3, Google Drive, GitHub Releases, or an artifact server).
* Keep repository focused on code + small sample images.

## 2) Use Git LFS for large binaries

Install git-lfs on both machines (Docker host and your local Windows):

**Linux / Debian:**

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

**Windows:** Download & run the installer from [https://git-lfs.github.com](https://git-lfs.github.com) or use `choco install git-lfs`.

Then track file types:

```bash
git lfs track "*.mp4"
git lfs track "Phase1_DataAnalysis/**/*.mp4"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "Phase3_eval_viz/**/predictions.json"
git add .gitattributes
git commit -m "Track large files with git-lfs"
```

**Caveat:** If the large file is already in history, you must remove it from the history (you already used `git filter-repo`, which is correct). After rewriting history, add the remote again and force-push.

## 3) If large file already pushed & rejected

* Use `git filter-repo` (you have) or `git filter-branch` (less recommended). You successfully removed the file with `git filter-repo` — good.
* After `git filter-repo`, re-add your remote and `git push --force`:

```bash
git remote add origin https://github.com/Utkarsh-shift/Utk_Bosch_bdd100K.git
git push --force origin main
```

(You earlier saw the origin removed warning — re-add it as above.)

## 4) Re-introducing Phase1 files deleted by filter

If `git filter-repo` removed tracked files from history, but you still have them locally in the working directory, simply re-add and commit:

```bash
git add Phase1_DataAnalysis/
git commit -m "Re-add Phase1_DataAnalysis contents after history rewrite"
git push origin main
```

If `git` reports "nothing to commit, working tree clean", ensure the files are not in `.gitignore` and that they are untracked. Use:

```bash
git status --ignored
```

To add binary large files (mp4) after enabling git-lfs, re-add them (they will be stored in LFS, not in normal git objects):

```bash
git lfs track "Phase1_DataAnalysis/**/*.mp4"
git add .gitattributes
git add Phase1_DataAnalysis/path/to/video.mp4
git commit -m "Add frontend mp4s via git-lfs"
git push origin main
```

---

# Testing & validation

* Unit-test precompute functions (histograms, mean calculations).
* Validate JSON shapes against front-end expectations (e.g., `class_counts` has keys with class names).
* Manual test: run `precompute_backend_data.py`, verify `precomputed_backend/*.json`, start backend and frontend, click Recompute, spot-check UI charts and sample images.

---

# Contributing

* Follow the repository code style and keep frontend keys synchronized with back-end precomputed JSON keys.
* When adding new precomputed metrics, update:

  * `precompute_backend_data.py` to generate the JSON,
  * the Django API serializer/view to serve it,
  * the Streamlit frontend to consume/display it.

---

# Example commands summary

```bash
# run precompute
python precompute_backend_data.py

# run Django backend
cd bdd100k_utk
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# run streamlit frontend
cd ../streamlit_app
streamlit run app.py

# run with docker-compose (recommended)
cd Phase1_DataAnalysis
docker-compose up --build
```

Git LFS quickstart (if you need it):

```bash
# install git-lfs first (system-specific)
git lfs install
git lfs track "*.mp4"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Enable git-lfs for large binaries"
```

---

