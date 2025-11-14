"""
Precompute and cache backend API responses for faster frontend loading.

Author: Utkarsh
Date: November 2025
"""

import os
import json
import requests
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
BASE_URL = "http://localhost:8000"       # Django backend base URL
OUTPUT_DIR = Path("app/precomputed_backend")

# Mapping of output file names → backend endpoints
ENDPOINTS = {
    "class_balance_insights": "/api/analysis/class-balance-insights/",
    "class_distribution": "/api/analysis/class-distribution/",
    "data_quality_per_class": "/api/analysis/data-quality-per-class/",
    "object_density": "/api/analysis/object-density/",
    "sample_visualization": "/api/analysis/sample-visualizations/",
    "visual-distributions": "/api/analysis/visual-distributions/",
    "class_balance": "/api/analysis/class-balance/",
    "dataset_summary": "/api/analysis/dataset-summary/",
}

# ============================================================
#  BACKUP FUNCTION
# ============================================================
def backup_existing_files(output_dir: Path):
    """Move existing files into a timestamped backup directory."""
    if not output_dir.exists():
        return None

    existing_files = list(output_dir.glob("*.json"))
    if not existing_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Backing up old JSON files to {backup_dir}")

    for file in existing_files:
        shutil.move(str(file), str(backup_dir / file.name))
        print(f"  Moved: {file.name}")

    return backup_dir

# ============================================================
#  MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Backup old cached files
    backup_existing_files(OUTPUT_DIR)

    # Step 2 — Fetch fresh API responses
    for file_key, endpoint in ENDPOINTS.items():
        url = f"{BASE_URL}{endpoint}"
        print(f"\n[REQUEST] Fetching: {url}")

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()

            output_path = OUTPUT_DIR / f"{file_key}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"[OK] Saved → {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed for {file_key}: {e}")

        except json.JSONDecodeError:
            print(f"[WARN] Non-JSON response for {file_key}, skipped.")

    print("\n🎉 Precomputation complete!")
    print(f"All JSON files saved in: {OUTPUT_DIR.resolve()}")
