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
BASE_URL = "http://localhost:8000"  # Django backend base URL
OUTPUT_DIR = Path("precomputed_backend")

ENDPOINTS = {
    "dataset_summary": "/api/analysis/dataset-summary/",
    "class_distribution": "/api/analysis/class-distribution/",
    "visual_distributions": "/api/analysis/visual-distributions/",
    "data_quality_per_class": "/api/analysis/data-quality-per-class/",
    "object_density": "/api/analysis/object-density/",
    "class_balance": "/api/analysis/class-balance/",
    "class_balance_insights": "/api/analysis/class-balance-insights/",
    "sample_visualizations": "/api/analysis/sample-visualizations/",
}

# ============================================================
#  HANDLE EXISTING FILES — BACKUP FIRST
# ============================================================
def backup_existing_files(output_dir: Path):
    """Move existing files to a timestamped backup folder before overwriting."""
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None  # no existing files to move

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"

    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Moving old files to: {backup_dir}")

    for file in output_dir.glob("*.json"):
        shutil.move(str(file), str(backup_dir / file.name))
        print(f"  Moved: {file.name}")

    return backup_dir


# ============================================================
#  MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1️⃣ — Backup old results
    backup_existing_files(OUTPUT_DIR)

    # Step 2️⃣ — Fetch and save new API responses
    for name, endpoint in ENDPOINTS.items():
        url = f"{BASE_URL}{endpoint}"
        print(f"📡 Fetching: {url}")

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            data = response.json()
            output_path = OUTPUT_DIR / f"{name}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"✅ Saved: {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {name}: {e}")
        except json.JSONDecodeError:
            print(f"⚠️ Non-JSON response for {name}, skipping.")

    print("\n🎉 Precomputation complete!")
    print(f"All fresh API responses saved in: {OUTPUT_DIR.resolve()}")
