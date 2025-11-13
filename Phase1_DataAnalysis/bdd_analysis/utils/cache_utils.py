"""Utility functions for lightweight result caching with auto-invalidations."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Cache directory (consistent across OS)
CACHE_DIR = Path("runtime_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# 🔹 Dataset Fingerprint Utilities
# ============================================================

def _dataset_checksum(dataset_dir: Path) -> str:
    """
    Generate a checksum based on file names and modification times.
    Detects dataset changes while remaining stable for existing files.
    """
    hash_obj = hashlib.sha256()
    dataset_dir = Path(dataset_dir).resolve()

    for root, _, files in os.walk(dataset_dir):
        for file in sorted(files):
            file_path = Path(root) / file
            try:
                stats = file_path.stat()
                # Use relative file path for cross-platform consistency
                rel_path = str(file_path.relative_to(dataset_dir)).replace("\\", "/")
                hash_obj.update(rel_path.encode())
                hash_obj.update(str(round(stats.st_mtime, 2)).encode())
            except FileNotFoundError:
                continue

    return hash_obj.hexdigest()


def _cache_key(name: str, dataset_dir: Path) -> str:
    """
    Generate a stable cache key from dataset path and analysis name.
    """
    dataset_dir = Path(dataset_dir).resolve()
    checksum = _dataset_checksum(dataset_dir)
    raw_key = f"{dataset_dir.as_posix()}::{name}::{checksum}"
    return hashlib.sha1(raw_key.encode()).hexdigest()


# ============================================================
# 🔹 Cache Operations
# ============================================================

def cache_get(name: str, dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """Return cached data if available and valid."""
    key = _cache_key(name, dataset_dir)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read cache file {cache_file}: {e}")
            return None
    return None


def cache_set(name: str, dataset_dir: Path, data: Dict[str, Any]) -> None:
    """Store result in cache with embedded checksum."""
    key = _cache_key(name, dataset_dir)
    cache_file = CACHE_DIR / f"{key}.json"

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[CACHE] Stored: {cache_file}")
    except Exception as e:
        print(f"[WARN] Could not write cache file {cache_file}: {e}")


def cache_clear(name: str, dataset_dir: Path) -> None:
    """
    Delete cached result(s) for a specific dataset and analysis name.
    Matches all cache files with the dataset + name, regardless of checksum.
    Works across Windows/Linux and handles checksum drift.
    """
    dataset_dir = Path(dataset_dir).resolve()
    dataset_sig = dataset_dir.as_posix()

    deleted = False
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                # Optional: detect cache content for name + dataset match
                data = f.read()
                if name in data or dataset_sig in data:
                    cache_file.unlink()
                    print(f"[CACHE] Cleared: {cache_file}")
                    deleted = True
        except Exception:
            # Fall back to filename-based match
            if name in str(cache_file):
                cache_file.unlink()
                print(f"[CACHE] Cleared (filename match): {cache_file}")
                deleted = True

    if not deleted:
        print(f"[CACHE] No cache found for '{name}' in {CACHE_DIR}")
