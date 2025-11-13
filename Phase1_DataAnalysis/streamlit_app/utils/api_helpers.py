import requests
import streamlit as st
import os 
import json 
API_BASE = "http://127.0.0.1:8000/api/analysis/"
BASE_URL = "http://127.0.0.1:8000/api/analysis/"
CACHE_DIR = "precomputed_backend"
@st.cache_data(show_spinner=False, ttl=600)
def fetch_api(endpoint: str):
    """
    Load data from a cached JSON file if available, else fetch from API and cache it.
    Example:
        data = fetch_api("visual-distributions")
    """
    # Define full cache path
    cache_path = os.path.join(CACHE_DIR, f"{endpoint}.json")

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            st.info(f"📁 Loaded '{endpoint}' from cache.")
            data.setdefault("meta", {"cache_status": "local"})
            return data
        except Exception as e:
            st.warning(f"⚠️ Failed to read cached file for '{endpoint}': {e}")

    # If cache not found or unreadable, fetch from API
    try:
        st.info(f"🌐 Fetching '{endpoint}' data from API...")
        response = requests.get(f"{BASE_URL}/api/analysis/{endpoint}/", timeout=None)
        response.raise_for_status()
        data = response.json()

        # Save to cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

        st.success(f"✅ '{endpoint}' data fetched and cached successfully.")
        data["meta"] = {"cache_status": "api"}
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch '{endpoint}' from API: {e}")
        return None

def clear_cache(endpoint):
    """Trigger cache clear API."""
    try:
        url = f"{API_BASE}clear-cache/"
        resp = requests.get(url, params={"name": endpoint})
        if resp.status_code == 200:
            st.success(resp.json().get("status"))
        else:
            st.error(resp.text)
    except Exception as e:
        st.error(f"Cache clear failed: {e}")
