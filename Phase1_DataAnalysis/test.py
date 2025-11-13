import requests
try:
    r = requests.get("http://bdd_backend:8000/api/analysis/dataset-summary/", timeout=5)
    print(" Status:", r.status_code)
    print(" Response:", r.text[:500])
except Exception as e:
    print(" Error:", e)