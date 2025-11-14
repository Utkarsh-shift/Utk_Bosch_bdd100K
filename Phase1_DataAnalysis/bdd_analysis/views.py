"""Views for BDD100K dataset analysis."""

import os
from pathlib import Path
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

from .utils.stats import compute_dataset_summary
from .utils.data_quality import compute_data_quality_metrics_per_class
from .utils.data_loader import load_annotations


from .utils.stats import compute_class_distribution

from .utils.object_density import compute_object_density
# Load environment variables from .env
load_dotenv()
from .utils.cache_utils import cache_clear
def handle_recompute(request, cache_key, dataset_dir):
    """Handle ?recompute=1 flag globally."""
    recompute = request.GET.get("recompute", "0") in ("1", "true", "True")
    if recompute:
        cache_clear(cache_key, dataset_dir)
        print(f"[CACHE] Cleared cache for {cache_key}")
    return recompute

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import os
from .utils.cache_utils import cache_clear


@csrf_exempt
def clear_cache_api(request):
    """
    API endpoint to clear cached data.
    Usage example:
      /api/analysis/clear-cache/?name=sample_visualizations
    """
    try:
        name = request.GET.get("name")
        if not name:
            return JsonResponse(
                {"error": "Missing ?name= parameter. Example: /api/analysis/clear-cache/?name=class_balance_summary"},
                status=400,
            )

        dataset_root = Path(os.getenv("BDD100K_ROOT", "."))
        cache_clear(name, dataset_root)

        return JsonResponse({"status": f"Cache '{name}' cleared successfully."}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_root(request):
    """
    Root API endpoint listing all available analysis routes.
    """
    return JsonResponse({
        "available_endpoints": {
            "dataset_summary": "/api/analysis/dataset-summary/",
            "class_distribution": "/api/analysis/class-distribution/",
            "visual_distributions": "/api/analysis/visual-distributions/",
            "data_quality_per_class": "/api/analysis/data-quality-per-class/",
            "object_density": "/api/analysis/object-density/",
            "class_balance": "/api/analysis/class-balance/",
            "class_balance_insights": "/api/analysis/class-balance-insights/",
            "sample_visualizations": "/api/analysis/sample-visualizations/",
            "clear_cache": "/api/analysis/clear-cache/?name=<cache_name>"  
        }
    })


import json
from pathlib import Path
from django.http import JsonResponse

def dataset_summary_api(request):
    dataset_root = os.getenv("BDD100K_ROOT") or os.getenv("DATASET_PATH")
    dataset_root = Path(dataset_root)
    print("Dataset root in API: *************", dataset_root)

    # Precomputed fallback file
    print(os.getcwd())
    precomputed_path = Path(os.getcwd() + "/precomputed_backend/data_summary.json")
    print("Precomputed path:^^^^^^^^^^", precomputed_path)
    if precomputed_path.exists():
            with open(precomputed_path, "r") as f:
                print("Precomputed file found")
                summary = json.load(f)
            summary["meta"] = {
                "cache_status": "precomputed",
                "note": "Dataset missing. Loaded precomputed summary."
            }
            return JsonResponse(summary, json_dumps_params={"indent": 2})
    # If recompute=1 → FORCE compute
    recompute = handle_recompute(request, "dataset_summary", dataset_root)

    # Try normal computation first
    if not recompute:
        try:
            summary = compute_dataset_summary(recompute=False)
            summary["meta"] = {"cache_status": "hit"}
            return JsonResponse(summary, json_dumps_params={"indent": 2})
        except FileNotFoundError:
            # dataset missing → fallback
            pass

    # Try recompute
    try:
        summary = compute_dataset_summary(recompute=True)
        summary["meta"] = {"cache_status": "miss"}
        return JsonResponse(summary, json_dumps_params={"indent": 2})

    except Exception as e:
        # FINAL FALLBACK → use precomputed file
        if precomputed_path.exists():
            with open(precomputed_path, "r") as f:
                summary = json.load(f)
            summary["meta"] = {
                "cache_status": "precomputed",
                "note": "Dataset missing. Loaded precomputed summary."
            }
            return JsonResponse(summary, json_dumps_params={"indent": 2})

        # nothing available → return error
        return JsonResponse(
            {
                "error": "Dataset files missing and precomputed summary not found.",
                "exception": str(e),
                "expected_dataset_root": str(dataset_root),
                "expected_precomputed_file": str(precomputed_path.resolve())
            },
            status=500
        )




import json
from pathlib import Path
from django.http import JsonResponse

def class_distribution_api(request):
    dataset_root = Path(os.getenv("BDD100K_ROOT") or os.getenv("DATASET_PATH"))
    precomputed_path = Path("precomputed_backend/class_distribution.json")
    precomputed_path = Path(os.getcwd() + "/precomputed_backend/class_distribution.json")
    print("Precomputed path:^^^^^^^^^^", precomputed_path)
    if precomputed_path.exists():
            with open(precomputed_path, "r") as f:
                summary = json.load(f)
            summary["meta"] = {
                "cache_status": "precomputed",
                "note": "Dataset missing. Loaded precomputed summary."
            }
            return JsonResponse(summary, json_dumps_params={"indent": 2})
    # Check for recompute
    recompute = handle_recompute(request, "class_distribution", dataset_root)

    # ================================
    # 1️⃣ Try NORMAL (cached) computation
    # ================================
    if not recompute:
        try:
            result = compute_class_distribution(recompute=False)
            result["meta"] = {"cache_status": "hit"}
            return JsonResponse(result, json_dumps_params={"indent": 2})
        except FileNotFoundError:
            pass  # dataset files missing → fallback to next steps

    # ================================
    # 2️⃣ Try FORCED recompute
    # ================================
    try:
        result = compute_class_distribution(recompute=True)
        result["meta"] = {"cache_status": "miss"}
        return JsonResponse(result, json_dumps_params={"indent": 2})

    except Exception as e:
        # Dataset missing OR compute failed
        pass

    # ================================
    # 3️⃣ FINAL FALLBACK → Precomputed File
    # ================================
    if precomputed_path.exists():
        with open(precomputed_path, "r") as f:
            result = json.load(f)

        result["meta"] = {
            "cache_status": "precomputed",
            "note": "Dataset missing. Loaded precomputed class distribution."
        }

        return JsonResponse(result, json_dumps_params={"indent": 2})

    # ================================
    # 4️⃣ No precomputed file → return error
    # ================================
    return JsonResponse(
        {
            "error": "Dataset files missing and no precomputed class_distribution.json found.",
            "expected_precomputed_path": str(precomputed_path.resolve()),
            "expected_dataset_root": str(dataset_root),
        },
        status=500
    )


from .utils.image_stats import compute_visual_distributions
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def visual_distributions_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    recompute = handle_recompute(request, "visual_distributions", Path(os.getenv("BDD100K_ROOT")))
    train_data = compute_visual_distributions(subset="train")
    val_data = compute_visual_distributions(subset="val")

    response = {
        "train": train_data,
        "val": val_data,
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss" if recompute else "hit",
        },
    }
    return JsonResponse(response, safe=False)


@csrf_exempt
def data_quality_per_class_api(request):
    train_dir = Path(os.getenv("BDD100K_TRAIN_IMAGES"))
    val_dir = Path(os.getenv("BDD100K_VAL_IMAGES"))
    train_ann = Path(os.getenv("BDD100K_TRAIN_JSON"))
    val_ann = Path(os.getenv("BDD100K_VAL_JSON"))

    if not train_dir.exists() or not val_dir.exists():
        return JsonResponse({"error": "Invalid dataset paths"}, status=500)

    recompute = handle_recompute(request, "data_quality_metrics_per_class", Path(os.getenv("BDD100K_ROOT")))
    sample_size = int(request.GET.get("sample_size", "300"))

    train_summary = compute_data_quality_metrics_per_class(train_dir, train_ann, sample_size)
    val_summary = compute_data_quality_metrics_per_class(val_dir, val_ann, sample_size)

    result = {
        "train": train_summary,
        "val": val_summary,
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sample Size": sample_size,
            "cache_status": "miss" if recompute else "hit",
        },
    }
    return JsonResponse(result, safe=False)

    
# @csrf_exempt
# def data_quality_per_class_api(request):
#     """Compute image data quality metrics per class (train + val)."""
#     from .utils.data_quality import compute_data_quality_metrics_per_class

#     # --- Load dataset paths from .env ---
#     train_dir_env = os.getenv("BDD100K_TRAIN_IMAGES")
#     val_dir_env = os.getenv("BDD100K_VAL_IMAGES")
#     train_json = os.getenv("BDD100K_TRAIN_JSON")
#     val_json = os.getenv("BDD100K_VAL_JSON")

#     if not train_dir_env or not val_dir_env:
#         return JsonResponse(
#             {
#                 "error": "Missing environment variables. Please ensure BDD100K_TRAIN_IMAGES and BDD100K_VAL_IMAGES exist in your .env file."
#             },
#             status=500,
#         )

#     # --- Convert to Paths ---
#     train_dir = Path(train_dir_env)
#     val_dir = Path(val_dir_env)
#     train_ann = Path(train_json)
#     val_ann = Path(val_json)

#     # --- Sample size from query param ---
#     sample_size = int(request.GET.get("sample_size", "300"))

#     # --- Compute data quality for train and val ---
#     train_summary = compute_data_quality_metrics_per_class(train_dir, train_ann, sample_size)
#     val_summary = compute_data_quality_metrics_per_class(val_dir, val_ann, sample_size)

#     result = {
#         "train": train_summary,
#         "val": val_summary,
#         "meta": {
#             "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "Sample Size": sample_size,
#         },
#     }

#     return JsonResponse(result, safe=False)



@csrf_exempt
def object_density_api(request):
    recompute = handle_recompute(request, "object_density", Path(os.getenv("BDD100K_ROOT")))

    train_result = compute_object_density(Path(os.getenv("BDD100K_TRAIN_JSON")), Path(os.getenv("BDD100K_TRAIN_IMAGES")))
    val_result = compute_object_density(Path(os.getenv("BDD100K_VAL_JSON")), Path(os.getenv("BDD100K_VAL_IMAGES")))

    return JsonResponse({
        "train": train_result,
        "val": val_result,
        "meta": {
            "Last Computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_status": "miss" if recompute else "hit"
        }
    }, safe=False)


from django.http import JsonResponse
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from .utils.data_loader import load_annotations

load_dotenv()

from django.http import JsonResponse
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import json, os

from .utils.cache_utils import cache_get, cache_set

from .utils.data_loader import load_annotations

load_dotenv()

@csrf_exempt
def class_balance_api(request):
   

        

    return JsonResponse(response, safe=False, json_dumps_params={"indent": 2})

from .utils.class_balance_insights import compute_class_balance_insights
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
from .utils.class_balance_insights import compute_class_balance_insights

@csrf_exempt
def class_balance_insights_api(request):
    recompute = handle_recompute(request, "class_balance_insights", Path(os.getenv("BDD100K_ROOT")))

    train_ann = os.getenv("BDD100K_TRAIN_JSON")
    val_ann = os.getenv("BDD100K_VAL_JSON")

    result = compute_class_balance_insights(train_ann, val_ann)
    result["meta"]["cache_status"] = "miss" if recompute else "hit"

    return JsonResponse(result, safe=False, json_dumps_params={"indent": 2})



@csrf_exempt
def sample_visualizations_api(request):
    from .utils.sample_visualizations import compute_sample_visualizations

    image_dir = Path(os.getenv("BDD100K_TRAIN_IMAGES"))
    ann_json = Path(os.getenv("BDD100K_TRAIN_JSON"))

    recompute = handle_recompute(request, "sample_visualizations", image_dir)

    result = compute_sample_visualizations(image_dir, ann_json, copy_files=True)
    result["meta"]["cache_status"] = "miss" if recompute else "hit"

    return JsonResponse(result, safe=False, json_dumps_params={"indent": 2})