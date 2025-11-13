from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_root, name="api_root"),  # 👈 add this first
    path(
        "api/analysis/dataset-summary/",
        views.dataset_summary_api,
        name="dataset_summary_api",
    ),
    path(
        "api/analysis/class-distribution/",
        views.class_distribution_api,
        name="class_distribution_api",
    ),
    path(
        "api/analysis/visual-distributions/",
        views.visual_distributions_api,
        name="visual_distributions_api",
    ),
    path(
    "api/analysis/data-quality-per-class/",
    views.data_quality_per_class_api,
    name="data_quality_per_class_api",
),

path(
    "api/analysis/object-density/",
    views.object_density_api,
    name="object_density_api",
),
path(
    "api/analysis/class-balance/",
    views.class_balance_api,
    name="class_balance_api",
),
path(
    "api/analysis/class-balance-insights/",
    views.class_balance_insights_api,
    name="class_balance_insights_api",
),
path(
    "api/analysis/sample-visualizations/",
    views.sample_visualizations_api,
    name="sample_visualizations_api",
),


]
