"""bdd100k_utk URL Configuration"""

from django.contrib import admin
from django.urls import path, include  

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("bdd_analysis.urls")),  
]
