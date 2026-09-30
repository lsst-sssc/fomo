"""URL configuration for the FOMO project."""

from django.urls import include, path

from solsys_code.scout_views import (
    RubinTooScoutListView,
    RubinTooScoutStatsView,
    ScoutTargetExportView,
    ScoutTargetListView,
)
from solsys_code.views import Ephemeris, MakeEphemerisView

urlpatterns = [
    # Shadow the 'targets/' list and export routes inside tom_common.urls (included below).
    # The stock target_list template still reverses `tom_targets:list` / `targets:export` to
    # these same paths, and the earlier pattern wins on resolution, so the HTMX table
    # refreshes and the "Export Filtered Targets" button both reach the Scout-aware views.
    path('targets/', ScoutTargetListView.as_view(), name='scout_target_list'),
    path('targets/export/', ScoutTargetExportView.as_view(), name='scout_target_export'),
    path('observatory/', include('solsys_code.solsys_code_observatory.urls', namespace='solsys_code_observatory')),
    path('ephem/<int:pk>/', Ephemeris.as_view(), name='ephem'),
    path('targets/<int:pk>/makeephem/', MakeEphemerisView.as_view(), name='makeephem'),
    path('scout/rubin-too/', RubinTooScoutListView.as_view(), name='scout_rubin_too'),
    path('scout/rubin-too/stats/', RubinTooScoutStatsView.as_view(), name='scout_rubin_too_stats'),
    path('', include('tom_common.urls')),
]
