"""URL configuration for the FOMO project."""

from django.urls import include, path

from solsys_code.scout_views import (
    RubinTooScoutListView,
    RubinTooScoutStatsView,
    ScoutTargetExportView,
    ScoutTargetListView,
)
from solsys_code.views import Ephemeris, MakeEphemerisView, ProtectedUserDeleteView

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
    # DISPLAY-09 — must precede tom_common.urls: tomtoolkit 3.0 now also registers a 'calendar'
    # namespace there (tom_calendar.urls) with identical URL names, which triggers Django's
    # urls.W005 "namespace isn't unique" check warning. That's expected and benign — this entry,
    # being first, is what `calendar:*` reversal and dispatch actually resolve to; the tom_common
    # one is fully shadowed. See solsys_code/calendar_urls.py's module docstring.
    path('calendar/', include('solsys_code.calendar_urls', namespace='calendar')),
    path('campaigns/', include('solsys_code.campaign_urls', namespace='campaigns')),  # VIEW-01 — before tom_common
    # WR-10 (37.1-REVIEW.md): must precede tom_common.urls -- TOM's own 'user-delete' view 500s on an
    # account that confirmed a campaign link or calendar event attribution (their confirmed_by is PROTECT).
    path('users/<int:pk>/delete/', ProtectedUserDeleteView.as_view(), name='user-delete'),
    path('', include('tom_common.urls')),
]
