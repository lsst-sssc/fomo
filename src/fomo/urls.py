"""django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

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
    # tomtoolkit 3.0.0 final dropped the 'alerts/' include from tom_common.urls (previously
    # registered there in 2.x/3.0.0a9) -- tom_alerts is still an installed app, so its urls
    # must now be wired up at the project level to keep the 'alerts' namespace resolvable.
    path('alerts/', include('tom_alerts.urls', namespace='alerts')),
    path('', include('tom_common.urls')),
]
