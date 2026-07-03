"""FOMO campaigns URL conf -- the per-campaign table read path (VIEW-01/03/04).

Mirrors solsys_code/calendar_urls.py's structure: app_name + a flat urlpatterns list.
"""

from django.urls import path

from solsys_code.campaign_views import CampaignListView, CampaignRunTableView

app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
