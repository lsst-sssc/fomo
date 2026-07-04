"""FOMO campaigns URL conf -- the per-campaign table read path (VIEW-01/03/04).

Mirrors solsys_code/calendar_urls.py's structure: app_name + a flat urlpatterns list.
"""

from django.urls import path
from django.views.generic import TemplateView

from solsys_code.campaign_views import (
    ApprovalQueueView,
    CampaignGapAnalysisView,
    CampaignListView,
    CampaignRunDecisionView,
    CampaignRunSubmissionView,
    CampaignRunTableView,
)

app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    path('submit/', CampaignRunSubmissionView.as_view(), name='submit'),
    path(
        'submission-thanks/',
        TemplateView.as_view(template_name='campaigns/submission_thanks.html'),
        name='submission_thanks',
    ),
    path('approval-queue/', ApprovalQueueView.as_view(), name='approval_queue'),
    path('<int:pk>/decide/', CampaignRunDecisionView.as_view(), name='decide'),
    path('<int:pk>/gaps/', CampaignGapAnalysisView.as_view(), name='gap_analysis'),
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
