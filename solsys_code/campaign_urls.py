"""FOMO campaigns URL conf -- the per-campaign table read path (VIEW-01/03/04).

Mirrors solsys_code/calendar_urls.py's structure: app_name + a flat urlpatterns list.
"""

from django.urls import path
from django.views.generic import TemplateView

from solsys_code.campaign_views import (
    ApprovalQueueView,
    AttributionDecisionView,
    AttributionQueueView,
    CampaignGapAnalysisView,
    CampaignListView,
    CampaignRunDecisionView,
    CampaignRunSubmissionView,
    CampaignRunTableView,
    SiteSearchView,
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
    # ATTRIB-01/D-02: deliberately NOT <int:pk>/-prefixed like campaigns:decide -- an
    # attribution action names a PAIR (an orphan of one of two kinds, and a run), so both
    # identifiers travel in the POST body where AttributionDecisionView re-validates them
    # together, mirroring CampaignRunDecisionView's single-dispatching-view-with-an-`action`-
    # POST-param shape rather than one URL per action.
    path('attribution/', AttributionQueueView.as_view(), name='attribution'),
    path('attribution/decide/', AttributionDecisionView.as_view(), name='attribution_decide'),
    path('site-search/', SiteSearchView.as_view(), name='site_search'),
    path('<int:pk>/decide/', CampaignRunDecisionView.as_view(), name='decide'),
    path('<int:pk>/gaps/', CampaignGapAnalysisView.as_view(), name='gap_analysis'),
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
