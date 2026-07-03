"""Views for the per-campaign table read path (VIEW-01/02/03/04).

Two views: ``CampaignRunTableView`` (the sortable/paginated/filterable per-campaign table,
PII-gated at the queryset layer per D-13/VIEW-03) and ``CampaignListView`` (D-03's campaigns
list page). Deliberately does not import ``solsys_code.views`` -- that module imports
``.ephem_utils`` at module load time, which triggers a ~1.6 GB SPICE kernel download
(CLAUDE.md "Heavy import side effect").
"""

from django.db.models import Count
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from tom_targets.models import TargetList

from .campaign_filters import CampaignRunFilterSet
from .campaign_tables import CampaignRunTable
from .models import CampaignRun

# D-13/VIEW-03/T-15-01: the exact D-09 column list for non-staff requests. Deliberately
# enumerated explicitly (not introspected from CampaignRun._meta) so contact_person/
# contact_email can never accidentally be included -- the SQL SELECT itself never fetches
# them for non-staff, per 15-RESEARCH.md Pitfall 1's "restrict the queryset, not just the
# rendered table" recommendation.
ALLOWED_FIELDS_FOR_NON_STAFF = [
    'pk',
    'telescope_instrument',
    'site__short_name',
    'site_raw',
    'site_needs_review',
    'obs_date',
    'ut_start',
    'ut_end',
    'filters_bandpass',
    'run_status',
    'approval_status',
    'open_to_collaboration',
    'observation_details',
    'weather',
    'observation_outcome',
    'publication_plans',
    'comments',
]


class CampaignRunTableView(SingleTableMixin, FilterView):
    """Sortable/paginated/filterable table of every CampaignRun for one campaign (VIEW-01/04).

    ``SingleTableMixin`` MUST be declared before ``FilterView`` in the class bases -- this MRO
    order is load-bearing (15-RESEARCH.md Pitfall 4): it ensures ``FilterView.get()`` has
    already set ``self.object_list = self.filterset.qs`` before ``SingleTableMixin`` builds the
    table from it. Reversing the order can silently unfilter the table.
    """

    model = CampaignRun
    table_class = CampaignRunTable
    filterset_class = CampaignRunFilterSet
    template_name = 'campaigns/campaignrun_table.html'
    table_pagination = {'per_page': 25}  # D-11

    def get_queryset(self):
        """Restrict to this campaign; non-staff get a PII-safe .values() queryset (D-13)."""
        campaign_pk = self.kwargs['pk']
        qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
        if self.request.user.is_staff:
            return qs.select_related('site')
        return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)

    def get_table_kwargs(self):
        """Belt-and-suspenders: also drop contact columns from the rendered table (D-13)."""
        if not self.request.user.is_staff:
            return {'exclude': ('contact_person', 'contact_email')}
        return {}

    def get_context_data(self, **kwargs):
        """Add the campaign (TargetList) to context for the page heading."""
        context = super().get_context_data(**kwargs)
        context['campaign'] = get_object_or_404(TargetList, pk=self.kwargs['pk'])
        return context


class CampaignListView(ListView):
    """Lists every TargetList that has >= 1 CampaignRun, each linking to its table (D-03).

    ``TargetList`` has no "is this a campaign" flag -- "campaign" is purely operational: a
    TargetList with campaign_runs__isnull=False (15-RESEARCH.md Pitfall 3). Never uses
    TargetList.objects.all(), which would include unrelated saved searches/groupings.
    """

    queryset = (
        TargetList.objects.filter(campaign_runs__isnull=False).distinct().annotate(run_count=Count('campaign_runs'))
    )
    template_name = 'campaigns/campaign_list.html'
    context_object_name = 'campaigns'
