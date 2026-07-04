"""Views for the per-campaign table read path (VIEW-01/02/03/04) and the public submission
write path (SUBMIT-01/04/05).

Views: ``CampaignRunTableView`` (the sortable/paginated/filterable per-campaign table,
PII-gated at the queryset layer per D-13/VIEW-03), ``CampaignListView`` (D-03's campaigns
list page), and ``CampaignRunSubmissionView`` (the public intake form). Deliberately does not
import ``solsys_code.views`` -- that module imports ``.ephem_utils`` at module load time,
which triggers a ~1.6 GB SPICE kernel download (CLAUDE.md "Heavy import side effect").
"""

from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.db import IntegrityError
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect
from django.urls import NoReverseMatch, reverse, reverse_lazy
from django.views.generic import FormView, ListView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from tom_targets.models import TargetList

from .campaign_filters import CampaignRunFilterSet
from .campaign_forms import CampaignRunSubmissionForm
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


class CampaignRunSubmissionView(FormView):
    """Public intake form for a single observing run, pending staff review (SUBMIT-01/04/05).

    A honeypot trip (``alt_contact_info`` populated) short-circuits to the exact same
    thanks-page redirect as a genuine submission -- no ``CampaignRun`` created, no email sent,
    no error signal (SUBMIT-04, 16-RESEARCH.md Pattern 3). A natural-key collision on
    ``.objects.create()`` (Pitfall 4) degrades to a friendly non-field form error, never a 500.
    """

    form_class = CampaignRunSubmissionForm
    template_name = 'campaigns/campaignrun_submit_form.html'
    success_url = reverse_lazy('campaigns:submission_thanks')

    def form_valid(self, form):
        """Create the CampaignRun (or silently drop a honeypot trip) and notify staff."""
        if form.cleaned_data.get('alt_contact_info'):
            # SUBMIT-04: bot tripped the honeypot -- fall straight through to the same success
            # redirect as a genuine submission. No create, no email, no error signal.
            return redirect('campaigns:submission_thanks')
        try:
            run = CampaignRun.objects.create(
                campaign=form.cleaned_data['campaign'],
                telescope_instrument=form.cleaned_data['telescope_instrument'],
                site_raw=form.cleaned_data['site_raw'],
                obs_date=form.cleaned_data['obs_date'],
                ut_start=form.cleaned_data['ut_start'],
                ut_end=form.cleaned_data['ut_end'],
                filters_bandpass=form.cleaned_data['filters_bandpass'],
                observation_details=form.cleaned_data['observation_details'],
                open_to_collaboration=form.cleaned_data['open_to_collaboration'],
                contact_person=form.cleaned_data['contact_person'],
                contact_email=form.cleaned_data['contact_email'],
                comments=form.cleaned_data['comments'],
                # approval_status intentionally not set -- model default is PENDING_REVIEW.
                # site/site_needs_review intentionally not set -- resolved at approval time (D-07).
            )
        except IntegrityError:
            # Pitfall 4: two submitters proposing the same campaign+telescope_instrument+
            # ut_start collide on CampaignRun's natural-key UniqueConstraint. Friendly form
            # error, never a 500.
            form.add_error(
                None,
                'A run for this telescope at this start time already exists for this campaign. '
                'Check the campaign table, or contact a coordinator if you believe this is a mistake.',
            )
            return self.form_invalid(form)
        self._notify_staff(run)
        return redirect('campaigns:submission_thanks')

    def _notify_staff(self, run):
        """Email every staff user with an email on file that a submission is pending (SUBMIT-05).

        Body/subject intentionally carry no PII (D-04) -- a bare ping plus the approval-queue
        link, nothing about the submitter, telescope, or campaign.
        """
        recipients = list(User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True))
        if not recipients:
            return  # no staff with an email on file -- nothing to notify, not an error
        try:
            queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
        except NoReverseMatch:
            # campaigns:approval_queue is added by Plan 03 (Wave 3), which has not landed yet
            # at this plan's point in the phase's sequential wave order. Fall back to the exact
            # path Plan 03 wires it to (src/fomo/urls.py mounts this app at 'campaigns/') so the
            # notification link is still correct; once Plan 03 lands, reverse() above succeeds
            # and this branch is dead code.
            queue_url = self.request.build_absolute_uri('/campaigns/approval-queue/')
        send_mail(
            subject='FOMO: new campaign run submission pending review',
            message=f'A new run submission is pending review: {queue_url}',
            from_email=None,
            recipient_list=recipients,
            fail_silently=True,  # Pitfall 6: a mail outage must never break the submission
        )
