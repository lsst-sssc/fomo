"""Views for the per-campaign table read path (VIEW-01/02/03/04) and the public submission
write path (SUBMIT-01/04/05).

Views: ``CampaignRunTableView`` (the sortable/paginated/filterable per-campaign table,
PII-gated at the queryset layer per D-13/VIEW-03), ``CampaignListView`` (D-03's campaigns
list page), and ``CampaignRunSubmissionView`` (the public intake form). Deliberately does not
import ``solsys_code.views`` -- that module imports ``.ephem_utils`` at module load time,
which triggers a ~1.6 GB SPICE kernel download (CLAUDE.md "Heavy import side effect").
"""

import logging

from django.contrib import messages
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.db.models import Count
from django.http import HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect
from django.urls import NoReverseMatch, reverse, reverse_lazy
from django.views.generic import FormView, ListView, TemplateView, View
from django_filters.views import FilterView
from django_tables2 import RequestConfig
from django_tables2.views import SingleTableMixin
from tom_targets.models import TargetList

from .calendar_utils import insert_or_create_calendar_event
from .campaign_filters import CampaignRunFilterSet
from .campaign_forms import CampaignRunSubmissionForm
from .campaign_tables import ApprovalQueueTable, CampaignRunTable
from .campaign_utils import resolve_site
from .mixins import StaffRequiredMixin
from .models import CampaignRun

logger = logging.getLogger(__name__)

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
        # D-09/SUBMIT-02: non-staff see approved AND rejected runs; only pending_review is
        # hidden. Queryset-level exclude (not a template conditional) so pending rows never
        # enter the non-staff SELECT -- mirrors D-13's existing discipline (T-16-07).
        qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
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

    def get_context_data(self, **kwargs):
        """Add pending_count for the staff-only "N pending review" banner (D-01).

        Computed unconditionally -- the template gates its display on request.user.is_staff,
        so it's harmless to compute for anonymous/non-staff visitors too (D-10: list
        membership itself is unchanged).
        """
        context = super().get_context_data(**kwargs)
        context['pending_count'] = CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).count()
        return context


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
            # Wrapped in its own atomic block (savepoint): without it, the IntegrityError
            # caught below poisons the outer request/test transaction, and any subsequent
            # query (e.g. re-rendering the form's ModelChoiceField) raises
            # TransactionManagementError instead of the intended friendly form error.
            with transaction.atomic():
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
                    # site/site_needs_review intentionally not set -- resolved at approval
                    # time (D-07).
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


class ApprovalQueueView(StaffRequiredMixin, TemplateView):
    """Staff-only two-section approval queue: pending review + recently decided (D-01/D-02).

    Two independent ``ApprovalQueueTable`` instances are built by hand from two separate
    querysets (16-RESEARCH.md Pattern 5) rather than routed through ``MultiTableMixin``, since
    the pending/decided querysets have genuinely asymmetric filtering (not a list of symmetric
    tables). ``StaffRequiredMixin`` gates the whole view -- this page must NOT follow Phase 15's
    soft-filter (``.values()``) pattern; anonymous/non-staff requests are redirected before any
    pending-submission content (which includes contact PII) is ever rendered (T-16-03).
    """

    template_name = 'campaigns/approval_queue.html'

    def get_context_data(self, **kwargs):
        """Build the pending (actionable) and recently-decided (read-only) tables."""
        context = super().get_context_data(**kwargs)
        pending_qs = CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).select_related('campaign', 'site')
        # Pitfall 1: CampaignRun has no modified/timestamp field -- order by -pk (a reasonable
        # recency proxy) and cap at 20 rows. Materialized to a list before handing it to the
        # table: django-tables2 applies its Meta.order_by (inherited '-obs_date' from
        # CampaignRunTable) by re-sorting the table's data on construction, and Django refuses
        # to call .order_by() again on an already-sliced queryset (`Cannot reorder a query once
        # a slice has been taken`). A plain list sidesteps that entirely (django-tables2 sorts
        # lists in Python via TableListData.order_by), and order_by=() below suppresses the
        # inherited default sort so the -pk selection order is preserved on first render.
        decided_qs = (
            CampaignRun.objects.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
            .select_related('campaign', 'site')
            .order_by('-pk')[:20]
        )
        pending_table = ApprovalQueueTable(
            pending_qs,
            prefix='pending-',
            request=self.request,
            empty_text='No submissions waiting for review.',
        )
        decided_table = ApprovalQueueTable(
            list(decided_qs),
            prefix='decided-',
            show_actions=False,
            empty_text='No decisions recorded yet.',
            order_by=(),
        )
        RequestConfig(self.request).configure(pending_table)
        RequestConfig(self.request).configure(decided_table)
        context['pending_table'] = pending_table
        context['decided_table'] = decided_table
        return context


class CampaignRunDecisionView(StaffRequiredMixin, View):
    """POST-only atomic approve/reject decision endpoint (SUBMIT-03) + calendar projection.

    A single conditional ``.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)`` proves
    the double-approve no-op (T-16-02): a second decision POST on an already-decided row
    matches zero rows, so the calendar projection below is never re-triggered (CAL-03).
    ``http_method_names = ['post']`` ensures a GET (crawler prefetch, bare ``<a href>``) can
    never trigger a state change (T-16-06).
    """

    http_method_names = ['post']

    def post(self, request, pk):
        """Atomically transition a CampaignRun and, on approve, project a CalendarEvent."""
        action = request.POST.get('action')
        if action not in ('approve', 'reject'):
            return HttpResponseBadRequest()
        new_status = CampaignRun.ApprovalStatus.APPROVED if action == 'approve' else CampaignRun.ApprovalStatus.REJECTED
        updated_count = CampaignRun.objects.filter(
            pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).update(approval_status=new_status)

        if updated_count == 1 and action == 'approve':
            try:
                run = CampaignRun.objects.get(pk=pk)
                # D-07: reuse the existing 3-tier site resolver rather than re-implementing it.
                site, needs_review = resolve_site(run.site_raw)
                run.site, run.site_needs_review = site, needs_review
                run.save(update_fields=['site', 'site_needs_review'])

                # CAL-01/Pitfall 2: CalendarEvent.start_time/end_time are non-nullable -- only
                # project when telescope_instrument, ut_start, AND ut_end are all present. A run
                # missing ut_end simply doesn't get a CalendarEvent yet.
                if run.telescope_instrument and run.ut_start and run.ut_end:
                    # Never construct CalendarEvent directly -- always route through the shared
                    # helper (Don't Hand-Roll) so the CAMPAIGN: namespace stays collision-safe
                    # against the LCO/Gemini/classical sync commands (T-16-09).
                    insert_or_create_calendar_event(
                        {'url': f'CAMPAIGN:{run.pk}'},
                        fields={
                            'title': f'{run.campaign.name}: {run.telescope_instrument}',
                            'description': run.observation_details,
                            'start_time': run.ut_start,
                            'end_time': run.ut_end,
                            'target_list': run.campaign,  # CAL-02
                            'telescope': run.telescope_instrument,
                        },
                    )
            except Exception:
                # CR-01: the conditional .update() above is its own auto-committed statement,
                # so the APPROVED transition has already landed. If site resolution (a network
                # call to the MPC Obscodes API) or calendar projection then fails, revert
                # approval_status back to PENDING_REVIEW so the run is never left permanently
                # "approved" with no CalendarEvent and no way to re-decide it -- without this,
                # the double-approve guard above makes that half-approved state unrecoverable
                # through the UI.
                logger.exception('Approve side-effects failed for CampaignRun %s; reverted to pending review.', pk)
                CampaignRun.objects.filter(pk=pk).update(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
                messages.error(
                    request,
                    'Approval failed while resolving the site or projecting the calendar event. '
                    'This run has been reset to pending review -- please try again.',
                )
                return redirect('campaigns:approval_queue')
            messages.success(request, 'Run approved.')
        elif updated_count == 1:
            messages.success(request, 'Run rejected.')
        else:
            messages.warning(request, 'This run was already decided by someone else.')
        return redirect('campaigns:approval_queue')
