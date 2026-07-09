"""Views for the per-campaign table read path (VIEW-01/02/03/04), the public submission write
path (SUBMIT-01/04/05), and the coverage-gap analysis view (GAP-02).

Views: ``CampaignRunTableView`` (the sortable/paginated/filterable per-campaign table,
PII-gated at the queryset layer per D-13/VIEW-03), ``CampaignListView`` (D-03's campaigns
list page), ``CampaignRunSubmissionView`` (the public intake form), and
``CampaignGapAnalysisView`` (GET-triggered, cached, server-side-validated coverage-gap
analysis). Deliberately does not import ``solsys_code.views`` -- that module imports
``.ephem_utils`` at module load time, which triggers a ~1.6 GB SPICE kernel download (CLAUDE.md
"Heavy import side effect"). ``campaign_gap`` is safe to import at module scope here: it only
depends on ``telescope_runs.sun_event``, never the heavy SPICE-loading ephemeris module.
"""

import logging
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timezone as dt_timezone

from django.contrib import messages
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.db.models import Count, F
from django.http import HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.views.generic import FormView, ListView, TemplateView, View
from django_filters.views import FilterView
from django_tables2 import RequestConfig
from django_tables2.views import SingleTableMixin
from tom_targets.models import TargetList

from solsys_code.solsys_code_observatory.models import Observatory

from .calendar_utils import insert_or_create_calendar_event
from .campaign_filters import CampaignRunFilterSet
from .campaign_forms import CampaignGapAnalysisForm, CampaignRunSubmissionForm
from .campaign_gap import clamp_date_range, get_or_compute_gap
from .campaign_tables import ApprovalQueueTable, CampaignRunTable
from .campaign_utils import resolve_site
from .mixins import StaffRequiredMixin
from .models import CampaignRun
from .telescope_runs import sun_event

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
    'window_start',
    'window_end',
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
        """Restrict to this campaign; non-staff get a PII-safe .values() queryset (D-13).

        D-04: default-sorts resolved rows first (most recent window_start first), TBD
        (window_start is NULL) rows last -- portably across SQLite/PostgreSQL, which
        default to opposite implicit NULL-ordering directions for DESC (RESEARCH.md
        Pattern 4/Anti-Patterns). Applied here (not django-tables2's Meta.order_by,
        which only compiles bare accessor strings) for both the staff and non-staff
        branches.
        """
        campaign_pk = self.kwargs['pk']
        qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
        if self.request.user.is_staff:
            return qs.select_related('site').order_by(F('window_start').desc(nulls_last=True))
        # D-09/SUBMIT-02: non-staff see approved AND rejected runs; only pending_review is
        # hidden. Queryset-level exclude (not a template conditional) so pending rows never
        # enter the non-staff SELECT -- mirrors D-13's existing discipline (T-16-07).
        qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        qs = qs.order_by(F('window_start').desc(nulls_last=True))
        return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)

    def get_table_kwargs(self):
        """Belt-and-suspenders: also drop contact columns from the rendered table (D-13).

        D-04: 'order_by': () suppresses django-tables2's own default sort so it doesn't
        clobber get_queryset()'s nulls-last ordering (mirrors the existing
        decided_table = ApprovalQueueTable(..., order_by=()) precedent in
        ApprovalQueueView below). Interactive column-header sorting (RequestConfig)
        still works normally on top of this.
        """
        if not self.request.user.is_staff:
            return {'exclude': ('contact_person', 'contact_email'), 'order_by': ()}
        return {'order_by': ()}

    def get_context_data(self, **kwargs):
        """Add the campaign (TargetList) and D-14 gap-analysis-button availability to context."""
        context = super().get_context_data(**kwargs)
        context['campaign'] = get_object_or_404(TargetList, pk=self.kwargs['pk'])
        # D-14: reuse gap_analysis_available() (defined below) rather than duplicating its
        # target-count / resolved-site logic here -- gates the "Show Coverage Gaps" button.
        context['gap_analysis_available'] = gap_analysis_available(context['campaign'])
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
                    # SCHED-02: single-night collapse -- the form's one observing-date field
                    # feeds both window_start and window_end.
                    window_start=form.cleaned_data['obs_date'],
                    window_end=form.cleaned_data['obs_date'],
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
            # observing date (a resolved single night) -- or the same campaign+
            # telescope_instrument+contact_person when the date is left blank (TBD) --
            # collide on one of CampaignRun's two partial natural-key UniqueConstraints.
            # Friendly form error, never a 500.
            form.add_error(
                None,
                'A run for this telescope on this observing date already exists for this campaign. '
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
        # WR-03: campaigns:approval_queue is wired up by campaign_urls.py/src/fomo/urls.py in
        # this same shipped changeset, so reverse() always succeeds here -- no NoReverseMatch
        # fallback needed.
        queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
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
        # table: django-tables2's table construction would otherwise re-sort the data, and
        # Django refuses to call .order_by() again on an already-sliced queryset (`Cannot
        # reorder a query once a slice has been taken`). A plain list sidesteps that entirely
        # (django-tables2 sorts lists in Python via TableListData.order_by), and order_by=()
        # below suppresses any default sort so the -pk selection order is preserved on first
        # render (D-04's nulls-last window sort is a CampaignRunTableView-only concern; this
        # queue view intentionally orders by recency, not window_start).
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
                # On approve we resolve the site but never auto-create a placeholder
                # Observatory for unresolvable public free-text (unlike the already-vetted
                # CSV import path) -- the run is still approved with site=None +
                # site_needs_review=True (site failure never blocks approval; D-06's calendar
                # projection below needs a resolved site, so an unresolved site simply means
                # no CalendarEvent yet, not a blocked approval).
                site, needs_review = resolve_site(run.site_raw, create_placeholder=False)
                run.site, run.site_needs_review = site, needs_review
                run.save(update_fields=['site', 'site_needs_review'])

                # D-06/CAL-01: CalendarEvent.start_time/end_time are non-nullable -- only
                # project a single concrete night (window_start == window_end); a resolved
                # site is required to pick the ground-vs-space branch. A range, TBD run, or
                # unresolved site simply doesn't get a CalendarEvent yet.
                if run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end:
                    event_fields = {
                        'title': f'{run.campaign.name}: {run.telescope_instrument}',
                        'description': run.observation_details,
                        'target_list': run.campaign,  # CAL-02
                        'telescope': run.telescope_instrument,
                    }
                    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
                        # Space-based observatory: no fixed horizon for sun_event() to work
                        # against -- use a midnight-UTC placeholder spanning the window date.
                        event_fields['start_time'] = datetime.combine(
                            run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc
                        )
                        event_fields['end_time'] = datetime.combine(
                            run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc
                        )
                        # Never construct CalendarEvent directly -- always route through the
                        # shared helper (Don't Hand-Roll) so the CAMPAIGN: namespace stays
                        # collision-safe against the LCO/Gemini/classical sync commands (T-16-09).
                        insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
                    else:
                        # Ground-based observatory: reuse the same dip-corrected sunset/sunrise
                        # convention the rest of the calendar feature already uses (kind='sun',
                        # not 'dark' -- Pitfall 6). A ValueError (e.g. blank site.timezone, or
                        # no 2 sun-altitude crossings) is logged and skipped, matching
                        # campaign_gap.observable_dates()'s established discipline -- it must
                        # never reach the broad except Exception below, which exists to revert
                        # a half-committed approval, not to handle expected messy site data.
                        try:
                            sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
                        except ValueError:
                            logger.debug(
                                'sun_event(sun) raised for site=%s date=%s; skipping projection.',
                                run.site,
                                run.window_start,
                            )
                        else:
                            event_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(
                                microsecond=0
                            )
                            event_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(
                                microsecond=0
                            )
                            insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
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
        elif CampaignRun.objects.filter(pk=pk).exists():
            # WR-01: the conditional .update() above returns 0 both when the row exists but
            # was already decided, and when pk never existed at all -- distinguish the two so a
            # deleted/stale/tampered pk gets an honest "no longer exists" message instead of the
            # factually-wrong "already decided by someone else".
            messages.warning(request, 'This run was already decided by someone else.')
        else:
            messages.error(request, 'This run no longer exists.')
        return redirect('campaigns:approval_queue')


def gap_analysis_available(campaign) -> bool:
    """D-14: whether coverage-gap analysis makes sense for this campaign.

    False when the campaign has zero ``Target``s, or none of its ``CampaignRun``s have a
    resolved ``site`` at all -- there is nothing to compute observability against either way.
    Reused by Plan 03's ``CampaignRunTableView`` to gate the "Show Coverage Gaps" button
    (disabled + explanatory helper text when unavailable, never a dead clickable button).
    """
    if campaign.targets.count() == 0:
        return False
    return CampaignRun.objects.filter(campaign=campaign, site__isnull=False).exists()


def _as_pk_or_none(raw: str | None) -> int | None:
    """Parse a raw GET-param string as a pk, or None if it isn't a valid integer (CR-01).

    Guards every `target`/`site` pk lookup in `CampaignGapAnalysisView` before it reaches
    `.filter(pk=...)` -- Django's `IntegerField.get_prep_value()` raises a bare `ValueError`
    for a non-integer string, which would otherwise crash the view with an unhandled 500
    instead of the documented `HttpResponseBadRequest` (T-17-01/Pitfall 3).
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class CampaignGapAnalysisView(TemplateView):
    """Coverage-gap analysis page (GAP-02): observable-but-unclaimed dates for a campaign
    target + site, computed on request or served from the 1-hour result cache (D-09/D-10).

    Public/read-only, same posture as ``CampaignRunTableView`` -- no ``StaffRequiredMixin``.
    A plain GET (not htmx) triggers computation, per D-09; the fast per-campaign table view
    never imports this module's computation path inline. Re-derives the campaign's allowed
    target/site sets server-side and validates any submitted ``target``/``site`` pk against
    them before either reaches a query or the cache key -- the campaign-scoped dropdown only
    constrains what's *offered*, never what a raw request can submit
    (``HttpResponseBadRequest`` on mismatch, T-17-01/Pitfall 3).
    """

    template_name = 'campaigns/campaignrun_gap_analysis.html'

    def get(self, request, *args, **kwargs):
        """Resolve campaign/target/site/range server-side, then render the form and any result."""
        campaign = get_object_or_404(TargetList, pk=self.kwargs['pk'])
        available = gap_analysis_available(campaign)
        form = CampaignGapAnalysisForm(request.GET or None, campaign=campaign)
        context = self.get_context_data(campaign=campaign, form=form, gap_analysis_available=available)

        if not available:
            # D-14: nothing to compute -- render the disabled-state page, no computation.
            return self.render_to_response(context)

        # D-12: a single-target campaign auto-uses its sole Target, ignoring any submitted
        # target pk; a multi-target campaign requires one and re-validates it server-side
        # against the campaign's own targets (never trusting the dropdown alone).
        if campaign.targets.count() == 1:
            target = campaign.targets.first()
        else:
            target_pk_raw = request.GET.get('target')
            if not target_pk_raw:
                # No selection submitted yet -- render just the form, no computation.
                return self.render_to_response(context)
            # CR-01: a non-numeric pk (e.g. ?target=abc) must never reach `.filter(pk=...)`
            # un-guarded -- Django's IntegerField.get_prep_value() raises a bare ValueError
            # for a non-integer string, which would otherwise crash this view with an
            # unhandled 500 instead of the documented HttpResponseBadRequest.
            target_pk = _as_pk_or_none(target_pk_raw)
            target = target_pk is not None and campaign.targets.filter(pk=target_pk).first()
            if not target:
                # T-17-01/Pitfall 3 (IDOR): never a raw 400 page -- re-render the selection
                # form with the UI-SPEC's alert-danger copy (17-03-PLAN.md Task 1).
                context['idor_error'] = True
                return self.render_to_response(context, status=400)

        # D-13: re-derive the campaign's allowed site set server-side (same query the form
        # uses) and validate the submitted site pk is a member before using it anywhere.
        site_pk_raw = request.GET.get('site')
        if not site_pk_raw:
            return self.render_to_response(context)
        allowed_sites = Observatory.objects.filter(campaign_runs__campaign=campaign).distinct()
        # CR-01/WR-04: guard the non-numeric-pk case the same way as `target` above, and
        # collapse the `.exists()` + `.get()` pair into a single `.filter(...).first()`
        # query to close the TOCTOU window between the existence check and the fetch.
        site_pk = _as_pk_or_none(site_pk_raw)
        site = site_pk is not None and allowed_sites.filter(pk=site_pk).first()
        if not site:
            # T-17-01/Pitfall 3 (IDOR): same treatment as the out-of-scope target case above.
            context['idor_error'] = True
            return self.render_to_response(context, status=400)

        # D-11/WR-03: use the already-bound, already-validated form's cleaned_data instead
        # of re-parsing raw request.GET by hand -- a form validation failure now renders the
        # form's own errors instead of silently substituting the 90-day default window.
        if not form.is_valid():
            return self.render_to_response(context, status=400)
        requested_end = form.cleaned_data.get('end_date')
        start, end = clamp_date_range(date.today(), requested_end)

        result = get_or_compute_gap(campaign, target, site, start, end)
        context.update({'target': target, 'site': site, 'start': start, 'end': end, 'result': result})
        return self.render_to_response(context)
