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
import re
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timezone as dt_timezone

from django.contrib import messages
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.db.models import Case, CharField, Count, EmailField, F, Value, When
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
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
from .campaign_utils import (
    _check_and_increment_throttle,
    build_site_candidates,
    is_placeholder_observatory,
    resolve_site,
    substring_or_fuzzy_match_candidates,
)
from .mixins import StaffRequiredMixin
from .models import CampaignRun
from .telescope_runs import sun_event

logger = logging.getLogger(__name__)

# 22-REVIEWS.md finding 2: a conservative DOM-id allowlist for the `input_id` GET param
# echoed back into SiteSearchView's rendered fragment. HTML auto-escaping alone is NOT
# sufficient for a value embedded in the inline `onclick=` JS-string context -- browsers
# decode HTML entities before the JS parser runs, so an HTML-escaped quote still
# terminates the JS string. A non-matching value is replaced server-side with the
# default 'id_site_raw' before it ever reaches the template context (belt-and-suspenders
# on top of the template's own `|escapejs` filter).
_INPUT_ID_RE = re.compile(r'^[-A-Za-z0-9_:.]+$')

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
        # VIEW-05/T-21-02: gate contact_person/contact_email at the SQL SELECT via a per-row
        # Case/When annotation keyed on the submitter's own opt-in flag -- an opted-out row's
        # real contact values are never fetched, only an empty string. ALLOWED_FIELDS_FOR_NON_STAFF
        # itself deliberately does NOT list contact_person/contact_email (RESEARCH.md
        # Anti-Pattern); they arrive only via this annotation.
        #
        # .values() MUST be called before .annotate() here (not after, despite that reading
        # more naturally): Django's annotate() rejects an alias that collides with a real model
        # field name ("The annotation 'contact_person' conflicts with a field on the model"),
        # and that check is against the model's full field list unless .values() has already
        # narrowed QuerySet._fields -- calling .values() first (without contact_person/
        # contact_email in the field list) makes the alias check pass. contact_public_opt_in
        # itself doesn't need to be in the .values() field list for the When() condition below
        # to reference it -- Django resolves F()/condition expressions against the underlying
        # column regardless of the projected .values() field list.
        qs = qs.values(*[f for f in ALLOWED_FIELDS_FOR_NON_STAFF if f not in ('contact_person', 'contact_email')])
        return qs.annotate(
            contact_person=Case(
                When(contact_public_opt_in=True, then=F('contact_person')),
                default=Value(''),
                output_field=CharField(),
            ),
            contact_email=Case(
                When(contact_public_opt_in=True, then=F('contact_email')),
                default=Value(''),
                output_field=EmailField(),
            ),
        )

    def get_table_kwargs(self):
        """D-04: 'order_by': () suppresses django-tables2's own default sort so it doesn't
        clobber get_queryset()'s nulls-last ordering (mirrors the existing
        decided_table = ApprovalQueueTable(..., order_by=()) precedent in
        ApprovalQueueView below). Interactive column-header sorting (RequestConfig)
        still works normally on top of this.

        VIEW-05: contact_person/contact_email are no longer excluded for non-staff -- they're
        always safe to render now (blank string for opted-out rows, populated for opted-in
        ones), gated at the SQL SELECT by get_queryset()'s Case/When annotation, not here.
        """
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
                    # SCHED-02: window_start/window_end come from the form's clean(), which
                    # runs the free-text obs_date through parse_obs_window() -- single-night
                    # collapse (start == end) for one date or an equal-endpoint range, a real
                    # start..end span for a multi-night range, and both None for a blank
                    # (TBD) submission.
                    window_start=form.cleaned_data['window_start'],
                    window_end=form.cleaned_data['window_end'],
                    filters_bandpass=form.cleaned_data['filters_bandpass'],
                    observation_details=form.cleaned_data['observation_details'],
                    open_to_collaboration=form.cleaned_data['open_to_collaboration'],
                    contact_person=form.cleaned_data['contact_person'],
                    contact_email=form.cleaned_data['contact_email'],
                    contact_public_opt_in=form.cleaned_data['contact_public_opt_in'],
                    comments=form.cleaned_data['comments'],
                    # approval_status intentionally not set -- model default is PENDING_REVIEW.
                    # site/site_needs_review intentionally not set -- resolved at approval
                    # time (D-07).
                )
        except IntegrityError:
            # Pitfall 4: two submitters proposing the same campaign+telescope_instrument+
            # resolved window (a single night OR an identical range) -- or the same
            # campaign+telescope_instrument+contact_person when the date is left blank
            # (TBD) -- collide on one of CampaignRun's two partial natural-key
            # UniqueConstraints. Friendly form error, never a 500. Requirement 7 (see
            # 260714-ilz-SUMMARY.md): this handler already covers the range case unchanged;
            # only the wording below was broadened to read correctly for a window as well
            # as a single date.
            form.add_error(
                None,
                'A run for this telescope for this observing window already exists for this campaign. '
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
        # SITE-01/Pitfall 5: build the merged local+MPC candidate pool exactly once per
        # request (never per row) -- build_site_candidates() is itself 24h-cached, but
        # calling it once here still avoids a per-row cache.get() round-trip. Never
        # raises (Plan 21-01's local-only fallback), so no try/except is needed here.
        candidate_pool = build_site_candidates()
        pending_table = ApprovalQueueTable(
            pending_qs,
            prefix='pending-',
            request=self.request,
            candidate_pool=candidate_pool,
            empty_text='No submissions waiting for review.',
        )
        decided_table = ApprovalQueueTable(
            list(decided_qs),
            prefix='decided-',
            show_actions=False,
            empty_text='No decisions recorded yet.',
            order_by=(),
        )
        # D-07: approved runs whose site never resolved -- the "dead end" this phase closes.
        # Deliberately NO row cap (unlike decided_qs's [:20] audit-log cap): this is a live
        # work queue of items genuinely needing staff action, and capping it would hide
        # actionable rows. Naturally includes the projection-failed retry state (site set,
        # flag still True) since the filter is on site_needs_review alone.
        review_qs = (
            CampaignRun.objects.filter(approval_status=CampaignRun.ApprovalStatus.APPROVED, site_needs_review=True)
            .select_related('campaign', 'site')
            .order_by('-pk')
        )
        review_table = ApprovalQueueTable(
            list(review_qs),
            prefix='review-',
            request=self.request,
            # Pitfall 5: reuse the SAME candidate_pool already computed above for
            # pending_table -- never call build_site_candidates() a second time per request.
            candidate_pool=candidate_pool,
            mode='resolve',
            empty_text='No sites currently need review.',
            order_by=(),
        )
        RequestConfig(self.request).configure(pending_table)
        RequestConfig(self.request).configure(decided_table)
        RequestConfig(self.request).configure(review_table)
        context['pending_table'] = pending_table
        context['decided_table'] = decided_table
        context['review_table'] = review_table
        return context


def _project_calendar_event(run: CampaignRun) -> bool:
    """CAL-01/CAL-02 CalendarEvent projection (D-08), extracted from the approve branch.

    Returns True when ``insert_or_create_calendar_event()`` was actually called (an event was
    created/updated), False when projection was skipped by design (range/TBD run, or missing
    telescope_instrument/site) -- 22-REVIEWS.md finding 6: this bool drives the resolve_site
    action's two distinct success messages. RAISES ValueError when ``sun_event()`` fails (e.g.
    a Tier-2-resolved site with a blank ``timezone`` -- CR-01), and MAY RAISE on any other
    unexpected failure (e.g. ``insert_or_create_calendar_event()`` itself failing) -- this
    helper does NO error-handling of its own for genuine failures; callers own
    revert-vs-non-revert behavior. ``resolve_site()`` must treat any raise here as "projection
    attempted but failed" (keep ``site_needs_review=True``, warn instead of claiming success);
    ``approve()`` has no retry surface to protect and instead catches-and-swallows the
    ValueError case specifically at its call site to preserve its original behavior (approval
    still succeeds even when the calendar entry couldn't be projected).
    """
    # D-06/CAL-01: CalendarEvent.start_time/end_time are non-nullable -- only project a
    # single concrete night (window_start == window_end); a resolved site is required to
    # pick the ground-vs-space branch. A range, TBD run, or unresolved site simply doesn't
    # get a CalendarEvent yet.
    if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
        return False
    event_fields = {
        'title': f'{run.campaign.name}: {run.telescope_instrument}',
        'description': run.observation_details,
        'target_list': run.campaign,  # CAL-02
        'telescope': run.telescope_instrument,
    }
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        # Space-based observatory: no fixed horizon for sun_event() to work against -- use
        # a midnight-UTC placeholder spanning the window date.
        event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
        event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
        # Never construct CalendarEvent directly -- always route through the shared helper
        # (Don't Hand-Roll) so the CAMPAIGN: namespace stays collision-safe against the
        # LCO/Gemini/classical sync commands (T-16-09).
        insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
        return True
    # Ground-based observatory: reuse the same dip-corrected sunset/sunrise convention the
    # rest of the calendar feature already uses (kind='sun', not 'dark' -- Pitfall 6). A
    # ValueError (e.g. blank site.timezone, or no 2 sun-altitude crossings) is logged and
    # re-raised (CR-01) -- callers decide whether that's a by-design skip (approve()) or a
    # real failure that must keep the retry surface open (resolve_site()).
    #
    # IN-02 (19-REVIEW.md): this branch also catches OCCULTATION_OBSTYPE and RADAR_OBSTYPE
    # sites, not just OPTICAL_OBSTYPE -- every non-SATELLITE Observatory.OBSTYPE_CHOICES
    # member unconditionally gets the dip-corrected dark-window treatment. That's a
    # deliberate simplification for this milestone; scope this to Observatory.OPTICAL_OBSTYPE
    # explicitly, with OCCULTATION/RADAR falling back to no projection, when those site types
    # get real support.
    try:
        sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
    except ValueError:
        logger.debug(
            'sun_event(sun) raised for site=%s date=%s; re-raising so callers that need the '
            'retry guarantee (resolve_site) see this as a failure, not a by-design skip.',
            run.site,
            run.window_start,
        )
        raise  # CR-01: never silently swallow this -- see docstring above.
    event_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    event_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
    return True


class CampaignRunDecisionView(StaffRequiredMixin, View):
    """POST-only atomic approve/reject decision endpoint (SUBMIT-03) + calendar projection,
    plus the resolve_site action (D-08) that resolves an approved run's still-unmatched site
    and retroactively projects the calendar event approval skipped.

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
        if action not in ('approve', 'reject', 'resolve_site'):
            return HttpResponseBadRequest()
        if action == 'resolve_site':
            return self._resolve_site(request, pk)
        new_status = CampaignRun.ApprovalStatus.APPROVED if action == 'approve' else CampaignRun.ApprovalStatus.REJECTED
        updated_count = CampaignRun.objects.filter(
            pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).update(approval_status=new_status)

        if updated_count == 1 and action == 'approve':
            try:
                run = CampaignRun.objects.get(pk=pk)
                # D-06: only resolve the site once. An already-resolved run.site (from CSV
                # import, tier 1/2 auto-resolution, or a prior staff-UI resolution) must never
                # be re-resolved on a later approve -- e.g. after the except Exception revert
                # below reverts approval_status back to PENDING_REVIEW while leaving run.site
                # set, a second approve POST would otherwise re-hit resolve_site()
                # unconditionally (RESEARCH.md Pitfall 3, the live clobbering bug this closes).
                # A satellite-type site_selection (250/274/289) still falls through to
                # (None, True) via resolve_site()'s to_observatory() TypeError path -- expected,
                # pre-existing behavior, not a Phase 21 regression (RESEARCH.md Pitfall 4).
                # WR-01 (22-REVIEW.md re-review): mirrors _resolve_site()'s placeholder-aware
                # guard below -- a run whose site is already a tier-3 placeholder (e.g. from
                # CSV import) is not a genuine resolution either, so it must still re-enter
                # resolution here, not only when site is None.
                if run.site is None or is_placeholder_observatory(run.site):
                    # D-07: reuse the existing 3-tier site resolver rather than
                    # re-implementing it. SITE-02: prefer the staff-submitted site_selection
                    # (Plan 21-03's inline input) over the originally-submitted site_raw,
                    # falling back to site_raw when blank. On approve we resolve the site but
                    # never auto-create a placeholder Observatory for unresolvable public free
                    # text (unlike the already-vetted CSV import path) -- the run is still
                    # approved with site=None + site_needs_review=True (site failure never
                    # blocks approval; the calendar projection below needs a resolved site, so
                    # an unresolved site simply means no CalendarEvent yet, not a blocked
                    # approval).
                    selection = request.POST.get('site_selection', '').strip() or run.site_raw
                    # CR-01: the datalist offered on the row (ApprovalQueueTable.render_site)
                    # lists MPC-sourced display strings (name_utf8/short_name/old_names), not
                    # obscodes -- resolve the submitted text back to its obscode via the same
                    # candidate pool the datalist was built from before calling resolve_site(),
                    # which otherwise treats its argument as a literal obscode. An exact match
                    # in the pool (e.g. the staff member picked/typed a datalist option
                    # verbatim) maps to its obscode; anything else (a value that was never a
                    # candidate, including a genuinely-typed obscode) passes through unchanged.
                    obscode_selection = build_site_candidates().get(selection, selection)
                    site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
                    run.site, run.site_needs_review = site, needs_review
                    run.save(update_fields=['site', 'site_needs_review'])

                # Projection extracted into the shared _project_calendar_event() helper
                # (22-REVIEWS.md finding 6); the approve branch ignores its bool return.
                # CR-01: _project_calendar_event() now raises ValueError when sun_event()
                # fails (e.g. a Tier-2-resolved site with a blank timezone) so resolve_site()
                # can treat it as a real failure. approve() has no retry surface to protect
                # (unlike resolve_site()'s "Sites Needing Review" row), so it swallows
                # specifically this expected-failure-mode ValueError here to preserve its
                # original behavior: the approval still succeeds without a CalendarEvent.
                # Anything else _project_calendar_event() raises (e.g.
                # insert_or_create_calendar_event() itself failing) is a genuine unexpected
                # failure and still falls through to the broader except Exception below,
                # which reverts the approval.
                try:
                    _project_calendar_event(run)
                except ValueError:
                    logger.debug(
                        'Calendar projection skipped for CampaignRun %s on approve '
                        '(sun_event ValueError, e.g. blank site timezone).',
                        pk,
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
        elif CampaignRun.objects.filter(pk=pk).exists():
            # WR-01: the conditional .update() above returns 0 both when the row exists but
            # was already decided, and when pk never existed at all -- distinguish the two so a
            # deleted/stale/tampered pk gets an honest "no longer exists" message instead of the
            # factually-wrong "already decided by someone else".
            messages.warning(request, 'This run was already decided by someone else.')
        else:
            messages.error(request, 'This run no longer exists.')
        return redirect('campaigns:approval_queue')

    def _resolve_site(self, request, pk):
        """D-07/D-08: resolve an approved run's still-unmatched site, then retroactively
        project the CalendarEvent approval skipped.

        Ordering is deliberately load-bearing (22-REVIEWS.md findings 3/5/6/8c):
        ``site_needs_review`` is cleared ONLY after ``_project_calendar_event()`` returns
        without raising -- never before, and never on a projection failure -- so a failed
        projection leaves the run visible in the Sites Needing Review table (its retry
        surface) instead of vanishing into a dead end. The site write itself is a single
        conditional queryset update (not a plain re-fetch + in-Python check) so two racing
        staff POSTs cannot both claim the write.

        22-06 gap closure (UAT gap 2B): a tier-3 PLACEHOLDER site (``resolve_site()``'s
        ``create_placeholder`` fallback -- name prefixed ``NEEDS REVIEW: ``) is not a
        genuine resolution, so it's also eligible for replacement here, alongside the
        site=None case -- see ``is_placeholder_observatory()`` below. A genuinely-resolved
        (non-placeholder) site is still never re-resolved (D-06).
        """
        # Pitfall 2: re-fetch fresh from the DB -- never trust a stale in-memory instance.
        run = get_object_or_404(CampaignRun, pk=pk)

        # Business-logic bypass guard (Security "business-logic bypass" domain): validate
        # state server-side, never just trust the button was only offered on eligible rows.
        if run.approval_status != CampaignRun.ApprovalStatus.APPROVED or not run.site_needs_review:
            messages.warning(request, 'This run is not awaiting site resolution.')
            return redirect('campaigns:approval_queue')

        # 22-06: capture the pre-read site pk (None when unresolved, the placeholder
        # Observatory's own pk when a placeholder) BEFORE any write -- the conditional
        # claim below keys on this exact value so two racing staff POSTs can never
        # double-write (D-06).
        previous_site_id = run.site_id

        # D-06 never-re-resolve guard, extended for 22-06: only resolve when the site
        # isn't set yet, OR is a tier-3 placeholder (not a genuine resolution). A run with
        # a REAL Observatory already set + site_needs_review still True is the
        # projection-failed retry state (finding 8c) -- resolve_site is never called again
        # for it; it falls straight through to the projection retry below.
        if run.site is None or is_placeholder_observatory(run.site):
            # SITE-02: prefer the staff-submitted site_selection over the originally-
            # submitted site_raw, falling back to site_raw when blank.
            selection = request.POST.get('site_selection', '').strip() or run.site_raw
            # CR-01: map the display-string selection back to its obscode via the same
            # candidate pool the widget was built from, before calling resolve_site(), which
            # otherwise treats its argument as a literal obscode.
            obscode_selection = build_site_candidates().get(selection, selection)
            site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
            if site is None:
                # Nothing was written -- D-09: never fabricate a second placeholder from
                # unresolvable input. The flag is already True, the row (still pointing at
                # its existing placeholder, if any) stays in the review table for another
                # attempt.
                messages.error(
                    request,
                    'Could not resolve that site. Try a different search term or an exact '
                    'MPC code, or use Create new Observatory.',
                )
                return redirect('campaigns:approval_queue')

            # 22-REVIEWS.md finding 5: claim the site write with a single conditional
            # queryset update mirroring the approve/reject staleness guard
            # (`updated_count == 1` discipline) -- deliberately writing `site` ONLY, never
            # `site_needs_review` (finding 3: the flag must never clear before a successful
            # projection). Not using transaction.atomic()+select_for_update() here -- the
            # conditional-update claim is this codebase's established guard and behaves
            # uniformly on SQLite.
            #
            # 22-06: keyed on `site_id=previous_site_id` (not the old hard-coded
            # `site__isnull=True`) so the same conditional-claim guard covers both the
            # unresolved case (Django treats `site_id=None` as IS NULL -- byte-equivalent
            # to the old filter) and the placeholder-replacement case: a competing POST
            # that already changed the site away from `previous_site_id` matches zero rows.
            claimed = CampaignRun.objects.filter(
                pk=pk,
                approval_status=CampaignRun.ApprovalStatus.APPROVED,
                site_needs_review=True,
                site_id=previous_site_id,
            ).update(site=site)
            if claimed == 0:
                # A racing staff POST resolved (or is resolving) this run first -- the
                # loser's site value is never written, and no projection fires for it.
                messages.warning(request, "This run's site was already resolved by someone else.")
                return redirect('campaigns:approval_queue')
            run.refresh_from_db()

            # WR-03 (22-REVIEW.md re-review): the just-replaced placeholder Observatory (if
            # any) is now orphaned by this run -- delete it so it stops satisfying
            # is_placeholder_observatory() and no longer pollutes the search-suggestion pool
            # (CR-02) for the next, unrelated resolution attempt. Guarded on no other
            # CampaignRun still referencing it: the same placeholder obscode can be shared by
            # more than one still-unresolved row (e.g. several CSV-imported runs at one
            # still-unconfigured site), so it's only safe to delete once nothing points to it
            # anymore.
            if previous_site_id is not None:
                try:
                    previous_site = Observatory.objects.get(pk=previous_site_id)
                except Observatory.DoesNotExist:
                    pass
                else:
                    if (
                        is_placeholder_observatory(previous_site)
                        and not CampaignRun.objects.filter(site_id=previous_site_id).exists()
                    ):
                        previous_site.delete()

        # Projection, inside its own NON-reverting try/except (never reuse the approve
        # branch's revert-to-PENDING_REVIEW except block -- reverting an already-APPROVED
        # run would resurrect it into the pending queue, reintroducing the dead end this
        # phase closes).
        try:
            created = _project_calendar_event(run)
        except Exception:
            logger.exception('Calendar projection failed for CampaignRun %s during resolve_site.', pk)
            messages.warning(
                request,
                "Site resolved, but the calendar entry couldn't be created automatically -- "
                'the run stays in Sites Needing Review; use Resolve to retry.',
            )
            return redirect('campaigns:approval_queue')

        # Only after the projection call returned without raising: clear the flag.
        run.site_needs_review = False
        run.save(update_fields=['site_needs_review'])
        if created:
            messages.success(request, 'Site resolved — run added to the calendar.')
        else:
            messages.success(request, 'Site resolved.')
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


class SiteSearchView(View):
    """Shared, anonymous, throttled HTMX live-search endpoint (D-01/D-02/D-03, Phase 22 P01).

    Public/read-only, same posture as ``CampaignGapAnalysisView``/``CampaignRunTableView``
    -- deliberately no ``StaffRequiredMixin``. The candidate pool is public MPC data
    (``build_site_candidates()``), and this endpoint backs the public submission form
    (Plan 02) as well as the approval-queue widgets (Plan 02/03) -- neither caller is
    staff-only. Returns a rendered HTML fragment (never JSON), per D-03.
    """

    http_method_names = ['get']

    def get(self, request):
        """Throttle, validate, min-length-gate, then render the suggestion fragment."""
        # D-02/Pitfall 5: staff triaging the approval queue must never trip the
        # anonymous-abuse throttle meant for the public form (Assumption A3) -- exempt
        # authenticated staff from the per-IP counter entirely.
        client_ip = request.META.get('REMOTE_ADDR')
        if not request.user.is_staff:
            if client_ip:
                if not _check_and_increment_throttle(client_ip):
                    return HttpResponse(status=429)
            else:
                # WR-02 (22-REVIEW.md): a missing REMOTE_ADDR must never fall back to an
                # empty-string cache key -- that would silently collapse every such
                # anonymous client into one shared throttle bucket (cross-client
                # interference). Treat "no client IP available" as "no throttle key
                # available" instead: skip throttling for this request and log it, so the
                # failure mode is "no rate limit" rather than one client's usage 429-ing
                # unrelated clients.
                logger.warning(
                    'SiteSearchView: REMOTE_ADDR missing from request.META; skipping the '
                    'per-IP throttle for this anonymous request rather than sharing a '
                    'single empty-string bucket across all such clients.'
                )

        # 22-REVIEWS.md finding 2: validate input_id server-side against a conservative
        # DOM-id allowlist before it ever reaches the template context -- HTML
        # auto-escaping alone is not sufficient inside the fragment's inline `onclick=`
        # JS-string context (see _INPUT_ID_RE comment above).
        input_id = request.GET.get('input_id', 'id_site_raw')
        if not _INPUT_ID_RE.fullmatch(input_id):
            input_id = 'id_site_raw'

        # 22-REVIEWS.md finding 4/T-22-02: a blank/1-char query must never reach
        # build_site_candidates() -- on a cache miss that would trigger
        # MPCObscodeFetcher().query_all(), and the widgets' client-side 2-char
        # hx-trigger filter only gates browser-originated requests, not a direct
        # anonymous GET. Gate here, AFTER the throttle check but BEFORE any pool access.
        #
        # gap_closure (22-04, debug/site-search-widget-query-param-mismatch.md): htmx's
        # hx-get serializes only the triggering element's own name-keyed value plus
        # hx-vals -- never an enclosing form's other fields, unlike POST. Neither widget
        # sends `q`: the public submission form's field is `name="site_raw"`
        # (campaign_forms.py) and the approval-queue/Sites-Needing-Review widgets are
        # `name="site_selection"` (campaign_tables.py). Resolve the term from `q` first
        # (so every existing `?q=` caller/test is unaffected), then `site_raw`, then
        # `site_selection`, preferring the first non-empty value.
        query = request.GET.get('q', '') or request.GET.get('site_raw', '') or request.GET.get('site_selection', '')
        if len(query.strip()) < 2:
            return render(
                request,
                'campaigns/partials/site_search_results.html',
                {'candidates': [], 'input_id': input_id, 'query': '', 'no_matches_copy': ''},
            )

        candidates = substring_or_fuzzy_match_candidates(query, build_site_candidates())
        # Copywriting Contract: distinguish the public form (free text is fine, staff
        # will resolve it) from the queue widgets (a different site is expected to
        # actually resolve) by input_id.
        no_matches_copy = (
            'No matches — free text is fine, a staff member will resolve it.'
            if input_id == 'id_site_raw'
            else 'No matches for this search.'
        )
        return render(
            request,
            'campaigns/partials/site_search_results.html',
            {'candidates': candidates, 'input_id': input_id, 'query': query, 'no_matches_copy': no_matches_copy},
        )
