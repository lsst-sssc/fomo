"""django-tables2 Table definition for the per-campaign CampaignRun read path (VIEW-01).

Renders identically whether ``record`` is a plain ``dict`` (the restricted ``.values()``
queryset used for non-staff requests, D-13/VIEW-03) or a full ``CampaignRun`` model instance
(staff requests). django-tables2's automatic ``get_FOO_display()`` choice-label lookup is
skipped for dict rows (15-RESEARCH.md Pitfall 2), so ``run_status``/``approval_status`` labels
are resolved manually here via the model's ``TextChoices`` rather than relied on automatically.
"""

import django_tables2 as tables
from django.middleware.csrf import get_token
from django.urls import reverse
from django.utils.html import format_html, format_html_join
from django.utils.http import urlencode
from django_tables2.utils import Accessor

from .campaign_utils import fuzzy_match_candidates
from .models import CampaignRun

# D-08 / UI-SPEC Approval-Status Badge Contract: fixed 3-entry dict, badge class never derived
# from the raw DB string (mirrors calendar_display_extras.py's constant-lookup pattern shape).
APPROVAL_BADGE_CLASSES = {
    CampaignRun.ApprovalStatus.PENDING_REVIEW: 'badge-warning',
    CampaignRun.ApprovalStatus.APPROVED: 'badge-success',
    CampaignRun.ApprovalStatus.REJECTED: 'badge-danger',
}

# UI-SPEC Run-Status Badge Contract: deliberately muted so it never competes with the
# mandatory approval_status badge. Dead-end outcomes use badge-light (+ grey border added in
# render_run_status), NOT badge-danger -- danger-red is reserved exclusively for
# approval_status=rejected (see UI-SPEC rationale).
RUN_STATUS_BADGE_CLASSES = {
    CampaignRun.RunStatus.REQUESTED: 'badge-secondary',
    CampaignRun.RunStatus.PLANNED: 'badge-secondary',
    CampaignRun.RunStatus.OBSERVED: 'badge-info',
    CampaignRun.RunStatus.REDUCED: 'badge-info',
    CampaignRun.RunStatus.PUBLISHED: 'badge-primary',
    CampaignRun.RunStatus.CANCELLED: 'badge-light',
    CampaignRun.RunStatus.NOT_AWARDED: 'badge-light',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: 'badge-light',
}

# Fields whose staff-vs-anonymous underlying key genuinely differs (dict path selects
# 'site__short_name' explicitly, never 'site' -- see campaign_views.ALLOWED_FIELDS_FOR_NON_STAFF),
# so the column needs an Accessor that resolves both a literal dict key and a model-instance
# attribute chain (confirmed against installed django_tables2.utils.Accessor.resolve source).
_FREE_TEXT_ATTRS = {'td': {'class': 'text-truncate', 'style': 'max-width: 200px;'}}


class CampaignRunTable(tables.Table):
    """Spreadsheet-parity CampaignRun table (D-09), PII-gated via the view's ``exclude=`` kwarg."""

    site = tables.Column(accessor='site__short_name', verbose_name='Site', empty_values=())

    class Meta:  # noqa: D106
        model = CampaignRun
        fields = (
            'telescope_instrument',
            'site',
            'window_start',
            'filters_bandpass',
            'run_status',
            'approval_status',
            'open_to_collaboration',
            'observation_details',
            'weather',
            'observation_outcome',
            'publication_plans',
            'comments',
            'contact_person',
            'contact_email',
        )
        template_name = 'django_tables2/bootstrap4-responsive.html'
        attrs = {'class': 'table table-bordered table-sm'}
        empty_text = 'No runs match these filters. Clear filters to see all runs for this campaign.'

    observation_details = tables.Column(attrs=_FREE_TEXT_ATTRS)
    weather = tables.Column(attrs=_FREE_TEXT_ATTRS)
    observation_outcome = tables.Column(attrs=_FREE_TEXT_ATTRS)
    publication_plans = tables.Column(attrs=_FREE_TEXT_ATTRS)
    comments = tables.Column(attrs=_FREE_TEXT_ATTRS)

    def render_run_status(self, record):
        """Render run_status as a muted Bootstrap badge (UI-SPEC Run-Status Badge Contract).

        Reads the raw stored value from ``record`` via Accessor rather than accepting
        django-tables2's pre-resolved ``value`` kwarg: for model-instance rows (staff),
        django-tables2's row machinery auto-calls ``get_run_status_display()`` *before*
        this method runs (since the field has ``choices``), silently handing us the
        already-humanized label instead of the raw code -- would break the
        ``CampaignRun.RunStatus(value)`` lookup below. Resolving from ``record`` directly
        sidesteps that pre-processing and gives the raw code for both dict and model rows.
        """
        value = Accessor('run_status').resolve(record, quiet=True)
        css = RUN_STATUS_BADGE_CLASSES.get(value, 'badge-secondary')
        label = CampaignRun.RunStatus(value).label
        style = 'border: 1px solid #6c757d;' if css == 'badge-light' else ''
        return format_html('<span class="badge {}" style="{}">{}</span>', css, style, label)

    def render_approval_status(self, record):
        """Render approval_status as a colored Bootstrap badge (D-08).

        See render_run_status docstring -- same raw-value-via-Accessor rationale applies.
        """
        value = Accessor('approval_status').resolve(record, quiet=True)
        css = APPROVAL_BADGE_CLASSES.get(value, 'badge-secondary')
        label = CampaignRun.ApprovalStatus(value).label
        return format_html('<span class="badge {}">{}</span>', css, label)

    def render_site(self, record):
        """Show Observatory.short_name when resolved, else the submitted site_raw text.

        Falls back to ``site_raw`` whenever the site is unresolved (``site__short_name``
        empty) and ``site_raw`` is non-empty, regardless of ``site_needs_review`` --
        pending runs (D-07) leave ``site_needs_review`` False until approval, so relying
        on that flag alone hid every pending submission's site text from staff. When
        resolution genuinely ran and failed (``site_needs_review`` True), keep the
        failure styling (warning triangle); otherwise (not yet attempted) show a plain
        muted-italic "pending review" presentation with no failure icon.
        """
        site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
        if site_short_name:
            return site_short_name
        site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
        if not site_raw:
            return ''
        if Accessor('site_needs_review').resolve(record, quiet=True):
            return format_html(
                '<span class="text-muted font-italic" title="Site could not be automatically resolved">'
                '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i> {}</span>',
                site_raw,
            )
        return format_html(
            '<span class="text-muted font-italic" title="Site not yet resolved -- pending review">{}</span>',
            site_raw,
        )

    def render_window_start(self, record):
        """Render the observing window as a TBD badge, single date, or 'start -> end' range.

        Resolves both window_start/window_end via Accessor (D-03/D-05) so this works
        identically whether record is a dict (non-staff) or a CampaignRun instance
        (staff) -- mirrors render_site()'s dict-vs-model dual-accessor precedent.

        The TBD badge additionally carries a ``title`` tooltip with
        ``original_obs_date_raw`` (D-08) when that field is non-empty, so staff can see
        exactly what the sheet said without new display machinery -- reuses render_site()'s
        format_html tooltip convention. The raw text is interpolated as a positional
        format_html argument so Django auto-escapes it (mitigates stored-XSS from
        community-editable sheet text, T-20-03); never mark_safe or string concatenation.
        """
        start = Accessor('window_start').resolve(record, quiet=True)
        end = Accessor('window_end').resolve(record, quiet=True)
        if start is None:  # both null by the model's own invariant
            original_obs_date_raw = Accessor('original_obs_date_raw').resolve(record, quiet=True) or ''
            if original_obs_date_raw:
                return format_html('<span class="badge badge-secondary" title="{}">TBD</span>', original_obs_date_raw)
            return format_html('<span class="badge badge-secondary">TBD</span>')
        if start == end:
            return start  # single-night row (D-05)
        return format_html('{} -&gt; {}', start, end)  # D-05: literal "->", not an en-dash

    def render_open_to_collaboration(self, value):
        """Render open_to_collaboration as a Yes/No icon (UI-SPEC column set)."""
        if value:
            return format_html('<i class="fa fa-check text-success" aria-hidden="true" title="Yes"></i>')
        return format_html('<i class="fa fa-times text-muted" aria-hidden="true" title="No"></i>')


class ApprovalQueueTable(CampaignRunTable):
    """CampaignRunTable plus an Actions column for the staff approval queue (D-01/D-02).

    The pending-review table (``show_actions=True``, the default) renders an Approve/Reject
    button pair per row that POST directly to ``campaigns:decide``; the recently-decided table
    (``show_actions=False``) renders the same columns with an empty Actions cell -- read-only
    per 16-RESEARCH.md Open Question 2. CSRF protection is handled inside ``render_actions``
    itself (via ``django.middleware.csrf.get_token``) rather than in the template's row loop,
    since ``{% render_table %}`` doesn't hand row-rendering control back to the template (see
    16-03-PLAN.md Task 1 planner note) -- the request object must be passed in explicitly at
    construction time so a CSRF token can be minted for each row's mini-forms.
    """

    actions = tables.Column(empty_values=(), orderable=False, verbose_name='Actions')

    # Triage-focused queue view (UAT Test 14 gap closure, 16-05): drop the three
    # post-observation columns (weather, observation_outcome, publication_plans) that have
    # no CampaignRunSubmissionForm field and are therefore structurally always blank on a
    # PENDING_REVIEW row, and front-load `actions` so Approve/Reject is reachable without
    # horizontal scrolling. CampaignRunTable itself is untouched -- it stays spreadsheet-parity
    # for Phase 15's D-09 read path.
    class Meta(CampaignRunTable.Meta):  # noqa: D106
        exclude = ('weather', 'observation_outcome', 'publication_plans')
        sequence = (
            'actions',
            'approval_status',
            'telescope_instrument',
            'site',
            'window_start',
            '...',
        )

    def __init__(self, *args, show_actions=True, request=None, candidate_pool=None, **kwargs):
        self.show_actions = show_actions
        self.request = request
        self.candidate_pool = candidate_pool
        super().__init__(*args, **kwargs)

    def render_site(self, record):
        """Unresolved actionable pending row: inline site input + fuzzy-matched datalist +
        an always-visible "Create new Observatory" link (SITE-01/D-04), submitted into the
        row's single decide-form via the HTML5 ``form=`` attribute. Resolved rows and the
        read-only decided table (``show_actions=False``) keep CampaignRunTable's existing
        plain-text ``render_site`` rendering unchanged.

        Only overridden here (not on ``CampaignRunTable``): only ``ApprovalQueueTable``
        instances carry ``self.show_actions``/``self.candidate_pool``, so overriding on the
        parent would raise ``AttributeError`` for the per-campaign ``CampaignRunTable``.
        """
        site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
        if site_short_name or not self.show_actions:
            return super().render_site(record)
        pk = Accessor('pk').resolve(record, quiet=True)
        site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
        datalist_id = f'site-candidates-{pk}'
        form_id = f'decide-form-{pk}'
        candidate_pairs = fuzzy_match_candidates(site_raw, self.candidate_pool) if self.candidate_pool else []
        # Only the MPC-sourced display string becomes the <option> value -- the resolved
        # obscode itself is read server-side from site_selection's submitted text, not
        # from a hidden option attribute (T-21-01: every candidate string still goes
        # through format_html_join's auto-escaping positional substitution).
        options = format_html_join('', '<option value="{}">', ((candidate,) for candidate, _obscode in candidate_pairs))
        create_url = '{}?{}'.format(
            reverse('solsys_code_observatory:create'),
            urlencode({'obscode': site_raw, 'next': reverse('campaigns:approval_queue')}),
        )
        return format_html(
            '<input type="text" name="site_selection" value="{0}" list="{1}" '
            'form="{2}" class="form-control form-control-sm" placeholder="MPC code or site name…">'
            '<datalist id="{1}">{3}</datalist>'
            '<a href="{4}" class="small ml-1">Create new Observatory</a>',
            site_raw,
            datalist_id,
            form_id,
            options,
            create_url,
        )

    def render_actions(self, record):
        """Render one form (Approve/Reject as named submit buttons), or nothing
        (decided-runs table). Single form (not two) so the Site column's ``form=`` input
        can target it via the HTML5 ``form=`` attribute (D-04)."""
        if not self.show_actions:
            return ''
        decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
        csrf_token = get_token(self.request) if self.request is not None else ''
        form_id = f'decide-form-{record.pk}'
        return format_html(
            '<form id="{0}" method="post" action="{1}">'
            '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
            '<div class="d-flex" style="gap: 0.5rem;">'
            '<button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>'
            '<button type="submit" name="action" value="reject" class="btn btn-sm btn-danger" '
            'onclick="return confirm(\'Reject this submission? '
            'The submitter will not be automatically notified.\')">Reject</button>'
            '</div></form>',
            form_id,
            decide_url,
            csrf_token,
        )
