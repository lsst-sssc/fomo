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
from django.utils.html import format_html
from django_tables2.utils import Accessor

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
            'contact_person',
            'contact_email',
        )
        order_by = ('-obs_date',)  # D-10
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
        """Show Observatory.short_name when resolved, else flagged site_raw text (UI-SPEC)."""
        site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
        if site_short_name:
            return site_short_name
        if Accessor('site_needs_review').resolve(record, quiet=True):
            site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
            return format_html(
                '<span class="text-muted font-italic" title="Site could not be automatically resolved">'
                '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i> {}</span>',
                site_raw,
            )
        return ''

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
            'obs_date',
            'ut_start',
            'ut_end',
            '...',
        )

    def __init__(self, *args, show_actions=True, request=None, **kwargs):
        self.show_actions = show_actions
        self.request = request
        super().__init__(*args, **kwargs)

    def render_actions(self, record):
        """Render side-by-side Approve/Reject mini-forms, or nothing (decided-runs table)."""
        if not self.show_actions:
            return ''
        decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
        csrf_token = get_token(self.request) if self.request is not None else ''
        return format_html(
            '<div class="d-flex" style="gap: 0.5rem;">'
            '<form method="post" action="{0}" class="d-inline">'
            '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
            '<input type="hidden" name="action" value="approve">'
            '<button type="submit" class="btn btn-sm btn-success">Approve</button>'
            '</form>'
            '<form method="post" action="{0}" class="d-inline">'
            '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
            '<input type="hidden" name="action" value="reject">'
            '<button type="submit" class="btn btn-sm btn-danger" '
            'onclick="return confirm(\'Reject this submission? '
            'The submitter will not be automatically notified.\')">Reject</button>'
            '</form>'
            '</div>',
            decide_url,
            csrf_token,
        )
