"""django-tables2 Table definition for the per-campaign CampaignRun read path (VIEW-01).

Renders identically whether ``record`` is a plain ``dict`` (the restricted ``.values()``
queryset used for non-staff requests, D-13/VIEW-03) or a full ``CampaignRun`` model instance
(staff requests). django-tables2's automatic ``get_FOO_display()`` choice-label lookup is
skipped for dict rows (15-RESEARCH.md Pitfall 2), so ``run_status``/``approval_status`` labels
are resolved manually here via the model's ``TextChoices`` rather than relied on automatically.
"""

import django_tables2 as tables
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

    def render_run_status(self, value):
        """Render run_status as a muted Bootstrap badge (UI-SPEC Run-Status Badge Contract)."""
        css = RUN_STATUS_BADGE_CLASSES.get(value, 'badge-secondary')
        label = CampaignRun.RunStatus(value).label
        style = 'border: 1px solid #6c757d;' if css == 'badge-light' else ''
        return format_html('<span class="badge {}" style="{}">{}</span>', css, style, label)

    def render_approval_status(self, value):
        """Render approval_status as a colored Bootstrap badge (D-08)."""
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
