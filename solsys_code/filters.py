"""Target list filters extending the TOM Toolkit defaults."""

import django_filters
from crispy_forms.layout import Column
from django import forms
from tom_targets.filters import TargetFilterSet

# Same HTMX widget attributes tom_targets uses on its own <select> filters, so a change
# refreshes the table in place rather than requiring a form submit.
_HTMX_SELECT_ATTRS = {
    'hx-get': '',
    'hx-trigger': 'change',
    'hx-target': 'div.table-container',
    'hx-swap': 'innerHTML',
    'hx-indicator': '.progress',
    'hx-include': 'closest form',
}

SCOUT_FILTER_CHOICES = [
    ('active', 'Active Scout candidates'),
    ('retired', 'Retired Scout candidates'),
    ('any', 'Any Scout candidate'),
    ('none', 'Not from Scout'),
]


class ScoutTargetFilterSet(TargetFilterSet):
    """``TargetFilterSet`` with a quick filter on a target's JPL Scout status.

    Scout-sourced targets are identified by the presence of a ``tom_jpl.models.ScoutDetail``
    row; ``ScoutDetail.active`` distinguishes candidates still on the Scout roster from
    ones that have left it (retired by ``updatescout``).
    """

    scout = django_filters.ChoiceFilter(
        label='Scout',
        choices=SCOUT_FILTER_CHOICES,
        empty_label='All targets',
        method='filter_scout',
        widget=forms.Select(attrs=_HTMX_SELECT_ATTRS),
    )

    def filter_scout(self, queryset, name, value):
        """Restrict ``queryset`` by Scout status; an unrecognised/empty ``value`` is a no-op."""
        if value == 'active':
            return queryset.filter(scout_detail__active=True)
        if value == 'retired':
            return queryset.filter(scout_detail__active=False)
        if value == 'any':
            return queryset.filter(scout_detail__isnull=False)
        if value == 'none':
            return queryset.filter(scout_detail__isnull=True)
        return queryset

    @property
    def form(self):
        """The parent's crispy form, with the Scout selector added to the primary search row."""
        first_build = not hasattr(self, '_form')
        form = super().form
        if first_build:
            # Row 0 of the parent layout holds the primary (non-"Advanced") search
            # controls; put the Scout selector there so it is always visible.
            form.helper.layout[0].append(Column('scout', css_class='form-group col-md-3'))
        return form
