"""django-filter FilterSet for the per-campaign CampaignRun read path (VIEW-04).

``run_status`` must be explicitly declared as a ``MultipleChoiceFilter`` -- ``Meta.fields``
auto-generation produces a single-value ``CharFilter`` for a ``choices`` ``CharField`` (no
special-casing for ``choices`` in django-filter's ``FILTER_FOR_DBFIELD_DEFAULTS``), which would
violate D-12's OR-semantics multi-select requirement. ``open_to_collaboration`` is a plain
``BooleanField`` and is left to ``Meta.fields`` auto-generation, which correctly produces a
``BooleanFilter``.
"""

import django_filters
from django import forms

from .models import CampaignRun


class CampaignRunFilterSet(django_filters.FilterSet):
    """VIEW-04: multi-select run_status (OR semantics, D-12) + boolean open_to_collaboration."""

    run_status = django_filters.MultipleChoiceFilter(
        choices=CampaignRun.RunStatus.choices,
        label='Run status',
        widget=forms.CheckboxSelectMultiple,
    )

    class Meta:  # noqa: D106
        model = CampaignRun
        fields = ['run_status', 'open_to_collaboration']
