"""Views over stored JPL Scout details.

Kept separate from :mod:`solsys_code.views` so these light-weight, Scout-only
views do not pull in the ephemeris machinery (REBOUND/ASSIST/sorcha and the
~1.6 GB SPICE kernel furnish) that importing ``solsys_code.views`` triggers.
"""

from collections import defaultdict

from django.views.generic import ListView, TemplateView
from tom_jpl.models import ScoutDetail, ScoutDetailHistory

from solsys_code.rubin_too import RUBIN_TOO_FILTERS, evaluate_filters, passes_filters


class RubinTooScoutListView(ListView):
    """The live "currently passing" snapshot of Scout NEO candidates.

    Lists every target whose *current* :class:`~tom_jpl.models.ScoutDetail`
    satisfies all the Section 2.1 Rubin ToO filters. Because the filters branch
    on declination (and so are awkward to express in the ORM), the queryset is
    materialised and filtered in Python; the candidate population is small.
    """

    template_name = 'solsys_code/scout_rubin_too_list.html'
    context_object_name = 'scout_details'

    def get_queryset(self):
        """Return the stored ScoutDetails that currently pass all Section 2.1 filters."""
        details = ScoutDetail.objects.select_related('target').filter(active=True)
        passing = [sd for sd in details if passes_filters(sd)]
        for sd in passing:
            # Convenience for the template: arc is stored in days, but is most
            # readable in hours for these short (<~14 day) candidate arcs.
            sd.arc_hours = sd.arc * 24.0 if sd.arc is not None else None
        return passing

    def get_context_data(self, **kwargs):
        """Add passing/total Scout candidate counts for the template summary line."""
        context = super().get_context_data(**kwargs)
        context['num_passing'] = len(context['scout_details'])
        context['num_total'] = ScoutDetail.objects.filter(active=True).count()
        return context


def _compute_first_pass_stats():
    """Walk ScoutDetailHistory and count first-pass events per year.

    For each target, walking its history in ``last_run`` order, a "first-pass
    event" is recorded whenever the candidate transitions from failing to passing
    (either combined or per individual filter). Each target is counted at most
    once per year per filter — subsequent re-entries within the same year are
    not counted again. The year boundary resets the counter so a target that
    fails then passes again in a new year is counted for that new year.

    Returns a list of ``{'year': int, 'combined': int, <filter_key>: int, ...}``
    dicts, one per year with at least one event, sorted ascending by year.
    """
    filter_keys = [key for key, _, _ in RUBIN_TOO_FILTERS]
    stats = defaultdict(lambda: defaultdict(int))

    target_ids = list(ScoutDetailHistory.objects.order_by().values_list('target_id', flat=True).distinct())

    for target_id in target_ids:
        rows = list(
            ScoutDetailHistory.objects.filter(target_id=target_id).select_related('target').order_by('last_run')
        )
        if not rows:
            continue

        abs_mag = rows[0].target.abs_mag
        prev_combined = False
        prev_per_filter = {key: False for key in filter_keys}
        counted_combined_years: set = set()
        counted_filter_years: dict = defaultdict(set)

        for row in rows:
            if row.last_run is None:
                continue
            year = row.last_run.year
            filter_results = evaluate_filters(row, abs_mag=abs_mag)
            combined = all(filter_results.values())

            if combined and not prev_combined and year not in counted_combined_years:
                stats[year]['combined'] += 1
                counted_combined_years.add(year)

            for key, result in filter_results.items():
                if result and not prev_per_filter[key] and year not in counted_filter_years[key]:
                    stats[year][key] += 1
                    counted_filter_years[key].add(year)

            prev_combined = combined
            prev_per_filter = dict(filter_results)

    return [
        {'year': year, 'combined': year_data.get('combined', 0), **{key: year_data.get(key, 0) for key in filter_keys}}
        for year, year_data in sorted(stats.items())
    ]


class RubinTooScoutStatsView(TemplateView):
    """Per-year first-pass event counts for Rubin ToO filter criteria.

    Walks :class:`~tom_jpl.models.ScoutDetailHistory` rows to identify the
    first moment each candidate crosses from failing to passing (both combined
    and per individual filter) within a calendar year. Candidates are counted
    at most once per year per filter, so a re-entry after a gap within the same
    year is not double-counted. Year boundaries reset the count.
    """

    template_name = 'solsys_code/scout_rubin_too_stats.html'

    def get_context_data(self, **kwargs):
        """Compute per-year first-pass stats and pass them to the template."""
        context = super().get_context_data(**kwargs)
        years_stats = _compute_first_pass_stats()
        context['years'] = [ys['year'] for ys in years_stats]
        context['combined_by_year'] = [ys['combined'] for ys in years_stats]
        context['filter_rows'] = [
            {'key': key, 'label': label, 'counts': [ys[key] for ys in years_stats]}
            for key, label, _ in RUBIN_TOO_FILTERS
        ]
        context['total_targets'] = ScoutDetailHistory.objects.values('target_id').distinct().count()
        return context
