"""Views over stored JPL Scout details.

Kept separate from :mod:`solsys_code.views` so these light-weight, Scout-only
views do not pull in the ephemeris machinery (REBOUND/ASSIST/sorcha and the
~1.6 GB SPICE kernel furnish) that importing ``solsys_code.views`` triggers.
"""

from django.views.generic import ListView
from tom_jpl.models import ScoutDetail

from solsys_code.rubin_too import passes_filters


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
