"""General-search callables for the TOM Toolkit HTMX filter sets."""

from django.db.models import Q


def target_general_search(queryset, name, value):
    """Match the target list's General Search box against target names *and* aliases.

    The TOM Toolkit default matches ``Target.name`` only, so a target renamed after
    ingest -- an NEOCP trksub such as ``CERNQ52`` becoming ``2026 RW1`` once the MPC
    designates it -- stops being findable under the name it was ingested with, even
    though the trksub survives as an alias. Registered via
    ``settings.GENERAL_SEARCH_FUNCTIONS``.
    """
    if not value:
        return queryset
    return queryset.filter(Q(name__icontains=value) | Q(aliases__name__icontains=value)).distinct()
