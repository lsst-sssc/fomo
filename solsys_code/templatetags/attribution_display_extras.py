"""Django template tag library for presentation glue over Phase 28's attribution matcher.

Provides a single simple_tag consumed by event_form.html (27-07 gap closure, 27-UAT.md
Test 9):

- high_band_attribution_candidates: HIGH-band candidate runs for an unlinked calendar event

Kept as its own file rather than folded into calendar_display_extras.py's color/visual-
encoding tags (a different concern -- proposal/telescope color and status rings, not
attribution) or into campaign_attribution.py itself, whose own module docstring forbids it
depending on the request-handling view layer or the template layer.
"""

from django import template
from tom_calendar.models import CalendarEvent

from solsys_code import campaign_attribution

register = template.Library()


@register.simple_tag
def high_band_attribution_candidates(event: CalendarEvent) -> list[campaign_attribution.AttributionCandidate]:
    """HIGH-band attribution candidates for one unlinked CalendarEvent (27-UAT.md Test 9).

    A thin filter over ``campaign_attribution.candidates_for_event()`` -- the existing,
    already dismissal-aware, campaign-boundary-gated scorer -- kept to the High band only,
    since that is the confidence tier this hint is meant to surface. Never raises: delegates
    entirely to ``candidates_for_event()``, which itself never raises.

    Args:
        event: the CalendarEvent to find candidates for (typically one with no
            CalendarEventMeta.run set yet).

    Returns:
        list[AttributionCandidate]: the event's candidates whose band is
        ``campaign_attribution.BAND_HIGH``, possibly empty. Never raises.
    """
    return [c for c in campaign_attribution.candidates_for_event(event) if c.band == campaign_attribution.BAND_HIGH]
