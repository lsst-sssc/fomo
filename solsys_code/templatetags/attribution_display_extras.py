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
from solsys_code.models import CalendarEventMeta

register = template.Library()


@register.simple_tag
def high_band_attribution_candidates(event: CalendarEvent) -> list[campaign_attribution.AttributionCandidate]:
    """HIGH-band attribution candidates for one unlinked CalendarEvent (27-UAT.md Test 9).

    A thin filter over ``campaign_attribution.candidates_for_event()`` -- the existing,
    already dismissal-aware, campaign-boundary-gated scorer -- kept to the High band only,
    since that is the confidence tier this hint is meant to surface. Never raises: returns ``[]``
    for any value that is not a CalendarEvent (the create form passes the empty-string
    placeholder for its missing ``event``) and otherwise delegates to ``candidates_for_event()``.

    WR-06 (37.1-REVIEW.md): an event drawn from an observation record is never offered
    (37.1 D-09) -- it is attributed only through its record, and the attribution queue no longer
    lists it, so a hint here would link to a page where neither the event nor (necessarily) its
    record appears. Returns ``[]`` for such an event.

    Args:
        event: the CalendarEvent to find candidates for (typically one with no
            CalendarEventMeta.run set yet).

    Returns:
        list[AttributionCandidate]: the event's candidates whose band is
        ``campaign_attribution.BAND_HIGH``, possibly empty -- always empty for a record's own
        event or for a value that is not a CalendarEvent. Never raises.
    """
    # The create-event form context has no `event` key, and Django resolves the missing variable
    # to the empty-string invalid-variable placeholder rather than raising -- the same guard as
    # campaign_decoration(), run_tally() and observation_series_decoration() in
    # calendar_display_extras.py. Without it the lookup below raised ValueError and a staff
    # viewer's New Event pop-up was a 500 (39-UAT.md G-39-4).
    if not isinstance(event, CalendarEvent):
        return []
    if CalendarEventMeta.objects.filter(event=event, observation_record__isnull=False).exists():
        return []
    return [c for c in campaign_attribution.candidates_for_event(event) if c.band == campaign_attribution.BAND_HIGH]
