"""Django template tag library for proposal color and status visual encoding.

Provides three simple_tags consumed by calendar.html (Plan 02):

- proposal_color: deterministic, colorblind-vetted palette color keyed by proposal code (DISPLAY-04)
- status_border_css: title-prefix → box-shadow CSS fragment (DISPLAY-06)
- visible_proposals: current-month legend data grouped by color (DISPLAY-07)

All values returned by proposal_color and status_border_css are drawn from fixed
internal constants — the raw proposal/title string is used only as a hash input or
startswith test and is never echoed into the output (T-09-01/T-09-02 mitigations).
"""

import hashlib
from collections import defaultdict

from django import template

register = template.Library()

# Colorblind-vetted, white-text-AA palette — 8 hex values locked by 09-UI-SPEC.md
# Color section.  Mutual distinguishability verified against CVD simulators for
# deuteranopia + protanopia (see 09-VALIDATION.md manual verification item A1).
PROPOSAL_PALETTE = [
    '#005f9e',
    '#a34000',
    '#5b2080',
    '#006b4e',
    '#9e1c1c',
    '#006b6b',
    '#6b2060',
    '#7a4500',
]

# D-05: dedicated neutral slot for calendar events with no proposal code.
# Separate from PROPOSAL_PALETTE so an empty-string hash cannot accidentally
# collide with this value (see 09-RESEARCH Pitfall 1).
NEUTRAL_SLOT_COLOR = '#5a6268'

# D-06: human-readable label for classical-schedule (empty-proposal) legend entry.
CLASSICAL_SCHEDULE_LABEL = 'Classical schedule'

# Title-prefix vocabulary emitted by sync_lco_observation_calendar.py (confirmed live).
# Terminal states: observations that reached an unrecoverable failure state.
# [QUEUED] is handled separately (its own branch in status_border_css).
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]')


@register.simple_tag
def proposal_color(proposal: str) -> str:
    """Return a deterministic hex color for a proposal code (DISPLAY-04).

    Normalizes via .strip().upper() before hashing so casing and whitespace
    variants share one color — D-04 premise, 09-RESEARCH Pitfall 1.  Uses
    hashlib.sha256 for deterministic output across process restarts (see
    STATE.md Key Technical Notes — the per-process-salted built-in is forbidden
    here).

    Args:
        proposal: Raw proposal string from CalendarEvent.proposal (may be
            blank, mixed-case, or have surrounding whitespace).

    Returns:
        A hex color string from PROPOSAL_PALETTE, or NEUTRAL_SLOT_COLOR for
        blank/missing proposals (D-05).
    """
    normalized = (proposal or '').strip().upper()
    if not normalized:
        return NEUTRAL_SLOT_COLOR
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return PROPOSAL_PALETTE[int(digest, 16) % len(PROPOSAL_PALETTE)]


@register.simple_tag
def status_border_css(title: str) -> str:
    """Return a CSS box-shadow fragment encoding the observation status (DISPLAY-06).

    Maps the title-prefix vocabulary from sync_lco_observation_calendar.py to a
    box-shadow ring (D-08 resolved=box-shadow).  The placed bucket ([UNVERIFIED]
    or no prefix) intentionally returns '' because Phase 8's D-09-reserved
    border treatment already owns the verified/fallback visual distinction —
    re-encoding it here would cause the two signals to merge into one style
    attribute branch instead of composing independently (09-RESEARCH Pitfall 3
    prevention).

    Args:
        title: CalendarEvent.title — may start with a known status prefix.

    Returns:
        A CSS fragment suitable for direct inclusion in a style attribute, e.g.
        'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'.  Returns '' for placed
        events.  The D-09-reserved border style is never emitted by this tag.
    """
    title = title or ''
    if title.startswith('[QUEUED] '):
        return 'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'
    if any(title.startswith(p) for p in _TERMINAL_PREFIXES):
        return 'box-shadow: 0 0 0 3px rgba(160, 0, 0, 0.55);'
    return ''


def _relative_luminance(hex_color: str) -> float:
    """Return relative luminance (0.0–1.0) for a #rrggbb hex color per WCAG 2.1."""
    if not hex_color or not isinstance(hex_color, str):
        return 0.0  # treat invalid input as black (worst case → white text returned)
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return 0.0
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def linearize(c: int) -> float:
        L = c / 255
        return L / 12.92 if L <= 0.04045 else ((L + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


@register.simple_tag
def text_color_for_bg(hex_color: str) -> str:
    """Return '#fff' or '#000' — whichever achieves WCAG AA 4.5:1 contrast against hex_color (DISPLAY-08).

    Uses the WCAG 2.1 relative luminance formula. White text achieves 4.5:1 against
    any background with luminance <= 0.183; all PROPOSAL_PALETTE and NEUTRAL_SLOT_COLOR
    entries are dark, so '#fff' is returned for all current palette members.

    Args:
        hex_color: A '#rrggbb' hex color string (e.g. '#005f9e').

    Returns:
        '#fff' if white text achieves >= 4.5:1 contrast; '#000' otherwise.
    """
    lum = _relative_luminance(hex_color)
    white_contrast = 1.05 / (lum + 0.05)
    return '#fff' if white_contrast >= 4.5 else '#000'


@register.simple_tag
def visible_proposals(weeks) -> list[dict]:
    """Compute the set of proposals visible in the currently-rendered month (DISPLAY-07).

    Iterates the weeks/day context already materialized by render_calendar() —
    no new database query (D-02).  Groups by resulting color so hash-colliding
    proposals share one legend entry (D-04, 09-RESEARCH Pitfall 4).  Neutral-slot
    events (empty proposal) appear as 'Classical schedule' and are forced last
    regardless of their hex sort position (D-06 / 09-UI-SPEC.md Legend Layout).

    Args:
        weeks: The weeks context list passed to calendar.html — a list of lists
            of day objects, each with .all_day_events and .events attributes
            containing objects with a .proposal attribute.

    Returns:
        List of dicts with keys 'color' (hex string), 'codes' (sorted list of
        proposal code strings or [CLASSICAL_SCHEDULE_LABEL] for the neutral
        slot), and 'label' (comma-joined string for display).  Sorted by color
        hex ascending, with the NEUTRAL_SLOT_COLOR entry appended last.
    """
    by_color: dict[str, set[str]] = defaultdict(set)
    for week in weeks:
        for day in week:
            # Support both dict-based days (tom_calendar view) and attribute-based
            # stubs (unit tests using SimpleNamespace or similar objects).
            if isinstance(day, dict):
                all_day = day['all_day_events']
                timed = day['events']
            else:
                all_day = day.all_day_events
                timed = day.events
            for event in list(all_day) + list(timed):
                normalized = (event.proposal or '').strip().upper()
                color = proposal_color(event.proposal)
                label = normalized if normalized else CLASSICAL_SCHEDULE_LABEL
                by_color[color].add(label)

    result = []
    for color, codes in sorted(by_color.items()):
        if color == NEUTRAL_SLOT_COLOR:
            continue
        result.append(
            {
                'color': color,
                'codes': sorted(codes),
                'label': ', '.join(sorted(codes)),
            }
        )

    if NEUTRAL_SLOT_COLOR in by_color:
        codes = by_color[NEUTRAL_SLOT_COLOR]
        result.append(
            {
                'color': NEUTRAL_SLOT_COLOR,
                'codes': sorted(codes),
                'label': ', '.join(sorted(codes)),
            }
        )

    return result
