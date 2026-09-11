"""Django template tag library for proposal color and status visual encoding.

Provides simple_tags consumed by calendar.html (Plan 02):

- proposal_color: deterministic, colorblind-vetted palette color keyed by proposal code (DISPLAY-04)
- status_border_css: title-prefix → box-shadow CSS fragment (DISPLAY-06)
- visible_proposals: current-month legend data grouped by color (DISPLAY-07)
- telescope_color: deterministic palette color keyed by telescope name (quick-260724-osc)
- telescope_stripe_color: deterministic stripe-palette color keyed by telescope name,
  parallel to telescope_color but gated against the gray fill (quick-260724-vb0)
- visible_classical_telescopes: current-month classical-schedule telescope legend data (quick-260724-osc)
- neutral_slot_color: assignment tag exposing NEUTRAL_SLOT_COLOR to templates (quick-260724-osc)
- campaign_decoration: read-only campaign attribution decoration for event_form.html,
  rendered from CalendarEventMeta.run at request time (ANNOT-02, Phase 33 D-10/D-11/D-13/D-14)
- observation_status_legend: fixed, ordered observation-projector marker legend for
  calendar.html (PROJ-03, D-02, Phase 34 Plan 03)
- observation_series_decoration: read-only "night n of N" series decoration for
  event_form.html, rendered from CalendarEventMeta.observation_group at request time
  (PROJ-04/PROJ-05, D-04, Phase 34 Plan 03)

All values returned by proposal_color, telescope_color, and status_border_css are drawn
from fixed internal constants — the raw proposal/telescope/title string is used only as
a hash input or startswith test and is never echoed into the output (T-09-01/T-09-02,
T-osc-01/T-osc-02 mitigations).
"""

import hashlib
from collections import defaultdict
from datetime import datetime
from datetime import timezone as dt_timezone

from django import template
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse
from tom_calendar.models import CalendarEvent

from solsys_code.calendar_utils import record_time_window
from solsys_code.models import NO_CAMPAIGN_LABEL

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

# quick-260724-vb0: legend-only palette for the telescope legend chip (.cal-legend-chip),
# whose only background is the page's white background -- gated against #ffffff at
# >= 3.5:1 by _contrast_ratio (see TestTelescopeStripeContrast). This was previously
# documented as "chosen for against-gray-fill stripe contrast" (quick-260724-tiz), but a
# contrast audit found that claim false: it was only ever validated against white, and
# 4 of its 8 entries measured 1.26-2.02:1 against the gray classical-fill it was
# actually rendered on. That fill-facing case is now TELESCOPE_STRIPE_PALETTE below,
# a separate array -- one 8-color palette cannot clear 3:1 against both white and the
# gray fill at once (see TestTelescopeStripeContrast for the luminance-band proof).
# quick-260724-vb0 also retuned 4 entries that measured 3.07-3.41 against white here
# (green, mustard, purple, red) to same-hue-family values that clear 3.5:1; the other 4
# already cleared it and are unchanged. Both this palette and TELESCOPE_STRIPE_PALETTE
# were re-screened for mutual distinguishability under simulated protanopia/deuteranopia
# and hold at or above the shipped palette's floor (see 260724-vb0-SUMMARY.md).
TELESCOPE_PALETTE = [
    '#3987e5',
    '#d95926',
    '#008a55',
    '#a18245',
    '#d55181',
    '#008300',
    '#7b5ff7',
    '#c0736d',
]

# quick-260724-vb0: stripe-only palette for the classical-schedule left-edge stripe
# (.cal-event-classical::before), whose only background neighbour is its own chip's
# NEUTRAL_SLOT_COLOR fill on its right (inner) edge. Gated against NEUTRAL_SLOT_COLOR
# at >= 3.4:1 by _contrast_ratio, not against white -- TELESCOPE_PALETTE above is the
# one gated against white, because the legend chip is its only background. This is a
# parallel array to TELESCOPE_PALETTE: same length, same shared hashing helper
# (_hash_to_palette_color), so a given telescope name resolves to the same index --
# and therefore the same hue family -- in both palettes. See TestTelescopeStripeContrast
# for the live luminance-band arithmetic proving one 8-color palette cannot clear 3:1
# against both white and NEUTRAL_SLOT_COLOR at once.
TELESCOPE_STRIPE_PALETTE = [
    '#8ac9ff',
    '#ffb370',
    '#33dba1',
    '#ffb524',
    '#f8bfce',
    '#5dea3e',
    '#c9bfe3',
    '#ffb09e',
]

# quick-260724-vb0: single source of truth for the stripe's outward-facing (left/top/
# bottom) opaque edge line, mirrored by the .cal-event-classical::before CSS rule in
# calendar.html. Deliberately absent on the fill-facing (right) edge -- every
# TELESCOPE_STRIPE_PALETTE entry already clears 3:1 against NEUTRAL_SLOT_COLOR there on
# its own, so no separator is needed on that side (see TestTelescopeStripeContrast).
STRIPE_OUTER_EDGE_COLOR = '#343a40'

# D-05: dedicated neutral slot for calendar events with no proposal code.
# Separate from PROPOSAL_PALETTE so an empty-string hash cannot accidentally
# collide with this value (see 09-RESEARCH Pitfall 1).
NEUTRAL_SLOT_COLOR = '#5a6268'

# D-06: human-readable label for classical-schedule (empty-proposal) legend entry.
CLASSICAL_SCHEDULE_LABEL = 'Classical schedule'

# Two title-prefix vocabularies live side by side here. The bracket-WORD prefixes are the
# legacy verbose vocabulary (the v1.3-era LCO/SOAR sync command's own prefixes, retired
# 34-02/D-18 but still emitted by load_telescope_runs.py and campaign_views.py), plus
# '[WEATHERED]' (D-03, campaign_views._RUN_STATUS_CALENDAR_PREFIX, Phase 23 Plan 02) -- both
# must stay byte-identical to their producers. The bracket-LETTER prefixes are the
# observation projector's own terse marker vocabulary (observation_projector.py, 34-01
# PROJ-03/D-02): '[X] ' (window expired), '[C] ' (cancelled), '[F] ' (failed) and '[?] '
# (inconsistent record -- projected with a half-set schedule, D-13; reads as terminal because
# an inconsistent record needs an operator's eye, not because anything actually failed).
# These four carry a trailing space deliberately, since this tuple is consumed by
# `title.startswith(p)` and a bare '[C]' would also match a hypothetical future
# '[COMPLETED]'-style bracket-word prefix. Terminal states: observations that reached an
# unrecoverable failure state (or, for '[?] ', a state that needs one). [QUEUED] is handled
# separately (its own branch below). Reconciling the two vocabularies into one is Phase 37's
# STATUS-01/02, not this phase's concern.
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X] ', '[C] ', '[F] ', '[?] ')

# Ordered legend vocabulary for observation_status_legend() (PROJ-03/D-02): every marker
# the observation projector (observation_projector.py, 34-01) can write, plus its
# human-readable label. Fixed and hand-maintained rather than derived from
# _TERMINAL_PREFIXES/status_border_css() -- deriving it would only let ring-vs-label drift
# in the other direction (a marker with a ring but no legend entry, say). Phase 37 owns the
# final wording of this vocabulary (STATUS-01/02); this is deliberately provisional.
_OBSERVATION_STATUS_LEGEND = (
    {'marker': '[Q]', 'label': 'Queued'},
    {'marker': '[S]', 'label': 'Scheduled'},
    {'marker': '[O]', 'label': 'Observed'},
    {'marker': '[X]', 'label': 'Window expired'},
    {'marker': '[C]', 'label': 'Cancelled'},
    {'marker': '[F]', 'label': 'Failed'},
    {'marker': '[?]', 'label': 'Inconsistent record'},
)


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

    Maps both title-prefix vocabularies (see ``_TERMINAL_PREFIXES`` above) to a box-shadow
    ring (D-08 resolved=box-shadow): the legacy verbose bracket-word prefixes from the
    retired sync command and campaign_views, and the observation projector's terse
    bracket-letter marker vocabulary (34-01 PROJ-03/D-02). The placed bucket ([UNVERIFIED],
    the projector's own '[S] '/'[O] ' markers, or no prefix at all) intentionally returns ''
    because Phase 8's D-09-reserved border treatment already owns the verified/fallback
    visual distinction — re-encoding it here would cause the two signals to merge into one
    style attribute branch instead of composing independently (09-RESEARCH Pitfall 3
    prevention). '[?] ' (the projector's inconsistent-record marker, D-13) reads as terminal
    here even though nothing actually failed — an inconsistent record needs an operator's
    eye, not a placed-looking chip.

    Args:
        title: CalendarEvent.title — may start with a known status prefix.

    Returns:
        A CSS fragment suitable for direct inclusion in a style attribute, e.g.
        'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'.  Returns '' for placed
        events.  The D-09-reserved border style is never emitted by this tag.
    """
    title = title or ''
    if title.startswith('[QUEUED] ') or title.startswith('[Q] '):
        return 'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'
    if any(title.startswith(p) for p in _TERMINAL_PREFIXES):
        # quick-260724-vb0: this ring is painted outside the chip's border box, so on
        # a classical chip its inner neighbour along the chip's left flank is the
        # stripe's outward-facing edge (STRIPE_OUTER_EDGE_COLOR). rgba(160, 0, 0, 0.55)
        # composited over the white day cell is #cb7373 (3.37:1 vs white on its own);
        # STRIPE_OUTER_EDGE_COLOR clears 3:1 against that composited color (see
        # TestTelescopeStripeContrast), which is what keeps the ring from merging into
        # a bare stripe entry (a bare entry there would measure ~1.87:1).
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


def _contrast_ratio(hex_a: str, hex_b: str) -> float:
    """Return the WCAG 2.1 contrast ratio between two '#rrggbb' colors.

    Implements the SC 1.4.11 (Non-text Contrast) / SC 1.4.3 (Contrast) formula:
    (L1 + 0.05) / (L2 + 0.05), where L1 is the lighter color's relative luminance
    and L2 the darker's -- order-independent and always >= 1.0. Source:
    https://www.w3.org/TR/WCAG21/#contrast-minimum (formula shared by 1.4.3 and 1.4.11).

    Args:
        hex_a: A '#rrggbb' hex color string.
        hex_b: A '#rrggbb' hex color string.

    Returns:
        The contrast ratio: 1.0 for identical/no-contrast colors, 21.0 for black
        against white.
    """
    lum_a, lum_b = _relative_luminance(hex_a), _relative_luminance(hex_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


@register.simple_tag
def text_color_for_bg(hex_color: str) -> str:
    """Return '#fff' or '#000' — whichever achieves WCAG AA 4.5:1 contrast against hex_color (DISPLAY-08).

    Uses _contrast_ratio against white. White text achieves 4.5:1 against any
    background with luminance <= 0.183; all PROPOSAL_PALETTE and NEUTRAL_SLOT_COLOR
    entries are dark, so '#fff' is returned for all current palette members.

    Args:
        hex_color: A '#rrggbb' hex color string (e.g. '#005f9e').

    Returns:
        '#fff' if white text achieves >= 4.5:1 contrast; '#000' otherwise.
    """
    white_contrast = _contrast_ratio(hex_color, '#ffffff')
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


@register.simple_tag
def neutral_slot_color() -> str:
    """Expose NEUTRAL_SLOT_COLOR to templates without a magic literal (quick-260724-osc).

    Lets calendar.html compare an already-computed bg_color to this value to decide
    whether an all-day event is classical-schedule (no proposal) vs. proposal-having,
    without re-deriving that distinction a second way.

    Returns:
        The NEUTRAL_SLOT_COLOR hex string.
    """
    return NEUTRAL_SLOT_COLOR


def _hash_to_palette_color(value: str, palette: list[str]) -> str:
    """Normalize a raw telescope-like string and hash it into a palette index.

    Shared by telescope_color and telescope_stripe_color (quick-260724-vb0) so a
    given telescope name always resolves to the same index regardless of which
    palette it indexes into -- TELESCOPE_PALETTE and TELESCOPE_STRIPE_PALETTE are
    parallel arrays for exactly this reason. Uses hashlib.sha256 for deterministic
    output across process restarts (see STATE.md Key Technical Notes -- the
    per-process-salted built-in is forbidden here). The return value is always a
    fixed palette constant or NEUTRAL_SLOT_COLOR, never derived from caller-supplied
    color data (T-vb0-01 mitigation).

    Args:
        value: Raw telescope string (may be blank, mixed-case, or have surrounding
            whitespace).
        palette: The palette list to index into.

    Returns:
        A hex color string from palette, or NEUTRAL_SLOT_COLOR for blank/missing
        input.
    """
    normalized = (value or '').strip().upper()
    if not normalized:
        return NEUTRAL_SLOT_COLOR
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return palette[int(digest, 16) % len(palette)]


@register.simple_tag
def telescope_color(telescope: str) -> str:
    """Return a deterministic hex color for a telescope name (quick-260724-osc).

    Delegates to the shared _hash_to_palette_color helper with TELESCOPE_PALETTE, which
    is gated against #ffffff -- the legend chip that consumes this tag's output is the
    only place TELESCOPE_PALETTE is rendered, and its only background is the page's
    white background (quick-260724-vb0). See telescope_stripe_color for the parallel
    stripe-palette tag that resolves the same telescope name to the same index in a
    different, fill-gated palette.

    Args:
        telescope: Raw telescope string from CalendarEvent.telescope (may be blank,
            mixed-case, or have surrounding whitespace).

    Returns:
        A hex color string from TELESCOPE_PALETTE, or NEUTRAL_SLOT_COLOR for
        blank/missing telescopes.
    """
    return _hash_to_palette_color(telescope, TELESCOPE_PALETTE)


@register.simple_tag
def telescope_stripe_color(telescope: str) -> str:
    """Return a deterministic hex color from TELESCOPE_STRIPE_PALETTE (quick-260724-vb0).

    Delegates to the same _hash_to_palette_color helper as telescope_color, so a
    telescope name always resolves to the same index in both palettes -- one hue
    family shared by the legend chip and the stripe, even though the two palettes
    hold different hex values tuned for their own background (TELESCOPE_PALETTE
    against white for the legend chip; TELESCOPE_STRIPE_PALETTE against
    NEUTRAL_SLOT_COLOR for the stripe's fill neighbour). See TestTelescopeStripeContrast
    for the arithmetic proving one palette cannot serve both.

    Args:
        telescope: Raw telescope string from CalendarEvent.telescope (may be blank,
            mixed-case, or have surrounding whitespace).

    Returns:
        A hex color string from TELESCOPE_STRIPE_PALETTE, or NEUTRAL_SLOT_COLOR for
        blank/missing telescopes.
    """
    return _hash_to_palette_color(telescope, TELESCOPE_STRIPE_PALETTE)


@register.simple_tag
def visible_classical_telescopes(weeks) -> list[dict]:
    """Compute the set of telescopes visible in the currently-rendered month, classical-schedule only.

    Mirrors visible_proposals's weeks iteration and dual dict/attribute day support, but
    scoped to classical-schedule events only (empty proposal) — proposal-having events
    already encode identity via their proposal fill and are excluded here even when their
    telescope field is set. Groups by resulting color so hash-colliding telescopes share
    one legend entry, same collision handling as visible_proposals.

    Args:
        weeks: The weeks context list passed to calendar.html — a list of lists of day
            objects, each with .all_day_events and .events attributes containing objects
            with .proposal and .telescope attributes.

    Returns:
        List of dicts with keys 'color' (hex string), 'codes' (sorted list of normalized
        telescope strings), and 'label' (comma-joined string for display). Sorted by
        color hex ascending.
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
                normalized_proposal = (event.proposal or '').strip().upper()
                if normalized_proposal:
                    continue  # only classical-schedule (empty-proposal) events contribute
                normalized_telescope = (event.telescope or '').strip().upper()
                color = telescope_color(event.telescope)
                by_color[color].add(normalized_telescope)

    result = []
    for color, codes in sorted(by_color.items()):
        result.append(
            {
                'color': color,
                'codes': sorted(codes),
                'label': ', '.join(sorted(codes)),
            }
        )

    return result


@register.simple_tag
def observation_status_legend() -> list[dict]:
    """Return the fixed, ordered observation-status marker legend (PROJ-03, D-02).

    Exposes ``_OBSERVATION_STATUS_LEGEND`` to calendar.html so a calendar visitor can read
    what every observation-projector marker (``[Q]``/``[S]``/``[O]``/``[X]``/``[C]``/``[F]``/
    ``[?]``) means directly off the page, without needing this module's source. Deliberately
    a fixed vocabulary rather than data-driven — making it read from the database would only
    let it drift out of sync with ``status_border_css()``'s own prefix matching. Phase 37
    owns the final wording of this vocabulary (STATUS-01/02); this is provisional.

    Takes no arguments, reads nothing from the database, and never raises.

    Returns:
        list[dict]: one ``{'marker': ..., 'label': ...}`` dict per marker, in the fixed
        order ``[Q]``, ``[S]``, ``[O]``, ``[X]``, ``[C]``, ``[F]``, ``[?]``.
    """
    return [dict(entry) for entry in _OBSERVATION_STATUS_LEGEND]


@register.simple_tag
def campaign_decoration(event: CalendarEvent) -> dict | None:
    """Read-only campaign attribution decoration for a CalendarEvent (ANNOT-02, D-10/D-11/D-13/D-14).

    Renders the campaign an event is attributed to from ``CalendarEventMeta.run`` at
    request time -- never from a value written into the event's own fields -- so a
    base-layer re-projection of this event's title/description cannot erase the decoration.
    Reads only: never calls ``.save()``, ``.update()``, ``.create()`` or ``get_or_create()``,
    and never imports the views module or the SPICE-kernel-loading ephemeris module.

    Never raises. Returns ``None`` for an event with no companion row (guards the reverse
    one-to-one ``telescope_label_meta`` accessor against ``ObjectDoesNotExist``), for a
    companion row whose ``run`` is unset, and for a run that is not publicly visible
    (``CampaignRun.is_publicly_visible`` -- keeps a pending-review run's campaign name off
    the public calendar).

    Args:
        event: the CalendarEvent to decorate.

    Returns:
        dict | None: exactly the keys ``campaign_name``, ``run_pk``, ``table_url``,
        ``telescope_instrument``, ``window_start``, ``window_end`` and
        ``run_status_display``, or ``None``. Never exposes any PII contact field or the
        run's provenance-only ingest field -- those stay behind their existing
        staff/PII gates.

        Rendered consumers (Phase 33 Plan 06, IN-01): ``run_pk`` is consumed by
        ``campaign_chip.html``'s no-campaign tooltip and accessible name -- the only
        place this key is rendered. ``table_url`` doubles as the campaign-presence
        signal that partial branches on (it is set only inside ``if run.campaign_id is
        not None`` above, so ``table_url is None`` is equivalent to "no campaign").
    """
    if not isinstance(event, CalendarEvent):
        # Rule 1 fix (33-11): the create-event form context has no `event` key at all --
        # Django's template engine resolves the missing variable to the empty-string
        # invalid-variable placeholder rather than raising, so this tag can be invoked
        # with a non-CalendarEvent value. The docstring promises "never raises"; guard it
        # here rather than requiring every template call site to wrap this tag in
        # `{% if event %}`.
        return None
    try:
        meta = event.telescope_label_meta
    except ObjectDoesNotExist:
        return None
    run = meta.run
    if run is None or not run.is_publicly_visible:
        return None

    table_url = None
    if run.campaign_id is not None:
        # Built with reverse() in Python, not {% url %} in the template (RESEARCH.md
        # Pitfall 1): 'campaigns:table' resolves against path('<int:pk>/', ...), and a null
        # campaign pk raises NoReverseMatch, which on the public calendar is a whole-page
        # failure -- is_publicly_visible alone does not cover campaign nullness.
        table_url = f"{reverse('campaigns:table', args=[run.campaign_id])}#run-{run.pk}"

    return {
        'campaign_name': run.campaign.name if run.campaign_id is not None else NO_CAMPAIGN_LABEL,
        'run_pk': run.pk,
        'table_url': table_url,
        'telescope_instrument': run.telescope_instrument,
        'window_start': run.window_start,
        'window_end': run.window_end,
        'run_status_display': run.get_run_status_display(),
    }


def _window_start_or_max(record) -> datetime:
    """Return record_time_window(record)[0], or a UTC-attached datetime.max on failure.

    Shared sort key for observation_series_decoration(): a sibling whose window cannot be
    derived (half-set schedule mid-projection, malformed parameters) sorts last rather than
    raising and taking the whole modal down with it -- the spike's own rule, reimplemented
    here rather than imported across modules (RESEARCH.md, no cross-module private import).

    Args:
        record: the ObservationRecord being sorted.

    Returns:
        datetime: a timezone-aware UTC datetime, always comparable to every other member's.
    """
    try:
        start, _ = record_time_window(record)
    except (KeyError, ValueError):
        return datetime.max.replace(tzinfo=dt_timezone.utc)
    return start


@register.simple_tag
def observation_series_decoration(event: CalendarEvent) -> dict | None:
    """Read-only "night n of N" series decoration for a CalendarEvent (PROJ-04/PROJ-05, D-04).

    Renders which night of how many an observation-projector-owned event's own record is,
    within its ObservationGroup, from ``CalendarEventMeta.observation_group`` /
    ``.observation_record`` at request time -- never from text written into the event's own
    title or description -- so a base-layer re-projection of this event cannot erase the
    decoration. Mirrors ``campaign_decoration()`` immediately above: same isinstance guard,
    same ``ObjectDoesNotExist`` guard, same reverse()-in-Python rule, same fixed-key return
    dict, same "never expose PII or provenance" discipline.

    Reads only -- performs no database write of any kind (no save, no bulk update, no
    row creation, no find-or-create call); never imports ``solsys_code.views`` or
    ``solsys_code.ephem_utils``.

    Never raises. Returns ``None`` for an event with no companion row, for a companion row
    with no ``observation_group`` or no ``observation_record`` link, for a group with fewer
    than two members (a series of one is not a series), and for a value that is not a
    CalendarEvent at all.

    Args:
        event: the CalendarEvent to decorate.

    Returns:
        dict | None: exactly the keys ``group_name``, ``group_pk``, ``index`` (1-based
        position of this event's own record within the group, ordered by window start then
        pk), ``size``, ``group_list_url`` and ``record_url``, or ``None``. Exposes no
        campaign, contact, submitter or provenance field -- this tag renders observation-group
        identity only.
    """
    if not isinstance(event, CalendarEvent):
        # Same reasoning as campaign_decoration(): the create-event form context has no
        # `event` key, and Django resolves the missing variable to the invalid-variable
        # placeholder rather than raising, so this tag can be called with a non-event value.
        return None
    try:
        meta = event.telescope_label_meta
    except ObjectDoesNotExist:
        return None
    if meta.observation_group_id is None or meta.observation_record_id is None:
        return None

    members = list(meta.observation_group.observation_records.select_related('target'))
    if len(members) < 2:
        return None

    members.sort(key=lambda member: (_window_start_or_max(member), member.pk))
    index = None
    for position, member in enumerate(members, start=1):
        if member.pk == meta.observation_record_id:
            index = position
            break
    if index is None:
        # This event's own record fell out of the group between the meta read above and
        # this loop (e.g. concurrent membership change) -- never raise on a request-time
        # decoration; render nothing rather than a broken "n of N".
        return None

    return {
        'group_name': meta.observation_group.name,
        'group_pk': meta.observation_group_id,
        'index': index,
        'size': len(members),
        'group_list_url': reverse('tom_observations:group-list'),
        'record_url': reverse('tom_observations:detail', args=[meta.observation_record_id]),
    }
