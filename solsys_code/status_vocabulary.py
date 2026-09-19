"""Single definition of every calendar status marker for FOMO (STATUS-01/02).

Before this module, three independent prefix vocabularies had to be kept "byte-identical"
to each other by convention: ``observation_projector.py``'s ``_STAGE_MARKER``/
``_FAILURE_MARKER_BY_STATUS``, ``campaign_reconciler.py``'s ``RUN_STATUS_CALENDAR_PREFIX``
(re-imported by ``allocation_projector.py``), and ``calendar_display_extras.py``'s
``_TERMINAL_PREFIXES``/``_OBSERVATION_STATUS_LEGEND``. This module is now the one place a
marker, a label, a legend entry or a terminal-state classification is defined; every other
module reads from it and holds no marker table of its own.

TALLY-03 invariant: nothing in this module, and nothing that reads from it, ever writes
``CampaignRun.run_status``. That field stays a staff decision made only through the
existing approval-queue/decision views -- this module only classifies and labels state that
already exists, it never sets it.

Mirrors ``campaign_gap.py``'s import discipline (``solsys_code/campaign_gap.py:1-14``): this
module must never import ``solsys_code.views`` or ``solsys_code.ephem_utils`` (or any module
that imports either) at module scope, because ``observation_projector``,
``campaign_reconciler``, ``allocation_projector`` and ``calendar_display_extras`` all import
this module, and those heavy modules trigger a ~1.6 GB SPICE-kernel download on import
(CLAUDE.md "Heavy import side effect") that every one of those callers would otherwise pay.
"""

from typing import Any

from solsys_code.models import CampaignRun


class OCSState:
    """Canonical LCO/SOAR observing-portal state names (D-05).

    These are the state strings the LCO Observation Portal API and TOM Toolkit's
    ``get_terminal_observing_states()``/``get_failed_observing_states()`` use verbatim.
    Note the portal's own single-L spelling of ``CANCELED``.
    """

    PENDING = 'PENDING'
    COMPLETED = 'COMPLETED'
    WINDOW_EXPIRED = 'WINDOW_EXPIRED'
    CANCELED = 'CANCELED'
    FAILURE_LIMIT_REACHED = 'FAILURE_LIMIT_REACHED'
    NOT_ATTEMPTED = 'NOT_ATTEMPTED'


class DisplayState:
    """The nine calendar-visible display states (D-01/D-04)."""

    QUEUED = 'QUEUED'
    SCHEDULED = 'SCHEDULED'
    OBSERVED = 'OBSERVED'
    WINDOW_EXPIRED = 'WINDOW_EXPIRED'
    CANCELLED = 'CANCELLED'
    FAILED = 'FAILED'
    WEATHERED = 'WEATHERED'
    INCONSISTENT = 'INCONSISTENT'
    UNUSED = 'UNUSED'


# D-01/D-02: display state -> the one marker every producer writes into a title.
MARKER: dict[str, str] = {
    DisplayState.QUEUED: '[Q]',
    DisplayState.SCHEDULED: '[S]',
    DisplayState.OBSERVED: '[O]',
    DisplayState.WINDOW_EXPIRED: '[X]',
    DisplayState.CANCELLED: '[C]',
    DisplayState.FAILED: '[F]',
    DisplayState.WEATHERED: '[W]',
    DisplayState.INCONSISTENT: '[?]',
    DisplayState.UNUSED: '[U]',
}

# D-03: the placed-but-unobserved state is named "Scheduled" (the LCO portal's own word) in
# every label, the runbook and every docstring -- "placed" stays an internal code word only.
LABEL: dict[str, str] = {
    DisplayState.QUEUED: 'Queued',
    DisplayState.SCHEDULED: 'Scheduled',
    DisplayState.OBSERVED: 'Observed',
    DisplayState.WINDOW_EXPIRED: 'Window expired',
    DisplayState.CANCELLED: 'Cancelled',
    DisplayState.FAILED: 'Failed',
    DisplayState.WEATHERED: 'Weather/technical failure',
    DisplayState.INCONSISTENT: 'Inconsistent record',
    DisplayState.UNUSED: 'Unused awarded night',
}

# D-04: one legend, every visible state, one fixed order -- never derived from a database
# query and never derived from the ring buckets below (deriving it would only let
# ring-vs-label drift in the other direction: a marker with a ring but no legend entry).
LEGEND: tuple[dict[str, str], ...] = tuple(
    {'marker': MARKER[state], 'label': LABEL[state]}
    for state in (
        DisplayState.QUEUED,
        DisplayState.SCHEDULED,
        DisplayState.OBSERVED,
        DisplayState.WINDOW_EXPIRED,
        DisplayState.CANCELLED,
        DisplayState.FAILED,
        DisplayState.WEATHERED,
        DisplayState.INCONSISTENT,
        DisplayState.UNUSED,
    )
)

# The observation projector's stage table (moved verbatim in value from
# observation_projector._STAGE_MARKER), now expressed through MARKER rather than re-typed
# literals.
STAGE_MARKER: dict[str, str] = {
    'queued': MARKER[DisplayState.QUEUED],
    'placed': MARKER[DisplayState.SCHEDULED],
    'observed': MARKER[DisplayState.OBSERVED],
    'completed-no-block': MARKER[DisplayState.OBSERVED],
}

# The observation projector's failure-status table (moved verbatim in value from
# observation_projector._FAILURE_MARKER_BY_STATUS), now expressed through MARKER and
# OCSState. An unrecognised failure state still falls back to the failed marker rather than
# being silently unmarked -- if the facility ever adds a fifth failure state, update this
# table.
FAILURE_MARKER_BY_STATUS: dict[str, str] = {
    OCSState.WINDOW_EXPIRED: MARKER[DisplayState.WINDOW_EXPIRED],
    OCSState.CANCELED: MARKER[DisplayState.CANCELLED],
    OCSState.FAILURE_LIMIT_REACHED: MARKER[DisplayState.FAILED],
    OCSState.NOT_ATTEMPTED: MARKER[DisplayState.FAILED],
}

# D-02: the only two CampaignRun.RunStatus values that ever surface a calendar marker. A
# staff-cancelled run night gets the same [C] a portal-cancelled record gets -- [C] means
# "cancelled, by whoever owns this event"; the pop-up's own "Run status:" line is what
# distinguishes the two layers. Every other RunStatus value (REQUESTED, PLANNED, OBSERVED,
# REDUCED, PUBLISHED, NOT_AWARDED) gains no marker. Importing CampaignRun from
# solsys_code.models at module scope is safe here: models.py imports
# campaign_reconciler/allocation_projector lazily inside a signal receiver, so there is no
# module-scope import cycle.
RUN_STATUS_MARKER: dict[str, str] = {
    CampaignRun.RunStatus.CANCELLED: MARKER[DisplayState.CANCELLED],
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: MARKER[DisplayState.WEATHERED],
}

# status_border_css()'s two ring buckets. SCHEDULED, OBSERVED and UNUSED are in neither
# bucket -- D-13 forbids a ring for UNUSED (it is a chip style + text token, never a ring).
RING_QUEUED_STATES: frozenset[str] = frozenset({DisplayState.QUEUED})
RING_TERMINAL_STATES: frozenset[str] = frozenset(
    {
        DisplayState.WINDOW_EXPIRED,
        DisplayState.CANCELLED,
        DisplayState.FAILED,
        DisplayState.WEATHERED,
        DisplayState.INCONSISTENT,
    }
)

# Migration complete (Phase 37 Plan 07): the legacy bracket-WORD title prefixes
# (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]`) this module used to also recognise
# as a temporary retirement list (`RETIRED_TITLE_PREFIXES`) are gone -- a re-title sweep
# (`reconcile_campaign_runs` + `project_observation_calendar`) proved the developer
# database held none of them, so there is exactly one spelling of every marker now. A
# deployment that has not yet run both sweeps re-titles itself within one unattended tick,
# because both sweeps run every tick (`solsys_code/unattended.py` `STEPS`) -- see the
# runbook's "One-time title change" note.
_MARKER_STATE: dict[str, str] = {marker: state for state, marker in MARKER.items()}


def state_for_title(title: str | None) -> str | None:
    """Resolve a stored CalendarEvent title's leading marker to a display state.

    Used by the status ring (``calendar_display_extras.status_border_css()``): matches a
    marker followed by a space (the trailing-space rule, so ``'[C] '`` cannot match a
    hypothetical future ``'[COMPLETED]'``-style word prefix), else returns ``None``.

    Args:
        title: a ``CalendarEvent.title`` value, or ``None``.

    Returns:
        str | None: the matching ``DisplayState`` member, or ``None`` if the title starts
            with no known marker. Never raises.
    """
    if not title:
        return None
    for marker, state in _MARKER_STATE.items():
        if title.startswith(f'{marker} '):
            return state
    return None


# D-05: FOMO-side override table -- a facility absent from this table is trusted to use the
# canonical LCO/SOAR OCS vocabulary; a facility present here is not. This is where a future
# facility-vocabulary change lands.
OBSERVED_STATES_BY_FACILITY: dict[str, frozenset[str]] = {
    # tom_gemini is a limited, ToO-only subset: its "terminal" states (TRIGGERED, ON_HOLD)
    # mean the ToO was submitted, never that it was observed, and may change with future GPP
    # support.
    'GEM': frozenset(),
    # ESO has no Phase 2 read-back vocabulary yet.
    'ESO': frozenset(),
}


def failed_states_for(facility: Any) -> frozenset[str]:
    """Return this facility's own failed-observing-state set, unmapped.

    Args:
        facility: a TOM Toolkit facility service instance.

    Returns:
        frozenset[str]: ``frozenset(facility.get_failed_observing_states())``.
    """
    return frozenset(facility.get_failed_observing_states())


def observed_states_for(facility: Any) -> frozenset[str]:
    """Return the set of this facility's statuses that mean an observation happened.

    D-05: the LCO/SOAR OCS vocabulary is canonical -- a facility absent from
    ``OBSERVED_STATES_BY_FACILITY`` is trusted to use it, and its observed set is its own
    terminal states minus its own failed states. A facility present in the table (currently
    ``GEM``/``ESO``) is not trusted at all: its terminal states mean "submitted", never
    "observed", so its override is an empty set. Reads the facility's name defensively (a
    plain ``getattr`` with the class name as fallback) so an unexpected facility object
    cannot raise inside a template-time caller.

    Args:
        facility: a TOM Toolkit facility service instance
            (``facility.get_terminal_observing_states()``/``.get_failed_observing_states()``).

    Returns:
        frozenset[str]: the observed-state set for this facility. Never raises.
    """
    name = getattr(facility, 'name', None) or type(facility).__name__
    if name in OBSERVED_STATES_BY_FACILITY:
        return OBSERVED_STATES_BY_FACILITY[name]
    return frozenset(facility.get_terminal_observing_states()) - failed_states_for(facility)


_FAILURE_STATE_BY_OCS_STATE: dict[str, str] = {
    OCSState.WINDOW_EXPIRED: DisplayState.WINDOW_EXPIRED,
    OCSState.CANCELED: DisplayState.CANCELLED,
    OCSState.FAILURE_LIMIT_REACHED: DisplayState.FAILED,
    OCSState.NOT_ATTEMPTED: DisplayState.FAILED,
}


def classify_record(record: Any, facility: Any) -> str:
    """Classify an ObservationRecord's lifecycle stage into a DisplayState (D-05).

    Makes no network call and never raises for messy record data: a half-set
    ``scheduled_start``/``scheduled_end`` pair is ``INCONSISTENT``; a status in the
    facility's failed set maps through the OCS-state keys to
    ``WINDOW_EXPIRED``/``CANCELLED``/``FAILED`` (``FAILED`` is the fallback for an
    unrecognised failure state); a status in ``observed_states_for(facility)`` is
    ``OBSERVED``; a record with both schedule fields set is ``SCHEDULED``; anything else is
    ``QUEUED``.

    Args:
        record: the ObservationRecord being classified (reads ``scheduled_start``,
            ``scheduled_end`` and ``status``).
        facility: the record's facility instance.

    Returns:
        str: one of the ``DisplayState`` members above. Never raises.
    """
    has_start = record.scheduled_start is not None
    has_end = record.scheduled_end is not None
    if has_start != has_end:
        return DisplayState.INCONSISTENT
    has_block = has_start and has_end
    failed = failed_states_for(facility)
    if record.status in failed:
        return _FAILURE_STATE_BY_OCS_STATE.get(record.status, DisplayState.FAILED)
    if record.status in observed_states_for(facility):
        return DisplayState.OBSERVED
    if has_block:
        return DisplayState.SCHEDULED
    return DisplayState.QUEUED
