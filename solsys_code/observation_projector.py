"""FOMO's observation projector: draws and keeps current one CalendarEvent per LCO/SOAR
ObservationRecord, with no operator command.

Implements PROJ-01 (one CalendarEvent per record, keyed by the facility's own observation
URL), PROJ-02 (the event's span narrows from request window to placed block to observed
block as the record's own fields change), TRIG-01 (a post_save signal trigger, not TOM's
observation_change_state hook, which is silent for a schedule-only placement save) and
TRIG-02 (the receiver runs inline in the caller's transaction, makes no network call, and
never raises out of a save).

Three ownership rules hold across every function in this module:

1. It owns only ``CalendarEvent`` rows whose ``url`` is a facility observation URL
   (``facility.get_observation_url(record.observation_id)``) -- never a ``RUN:`` reconciler
   event, a ``GEM:`` Gemini echo event, or a blank-url classical event.
2. It writes exactly ``CalendarEventMeta.observation_record``, ``.observation_group`` and
   ``.is_verified`` on its own event's companion row -- the campaign attribution link and
   its two human-confirmation stamp fields are written only by the Phase 28 attribution
   queue and must survive a projection untouched.
3. It never raises out of a signal receiver -- a calendar-layer fault must never cost an
   operator their observation record.
"""

import logging
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from tom_observations.facility import get_service_class
from tom_observations.models import ObservationGroup, ObservationRecord

from solsys_code.calendar_utils import (
    InstrumentExtractionError,
    coarse_telescope_label,
    extract_instrument,
    insert_or_create_calendar_event,
    record_time_window,
)
from solsys_code.models import CalendarEventMeta

logger = logging.getLogger(__name__)

# PROJ-01/D-16: the only facilities this module owns. Gemini has no queue read-back
# (GEMFacility's status/URL methods are stubs) and stays with the submission-echo command.
PROJECTED_FACILITIES = ('LCO', 'SOAR')

# One facility instance per record.facility value -- never a single shared instance across
# LCO and SOAR (the promote-decision invariant this plan's assumption_delta_decision names).
_facilities: dict[str, Any] = {}


def facility_for(record: ObservationRecord) -> Any:
    """Return the cached facility instance for a record's facility, creating one if absent.

    Args:
        record: the ObservationRecord whose ``facility`` selects the instance.

    Returns:
        Any: the facility service instance for this record's facility (e.g. the LCO or SOAR
            service class), one per distinct ``record.facility`` value, never shared across
            two different facility names.
    """
    name = record.facility
    if name not in _facilities:
        _facilities[name] = get_service_class(name)()
    return _facilities[name]


def reset_facility_cache() -> None:
    """Clear the cached facility instances (test-only helper).

    Used between test cases so a facility class patched in one test (e.g. a monkeypatched
    ``get_observation_status``) cannot leak a stale cached instance into the next.
    """
    _facilities.clear()


# D-02: a hand-typed snapshot of the four failure states get_failed_observing_states()
# returns today. An unrecognised failure state still falls back to '[F]' rather than being
# silently unmarked -- if the facility ever adds a fifth failure state, update this table.
_FAILURE_MARKER_BY_STATUS = {
    'WINDOW_EXPIRED': '[X]',
    'CANCELED': '[C]',
    'FAILURE_LIMIT_REACHED': '[F]',
    'NOT_ATTEMPTED': '[F]',
}

_STAGE_MARKER = {
    'queued': '[Q]',
    'placed': '[S]',
    'observed': '[O]',
    'completed-no-block': '[O]',
}


def _failure_marker(status: str, facility: Any) -> str | None:
    """Return the D-02 failure marker for a status, or None if it is not a failure state."""
    if status not in set(facility.get_failed_observing_states()):
        return None
    return _FAILURE_MARKER_BY_STATUS.get(status, '[F]')


def stage_for(record: ObservationRecord, facility: Any) -> str:
    """Classify a record's lifecycle stage from its own fields alone -- no network call.

    Args:
        record: the ObservationRecord being classified.
        facility: the record's facility instance (``facility_for(record)``), used to read
            the failed/terminal observing-state vocabularies.

    Returns:
        str: one of 'inconsistent' (half-set scheduled_start/scheduled_end, D-13),
            'terminal-negative' (status in the facility's failed states, D-11), 'observed'
            or 'completed-no-block' (status in terminal-minus-failed, with or without a
            placed block, D-12), or 'placed'/'queued' (with or without a placed block).
    """
    has_start = record.scheduled_start is not None
    has_end = record.scheduled_end is not None
    if has_start != has_end:
        return 'inconsistent'
    has_block = has_start and has_end
    failed = set(facility.get_failed_observing_states())
    successful = set(facility.get_terminal_observing_states()) - failed
    if record.status in failed:
        return 'terminal-negative'
    if record.status in successful:
        return 'observed' if has_block else 'completed-no-block'
    return 'placed' if has_block else 'queued'


def telescope_token(record: ObservationRecord, stage: str, instrument: str) -> str:
    """Return the telescope token used in both the title and CalendarEvent.telescope.

    Always the coarse aperture class today. This is the point plan 34-02 extends with the
    observed-telescope token (D-07) once a record reaches a successful terminal state.

    Args:
        record: the ObservationRecord being projected.
        stage: the record's classified stage (``stage_for(record, facility)``); unused for
            now, kept in the signature as the extension point plan 34-02 branches on.
        instrument: the record's extracted instrument string (``extract_instrument()``).

    Returns:
        str: the coarse aperture-class label (``calendar_utils.coarse_telescope_label()``).
    """
    return coarse_telescope_label(instrument, record.facility)


def title_for(record: ObservationRecord, stage: str, token: str, target_name: str, facility: Any) -> str:
    """Build the D-01 event title: exactly one marker, always present (D-03).

    Args:
        record: the ObservationRecord being titled.
        stage: the record's classified stage.
        token: the telescope token (``telescope_token()``).
        target_name: ``record.target.name``.
        facility: the record's facility instance, used to resolve a failure marker.

    Returns:
        str: ``f'{marker} {token} {target_name}'`` truncated to 200 characters. A failure
            marker wins over a stage marker; an unrecognised stage falls back to '[?]'.
    """
    marker = _failure_marker(record.status, facility)
    if marker is None:
        marker = _STAGE_MARKER.get(stage, '[?]')
    return f'{marker} {token} {target_name}'[:200]


def event_url(record: ObservationRecord, facility: Any) -> str:
    """Return this record's facility observation URL -- the projector's only identity key.

    Args:
        record: the ObservationRecord being projected.
        facility: the record's facility instance.

    Returns:
        str: ``facility.get_observation_url(record.observation_id)``.
    """
    return facility.get_observation_url(record.observation_id)


def series_group_for(record: ObservationRecord) -> ObservationGroup | None:
    """Return the lowest-pk ObservationGroup this record belongs to, or None.

    Args:
        record: the ObservationRecord being projected.

    Returns:
        ObservationGroup | None: the group with the lowest pk among every group containing
            this record, or None if it belongs to no group. Lowest pk wins deterministically
            when a record is in more than one group.
    """
    return ObservationGroup.objects.filter(observation_records=record).order_by('pk').first()


def event_fields_for(record: ObservationRecord, facility: Any) -> tuple[dict[str, Any], str]:
    """Build the CalendarEvent field values for a record; raises if it cannot be projected.

    Args:
        record: the ObservationRecord being projected.
        facility: the record's facility instance.

    Returns:
        tuple[dict[str, Any], str]: the field dict (title, description, start_time,
            end_time, telescope, instrument, proposal, target_list) and the classified
            stage string.

    Raises:
        InstrumentExtractionError: if ``extract_instrument()`` finds no usable config.
        KeyError: if the record has no usable request window (missing
            ``parameters['start']``/``['end']``).
        ValueError: if ``parameters['start']``/``['end']`` cannot be parsed as datetimes, or
            (for a stage other than 'inconsistent') if the record's schedule fields are
            otherwise unusable per ``record_time_window()``'s own raising contract.
    """
    instrument = extract_instrument(record.parameters)
    if instrument is None:
        raise InstrumentExtractionError(
            f'No recognized configuration_type or exposure signal found in observation_id='
            f'{record.observation_id!r} parameters'
        )
    stage = stage_for(record, facility)
    token = telescope_token(record, stage, instrument)
    target_name = record.target.name
    if stage == 'inconsistent':
        # D-13: an inconsistent record is still projectable -- span the request window
        # directly rather than through record_time_window(), which raises for a half-set
        # schedule. A missing/unparsable key here raises through to project_record's catch,
        # correctly marking the record unprojectable.
        start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
        end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
    else:
        start_time, end_time = record_time_window(record)
    proposal = record.parameters.get('proposal', '')
    target_list = record.target.targetlist_set.order_by('name').first()
    description = (
        f'Proposal: {proposal}\n'
        f'Status: {record.status}\n'
        f'Stage: {stage}\n'
        f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime("%Y-%m-%dT%H:%M:%S")}'
    )
    fields = {
        'title': title_for(record, stage, token, target_name, facility),
        'description': description,
        'start_time': start_time,
        'end_time': end_time,
        'telescope': token,
        'instrument': instrument,
        'proposal': proposal,
        'target_list': target_list,
    }
    return fields, stage


def write_event_meta(event: Any, record: ObservationRecord) -> None:
    """Create or update ``event``'s CalendarEventMeta row with this record's own links.

    First clears any stale one-to-one claim on ``observation_record`` from a companion row
    that still points at a different event (the takeover sweep meets rows exactly like
    this), then writes exactly ``is_verified``/``observation_record``/``observation_group``
    -- never the campaign attribution link or its confirmation stamps.

    Args:
        event: the CalendarEvent this record was just projected onto.
        record: the ObservationRecord being projected.
    """
    CalendarEventMeta.objects.filter(observation_record=record).exclude(event_id=event.pk).update(
        observation_record=None
    )
    CalendarEventMeta.objects.update_or_create(
        event=event,
        defaults={
            'is_verified': True,
            'observation_record': record,
            'observation_group': series_group_for(record),
        },
    )


def project_record(record: ObservationRecord) -> tuple[str, str]:
    """Create/update/leave-unchanged the record's event. Never raises (TRIG-02).

    Args:
        record: the ObservationRecord being projected.

    Returns:
        tuple[str, str]: (action, stage) where action is one of
            ``insert_or_create_calendar_event()``'s 'created'/'updated'/'unchanged', or
            'unprojectable' (stage then carries the caught exception's class name).
    """
    facility = facility_for(record)
    try:
        fields, stage = event_fields_for(record, facility)
    except Exception as exc:  # noqa: BLE001 -- a projector must never break the triggering save
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, type(exc).__name__)
        return 'unprojectable', type(exc).__name__
    event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
    write_event_meta(event, record)
    return action, stage


def receiver_on_record_save(sender: Any, instance: ObservationRecord, created: bool, raw: bool, **kwargs: Any) -> None:
    """post_save receiver (TRIG-01): projects a record's event with no operator command.

    Returns immediately for a fixture load (``raw=True``) and for any facility other than
    LCO/SOAR (D-16, Gemini records stay with the submission-echo command). The call to
    ``project_record()`` is wrapped in its own try/except so even an unexpected ORM error
    cannot abort the caller's save (TRIG-02); logged at debug level, not warning, since this
    fires on every ObservationRecord save in production.

    Args:
        sender: the model class Django's signal framework passes (ObservationRecord).
        instance: the ObservationRecord that was just saved.
        created: True if this save created a new row.
        raw: True if this save came from a fixture load (``loaddata``).
        **kwargs: the remaining signal kwargs (``using``, ``update_fields``), unused.
    """
    if raw:
        return
    if instance.facility not in PROJECTED_FACILITIES:
        return
    try:
        action, stage = project_record(instance)
    except Exception as exc:  # noqa: BLE001 -- TRIG-02: never abort the caller's save
        logger.warning(
            'receiver_on_record_save failed for observation_id=%r: %s', instance.observation_id, type(exc).__name__
        )
        return
    logger.debug(
        'post_save projected observation_id=%r created=%s -> %s (%s)',
        instance.observation_id,
        created,
        action,
        stage,
    )
