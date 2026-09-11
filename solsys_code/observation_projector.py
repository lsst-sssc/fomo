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
from collections.abc import Callable
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from tom_calendar.models import CalendarEvent
from tom_observations.facility import get_service_class
from tom_observations.models import ObservationGroup, ObservationRecord

from solsys_code.calendar_utils import (
    InstrumentExtractionError,
    coarse_telescope_label,
    derive_telescope,
    extract_instrument,
    insert_or_create_calendar_event,
    preview_calendar_event_action,
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


def observed_token(record: ObservationRecord) -> str | None:
    """Return the D-07 observed-telescope token stored on this record, or None.

    Reads the site/telescope the sweep's one-time lookup stored on
    ``record.parameters`` (plan 34-02 Task 3's ``resolve_observed_site()``) and maps them
    through ``calendar_utils.derive_telescope()`` -- the same function ``telescope_token()``
    would otherwise use for any other stage. Makes no network call: this is a pure read of
    already-stored data, never a live lookup.

    Args:
        record: the ObservationRecord being projected.

    Returns:
        str | None: the observed-telescope label (e.g. 'FTN', or a SITE-aperture label for a
            1m0/0m4 site) if both keys are stored and map to a known site, else None --
            ``derive_telescope()`` is already None-safe on both a missing key and an
            unmapped pair.
    """
    site = record.parameters.get('observed_site')
    telescope = record.parameters.get('observed_telescope')
    return derive_telescope(site, telescope)


# D-07: only these two stages ever read the stored observed-telescope token -- a record
# still queued or placed has nothing to read yet (the lookup only ever fires at a
# successful-terminal stage, plan 34-02 Task 3), so it always falls through to the coarse
# label below.
_OBSERVED_TOKEN_STAGES = ('observed', 'completed-no-block')


def telescope_token(record: ObservationRecord, stage: str, instrument: str) -> str:
    """Return the telescope token used in both the title and CalendarEvent.telescope.

    For an 'observed'/'completed-no-block' record whose observed site has already been
    resolved (D-07), returns that observed-telescope token instead of the coarse aperture
    class -- e.g. 'FTN' rather than '2m0'. Every other stage, and a successful-terminal
    record whose lookup has not (yet) succeeded, keeps the coarse aperture-class label as
    its standing fallback -- the same label used throughout the record's lifecycle up to
    that point, so the token never regresses to something coarser once observed.

    Args:
        record: the ObservationRecord being projected.
        stage: the record's classified stage (``stage_for(record, facility)``).
        instrument: the record's extracted instrument string (``extract_instrument()``).

    Returns:
        str: the observed-telescope label when available for 'observed'/'completed-no-block'
            (D-07), else the coarse aperture-class label
            (``calendar_utils.coarse_telescope_label()``).
    """
    if stage in _OBSERVED_TOKEN_STAGES:
        token = observed_token(record)
        if token is not None:
            return token
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
        ValueError: if ``record.observation_id`` is blank or whitespace-only (CR-01) --
            ``event_url()`` keys the event on ``facility.get_observation_url(observation_id)``,
            and every LCO/SOAR facility maps a blank id to the same bare listing URL, so an
            unusable id must be rejected before it can collide with another blank-id record's
            event. Also raised if ``parameters['start']``/``['end']`` cannot be parsed as
            datetimes, or (for a stage other than 'inconsistent') if the record's schedule
            fields are otherwise unusable per ``record_time_window()``'s own raising contract.
        InstrumentExtractionError: if ``extract_instrument()`` finds no usable config.
        KeyError: if the record has no usable request window (missing
            ``parameters['start']``/``['end']``).
    """
    if not (record.observation_id or '').strip():
        raise ValueError(f'record pk={record.pk} has no observation_id; cannot key an event')
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
        # WR-01: telescope/instrument/proposal are externally sourced (the last two come
        # straight from record.parameters) and write into CharField(max_length=200)
        # columns; title is already truncated above (title_for()'s own [:200]). SQLite
        # accepts an over-length value silently, but PostgreSQL (CLAUDE.md's documented
        # production target) raises DataError, which would otherwise escape as an
        # unhandled database error inside the caller's transaction (see WR-01's
        # transaction.atomic() fix at the post_save receiver).
        'telescope': token[:200],
        'instrument': instrument[:200],
        'proposal': proposal[:200],
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

    CR-02: every write this function makes -- resolving the facility, building the field
    dict, the create-or-update itself, and the companion-row write -- lives inside one
    ``try``, so a ``CalendarEvent.objects.get_or_create()`` collision
    (``MultipleObjectsReturned`` from a duplicate-url row, reachable through the
    unauthenticated event form) or an ``ImportError`` from ``get_service_class()`` is
    caught here rather than escaping to whichever caller happens to wrap this call --
    matching the "Never raises" contract this docstring already promised.

    Args:
        record: the ObservationRecord being projected.

    Returns:
        tuple[str, str]: (action, stage) where action is one of
            ``insert_or_create_calendar_event()``'s 'created'/'updated'/'unchanged', or
            'unprojectable' (stage then carries the caught exception's class name).
    """
    try:
        facility = facility_for(record)
        fields, stage = event_fields_for(record, facility)
        event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
        write_event_meta(event, record)
    except Exception as exc:  # noqa: BLE001 -- a projector must never break the triggering save
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, type(exc).__name__)
        return 'unprojectable', type(exc).__name__
    return action, stage


# TRIG-03/D-17: the six sweep counters project_queryset() accumulates per facility. Kept
# private to this module -- the command module (project_observation_calendar.py) keeps its
# own copy of the same six-tuple for seeding a facility that is in scope but contributed no
# records, since project_queryset() only ever returns keys for facilities it actually saw.
_SWEEP_COUNTER_KEYS = ('created', 'updated', 'unchanged', 'unprojectable', 'site_lookups', 'site_lookup_failed')


def _new_sweep_counters() -> dict[str, int]:
    """Return a fresh zeroed counter dict for one facility (a NEW dict every call)."""
    return dict.fromkeys(_SWEEP_COUNTER_KEYS, 0)


def project_queryset(
    records: Any,
    *,
    dry_run: bool = False,
    pre_fields_hook: Callable[[ObservationRecord, Any], dict[str, int] | None] | None = None,
) -> dict[str, Any]:
    """Sweep every record in ``records`` through the projector -- the TRIG-03 backstop for
    ``QuerySet.update()``/``bulk_create()`` paths the ``post_save`` receiver never sees.

    One counting rule for both run modes (real and ``dry_run``): per record, read the
    pre-sweep event snapshot (``before``) BEFORE anything in this iteration writes, build the
    intended field values, and take the counted action from
    ``calendar_utils.preview_calendar_event_action(before, fields)`` -- never from
    ``project_record()``'s own return value, which can report 'unchanged' even when this
    iteration's own write already changed the event (e.g. Task 3's observed-site save firing
    the ``post_save`` receiver mid-iteration). This is what keeps a dry-run count in agreement
    with what a real run would do for every field ``event_fields_for()`` derives from the
    record's own already-stored state: both modes read the same ``before`` and apply the same
    comparison helper.

    WR-02: this agreement has one documented exception. ``pre_fields_hook`` -- the one-time
    observed-site lookup -- is never called when ``dry_run`` is True, so the observed-telescope
    token (D-07) is never resolved in a dry run. A record whose only pending change is the
    coarse-to-observed token (e.g. ``'2m0'`` -> ``'FTN'``) is therefore reported ``unchanged``
    by ``--dry-run`` and ``updated`` by the real run that follows it, and ``site_lookups``
    itself is always 0 in a dry run. A dry-run count is otherwise structurally unable to
    disagree with what a real run would do -- this is the one field it cannot predict without
    making the network call it exists to avoid.

    ``pre_fields_hook``, when given, is called once per record with ``(record, facility)``
    AFTER ``before`` is captured and BEFORE ``fields``/``stage`` are built -- the extension
    point plan 34-02 Task 3 uses for the one-time observed-site lookup, whose stored token
    must already be on the record by the time ``event_fields_for()`` runs, while ``before``
    still holds the event as it stood before this sweep touched it. The hook may return a
    counter-increment dict (e.g. ``{'site_lookups': 1}``) to add to this record's facility
    counters, or None to add nothing. Never called when ``dry_run`` is True or the hook is
    None -- the caller decides whether a hook applies at all.

    This function itself never raises: each record's whole processing (the hook call,
    ``event_fields_for()``, and the real write) is wrapped so one bad row can never end the
    sweep, matching ``project_record()``'s own never-raise contract.

    Args:
        records: an ``ObservationRecord`` queryset to sweep (unfiltered ordering -- this
            function imposes its own ``order_by('pk')``).
        dry_run: if True, no ``CalendarEvent``/``CalendarEventMeta``/``ObservationRecord`` row
            is written and ``pre_fields_hook`` is never called, regardless of whether the
            caller passed one.
        pre_fields_hook: optional per-record hook called between capturing ``before`` and
            building ``fields`` (see above).

    Returns:
        dict[str, Any]: ``{'counters': {facility_name: {counter_key: int, ...}, ...},
            'rows': [{'observation_id': str, 'status': str, 'stage': str, 'action': str}, ...]}``.
            ``rows`` is in the same primary-key order the sweep iterated.
    """
    counters: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for record in records.select_related('target').order_by('pk'):
        facility_name = record.facility
        facility_counters = counters.setdefault(facility_name, _new_sweep_counters())
        try:
            facility = facility_for(record)
            url = event_url(record, facility)
            # Deliberately stale on purpose: this is the pre-sweep instance, held in memory
            # so a receiver's write later in this same iteration cannot silently refresh it
            # out from under the comparison -- see the docstring above.
            before = CalendarEvent.objects.filter(url=url).first()

            if not dry_run and pre_fields_hook is not None:
                increment = pre_fields_hook(record, facility)
                if increment:
                    for key, value in increment.items():
                        facility_counters[key] += value

            try:
                fields, stage = event_fields_for(record, facility)
            except Exception as exc:  # noqa: BLE001 -- a bad row must never end the sweep
                logger.warning('unprojectable observation_id=%r: %s', record.observation_id, type(exc).__name__)
                facility_counters['unprojectable'] += 1
                rows.append(
                    {
                        'observation_id': record.observation_id,
                        'status': record.status,
                        'stage': type(exc).__name__,
                        'action': 'unprojectable',
                    }
                )
                continue

            action = preview_calendar_event_action(before, fields)
            facility_counters[action] += 1
            if not dry_run:
                # A no-op write when pre_fields_hook's own save already triggered the
                # post_save receiver's own project_record() call for this record; the
                # guarantee that this call provides is for a record whose receiver path
                # was skipped (raw=True saves, receiver disconnected around a fixture, etc).
                project_record(record)
            rows.append(
                {'observation_id': record.observation_id, 'status': record.status, 'stage': stage, 'action': action}
            )
        except Exception as exc:  # noqa: BLE001 -- one bad row must never end the whole sweep
            logger.warning(
                'project_queryset failed for observation_id=%r: %s', record.observation_id, type(exc).__name__
            )
            facility_counters['unprojectable'] += 1
            rows.append(
                {
                    'observation_id': record.observation_id,
                    'status': record.status,
                    'stage': type(exc).__name__,
                    'action': 'unprojectable',
                }
            )
    return {'counters': counters, 'rows': rows}


def receiver_on_record_save(sender: Any, instance: ObservationRecord, created: bool, raw: bool, **kwargs: Any) -> None:
    """post_save receiver (TRIG-01): projects a record's event with no operator command.

    Returns immediately for a fixture load (``raw=True``) and for any facility other than
    LCO/SOAR (D-16, Gemini records stay with the submission-echo command). ``project_record()``
    runs inside its own ``transaction.atomic()`` savepoint (WR-01): TRIG-02 says this receiver
    runs inline in the caller's own transaction, and per Django's documented rule, catching a
    *database* error inside an ``atomic`` block without a savepoint leaves that whole
    transaction unusable for every later query. A savepoint lets a database error here roll
    back only the projector's own work, so the broad ``except`` below still protects the
    caller's save rather than converting a clear error into a confusing
    ``TransactionManagementError`` later. Logged at debug level, not warning, since this fires
    on every ObservationRecord save in production.

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
        with transaction.atomic():
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


# D-15/Pattern 3: captures a forward-direction ObservationGroup.observation_records.clear()'s
# former members on 'pre_clear' (before they're gone), keyed by (sender, group pk), so
# 'post_clear' -- which always arrives with pk_set=None -- can still re-project them.
_cleared_group_members: dict[tuple[Any, int], list[int]] = {}


def receiver_on_group_membership_changed(
    sender: Any, instance: Any, action: str, reverse: bool, pk_set: set[int] | None, **kwargs: Any
) -> None:
    """m2m_changed receiver (D-15): re-projects only the records named in the signal.

    Closes the gap a plain ``post_save`` receiver never sees: ``backfill_lco_observations``
    (and any other caller) adds group membership via ``group.observation_records.add(...)``
    *after* each record's own save, so a record's group link would otherwise never reach its
    event until the next sweep.

    Args:
        sender: the through model for ``ObservationGroup.observation_records``.
        instance: the ``ObservationGroup`` (forward direction, ``reverse=False``) or the
            ``ObservationRecord`` (reverse direction, ``reverse=True``) whose membership
            changed.
        action: one of Django's m2m_changed actions; only 'pre_clear'/'post_add'/
            'post_remove'/'post_clear' are handled, everything else returns immediately.
        reverse: True when the change was made from the ``ObservationRecord`` side (e.g.
            ``record.observationgroup_set.add(group)``).
        pk_set: the set of pks added/removed (forward direction), or the set of group pks
            (reverse direction); None for a 'pre_clear'/'post_clear' pair.
        **kwargs: the remaining signal kwargs (``using``, ``model``), unused.
    """
    if action not in ('pre_clear', 'post_add', 'post_remove', 'post_clear'):
        return
    if action == 'pre_clear':
        if not reverse:
            # Forward-direction .clear(): capture the about-to-be-cleared members now, while
            # they are still present, so post_clear (pk_set=None) can still re-project them.
            _cleared_group_members[(sender, instance.pk)] = list(
                instance.observation_records.values_list('pk', flat=True)
            )
        return
    if reverse:
        # instance is the ObservationRecord itself -- re-project it alone, regardless of
        # which action fired (post_add/post_remove/post_clear all mean "this record's own
        # group membership changed").
        record_pks: list[int] = [instance.pk]
    elif action == 'post_clear':
        record_pks = _cleared_group_members.pop((sender, instance.pk), [])
    else:
        record_pks = list(pk_set or [])

    for record in ObservationRecord.objects.filter(pk__in=record_pks):
        if record.facility not in PROJECTED_FACILITIES:
            continue
        try:
            project_record(record)
        except Exception as exc:  # noqa: BLE001 -- never break the caller's membership change
            logger.warning(
                'receiver_on_group_membership_changed failed for observation_id=%r: %s',
                record.observation_id,
                type(exc).__name__,
            )


def receiver_on_record_delete(sender: Any, instance: ObservationRecord, **kwargs: Any) -> None:
    """pre_delete receiver (D-14): deletes the record's own projector-owned event.

    ``pre_delete``, not ``post_delete``, because ``CalendarEventMeta.observation_record``'s
    ``on_delete=SET_NULL`` clears the reverse one-to-one link before ``post_delete`` fires --
    by then the companion row could no longer be found through the record at all. Deletes
    ``meta.event`` only when its ``url`` is this record's own facility observation URL, so an
    event a staff member has since re-attributed elsewhere (or any event outside this
    module's namespace) is never destroyed. Deleting the event cascades to its
    ``CalendarEventMeta`` companion row via that row's own ``on_delete=CASCADE``.

    The whole body is wrapped in one try/except so a projector problem here can never block
    an operator deleting an observation record (TRIG-02).

    Args:
        sender: the model class Django's signal framework passes (ObservationRecord).
        instance: the ObservationRecord about to be deleted.
        **kwargs: the remaining signal kwargs (``using``), unused.
    """
    if instance.facility not in PROJECTED_FACILITIES:
        return
    try:
        meta = instance.calendar_event_meta
        facility = facility_for(instance)
        if meta.event.url == event_url(instance, facility):
            meta.event.delete()
    except ObjectDoesNotExist:
        return
    except Exception as exc:  # noqa: BLE001 -- TRIG-02: never block an operator's delete
        logger.warning(
            'receiver_on_record_delete failed for observation_id=%r: %s', instance.observation_id, type(exc).__name__
        )
