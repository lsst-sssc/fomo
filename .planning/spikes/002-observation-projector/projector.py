"""Spike 002: a campaign-free projector from ObservationRecord to tom_calendar.CalendarEvent.

Throwaway-shaped: lives under .planning/spikes/, imports FOMO's existing helpers READ-ONLY
(calendar_utils, the LCO sync command's prefix vocabulary) and writes only CalendarEvent rows
keyed by the facility's own observation URL. It never touches the campaign reconciler's
``RUN:`` rows, never creates a CalendarEventMeta row (absent meta == "not owned by any run",
exactly the ownership rule decision D2 wants), and never calls the network.

One event per ObservationRecord (D1); the ObservationGroup contributes identity only (title
suffix + description line); the span is re-derived from the record's current state on every
call (D4): request window while queued, the placed block once scheduled, the observed block
once completed. Terminal-negative records keep their window and get a failure prefix — marked,
never dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any

from django.db.models.signals import post_save
from tom_observations.facility import get_service_class
from tom_observations.models import ObservationGroup, ObservationRecord

from solsys_code.calendar_utils import (
    coarse_telescope_label,
    extract_instrument,
    insert_or_create_calendar_event,
    record_time_window,
)
from solsys_code.management.commands.sync_lco_observation_calendar import _failure_prefix

logger = logging.getLogger('spike002')
DISPATCH_UID = 'spike-002-observation-projector'
SPIKE_MARK = 'Spike: 002-observation-projector (spike-owned base-layer event; not a campaign run)'

_facilities: dict[str, Any] = {}
_series_cache: dict[int, dict[int, 'Series']] = {}


@dataclass(frozen=True)
class Series:
    group_name: str
    group_pk: int
    index: int
    size: int


def facility_for(record: ObservationRecord):
    """One facility instance per facility name (LCO -> LCOFacility, SOAR -> SOARFacility, ...)."""
    name = record.facility
    if name not in _facilities:
        _facilities[name] = get_service_class(name)()
    return _facilities[name]


def _window_start_or_max(record: ObservationRecord) -> datetime:
    try:
        return record_time_window(record)[0]
    except (KeyError, ValueError, TypeError):
        return datetime.max.replace(tzinfo=dt_timezone.utc)


def series_for(record: ObservationRecord) -> Series | None:
    """Series identity for a record that belongs to a multi-member ObservationGroup, else None.

    Members are numbered 1..n in order of their window start, so a nightly cadence reads
    "night 3 of 14" no matter which member is projected first. Cached per group for the sweep.
    """
    group = ObservationGroup.objects.filter(observation_records=record).order_by('pk').first()
    if group is None:
        return None
    if group.pk not in _series_cache:
        members = list(group.observation_records.all())
        if len(members) < 2:
            _series_cache[group.pk] = {}
        else:
            members.sort(key=lambda r: (_window_start_or_max(r), r.pk))
            _series_cache[group.pk] = {
                r.pk: Series(group.name, group.pk, i, len(members)) for i, r in enumerate(members, 1)
            }
    return _series_cache[group.pk].get(record.pk)


def reset_caches() -> None:
    _series_cache.clear()


def stage_for(record: ObservationRecord, facility) -> str:
    """Classify the record's lifecycle stage from its current fields alone (no network)."""
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


_STAGE_PREFIX = {'queued': '[QUEUED]', 'placed': '[SCHEDULED]'}


def title_for(record: ObservationRecord, stage: str, telescope: str, instrument: str, facility, series) -> str:
    prefix = _failure_prefix(record.status, facility)
    if prefix is None:
        prefix = _STAGE_PREFIX.get(stage)
    stem = f'{record.target.name} {telescope} {instrument}'
    if series is not None:
        stem = f'{stem} · {series.group_name} {series.index}/{series.size}'
    title = f'{prefix} {stem}' if prefix else stem
    return title[:200]


def event_url(record: ObservationRecord, facility) -> str:
    return facility.get_observation_url(record.observation_id)


def event_fields_for(record: ObservationRecord, facility) -> tuple[dict[str, Any], str]:
    """Build the CalendarEvent field values for a record; raises if the record cannot be projected."""
    instrument = extract_instrument(record.parameters) or 'unknown-instrument'
    telescope = coarse_telescope_label(instrument, record.facility)
    stage = stage_for(record, facility)
    if stage == 'inconsistent':
        raise ValueError(f'inconsistent scheduled_start/scheduled_end on observation_id={record.observation_id!r}')
    start_time, end_time = record_time_window(record)
    series = series_for(record)
    proposal = record.parameters.get('proposal', '')
    target_list = record.target.targetlist_set.order_by('name').first()
    description = (
        f'Proposal: {proposal}\n'
        f'Status: {record.status}\n'
        f'Stage: {stage}\n'
        f'Span (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime("%Y-%m-%dT%H:%M:%S")}\n'
        f'{SPIKE_MARK}'
    )
    if series is not None:
        description += f'\nSeries: {series.group_name} ({series.index}/{series.size}, ObservationGroup #{series.group_pk})'
    fields = {
        'title': title_for(record, stage, telescope, instrument, facility, series),
        'description': description,
        'start_time': start_time,
        'end_time': end_time,
        'telescope': telescope,
        'instrument': instrument,
        'proposal': proposal,
        'target_list': target_list,
    }
    return fields, stage


def project_record(record: ObservationRecord) -> tuple[str, str]:
    """Create/update/leave-unchanged the record's event. Returns (action, stage).

    action is one of insert_or_create_calendar_event's 'created' | 'updated' | 'unchanged', or
    'unprojectable' (stage then carries the reason) — never raises, so it is safe inside a
    post_save receiver.
    """
    facility = facility_for(record)
    try:
        fields, stage = event_fields_for(record, facility)
    except Exception as exc:  # noqa: BLE001 — a projector must never break the save that triggered it
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, exc)
        return 'unprojectable', f'{type(exc).__name__}: {exc}'
    _event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
    return action, stage


def project_queryset(records) -> dict[str, Any]:
    """Sweep: project every record in the queryset; return counters and a per-record log."""
    reset_caches()
    counters: dict[str, int] = {}
    per_stage: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for record in records.select_related('target').order_by('pk'):
        action, stage = project_record(record)
        counters[action] = counters.get(action, 0) + 1
        per_stage.setdefault(stage, {})
        per_stage[stage][action] = per_stage[stage].get(action, 0) + 1
        rows.append({'observation_id': record.observation_id, 'status': record.status, 'stage': stage, 'action': action})
    return {'counters': counters, 'per_stage': per_stage, 'rows': rows}


def _receiver(sender, instance: ObservationRecord, created: bool, raw: bool, **kwargs) -> None:
    if raw:
        return
    reset_caches()
    action, stage = project_record(instance)
    logger.info('post_save projected observation_id=%r created=%s -> %s (%s)', instance.observation_id, created, action, stage)


def connect() -> None:
    """Connect the per-save trigger (decision D5, refined by spike 001b)."""
    post_save.connect(_receiver, sender=ObservationRecord, weak=False, dispatch_uid=DISPATCH_UID)


def disconnect() -> None:
    post_save.disconnect(_receiver, sender=ObservationRecord, dispatch_uid=DISPATCH_UID)
