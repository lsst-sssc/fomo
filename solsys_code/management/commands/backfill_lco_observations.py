"""Backfill ObservationRecords, Targets, and ObservationGroups from LCO RequestGroups, and
collect every touched Target into a TargetList.

Campaign-agnostic sibling of ``backfill_lco_observation_records`` (see that module's
docstring): this command needs no campaign, updates existing records in place instead of
skipping them, links multi-request RequestGroups into ``ObservationGroup``s, and only ever
creates non-sidereal Targets (never a sidereal field Target). It is deliberately
self-contained -- it does not import the sibling's helpers -- so it stays a plausible
standalone contribution back to ``tom_toolkit``.
"""

import io
import logging
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timezone as dt_timezone
from typing import Any, TextIO
from urllib.parse import urlencode, urljoin

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.base_models import REQUIRED_NON_SIDEREAL_FIELDS, REQUIRED_NON_SIDEREAL_FIELDS_PER_SCHEME
from tom_targets.models import Target, TargetList

# The parameters-key reservation shared with the projector sweep's one-time observed-site
# lookup (calendar_utils D-09). Discovery must carry these keys forward and never erase them
# (F1, v2.4-INTENT-REVIEW.md).
from solsys_code.calendar_utils import OBSERVED_SITE_PARAMETER_KEYS
from solsys_code.campaign_system_links import (
    OUTCOME_LINKED,
    OUTCOME_SKIPPED,
    OUTCOME_WOULD_LINK,
    attempt_system_link,
)
from solsys_code.models import WatchedProposal
from solsys_code.observation_blocks import FomoLCOFacility, select_schedule_block

logger = logging.getLogger(__name__)

# The ObservationRecord.parameters key this sweep writes on a record it created while the live block
# lookup failed, and on an existing record whose --recheck-unscheduled lookup failed while the ordinary
# tick would otherwise skip it. _schedule_lookup_is_needed() treats a marked record as needing a lookup
# until one succeeds, and the successful write removes the key (the freshly built parameters never
# contain it). Only the sweep writes it: _build_parameters() copies nothing from the portal payload
# into this key.
SCHEDULE_LOOKUP_FAILED_KEY = 'schedule_lookup_failed'

# Portal wire key -> TOM Target field name, the inverse of OCSFacility._build_target_fields'
# own field_mapping (tom_observations/facilities/ocs.py:843-850), verified at plan time (D-E).
_ELEMENT_WIRE_TO_TOM_FIELD = {
    'orbinc': 'inclination',
    'longascnode': 'lng_asc_node',
    'argofperih': 'arg_of_perihelion',
    'meandist': 'semimajor_axis',
    'meananom': 'mean_anomaly',
    'dailymot': 'mean_daily_motion',
    'epochofel': 'epoch_of_elements',
    'epochofperih': 'epoch_of_perihelion',
}

# Fields that carry across the portal payload unchanged (D-E).
_ELEMENT_PASSTHROUGH_FIELDS = ('perihdist', 'eccentricity')


def _iter_request_groups(facility: LCOFacility, proposal: str, created_after: str | None, created_before: str | None):
    """Page through GET /api/requestgroups/ for a proposal, yielding every RequestGroup.

    'created_after'/'created_before' are sent as server-side query parameters (D-C) purely
    as a pre-filter to cut payload size -- the caller must still re-check
    request_group['created'] client-side with ``_within_created_window()``, since a portal
    that ignores an unrecognised query parameter would otherwise silently backfill the
    whole proposal.

    Args:
        facility: an LCOFacility instance (for portal_url/api_key settings and headers).
        proposal: LCO proposal code, exact match.
        created_after: raw --created-after CLI value (ISO-8601), or None.
        created_before: raw --created-before CLI value (ISO-8601), or None.

    Yields:
        dict: each RequestGroup object (with its nested 'requests' list) returned by the
            portal, regardless of whether it passes the client-side created-window check.
    """
    params: dict[str, Any] = {'proposal': proposal, 'limit': 100}
    if created_after:
        params['created_after'] = created_after
    if created_before:
        params['created_before'] = created_before
    query = urlencode(params)
    url = urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/requestgroups/?{query}')
    while url:
        response = make_request('GET', url, headers=facility._portal_headers())
        payload = response.json()
        yield from payload.get('results', [])
        url = payload.get('next')


def _parse_datetime_value(value: Any) -> datetime | None:
    """Parse a portal or CLI ISO-8601 value into an aware UTC datetime.

    Accepts a full ISO-8601 datetime string, a bare ISO-8601 date string (assumed
    midnight), or an already-a-datetime value; a naive datetime is assumed UTC. Never
    raises -- a value that cannot be parsed at all returns None, so a single malformed
    portal field degrades a comparison rather than aborting the run.

    Args:
        value: a str, datetime, or falsy value read from a portal payload or CLI arg.

    Returns:
        datetime | None: an aware UTC datetime, or None if 'value' is falsy or unparseable.
    """
    if not value:
        return None
    if isinstance(value, str):
        parsed = parse_datetime(value)
        if parsed is None:
            try:
                parsed = datetime.combine(date.fromisoformat(value), dt_time.min)
            except ValueError:
                return None
        value = parsed
    if timezone.is_naive(value):
        value = timezone.make_aware(value, dt_timezone.utc)
    return value


def _parse_created_bound(raw_value: str | None) -> datetime | None:
    """Parse a --created-after/--created-before CLI value into an aware datetime.

    Args:
        raw_value: the raw --created-after/--created-before CLI string, or None.

    Returns:
        datetime | None: the parsed, timezone-aware bound, or None if 'raw_value' is None.

    Raises:
        CommandError: 'raw_value' is set but not a valid ISO-8601 timestamp/date.
    """
    if not raw_value:
        return None
    parsed = _parse_datetime_value(raw_value)
    if parsed is None:
        raise CommandError(f'Invalid ISO-8601 timestamp: {raw_value!r}')
    return parsed


def _within_created_window(
    request_group: dict[str, Any], created_after: datetime | None, created_before: datetime | None
) -> bool:
    """Client-side re-check of a RequestGroup's 'created' timestamp against the window (D-C).

    Args:
        request_group: the RequestGroup object, for its 'created' field.
        created_after: the parsed --created-after bound, or None.
        created_before: the parsed --created-before bound, or None.

    Returns:
        bool: True if no window is set, or the RequestGroup's 'created' timestamp falls
            inside it. False if a window is set and 'created' is missing/unparseable
            (fail closed, matching D-C's "never silently backfill the whole proposal")
            or falls outside the window.
    """
    if created_after is None and created_before is None:
        return True
    created = _parse_datetime_value(request_group.get('created'))
    if created is None:
        return False
    if created_after is not None and created < created_after:
        return False
    if created_before is not None and created > created_before:
        return False
    return True


def _first_named_target(request: dict[str, Any]) -> dict[str, Any] | None:
    """Return the target dict of a request's first configuration that has a name.

    Args:
        request: a single request from request_group['requests'].

    Returns:
        dict[str, Any] | None: the first configuration's 'target' dict with a 'name', or
            None if no configuration has one.
    """
    for configuration in request.get('configurations', []):
        target = configuration.get('target') or {}
        if target.get('name'):
            return target
    return None


def _extract_orbital_elements(target_dict: dict[str, Any]) -> dict[str, Any]:
    """Map a portal ORBITAL_ELEMENTS target dict onto TOM Target field names (D-E).

    Reads each field by its LCO/OCS wire key first (e.g. 'orbinc'), falling back to the
    TOM field name itself (e.g. 'inclination') when the payload already uses that
    spelling -- so both 'epochofel' and 'epoch_of_elements' (etc.) are accepted.

    Args:
        target_dict: a request configuration's 'target' dict.

    Returns:
        dict[str, Any]: TOM Target field names mapped to their portal values, for every
            field the payload actually carries a value for.
    """
    elements: dict[str, Any] = {}
    for wire_key, tom_field in _ELEMENT_WIRE_TO_TOM_FIELD.items():
        value = target_dict.get(wire_key, target_dict.get(tom_field))
        if value is not None:
            elements[tom_field] = value
    for tom_field in _ELEMENT_PASSTHROUGH_FIELDS:
        value = target_dict.get(tom_field)
        if value is not None:
            elements[tom_field] = value
    # 'scheme' is read last so an alternate 'orbital_elements' spelling (occasionally seen
    # on hand-built portal fixtures) is accepted as a fallback for the canonical key.
    scheme = target_dict.get('scheme', target_dict.get('orbital_elements'))
    if scheme is not None:
        elements['scheme'] = scheme
    return elements


def _validate_orbital_elements(elements: dict[str, Any]) -> str | None:
    """Check that 'elements' carries every field REQUIRED_NON_SIDEREAL_FIELDS(_PER_SCHEME)
    demands for its declared scheme.

    Args:
        elements: the dict returned by _extract_orbital_elements().

    Returns:
        str | None: a short reason the elements are incomplete/unusable, or None if they
            are sufficient to build a non-sidereal Target.
    """
    scheme = elements.get('scheme')
    if scheme not in REQUIRED_NON_SIDEREAL_FIELDS_PER_SCHEME:
        return f'unrecognised or missing orbital-element scheme {scheme!r}'
    required = REQUIRED_NON_SIDEREAL_FIELDS + REQUIRED_NON_SIDEREAL_FIELDS_PER_SCHEME[scheme]
    missing = [field for field in required if elements.get(field) is None]
    if missing:
        return f'missing required orbital elements for scheme {scheme!r}: {", ".join(missing)}'
    return None


def _build_non_sidereal_target(target_dict: dict[str, Any]) -> tuple[Target | None, str | None]:
    """Build an unsaved non-sidereal Target from a request's orbital-element target dict.

    Never assigns the sidereal Target type anywhere (D-G): a target dict this function
    can't map to a complete non-sidereal Target simply isn't built at all.

    Args:
        target_dict: a request configuration's 'target' dict, per _first_named_target().

    Returns:
        tuple[Target | None, str | None]: (unsaved non-sidereal Target, None) on success,
            or (None, reason) if the payload doesn't carry enough to build one.
    """
    name = target_dict.get('name')
    if not name:
        return None, 'target has no name'
    if target_dict.get('type') != 'ORBITAL_ELEMENTS':
        return None, f'target type {target_dict.get("type")!r} is not ORBITAL_ELEMENTS'
    elements = _extract_orbital_elements(target_dict)
    reason = _validate_orbital_elements(elements)
    if reason:
        return None, reason
    return Target(name=name, type=Target.NON_SIDEREAL, **elements), None


def _build_parameters(request_group: dict[str, Any], request: dict[str, Any]) -> dict[str, Any] | None:
    """Build a minimal flat ObservationRecord.parameters dict for a backfilled request.

    Matches the legacy single-config flat shape ('proposal', 'instrument_type', 'start',
    'end') that solsys_code.calendar_utils.extract_instrument() already falls back to when
    no c_N_*-prefixed multi-configuration keys are present, so backfilled records stay
    readable by the existing sync command.

    Args:
        request_group: the parent RequestGroup object (for 'proposal').
        request: a single request from request_group['requests'] (for 'windows' and the
            first configuration with an 'instrument_type').

    Returns:
        dict[str, Any] | None: the parameters dict, or None if the request has no
            configuration with a usable instrument_type.
    """
    for configuration in request.get('configurations', []):
        instrument_type = configuration.get('instrument_type')
        if not instrument_type:
            continue
        parameters = {
            'proposal': request_group.get('proposal'),
            'instrument_type': instrument_type,
        }
        windows = request.get('windows') or []
        if windows:
            if windows[0].get('start'):
                parameters['start'] = windows[0]['start']
            if windows[0].get('end'):
                parameters['end'] = windows[0]['end']
        return parameters
    return None


def _resolve_schedule(
    facility: LCOFacility, request: dict[str, Any], dry_run: bool, skip_live_lookup: bool = False
) -> tuple[Any, Any, bool, bool]:
    """Resolve a request's scheduled_start/scheduled_end (D-B).

    Reads an embedded 'observations' block list from the request payload when present
    (no extra HTTP call either way, so this is not skipped under --dry-run) and chooses
    the block with FOMO's rule, ``select_schedule_block()``; otherwise falls back to a live
    facility.get_observation_status() call -- on ``FomoLCOFacility``, so the live path
    chooses the block with the same rule -- which *is* skipped entirely under --dry-run.
    A failed fallback call is caught and reported via the
    returned 'lookup_failed' flag -- never fatal, never counted as a skipped request. That includes a
    reply whose block list is not a list: ``FomoLCOFacility`` raises ``UnexpectedBlockPayloadError``
    for it, which is caught here like any other failure, so it is a failed lookup, never "no block".
    A failed lookup resolved nothing: the caller then writes neither schedule field nor a status change
    for an existing record (see ``_changed_record_fields``), so a stored time is never erased.

    Args:
        facility: an LCOFacility instance (``sweep_proposal()`` passes a ``FomoLCOFacility``).
        request: a single request from request_group['requests'].
        dry_run: whether the command is running with --dry-run.
        skip_live_lookup: when True, a request without an embedded block returns
            (None, None, False, embedded) without any live call, exactly like a dry run.
            The caller sets this for a record already finished at the same portal state
            (F2). An embedded block is still read either way, because that costs nothing.

    Returns:
        tuple[Any, Any, bool, bool]: (scheduled_start, scheduled_end, lookup_failed,
            embedded), where the first two are raw portal values (str or None), the third
            is True only when the live fallback call was attempted and failed, and the
            fourth is True when the request payload carried an embedded 'observations'
            list at all (regardless of whether a usable block could be selected from it)
            -- so 'handle' can count which schedule path the portal actually exercised
            without re-deriving that from the two schedule values.
    """
    blocks = request.get('observations')
    embedded = blocks is not None
    if embedded:
        current_block = select_schedule_block(blocks)
        if current_block:
            return current_block.get('start'), current_block.get('end'), False, embedded
        return None, None, False, embedded

    if dry_run or skip_live_lookup:
        return None, None, False, embedded

    observation_id = str(request.get('id'))
    try:
        result = facility.get_observation_status(observation_id)
    except Exception as exc:
        logger.debug(f'Observed-block lookup failed for observation_id={observation_id!r}: {exc}')
        return None, None, True, embedded
    return result.get('scheduled_start'), result.get('scheduled_end'), False, embedded


def _has_failed_lookup_marker(parameters: Any) -> bool:
    """Return True when 'parameters' carries a truthy SCHEDULE_LOOKUP_FAILED_KEY.

    Args:
        parameters: an ObservationRecord's parameters value. Anything that is not a dict carries
            no marker.

    Returns:
        bool: True only for a dict whose value under SCHEDULE_LOOKUP_FAILED_KEY is truthy.
    """
    return isinstance(parameters, dict) and bool(parameters.get(SCHEDULE_LOOKUP_FAILED_KEY))


def _schedule_lookup_is_needed(
    existing_record: ObservationRecord | None,
    portal_state: str,
    terminal_states: frozenset[str],
    failed_states: frozenset[str],
    *,
    recheck_unscheduled: bool = False,
) -> bool:
    """Decide whether a request needs a live get_observation_status() lookup (F2, option A).

    The /api/requestgroups/ list payload carries no 'observations' blocks, so without this
    gate every request costs a live get_observation_status() call (two portal GETs) on every
    tick, and a tick lengthens as a proposal ages. A record that is already finished and is
    still reported in that same state has an observed block that can no longer change; this
    is the same premise update_all_observation_statuses() uses when it leaves terminal
    records out of the status refresh. Such a record needs no lookup.

    A completed record (terminal, but not a failed state) that is still missing its
    'scheduled_start' or 'scheduled_end' keeps being looked up until the portal supplies
    them, so a lookup that failed when the record was created or completed is retried on
    later ticks rather than frozen at no schedule. A record in a failed state (window
    expired, cancelled and the like) with no times is skipped too, because no FOMO path that updates an
    existing record stores a state change without its block: this sweep writes a state
    change only together with a lookup that succeeded (a failed lookup holds the change back and
    writes no schedule field, so the state-change branch retries it next tick); a record this sweep
    creates while its lookup fails, or whose ``recheck_unscheduled`` lookup fails while this skip
    would otherwise apply, carries ``SCHEDULE_LOOKUP_FAILED_KEY`` and is looked up on every run until
    a lookup succeeds; and the unattended status refresh writes nothing for a request whose lookup
    raises, including a block-list reply that is not a list (``UnexpectedBlockPayloadError``).

    A failed request can carry a block that started and was aborted after taking data, and that
    block counts (Phase 35 D-05/D-06). Since 37.1-07 every FOMO-owned path that writes a state
    change -- the unattended status refresh, this sweep and the Didymos backfill command --
    resolves the block with FOMO's rule (``select_schedule_block()``), so a record that reaches
    a failed state through one of them already holds its block and this skip stays correct for
    it. TOM Toolkit's own status routes run on TOM's registered ``LCOFacility``, which ignores an
    aborted block: its stock ``updatestatus`` command and the observation list's "Update status"
    button that runs it, the Cancel button on its observation page, and its REST cancel route
    ``PATCH /api/observations/<pk>/cancel/``. A record finished through one of them is stored
    without its aborted block's times, and this skip then leaves it alone exactly like a record
    stored before 37.1-07 (T-37.1-42, accepted). The Didymos backfill command creates its record with
    the listing's state before its own lookup and reports a failure under ``status sync failed``, so
    such a record is recovered with ``recheck_unscheduled=True`` like a legacy one. Both kinds are
    brought up to date by an operator run with ``recheck_unscheduled=True``, which also looks up a
    failed-state record that is missing either time, once per run. The unattended runner never passes
    it, so F2's bounded per-tick lookups stay as they are. If that recheck lookup fails, the record is
    marked as above, so the unattended runner, which never passes the flag, retries it.

    This is F2 option A from .planning/v2.4-INTENT-REVIEW.md. The function makes no query, no
    network call and no mutation, and trusts the two state lists it is given.

    Args:
        existing_record: the stored ObservationRecord for this request, or None if there is none.
        portal_state: the request's current state as reported by the portal.
        terminal_states: the facility's terminal observing states.
        failed_states: the facility's failed observing states (a subset of the terminal ones).
        recheck_unscheduled: when True, a record missing either scheduled time is looked up even
            in a failed state. Keyword-only; the unattended runner never sets it.

    Returns:
        bool: True when a live lookup is needed; False when it can be skipped.
    """
    if existing_record is None:
        return True
    if _has_failed_lookup_marker(existing_record.parameters):
        return True
    if existing_record.status != portal_state:
        return True
    if existing_record.status not in terminal_states:
        return True
    if existing_record.scheduled_start is None or existing_record.scheduled_end is None:
        return recheck_unscheduled or existing_record.status not in failed_states
    return False


def _failed_lookup_needs_marker(
    record: ObservationRecord,
    portal_state: str,
    terminal_states: frozenset[str],
    failed_states: frozenset[str],
    *,
    lookup_failed: bool,
) -> bool:
    """Decide whether a failed live lookup on an existing record must mark it for a retry.

    A failed lookup resolves nothing and writes nothing, so a record the next ordinary tick would
    skip stays frozen with no times -- in practice an unmarked failed-state record with no times at
    the same portal state, which only ``--recheck-unscheduled`` looks up. Such a record is marked
    with ``SCHEDULE_LOOKUP_FAILED_KEY`` so the unattended runner, which never passes the flag, looks
    it up again until a lookup succeeds. Any record the ordinary gate already looks up (a state
    change, a non-terminal record, a completed record missing a time, an already-marked record)
    needs no new mark.

    Args:
        record: the stored ObservationRecord the failed lookup was for.
        portal_state: the request's current state as reported by the portal.
        terminal_states: the facility's terminal observing states.
        failed_states: the facility's failed observing states (a subset of the terminal ones).
        lookup_failed: True when this run's live lookup was attempted and raised.

    Returns:
        bool: True only when the lookup failed and the ordinary per-tick gate would skip the record.
    """
    return lookup_failed and not _schedule_lookup_is_needed(
        record, portal_state, terminal_states, failed_states, recheck_unscheduled=False
    )


def _preserve_observed_site_keys(existing: Any, rebuilt: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of 'rebuilt' that also carries the sweep's observed-site keys from 'existing'.

    The projector sweep stores the OBSERVED_SITE_PARAMETER_KEYS in ObservationRecord.parameters
    once per record, ever, and treats their presence as "already looked up".
    _build_parameters() never produces them, so a whole-dict comparison would see them as
    portal drift on every tick: discovery would erase them, the post_save trigger would
    re-draw the event with the coarse telescope token, and the next sweep would repeat a live
    portal lookup for every tagged record (F1).

    Presence is tested by membership, never truthiness, so a key the sweep stored as None
    (for example 'observed_enclosure' when the portal block has no enclosure) survives too.

    Args:
        existing: the record's current parameters. Anything that is not a dict carries nothing.
        rebuilt: the freshly built parameters dict. Its portal-owned keys always win.

    Returns:
        dict[str, Any]: a new dict; neither argument is mutated.
    """
    merged = dict(rebuilt)
    if isinstance(existing, dict):
        for key in OBSERVED_SITE_PARAMETER_KEYS:
            if key in existing:
                merged[key] = existing[key]
    return merged


def _changed_record_fields(
    record: ObservationRecord,
    status: str,
    scheduled_start: datetime | None,
    scheduled_end: datetime | None,
    parameters: dict[str, Any],
    compare_schedule: bool = True,
    *,
    lookup_failed: bool = False,
    mark_lookup_failed: bool = False,
) -> dict[str, Any]:
    """Return the ObservationRecord fields whose desired value differs from the record's.

    The single comparison the write branch and the dry-run branch both call, so an
    updated-vs-unchanged decision can never drift between the two modes (T-ik7-02).

    Args:
        record: the existing ObservationRecord being compared against.
        status: the request's current portal state.
        scheduled_start: the resolved scheduled start, or None.
        scheduled_end: the resolved scheduled end, or None.
        parameters: the freshly built ObservationRecord.parameters dict. Any of the
            OBSERVED_SITE_PARAMETER_KEYS already on 'record.parameters' are carried into it
            before comparing (F1), so the sweep's one-time site lookup is never erased or
            counted as portal drift. The SCHEDULE_LOOKUP_FAILED_KEY marker is carried the same
            way when this run resolved no schedule (``compare_schedule`` False, which includes a
            failed lookup), so a dry run, a skipped lookup or another failed lookup never reports
            it as drift; when a schedule was resolved the freshly built dict, which never has the
            key, wins, so the marker is removed in the same save that writes the resolved times.
            When 'parameters' appears in the returned dict, it is this merged dict.
        compare_schedule: whether 'scheduled_start'/'scheduled_end' are compared at all.
            Under --dry-run, a request with no embedded 'observations' block has no
            resolved schedule -- the live fallback lookup that would otherwise produce one
            is deliberately skipped (D-B) -- so comparing the two schedule fields against
            None here would report a spurious change on every record that already has real
            times. The same holds for a real-run request whose lookup was skipped because the
            record is already finished at the same portal state (F2). Callers pass True only
            when this run resolved a schedule (an embedded block, or a real-run live lookup)
            and False for a dry-run fallback or a skipped lookup, so such a request is
            compared on status/parameters only, matching what the run can actually know
            without making the network call it exists to avoid.
        lookup_failed: True when this run's live block lookup was attempted and raised. Keyword-only.
            A failed lookup resolved nothing, so neither schedule field is compared (it behaves as
            ``compare_schedule=False``, so a stored time is never overwritten with None) and the
            ``status`` change is held back too: the record keeps its stored state, so the next
            sweep still sees a state change, runs the lookup again, and stores the new status and
            the block's times together in one save. Writing the status without the block would
            disarm that retry and leave a finished record looking resolved (37.1-REVIEW CR-01).
            ``parameters`` are still compared.
        mark_lookup_failed: True to write ``SCHEDULE_LOOKUP_FAILED_KEY`` into the compared parameters,
            so the mark lands in the same save and the record counts as updated. Keyword-only; pass
            True only together with ``lookup_failed`` (see ``_failed_lookup_needs_marker``).

    Returns:
        dict[str, Any]: field name -> new value for each of the (up to four) fields whose
            desired value differs from what 'record' currently holds. An empty dict means
            nothing would change.
    """
    parameters = _preserve_observed_site_keys(record.parameters, parameters)
    changes: dict[str, Any] = {}
    if lookup_failed:
        compare_schedule = False
    elif record.status != status:
        changes['status'] = status
    if not compare_schedule and _has_failed_lookup_marker(record.parameters):
        parameters[SCHEDULE_LOOKUP_FAILED_KEY] = record.parameters[SCHEDULE_LOOKUP_FAILED_KEY]
    if mark_lookup_failed:
        parameters[SCHEDULE_LOOKUP_FAILED_KEY] = True
    if compare_schedule:
        if record.scheduled_start != scheduled_start:
            changes['scheduled_start'] = scheduled_start
        if record.scheduled_end != scheduled_end:
            changes['scheduled_end'] = scheduled_end
    if record.parameters != parameters:
        changes['parameters'] = parameters
    return changes


def _group_name(request_group: dict[str, Any]) -> str:
    """Build a deterministic ObservationGroup name that always fits max_length=50 (D-D).

    Args:
        request_group: the parent RequestGroup object (for 'name' and 'id').

    Returns:
        str: the RequestGroup name, truncated so a ' (<requestgroup id>)' suffix always
            fits inside 50 characters.
    """
    suffix = f' ({request_group.get("id")})'
    max_name_length = 50 - len(suffix)
    name = (request_group.get('name') or '')[:max_name_length]
    return f'{name}{suffix}'


def sweep_proposal(
    proposal: str,
    *,
    target_list_name: str | None = None,
    user: Any = None,
    created_after: str | None = None,
    created_before: str | None = None,
    dry_run: bool = False,
    recheck_unscheduled: bool = False,
    stdout: Any = None,
    stderr: Any = None,
) -> str:
    """Backfill ObservationRecords, non-sidereal Targets and an ObservationGroup for every
    request/RequestGroup returned for one LCO proposal, and collect every touched Target
    into a TargetList.

    Extracted from ``Command.handle()`` (36-CONTEXT.md D-07/36-RESEARCH.md Open Question 2)
    so the sweep for a single proposal is callable directly -- by the bare-invocation
    watched-list loop (Task 3) and by the unattended runner (36-01/Plan 03) -- without going
    through ``call_command()``. Constructs its own ``FomoLCOFacility`` (TOM's LCO facility with
    FOMO's block rule, so an aborted block's times are stored) and calls
    ``facility.set_user(user)`` here so each call gets a fresh instance (Phase 34 D-10: a
    facility instance is never shared across calls).

    Args:
        proposal: LCO proposal code to filter RequestGroups by (exact match).
        target_list_name: override for the derived ``'<proposal>_targets'`` TargetList
            name; None (the default) keeps the derived name.
        user: a resolved ``User`` instance to attribute created/updated ObservationRecords
            to, or None to leave them unattributed. Username-to-User resolution (including
            the CommandError-on-unknown-username check) is CLI argument validation and
            stays in ``Command.handle()``.
        created_after: raw ISO-8601 CLI value; only RequestGroups created on/after this
            timestamp are backfilled, or None for no lower bound.
        created_before: raw ISO-8601 CLI value; only RequestGroups created on/before this
            timestamp are backfilled, or None for no upper bound.
        dry_run: whether to report what would be created/updated without writing anything.
        recheck_unscheduled: when True, every record still missing a scheduled time is looked up
            once, including a failed-state record the finished-record skip (F2) otherwise leaves
            alone, so records stored under TOM's old block rule pick up an aborted block's times.
            One portal lookup per such record. A lookup that fails on such a record marks it with
            ``SCHEDULE_LOOKUP_FAILED_KEY``, so the ordinary sweep the unattended runner makes retries it
            until a lookup succeeds. Default False; the unattended runner never passes it.
        stdout: a file-like sink for progress/summary lines (defaults to a fresh
            ``io.StringIO()`` so this function is callable with no sink at all).
        stderr: a file-like sink for skip/failure lines (defaults to a fresh
            ``io.StringIO()``).

    System links (ALLOC-06): every record the sweep touches -- created, updated or unchanged
    alike, whatever its own status -- is offered to
    ``campaign_system_links.attempt_system_link()``, because runs are often created after their
    records (D-01). A record that exactly matches one approved run by proposal code, target (or
    campaign membership when the proposal is unique in the campaign) and window is linked to it
    and its allocation night retires on this sweep; one stdout line per link names the
    ``observation_id``, the ``CampaignRun`` pk and the match basis. The summary ends with
    ``system links: N, links skipped: M`` (``would link: N, links skipped: M`` under
    ``dry_run``); ``links skipped`` counts a match check or write that raised, which never fails
    the sweep. A target this sweep adds to its own ``<proposal>_targets`` list counts as a
    campaign member from the NEXT tick, because that list is written after the loop. Under
    ``dry_run`` an existing record is judged on its stored window and a not-yet-created record on
    its request window; a would-be-new target is never asked.

    Returns:
        str: a one-line summary of the counts described in the ``Command`` class docstring.

    Raises:
        CommandError: 'created_after'/'created_before' is set but not a valid ISO-8601
            timestamp/date.
    """
    if stdout is None:
        stdout = io.StringIO()
    if stderr is None:
        stderr = io.StringIO()

    parsed_created_after = _parse_created_bound(created_after)
    parsed_created_before = _parse_created_bound(created_before)

    facility = FomoLCOFacility()
    facility.set_user(user)
    # Read once per sweep from the facility's own lists, never hard-coded, so this gate cannot
    # diverge from TOM's notion of which states are finished (F2).
    terminal_states = frozenset(facility.get_terminal_observing_states())
    failed_states = frozenset(facility.get_failed_observing_states())

    requestgroups_seen = 0
    created = 0
    updated = 0
    unchanged = 0
    skipped = 0
    targets_created = 0
    groups_created = 0
    groups_reused = 0
    block_lookups_failed = 0
    embedded_blocks = 0
    fallback_lookups_needed = 0
    fallback_lookups_skipped = 0
    # ALLOC-06: exact-identity system links written (or, under --dry-run, that would be
    # written) and link attempts that raised, both reported at the end of the summary.
    system_links = 0
    links_skipped = 0
    # De-dups the dry-run target counter within this invocation only (see the dry-run
    # branch below): a real run saves the target on the first request in a group and
    # matches it on the second, but a dry run never saves anything, so without this set
    # an unsaved shared target would be counted as newly-missing on every repeat.
    dry_run_target_names_seen: set[str] = set()
    # D-01/D-02: every touched Target (matched or newly built), keyed by pk in a real
    # run and, under --dry-run only, by name for a would-be-new target that was never
    # saved and so has no pk -- an already-existing target is keyed by pk in both modes
    # so two portal names fuzzy-matching one Target are collected once, not twice.
    # Values are the Target instances, so the post-loop TargetList step can .add() them
    # directly with no extra query. This is a separate container from
    # dry_run_target_names_seen, which guards a different counter and must not be
    # perturbed here.
    collected_targets: dict[Any, Target] = {}

    for request_group in _iter_request_groups(facility, proposal, created_after, created_before):
        requestgroups_seen += 1
        if not _within_created_window(request_group, parsed_created_after, parsed_created_before):
            continue

        requests_in_group = request_group.get('requests', [])
        processed_in_group: list[Any] = []

        for request in requests_in_group:
            observation_id_raw = request.get('id')
            if observation_id_raw is None:
                skipped += 1
                stderr.write('Skipping request: payload has no id.\n')
                continue
            observation_id = str(observation_id_raw)

            target_dict = _first_named_target(request)
            if target_dict is None:
                skipped += 1
                stderr.write(f'Skipping request {observation_id}: no configuration with a named target.\n')
                continue

            target_name = target_dict.get('name')
            target = Target.matches.match_fuzzy_name(target_name).first()
            is_new_target = False
            if target is None:
                target, reason = _build_non_sidereal_target(target_dict)
                if target is None:
                    skipped += 1
                    stderr.write(f'Skipping request {observation_id}: {reason}.\n')
                    continue
                is_new_target = True

            parameters = _build_parameters(request_group, request)
            if parameters is None:
                skipped += 1
                stderr.write(f'Skipping request {observation_id}: no configuration with a usable instrument_type.\n')
                continue

            status = request.get('state', '')
            # Read once, before the schedule is resolved, so the F2 skip decision sees the
            # stored state. The dry-run branch reuses it; the real branch keeps its own
            # get_or_create below, so a duplicate (facility, observation_id) pair still
            # fails the sweep exactly as before.
            existing_record = ObservationRecord.objects.filter(
                facility=facility.name, observation_id=observation_id
            ).first()
            lookup_needed = _schedule_lookup_is_needed(
                existing_record,
                status,
                terminal_states,
                failed_states,
                recheck_unscheduled=recheck_unscheduled,
            )
            scheduled_start, scheduled_end, lookup_failed, embedded = _resolve_schedule(
                facility, request, dry_run, skip_live_lookup=not lookup_needed
            )
            # Requests skipped above (no id, no named target, no usable elements/
            # instrument_type) never reach here, so they are never counted under any
            # schedule-path counter.
            if embedded:
                embedded_blocks += 1
            elif lookup_needed:
                fallback_lookups_needed += 1
            else:
                fallback_lookups_skipped += 1
            if lookup_failed:
                block_lookups_failed += 1
                stderr.write(f'Failed to resolve observed block for observation_id={observation_id!r}.\n')
            scheduled_start = _parse_datetime_value(scheduled_start)
            scheduled_end = _parse_datetime_value(scheduled_end)
            # The schedule is compared only when this run actually resolved one. A skipped
            # lookup, or a dry-run fallback, compares status and parameters only, so both
            # modes decide identically (T-ik7-02). A failed live lookup resolved nothing, so it
            # never writes a schedule field either: an existing record keeps its stored times
            # (and its stored status, see _changed_record_fields) instead of having them erased.
            compare_schedule = (embedded or (lookup_needed and not dry_run)) and not lookup_failed

            if dry_run:
                if existing_record is None:
                    created += 1
                    record_verb = 'create'
                else:
                    changes = _changed_record_fields(
                        existing_record,
                        status,
                        scheduled_start,
                        scheduled_end,
                        parameters,
                        compare_schedule=compare_schedule,
                        lookup_failed=lookup_failed,
                        mark_lookup_failed=_failed_lookup_needs_marker(
                            existing_record, status, terminal_states, failed_states, lookup_failed=lookup_failed
                        ),
                    )
                    if changes:
                        updated += 1
                        record_verb = 'update'
                    else:
                        unchanged += 1
                        record_verb = 'leave unchanged'

                if is_new_target and target_name not in dry_run_target_names_seen:
                    dry_run_target_names_seen.add(target_name)
                    targets_created += 1
                    target_verb = 'create'
                else:
                    target_verb = 'reuse'

                # D-01/D-02: keyed by pk when the target already exists (so it is
                # collected once even if a second portal name fuzzy-matches it later in
                # the same run), or by name for a would-be-new target with no pk yet.
                collected_targets[target.pk if target.pk else target_name] = target

                stdout.write(
                    f'Would {target_verb} target {target_name!r}; '
                    f'would {record_verb} ObservationRecord '
                    f'observation_id={observation_id!r} status={status!r}.\n'
                )
                # ALLOC-06 / T-kpy-01: ask the matcher about the record this sweep would see, with
                # dry_run=True so nothing is written. An existing record is judged on its STORED
                # window (a dry run cannot see a schedule change the real pass would apply); a
                # not-yet-created record is judged as an unsaved instance (instantiating fires no
                # signal and writes nothing) on its request window or embedded block. A would-be-new
                # target has no pk, is in no campaign and carries no run, so nothing is asked.
                if existing_record is not None:
                    link_subject = existing_record
                elif target.pk is not None:
                    link_subject = ObservationRecord(
                        target=target,
                        facility=facility.name,
                        observation_id=observation_id,
                        status=status,
                        parameters=parameters,
                        scheduled_start=scheduled_start,
                        scheduled_end=scheduled_end,
                    )
                else:
                    link_subject = None
                if link_subject is not None:
                    link_outcome = attempt_system_link(link_subject, dry_run=True, stdout=stdout, stderr=stderr)
                    if link_outcome in (OUTCOME_LINKED, OUTCOME_WOULD_LINK):
                        system_links += 1
                    elif link_outcome == OUTCOME_SKIPPED:
                        links_skipped += 1
                processed_in_group.append(True)
                continue

            if is_new_target:
                target.save()
                targets_created += 1

            # D-01/D-02: every target reaching this line has a pk -- a matched one
            # already had it, a new one was just saved above.
            collected_targets[target.pk] = target

            record, record_created = ObservationRecord.objects.get_or_create(
                facility=facility.name,
                observation_id=observation_id,
                defaults={
                    'target': target,
                    'user': user,
                    'status': status,
                    # A record created while its lookup failed is marked, so the gate looks it up on
                    # every later run until a lookup succeeds (37.1-REVIEW CR-01).
                    'parameters': {**parameters, SCHEDULE_LOOKUP_FAILED_KEY: True} if lookup_failed else parameters,
                    'scheduled_start': scheduled_start,
                    'scheduled_end': scheduled_end,
                },
            )
            if record_created:
                created += 1
            else:
                changes = _changed_record_fields(
                    record,
                    status,
                    scheduled_start,
                    scheduled_end,
                    parameters,
                    compare_schedule=compare_schedule,
                    lookup_failed=lookup_failed,
                    mark_lookup_failed=_failed_lookup_needs_marker(
                        record, status, terminal_states, failed_states, lookup_failed=lookup_failed
                    ),
                )
                if changes:
                    for field, value in changes.items():
                        setattr(record, field, value)
                    record.save()
                    updated += 1
                else:
                    unchanged += 1

            # ALLOC-06 / D-01: every record the sweep touches -- created, updated or unchanged --
            # is offered to the exact-identity matcher, because runs are often created after their
            # records. The record's own status has no effect on whether it links (D-03).
            link_outcome = attempt_system_link(record, dry_run=False, stdout=stdout, stderr=stderr)
            if link_outcome in (OUTCOME_LINKED, OUTCOME_WOULD_LINK):
                system_links += 1
            elif link_outcome == OUTCOME_SKIPPED:
                links_skipped += 1

            processed_in_group.append(record)

        if len(requests_in_group) > 1 and processed_in_group:
            group_name = _group_name(request_group)
            if dry_run:
                would_reuse = ObservationGroup.objects.filter(name=group_name).exists()
                if would_reuse:
                    groups_reused += 1
                else:
                    groups_created += 1
                stdout.write(f'Would {"reuse" if would_reuse else "create"} ObservationGroup {group_name!r}.\n')
            else:
                group, group_was_created = ObservationGroup.objects.get_or_create(name=group_name)
                group.observation_records.add(*processed_in_group)
                if group_was_created:
                    groups_created += 1
                else:
                    groups_reused += 1

    list_name = target_list_name or f'{proposal}_targets'
    targets_added = len(collected_targets)
    if dry_run:
        # T-kpy-01: no get_or_create, no .add() -- reads only, so a dry run writes
        # nothing at all.
        list_reused = TargetList.objects.filter(name=list_name).exists()
    else:
        # D-05: the list is created unconditionally, even when the sweep touched zero
        # targets, because get_or_create runs before .add() regardless.
        target_list, list_was_created = TargetList.objects.get_or_create(name=list_name)
        # The M2M add is set-like, which is what makes a re-run non-duplicating.
        target_list.targets.add(*collected_targets.values())
        list_reused = not list_was_created

    if dry_run:
        list_verb = 'would reuse' if list_reused else 'would create'
    else:
        list_verb = 'reused' if list_reused else 'created'

    summary = (
        f'requestgroups seen: {requestgroups_seen}, '
        f'{"would create" if dry_run else "created"}: {created}, '
        f'{"would update" if dry_run else "updated"}: {updated}, '
        f'unchanged: {unchanged}, skipped: {skipped}, '
        f'{"targets would create" if dry_run else "targets created"}: {targets_created}, '
        f'{"groups would create" if dry_run else "groups created"}: {groups_created}, '
        f'{"groups would reuse" if dry_run else "groups reused"}: {groups_reused}, '
        f'embedded blocks: {embedded_blocks}, fallback lookups needed: {fallback_lookups_needed}, '
        f'fallback lookups skipped: {fallback_lookups_skipped}, '
        f'block lookups failed: {"n/a (dry-run)" if dry_run else block_lookups_failed}, '
        f'target list: {list_verb} {list_name!r}, '
        f'{"targets would add to list" if dry_run else "targets added to list"}: {targets_added}, '
        f'{"would link" if dry_run else "system links"}: {system_links}, links skipped: {links_skipped}'
    )
    return summary


def watched_rows():
    """Return every active `WatchedProposal` row, ready for the bare-invocation sweep.

    Ordering is `proposal_code` ascending -- inherited from `WatchedProposal.Meta.ordering`
    (36-CONTEXT.md D-06), not restated here, so a future change to that ordering needs no
    matching edit in this function.

    Returns:
        QuerySet[WatchedProposal]: every row with `is_active=True`, with `attributed_to`
            pre-fetched (each row's sweep reads it).
    """
    return WatchedProposal.objects.filter(is_active=True).select_related('attributed_to')


def sweep_watched_rows(
    *,
    dry_run: bool,
    recheck_unscheduled: bool = False,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> tuple[int, int, list[str]]:
    """Sweep every active ``WatchedProposal`` row through ``sweep_proposal()`` (D-07..D-09).

    IN-13 (36-REVIEW.md): the single source of the watched-list sweep loop -- ``Command.
    handle()``'s bare-invocation path and ``unattended.step_discovery()`` were two
    near-identical copies of this same query/sweep/bookkeeping/failure-isolation logic,
    with CR-02's own `--proposal`-only guard living in only one of them (correct today
    only because the runner bypasses the CLI). Each caller does its own terminal
    reporting (a `StepResult` vs. a `CommandError`) from the counts this returns.

    Args:
        dry_run: report what would change without writing any ``WatchedProposal``
            bookkeeping when True.
        recheck_unscheduled: forwarded to ``sweep_proposal()`` for every row. ``Command.handle()``
            sets it from ``--recheck-unscheduled``; ``unattended.step_discovery()`` never does.
        stdout: forwarded to ``sweep_proposal()``. ``Command.handle()``'s bare-invocation
            path passes ``self.stdout`` (a Django ``OutputWrapper``); ``unattended.
            step_discovery()`` passes a throwaway ``io.StringIO()`` it captures and logs
            (IN-02, 36-REVIEW.md), since the runner has no terminal of its own to write
            progress lines to.
        stderr: forwarded to ``sweep_proposal()``, and used to report the class name of
            any per-row failure (D-17) when supplied.

    Returns:
        tuple[int, int, list[str]]: ``(rows_swept, failed_count, failed_codes)``.
            ``rows_swept`` is 0 for a legitimately empty watch list (D-08, not a
            failure). A per-row failure is caught and recorded on that row's own
            ``last_run_summary`` (class name only, D-17) without stopping the remaining
            rows (D-09).
    """
    rows = list(watched_rows())
    failed_count = 0
    failed_codes: list[str] = []
    for row in rows:
        try:
            summary = sweep_proposal(
                row.proposal_code,
                target_list_name=row.target_list_name or None,
                user=row.attributed_to,
                dry_run=dry_run,
                recheck_unscheduled=recheck_unscheduled,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as exc:  # noqa: BLE001 -- the only catch point, D-09
            # D-17: a portal/facility/network exception's message can embed request or
            # response content, so only the class name ever reaches the row, stderr, or
            # the log -- never str(exc).
            summary = f'failed: {type(exc).__name__}'
            logger.debug('sweep_proposal() failed for proposal_code=%r: %s', row.proposal_code, type(exc).__name__)
            if stderr is not None:
                stderr.write(f'Proposal {row.proposal_code!r}: {summary}')
            failed_count += 1
            failed_codes.append(row.proposal_code)

        # D-06/D-09: bookkeeping is written either way (success or failure) so the
        # admin's last_run_at/last_run_summary columns are never stale for a row that
        # was actually swept -- but a dry run writes nothing at all (D-07's own
        # dry-run contract: report, never persist).
        if not dry_run:
            row.last_run_at = timezone.now()
            row.last_run_summary = summary
            row.save(update_fields=['last_run_at', 'last_run_summary'])

    return len(rows), failed_count, failed_codes


class Command(BaseCommand):
    """Backfill ObservationRecords, non-sidereal Targets, and ObservationGroups for LCO
    RequestGroups, campaign-agnostic and safe to re-run.

    Queries the LCO Observation Portal's 'Get All RequestGroups' API
    (GET /api/requestgroups/) for a proposal and creates one ObservationRecord per child
    request (facility='LCO', observation_id=<request id>). Unlike
    backfill_lco_observation_records, a request whose ObservationRecord already exists is
    updated in place (status/scheduled_start/scheduled_end/parameters) rather than skipped,
    and a request whose target isn't already in FOMO gets a newly built non-sidereal Target
    from its orbital elements -- never a sidereal one, and never interactively prompted for.

    An LCO RequestGroup carrying more than one request is linked into a reusable
    ObservationGroup; a single-request RequestGroup gets no group.

    A --dry-run pass reports what a real pass over the same portal payload would do -- same
    created/updated/unchanged/target/group counts, labelled with the would-forms -- with one
    honest caveat: for a request that would need the live fallback schedule lookup (no
    embedded 'observations' block), a dry run compares status and parameters only, since the
    schedule fields it would otherwise compare are never resolved under --dry-run. Such a
    record can therefore be reported unchanged by a dry run when only its schedule times
    would actually move on a real pass. A record whose lookup is skipped (see below) is
    compared the same way in both modes: status and parameters only.

    The list payload carries no observed blocks, so each request would otherwise cost a live
    get_observation_status() lookup (two portal GETs) on every tick (F2). A record already in
    one of the facility's terminal states, which the portal still reports in that same state,
    is therefore not looked up: its status and parameters are compared and its stored
    scheduled_start/scheduled_end stay as they are. It is counted under 'fallback lookups
    skipped'. A brand-new request, a record still in a non-terminal state, a record whose
    state changed, and a completed record still missing its scheduled times are all looked up
    (the last so a failed lookup is retried rather than frozen). A request with an embedded
    'observations' block is unaffected, since reading it costs no network call.

    A failed lookup never writes either schedule field: an existing record keeps its stored times
    and its stored status (the state change is held back, so the next run sees it and looks the
    request up again, and a successful lookup stores the new status and the block's times in the
    same save). A record created while its lookup fails is marked with the 'schedule_lookup_failed'
    parameters key and is looked up on every later run until a lookup succeeds, which removes the mark.
    A failed --recheck-unscheduled lookup on a record the per-tick skip would otherwise leave alone marks
    it the same way, so the unattended runner retries it until a lookup succeeds. A block-list reply that
    is not a list is a failed lookup too (FomoLCOFacility raises UnexpectedBlockPayloadError for it).

    A request that expired or was cancelled can still carry a block that started and was aborted
    after taking data, and FOMO's block rule (first completed block, else the last aborted or
    in-progress block, else the last pending one) stores that block's times, for both the embedded
    list and the live lookup. Records stored before that rule, and records finished through one of
    TOM Toolkit's own status routes, have no times and the skip above leaves them alone.
    --recheck-unscheduled (opt-in, works with --proposal and on the bare form) looks up once every
    record missing a scheduled time, including those finished records, so they pick up an aborted
    block's times; a linked record that gains them retires its run's allocation night in that same
    sweep. It costs one portal lookup per such record and is never used by the unattended runner.

    Every Target the sweep touches -- matched by fuzzy name or newly built from orbital
    elements -- is collected into a TargetList named '<proposal>_targets', created on the
    first run and reused on every re-run; --target-list NAME overrides the derived name. A
    skipped request contributes nothing to the list. A dry run reports which list it would
    create or reuse and how many targets it would add, without creating the list at all.

    36-CONTEXT.md D-07: --proposal is now optional. Given, this behaves exactly as before --
    a one-off manual sweep of that single code, which does not have to be a WatchedProposal
    row and writes no WatchedProposal bookkeeping. Omitted (the invocation the unattended
    runner uses), the command instead sweeps every active WatchedProposal row in
    proposal_code order, applying that row's target_list_name/attributed_to overrides,
    isolating a portal or data error to that row alone (D-09), and recording
    last_run_at/last_run_summary on every row it swept. An empty watched list is a quiet,
    zero-exit no-op (D-08).

    System links (ALLOC-06): after each request is written (or, under --dry-run, previewed),
    the record is checked for an exact match with one approved campaign run -- same proposal
    code, same target (or, for a field-pointing record, the one run its campaign holds for that
    proposal) and a run window that contains the record's. A match is linked on the spot as a
    system link (no staff confirmation, undoable from the attribution queue), which retires the
    run's allocation night for that block on the same sweep. Re-running writes no second link.
    The summary ends with 'system links: N, links skipped: M' ('would link' under --dry-run);
    every link also writes one line naming the observation id, the run and the match basis.
    The similarity scorer is never consulted: an ambiguous or partial match stays in the
    attribution queue for a person to decide.
    """

    help = 'Backfill ObservationRecords, non-sidereal Targets and ObservationGroups from LCO RequestGroups'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--proposal',
            required=False,
            default=None,
            help=(
                'LCO proposal code to filter RequestGroups by (exact match). Omit to sweep every '
                'active WatchedProposal row instead (36-CONTEXT.md D-07); the named code need not '
                'be a WatchedProposal row.'
            ),
        )
        parser.add_argument(
            '--created-after',
            required=False,
            help=(
                'Only backfill RequestGroups created on/after this ISO-8601 timestamp/date. '
                'Requires --proposal (CR-02, 36-REVIEW.md): the watched-list sweep has no window '
                'argument and takes its overrides from each WatchedProposal row.'
            ),
        )
        parser.add_argument(
            '--created-before',
            required=False,
            help=(
                'Only backfill RequestGroups created on/before this ISO-8601 timestamp/date. '
                'Requires --proposal -- see --created-after.'
            ),
        )
        parser.add_argument(
            '--username',
            required=False,
            help=(
                'Attribute created/updated ObservationRecords to this username (default: unattributed). '
                'Requires --proposal -- see --created-after.'
            ),
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be created/updated without writing anything.',
        )
        parser.add_argument(
            '--target-list',
            required=False,
            help=(
                'Override the derived "<proposal>_targets" name for the TargetList the sweep collects '
                'into. Requires --proposal -- see --created-after.'
            ),
        )
        parser.add_argument(
            '--recheck-unscheduled',
            action='store_true',
            help=(
                'Also look up once every record that is missing a scheduled time, including finished '
                'records the per-tick skip leaves alone, using the block rule FOMO applies (a block that '
                'started and was aborted after taking data counts), so records stored before that rule '
                'pick up their block times. One portal lookup per such record. Works with or without '
                '--proposal. Never used by the unattended runner. A failed lookup on a record the per-tick skip '
                'would otherwise leave alone marks it schedule_lookup_failed; the unattended runner retries a '
                'marked record only while its proposal is an active watched proposal, so for any other '
                '--proposal code re-run the command (no flag needed) until block lookups failed is 0. '
                'Default off.'
            ),
        )

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Resolve CLI-only arguments, then either sweep the single --proposal override or
        every active WatchedProposal row (D-07).

        Returns:
            str | None: a one-line summary for the override path (unchanged from Task 2);
                None for the watched path, which writes its own summary lines directly so
                BaseCommand.execute()'s auto-write of a return value never duplicates them.

        Raises:
            CommandError: --username names an unknown user; --created-after/--created-before
                (override path only) is not a valid ISO-8601 timestamp/date; --proposal is
                omitted while --created-after/--created-before/--username/--target-list is
                given (CR-02, 36-REVIEW.md -- the watched-list sweep has no window argument
                and would otherwise silently discard it); or (watched path) one or more
                watched proposals failed, naming the failing code(s).
        """
        proposal = options.get('proposal')
        dry_run = options['dry_run']
        # Not one of the CR-02 proposal-only flags below: the bare watched-list sweep honours it.
        recheck_unscheduled = options['recheck_unscheduled']

        # CR-02 (36-REVIEW.md): the watched-list sweep takes its overrides from each
        # WatchedProposal row and has no window argument at all, so --created-after/
        # --created-before/--username/--target-list are silently discarded on this path
        # -- argparse accepts them, but they never reach sweep_proposal(). Fail closed
        # instead of letting an operator believe a window/attribution override applied
        # to a full-history sweep of every watched proposal.
        #
        # IN-12 (36-REVIEW.md): checked before --username is resolved to a User below (a
        # DB query that raises its own, less useful CommandError on an unknown name), and
        # against "was the flag supplied at all" (`is not None`), not truthiness -- a
        # flag given as an empty string (e.g. --target-list '') was still supplied and
        # must still trip this guard.
        if not proposal:
            ignored = [
                flag
                for flag, key in (
                    ('--created-after', 'created_after'),
                    ('--created-before', 'created_before'),
                    ('--username', 'username'),
                    ('--target-list', 'target_list'),
                )
                if options.get(key) is not None
            ]
            if ignored:
                verb = 'requires' if len(ignored) == 1 else 'require'
                raise CommandError(
                    f'{", ".join(ignored)} {verb} --proposal; the watched-list sweep takes its '
                    'overrides from each WatchedProposal row.'
                )

        user = None
        if options.get('username'):
            try:
                user = get_user_model().objects.get(username=options['username'])
            except get_user_model().DoesNotExist as exc:
                raise CommandError(f'Invalid username: {options["username"]!r}') from exc

        if proposal:
            # The override path: unchanged from Task 2. The named code does not have to be
            # a WatchedProposal row, and no WatchedProposal bookkeeping is written here.
            return sweep_proposal(
                proposal,
                target_list_name=options.get('target_list'),
                user=user,
                created_after=options.get('created_after'),
                created_before=options.get('created_before'),
                dry_run=dry_run,
                recheck_unscheduled=recheck_unscheduled,
                stdout=self.stdout,
                stderr=self.stderr,
            )

        # IN-13 (36-REVIEW.md): the query/sweep/bookkeeping/failure-isolation loop itself
        # lives in sweep_watched_rows() -- shared with unattended.step_discovery() -- so
        # this method only needs its own terminal reporting (a written summary line and a
        # CommandError) from the counts that helper returns.
        rows_swept, failed_count, failed_codes = sweep_watched_rows(
            dry_run=dry_run, recheck_unscheduled=recheck_unscheduled, stdout=self.stdout, stderr=self.stderr
        )
        if not rows_swept:
            # D-08: a legitimately empty watch list is a healthy, quiet no-op -- not a
            # failure -- so this is INFO, not a warning/error, and the command still exits 0.
            message = '0 watched proposals, nothing to discover'
            logger.info(message)
            self.stdout.write(message)
            return None

        self.stdout.write(f'Swept {rows_swept} watched proposal(s), failed: {failed_count}')

        if failed_count:
            # CommandError is FOMO's own exception (D-17's second bucket): its message may
            # name the failing proposal codes, since that string never embeds a portal
            # response body or header.
            raise CommandError(f'{failed_count} watched proposal(s) failed: {", ".join(failed_codes)}')
        return None
