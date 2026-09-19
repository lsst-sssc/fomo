"""Pure-logic core of the coverage-gap analysis feature (GAP-01/GAP-02).

Composes ``telescope_runs.sun_event()`` (the observable side, dark-window-only per
``17-GAP-01-DECISION.md``) with a ``CampaignRun`` query (the claimed side) into a set
difference, cached via Django's low-level cache framework with a 1-hour TTL. Mirrors
``campaign_utils.py``'s role: a pure-logic helper module with no view/request concerns,
structured with the same "never raise for expected messy data" discipline.

This module depends only on the heavy SPICE-loading ephemeris module's read-only,
already-tested sun-event helper for its ephemeris needs -- it must never import the heavy
SPICE-loading ephemeris module (or any module that imports it, such as ``solsys_code.views``)
at module scope. That module's ~1.6 GB SPICE-kernel download side effect (CLAUDE.md "Heavy
import side effect") would otherwise be paid by every process that imports this module.
"""

import logging
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.core.cache import cache
from django.db.models import F, Q
from django.utils import timezone
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import derive_telescope, record_time_window
from solsys_code.campaign_attribution import (
    LCO_SITE_CODE_TO_OBSCODE,
    OBSERVED_TELESCOPE_OBSCODES,
    _extract_lco_site_code,
)
from solsys_code.models import CampaignRun
from solsys_code.observation_projector import facility_for_or_none
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.status_vocabulary import DisplayState, classify_record
from solsys_code.telescope_runs import observing_night, sun_event

logger = logging.getLogger(__name__)

GAP_CACHE_TTL_SECONDS = 3600  # D-10: 1-hour result cache
DEFAULT_WINDOW_DAYS = 90  # D-11: default date-range window
MAX_WINDOW_DAYS = 180  # D-11: hard cap on requested date-range span

# D-05: a CampaignRun in one of these run_status values never "claims" a date, even if
# approval_status=APPROVED -- a run that fell through in the real world frees its date
# back up as a gap.
_EXCLUDED_RUN_STATUSES = frozenset(
    {
        CampaignRun.RunStatus.CANCELLED,
        CampaignRun.RunStatus.NOT_AWARDED,
        CampaignRun.RunStatus.WEATHER_TECH_FAILURE,
    }
)

# D-16 (GAPB-01): only a real placed block claims a night -- a queue window is not a set of
# owned nights (the Phase 26/35 domain correction), and a record that expired, was cancelled
# or failed claims nothing.
_CLAIMING_DISPLAY_STATES = frozenset({DisplayState.OBSERVED, DisplayState.SCHEDULED})


def clamp_date_range(today: date, requested_end: date | None) -> tuple[date, date]:
    """Enforce D-11's 90-day default / 180-day max span, independent of client input.

    Args:
        today: the local "today" the range starts from (always the start of the range).
        requested_end: a client-supplied end date, or None to use the 90-day default.

    Returns:
        tuple[date, date]: (start, end), where start is always `today` and end is never
            later than `today + MAX_WINDOW_DAYS` days, regardless of `requested_end`.
    """
    start = today
    default_end = start + timedelta(days=DEFAULT_WINDOW_DAYS)
    max_end = start + timedelta(days=MAX_WINDOW_DAYS)
    if requested_end is None:
        return start, default_end
    # WR-02: also floor at `start` -- otherwise a past `requested_end` (e.g. a client
    # submitting end_date=2020-01-01) produces end < start, an empty range, and a
    # misleading "no gaps found" instead of reflecting that nothing was actually searched.
    return start, max(start, min(requested_end, max_end))


def build_gap_cache_key(campaign_pk: int, target_pk: int | None, site_pk: int, start: date, end: date) -> str:
    """Build a stable, collision-free cache key for a gap-analysis request (D-10).

    Args:
        campaign_pk: pk of the campaign (TargetList).
        target_pk: pk of the selected Target, or None for a single-target campaign that
            has no per-target disambiguation need (D-12). Encoded explicitly as the
            literal 'none' rather than omitted, so a null-target request never collides
            with a differently-scoped one (D-10 / Information Disclosure control).
        site_pk: pk of the selected Observatory.
        start: inclusive start date of the requested range.
        end: inclusive end date of the requested range.

    Returns:
        str: a delimited cache key including all four dimensions (campaign, target,
            site, date range).
    """
    target_segment = str(target_pk) if target_pk is not None else 'none'
    return f'campaign_gap:{campaign_pk}:{target_segment}:{site_pk}:{start.isoformat()}:{end.isoformat()}'


def observable_dates(site, start: date, end: date) -> set[date]:
    """Return the set of dates in [start, end] with a non-zero -15 degree dark window.

    D-04: any non-zero dark window counts as observable -- no minimum-duration threshold.
    D-03: a per-date `sun_event(kind='dark')` ValueError (e.g. a hypothetical future
    polar/midnight-sun Observatory) skips that one date as "unknown"; it never aborts the
    rest of the loop, matching this codebase's established per-record log+skip discipline.

    Args:
        site: an Observatory instance (sun_event() accepts any Observatory, not just a
            SITES-dict-registered one).
        start: inclusive start date.
        end: inclusive end date.

    Returns:
        set[date]: dates whose dark window is non-zero.
    """
    observable = set()
    n_days = (end - start).days + 1
    for i in range(n_days):
        d = start + timedelta(days=i)
        try:
            sun_event(site, d, kind='dark')
            observable.add(d)
        except ValueError:
            logger.debug('sun_event(dark) raised for site=%s date=%s; skipping as unknown (D-03).', site, d)
    return observable


def observation_site_obscode(record, attributed_site_obscode: str | None) -> str | None:
    """D-17 site-resolution ladder for one observation event.

    Resolution order: (1) the record's own ``parameters['observed_site']``/
    ``['observed_telescope']`` (the observation projector's one-time observed-site lookup,
    Phase 34 D-09), mapped through ``calendar_utils.derive_telescope()`` to a verified
    telescope label and then to an obscode via ``campaign_attribution``'s label-keyed
    ``OBSERVED_TELESCOPE_OBSCODES`` (the three renamed 2m0/4m0 telescopes) or, for a
    SITECODE-CLASS label, its site-keyed ``LCO_SITE_CODE_TO_OBSCODE``; (2) failing that, the
    obscode of the site of the CampaignRun this event is attributed to (``attributed_site_obscode``,
    read from the caller's own ``CalendarEventMeta.run.site`` annotation); (3) otherwise
    ``None`` -- the event is not assignable to any site.

    A ``None`` result must be reported by the caller as claimed-but-site-unknown and must
    never be treated as a match for the caller's selected site.

    Args:
        record: the ObservationRecord being resolved (reads ``parameters`` only).
        attributed_site_obscode: the obscode of the site of the CampaignRun this event is
            attributed to via ``CalendarEventMeta.run``, or None if unattributed or that
            run's own site is unset.

    Returns:
        str | None: the resolved Observatory obscode, or None. Never raises: a missing
            parameter key, an unmapped (site, telescope) pair and an unknown site code all
            fall through to the next rung rather than raising.
    """
    parameters = record.parameters or {}
    label = derive_telescope(parameters.get('observed_site'), parameters.get('observed_telescope'))
    if label:
        obscode = OBSERVED_TELESCOPE_OBSCODES.get(label)
        if obscode is None:
            lco_site_code = _extract_lco_site_code(label)
            if lco_site_code:
                obscode = LCO_SITE_CODE_TO_OBSCODE.get(lco_site_code)
        if obscode is not None:
            return obscode
    return attributed_site_obscode


def observation_claimed_dates(campaign, target, site) -> tuple[set[date], int]:
    """The second GAPB-01 claim source: observed/scheduled events on the campaign calendar.

    D-18: "on the campaign calendar" is the union of an event attributed to one of the
    campaign's runs (``CalendarEventMeta.run``) OR the record's own target belonging to the
    campaign's TargetList -- an unattributed classical/queue observation of the campaign's
    own target counts even before anyone works the attribution queue. D-16: only an
    OBSERVED- or SCHEDULED-classified record's placed block claims a night; a queued
    request's window claims nothing, and an expired/cancelled/failed/inconsistent record
    claims nothing. D-17: an event whose site cannot be resolved increments the
    site-unknown counter and closes no gap.

    T-37-09: this queryset carries its own explicit ``.only()`` field restriction and never
    eagerly joins the whole attributed-run row into memory -- a wide eager join could later
    be widened to pull ``CampaignRun.contact_person``/``.contact_email`` onto a page that is
    public. The attributed run's site obscode is read instead as a single annotated scalar
    column.

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.

    Returns:
        tuple[set[date], int]: (observation_claimed_dates, site_unknown_count). Not
            range-bounded, mirroring ``claimed_dates()``'s own WR-05 note.
    """
    try:
        site_zone = ZoneInfo(site.timezone)
    except (ZoneInfoNotFoundError, TypeError, ValueError):
        # A site with no usable IANA timezone can't derive a site-local observing night for
        # any observation event -- never abort the gap page for it (D-16/17 concern the
        # per-record site, not this query-level site's own timezone).
        logger.debug('observation_claimed_dates: site pk=%s has no usable timezone; skipping.', site.pk)
        return set(), 0

    qs = ObservationRecord.objects.filter(
        Q(calendar_event_meta__run__campaign=campaign) | Q(target__in=campaign.targets.all())
    )
    if campaign.targets.count() != 1:
        # Multi-target campaign: the same target rule claimed_dates() uses above.
        qs = qs.filter(target=target)
    qs = qs.annotate(attributed_site_obscode=F('calendar_event_meta__run__site__obscode'))
    qs = qs.only('pk', 'status', 'facility', 'scheduled_start', 'scheduled_end', 'parameters')

    observation_claimed: set[date] = set()
    site_unknown_count = 0
    for record in qs:
        # CR-03 (37-REVIEW.md): facility_for_or_none() never raises for a stale/
        # unconfigured facility name. A record with no resolvable facility is counted as
        # site-unknown, the same bucket a resolvable-but-unattributed site already falls
        # into just below -- never a 500 on the gap-analysis page.
        facility = facility_for_or_none(record)
        if facility is None:
            site_unknown_count += 1
            continue
        if classify_record(record, facility) not in _CLAIMING_DISPLAY_STATES:
            continue
        obscode = observation_site_obscode(record, record.attributed_site_obscode)
        if obscode is None:
            site_unknown_count += 1
            continue
        if obscode != site.obscode:
            continue
        try:
            start_time, _end_time = record_time_window(record)
        except (KeyError, ValueError):
            logger.debug('record_time_window() raised for observation record pk=%s; skipping as unknown.', record.pk)
            continue
        observation_claimed.add(observing_night(start_time, site_zone))

    return observation_claimed, site_unknown_count


def claimed_dates(campaign, target, site) -> tuple[set[date], list, list, list, set[date], int]:
    """Return the set of dates claimed by approved, non-terminal-failure CampaignRuns.

    D-05: a date is claimed when a CampaignRun has approval_status=APPROVED and
    run_status is not in {cancelled, not_awarded, weather_tech_failure}.

    Target attribution (Pitfall 4 / D-12): if the campaign has exactly one Target, the
    query does NOT filter by target -- the single target is implied, and real imported
    runs commonly have target=None (per import_campaign_csv's single-target
    auto-assignment precedent). If the campaign has more than one Target, the query
    filters target=<selected target> strictly, and target=None rows are collected into a
    separate "unattributed" list rather than being counted as claiming (or not claiming)
    any specific target's dates -- a data-quality signal, not a silent guess either way.

    Ground-vs-space asset-awareness (ASSET-01/ASSET-02, D-09): the classification is
    computed once, before the loop, from the ``site`` parameter (``site.observations_type
    == Observatory.SATELLITE_OBSTYPE``) -- never re-read per-row (Pitfall 3), since the
    queryset is already filtered to this single site. For a ground run, every date in the
    inclusive range [window_start, window_end] is claimed (a single-night run has
    window_start == window_end, so exactly one date is claimed). A space-mission run whose
    window hasn't narrowed to a single night (window_start != window_end) claims nothing
    and is collected into a separate "pending narrowing" list instead -- a space
    observatory has no fixed horizon, so claiming every date in a broad window would
    wrongly mark those nights as covered. A run with window_start is None (TBD) cannot be
    attributed to any date regardless of site type -- it is collected into a separate
    "undated" list, never added to the claimed set and never added to "pending narrowing"
    (D-09 explicit distinction: "no info at all" vs. "a real space-mission run with a
    range, just not scheduled tight enough yet").

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.

    GAPB-01/D-19: after the run-window loop below, ``observation_claimed_dates()`` is called
    and its result is unioned into the same ``claimed`` set -- an observed/scheduled
    observation block claims a night alongside, never instead of, an approved run window's
    claims (the ``_EXCLUDED_RUN_STATUSES`` rule above is unchanged). The observation-only
    subset is also returned separately (as ``observation_claimed``) so a caller can say
    which kind of claim covered a given night, along with the count of observation events
    this call could not assign to any site (``site_unknown_count``, D-17) -- a listed count,
    never a silent drop.

    WR-05: unlike ``observable_dates(site, start, end)``, this function takes no date-range
    parameters -- it returns every approved, non-excluded ``CampaignRun`` for the campaign/
    site combination regardless of any requested window. ``_compute_gap()`` only ever
    evaluates the range-bounded ``gap = obs - claimed`` against the range-bounded ``obs``
    set, so ``gap_dates`` is correct -- but the returned ``claimed_dates``/``undated_runs``/
    ``unattributed_runs``/``pending_narrowing_runs``/``observation_claimed`` are
    campaign/site-wide, NOT scoped to ``[start, end]``, even though the cached result they
    end up in (``build_gap_cache_key()``) is keyed by a date range. Do not assume a
    range-keyed cache entry's ``claimed_dates`` is itself range-bounded.

    Returns:
        tuple[set[date], list, list, list, set[date], int]: (claimed_dates, undated_runs,
            unattributed_runs, pending_narrowing_runs, observation_claimed,
            site_unknown_count).
    """
    # D-13/WR-01: restrict the columns actually fetched to a PII-free field set (never
    # contact_person/contact_email) before anything is collected into
    # `undated_runs`/`unattributed_runs` and cached -- mirrors CampaignRunTableView's
    # "restrict the queryset, not just the rendered output" discipline. `.only()` (not
    # `.values()`) keeps these as CampaignRun instances so existing pk-based equality and
    # attribute access downstream keep working; only pk/window_start/window_end are fetched.
    qs = CampaignRun.objects.filter(campaign=campaign, site=site, approval_status=CampaignRun.ApprovalStatus.APPROVED)
    qs = qs.exclude(run_status__in=_EXCLUDED_RUN_STATUSES)
    qs = qs.only('pk', 'window_start', 'window_end')

    unattributed_runs: list[CampaignRun] = []
    single_target = campaign.targets.count() == 1
    if not single_target:
        # Multi-target campaign: target=None rows are ambiguous -- don't count them as
        # claiming this specific target's dates, but don't silently drop them either.
        unattributed_runs = list(qs.filter(target__isnull=True))
        qs = qs.filter(target=target)
    # Single-target campaign: don't filter by target at all -- the single target is
    # implied and target=None is the common real-data case (Pitfall 4).

    # ASSET-01: classification computed once from the site parameter, before the loop --
    # never a per-row run.site read (Pitfall 3), which would force widening the
    # PII-minimizing .only('pk', 'window_start', 'window_end') queryset above.
    is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE

    claimed: set[date] = set()
    undated_runs: list[CampaignRun] = []
    pending_narrowing_runs: list[CampaignRun] = []
    for run in qs:
        if run.window_start is None or run.window_end is None:
            # TBD -- can't be attributed to any date (unchanged bucketing rule), regardless
            # of site type (D-09 explicit distinction from pending_narrowing_runs below).
            # WR-02: also catches the DB-CheckConstraint-should-prevent-but-defend-anyway
            # case of a mismatched pair (one set, one NULL) so this never raises a
            # TypeError on read.
            undated_runs.append(run)
            continue
        if is_space_mission and run.window_start != run.window_end:
            # ASSET-02/D-09: a space-mission run with an un-narrowed range claims nothing
            # until a staff edit or CSV re-import narrows it to window_start == window_end
            # (D-10: no automated narrowing mechanism).
            pending_narrowing_runs.append(run)
            continue
        n_days = (run.window_end - run.window_start).days + 1
        for i in range(n_days):
            claimed.add(run.window_start + timedelta(days=i))

    # GAPB-01/D-19: union, never substitution -- the run-window claims above are unchanged;
    # an observed/scheduled observation block adds to the same claimed set.
    observation_claimed, site_unknown_count = observation_claimed_dates(campaign, target, site)
    claimed |= observation_claimed

    return claimed, undated_runs, unattributed_runs, pending_narrowing_runs, observation_claimed, site_unknown_count


def _compute_gap(campaign, target, site, start: date, end: date) -> dict:
    """Compute the coverage-gap result dict (no caching).

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.
        start: inclusive start date.
        end: inclusive end date.

    Returns:
        dict: gap_dates, claimed_dates, observable_dates, observation_claimed_dates (each a
            sorted list of dates), undated_runs, unattributed_runs, pending_narrowing_runs
            (lists of CampaignRun), unknown_date_count (number of dates in range whose
            sun_event() call raised, i.e. dates in range that are neither observable nor
            known-unavailable), and claimed_site_unknown_count (GAPB-01/D-17: number of
            observation events on the campaign calendar that could not be assigned to any
            site -- counted, never silently dropped).
    """
    obs = observable_dates(site, start, end)
    (
        claimed,
        undated_runs,
        unattributed_runs,
        pending_narrowing_runs,
        observation_claimed,
        site_unknown_count,
    ) = claimed_dates(campaign, target, site)
    gap = obs - claimed

    n_days = (end - start).days + 1
    # observable_dates() only ever adds a date when sun_event() succeeds (D-04: any
    # non-zero dark window -- i.e. any successful 2-crossing evaluation -- counts as
    # observable), so every date in range that did NOT end up in `obs` is exactly a date
    # whose sun_event() call raised ValueError (D-03) and was skipped as unknown.
    unknown_date_count = n_days - len(obs)

    return {
        'gap_dates': sorted(gap),
        'claimed_dates': sorted(claimed),
        'observable_dates': sorted(obs),
        'undated_runs': undated_runs,
        'unattributed_runs': unattributed_runs,
        'pending_narrowing_runs': pending_narrowing_runs,
        'unknown_date_count': unknown_date_count,
        'observation_claimed_dates': sorted(observation_claimed),
        'claimed_site_unknown_count': site_unknown_count,
    }


def get_or_compute_gap(campaign, target, site, start: date, end: date) -> dict:
    """Cache-or-compute wrapper for the coverage-gap result (D-10).

    On a cache hit, returns the cached dict unchanged -- its original `computed_at` must
    survive so the "last computed at" display reflects when the result was actually
    computed, not the time of this (cache-hit) request. On a cache miss, computes the
    result, stamps `computed_at`, caches it for GAP_CACHE_TTL_SECONDS, and returns it.

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.
        start: inclusive start date.
        end: inclusive end date.

    Returns:
        dict: see `_compute_gap`'s return value, plus a `computed_at` key.
    """
    key = build_gap_cache_key(campaign.pk, target.pk if target else None, site.pk, start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _compute_gap(campaign, target, site, start, end)
    result['computed_at'] = timezone.now()
    cache.set(key, result, timeout=GAP_CACHE_TTL_SECONDS)
    return result
