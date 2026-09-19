"""Pure-logic core of the public run/campaign tallies (TALLY-01/02).

Every value this module returns is a read-only aggregate. **No function here ever writes
``CampaignRun.run_status``, which remains a staff decision made only through the existing
approval-queue/decision views (TALLY-03).** This module only counts and classifies state
that already exists elsewhere -- it never sets any of it.

Mirrors ``campaign_gap.py``'s import discipline (``solsys_code/campaign_gap.py:1-14``): this
module must never import FOMO's heavy ephemeris-computation module or the view layer that
imports it (or any module that imports either) at module scope, because it is imported by
template tags and by the campaign table -- both would otherwise pay that heavy module's
~1.6 GB SPICE-kernel download side effect (CLAUDE.md "Heavy import side effect") on every
import.
"""

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.core.cache import cache
from django.db.models import Count, F, Max
from tom_observations.models import ObservationGroup, ObservationRecord

from solsys_code.calendar_utils import record_time_window
from solsys_code.models import CampaignRun, CampaignRunObservation
from solsys_code.observation_projector import facility_for
from solsys_code.status_vocabulary import DisplayState, classify_record
from solsys_code.telescope_runs import observing_night

logger = logging.getLogger(__name__)

# D-08/folded-todo (2026-09-01-add-ttl-cache-to-attribution-banner-count.md): these counts
# are computed for every visitor on every campaign-table load -- the same exposure that todo
# measured for the attribution banner count. The SQL-expressible counts
# (link_counts_for_runs()) cost one annotated query for the WHOLE table, never a per-row
# query; the site-local-night counts (night_counts_for_run()) cannot be pushed into SQL, so
# they go behind this TTL cache instead of running per row on every page load.
#
# What this TTL is NOT: it is an eviction bound on PURELY TIME-DRIVEN transitions only -- a
# night elapsing into "unused", a refreshed proposal allocation. It is NOT what makes the
# tally see a record-driven change (a saved linked observation record): that happens
# immediately, with no TTL wait, because the cache key itself carries the newest
# linked-record change stamp -- see build_tally_cache_key().
TALLY_CACHE_TTL_SECONDS = 3600

# The literal cache-key segment for a run with no linked records at all (records_version is
# None) -- normalised to a fixed textual form distinct from any real ISO-format stamp, so the
# key stays stable across processes.
_NO_RECORDS_VERSION_TOKEN = 'none'

# D-11: the three per-run night-state buckets a linked record's classification maps onto.
# OBSERVED/SCHEDULED each get their own set; the three failure-family states share one
# "failed" set. QUEUED and INCONSISTENT are deliberately absent -- they contribute to no
# night set at all.
_NIGHT_CLAIMING_STATES = frozenset(
    {
        DisplayState.OBSERVED,
        DisplayState.SCHEDULED,
        DisplayState.WINDOW_EXPIRED,
        DisplayState.CANCELLED,
        DisplayState.FAILED,
    }
)


def build_tally_cache_key(run_pk: int, records_version: datetime | None) -> str:
    """Build a stable, freshness-sensitive cache key for a run's tally (TALLY-01).

    A key built from the run pk alone would leave a request the projector has just narrowed
    from queued to a placed block reading as queued for up to an hour -- folding the newest
    linked-record change stamp into the key is what makes a narrowing visible on the very
    next page load instead of waiting out the TTL.

    Args:
        run_pk: pk of the CampaignRun.
        records_version: the newest ``modified`` timestamp across the run's linked
            ObservationRecords (see ``link_counts_for_runs()``), or ``None`` for a run with
            no linked records at all.

    Returns:
        str: a stable key; two calls with the same ``(run_pk, records_version)`` pair
            produce identical keys, and any change to ``records_version`` produces a
            different one.
    """
    version_segment = records_version.isoformat() if records_version is not None else _NO_RECORDS_VERSION_TOKEN
    return f'campaign_tally:{run_pk}:{version_segment}'


def link_counts_for_runs(run_pks: list[int]) -> dict[int, dict[str, Any]]:
    """The SQL-expressible half of the tally: group/record counts and the freshness stamp,
    for a whole set of runs in two queries total -- never one query per run (D-08).

    Args:
        run_pks: pks of the CampaignRuns to count.

    Returns:
        dict[int, dict[str, Any]]: ``{run_pk: {'groups': int, 'records': int,
            'records_version': datetime | None}}`` with zeros and ``None`` for a pk that
            appears in neither underlying query (a run with no linked records/groups at
            all). Every pk in ``run_pks`` is guaranteed a key in the result.
    """
    run_pks = list(run_pks)
    result: dict[int, dict[str, Any]] = {pk: {'groups': 0, 'records': 0, 'records_version': None} for pk in run_pks}

    # One aggregate query: distinct linked-record count AND the newest linked-record change
    # stamp, in the same pass -- taking the stamp here is why freshness costs no extra query.
    record_rows = (
        CampaignRunObservation.objects.filter(run_id__in=run_pks)
        .values('run_id')
        .annotate(
            records=Count('observation_record', distinct=True),
            records_version=Max('observation_record__modified'),
        )
    )
    for row in record_rows:
        result[row['run_id']]['records'] = row['records']
        result[row['run_id']]['records_version'] = row['records_version']

    # A second aggregate query: distinct linked-group count per run.
    group_rows = (
        ObservationGroup.objects.filter(observation_records__campaign_run_links__run_id__in=run_pks)
        .values(run_id=F('observation_records__campaign_run_links__run_id'))
        .annotate(groups=Count('pk', distinct=True))
    )
    for row in group_rows:
        result[row['run_id']]['groups'] = row['groups']

    return result


def night_counts_for_run(run: CampaignRun) -> dict[str, int]:
    """The Python half of the tally: per-run night counts by state, de-duplicated per
    site-local observing night (D-11).

    Args:
        run: the CampaignRun being counted.

    Returns:
        dict[str, int]: ``{'nights_observed': int, 'nights_scheduled': int, 'nights_failed':
            int}``. A run with ``site`` unset, or a site with a blank/unusable timezone,
            returns three zeros with a debug log line naming the run pk -- never raises.
    """
    zero_counts = {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0}
    if run.site_id is None or not run.site.timezone:
        logger.debug('night_counts_for_run: run pk=%s has no resolvable site timezone; reporting zero.', run.pk)
        return zero_counts
    try:
        site_zone = ZoneInfo(run.site.timezone)
    except (ZoneInfoNotFoundError, TypeError, ValueError):
        logger.debug('night_counts_for_run: run pk=%s site timezone is unusable; reporting zero.', run.pk)
        return zero_counts

    # Restrict the fetched columns explicitly -- pk/status/facility/scheduled_start/
    # scheduled_end only, exactly what classify_record()/record_time_window() read.
    records = ObservationRecord.objects.filter(
        pk__in=run.observation_links.values_list('observation_record_id', flat=True)
    ).only('pk', 'status', 'facility', 'scheduled_start', 'scheduled_end')

    observed_nights: set = set()
    scheduled_nights: set = set()
    failed_nights: set = set()
    for record in records:
        facility = facility_for(record)
        state = classify_record(record, facility)
        if state not in _NIGHT_CLAIMING_STATES:
            # QUEUED (no block at all) or INCONSISTENT -- contributes to no night set.
            continue
        try:
            start_time, _end_time = record_time_window(record)
        except (KeyError, ValueError):
            logger.debug('night_counts_for_run: record_time_window() raised for pk=%s; skipping.', record.pk)
            continue
        night = observing_night(start_time, site_zone)
        if state == DisplayState.OBSERVED:
            observed_nights.add(night)
        elif state == DisplayState.SCHEDULED:
            scheduled_nights.add(night)
        else:
            failed_nights.add(night)

    return {
        'nights_observed': len(observed_nights),
        'nights_scheduled': len(scheduled_nights),
        'nights_failed': len(failed_nights),
    }


def _combine_tally(counts: dict[str, Any], nights: dict[str, int]) -> dict[str, Any]:
    """Combine one run's SQL-expressible counts and night counts into the eight-key tally
    dict. The three ``unused_*`` keys are left at their not-yet-known defaults here --
    ``tally_for_run()``/``tallies_for_runs()`` fill them in from the unused-night rule.
    """
    return {
        'groups': counts['groups'],
        'records': counts['records'],
        'nights_observed': nights['nights_observed'],
        'nights_scheduled': nights['nights_scheduled'],
        'nights_failed': nights['nights_failed'],
        'nights_unused': None,
        'unused_is_estimate': True,
        'unused_known': False,
    }


def tally_for_run(run: CampaignRun) -> dict[str, Any]:
    """The full eight-key tally for one run: "what has this run actually got".

    Combines ``link_counts_for_runs()`` and ``night_counts_for_run()``. This is a
    single-event entry point (e.g. the calendar pop-up's tag) -- a caller counting MANY runs
    at once (e.g. a table page) must use ``tallies_for_runs()`` instead, never this function
    in a loop (D-08).

    Args:
        run: the CampaignRun being tallied.

    Returns:
        dict[str, Any]: the eight-key tally dict (``groups``, ``records``,
            ``nights_observed``, ``nights_scheduled``, ``nights_failed``, ``nights_unused``,
            ``unused_is_estimate``, ``unused_known``).
    """
    counts = link_counts_for_runs([run.pk])[run.pk]
    nights = night_counts_for_run(run)
    return _combine_tally(counts, nights)


def get_or_compute_tally(run: CampaignRun, records_version: datetime | None = None) -> dict[str, Any]:
    """Cache-or-compute wrapper for one run's tally, mirroring
    ``campaign_gap.get_or_compute_gap()``'s shape.

    Args:
        run: the CampaignRun being tallied.
        records_version: the newest linked-record change stamp to key the cache on. When
            omitted, this is derived with one aggregate query
            (``link_counts_for_runs([run.pk])``) rather than left out of the key entirely --
            a single-event caller (e.g. the calendar pop-up's tag) pays one extra query,
            which is far cheaper than showing an hour-stale tally.

    Returns:
        dict[str, Any]: the cached dict unchanged on a hit; a freshly computed, cached dict
            on a miss.
    """
    if records_version is None:
        records_version = link_counts_for_runs([run.pk])[run.pk]['records_version']
    key = build_tally_cache_key(run.pk, records_version)
    cached = cache.get(key)
    if cached is not None:
        return cached
    tally = tally_for_run(run)
    cache.set(key, tally, timeout=TALLY_CACHE_TTL_SECONDS)
    return tally


def tallies_for_runs(runs) -> dict[int, dict[str, Any]]:
    """The bulk entry point the table uses: one ``link_counts_for_runs()`` call for the
    whole set, then the per-run cached night counts keyed with each run's own
    ``records_version`` from that call.

    Callers must NOT call ``tally_for_run()`` inside a row loop -- that would re-run the
    SQL-expressible aggregate once per row, exactly the per-row query loop D-08 forbids.
    This function exists so a table page pays one aggregate-query pass regardless of how
    many runs it renders.

    Args:
        runs: an iterable of CampaignRun instances.

    Returns:
        dict[int, dict[str, Any]]: ``{run_pk: tally_dict}``.
    """
    runs = list(runs)
    counts_by_pk = link_counts_for_runs([run.pk for run in runs])
    result: dict[int, dict[str, Any]] = {}
    for run in runs:
        counts = counts_by_pk[run.pk]
        key = build_tally_cache_key(run.pk, counts['records_version'])
        cached = cache.get(key)
        if cached is not None:
            result[run.pk] = cached
            continue
        nights = night_counts_for_run(run)
        tally = _combine_tally(counts, nights)
        cache.set(key, tally, timeout=TALLY_CACHE_TTL_SECONDS)
        result[run.pk] = tally
    return result
