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

from django.core.cache import cache
from django.utils import timezone

from solsys_code.models import CampaignRun
from solsys_code.telescope_runs import sun_event

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


def claimed_dates(campaign, target, site) -> tuple[set[date], list, list]:
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

    For each counted run, every date in the inclusive range
    [window_start, window_end] is claimed (a single-night run has window_start ==
    window_end, so exactly one date is claimed). A run with window_start is None (TBD)
    cannot be attributed to any date -- it is collected into a separate "undated" list,
    never added to the claimed set. This does not distinguish ground vs. space-mission
    runs (ASSET-02 asset-awareness is explicitly Phase 20's job) -- every counted run
    claims its full window regardless of site type.

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.

    WR-05: unlike ``observable_dates(site, start, end)``, this function takes no date-range
    parameters -- it returns every approved, non-excluded ``CampaignRun`` for the campaign/
    site combination regardless of any requested window. ``_compute_gap()`` only ever
    evaluates the range-bounded ``gap = obs - claimed`` against the range-bounded ``obs``
    set, so ``gap_dates`` is correct -- but the returned ``claimed_dates``/``undated_runs``/
    ``unattributed_runs`` are campaign/site-wide, NOT scoped to ``[start, end]``, even though
    the cached result they end up in (``build_gap_cache_key()``) is keyed by a date range.
    Do not assume a range-keyed cache entry's ``claimed_dates`` is itself range-bounded.

    Returns:
        tuple[set[date], list, list]: (claimed_dates, undated_runs, unattributed_runs).
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

    claimed: set[date] = set()
    undated_runs: list[CampaignRun] = []
    for run in qs:
        if run.window_start is None or run.window_end is None:
            # TBD -- can't be attributed to any date (unchanged bucketing rule). WR-02:
            # also catches the DB-CheckConstraint-should-prevent-but-defend-anyway case of a
            # mismatched pair (one set, one NULL) so this never raises a TypeError on read.
            undated_runs.append(run)
            continue
        n_days = (run.window_end - run.window_start).days + 1
        for i in range(n_days):
            claimed.add(run.window_start + timedelta(days=i))

    return claimed, undated_runs, unattributed_runs


def _compute_gap(campaign, target, site, start: date, end: date) -> dict:
    """Compute the coverage-gap result dict (no caching).

    Args:
        campaign: the campaign TargetList.
        target: the selected Target, or None.
        site: the selected Observatory.
        start: inclusive start date.
        end: inclusive end date.

    Returns:
        dict: gap_dates, claimed_dates, observable_dates (each a sorted list of dates),
            undated_runs, unattributed_runs (lists of CampaignRun), and
            unknown_date_count (number of dates in range whose sun_event() call raised,
            i.e. dates in range that are neither observable nor known-unavailable).
    """
    obs = observable_dates(site, start, end)
    claimed, undated_runs, unattributed_runs = claimed_dates(campaign, target, site)
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
        'unknown_date_count': unknown_date_count,
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
