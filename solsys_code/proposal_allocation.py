"""Fetch and store LCO Observation Portal proposal time allocations (Phase 37 D-06/D-07).

This module is the ONLY writer of ``ProposalTimeAllocation`` -- everything else (the public
tally columns, the calendar's unused-night decoration) reads the stored rows only, so a
public page never triggers a credentialed portal call at request time. Nothing in this
module reads or writes ``CampaignRun.run_status`` (TALLY-03's guard).

``HOURS_PER_NIGHT`` is a deliberate, fixed rule of thumb (D-06) -- 10 hours of awarded time
per observing night, with precedent in the NOIRLab/LCO proposal process -- NOT a
measurement. Every "unused nights" figure this module derives must be presented as an
estimate, and callers must render ``None`` as "not yet fetched", never as zero.

Mirrors ``campaign_gap.py``'s import discipline (``solsys_code/campaign_gap.py:1-14``): this
module must never import ``solsys_code.views`` or ``solsys_code.ephem_utils`` (or any module
that imports either) at module scope -- that module's ~1.6 GB SPICE-kernel download side
effect (CLAUDE.md "Heavy import side effect") must never be paid by a process that only
needs to fetch or read a proposal's time allocation.
"""

import math
from typing import Any
from urllib.parse import urljoin

import requests
from django import forms
from django.db.models import F, Sum
from django.utils import timezone
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request

from solsys_code.models import CampaignRun, ProposalTimeAllocation, WatchedProposal

# D-06: a deliberate, fixed rule of thumb with precedent in the NOIRLab/LCO proposal
# process, not a measurement -- the reason every figure derived from it is labelled an
# estimate. A future deferred per-telescope-class hours-per-night table would replace this
# single constant without touching the model or the fetch.
HOURS_PER_NIGHT = 10.0

# Task 1 checkpoint decision (a-store-all-types, "sum only standard time"): every allocation
# type the portal returns is stored (see _ALLOCATION_TYPES below), but only these types feed
# the unused-hours estimate. Real proposal UTX2026A-002 was confirmed (live portal check) to
# hold nonzero Time-Critical (`tc`) hours -- deliberately stored, never summed here, so the
# summation rule can change later (e.g. adding 'tc') without a second portal round trip.
ESTIMATE_ALLOCATION_TYPES = ('std',)

# The portal's own time-allocation-type prefixes -- confirmed against a live
# `GET /api/proposals/` call (Task 1 checkpoint): std, rr, tc, AND an undocumented
# `realtime_allocation`/`realtime_time_used` pair. All four are stored; only
# ESTIMATE_ALLOCATION_TYPES above are summed into the estimate.
_ALLOCATION_TYPES = ('std', 'rr', 'tc', 'realtime')

_API_TIMEOUT_SECONDS = 10


class PortalUnavailable(Exception):  # noqa: N818 -- exact symbol name given by 37-02-PLAN.md Task 3
    """Raised by :func:`fetch_proposal_allocations` on any portal-call failure.

    Carries ONLY the caught exception's class name (``str(exc)``) -- never the caught
    exception's own message or the response body, both of which can embed the portal
    request's content and, for ``ImproperCredentialsException``/``forms.ValidationError``,
    the API key (mirrors ``calendar_utils.resolve_placement_block()``'s SYNC-09/D-11
    discipline).
    """


def proposal_codes_to_fetch() -> list[str]:
    """The union of every active ``WatchedProposal`` code and every distinct non-blank
    ``CampaignRun.proposal_code``, as a sorted list of unique codes.

    Uses ``.values_list(..., flat=True).distinct()`` on both sides so no ``CampaignRun``
    row -- and in particular no contact field -- is ever fetched into this process (T-37-06).

    Returns:
        list[str]: sorted, de-duplicated proposal codes.
    """
    watched = set(WatchedProposal.objects.filter(is_active=True).values_list('proposal_code', flat=True))
    run_codes = set(CampaignRun.objects.exclude(proposal_code='').values_list('proposal_code', flat=True).distinct())
    return sorted(watched | run_codes)


def fetch_proposal_allocations(proposal_code: str, facility: LCOFacility) -> list[dict[str, Any]]:
    """One timeout-bounded GET to the LCO Observation Portal's single-proposal endpoint.

    Copies ``calendar_utils.resolve_placement_block()``'s call shape and except-clause
    discipline verbatim: the same exception set is caught, and the caught exception is
    never referenced, stringified, or logged, because ``ImproperCredentialsException``/
    ``forms.ValidationError`` embed response content and the request carried the API key
    (T-37-04).

    Args:
        proposal_code: the proposal code to fetch.
        facility: a shared ``LCOFacility``/``SOARFacility`` instance (for
            ``portal_url``/``api_key`` settings and auth header construction).

    Returns:
        list[dict[str, Any]]: the parsed ``timeallocation_set`` list.

    Raises:
        PortalUnavailable: on any network error, auth failure, or non-JSON/malformed body --
            carrying only the caught exception's class name.
    """
    try:
        response = make_request(
            'GET',
            urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/proposals/{proposal_code}/'),
            headers=facility._portal_headers(),
            timeout=_API_TIMEOUT_SECONDS,
        )
        parsed = response.json()
    except (
        requests.exceptions.RequestException,
        ImproperCredentialsException,
        forms.ValidationError,
        ValueError,
    ) as exc:
        raise PortalUnavailable(type(exc).__name__) from None

    if not isinstance(parsed, dict):
        raise PortalUnavailable('ValueError')

    timeallocation_set = parsed.get('timeallocation_set')
    if not isinstance(timeallocation_set, list):
        raise PortalUnavailable('ValueError')

    return timeallocation_set


def store_proposal_allocations(proposal_code: str, rows: list[dict[str, Any]]) -> int:
    """Create or update the matching ``ProposalTimeAllocation`` row for every allocation
    type each ``timeallocation_set`` entry carries.

    One row per (semester, instrument_type, allocation_type) triple, per entry. A time type
    the response does not carry (missing BOTH its ``<type>_allocation`` and
    ``<type>_time_used`` keys) is treated as absent, not zero -- no row is written for it.

    Args:
        proposal_code: the proposal code every written row is keyed on.
        rows: the ``timeallocation_set`` list :func:`fetch_proposal_allocations` returned.

    Returns:
        int: the number of rows created or updated.
    """
    written = 0
    fetched_at = timezone.now()
    for entry in rows:
        semester = entry.get('semester') or ''
        instrument_type = entry.get('instrument_type') or ''
        for allocation_type in _ALLOCATION_TYPES:
            allocation_key = f'{allocation_type}_allocation'
            used_key = f'{allocation_type}_time_used'
            if allocation_key not in entry or used_key not in entry:
                continue
            ProposalTimeAllocation.objects.update_or_create(
                proposal_code=proposal_code,
                semester=semester,
                instrument_type=instrument_type,
                allocation_type=allocation_type,
                defaults={
                    'allocated_hours': entry[allocation_key] or 0.0,
                    'used_hours': entry[used_key] or 0.0,
                    'fetched_at': fetched_at,
                },
            )
            written += 1
    return written


def unused_hours_for(proposal_code: str) -> float | None:
    """Summed ``allocated_hours - used_hours`` over the stored rows in
    ``ESTIMATE_ALLOCATION_TYPES``, floored at zero.

    Args:
        proposal_code: the proposal code to sum.

    Returns:
        float | None: the summed unused hours (never negative), or ``None`` when the
            proposal has no stored rows at all -- callers must render this as "not yet
            fetched", never as zero.
    """
    rows = ProposalTimeAllocation.objects.filter(
        proposal_code=proposal_code, allocation_type__in=ESTIMATE_ALLOCATION_TYPES
    )
    if not rows.exists():
        return None
    total = rows.aggregate(total=Sum(F('allocated_hours') - F('used_hours')))['total']
    return max(total or 0.0, 0.0)


def estimated_unused_nights(proposal_code: str) -> int | None:
    """``unused_hours_for()`` divided by ``HOURS_PER_NIGHT``, rounded to a whole number of
    nights (ties away from zero -- e.g. 25 unused standard hours reports 3 nights).

    This is a deliberate ESTIMATE (D-06), never a measurement. ``None`` means "not yet
    fetched" -- callers must render that as unknown, never as zero.

    Args:
        proposal_code: the proposal code to estimate.

    Returns:
        int | None: the estimated unused night count, or ``None`` when
            :func:`unused_hours_for` returns ``None``.
    """
    unused_hours = unused_hours_for(proposal_code)
    if unused_hours is None:
        return None
    # unused_hours is always >= 0 (floored above), so floor(x + 0.5) is round-half-up.
    return math.floor(unused_hours / HOURS_PER_NIGHT + 0.5)


def refresh_all(facility: LCOFacility) -> tuple[int, int, int, str | None]:
    """Fetch and store every code :func:`proposal_codes_to_fetch` names, isolating a
    per-proposal failure so one bad code does not abandon the rest.

    Args:
        facility: a shared ``LCOFacility`` instance (for ``portal_url``/``api_key``
            settings and auth header construction), reused across every fetch.

    Returns:
        tuple[int, int, int, str | None]: (proposals attempted, rows written, proposals
            that failed, the first failing proposal's exception class name or ``None``).
    """
    attempted = 0
    rows_written = 0
    failed = 0
    first_exception: str | None = None
    for proposal_code in proposal_codes_to_fetch():
        attempted += 1
        try:
            rows = fetch_proposal_allocations(proposal_code, facility)
            rows_written += store_proposal_allocations(proposal_code, rows)
        except PortalUnavailable as exc:
            failed += 1
            if first_exception is None:
                first_exception = str(exc)
    return attempted, rows_written, failed, first_exception
