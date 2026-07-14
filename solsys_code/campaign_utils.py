"""Shared helpers for the campaign-coordination CSV bootstrap import.

Provides the D-08 3-tier site resolver, best-effort UT-time-window parsing, the
Observation-Status translation table, and the no-churn CampaignRun create-or-update
helper used by the ``import_campaign_csv`` management command. Mirrors
``calendar_utils.py``'s role for the three CalendarEvent sync commands: every function
here is structured as "never raise for expected messy data; return a usable value plus
an explicit flag" per the ``_derive_telescope_class`` precedent in ``calendar_utils.py``.
"""

import difflib
import logging
import re
from datetime import date, datetime
from datetime import timezone as dt_timezone
from typing import Any

import requests
from django.core.cache import cache
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher

logger = logging.getLogger(__name__)

# D-08 Pitfall 2: Observatory.obscode is CharField(max_length=4). Computed from the
# field itself (not hardcoded) so a future schema change can't silently desync this guard.
_MAX_OBSCODE_LEN = Observatory._meta.get_field('obscode').max_length

# D-02/A2: the MPC obscode list changes far less often than gap-analysis results, so this
# mirrors campaign_gap.py's cache pattern (GAP_CACHE_TTL_SECONDS = 3600) with a much
# longer TTL. Single global pool -> a fixed cache key, no per-request parameters needed.
MPC_CANDIDATE_CACHE_TTL_SECONDS = 86400  # 24h
_MPC_CANDIDATE_CACHE_KEY = 'mpc_obscode_candidates'

# UT Time Range formats confirmed against the real 3I/ATLAS sheet export (RESEARCH.md
# "Real 3I/ATLAS Sheet -- Verified Shape"): 'HH:MM - HH:MM' (tolerant of a ';' typo in
# place of ':'), a tilde-prefixed approximate hour ('~1 am', '~7:00:00 AM'), and a bare
# hour with an explicit 'UTC' marker ('5 UTC', '1 UTC'). Deliberately NOT a permissive
# general-purpose date/time parser (RESEARCH.md Anti-Patterns) -- each pattern requires
# an unambiguous marker (colon/semicolon range, leading '~', or trailing 'UTC') so a
# stray date-range or free-text garbage cell never "succeeds" into a wrong-but-plausible
# time.
_HHMM_RANGE = re.compile(r'(\d{1,2})[:;](\d{2})\s*(am|pm)?\s*-\s*(\d{1,2})[:;](\d{2})\s*(am|pm)?', re.IGNORECASE)
_APPROX_HOUR = re.compile(r'~\s*(\d{1,2})(?::\d{2})?(?::\d{2})?\s*(am|pm)?', re.IGNORECASE)
_BARE_HOUR_UTC = re.compile(r'(\d{1,2})\s*UTC\b', re.IGNORECASE)

# D-12: full-date range, en-dash/em-dash/hyphen(s) or literal "to" separator. Anchored
# start-to-end (never .search()) so it never partially matches inside a longer garbage
# string -- Obs. Date is a structured column, not free prose (unlike the UT-time regexes
# above, which use .search() against genuinely free text). `-{1,2}` (260714-ilz) also
# accepts a double-hyphen separator ('2027-04-20 -- 2027-05-11') -- the ASCII-typable
# stand-in for an en/em dash that a public form submitter is far more likely to type than
# an actual Unicode dash character; the original CSV-sheet-derived single-hyphen/en-dash/
# em-dash/"to" shapes are unaffected.
_DATE_RANGE_FULL = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s*(?:to|-{1,2}|[–—])\s*(\d{4}-\d{2}-\d{2})$',
    re.IGNORECASE,
)

# D-11: compact same-month/rollover range, e.g. '2025-11-02 -25' or '2025-11-28 -05'.
# Second group is the day-of-month only (1-2 digits); rollover logic lives in the caller,
# not the regex, matching this module's existing convention (_HHMM_RANGE also keeps
# am/pm interpretation out of the pattern itself).
_DATE_RANGE_COMPACT = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\s*-\s*(\d{1,2})$')


def _to_24h(hour: int, meridiem: str | None) -> int:
    """Apply an optional am/pm marker to an hour parsed from a 12-hour-ish UT time cell.

    CR-01: the sheet's UT Time Range cells sometimes carry an explicit am/pm marker
    (e.g. ``'~7:00:00 PM'``); the marker must be applied rather than silently discarded,
    or a PM time parses as if it were AM (12 hours wrong, with no error and no flag).

    Args:
        hour: the hour digits as parsed (1-12 for am/pm-marked input, 0-23 otherwise).
        meridiem: ``'am'``/``'pm'`` (any case) if a marker was present, else ``None``.

    Returns:
        int: the 24-hour-clock hour.
    """
    if not meridiem:
        return hour
    meridiem = meridiem.lower()
    if meridiem == 'am':
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


# Observation Status -> RunStatus translation (Pitfall 3): case-insensitive substring
# match, most-specific first, conservative REQUESTED default for anything unrecognized
# (a non-key field per D-05 -- an imprecise default must never block the row).
_STATUS_MAP = [
    ('cancel', CampaignRun.RunStatus.CANCELLED),
    ('not awarded', CampaignRun.RunStatus.NOT_AWARDED),
    ('weather', CampaignRun.RunStatus.WEATHER_TECH_FAILURE),
    ('technical', CampaignRun.RunStatus.WEATHER_TECH_FAILURE),
    ('publish', CampaignRun.RunStatus.PUBLISHED),
    ('reduc', CampaignRun.RunStatus.REDUCED),
    ('complet', CampaignRun.RunStatus.OBSERVED),
    ('observ', CampaignRun.RunStatus.OBSERVED),
    ('upcoming', CampaignRun.RunStatus.PLANNED),
    ('planned', CampaignRun.RunStatus.PLANNED),
]

# WR-08: a bare negation ("Not observed", no other keyword) contains the 'observ'
# substring but means the opposite -- map_observation_status skips the ('observ', ...)
# entry when this matches and no earlier, more-specific keyword already matched.
_NOT_OBSERVED_RE = re.compile(r'\bnot\s+observ', re.IGNORECASE)


def resolve_site(site_code_raw: str, *, create_placeholder: bool = True) -> tuple[Observatory | None, bool]:
    """Resolve a raw Site Code string to an Observatory (D-08 3-tier resolution).

    Tier 1: match against an existing ``Observatory`` record. Tier 2: query the MPC
    Obscodes API via ``MPCObscodeFetcher`` and create an ``Observatory`` row if found.
    Tier 3: create a placeholder ``Observatory`` row, flagged for manual review -- unless
    ``create_placeholder`` is False, in which case tier 3 is skipped entirely and the code
    is flagged for manual review with no Observatory row created. A blank or oversized
    (> ``Observatory.obscode``'s max length) code never reaches tier 1/2/3 at all -- it is
    flagged immediately with no Observatory row created, so a code that can't possibly be
    a real MPC obscode (e.g. JWST's 8-character spacecraft-style ``'500@-170'``) is never
    truncated or fabricated (D-09/Pitfall 2).

    Args:
        site_code_raw: the CSV row's raw ``Site Code`` cell value (may be blank, ``None``,
            or contain leading/trailing whitespace).
        create_placeholder: whether tier 3 may fabricate a placeholder ``Observatory`` row
            when tiers 1 and 2 both miss. Defaults to ``True`` so the existing CSV-import
            caller (already-vetted sheet data) is unaffected. Pass ``False`` for
            unvetted/public free-text input (e.g. the campaign approval endpoint) so an
            unresolvable site is flagged for manual review instead of fabricating a fake
            Observatory row.

    Returns:
        tuple[Observatory | None, bool]: ``(observatory_or_none, needs_review)``. Never
            raises for expected messy-data cases.
    """
    code = (site_code_raw or '').strip()
    if not code:
        return None, True  # no code at all -- flag, no placeholder possible

    if len(code) > _MAX_OBSCODE_LEN:
        # e.g. JWST's '500@-170' -- can't fit Observatory.obscode; don't fabricate a
        # truncated/wrong site. Flag for manual review instead (Pitfall 2).
        return None, True

    # Tier 1: existing Observatory record.
    try:
        return Observatory.objects.get(obscode=code), False
    except Observatory.DoesNotExist:
        pass

    # Tier 2: MPC Obscodes API (same call CreateObservatory.form_valid makes). The
    # `errors` return value is intentionally unused beyond triggering the
    # MissingDataException path below -- MPCObscodeFetcher.query() already logs the API
    # error internally; don't double-log.
    fetcher = MPCObscodeFetcher()
    try:
        fetcher.query(code, timeout=10)
    except requests.exceptions.RequestException:
        # WR-01: the MPC API call hung/failed at the network layer (timeout, DNS,
        # connection reset, ...) -- import_campaign_csv calls resolve_site once per CSV
        # row in a synchronous loop, so an unhandled network exception here would crash
        # the whole batch import. Treat a network failure like an MPC miss and fall
        # through to tier 3 rather than losing the rest of the import.
        pass
    else:
        try:
            return fetcher.to_observatory(), False
        except MissingDataException:
            pass  # no such obscode at MPC either -- fall through to tier 3
        except (KeyError, ValueError, TypeError):
            # WR-04: to_observatory() reads several dict keys with no `.get()`/default,
            # so a live MPC API response that's 200 OK but missing/malformed an expected
            # key (KeyError) or has a value that fails a `float(...)`/similar conversion
            # (ValueError/TypeError) would otherwise crash the whole import. Treat a
            # malformed-but-"ok" response like an MPC miss and fall through to tier 3.
            pass
        except IntegrityError:
            # Race: another row in this same import (or a concurrent process) already
            # created it -- re-fetch instead of losing the row.
            try:
                return Observatory.objects.get(obscode=code), False
            except Observatory.DoesNotExist:
                # WR-02: Observatory.name is also unique=True, so an IntegrityError here
                # isn't necessarily an obscode race -- it could be a name collision with
                # a *different* obscode, in which case no Observatory exists for `code`
                # and the re-fetch above would otherwise raise uncaught. Fall through to
                # tier 3 instead of letting DoesNotExist propagate out of resolve_site.
                pass

    if not create_placeholder:
        # Public free-text submissions (unlike the already-vetted CSV import) should not
        # auto-create a placeholder Observatory on approve -- flag for manual review only.
        return None, True

    # Tier 3: placeholder, flagged for review (D-09 -- flag, don't silently guess).
    try:
        placeholder = Observatory.objects.create(
            obscode=code,
            name=f'NEEDS REVIEW: {code}',
            short_name=code,
        )
    except IntegrityError:
        # WR-03: race protection matching tier 2's -- another row in this same import
        # (or a concurrent process) already created an Observatory (placeholder or real)
        # for this obscode. Re-fetch instead of crashing the import.
        return Observatory.objects.get(obscode=code), True
    return placeholder, True


def _flatten_mpc_candidates(obscode_dict: dict) -> dict[str, str]:
    """Flatten a bulk MPC obscodes dict into a fuzzy-matchable ``{string: obscode}`` map.

    Per RESEARCH.md Pattern 3 / Open Question 2: each record contributes its obscode,
    ``name_utf8``, ``short_name``, and ``old_names`` (as one whole string, not split) as
    candidate display strings. First-seen wins on collision (rare -- distinct sites
    practically never share a name); blank/falsy strings are skipped.

    Args:
        obscode_dict: dict keyed by 3-char obscode, as returned by
            ``MPCObscodeFetcher.query_all()``.

    Returns:
        dict[str, str]: candidate display string -> obscode. Never raises for expected
            messy data (a missing/None field is treated as an empty string and skipped).
    """
    mapping: dict[str, str] = {}
    for code, rec in obscode_dict.items():
        for candidate in (code, rec.get('name_utf8') or '', rec.get('short_name') or '', rec.get('old_names') or ''):
            if candidate and candidate not in mapping:
                mapping[candidate] = code
    return mapping


def _local_observatory_candidates() -> dict[str, str]:
    """Build a ``{string: obscode}`` map from every local ``Observatory`` row.

    Candidate strings mirror ``_flatten_mpc_candidates()``'s field selection (obscode,
    name, short_name, old_names) so the local and MPC-sourced pools merge uniformly.
    First-seen wins on collision.

    Returns:
        dict[str, str]: candidate display string -> obscode. Never raises.
    """
    mapping: dict[str, str] = {}
    for obs in Observatory.objects.all():
        for candidate in (obs.obscode, obs.name or '', obs.short_name or '', obs.old_names or ''):
            if candidate and candidate not in mapping:
                mapping[candidate] = obs.obscode
    return mapping


def build_site_candidates() -> dict[str, str]:
    """Build (and cache) the merged local + MPC fuzzy-match candidate pool (D-01/D-02).

    On a cache hit under ``'mpc_obscode_candidates'``, returns the cached pool without
    re-fetching. On a miss, bulk-fetches the full MPC obscode list via
    ``MPCObscodeFetcher.query_all()``, flattens it (``_flatten_mpc_candidates()``), merges
    in every local ``Observatory`` row's candidate strings, caches the merged result for
    ``MPC_CANDIDATE_CACHE_TTL_SECONDS``, and returns it.

    Mirrors ``resolve_site()``'s "never raise for expected messy data; return a usable
    value plus an explicit flag" discipline (here: no explicit flag, since a network
    failure degrades gracefully to a still-usable local-only pool rather than needing a
    caller-visible error state): a bulk-fetch network/parse failure is caught narrowly and
    falls back to the local-only ``Observatory`` pool, never raising into
    ``ApprovalQueueView``'s page render (RESEARCH.md Environment Availability fallback).

    Returns:
        dict[str, str]: candidate display string -> obscode, merged from the cached MPC
            bulk list (or local-only on MPC failure) and every local ``Observatory`` row.
            Never raises.
    """
    cached = cache.get(_MPC_CANDIDATE_CACHE_KEY)
    if cached is not None:
        return cached

    mpc_candidates: dict[str, str] = {}
    try:
        obscode_dict = MPCObscodeFetcher().query_all()
        mpc_candidates = _flatten_mpc_candidates(obscode_dict)
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
        # WR-style fallback (mirrors resolve_site()'s tier-2 network-failure handling):
        # an MPC outage must never break the approval-queue page render -- fall through to
        # a local-only pool below. WR-01: AttributeError is included because
        # _flatten_mpc_candidates() calls .items()/.get() assuming obscode_dict (and each
        # record) is a dict -- a bulk endpoint response that's drifted to a non-dict shape
        # (e.g. a list or None) raises AttributeError, not one of the other caught types.
        logger.debug('MPC bulk obscode fetch failed; falling back to local-only candidate pool.', exc_info=True)

    merged = dict(mpc_candidates)
    # Local Observatory rows merge in last so an already-vetted local record's display
    # string always wins a first-seen collision over the raw MPC bulk data.
    for candidate, obscode in _local_observatory_candidates().items():
        merged.setdefault(candidate, obscode)

    cache.set(_MPC_CANDIDATE_CACHE_KEY, merged, timeout=MPC_CANDIDATE_CACHE_TTL_SECONDS)
    return merged


def fuzzy_match_candidates(site_raw: str, candidate_pool: dict[str, str]) -> list[tuple[str, str]]:
    """Fuzzy-match a raw submitted site string against a candidate pool (D-01/A3).

    Wraps ``difflib.get_close_matches`` (``n=5, cutoff=0.6`` -- difflib's own documented
    default cutoff, per RESEARCH.md Assumption A3) and resolves each matched display
    string back to its obscode via ``candidate_pool``.

    Args:
        site_raw: the raw submitted/typed site text to match against the pool. Blank
            input returns an empty list without invoking difflib.
        candidate_pool: a ``{candidate_display_string: obscode}`` mapping, typically from
            ``build_site_candidates()``.

    Returns:
        list[tuple[str, str]]: ranked ``(display_string, obscode)`` pairs, best match
            first. Empty list when nothing clears the cutoff (e.g. an acronym/nickname
            like ``'DCT'`` that difflib cannot bridge -- Pitfall 2). Never raises.
    """
    text = (site_raw or '').strip()
    if not text:
        return []
    matches = difflib.get_close_matches(text, candidate_pool.keys(), n=5, cutoff=0.6)
    return [(match, candidate_pool[match]) for match in matches]


def parse_obs_window(
    obs_date_raw: str, ut_range_raw: str
) -> tuple[date | None, date | None, str, bool, datetime | None, datetime | None, bool]:
    """Best-effort parse of the sheet's Obs. Date + UT Time Range columns (D-11/D-12/D-13).

    Tries, in order, an exact ``YYYY-MM-DD`` date, a full-date range (``' to '``/en-dash/
    em-dash/hyphen separated), and a compact same-month/rollover range (``'2025-11-02
    -25'``). Anything that doesn't match any of those shapes -- including blank text, a
    ``'YYYY-MM-?'`` marker, and free-text prose -- is a TBD row: ``window_start`` and
    ``window_end`` are both ``None``, ``original_obs_date_raw`` carries the verbatim raw
    text, and ``window_needs_review`` is ``True``. ``parse_obs_window()`` never raises for
    any ``obs_date_raw`` input (D-13) -- this is a contract change from the previous
    exact-date-or-raise behavior.

    ``ut_range_raw`` is only parsed for the single-night case (``window_start ==
    window_end``, both not ``None``); a range or TBD row skips UT parsing entirely (A1 --
    ``ut_start``/``ut_end``/``ut_needs_review`` are unused by every real caller, and a
    multi-night window has no single night to anchor a UT time to).

    Args:
        obs_date_raw: the CSV row's raw ``Obs. Date`` cell value -- an exact date, a
            range (full-date or compact same-month/rollover), or unparseable free text.
        ut_range_raw: the CSV row's raw ``UT Time Range`` cell value -- highly variable
            free text in the real sheet (HH:MM ranges, semicolon typos, approximate
            hours, bare-hour-plus-UTC shorthand, blank, or unparseable prose).

    Returns:
        tuple[date | None, date | None, str, bool, datetime | None, datetime | None, bool]:
            ``(window_start, window_end, original_obs_date_raw, window_needs_review,
            ut_start, ut_end, ut_needs_review)``. ``ut_start``/``ut_end`` are tz-aware UTC
            when present. Never raises.
    """
    text = (obs_date_raw or '').strip()

    window_start: date | None = None
    window_end: date | None = None

    try:
        window_start = window_end = datetime.strptime(text, '%Y-%m-%d').date()
    except ValueError:
        match = _DATE_RANGE_FULL.match(text)
        if match:
            start_s, end_s = match.groups()
            try:
                window_start = datetime.strptime(start_s, '%Y-%m-%d').date()
                window_end = datetime.strptime(end_s, '%Y-%m-%d').date()
                if window_end < window_start:
                    # Reversed range (operands swapped, e.g. a source-sheet typo) --
                    # treat like any other unparseable shape rather than silently
                    # accepting a window that claims zero dates (WR-01).
                    window_start = window_end = None
            except ValueError:
                window_start = window_end = None
        else:
            match = _DATE_RANGE_COMPACT.match(text)
            if match:
                year_s, month_s, day1_s, day2_s = match.groups()
                year, month, day1, day2 = int(year_s), int(month_s), int(day1_s), int(day2_s)
                try:
                    window_start = date(year, month, day1)
                    if day2 < day1:
                        # D-11 rollover: second number is smaller than the first
                        # day-of-month -- roll into the next month (and next year for a
                        # Dec -> Jan crossing).
                        window_end = date(year + 1, 1, day2) if month == 12 else date(year, month + 1, day2)
                    else:
                        window_end = date(year, month, day2)
                except ValueError:
                    # e.g. day2=35 for a 28/29/30/31-day month, or day1 itself invalid --
                    # stdlib date() already validates this; treat like any other
                    # unparseable shape and fall through to TBD.
                    window_start = window_end = None

    if window_start is None:
        # No shape matched (blank, 'YYYY-MM-?', or genuine garbage) -- D-03/D-06/D-13 TBD
        # state. No dedicated 'YYYY-MM-?' regex is needed: it falls through here naturally.
        return None, None, text, True, None, None, False

    if window_start != window_end:
        # Range row -- no single night to anchor a UT time to (A1); skip UT parsing.
        return window_start, window_end, '', False, None, None, False

    # Single-night case (window_start == window_end): UT-Time-Range parsing unchanged.
    obs_date = window_start

    match = _HHMM_RANGE.search(ut_range_raw or '')
    if match:
        h1_raw, m1, meridiem1, h2_raw, m2, meridiem2 = match.groups()
        h1 = _to_24h(int(h1_raw), meridiem1)
        h2 = _to_24h(int(h2_raw), meridiem2)
        m1, m2 = int(m1), int(m2)
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h1, m1, tzinfo=dt_timezone.utc)
        end = datetime(obs_date.year, obs_date.month, obs_date.day, h2, m2, tzinfo=dt_timezone.utc)
        return window_start, window_end, '', False, start, end, False

    match = _APPROX_HOUR.search(ut_range_raw or '')
    if match:
        h = _to_24h(int(match.group(1)), match.group(2))
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h, 0, tzinfo=dt_timezone.utc)
        return window_start, window_end, '', False, start, None, False

    match = _BARE_HOUR_UTC.search(ut_range_raw or '')
    if match:
        h = int(match.group(1))
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h, 0, tzinfo=dt_timezone.utc)
        return window_start, window_end, '', False, start, None, False

    # Fallback: obs_date is valid but UT range isn't parseable at all (blank, garbage
    # text, or a misplaced date-range) -- use midnight UTC (Pitfall 1), never skip here.
    # Flagged via ut_needs_review=True (CR-02) since this fallback always resolves to the
    # same timestamp for a given obs_date, so two distinct rows sharing telescope+date
    # both falling back here would otherwise collide on the natural key.
    start = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, tzinfo=dt_timezone.utc)
    return window_start, window_end, '', False, start, None, True


def map_observation_status(raw: str) -> str:
    """Translate the sheet's free-text Observation Status into a RunStatus value (Pitfall 3).

    Case-insensitive substring match against a small, ordered translation table. Any
    unrecognized string (including blank) falls back to the conservative
    ``RunStatus.REQUESTED`` default rather than raising or guessing at a more specific
    status -- ``run_status`` is a non-key field (D-05), so an imprecise default must
    never block the row. A bare negation like ``'Not observed'`` (WR-08) is deliberately
    *not* classified as ``OBSERVED`` even though it contains that substring -- unless a
    more specific keyword also co-occurs (e.g. ``'Not observed -- weather'`` still maps to
    ``WEATHER_TECH_FAILURE`` via the earlier, more-specific ``'weather'`` entry).

    Args:
        raw: the CSV row's raw ``Observation Status`` cell value.

    Returns:
        str: one of ``CampaignRun.RunStatus``'s values. Never raises.
    """
    normalized = (raw or '').strip().lower()
    for needle, status in _STATUS_MAP:
        if needle == 'observ' and _NOT_OBSERVED_RE.search(normalized):
            # WR-08: no more-specific keyword matched by this point (they're all earlier
            # in the table), so this is a bare negation -- skip straight to REQUESTED
            # rather than mis-classifying it as OBSERVED.
            continue
        if needle in normalized:
            return status
    return CampaignRun.RunStatus.REQUESTED


def insert_or_create_campaign_run(lookup: dict[str, Any], fields: dict[str, Any]) -> tuple[CampaignRun, str]:
    """Create or update a CampaignRun, or leave it unchanged if no fields differ.

    Mirrors ``calendar_utils.insert_or_create_calendar_event()``'s no-churn
    create-or-update contract: create a new ``CampaignRun`` if none exists for the given
    lookup key (D-04's natural key), update it in place if any fields changed, or leave
    it untouched if nothing changed (idempotent re-run, no spurious writes).
    ``CampaignRun`` has no ``modified``/auto-now field, so an update issues
    ``save(update_fields=list(fields))`` only -- unlike ``CalendarEvent``, there is no
    timestamp field to include.

    Args:
        lookup: keyword-argument mapping used as the unique lookup key for
            ``CampaignRun.objects.get_or_create`` (D-04). Two shapes: resolved-window rows
            key on (campaign, telescope_instrument, window_start, window_end); TBD rows key
            on (campaign, telescope_instrument, contact_person, window_start__isnull=True) --
            the `window_start__isnull=True` guard is required so a TBD row never collides
            with a resolved row sharing the same campaign/telescope/contact_person.
        fields: field-value mapping of ``CampaignRun`` attributes to set when creating or
            updating. Not merged with `lookup`; the caller is responsible for ensuring
            the combined key+fields set is complete.

    Returns:
        tuple[CampaignRun, str]: ``(run, action)`` where action is one of ``'created'``
            (new record written), ``'updated'`` (existing record changed and saved), or
            ``'unchanged'`` (existing record matched all fields; no save issued).
    """
    run, created = CampaignRun.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return run, 'created'
    changed = [f for f, v in fields.items() if getattr(run, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(run, f, v)
        run.save(update_fields=list(fields.keys()))
        return run, 'updated'
    return run, 'unchanged'
