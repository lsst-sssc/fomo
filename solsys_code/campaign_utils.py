"""Shared helpers for the campaign-coordination CSV bootstrap import.

Provides the D-08 3-tier site resolver, best-effort UT-time-window parsing, the
Observation-Status translation table, and the no-churn CampaignRun create-or-update
helper used by the ``import_campaign_csv`` management command. Mirrors
``calendar_utils.py``'s role for the three CalendarEvent sync commands: every function
here is structured as "never raise for expected messy data; return a usable value plus
an explicit flag" per the ``_derive_telescope_class`` precedent in ``calendar_utils.py``.
"""

import re
from datetime import date, datetime
from datetime import timezone as dt_timezone
from typing import Any

import requests
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher

# D-08 Pitfall 2: Observatory.obscode is CharField(max_length=4). Computed from the
# field itself (not hardcoded) so a future schema change can't silently desync this guard.
_MAX_OBSCODE_LEN = Observatory._meta.get_field('obscode').max_length

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


def resolve_site(site_code_raw: str) -> tuple[Observatory | None, bool]:
    """Resolve a raw Site Code string to an Observatory (D-08 3-tier resolution).

    Tier 1: match against an existing ``Observatory`` record. Tier 2: query the MPC
    Obscodes API via ``MPCObscodeFetcher`` and create an ``Observatory`` row if found.
    Tier 3: create a placeholder ``Observatory`` row, flagged for manual review. A blank
    or oversized (> ``Observatory.obscode``'s max length) code never reaches tier 1/2/3 at
    all -- it is flagged immediately with no Observatory row created, so a code that can't
    possibly be a real MPC obscode (e.g. JWST's 8-character spacecraft-style
    ``'500@-170'``) is never truncated or fabricated (D-09/Pitfall 2).

    Args:
        site_code_raw: the CSV row's raw ``Site Code`` cell value (may be blank, ``None``,
            or contain leading/trailing whitespace).

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

    # Tier 3: placeholder, flagged for review (D-09 -- flag, don't silently guess).
    placeholder = Observatory.objects.create(
        obscode=code,
        name=f'NEEDS REVIEW: {code}',
        short_name=code,
    )
    return placeholder, True


def parse_obs_window(obs_date_raw: str, ut_range_raw: str) -> tuple[date, datetime, datetime | None, bool]:
    """Best-effort parse of the sheet's Obs. Date + UT Time Range columns (Pitfall 1).

    ``obs_date_raw`` must parse as ``%Y-%m-%d`` or the row is a true natural-key failure
    (D-05) and this raises. ``ut_range_raw`` is always best-effort: an unparseable or
    blank time range never raises -- it falls back to ``obs_date`` at 00:00:00 UTC, so a
    malformed non-key field never skips the row (D-05/D-09 discipline extended to time
    parsing).

    Args:
        obs_date_raw: the CSV row's raw ``Obs. Date`` cell value, expected ``YYYY-MM-DD``.
        ut_range_raw: the CSV row's raw ``UT Time Range`` cell value -- highly variable
            free text in the real sheet (HH:MM ranges, semicolon typos, approximate
            hours, bare-hour-plus-UTC shorthand, blank, or unparseable prose).

    Returns:
        tuple[date, datetime, datetime | None, bool]: ``(obs_date, ut_start, ut_end,
            ut_needs_review)``, both datetimes tz-aware UTC. ``ut_end`` is ``None``
            whenever only a start time (or no time at all) could be determined.
            ``ut_needs_review`` mirrors ``resolve_site``'s ``needs_review`` flag: ``True``
            when ``ut_start`` is the midnight-UTC fallback rather than an actual parsed
            time (CR-02) -- callers can use this to detect and disambiguate natural-key
            collisions between distinct rows that both fell back to midnight.

    Raises:
        ValueError: if ``obs_date_raw`` itself can't be parsed to a date (true
            natural-key failure per D-05) -- a bad/missing ``ut_range_raw`` never raises.
    """
    obs_date = datetime.strptime((obs_date_raw or '').strip(), '%Y-%m-%d').date()  # ValueError propagates

    match = _HHMM_RANGE.search(ut_range_raw or '')
    if match:
        h1_raw, m1, meridiem1, h2_raw, m2, meridiem2 = match.groups()
        h1 = _to_24h(int(h1_raw), meridiem1)
        h2 = _to_24h(int(h2_raw), meridiem2)
        m1, m2 = int(m1), int(m2)
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h1, m1, tzinfo=dt_timezone.utc)
        end = datetime(obs_date.year, obs_date.month, obs_date.day, h2, m2, tzinfo=dt_timezone.utc)
        return obs_date, start, end, False

    match = _APPROX_HOUR.search(ut_range_raw or '')
    if match:
        h = _to_24h(int(match.group(1)), match.group(2))
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h, 0, tzinfo=dt_timezone.utc)
        return obs_date, start, None, False

    match = _BARE_HOUR_UTC.search(ut_range_raw or '')
    if match:
        h = int(match.group(1))
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h, 0, tzinfo=dt_timezone.utc)
        return obs_date, start, None, False

    # Fallback: obs_date is valid but UT range isn't parseable at all (blank, garbage
    # text, or a misplaced date-range) -- use midnight UTC (Pitfall 1), never skip here.
    # Flagged via ut_needs_review=True (CR-02) since this fallback always resolves to the
    # same timestamp for a given obs_date, so two distinct rows sharing telescope+date
    # both falling back here would otherwise collide on the natural key.
    start = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, tzinfo=dt_timezone.utc)
    return obs_date, start, None, True


def map_observation_status(raw: str) -> str:
    """Translate the sheet's free-text Observation Status into a RunStatus value (Pitfall 3).

    Case-insensitive substring match against a small, ordered translation table. Any
    unrecognized string (including blank) falls back to the conservative
    ``RunStatus.REQUESTED`` default rather than raising or guessing at a more specific
    status -- ``run_status`` is a non-key field (D-05), so an imprecise default must
    never block the row.

    Args:
        raw: the CSV row's raw ``Observation Status`` cell value.

    Returns:
        str: one of ``CampaignRun.RunStatus``'s values. Never raises.
    """
    normalized = (raw or '').strip().lower()
    for needle, status in _STATUS_MAP:
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
            ``CampaignRun.objects.get_or_create`` (D-04: campaign, telescope_instrument,
            ut_start).
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
