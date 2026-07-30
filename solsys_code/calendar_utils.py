"""Shared LCO/SOAR telescope-mapping helpers and CalendarEvent create-or-update helper.

Provides the instrument-extraction chain and telescope-mapping constants extracted from
sync_lco_observation_calendar so all three management commands (sync_lco,
sync_gemini, load_telescope_runs) can share a single implementation, plus the
no-churn CalendarEvent create-or-update function used by all three consumers.
"""

import re
from datetime import timedelta
from typing import Any
from urllib.parse import urljoin

import requests
from django import forms
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request

from solsys_code.observer_codes import HORIZONS_OBSERVER_TO_OBSCODE

# (site, aperture_class) -> 'SITECODE-CLASS' telescope label (TELESCOPE-01/D-03/D-04).
# Verified, real-data-grounded inventory of the 7 real LCO-network sites this
# codebase's installed LCOSettings/SOARSettings actually confirm (tlv/Wise Observatory
# is deliberately excluded -- confirmed absent from both installed get_sites()
# implementations at the 07-01 Task 1 checkpoint; see 07-01-SUMMARY.md Deviations).
# 'coj'/'ogg'/'sor' migrate the 3 pre-existing entries (D-05) -- 'coj'/'ogg' confirmed
# 2m0 (FTS/FTN), 'sor' confirmed 4m0 (SOAR, tom_observations.facilities.soar hardcodes
# 'sitecode': 'sor'). 'elp'/'lsc'/'cpt'/'tfn' confirmed by operator (LCO staff) at the
# 07-01 Task 1 checkpoint -- see 07-01-SUMMARY.md -- as standard 1m-network sites
# hosting both 1m0 and 0m4 telescope classes. 'coj' (Siding Spring) and 'ogg' (Haleakala)
# additionally host 0m4/1m0 (coj) and 0m4 (ogg) -- CONFIRMED (not [ASSUMED]) against the
# authoritative public source https://lco.global/observatory/sites/mpccodes/ (SITEID
# column combined with the first 3 chars of TELID, deduped across all rows); this is
# stronger evidence than the operator-confirmation basis for the original Plan 07-01
# entries above. Closes the SITE_TELESCOPE_MAP completeness gap found in Phase 7 UAT
# Test 1 (07-UAT.md Gaps section), where a real placed record (observation_id=4213127)
# resolved to ('coj', '1m0') but fell back to [UNVERIFIED] for lack of this entry.
SITE_TELESCOPE_MAP = {
    ('coj', '2m0'): 'COJ-2m0',
    ('coj', '1m0'): 'COJ-1m0',
    ('coj', '0m4'): 'COJ-0m4',
    ('ogg', '2m0'): 'OGG-2m0',
    ('ogg', '0m4'): 'OGG-0m4',
    ('sor', '4m0'): 'SOR-4m0',
    ('elp', '1m0'): 'ELP-1m0',
    ('elp', '0m4'): 'ELP-0m4',
    ('lsc', '1m0'): 'LSC-1m0',
    ('lsc', '0m4'): 'LSC-0m4',
    ('cpt', '1m0'): 'CPT-1m0',
    ('cpt', '0m4'): 'CPT-0m4',
    ('tfn', '1m0'): 'TFN-1m0',
    ('tfn', '0m4'): 'TFN-0m4',
}

# SYNC-08/D-10: explicit timeout, single attempt, no retry/backoff loop. This is the
# first explicit HTTP timeout introduced anywhere in solsys_code/ -- there is no
# existing precedent to follow (JPLSBDBQuery.run_query() calls requests.get() with no
# timeout at all, a known anti-pattern, not a convention to mirror here).
_API_TIMEOUT_SECONDS = 10

# EXTRACT-01/D-01: configuration_type values that mark a config as the scientifically
# meaningful one, as opposed to a calibration config (ARC/LAMP_FLAT, SOAR) or an
# NRES-specific config (never in scope). Confirmed against installed tom_observations:
# ocs.py:1025-1030/1213 (flat c_{N}_configuration_type key), lco.py:740-743,757-760,998
# (EXPOSE/REPEAT_EXPOSE/SPECTRUM/REPEAT_SPECTRUM), soar.py:103,118 (SPECTRUM/ARC/
# LAMP_FLAT), blanco.py:177 (STANDARD -- vocabulary adopted now for forward
# compatibility per CONTEXT.md D-01; Blanco facility scope itself stays deferred).
_SCIENCE_CONFIGURATION_TYPES = {'EXPOSE', 'REPEAT_EXPOSE', 'SPECTRUM', 'REPEAT_SPECTRUM', 'STANDARD'}

# D-04: LCO MUSCAT records have no flat c_N_exposure_time, only per-channel
# c_N_ic_M_exposure_time_{suffix} keys (confirmed lco.py:585-596,
# LCOMuscatImagingObservationForm). Detect population by ANY of the 4 channels being
# truthy -- more lenient than the real submission form's all-4-required validation.
_MUSCAT_CHANNEL_SUFFIXES = ('g', 'r', 'i', 'z')


class InstrumentExtractionError(Exception):
    """Raised when extract_instrument finds no usable config (D-06 total extraction failure).

    Caught separately in handle() so a fully-malformed record is routed to the
    dedicated 'extraction_failed' counter, never silently merged into 'skipped'.
    """


def aperture_class_from_telescope_code(telescope_code: str | None) -> str | None:
    """Extract the aperture-class token (D-04 vocabulary) from a 4-char telescope code.

    Args:
        telescope_code: e.g. '0m4b', '1m0a', '2m0a' (from the API response's
            'telescope' key) -- a 3-char aperture-class token plus a trailing
            dome-instance letter suffix. May be None if a malformed/tampered
            API block omitted the 'telescope' key (T-07-03); routes to fallback.

    Returns:
        str | None: '0m4'/'1m0'/'2m0'/'4m0' (strips the trailing dome-instance
            letter), or None if telescope_code is None or doesn't match the
            expected 3-char-class + 1-char-suffix shape (routes the caller to
            fallback per TELESCOPE-03). Never raises.
    """
    if not telescope_code:
        return None
    if len(telescope_code) >= 4 and telescope_code[:3] in {'0m4', '1m0', '2m0', '4m0'}:
        return telescope_code[:3]
    return None


# D-11/D-16: JPL Horizons/SPICE observer notation (`500@<NAIF SPK ID>`) names a space
# observatory that has NO MPC obscode assigned at all -- SPACE's one specific meaning.
# JUICE (site_raw='500@-28') is a real example, but its dev-DB row also carries a blank
# site_raw for some real ingest paths, so a site_raw-only check (tier a, below) cannot
# see it. This constant covers that "telescope_instrument names a known no-obscode space
# observatory" case (tier b). Extension rule, same as HORIZONS_OBSERVER_TO_OBSCODE
# (campaign_utils.py): only add a name after verifying on BOTH sides that the
# observatory has a Horizons code and genuinely no MPC obscode -- never infer from the
# name alone. Swift (obscode 'C52'), HST ('250') and JWST ('274') are deliberately NOT
# members: they resolve like any ground site, and widening SPACE back to "any space
# mission" is exactly the premise D-11 falsified.
NO_OBSCODE_SPACE_OBSERVATORIES: frozenset[str] = frozenset({'juice'})

# T-27-04: linear, non-backtracking patterns over CSV free text (CampaignRun.
# telescope_instrument, max_length=255) -- a bounded numeric group plus optional
# whitespace plus a literal 'm', no nested quantifiers, no alternation spanning the
# whole string.
# Matches an already-canonical 3-char aperture-class token ('0m4'/'1m0'/'2m0'/'4m0'),
# word-bounded so it can't match inside a longer token.
_APERTURE_TOKEN_PATTERN = re.compile(r'\b(0m4|1m0|2m0|4m0)\b')
# Matches a metre-aperture phrase: '0.4m', '1m'/'1.0m', '2m'/'2.0m', '4m'/'4.0m', with
# optional whitespace between the number and 'm'. The digit must be word-bounded on the
# left and 'm' must be word-bounded on the right, so 'MuSCAT4' (m before the digit, not
# after) and '43cm' (no 'm' immediately after the digits) can never match.
_APERTURE_METRE_PATTERN = re.compile(r'\b(\d(?:\.\d)?)\s?m\b')

# Canonicalises a matched metre-phrase capture group to the same 3-char token vocabulary
# aperture_class_from_telescope_code uses.
_METRE_TO_APERTURE_TOKEN = {
    '0.4': '0m4',
    '1': '1m0',
    '1.0': '1m0',
    '2': '2m0',
    '2.0': '2m0',
    '4': '4m0',
    '4.0': '4m0',
}


def derive_telescope_class(site_raw: str | None, telescope_instrument: str | None) -> str:
    """Derive a CampaignRun.telescope_class value from primitives, never a model instance.

    D-20: this helper takes only primitives (not a CampaignRun) so a data migration's
    RunPython step can import and call it without coupling to a model that keeps
    changing through Phases 28-29. Note that this module still imports
    ``tom_calendar.models.CalendarEvent`` at module scope, so "no live models" was never
    literally true of the whole module -- what D-20 buys is that the *signature* stays
    model-free. Both call sites (the Phase 27-04 backfill migration
    and Phase 27-06's import_campaign_csv) gate on "the run has no resolved site" --
    the migration filters site__isnull=True and the importer only calls this when
    resolve_site() returned None. A site-resolved run must never carry a
    telescope_class; that contract lives with the callers, not here.

    Args:
        site_raw: the run's free-text site string (e.g. '500@-28', '250', '' or None).
        telescope_instrument: the run's free-text telescope/instrument string (e.g.
            'LCO 1m', 'JUICE', 'SOAR 4m', '' or None).

    Returns:
        str: '2m0'/'1m0'/'0m4' (D-21: stored lowercase, matching
            aperture_class_from_telescope_code's existing vocabulary, so D-12's subset
            assertion compares directly with no case-folding), 'SPACE' (D-11: a space
            observatory with a Horizons code but no MPC obscode assigned -- deliberately
            kept uppercase since it has no calendar_utils aperture counterpart to
            normalise against), or '' if neither signal fires -- the correct value for a
            genuine site-resolution failure, since site_needs_review already carries
            "unresolved" (D-13). Never raises.
    """
    if site_raw:
        stripped_site = site_raw.strip()
        # WR-07: HORIZONS_OBSERVER_TO_OBSCODE comes from solsys_code.observer_codes (imported
        # at module scope above), which imports no Django models at all. It used to be a
        # function-local import from campaign_utils, with a comment claiming that kept the
        # live CampaignRun model out of a data migration's import graph -- but a
        # function-local import still executes at CALL time, and migration 0011 calls this
        # function once per site-less row, so the very first '500@' row pulled
        # solsys_code.models in mid-migration regardless. The claim is now true.
        if stripped_site.startswith('500@') and stripped_site not in HORIZONS_OBSERVER_TO_OBSCODE:
            # WR-09: `500@<N>` is Horizons *observer* notation for "geocentric observer at
            # body N", and body N need not be a spacecraft -- 500@399 is the Earth's centre,
            # 500@10 the Sun, 500@301 the Moon, and '500@oops' is simply a typo. Only
            # NEGATIVE NAIF IDs are spacecraft, so only those can mean "a space observatory
            # with a Horizons code but no MPC obscode" (D-11). Everything else falls through
            # to the telescope_instrument tier and, failing that, to '' -- which is the
            # correct value for "we could not resolve this", since site_needs_review already
            # carries exactly that meaning (D-13). This mirrors NO_OBSCODE_SPACE_OBSERVATORIES'
            # own extension rule above: never infer a space observatory from string shape alone.
            naif_id = stripped_site[len('500@') :]
            if naif_id.startswith('-') and naif_id[1:].isdigit():
                # D-11: a Horizons observer code with no MPC-obscode alias is exactly
                # what SPACE means.
                return 'SPACE'

    if telescope_instrument:
        lowered = telescope_instrument.lower()
        tokens = re.split(r'[^a-z0-9]+', lowered)
        if NO_OBSCODE_SPACE_OBSERVATORIES.intersection(tokens):
            return 'SPACE'

        token_match = _APERTURE_TOKEN_PATTERN.search(lowered)
        if token_match:
            aperture = token_match.group(1)
        else:
            metre_match = _APERTURE_METRE_PATTERN.search(lowered)
            aperture = _METRE_TO_APERTURE_TOKEN.get(metre_match.group(1)) if metre_match else None

        if aperture == '4m0':
            # D-12: 4m0 (SOAR) is a real, deliberately-excluded value -- it stays in
            # calendar_utils' aperture-class set (SITE_TELESCOPE_MAP has ('sor', '4m0'))
            # but CampaignRun.TelescopeClass's vocabulary is only 2m0/1m0/0m4. Do not
            # "fix" this by adding 4m0 to the model.
            return ''
        if aperture in {'2m0', '1m0', '0m4'}:
            return aperture

    return ''


def derive_telescope(site: str | None, telescope_code: str | None) -> str | None:
    """Map a resolved (site, telescope_code) pair to a verified label via SITE_TELESCOPE_MAP.

    Args:
        site: 3-letter site code from the API response (e.g. 'lsc'). May be
            None if a malformed/tampered API block omitted the 'site' key
            (T-07-03); routes to fallback.
        telescope_code: 4-char telescope code from the API response (e.g.
            '1m0a'). May be None for the same reason; routes to fallback.

    Returns:
        str | None: the verified label (e.g. 'LSC-1m0'), or None if either
            site or telescope_code is None, the (site, class) pair isn't in
            SITE_TELESCOPE_MAP, or telescope_code's aperture class couldn't be
            parsed -- caller falls back to the coarse instrument-class label
            (TELESCOPE-03). Never raises.
    """
    aperture_class = aperture_class_from_telescope_code(telescope_code)
    if aperture_class is None:
        return None
    return SITE_TELESCOPE_MAP.get((site, aperture_class))


def resolve_placement_block(observation_id: str, facility: LCOFacility) -> dict[str, Any] | None:
    """Call the LCO Observation Portal API once to resolve a placed record's block.

    Issues a single, timeout-bounded GET to /api/requests/{observation_id}/observations/
    and selects the same COMPLETED-first-else-PENDING block that
    OCSFacility.get_observation_status() selects for scheduled_start/scheduled_end, so
    telescope resolution and timing always come from the same block (Pitfall 3).

    Args:
        observation_id: the record's LCO observation_id.
        facility: a shared LCOFacility/SOARFacility instance (for portal_url/api_key
            settings and auth header construction).

    Returns:
        dict[str, Any] | None: the matched block dict (with 'site'/'enclosure'/
            'telescope'/'state' keys) on success, or None if the API call failed,
            timed out, or returned no usable COMPLETED/PENDING block. Never raises --
            every failure mode (network error, library auth/validation exception,
            malformed/non-JSON body, missing 'state' key) is caught and converted to
            None so the caller always falls through to the coarse fallback (SYNC-07:
            a per-record failure never aborts the run). The except clause never
            references, stringifies, or logs the caught exception (SYNC-09/D-11) --
            ImproperCredentialsException/forms.ValidationError embed response.content
            directly and must never be logged verbatim.
    """
    try:
        response = make_request(
            'GET',
            urljoin(
                facility.facility_settings.get_setting('portal_url'),
                f'/api/requests/{observation_id}/observations/',
            ),
            headers=facility._portal_headers(),
            timeout=_API_TIMEOUT_SECONDS,
        )
        blocks = response.json()
    except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):
        return None

    if not isinstance(blocks, list):
        return None

    current_block = None
    for block in blocks:
        if block.get('state') == 'COMPLETED':
            current_block = block
            break
        elif block.get('state') == 'PENDING':
            current_block = block
    return current_block


def _has_muscat_exposure_signal(parameters: dict[str, Any], n: int) -> bool:
    """Check whether config c_{n} has a populated MUSCAT per-channel exposure key (D-04).

    Args:
        parameters: the record's parameters dict.
        n: config index (1-5).

    Returns:
        bool: True if any of c_{n}_ic_1_exposure_time_{g,r,i,z} is truthy.
    """
    return any(parameters.get(f'c_{n}_ic_1_exposure_time_{suffix}') for suffix in _MUSCAT_CHANNEL_SUFFIXES)


def _find_science_config(parameters: dict[str, Any]) -> int | None:
    """Scan c_1..c_5 for the first config whose configuration_type is a science type (D-01).

    Args:
        parameters: the record's parameters dict.

    Returns:
        int | None: the config index (1-5) of the first config whose
            c_{n}_configuration_type is in _SCIENCE_CONFIGURATION_TYPES, or None if no
            config has a recognized science configuration_type.
    """
    for n in range(1, 6):
        configuration_type = parameters.get(f'c_{n}_configuration_type')
        if configuration_type in _SCIENCE_CONFIGURATION_TYPES:
            return n
    return None


def _find_exposure_signal_config(parameters: dict[str, Any]) -> int | None:
    """Scan c_1..c_5 for the first config with a populated exposure signal (D-02 fallback).

    Args:
        parameters: the record's parameters dict.

    Returns:
        int | None: the config index (1-5) of the first config with a truthy flat
            c_{n}_exposure_time, or (D-04) a populated MUSCAT per-channel exposure key,
            or None if no config has any exposure signal at all.
    """
    for n in range(1, 6):
        if parameters.get(f'c_{n}_exposure_time') or _has_muscat_exposure_signal(parameters, n):
            return n
    return None


def extract_instrument(parameters: dict[str, Any]) -> str | None:
    """Extract the scientifically meaningful instrument_type from a record's parameters.

    Scans the real c_1..c_5-prefixed multi-configuration shape (D-01..D-06): first by
    configuration_type whitelist (science vs. SOAR calibration/NRES configs), falling
    back to the first config with a populated exposure signal (flat or MUSCAT
    per-channel) if no config has a recognized configuration_type. If no c_N_* config
    exists at all (today's legacy single-config shape, pre-dating the c_N_* fields),
    falls back to the flat 'instrument_type' key itself -- D-02's "original EXTRACT-01
    heuristic" applied to the degenerate single-config case.

    Args:
        parameters: the record's parameters dict.

    Returns:
        str | None: the selected config's c_{n}_instrument_type value (D-03, unchanged
            in format), the flat 'instrument_type' value for the legacy shape, or None
            if neither signal selects any config and no flat key is present (D-06 total
            extraction failure -- the caller routes this to a dedicated counter, never
            the existing 'skipped' counter).
    """
    n = _find_science_config(parameters)
    if n is None:
        n = _find_exposure_signal_config(parameters)
    if n is not None:
        return parameters.get(f'c_{n}_instrument_type')
    return parameters.get('instrument_type')


def coarse_telescope_label(instrument_type: str, facility_name: str) -> str:
    """Derive the coarse aperture-class fallback label from instrument_type and facility.

    LCO instrument type codes are prefixed with the aperture class token (e.g.
    '1M0-SCICAM-SINISTRO', '0M4-SCICAM-SBIG', '2M0-SPECTRAL-AG' -- confirmed
    lco.py:792), mirroring the installed library's own
    `self._get_instruments()[instrument_type]['class']` convention of treating
    instrument type as implying aperture class. SOAR instrument type codes (e.g.
    'SOAR_GHTS_REDCAM') do NOT follow this prefix convention, so they never match
    and previously fell through to the raw, non-coarse string -- closing the
    v1.3 milestone-audit gap (TELESCOPE-03/TELESCOPE-04/SYNC-06): SOAR has exactly
    one site and one aperture class per the single `('sor', '4m0')` entry in
    SITE_TELESCOPE_MAP, so any SOAR record's fallback label is unconditionally
    '4m0', regardless of its raw instrument_type string.

    Args:
        instrument_type: the record's extracted instrument_type (D-04 fallback
            vocabulary source, e.g. '1M0-SCICAM-SINISTRO', 'SOAR_GHTS_REDCAM').
        facility_name: the record's facility string (`record.facility`, e.g.
            'LCO'/'SOAR') -- NOT an LCOFacility/SOARFacility instance.

    Returns:
        str: '4m0' unconditionally if facility_name is SOAR (case-insensitive);
            otherwise '0m4'/'1m0'/'2m0' (case-normalized, D-04 vocabulary) if
            instrument_type has a recognized leading aperture-class prefix, or the
            raw instrument_type string itself if it doesn't -- so the fallback
            label is never empty and this never raises. This only affects the
            coarse label's text -- it never decides whether a record syncs
            (TELESCOPE-03).
    """
    if facility_name.upper() == 'SOAR':
        return '4m0'
    if len(instrument_type) >= 3:
        candidate = instrument_type[:3].lower()
        if candidate in {'0m4', '1m0', '2m0', '4m0'}:
            return candidate
    return instrument_type


def _update_or_unchanged(event: CalendarEvent, fields: dict[str, Any]) -> tuple[CalendarEvent, str]:
    """Apply `fields` to an existing event, saving only if something actually changed.

    Args:
        event: the matched CalendarEvent to reconcile.
        fields: field-value mapping to set on the event.

    Returns:
        tuple[CalendarEvent, str]: (event, 'updated') if any field differed and the
            row was saved, or (event, 'unchanged') if every field already matched and
            no save was issued (no-churn contract).
    """
    changed = [f for f, v in fields.items() if getattr(event, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(event, f, v)
        event.save(update_fields=list(fields.keys()) + ['modified'])
        return event, 'updated'
    return event, 'unchanged'


def insert_or_create_calendar_event(
    lookup: dict[str, Any],
    fields: dict[str, Any],
    *,
    start_time_tolerance: timedelta | None = None,
) -> tuple[CalendarEvent, str]:
    """Create or update a CalendarEvent, or leave it unchanged if no fields differ.

    Implements the no-churn create-or-update contract shared by all three management
    commands (sync_lco_observation_calendar, sync_gemini_observation_calendar,
    load_telescope_runs): create a new CalendarEvent if none exists for the given
    lookup key, update it in place if any fields changed, or leave it untouched if
    nothing changed (SYNC-04 idempotency).

    Args:
        lookup: keyword-argument mapping used as the unique lookup key for
            CalendarEvent.objects.get_or_create (e.g. {'url': url} for LCO/SOAR
            and Gemini sync commands, or {'telescope': ..., 'instrument': ...,
            'start_time': ...} for the load_telescope_runs command).
        fields: field-value mapping of CalendarEvent attributes to set when
            creating or updating. Not merged with `lookup`; the caller is
            responsible for ensuring the combined key+fields set is complete.
        start_time_tolerance: if given, the lookup's `start_time` is matched by
            proximity (an existing event whose start_time is within +/- this
            tolerance of the lookup value counts as the same event) rather than by
            exact datetime equality. This exists for load_telescope_runs, whose
            `start_time` is a computed sun-event time (telescope_runs.sun_event())
            that drifts by a second or two between independent ingests of the same
            (site, night) as astropy's IERS Earth-orientation data is refreshed --
            an exact key would silently create a near-duplicate row on re-ingest.
            A proximity WINDOW is used deliberately rather than rounding/truncating
            start_time to the minute: any fixed bucketing still splits an event that
            drifts across a bucket boundary, whereas a centred window never does. The
            URL-keyed sync callers pass None and keep exact-equality behaviour
            unchanged. Ignored when the lookup has no `start_time` key.

    Returns:
        tuple[CalendarEvent, str]: (event, action) where action is one of
            'created' (new record written), 'updated' (existing record changed
            and saved), or 'unchanged' (existing record matched all fields; no
            save issued). Callers own counter updates and any sidecar writes.
    """
    if start_time_tolerance is not None and 'start_time' in lookup:
        start_time = lookup['start_time']
        key = {k: v for k, v in lookup.items() if k != 'start_time'}
        window = (start_time - start_time_tolerance, start_time + start_time_tolerance)
        # order_by makes the match deterministic; in this domain at most one event
        # per (telescope, instrument) ever falls in the window (nights are ~24h apart,
        # far wider than any plausible IERS-driven drift).
        existing = CalendarEvent.objects.filter(**key, start_time__range=window).order_by('start_time').first()
        if existing is not None:
            # start_time itself is intentionally NOT in `fields`: leaving the stored
            # (first-ingested) value pinned avoids churning `modified` every re-ingest
            # just because the recomputed sun-event time drifted within tolerance.
            return _update_or_unchanged(existing, fields)
        return CalendarEvent.objects.create(**lookup, **fields), 'created'

    event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return event, 'created'
    return _update_or_unchanged(event, fields)
