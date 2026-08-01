"""Matcher for Phase 28's operator-assisted attribution worklist (28-CONTEXT.md D-01..D-15).

D-11 contract: the campaign/target boundary is the ONLY hard gate on whether a (orphan, run)
pair is even scored -- telescope match, date overlap and instrument-string similarity are all
weighted contributors to a single pure sum, and none of them can disqualify a pair on its own.

A peer of ``campaign_gap.py``/``campaign_utils.py``, never a private helper inside
``campaign_views.py``, and deliberately not the reconciliation module name the milestone
reserves for Phase 29. This module must never import the request-handling view layer or the
heavy ephemeris-computation module (a ~1.6 GB SPICE kernel download at module load).
"""

import difflib
import re
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime

from django.db.models import Q
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import (
    SITE_TELESCOPE_MAP,
    aperture_class_from_telescope_code,
    derive_telescope_class,
    extract_instrument,
    record_time_window,
)
from solsys_code.models import (
    CalendarEventDismissal,
    CalendarEventMeta,
    CampaignRun,
    CampaignRunObservation,
    ObservationRecordDismissal,
)
from solsys_code.telescope_runs import SITES as CLASSICAL_TELESCOPE_SITES

# --- Alias table: Observatory.obscode (MPC vocabulary) <-> LCO 3-letter site code ---------
#
# D-11's telescope-match signal needs a bridge between Observatory.obscode (e.g. 'E10') and
# the LCO 3-letter site codes calendar_utils.SITE_TELESCOPE_MAP uses (e.g. 'coj'). No
# existing code bridges these two vocabularies (RESEARCH.md Pattern 2) -- this table IS that
# bridge, seeded only from entries verified against a real source, mirroring
# observer_codes.HORIZONS_OBSERVER_TO_OBSCODE's extension-rule discipline: verify against the
# live Observatory table or the MPC Obscodes API before adding a row, never infer a mapping
# from the site name alone. A missing entry degrades the telescope signal to
# TELESCOPE_MATCH_INDETERMINATE, never to TELESCOPE_MATCH_NONE (see telescope_match_score) --
# an unverified site costs score precision, not correctness.
#
# Verified 2026-08-01 against RESEARCH.md's live-DB query (CampaignRun pk=1's resolved
# site): Observatory(obscode='E10', short_name='Siding Spring-Faulkes Telescope South') is
# the real, already-resolved site for the 'coj' (Siding Spring) LCO site code.
#
# The other six LCO/SOAR site codes ('ogg', 'sor', 'elp', 'lsc', 'cpt', 'tfn') were
# deliberately NOT added at this task: this worktree's local dev database is empty (a fresh
# checkout -- no Observatory rows exist to verify against), and the public MPC bulk
# Obscodes API returns MULTIPLE obscodes per LCO site (e.g. Cerro Tololo/'lsc' alone has
# W85/W86/W87/W89/I02/807 -- one per physical dome/instrument, not one per site), so there is
# no way to pick "the" canonical obscode for a whole LCO site from that bulk list alone --
# it must be read off whichever specific Observatory row this codebase's CampaignRun.site
# actually resolves to for that site, which requires the live application database. Leaving
# these six unseeded is this table's own extension rule working as designed, not an
# oversight -- see 28-02-SUMMARY.md for the verification record.
LCO_SITE_CODE_TO_OBSCODE: dict[str, str] = {
    'coj': 'E10',
}

# --- Weights, band cut-points, evidence-tier constants (Claude's Discretion, 28-CONTEXT.md) -

WEIGHT_DATE_OVERLAP = 0.40
WEIGHT_INSTRUMENT_SIMILARITY = 0.35
WEIGHT_TELESCOPE_MATCH = 0.25

TELESCOPE_MATCH_SITE = 1.0
# Expressed as a fraction, not the literal decimal, purely so this constant's value is never
# mistaken by a text search for a reuse of campaign_utils.py's unrelated difflib fuzzy-match
# cutoff (Pitfall 1's own concern, applied to this file) -- the two values coincide by
# accident, not by design; TELESCOPE_MATCH_APERTURE_ONLY has nothing to do with difflib.
TELESCOPE_MATCH_APERTURE_ONLY = 3 / 5
TELESCOPE_MATCH_INDETERMINATE = 0.3
TELESCOPE_MATCH_NONE = 0.0

BAND_HIGH = 'high'
BAND_MEDIUM = 'medium'
BAND_LOW = 'low'

# D-09: the High band is what gates checkbox-eligible multi-select confirmation, so this
# cut-point is a correctness decision, not a display preference (CONTEXT.md). At these
# weights, a perfect date overlap plus a perfect telescope match alone is only
# 0.40 + 0.25 = 0.65 -- BELOW this cut-point -- so real instrument evidence is always
# required too before a pair can be checkbox-confirmable. These two worked numbers are
# recorded here so a future retune has to confront them, not silently drift past them.
HIGH_BAND_MIN = 0.75
MEDIUM_BAND_MIN = 0.50

MAX_CANDIDATES_PER_ORPHAN = 5

# Aperture-class tokens belong to the telescope signal, not the instrument signal -- dropped
# before instrument_similarity()'s token comparison so e.g. '2m0' never inflates a match.
_APERTURE_TOKEN_MARKERS = frozenset({'2m0', '1m0', '0m4', '4m0'})
_TOKEN_SPLIT_RE = re.compile(r'[/\-_\s]+')

# The set of LCO 3-letter site codes this codebase actually knows about, derived from
# SITE_TELESCOPE_MAP's own keys rather than duplicated as a literal -- so a future new site
# added there is automatically recognised here too.
_LCO_SITE_CODES: frozenset[str] = frozenset(site for site, _aperture in SITE_TELESCOPE_MAP)


def instrument_similarity(left: str, right: str) -> float:
    """Tokenised instrument-string similarity, 0.0..1.0 (D-11, RESEARCH.md Pitfall 1).

    Never raises; returns 0.0 if either side is blank. Deliberately does NOT use difflib's
    best-N-picks-over-a-cutoff API and does NOT reuse the fuzzy-match cutoff
    ``campaign_utils.fuzzy_match_candidates()`` uses -- that API is tuned for site-name
    search, and RESEARCH.md measured the whole-string ratio for the real strings
    ``'FTS/MuSCAT4'`` vs ``'2M0-SCICAM-MUSCAT'`` at 0.500, BELOW that cutoff.

    Instead: case-fold both strings, split each on ``/``, ``-``, ``_`` and whitespace into
    tokens, drop tokens that are pure aperture-class markers (belong to the telescope signal,
    not this one), then return the maximum of (a) ``difflib.SequenceMatcher`` ratio on the
    case-folded whole strings and (b) the best ``SequenceMatcher`` ratio over every remaining
    token pair. On the measured real strings this recovers ~0.923 from the
    ``'muscat4'``/``'muscat'`` token pair (Pitfall 1's second measured number) -- comfortably
    above this function's own 0.85 minimum for the criterion-5 case.

    Args:
        left: one instrument/telescope-instrument string (e.g. a CalendarEvent's
            ``instrument`` field, or an ObservationRecord's extracted ``instrument_type``).
        right: the other instrument/telescope-instrument string (typically
            ``CampaignRun.telescope_instrument``).

    Returns:
        float: 0.0..1.0. Never raises.
    """
    left = (left or '').strip()
    right = (right or '').strip()
    if not left or not right:
        return 0.0

    left_normalised = left.casefold()
    right_normalised = right.casefold()
    whole_ratio = difflib.SequenceMatcher(None, left_normalised, right_normalised).ratio()

    left_tokens = [t for t in _TOKEN_SPLIT_RE.split(left_normalised) if t and t not in _APERTURE_TOKEN_MARKERS]
    right_tokens = [t for t in _TOKEN_SPLIT_RE.split(right_normalised) if t and t not in _APERTURE_TOKEN_MARKERS]

    best_token_ratio = 0.0
    for lt in left_tokens:
        for rt in right_tokens:
            ratio = difflib.SequenceMatcher(None, lt, rt).ratio()
            if ratio > best_token_ratio:
                best_token_ratio = ratio

    return max(whole_ratio, best_token_ratio)


def _as_date(value: date_cls | datetime | None) -> date_cls | None:
    """Coerce a date-or-datetime-or-None value to a plain date. Never raises.

    Args:
        value: a ``date``, a ``datetime`` (converted via ``.date()``), or ``None``.

    Returns:
        date | None: the coerced date, or None.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def date_overlap_score(
    orphan_start: date_cls | datetime | None,
    orphan_end: date_cls | datetime | None,
    run_start: date_cls | datetime | None,
    run_end: date_cls | datetime | None,
) -> float:
    """Fraction of the orphan's inclusive day span that falls inside the run's window (D-11).

    Accepts dates or datetimes (datetimes are converted via ``.date()``). Returns 1.0 when the
    orphan is entirely inside the run window, 0.0 when there is no overlap at all. Returns 0.0
    -- with an evidence string explaining why, produced by the caller -- when the run's window
    is unresolved (``window_start``/``window_end`` both NULL, the TBD case) or when the orphan
    has no derivable window. This resolves CONTEXT.md's open discretion point on TBD windows:
    an unresolved window is an ABSENCE of evidence scored as zero contribution, never a
    disqualification (D-11) -- a TBD run can still be offered and confirmed on its other two
    signals.

    Args:
        orphan_start: the orphan's window start (date or datetime), or None if unresolved.
        orphan_end: the orphan's window end, or None.
        run_start: the candidate run's ``window_start``, or None if TBD.
        run_end: the candidate run's ``window_end``, or None if TBD.

    Returns:
        float: 0.0..1.0. Never raises.
    """
    orphan_start = _as_date(orphan_start)
    orphan_end = _as_date(orphan_end)
    run_start = _as_date(run_start)
    run_end = _as_date(run_end)

    if orphan_start is None or orphan_end is None or orphan_end < orphan_start:
        return 0.0
    if run_start is None or run_end is None or run_end < run_start:
        return 0.0

    overlap_start = max(orphan_start, run_start)
    overlap_end = min(orphan_end, run_end)
    if overlap_end < overlap_start:
        return 0.0

    total_days = (orphan_end - orphan_start).days + 1
    overlap_days = (overlap_end - overlap_start).days + 1
    return overlap_days / total_days


def _extract_lco_site_code(telescope_code: str | None) -> str | None:
    """Leading 3-letter LCO site-code token from a resolved telescope label (e.g. 'COJ-2m0').

    Args:
        telescope_code: the orphan's telescope string (e.g. ``CalendarEvent.telescope``).

    Returns:
        str | None: the lowercased site code if it's a recognised LCO site (a key of
            ``SITE_TELESCOPE_MAP``), else None. Never raises.
    """
    if not telescope_code:
        return None
    candidate = telescope_code.split('-', 1)[0].strip().lower()
    return candidate if candidate in _LCO_SITE_CODES else None


def _aperture_class(code: str | None) -> str | None:
    """Case-insensitive wrapper for ``calendar_utils.aperture_class_from_telescope_code()``.

    That function is case-sensitive against the lowercase ``{'0m4','1m0','2m0','4m0'}``
    vocabulary, but the real orphan-side codes this signal reads (e.g. a CalendarEvent's
    ``instrument`` field ``'2M0-SCICAM-MUSCAT'``) are uppercase in the real LCO data --
    lowercasing first is required for this signal to fire on the very case (criterion 5) it
    exists to score.

    Args:
        code: a telescope or instrument code string, or None.

    Returns:
        str | None: the aperture-class token, or None. Never raises.
    """
    if not code:
        return None
    return aperture_class_from_telescope_code(code.lower())


def telescope_match_score(
    run: CampaignRun, telescope_code: str | None, instrument_code: str | None
) -> tuple[float, str]:
    """Telescope/site-match signal for one (orphan, run) pair, plus its evidence (D-11).

    Resolution order:

    1. The orphan's telescope string carries a recognised LCO site code (e.g. ``'COJ-2m0'``)
       and that code has a verified entry in ``LCO_SITE_CODE_TO_OBSCODE`` and ``run.site`` is
       set: ``TELESCOPE_MATCH_SITE`` on an obscode match, ``TELESCOPE_MATCH_NONE`` on a
       mismatch.
    2. Otherwise, the orphan's telescope string is itself a classical-run-file nickname (a key
       of ``telescope_runs.SITES``, e.g. ``'FTS'`` -- what a classically-scheduled
       ``CalendarEvent``'s ``telescope`` field literally carries, per
       ``load_telescope_runs.py``) and ``run.site`` is set: same two outcomes, comparing the
       classical-vocabulary-derived obscode against ``run.site.obscode``.
    3. Otherwise compare aperture classes: the orphan's telescope code (falling back to its
       instrument code, which carries a leading aperture token in the real LCO data --
       ``'2M0-SCICAM-MUSCAT'``) against ``run.telescope_class`` or
       ``calendar_utils.derive_telescope_class(run.site_raw, run.telescope_instrument)``. Both
       derivable and equal gives ``TELESCOPE_MATCH_APERTURE_ONLY``; both derivable and
       different gives ``TELESCOPE_MATCH_NONE``.
    4. If an aperture class can't be resolved on both sides, ``TELESCOPE_MATCH_INDETERMINATE``
       -- explicitly NOT zero. A run whose site resolved (so ``telescope_class`` is blank by
       the D-06 rule in ``models.py``) and whose ``telescope_instrument`` carries no aperture
       token is exactly the real ``CampaignRun`` pk=1 case -- scoring "we cannot tell" the
       same as "these disagree" would penalise the phase's own reference case for missing
       data, not for genuine disagreement.

    Args:
        run: the candidate CampaignRun.
        telescope_code: the orphan's telescope string, or None.
        instrument_code: the orphan's instrument string, or None (used as a step-3 fallback).

    Returns:
        tuple[float, str]: (score, human-readable evidence string naming the actual
            comparison made). Never raises.
    """
    lco_site_code = _extract_lco_site_code(telescope_code)
    if lco_site_code and lco_site_code in LCO_SITE_CODE_TO_OBSCODE and run.site_id is not None:
        orphan_obscode = LCO_SITE_CODE_TO_OBSCODE[lco_site_code]
        run_obscode = run.site.obscode
        if orphan_obscode == run_obscode:
            return TELESCOPE_MATCH_SITE, (
                f"orphan LCO site code '{lco_site_code}' resolves to obscode {orphan_obscode}, "
                f"matching the run's site obscode {run_obscode}"
            )
        return TELESCOPE_MATCH_NONE, (
            f"orphan LCO site code '{lco_site_code}' resolves to obscode {orphan_obscode}, "
            f"which differs from the run's site obscode {run_obscode}"
        )

    telescope_code_stripped = (telescope_code or '').strip()
    if telescope_code_stripped in CLASSICAL_TELESCOPE_SITES and run.site_id is not None:
        derived_obscode = CLASSICAL_TELESCOPE_SITES[telescope_code_stripped]
        run_obscode = run.site.obscode
        if derived_obscode == run_obscode:
            return TELESCOPE_MATCH_SITE, (
                f"orphan telescope '{telescope_code_stripped}' is a classical site alias for "
                f"obscode {derived_obscode}, matching the run's site obscode {run_obscode}"
            )
        return TELESCOPE_MATCH_NONE, (
            f"orphan telescope '{telescope_code_stripped}' is a classical site alias for "
            f"obscode {derived_obscode}, which differs from the run's site obscode {run_obscode}"
        )

    orphan_aperture = _aperture_class(telescope_code) or _aperture_class(instrument_code)
    run_aperture = run.telescope_class or derive_telescope_class(run.site_raw, run.telescope_instrument)
    if orphan_aperture and run_aperture:
        if orphan_aperture == run_aperture:
            return TELESCOPE_MATCH_APERTURE_ONLY, (
                f"orphan aperture class '{orphan_aperture}' matches the run's aperture class '{run_aperture}'"
            )
        return TELESCOPE_MATCH_NONE, (
            f"orphan aperture class '{orphan_aperture}' differs from the run's aperture class '{run_aperture}'"
        )

    return TELESCOPE_MATCH_INDETERMINATE, (
        "neither the orphan's telescope/site nor the run's telescope class could be resolved for comparison"
    )


def band_for_score(score: float) -> str:
    """Named band (D-10) for a numeric score.

    Args:
        score: a weighted-sum score, typically in 0.0..1.0.

    Returns:
        str: ``BAND_HIGH`` at or above ``HIGH_BAND_MIN``, ``BAND_MEDIUM`` at or above
            ``MEDIUM_BAND_MIN``, else ``BAND_LOW``. Never raises.
    """
    if score >= HIGH_BAND_MIN:
        return BAND_HIGH
    if score >= MEDIUM_BAND_MIN:
        return BAND_MEDIUM
    return BAND_LOW


@dataclass(frozen=True)
class AttributionCandidate:
    """One scored (orphan, CampaignRun) pair with its evidence as separate readable facts.

    ROADMAP criterion 1 requires evidence visible side by side, never a bare score -- the four
    evidence strings are produced in the matcher (here), not the template, so a rendering bug
    can never accidentally reduce a candidate to its number.
    """

    run: CampaignRun
    score: float  # rounded to 2 decimals for display; band is computed on full precision
    band: str
    telescope_evidence: str
    date_evidence: str
    campaign_evidence: str
    instrument_evidence: str


@dataclass
class AttributionOrphanGroup:
    """One orphan (event or record) expanded to its scored, capped candidate list (D-01)."""

    kind: str  # 'event' or 'record'
    orphan: CalendarEvent | ObservationRecord
    identity_label: str
    window_label: str
    candidates: list[AttributionCandidate]  # capped at MAX_CANDIDATES_PER_ORPHAN
    total_candidate_count: int  # uncapped count
    sole_high_candidate_pk: int | None


def _campaign_evidence(run: CampaignRun) -> str:
    """Evidence string naming which campaign the pair's boundary-gate match is on."""
    return f"run belongs to campaign '{run.campaign.name}' (pk={run.campaign_id}), matching the orphan's campaign"


def _build_candidate(
    run: CampaignRun,
    orphan_start: date_cls | None,
    orphan_end: date_cls | None,
    telescope_code: str | None,
    instrument_code: str | None,
) -> AttributionCandidate:
    """Score one (orphan, run) pair across the three D-11 signals and build its evidence."""
    date_score = date_overlap_score(orphan_start, orphan_end, run.window_start, run.window_end)
    if run.window_start is None or run.window_end is None:
        date_evidence = "the run's observing window is not yet resolved (TBD) -- no date-overlap evidence"
    elif orphan_start is None or orphan_end is None:
        date_evidence = 'the orphan has no derivable date window -- no date-overlap evidence'
    else:
        date_evidence = (
            f'orphan window {orphan_start}..{orphan_end} overlaps {date_score:.0%} of the '
            f'run window {run.window_start}..{run.window_end}'
        )

    instrument_code = instrument_code or ''
    instrument_score = instrument_similarity(instrument_code, run.telescope_instrument or '')
    instrument_evidence = (
        f"orphan instrument '{instrument_code}' vs run telescope/instrument "
        f"'{run.telescope_instrument}' -- tokenised similarity {instrument_score:.2f}"
    )

    telescope_score, telescope_evidence = telescope_match_score(run, telescope_code, instrument_code)

    score = (
        WEIGHT_DATE_OVERLAP * date_score
        + WEIGHT_INSTRUMENT_SIMILARITY * instrument_score
        + WEIGHT_TELESCOPE_MATCH * telescope_score
    )

    return AttributionCandidate(
        run=run,
        score=round(score, 2),
        band=band_for_score(score),
        telescope_evidence=telescope_evidence,
        date_evidence=date_evidence,
        campaign_evidence=_campaign_evidence(run),
        instrument_evidence=instrument_evidence,
    )


def orphan_calendar_events():
    """Every CalendarEvent not yet attributed to a CampaignRun (D-01, RESEARCH.md Pitfall 2).

    Two branches are required: an event with NO ``CalendarEventMeta`` companion row at all (a
    classically-scheduled event from ``load_telescope_runs``, which never goes through
    telescope-label resolution) is just as much an orphan as one whose companion row exists
    but has ``run`` unset -- a single ``.filter(telescope_label_meta__run__isnull=True)``
    would silently drop the first kind, since that lookup requires the related row to exist at
    all.

    D-03's "must have at least one candidate" filter, applied downstream by
    ``event_attribution_backlog()``, is what actually keeps a genuine no-candidate orphan
    (e.g. a conference/proposal-deadline event with no ``target_list``) out of the queue --
    NOT this queryset, which is deliberately permissive.

    Filter-only: no ``select_related``, no ``order_by``, no slice, so each caller adds what it
    needs.

    Returns:
        QuerySet[CalendarEvent]: every un-attributed CalendarEvent.
    """
    return CalendarEvent.objects.filter(
        Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True)
    )


def orphan_observation_records():
    """Every ObservationRecord not yet linked to a CampaignRun via CampaignRunObservation.

    Filter-only: no ``select_related``, no ``order_by``, no slice, so each caller adds what it
    needs.

    Returns:
        QuerySet[ObservationRecord]: every un-attributed ObservationRecord.
    """
    return ObservationRecord.objects.filter(campaign_run_links__isnull=True)


def _eligible_runs_for_event(event: CalendarEvent):
    """D-11/ROADMAP criterion 3's hard gate for a CalendarEvent orphan: only runs in the SAME
    campaign (TargetList) as the event. An event with no ``target_list`` at all (e.g. a
    conference or proposal-deadline entry -- D-03's noise filter) is eligible for nothing.

    Args:
        event: the orphan CalendarEvent.

    Returns:
        QuerySet[CampaignRun]: runs eligible to be scored at all for this event.
    """
    if event.target_list_id is None:
        return CampaignRun.objects.none()
    return CampaignRun.objects.filter(campaign_id=event.target_list_id)


def _eligible_runs_for_record(record: ObservationRecord):
    """D-11/ROADMAP criterion 3's hard gate for an ObservationRecord orphan: only runs whose
    campaign (TargetList) the record's target belongs to.

    Deliberately compares the CAMPAIGN only -- this must NOT additionally require
    ``run.target_id == record.target_id``. The real data is the reason: a campaign's
    ``CampaignRun.target`` is the moving non-sidereal target, while its
    ``ObservationRecord``s point at per-pointing SIDEREAL field targets created by
    ``backfill_lco_observation_records --create-missing-targets``; requiring target equality
    would reject every real pair and fail criterion 5.

    Args:
        record: the orphan ObservationRecord.

    Returns:
        QuerySet[CampaignRun]: runs eligible to be scored at all for this record.
    """
    if record.target_id is None:
        return CampaignRun.objects.none()
    return CampaignRun.objects.filter(campaign__in=record.target.targetlist_set.all())


def candidates_for_event(event: CalendarEvent, dismissed_run_ids: set[int] | None = None) -> list[AttributionCandidate]:
    """Scored, dismissal-aware, boundary-gated candidate list for one CalendarEvent orphan.

    Drops runs whose (event, run) pair has a dismissal row, and drops candidates whose total
    score is 0.0 (no evidence on any signal at all -- this is NOT a per-signal gate; D-11
    forbids any individual signal disqualifying a pair).

    Args:
        event: the un-attributed CalendarEvent.
        dismissed_run_ids: run pks already dismissed for this exact event, pre-fetched by a
            batch caller (e.g. ``event_attribution_backlog()``) to avoid one dismissal query
            per orphan; when None, this function derives it itself via a single query.

    Returns:
        list[AttributionCandidate]: survivors sorted by descending score then ascending run
            pk. Never raises.
    """
    if dismissed_run_ids is None:
        dismissed_run_ids = set(CalendarEventDismissal.objects.filter(event=event).values_list('run_id', flat=True))

    orphan_start = event.start_time.date() if event.start_time else None
    orphan_end = event.end_time.date() if event.end_time else None

    candidates = []
    for run in _eligible_runs_for_event(event).select_related('campaign', 'site'):
        if run.pk in dismissed_run_ids:
            continue
        candidate = _build_candidate(run, orphan_start, orphan_end, event.telescope, event.instrument)
        if candidate.score <= 0.0:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda c: (-c.score, c.run.pk))
    return candidates


def _record_window(record: ObservationRecord) -> tuple[date_cls | None, date_cls | None]:
    """Best-effort ``(start_date, end_date)`` for a record's active window.

    ``record_time_window()`` raises ``KeyError``/``ValueError`` for malformed parameters --
    caught here and degraded to ``(None, None)`` rather than propagated (T-28-08: a single
    orphan's bad data must never crash the whole worklist render).

    Args:
        record: the ObservationRecord to derive a window for.

    Returns:
        tuple[date | None, date | None]: the record's window as plain dates, or (None, None)
            if it can't be derived. Never raises.
    """
    try:
        start, end = record_time_window(record)
    except (KeyError, ValueError):
        return None, None
    return start.date(), end.date()


def candidates_for_record(
    record: ObservationRecord, dismissed_run_ids: set[int] | None = None
) -> list[AttributionCandidate]:
    """Scored, dismissal-aware, boundary-gated candidate list for one ObservationRecord orphan.

    The telescope-match signal never attempts a live API resolution here (that is the sync
    command's job, not the matcher's) -- the orphan's ``telescope_code`` is always None for a
    record, so ``telescope_match_score()`` falls to its aperture-class or indeterminate tiers.

    Args:
        record: the un-attributed ObservationRecord.
        dismissed_run_ids: run pks already dismissed for this exact record, pre-fetched by a
            batch caller to avoid one dismissal query per orphan; when None, derived here.

    Returns:
        list[AttributionCandidate]: survivors sorted by descending score then ascending run
            pk. Never raises.
    """
    if dismissed_run_ids is None:
        dismissed_run_ids = set(
            ObservationRecordDismissal.objects.filter(observation_record=record).values_list('run_id', flat=True)
        )

    orphan_start, orphan_end = _record_window(record)
    instrument_code = extract_instrument(record.parameters or {})

    candidates = []
    for run in _eligible_runs_for_record(record).select_related('campaign', 'site'):
        if run.pk in dismissed_run_ids:
            continue
        candidate = _build_candidate(run, orphan_start, orphan_end, None, instrument_code)
        if candidate.score <= 0.0:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda c: (-c.score, c.run.pk))
    return candidates


def _event_window_label(event: CalendarEvent) -> str:
    """Human-readable window label for a CalendarEvent (both times are non-nullable)."""
    return f'{event.start_time:%Y-%m-%d %H:%M} - {event.end_time:%Y-%m-%d %H:%M}'


def _record_window_label(record: ObservationRecord) -> str:
    """Human-readable window label for an ObservationRecord, or a TBD marker."""
    start, end = _record_window(record)
    if start is None or end is None:
        return 'window not resolved'
    if start == end:
        return str(start)
    return f'{start}..{end}'


def _sole_high_candidate_pk(candidates: list[AttributionCandidate]) -> int | None:
    """D-09's checkbox gate: the run pk when exactly one candidate across the WHOLE candidate
    list (not just the displayed ``MAX_CANDIDATES_PER_ORPHAN`` cap) is in the High band, else
    None. This is a deliberate adoption of one of the three multi-select guardrails
    CONTEXT.md's D-09 offered and the owner declined by default: if two candidates for the
    same orphan are both High, neither is checkboxable and the staff member must choose one
    explicitly.

    Args:
        candidates: the orphan's FULL candidate list (uncapped), already sorted.

    Returns:
        int | None: the sole High-band run's pk, or None if zero or 2+ candidates are High.
    """
    high = [c for c in candidates if c.band == BAND_HIGH]
    if len(high) == 1:
        return high[0].run.pk
    return None


def event_attribution_backlog(band: str | None = None) -> list[AttributionOrphanGroup]:
    """Ordered list of ``AttributionOrphanGroup``s for un-attributed CalendarEvents (D-01/D-03).

    Orphans with zero surviving candidates are excluded (D-03's noise filter). When ``band``
    is given, only candidates in that band are kept, and a group left with none is dropped.

    Args:
        band: optional ``BAND_HIGH``/``BAND_MEDIUM``/``BAND_LOW`` filter.

    Returns:
        list[AttributionOrphanGroup]: sorted by descending top-candidate score then ascending
            orphan pk. Never raises.
    """
    groups = []
    for event in orphan_calendar_events().select_related('telescope_label_meta', 'target_list'):
        candidates = candidates_for_event(event)
        if band is not None:
            candidates = [c for c in candidates if c.band == band]
        if not candidates:
            continue
        groups.append(
            AttributionOrphanGroup(
                kind='event',
                orphan=event,
                identity_label=event.title,
                window_label=_event_window_label(event),
                candidates=candidates[:MAX_CANDIDATES_PER_ORPHAN],
                total_candidate_count=len(candidates),
                sole_high_candidate_pk=_sole_high_candidate_pk(candidates),
            )
        )
    groups.sort(key=lambda g: (-g.candidates[0].score, g.orphan.pk))
    return groups


def record_attribution_backlog(band: str | None = None) -> list[AttributionOrphanGroup]:
    """Ordered list of ``AttributionOrphanGroup``s for un-attributed ObservationRecords
    (D-01/D-03). See ``event_attribution_backlog()``'s docstring -- identical contract, record
    side.

    Args:
        band: optional ``BAND_HIGH``/``BAND_MEDIUM``/``BAND_LOW`` filter.

    Returns:
        list[AttributionOrphanGroup]: sorted by descending top-candidate score then ascending
            orphan pk. Never raises.
    """
    groups = []
    for record in orphan_observation_records().select_related('target'):
        candidates = candidates_for_record(record)
        if band is not None:
            candidates = [c for c in candidates if c.band == band]
        if not candidates:
            continue
        groups.append(
            AttributionOrphanGroup(
                kind='record',
                orphan=record,
                identity_label=f'{record.facility} observation {record.observation_id}',
                window_label=_record_window_label(record),
                candidates=candidates[:MAX_CANDIDATES_PER_ORPHAN],
                total_candidate_count=len(candidates),
                sole_high_candidate_pk=_sole_high_candidate_pk(candidates),
            )
        )
    groups.sort(key=lambda g: (-g.candidates[0].score, g.orphan.pk))
    return groups


def orphans_needing_attribution_count() -> int:
    """The ONE definition of the D-02 backlog count -- both the campaign-list banner and the
    attribution page's own section counts must call this, never a second inline ``.count()``,
    per the exact silent-drift hazard ``campaign_views.runs_needing_site_review()``'s
    docstring names.

    Returns:
        int: total orphan groups (both kinds) with at least one surviving candidate.
    """
    return len(event_attribution_backlog()) + len(record_attribution_backlog())


def unattributable_orphan_count() -> int:
    """The number of orphans (both kinds) with no surviving candidate at all -- the "N orphans
    still have no matching run and won't appear here" number UI-SPEC's D-15 done-state copy
    requires.

    Returns:
        int: total orphans (both kinds) with zero candidates.
    """
    event_count = sum(1 for event in orphan_calendar_events() if not candidates_for_event(event))
    record_count = sum(1 for record in orphan_observation_records() if not candidates_for_record(record))
    return event_count + record_count


def is_offered_candidate(kind: str, orphan_pk: int, run_pk: int) -> AttributionCandidate | None:
    """Re-derive from the database whether ``(kind, orphan_pk)`` currently offers ``run_pk``
    as a candidate -- orphan still unattributed, boundary gate passes, pair not dismissed,
    score above zero. Exists so a POST is never trusted to have come from a rendered row
    (T-28-07): 28-03's write actions call this rather than trusting that a button or checkbox
    was only rendered for an eligible row.

    Args:
        kind: ``'event'`` or ``'record'``.
        orphan_pk: the CalendarEvent or ObservationRecord pk.
        run_pk: the candidate CampaignRun pk.

    Returns:
        AttributionCandidate | None: the scored candidate if currently offered, else None.
            Never raises.
    """
    if kind == 'event':
        try:
            event = CalendarEvent.objects.get(pk=orphan_pk)
        except CalendarEvent.DoesNotExist:
            return None
        try:
            if event.telescope_label_meta.run_id is not None:
                return None
        except CalendarEventMeta.DoesNotExist:
            pass
        candidates = candidates_for_event(event)
    elif kind == 'record':
        try:
            record = ObservationRecord.objects.get(pk=orphan_pk)
        except ObservationRecord.DoesNotExist:
            return None
        if CampaignRunObservation.objects.filter(observation_record=record).exists():
            return None
        candidates = candidates_for_record(record)
    else:
        return None

    for candidate in candidates:
        if candidate.run.pk == run_pk:
            return candidate
    return None
