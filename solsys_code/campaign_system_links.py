"""Exact-identity system links from the discovery sweep (ALLOC-06).

A *system link* is a ``CampaignRunObservation`` row written by an ingest management command
rather than by a staff member: ``confirmed_by`` is ``None`` (the provenance) and
``confirmed_at`` is stamped when the sweep wrote it. The row's existence is still the
confirmation (Phase 27/28), so the ``CampaignRunObservation`` post_save receiver
(``allocation_projector.receiver_on_run_observation_save``, 35 D-11) retires the record's
allocation night, or attributes the record's own event to a container run, on the same sweep.
This module never calls ``reconcile_run()`` itself.

The match is exact identity only -- proposal code, target (or, failing that, campaign
membership when the proposal is unique in the record's campaigns) and window. It never
consults ``campaign_attribution``'s similarity scorer, its weights or its bands.

Only the ingest management commands call :func:`attempt_system_link`: never a view and never
the observation projector's post_save receiver. A request path must not be able to write a
machine attribution.

Proposal comparison: the record's ``parameters['proposal']`` is trimmed of surrounding
whitespace and then compared exactly and case-sensitively to ``CampaignRun.proposal_code``,
the same exact comparison ``WatchedProposal`` (which strips on save) and
``proposal_allocation`` (exact set membership) use. A blank trimmed proposal never matches:
live data carries many runs with ``proposal_code=''``.

Window containment is inclusive and compared on UTC dates, so a block observed on a run's
last local night west of UTC can fall on ``window_end + 1`` and stay in the attribution queue
(the safe direction).

Dry-run parity: a dry run judges an existing record on its stored window (a schedule change
the real pass would apply is not yet visible) and a not-yet-created record on its request
window or embedded block; a would-be-new target (no pk) is in no campaign and carries no run,
so it is never asked.

This module depends only on the models and the attribution window helper -- it must never
import ``solsys_code.views`` or ``solsys_code.ephem_utils`` (or any module that imports
them) at module scope. Their ~1.6 GB SPICE-kernel download side effect (CLAUDE.md "Heavy
import side effect") would otherwise be paid by every process that imports this module.
"""

import logging
from typing import NamedTuple

from tom_observations.models import ObservationRecord

from solsys_code.campaign_attribution import _record_window
from solsys_code.campaign_utils import create_system_link
from solsys_code.models import CampaignRun, CampaignRunObservation, ObservationRecordDismissal

logger = logging.getLogger(__name__)

# The two match bases, in the wording the per-link stdout line and the runbook use (D-07).
BASIS_TARGET = 'proposal + target + window'
BASIS_CAMPAIGN = 'proposal unique within campaign + window'

# attempt_system_link() outcomes.
OUTCOME_LINKED = 'linked'
OUTCOME_WOULD_LINK = 'would_link'
OUTCOME_NO_MATCH = 'no_match'
OUTCOME_SKIPPED = 'skipped'


class SystemLinkMatch(NamedTuple):
    """The one run an ObservationRecord exactly matches, and why."""

    run: CampaignRun
    basis: str


def _proposal_code(record: ObservationRecord) -> str:
    """Return the record's trimmed ``parameters['proposal']``, or '' when it has none.

    Args:
        record: the ObservationRecord to read.

    Returns:
        str: the proposal code with surrounding whitespace removed; '' when ``parameters`` is
            not a dict or ``'proposal'`` is missing or not a string.
    """
    parameters = record.parameters
    if not isinstance(parameters, dict):
        return ''
    proposal = parameters.get('proposal')
    if not isinstance(proposal, str):
        return ''
    return proposal.strip()


def _windowed_approved_runs(proposal: str):
    """APPROVED runs of one proposal that carry both window dates (D-13, D-14).

    The database check constraint keeps ``window_start`` and ``window_end`` null together, so
    filtering both is belt and braces. ``run_status`` is deliberately ignored.

    Args:
        proposal: the exact proposal code.

    Returns:
        QuerySet[CampaignRun]: the candidate runs, before any window or target filter.
    """
    return CampaignRun.objects.filter(
        approval_status=CampaignRun.ApprovalStatus.APPROVED,
        proposal_code=proposal,
        window_start__isnull=False,
        window_end__isnull=False,
    )


def _campaign_fallback_run(record: ObservationRecord, proposal: str, start, end) -> CampaignRun | None:
    """D-11 step 2: the campaign fallback, for a record whose own target carries no run.

    A field-pointing record's target is a per-pointing target, never the run's moving target
    (see ``campaign_attribution._eligible_runs_for_record``), so step 1 cannot match it. The
    record then links to the run of its campaigns (its target's ``TargetList``s) when, and
    only when:

    * no APPROVED, windowed same-proposal run anywhere carries the record's target -- this
      includes a run whose window does not contain the record, which makes the record's own
      target ambiguous rather than absent (roadmap SC2; stricter reading of the amended D-11);
    * the record's campaigns hold exactly one APPROVED, windowed same-proposal run, counted
      BEFORE any window filter (RESEARCH Pitfall 1), so a second run with a different window
      still makes the proposal ambiguous; and
    * that run's window contains the record's window (inclusive, UTC dates).

    Args:
        record: the ObservationRecord being matched; may be unsaved.
        proposal: the record's trimmed proposal code, already known to be non-blank.
        start: the record's window start date.
        end: the record's window end date.

    Returns:
        CampaignRun | None: the unique containing campaign run, or None.
    """
    if record.target_id is None or record.target is None or record.target.pk is None:
        return None
    if _windowed_approved_runs(proposal).filter(target_id=record.target_id).exists():
        return None
    campaign_runs = list(_windowed_approved_runs(proposal).filter(campaign__in=record.target.targetlist_set.all()))
    if len(campaign_runs) != 1:
        return None
    run = campaign_runs[0]
    if run.window_start <= start and run.window_end >= end:
        return run
    return None


def find_exact_run(record: ObservationRecord) -> SystemLinkMatch | None:
    """Return the one run the record exactly matches, or None (ALLOC-06, D-11..D-15).

    A pure read; it also accepts an UNSAVED record (``record.pk is None``) so a dry run can
    ask about a would-be record. In order:

    1. Guards: a blank trimmed proposal, a record that already has any
       ``CampaignRunObservation`` row (human outranks machine), or a window that cannot be
       read (or runs backwards) all give None.
    2. Candidates: APPROVED, windowed runs of the same proposal whose window contains the
       record's window (inclusive, UTC dates).
    3. Step 1 (D-11/D-15): exactly one candidate carrying the record's target wins with
       :data:`BASIS_TARGET`; two or more give None with no fall-through.
    4. Step 2 (D-11 as amended at plan time, D-15), only when no candidate carries the target:
       see :func:`_campaign_fallback_run`. It wins with :data:`BASIS_CAMPAIGN`.
    5. Dismissal veto (applied to the winner after the pick, never as a pre-filter, so a
       record is never linked to a different run instead): a staff dismissal of the
       (record, winner) pair gives None.

    Args:
        record: the ObservationRecord to match; may be unsaved.

    Returns:
        SystemLinkMatch | None: the winning run and the basis, or None when there is no
            unambiguous exact match.
    """
    proposal = _proposal_code(record)
    if not proposal:
        return None
    if record.pk is not None and CampaignRunObservation.objects.filter(observation_record_id=record.pk).exists():
        return None
    start, end = _record_window(record)
    if start is None or end is None or end < start:
        return None

    candidates = list(_windowed_approved_runs(proposal).filter(window_start__lte=start, window_end__gte=end))

    winner: CampaignRun | None = None
    basis = BASIS_TARGET
    by_target = [run for run in candidates if run.target_id is not None and run.target_id == record.target_id]
    if len(by_target) == 1:
        winner = by_target[0]
    elif len(by_target) > 1:
        return None
    else:
        winner = _campaign_fallback_run(record, proposal, start, end)
        basis = BASIS_CAMPAIGN
    if winner is None:
        return None

    if (
        record.pk is not None
        and ObservationRecordDismissal.objects.filter(observation_record_id=record.pk, run_id=winner.pk).exists()
    ):
        return None
    return SystemLinkMatch(winner, basis)


def attempt_system_link(record: ObservationRecord, *, dry_run: bool, stdout, stderr) -> str:
    """Offer one record to the exact-identity matcher and, if it matches, link it.

    The single entry point the ingest commands call. It never raises for a bad record or a
    failed write (D-08): one malformed record must never stop a sweep, and only the exception
    CLASS NAME is ever written (the D-17 convention -- never ``str(exc)``).

    Every line written ends with a newline: the unattended runner captures stdout and stderr
    in a plain ``io.StringIO`` and splits on lines, while Django's ``OutputWrapper`` does not
    double an existing newline.

    Args:
        record: the ObservationRecord to offer; unsaved only under ``dry_run``.
        dry_run: when True, report the would-be link and write nothing at all.
        stdout: a file-like sink for the per-link line.
        stderr: a file-like sink for failure lines.

    Returns:
        str: one of :data:`OUTCOME_LINKED`, :data:`OUTCOME_WOULD_LINK`,
            :data:`OUTCOME_NO_MATCH`, :data:`OUTCOME_SKIPPED`.
    """
    try:
        match = find_exact_run(record)
    except Exception as exc:  # noqa: BLE001 -- D-08, RESEARCH Pitfall 3: _record_window() does not catch TypeError
        stderr.write(f'System link check failed for observation_id={record.observation_id!r}: {type(exc).__name__}\n')
        return OUTCOME_SKIPPED
    if match is None:
        return OUTCOME_NO_MATCH

    if dry_run:
        stdout.write(
            f'Would system-link ObservationRecord observation_id={record.observation_id!r} '
            f'to CampaignRun #{match.run.pk} ({match.basis}).\n'
        )
        return OUTCOME_WOULD_LINK

    try:
        created = create_system_link(record, match.run)
    except Exception as exc:  # noqa: BLE001 -- D-08: a failed link write is counted, never fatal to the sweep
        stderr.write(
            f'Could not system-link observation_id={record.observation_id!r} '
            f'to CampaignRun #{match.run.pk}: {type(exc).__name__}\n'
        )
        return OUTCOME_SKIPPED
    if not created:
        # Someone linked it first; not a system link.
        return OUTCOME_NO_MATCH
    stdout.write(
        f'System-linked ObservationRecord observation_id={record.observation_id!r} '
        f'to CampaignRun #{match.run.pk} ({match.basis}).\n'
    )
    return OUTCOME_LINKED
