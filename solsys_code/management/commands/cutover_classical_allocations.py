"""One-time cutover (D-15 step 3, D-17, D-18, Phase 35 Task 2): converts the legacy
blank-url classical ``CalendarEvent`` rows -- written by ``load_telescope_runs`` before
this phase's allocation rewrite -- into ``CampaignRun``s plus ``ALLOC:``-keyed events, so
the calendar has exactly one writer per night.

Why this command exists: before Phase 35, a classical schedule line produced a bare
``CalendarEvent`` with ``url=''``, no owning ``CampaignRun`` and no ``CalendarEventMeta``
companion row. The rewritten ``load_telescope_runs`` (plan 35-05) now writes a
``CampaignRun`` and lets the allocation projector draw its per-night calendar, but every
event a PRE-cutover import already wrote is stranded in the old blank-url shape. This
command recovers each stranded event's own schedule line from its own description,
re-derives the run it would have produced under the new pipeline, and re-keys the event
onto that run's ``ALLOC:{run_pk}:{night}`` slot in place -- so a converted night is
indistinguishable from one a fresh import would have produced (D-17), and the calendar
ends up with exactly the same per-night entries it would have had if the rewritten command
had always been the one that created them.

It needs no schedule file: every fact it uses -- the telescope, instrument, status, date
range, sub-night window and optional proposal -- is already recoverable from the ``Source
line:`` the pre-cutover command itself wrote as part of each event's own description
(re-parsed with the identical ``telescope_runs.parse_run_line()`` a fresh import would use).

It deliberately does NOT touch any event outside the blank-url classical set: a
``RUN:``-keyed event is the reconciler's and the sweep's business (D-16's other half, Task
1), and a facility-url-keyed observation event is the observation projector's. It never
infers a run's source from a site or a telescope name -- every converted run is
unconditionally ``source=CampaignRun.Source.CLASSICAL_FILE``, the provenance a schedule
line actually has.

It never removes a ``CalendarEvent`` row from the database, on any path, including a
failure path: what it cannot explain, it reports and leaves completely untouched (D-18).
The reasons an event or its group can be left unexplained are: no ``Source line:`` marker
at all in the description; a ``Source line:`` that does not parse or whose telescope does
not resolve to a known site; a resolved site with no ``timezone`` set; a group whose
member events disagree on their campaign (``target_list``); an event already attributed to
a DIFFERENT run; a second event claiming a night this run has already claimed, or a night
whose ``ALLOC:`` url another ``CalendarEvent`` already holds -- which is what an import of
the same schedule file running before the cutover leaves behind (WR-11, 35-REVIEW.md; only
the first claimant of a night is re-keyed, the rest are reported untouched, because
``CalendarEvent.url`` carries no unique constraint for the database to enforce it); an
event whose own independently-derived observing night falls outside the window its own
schedule line implies, corrected by fixing the event's stored start time or the schedule
line's date range so the two agree (``window_mismatch``, NF-02, 35-REVIEW.md); a second
group whose ``Source line:`` resolves to the SAME run identity key as an earlier group --
``_source_identifier()`` deliberately ignores the parsed status word, so two lines differing
only in status (e.g. an allocation line and a cancelled line for the same telescope,
instrument and window) collide -- reported (never silently merged into the earlier group's
run) under its own reason. The database check also catches a claimant left by an earlier
cutover invocation or by ``load_telescope_runs``, not only this process's own in-memory
bookkeeping -- but only WHEN that claimant's own stored ``Source line:`` is recoverable and
matches; a claimant whose marker is absent or differs is reported under
``duplicate_identity`` instead of converted, because ``observation_details`` is writable
from the Django admin, from ``import_campaign_csv.py`` and from the campaign submission
form, so this command cannot prove such a row came from the line in hand
(``duplicate_identity``, NF-14/NF-19/CR-01, 35-REVIEW.md). The remedy this command can
actually carry out: when the two Source lines differ, edit the affected events' description
``Source line:`` text in the Django admin so it carries a different bracketed
``[proposal]`` token from the earlier group's; when no marker is recoverable at all, restore
or correct the claimant's ``observation_details`` ``Source line:`` text in the Django admin
so it matches, or disambiguate the two lines -- then re-run in either case, since this
command reads no schedule file, so editing one changes nothing it will ever see (NF-25,
35-REVIEW.md); and any
other exception, recorded with its own type name. Every reason is printed with the event's
primary key and title so an operator can find and correct the row in the admin. When EVERY
event in a group is attributed elsewhere, no ``CampaignRun`` is created or updated for that
group at all (WR-07, 35-REVIEW.md) -- the command writes nothing for a group it can convert
nothing in.

Re-run gotcha: a claimant whose stored ``Source line:`` marker MATCHES the group's is
find-and-updated, so ``source``, ``approval_status``, ``run_status``, ``target``,
``campaign``, ``site``/``site_raw``, ``window_start``/``window_end``, the two sub-night
fields and ``observation_details`` are all re-applied from the schedule line on every
invocation. A post-import staff edit to any of them does not survive the next cutover run.
This is the same file-authoritative behaviour ``load_telescope_runs`` has on every
re-import and ``import_campaign_csv`` has on every re-import (documented at length in the
runbook's "Re-import gotcha" section), and it is expected behaviour for this class of
command rather than a defect -- the command CAN prove this row came from this line, which
is exactly the distinction the identity guard above establishes. This closes a loop worth
naming explicitly: the no-marker remedy above tells an operator to restore the marker,
which converts a refused claimant into a matching one that will then be re-applied on the
next run -- so restore the marker deliberately, not reflexively.

Because nothing is ever removed, a non-zero exit here is purely operator-facing: it tells
a human which rows to look at and re-run once they are fixed. Nothing in this repository
gates a pipeline step, a CI step, or another command's exit code on this command's exit
code (35-RESEARCH.md Open Question 2, resolved) -- a caller that drives this command
through ``call_command()`` rather than a shell (the 35-07 demo notebook does exactly that)
catches the resulting exception and has only its message to show, which is why the message
itself carries the full count and reason breakdown, not just the preceding stdout lines.

It is one-time (a converted event's ``url`` is no longer blank, so it drops out of this
command's own candidate set) but safe to re-run: re-running it after a fully successful
run reports zero conversions and writes nothing, and it is documented as such in the
operator runbook alongside ``repair_stale_campaign_run_sites``, the other one-time,
safe-to-repeat command in this codebase.

``--dry-run`` performs read-only checks: it creates no ``CampaignRun`` row and re-keys no
event. It still exits non-zero when it finds an event it cannot explain, because that is
exactly the condition the operator must clear before the real run -- a dry run that
silently exited 0 in the presence of an unexplainable row would hide the one thing the
operator most needs to see before running for real. ``--dry-run`` applies every per-event
precondition the real pass applies -- window containment, in-run collision, and
existing-``ALLOC:``-url collision -- through the same shared helper the real pass calls
(NF-02, 35-REVIEW.md), so the two passes agree on the re-key count, the unexplained count,
every per-reason count and the exit status. A second GROUP whose ``Source line:`` resolves
to the same run identity key as an earlier one (``duplicate_identity``, NF-14,
35-REVIEW.md) is rejected outright, identically on both passes, before either one ever
attempts a write for it -- so the two passes also agree about which group exists at all,
not only which per-event preconditions a surviving group's events must clear. That
agreement is what makes the promise in the preceding sentence true rather than
aspirational: a dry run can no longer exit 0 over a fixture the immediately following
real run rejects.
"""

import logging
from collections import defaultdict
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import transaction
from tom_calendar.models import CalendarEvent

from solsys_code.allocation_projector import (
    allocation_night_description,
    allocation_night_title,
    allocation_night_url,
    preserved_dark_window_line,
)
from solsys_code.calendar_utils import update_calendar_event_key_and_fields
from solsys_code.campaign_utils import adopt_event_into_run, insert_or_create_campaign_run, preview_campaign_run_action

# Reused verbatim from the rewritten load_telescope_runs (plan 35-05), not re-derived: this
# is exactly what guarantees a converted run is byte-for-byte indistinguishable from one a
# fresh import of the same line would have produced (D-17). Importing these underscore-named
# helpers across a management-command module boundary is deliberate here -- the alternative,
# copying their logic, is precisely the drift risk this reuse closes.
from solsys_code.management.commands.load_telescope_runs import (
    _CLASSICAL_RUN_STATUS,
    _iter_run_nights,
    _source_identifier,
    _window_token_to_time,
)
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import get_site, observing_night, parse_run_line

logger = logging.getLogger(__name__)

_SOURCE_LINE_MARKER = 'Source line: '

# D-18's named reason vocabulary, in report order.
_NO_SOURCE_LINE = 'no_source_line'
_UNPARSEABLE_SOURCE_LINE = 'unparseable_source_line'
_UNRESOLVABLE_SITE = 'unresolvable_site'
_CAMPAIGN_MISMATCH = 'campaign_mismatch'
_FOREIGN_ATTRIBUTION = 'foreign_attribution'
# WR-11 (35-REVIEW.md): a dedicated category, not folded into _OTHER. The module's whole
# contract is that every printed reason tells the operator what to DO -- and a collision
# calls for a different action (find and delete or re-attribute the duplicate row) than a
# generic unexpected error, so it needs its own name in both the summary breakdown and the
# CommandError message for an operator to recognise and act on.
_KEY_COLLISION = 'key_collision'
# NF-02 (35-REVIEW.md): a dedicated category, not folded into _OTHER, for the same reason
# _KEY_COLLISION is not -- an out-of-window derived night is a known, named condition with
# its own operator action (correct the event's stored start time, or the schedule line's
# date range, so the two agree), not an unexpected error. The module's contract is that
# every printed reason tells the operator what to DO.
_WINDOW_MISMATCH = 'window_mismatch'
# NF-14 (35-REVIEW.md): a dedicated category, not folded into _KEY_COLLISION or _OTHER.
# _source_identifier() deliberately ignores parsed.status (load_telescope_runs.py), so two
# Source lines that differ only in status word (e.g. an "allocation" and a "cancelled" line
# for the same telescope/instrument/window) resolve to the SAME run identity key -- two
# GROUPS (keyed on the raw source_line string) that would find-or-update the SAME
# CampaignRun row. Before this guard, the second group to be processed silently overwrote
# the first group's run fields (whichever group's dict-insertion order came second won),
# and its own per-event loop then reported the first group's already-re-keyed events as
# _KEY_COLLISION -- a reason whose documented operator action ("delete or re-attribute the
# duplicate row") is exactly wrong here: both lines are a legitimate second schedule entry,
# and the correct action is the one load_telescope_runs.py's own seen_keys/skipped_collision
# guard already documents -- add a bracketed proposal token to disambiguate.
_DUPLICATE_IDENTITY = 'duplicate_identity'
_OTHER = 'other'

_REASON_LABELS = {
    _NO_SOURCE_LINE: 'no parseable Source line: marker',
    _UNPARSEABLE_SOURCE_LINE: 'Source line: does not parse or its telescope is unknown/ambiguous',
    _UNRESOLVABLE_SITE: 'resolved site has no Observatory record or no timezone set',
    _CAMPAIGN_MISMATCH: "group's events disagree on their campaign",
    _FOREIGN_ATTRIBUTION: 'already attributed to a different CampaignRun',
    _KEY_COLLISION: 'derived ALLOC: night is already claimed by another event',
    _WINDOW_MISMATCH: "derived observing night falls outside the run's own window",
    _DUPLICATE_IDENTITY: (
        'a second Source line resolves to the same run identity key as an earlier group, or a '
        'CampaignRun already holds the derived identity key and cannot be proved to have come '
        'from this line'
    ),
    _OTHER: 'unexpected error',
}


class _WindowMismatchError(Exception):
    """Raised by the shared precondition helper (`_check_event_night()`) when an event's
    independently-derived observing night lies outside the window the group's schedule
    line implies. Raised, not branched, for the same reason `_KeyCollisionError` is: a
    dedicated `except _WindowMismatchError` clause ahead of the broad `except Exception`
    below routes it to the named _WINDOW_MISMATCH reason instead of _OTHER."""


class _KeyCollisionError(Exception):
    """Raised inside the per-event savepoint (real path) or the read-only preview loop
    (dry-run path) when a derived observing night is already claimed -- either by another
    event in this same cutover run, or by a CalendarEvent row that already holds the
    derived ALLOC:{run_pk}:{night} url. Raising (rather than branching) lets this check
    live inside the same savepoint boundary that guarantees the losing event stays
    byte-identical (D-18), and a dedicated `except _KeyCollisionError` clause ahead of the
    broad `except Exception` below routes it to the _KEY_COLLISION reason instead of
    _OTHER."""


def _extract_source_line(description: str) -> str | None:
    """Returns the text after the literal ``'Source line: '`` marker on the first
    description line that starts with it, or None when no line does.

    Args:
        description: a candidate event's stored ``description``.

    Returns:
        str | None: the recovered schedule-line text, or None.
    """
    for raw_line in (description or '').split('\n'):
        if raw_line.startswith(_SOURCE_LINE_MARKER):
            return raw_line[len(_SOURCE_LINE_MARKER) :]
    return None


def _check_event_night(
    event: CalendarEvent,
    run: CampaignRun | None,
    site_zone: ZoneInfo,
    window_start: date,
    window_end: date,
    claimed_nights: set[date],
) -> date:
    """Applies, in order, the three per-event preconditions both the dry-run and the real
    cutover branch must agree on, and returns the resolved observing night only when all
    three hold.

    NF-02 (35-REVIEW.md): this is the single place all three preconditions live, so the
    dry-run and real branches cannot drift apart again -- a future fourth check added here
    automatically applies to both callers.

    ``window_start``/``window_end`` are parameters rather than read off ``run`` so both
    branches provably evaluate the SAME window: the dry-run branch passes the previewed
    values out of ``fields`` (what the real pass would write), and the real branch passes
    the values it just wrote. This is also what lets the check still run on the dry-run
    path when no ``CampaignRun`` exists yet (``run is None``) -- the window is available
    from the preview even though the run row is not.

    The existing-ALLOC-url probe (the third precondition) is the one check that is
    legitimately skipped when ``run is None``: a run with no primary key can hold no
    ``ALLOC:{run_pk}:{night}`` url in the table, so there is nothing to probe. That skip
    preserves parity rather than breaking it -- when the real pass then creates the run,
    its fresh primary key cannot collide with any pre-existing row either. When a
    ``CampaignRun`` already exists for the group, the dry run sees the same primary key
    the real pass will update, so the probe runs identically on both.

    Args:
        event: the candidate legacy blank-url event.
        run: the group's CampaignRun, or None on the dry-run path before it exists.
        site_zone: the site's timezone, for deriving the event's observing night.
        window_start: the run's (or previewed run's) window start date, inclusive.
        window_end: the run's (or previewed run's) window end date, inclusive.
        claimed_nights: nights already claimed by an earlier event in this same group.
            Read here; the CALLER adds the night after its own write (or preview) succeeds,
            so a failed event leaves the night free for a later one.

    Returns:
        date: the event's resolved observing night, when all three preconditions hold.

    Raises:
        _WindowMismatchError: the derived night lies outside window_start..window_end.
        _KeyCollisionError: the night is already in claimed_nights, or (when run is not
            None) another CalendarEvent already holds the derived ALLOC: url.
    """
    night = observing_night(event.start_time, site_zone)
    if not (window_start <= night <= window_end):
        raise _WindowMismatchError(f"derived night {night} falls outside the run's window {window_start}..{window_end}")
    if night in claimed_nights:
        raise _KeyCollisionError(f'a second event already claims night {night}')
    if run is not None:
        candidate_url = allocation_night_url(run, night)
        holder = CalendarEvent.objects.filter(url=candidate_url).exclude(pk=event.pk).first()
        if holder is not None:
            raise _KeyCollisionError(f'night {night} url is already held by CalendarEvent pk={holder.pk}')
    return night


class Command(BaseCommand):
    """One-time cutover: convert every legacy blank-url classical CalendarEvent into an
    allocation night owned by a CampaignRun (D-17), leaving anything it cannot explain
    completely untouched and reported (D-18)."""

    help = (
        'One-time cutover (D-15/D-17): converts legacy blank-url classical CalendarEvent '
        'rows -- written by load_telescope_runs before its Phase 35 allocation rewrite -- '
        'into CampaignRuns plus ALLOC:-keyed events. Needs no schedule file: every fact it '
        "uses is recovered by re-parsing each event's own description. Never deletes a "
        'CalendarEvent row on any path -- an event it cannot explain is left byte-identical '
        'and reported, and the command exits non-zero so the operator resolves it in the '
        'admin. Safe to re-run: a converted event drops out of the candidate set, so a '
        'second run over already-converted data reports zero conversions.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help=(
                'Report what would be converted without creating a CampaignRun or re-keying '
                'any event. Still exits non-zero when an unexplainable event is found -- that '
                'is the condition the operator must clear before the real run.'
            ),
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Convert every legacy blank-url classical event it can explain into an allocation
        night, reporting (never touching) everything it cannot.

        Returns:
            str | None: None on completion (only reached when nothing is unexplained).

        Raises:
            CommandError: one or more events could not be explained; the operator must
                resolve them in the admin and re-run.
        """
        dry_run = options['dry_run']

        candidates = list(CalendarEvent.objects.filter(url='').order_by('pk'))

        groups: dict[str, list[CalendarEvent]] = defaultdict(list)
        unexplained: list[tuple[CalendarEvent, str, str]] = []
        reason_counts: dict[str, int] = defaultdict(int)

        def _mark_unexplained(events: list[CalendarEvent], category: str, reason: str) -> None:
            for event in events:
                unexplained.append((event, category, reason))
                reason_counts[category] += 1

        for event in candidates:
            source_line = _extract_source_line(event.description)
            if source_line is None:
                _mark_unexplained([event], _NO_SOURCE_LINE, _REASON_LABELS[_NO_SOURCE_LINE])
                continue
            groups[source_line].append(event)

        groups_found = len(groups)
        runs_created = runs_updated = runs_unchanged = 0
        events_rekeyed = 0

        # NF-14 (35-REVIEW.md): seen_keys is the load_telescope_runs.py-style guard --
        # keyed on the run identity key, populated the moment a group first claims it, so
        # a SECOND group sharing that key is rejected outright, before either pass ever
        # attempts a write for it. claimed_by_key is keyed the same way (rather than a
        # fresh set() per group): in practice seen_keys already stops a second group from
        # ever reaching the per-event loop, so the two dicts agree, but sharing the same
        # key-scoped set here (instead of a per-group one) is what lets a dry run detect a
        # collision even while `run is None` -- the actual mechanism this fix depends on --
        # rather than relying solely on the seen_keys `continue` never being bypassed by a
        # future refactor.
        seen_keys: dict[str, str] = {}
        claimed_by_key: dict[str, set[date]] = defaultdict(set)

        for source_line, events in groups.items():
            try:
                parsed = parse_run_line(source_line)
            except ValueError as exc:
                reason = f'{_REASON_LABELS[_UNPARSEABLE_SOURCE_LINE]}: {exc}'
                _mark_unexplained(events, _UNPARSEABLE_SOURCE_LINE, reason)
                continue

            try:
                site = get_site(parsed.telescope)
            except Observatory.DoesNotExist as exc:
                _mark_unexplained(events, _UNRESOLVABLE_SITE, f'{_REASON_LABELS[_UNRESOLVABLE_SITE]}: {exc}')
                continue

            if not site.timezone:
                reason = (
                    f'{_REASON_LABELS[_UNRESOLVABLE_SITE]}: Observatory {site.short_name!r} '
                    f'(obscode={site.obscode}) has no timezone set'
                )
                _mark_unexplained(events, _UNRESOLVABLE_SITE, reason)
                continue

            try:
                nights = _iter_run_nights(parsed)
                window_start, window_end = nights[0], nights[-1]
                key = _source_identifier(parsed, window_start, window_end)
            except ValueError as exc:
                reason = f'{_REASON_LABELS[_UNPARSEABLE_SOURCE_LINE]}: {exc}'
                _mark_unexplained(events, _UNPARSEABLE_SOURCE_LINE, reason)
                continue

            # NF-14 (35-REVIEW.md): reject a second group whose Source line resolves to the
            # SAME run identity key as an earlier group, BEFORE any write is attempted for
            # it -- otherwise insert_or_create_campaign_run()/preview_campaign_run_action()
            # would find-or-update the SAME CampaignRun row the earlier group already
            # claimed, silently merging two schedule lines into one run (whichever group's
            # dict-insertion order came second would win) rather than reporting the
            # collision. Checked identically on both --dry-run and the real pass, in the
            # same loop, so the two agree about which GROUP exists at all -- not only which
            # per-event preconditions a surviving group's events must clear.
            if key in seen_keys:
                _mark_unexplained(
                    events,
                    _DUPLICATE_IDENTITY,
                    f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: {seen_keys[key]!r} already claimed {key!r}; '
                    "edit the affected events' description 'Source line:' text to "
                    'disambiguate the two groups in the Django admin, then re-run',
                )
                continue

            # NF-14/NF-19/CR-01 (35-REVIEW.md, BLOCKER): seen_keys above only protects THIS
            # process -- the thing it protects, insert_or_create_campaign_run()/
            # preview_campaign_run_action() below, find-or-update against the DATABASE,
            # where a claimant can already exist from an earlier cutover invocation or from
            # load_telescope_runs. The guard has to look there too, or the re-run this
            # command's own CommandError prescribes silently merges a second group into the
            # first group's run (the harm NF-19 reproduced). The claimant's own schedule
            # line is recoverable from its stored observation_details with the SAME
            # _extract_source_line() parser this module already uses -- no new parsing
            # code. A recovered line of None is NOT permissive: observation_details is
            # writable from the Django admin (solsys_code/admin.py:165), from
            # import_campaign_csv.py:321 and from campaign_forms.py:65, so its absence is
            # not evidence of agreement -- it is precisely the row this one-time destructive
            # migration cannot prove it owns, and it must be refused, never find-and-updated
            # (CR-01, 35-REVIEW.md).
            existing_run = CampaignRun.objects.filter(source_identifier=key).first()
            if existing_run is not None:
                existing_source_line = _extract_source_line(existing_run.observation_details)
                if existing_source_line != source_line:
                    if existing_source_line is not None:
                        reason = (
                            f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: CampaignRun pk={existing_run.pk} already '
                            f'claimed {key!r} for a different Source line; '
                            "edit the affected events' description 'Source line:' text to "
                            'disambiguate the two groups in the Django admin, then re-run'
                        )
                    else:
                        reason = (
                            f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: CampaignRun pk={existing_run.pk} already '
                            f"claimed {key!r} with no recoverable 'Source line:' marker in its "
                            'observation_details, so this command cannot prove the run came from this '
                            "schedule line; restore or correct that run's observation_details 'Source "
                            "line:' text in the Django admin so it matches, or disambiguate the two "
                            'lines, then re-run -- note a run whose marker then matches will have its '
                            'fields re-applied from the schedule line on that re-run'
                        )
                    _mark_unexplained(events, _DUPLICATE_IDENTITY, reason)
                    continue

            target_list_ids = {event.target_list_id for event in events}
            if len(target_list_ids) > 1:
                _mark_unexplained(events, _CAMPAIGN_MISMATCH, _REASON_LABELS[_CAMPAIGN_MISMATCH])
                continue

            campaign = events[0].target_list

            # WR-09 (35-REVIEW.md): this lookup sits outside any other try/except in this
            # loop. load_telescope_runs.py's own structural assertion (asserted at import
            # time, so it also protects this call site since both modules share the same
            # dict) keeps a KeyError here practically unreachable today, but a defensive
            # catch here means a future divergence still gets reported per-group instead of
            # aborting the whole cutover mid-run, after partial commits, with no reason.
            try:
                run_status = _CLASSICAL_RUN_STATUS[parsed.status]
            except KeyError as exc:
                _mark_unexplained(events, _OTHER, f'unknown classical status {exc}')
                continue

            observation_details = f'Status: {parsed.status}\nSource line: {source_line}'
            if parsed.proposal is not None:
                observation_details += f'\nProposal: {parsed.proposal}'

            fields = {
                'source': CampaignRun.Source.CLASSICAL_FILE,
                'approval_status': CampaignRun.ApprovalStatus.APPROVED,
                'run_status': run_status,
                'campaign': campaign,
                'target': None,
                'site': site,
                'site_raw': parsed.telescope,
                'site_needs_review': False,
                'telescope_instrument': f'{parsed.telescope}/{parsed.instrument}',
                'window_start': window_start,
                'window_end': window_end,
                'night_start_utc': _window_token_to_time(parsed.start_window),
                'night_end_utc': _window_token_to_time(parsed.end_window),
                'observation_details': observation_details,
            }

            # D-18's foreign-attribution guard runs BEFORE any write -- the allocation
            # projector's own legacy-night takeover checks ownership first and re-keys
            # only what is writable (_may_write()-first idiom); mirroring that order here
            # is what keeps an unexplainable event byte-identical rather than re-keyed and
            # then discovered unsafe.
            unattributed_events = []
            for event in events:
                meta = CalendarEventMeta.objects.filter(event=event).first()
                if meta is not None and meta.run_id is not None:
                    _mark_unexplained([event], _FOREIGN_ATTRIBUTION, _REASON_LABELS[_FOREIGN_ATTRIBUTION])
                else:
                    unattributed_events.append(event)

            # WR-07 (35-REVIEW.md): a group whose EVERY event was just rejected above has
            # nothing writable left -- an unconditional run write here would still create
            # (or update) an APPROVED, site-resolved, windowed CampaignRun that owns zero
            # events, and the next reconcile_campaign_runs sweep projects that empty run
            # into a full duplicate set of ALLOC:{pk}:{night} nights over the same nights
            # the foreign run already owns. Every event has already been passed to
            # _mark_unexplained() by the loop above, so `continue` here preserves each
            # event's own reporting and the command's non-zero exit -- it only skips the
            # write this group has nothing left to justify.
            if not unattributed_events:
                continue

            # IN-02 (35-REVIEW.md): claimed only NOW, after the group is known to have
            # something to write -- not the moment its key first resolves. Before this
            # fix, an unconvertible group (campaign mismatch, unknown status, or every
            # event foreign-attributed) permanently claimed the key anyway, causing a
            # convertible sibling group to be reported under duplicate_identity, naming a
            # line that converted nothing. seen_keys is what the in-process guard above
            # reads; claimed_by_key[key] (populated below) is the per-night set, unaffected
            # by this move.
            seen_keys[key] = source_line

            # NF-05 (35-REVIEW.md): these four counters are LOCAL to this group and folded
            # into the outer totals only after the `with transaction.atomic()` block below
            # exits successfully. Before this fix they were the outer totals themselves,
            # incremented from inside the savepoint the very next comment describes -- so a
            # group-level rollback rolled back every write the savepoint made but left the
            # counters at their post-write values, and the operator-facing summary for a
            # one-time production migration reported work that had just been undone.
            group_created = group_updated = group_unchanged = group_rekeyed = 0

            # WR-06 (35-REVIEW.md): wrap this GROUP's writes (the run write plus every
            # event's re-key) in one savepoint, so an interruption (Ctrl-C, a connection
            # drop, an IntegrityError from a path not covered by the per-event `except`
            # below) rolls this group back cleanly rather than leaving a run with only a
            # subset of its nights re-keyed. A per-event failure is still caught and
            # reported individually (D-18's own contract: "an event it cannot explain is
            # left byte-identical and reported") via its OWN nested savepoint, so one bad
            # event never rolls back the group's run write or any other event in it --
            # only a failure NOT already handled per-event reaches the outer `except` below.
            try:
                with transaction.atomic():
                    if dry_run:
                        # IN-01 (35-REVIEW.md): existing_run is already bound above by the
                        # NF-19/CR-01 guard's own lookup -- nothing between that binding and
                        # here creates a run holding this key, so re-querying it here was a
                        # redundant read and a name shadowed a reader could mistake for a
                        # fresh one.
                        action = preview_campaign_run_action(existing_run, fields)
                        run = existing_run
                    else:
                        run, action = insert_or_create_campaign_run({'source_identifier': key}, fields)

                    if action == 'created':
                        group_created += 1
                    elif action == 'updated':
                        group_updated += 1
                    else:
                        group_unchanged += 1

                    # WR-11 (35-REVIEW.md): site_zone is needed by both paths now, so it is
                    # computed once here rather than inside the (former) real-only branch.
                    # claimed_nights is scoped to this group's identity KEY (NF-14,
                    # 35-REVIEW.md), via claimed_by_key[key] rather than a fresh set() --
                    # never shared across two DIFFERENT keys, since two different runs
                    # legitimately own nights of the same date (the url is keyed by run
                    # primary key too). The seen_keys guard above means at most one group
                    # ever reaches this line for a given key, so in every reachable case
                    # this is still exactly one set per group -- populated only after a
                    # night's own write (or preview) succeeds.
                    site_zone = ZoneInfo(site.timezone)
                    claimed_nights = claimed_by_key[key]

                    if dry_run:
                        # Read-only preview: apply the identical three preconditions the
                        # real path applies (via the shared _check_event_night() helper),
                        # but write nothing. Every candidate event is blank-url by
                        # construction (the query above), so an event that claims its
                        # night cleanly always would re-key once its group resolves --
                        # there is no "would be unchanged" outcome to preview at the event
                        # level. The window is read from `fields`, i.e. what the real pass
                        # would write, NOT off `existing_run`, whose window may be stale
                        # from an earlier import (NF-02, 35-REVIEW.md).
                        for event in unattributed_events:
                            try:
                                night = _check_event_night(
                                    event,
                                    run,
                                    site_zone,
                                    fields['window_start'],
                                    fields['window_end'],
                                    claimed_nights,
                                )
                            except _WindowMismatchError as exc:
                                _mark_unexplained([event], _WINDOW_MISMATCH, str(exc))
                                continue
                            except _KeyCollisionError as exc:
                                _mark_unexplained([event], _KEY_COLLISION, str(exc))
                                continue
                            except Exception as exc:  # noqa: BLE001 -- D-18's catch-all, per event
                                _mark_unexplained([event], _OTHER, f'{type(exc).__name__}: {exc}')
                                continue
                            claimed_nights.add(night)
                            group_rekeyed += 1
                    else:
                        for event in unattributed_events:
                            try:
                                with transaction.atomic():  # per-event savepoint
                                    # NF-02 (35-REVIEW.md): all three preconditions --
                                    # window containment, in-run collision, existing-url
                                    # collision -- now live in _check_event_night(), the
                                    # same helper the dry-run branch calls above, so the
                                    # two branches cannot drift apart again. The check
                                    # runs BEFORE update_calendar_event_key_and_fields()/
                                    # adopt_event_into_run() inside this same per-event
                                    # savepoint -- that ordering is what keeps a rejected
                                    # event byte-identical (D-18).
                                    night = _check_event_night(
                                        event, run, site_zone, run.window_start, run.window_end, claimed_nights
                                    )
                                    url = allocation_night_url(run, night)
                                    dark_line = preserved_dark_window_line(event)
                                    rekey_fields = {
                                        'title': allocation_night_title(run),
                                        'description': allocation_night_description(run, dark_line),
                                        'target_list': run.campaign,
                                    }
                                    rekeyed_event, _action = update_calendar_event_key_and_fields(
                                        event, url, rekey_fields
                                    )
                                    adopt_event_into_run(rekeyed_event, run)
                                # A night is claimed only AFTER the savepoint's `with` block
                                # exits successfully: an event whose re-key failed for some
                                # OTHER reason wrote no url, so a later event must still be
                                # free to claim that night.
                                claimed_nights.add(night)
                                group_rekeyed += 1
                            except _WindowMismatchError as exc:
                                _mark_unexplained([event], _WINDOW_MISMATCH, str(exc))
                            except _KeyCollisionError as exc:
                                _mark_unexplained([event], _KEY_COLLISION, str(exc))
                            except Exception as exc:  # noqa: BLE001 -- D-18's catch-all, per event
                                _mark_unexplained([event], _OTHER, f'{type(exc).__name__}: {exc}')
                # NF-05 (35-REVIEW.md): folded into the outer totals only now that the
                # `with` block above has exited successfully -- a group-level exception
                # below never reaches this line, so a rollback cannot leave these totals at
                # their post-write values.
                runs_created += group_created
                runs_updated += group_updated
                runs_unchanged += group_unchanged
                events_rekeyed += group_rekeyed
            except Exception as exc:  # noqa: BLE001 -- D-18's catch-all for a genuinely
                # unexpected, group-level failure -- the savepoint above has already rolled
                # back this group's run write and every event re-key. NF-05 (35-REVIEW.md):
                # mark only the events NOT already marked by the per-event handling above
                # (e.g. a foreign-attribution rejection before the savepoint even opened) --
                # marking the whole group unconditionally reported (and counted) an
                # already-explained event a second time, under a second, misleading reason.
                already_marked = {marked_event.pk for marked_event, _category, _reason in unexplained}
                _mark_unexplained(
                    [event for event in events if event.pk not in already_marked],
                    _OTHER,
                    f'{type(exc).__name__}: {exc}',
                )
                continue

        for event, _category, reason in unexplained:
            self.stderr.write(f'pk={event.pk} ({event.title!r}): {reason}')

        prefix = 'Done (dry run).' if dry_run else 'Done.'
        total_unexplained = len(unexplained)
        alloc_count = CalendarEvent.objects.filter(url__startswith='ALLOC:').count()
        self.stdout.write(
            f'{prefix} candidates: {len(candidates)}, groups: {groups_found}, '
            f'runs created: {runs_created}, updated: {runs_updated}, unchanged: {runs_unchanged}, '
            f'events re-keyed: {events_rekeyed}, unexplained: {total_unexplained}'
        )
        for category, count in reason_counts.items():
            self.stdout.write(f'  unexplained ({category}): {count} -- {_REASON_LABELS[category]}')
        self.stdout.write(f'{prefix} total ALLOC:-keyed calendar events now in the database: {alloc_count}')

        if total_unexplained:
            breakdown = ', '.join(f'{category}={count}' for category, count in sorted(reason_counts.items()))
            raise CommandError(
                f'{total_unexplained} event(s) could not be explained and were left untouched ({breakdown}). '
                'Resolve the listed events in the admin (see the stderr lines above for each pk, title and '
                'reason), then re-run this command -- it is safe to repeat: a repeat pass converts nothing '
                "it has not already explained, and updates an existing CampaignRun only when that run's "
                'stored Source line: matches the line being converted, reporting anything else instead '
                '(NF-19/CR-01, 35-REVIEW.md).'
            )
        return None
