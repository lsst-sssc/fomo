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
a DIFFERENT run; and any other exception, recorded with its own type name. Every reason is
printed with the event's primary key and title so an operator can find and correct the row
in the admin.

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
operator most needs to see before running for real.
"""

import logging
from collections import defaultdict
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
_OTHER = 'other'

_REASON_LABELS = {
    _NO_SOURCE_LINE: 'no parseable Source line: marker',
    _UNPARSEABLE_SOURCE_LINE: 'Source line: does not parse or its telescope is unknown/ambiguous',
    _UNRESOLVABLE_SITE: 'resolved site has no Observatory record or no timezone set',
    _CAMPAIGN_MISMATCH: "group's events disagree on their campaign",
    _FOREIGN_ATTRIBUTION: 'already attributed to a different CampaignRun',
    _OTHER: 'unexpected error',
}


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
            writable_events = []
            for event in events:
                meta = CalendarEventMeta.objects.filter(event=event).first()
                if meta is not None and meta.run_id is not None:
                    _mark_unexplained([event], _FOREIGN_ATTRIBUTION, _REASON_LABELS[_FOREIGN_ATTRIBUTION])
                else:
                    writable_events.append(event)

            # WR-07 (35-REVIEW.md): a group whose EVERY event was just rejected above has
            # nothing writable left -- an unconditional run write here would still create
            # (or update) an APPROVED, site-resolved, windowed CampaignRun that owns zero
            # events, and the next reconcile_campaign_runs sweep projects that empty run
            # into a full duplicate set of ALLOC:{pk}:{night} nights over the same nights
            # the foreign run already owns. Every event has already been passed to
            # _mark_unexplained() by the loop above, so `continue` here preserves each
            # event's own reporting and the command's non-zero exit -- it only skips the
            # write this group has nothing left to justify.
            if not writable_events:
                continue

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
                        existing_run = CampaignRun.objects.filter(source_identifier=key).first()
                        action = preview_campaign_run_action(existing_run, fields)
                        run = existing_run
                    else:
                        run, action = insert_or_create_campaign_run({'source_identifier': key}, fields)

                    if action == 'created':
                        runs_created += 1
                    elif action == 'updated':
                        runs_updated += 1
                    else:
                        runs_unchanged += 1

                    if dry_run:
                        # Every candidate event is blank-url by construction (the query
                        # above), so its url always changes once its own group resolves --
                        # there is no "would be unchanged" outcome to preview at the event
                        # level; a real run's events_rekeyed count is exactly this same
                        # len(writable_events) total.
                        events_rekeyed += len(writable_events)
                    else:
                        site_zone = ZoneInfo(site.timezone)
                        for event in writable_events:
                            try:
                                with transaction.atomic():  # per-event savepoint
                                    night = observing_night(event.start_time, site_zone)
                                    # WR-08 (35-REVIEW.md): the run's window comes from
                                    # _iter_run_nights(parsed) (the schedule line's own day
                                    # range), while the event's own night is an independent
                                    # derivation from its stored start_time -- nothing
                                    # asserts the two agree. A mismatch (an off-by-one ESO
                                    # boundary, or a CR-06-shaped bug in a stored event) would
                                    # re-key the event to an ALLOC:{pk}:{night} url OUTSIDE
                                    # the run's own window, which project_allocation()'s
                                    # convergence step then classifies as stale and DELETES
                                    # on the very next sweep -- silently converting this
                                    # command's own "never removes a CalendarEvent row, on
                                    # any path" guarantee into "hands the next sweep a row to
                                    # remove". Validate before re-keying.
                                    if not (run.window_start <= night <= run.window_end):
                                        raise ValueError(
                                            f"derived night {night} falls outside the run's window "
                                            f'{run.window_start}..{run.window_end}'
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
                                events_rekeyed += 1
                            except Exception as exc:  # noqa: BLE001 -- D-18's catch-all, per event
                                _mark_unexplained([event], _OTHER, f'{type(exc).__name__}: {exc}')
            except Exception as exc:  # noqa: BLE001 -- D-18's catch-all for a genuinely
                # unexpected, group-level failure -- the savepoint above has already rolled
                # back this group's run write and every event re-key.
                _mark_unexplained(events, _OTHER, f'{type(exc).__name__}: {exc}')
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
                'reason), then re-run this command -- it is safe to repeat.'
            )
        return None
