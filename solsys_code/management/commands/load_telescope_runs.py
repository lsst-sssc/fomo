from datetime import date, time, timedelta
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import transaction
from tom_targets.models import TargetList

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.campaign_utils import preview_campaign_run_action, write_and_reconcile_campaign_run
from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ESO_NOON_TO_NOON_SITES, KNOWN_STATUSES, ParsedRun, get_site, parse_run_line

# Classical-schedule parser status -> real-world CampaignRun.RunStatus (D-03). A schedule
# file is operator-vetted, so every line this command writes is APPROVED regardless of its
# status word -- the parser status carries real-world lifecycle state only, via run_status.
# The event's visible title prefix and status ring follow from run_status through the
# shared reconciler vocabulary (RUN_STATUS_CALENDAR_PREFIX), not from a command-local dict.
_CLASSICAL_RUN_STATUS = {
    'cancelled': CampaignRun.RunStatus.CANCELLED,
    'confirmed': CampaignRun.RunStatus.PLANNED,
    'allocation': CampaignRun.RunStatus.PLANNED,
    'proposed': CampaignRun.RunStatus.REQUESTED,
    'not confirmed': CampaignRun.RunStatus.REQUESTED,
}

# WR-09 (35-REVIEW.md): this module and cutover_classical_allocations.py both index
# _CLASSICAL_RUN_STATUS[parsed.status] unconditionally, safe today only because this dict's
# key set happens to match telescope_runs.KNOWN_STATUSES -- an invariant nothing enforced.
# Adding one status word to KNOWN_STATUSES without a matching entry here would turn that
# indexing into an uncaught KeyError that aborts the whole import/cutover mid-run, after
# partial commits, with no reason report. Fail loudly at import time instead.
assert (
    set(_CLASSICAL_RUN_STATUS) == KNOWN_STATUSES
), 'every telescope_runs.KNOWN_STATUSES member needs a CampaignRun.RunStatus mapping in _CLASSICAL_RUN_STATUS'


def _window_token_to_time(token: str | None) -> time | None:
    """Converts a ParsedRun start/end window token to a stored sub-night UTC time-of-day.

    Args:
        token: None, or a case-insensitive 'BoN'/'EoN' spelling (both mean "use the
            computed sun event for this night" -- D-04's null convention), or a 4-digit
            HHMM UTC string.

    Returns:
        time | None: the UTC time of day the token names, or None for a missing token or
            'BoN'/'EoN'.
    """
    if token is None:
        return None
    upper = token.upper()
    if upper in ('BON', 'EON'):
        return None
    return time(int(token[:2]), int(token[2:]))


def _source_identifier(parsed: ParsedRun, window_start: date, window_end: date) -> str:
    """Builds the deterministic, collision-safe key a classical run is matched on (D-01).

    The dates used are the run's OWN STORED window_start/window_end -- the observing
    nights after the site's night-convention adjustment (_iter_run_nights()), not the raw
    day range parsed from the line -- so re-importing the same line recomputes a
    byte-identical key from the row it is about to match. The proposal token is what makes
    two proposals sharing a telescope, an instrument and a set of nights distinguishable,
    which is exactly the real collision Phase 31's identity spike measured and which the
    telescope/instrument/start-time tolerance match alone could not resolve.

    Args:
        parsed: the ParsedRun for this line.
        window_start: the run's first observing night (post night-convention adjustment).
        window_end: the run's last observing night (post night-convention adjustment).

    Returns:
        str: e.g. 'CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-12:BoN:EoN', with a trailing
            ':{proposal}' segment appended when the line named one.
    """
    parts = [
        'CLASSICAL',
        parsed.telescope,
        parsed.instrument,
        window_start.isoformat(),
        window_end.isoformat(),
        parsed.start_window or 'BoN',
        parsed.end_window or 'EoN',
    ]
    if parsed.proposal is not None:
        parts.append(parsed.proposal)
    return ':'.join(parts)


def _iter_run_nights(parsed: ParsedRun) -> list[date]:
    """Returns one evening date per observing night, per the site's night convention.

    Las Campanas (Magellan) Start and End dates are BOTH inclusive observing
    nights, so a run yields E - S + 1 nights (INGEST-01;
    docs/design/telescope_runs_calendar.rst "Night convention"). ESO sites
    (``ESO_NOON_TO_NOON_SITES``, e.g. NTT / La Silla) transcribe their ranges
    verbatim from ESO's Tatoo tool, whose displayed END date is the noon-to-noon
    closing boundary of the last night rather than an observing night itself, so
    their last observing night is day2 - 1 (E - S nights).

    Args:
        parsed: a ParsedRun from parse_run_line().

    Returns:
        list[date]: evening dates for each night of the run.

    Raises:
        ValueError: if day2 < day1 (a descending or malformed same-month day
            range -- e.g. a typo like '20-5 July' -- that parse_run_line does
            not reject upstream; genuine cross-month ranges are already
            rejected at parse time by parse_run_line, PR-REVIEW-F2), or if an
            ESO noon-to-noon range leaves no observing nights after dropping
            its closing boundary (day2 <= day1).
    """
    if parsed.day2 < parsed.day1:
        raise ValueError(f'Invalid or descending same-month day range (day2 < day1): {parsed!r}')
    n_nights = parsed.day2 - parsed.day1 + 1
    if parsed.telescope in ESO_NOON_TO_NOON_SITES:
        # Tatoo's End date is the closing noon boundary of the last night, not an
        # observing night -- drop it so E - S nights remain.
        n_nights -= 1
        if n_nights < 1:
            raise ValueError(
                f'ESO noon-to-noon run range has no observing nights after dropping its '
                f'closing boundary (day1={parsed.day1}, day2={parsed.day2}): {parsed!r}'
            )
    first_night = date(parsed.year, parsed.month, parsed.day1)
    return [first_night + timedelta(days=i) for i in range(n_nights)]


class Command(BaseCommand):
    """Load classical telescope run lines from a file and create or update CampaignRuns.

    Each schedule line becomes one campaign-less CampaignRun (source=CLASSICAL_FILE), keyed
    on a deterministic, collision-safe source_identifier (D-01); the allocation projector
    (reconcile_run() -> allocation_projector.project_allocation()) draws the per-night
    calendar from the run, so this command writes no CalendarEvent of its own (D-02). An
    optional --campaign associates every run created or updated from the file with the
    named campaign (a tom_targets.TargetList); when omitted, campaign is left unset (None)
    on every run, matching prior behavior exactly.
    """

    help = (
        'Load classical telescope run lines from a file and create/update one CampaignRun '
        'per line; the allocation projector draws the per-night calendar. Optionally '
        'associate every run with a campaign (tom_targets.TargetList) via --campaign.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to a text file of classical run lines (one per line)',
        )
        parser.add_argument(
            '--campaign',
            required=False,
            help=(
                'Name of the campaign (tom_targets.TargetList) to associate every CampaignRun '
                'from this file with. If omitted, no campaign is set.'
            ),
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help=(
                'Report what would be created or updated without writing anything. For a run '
                'that already exists, the night-level preview comes from reconcile_run(dry_run=True). '
                'For a line that would create a brand-new run, a first-time dry run predicts night '
                'counts from the window length rather than from a sun-event computation.'
            ),
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def _resolve_campaign(self, campaign_name: str | None) -> TargetList | None:
        """Resolve --campaign to a TargetList by exact name match, or None if omitted.

        Mirrors only the explicit-name lookup branch of
        backfill_lco_observation_records.Command._resolve_campaign() -- deliberately
        no interactive prompt-when-omitted branch, since omitting --campaign here means
        "no campaign", not "ask me".

        Args:
            campaign_name: the --campaign value, or None if the flag was omitted.

        Returns:
            TargetList | None: the resolved campaign, or None if campaign_name is None.

        Raises:
            CommandError: no TargetList by that name exists, or more than one does.
        """
        if not campaign_name:
            return None
        matches = TargetList.objects.filter(name=campaign_name)
        if not matches.exists():
            raise CommandError(f'No campaign (TargetList) named {campaign_name!r} found.')
        if matches.count() > 1:
            raise CommandError(f'Multiple campaigns (TargetLists) named {campaign_name!r} found.')
        return matches.first()

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Load classical schedule lines and create or update one CampaignRun per line.

        For each schedule line: resolve the site, derive its observing nights and
        deterministic source_identifier key (D-01), then create-or-update the CampaignRun
        through write_and_reconcile_campaign_run() (or preview it under --dry-run) --
        never write a CalendarEvent directly. A key already claimed by an earlier line in
        this file is reported as a collision and skipped, never merged. If --campaign is
        given, it is resolved once upfront (fail fast on a bad name before any line is
        processed) and associated with every created/updated run; if omitted, campaign is
        left unset (None) on every run.

        Returns:
            str | None: None on completion.
        """
        filepath = options['filepath']
        dry_run = options['dry_run']
        campaign = self._resolve_campaign(options.get('campaign'))

        lines_processed = 0
        run_created = run_updated = run_unchanged = run_skipped = 0
        skipped_collision = 0

        night_created = night_updated = night_unchanged = 0
        night_retired = night_rekeyed = night_blocked = night_skipped = 0

        seen_keys: dict[str, int] = {}

        try:
            with open(filepath, encoding='utf-8') as f:
                file_lines = list(f)
        except OSError as exc:
            raise CommandError(f'Cannot open schedule file {filepath!r}: {exc}') from exc

        for line_num, line in enumerate(file_lines, start=1):
            if not line.strip():
                continue
            lines_processed += 1
            try:
                parsed = parse_run_line(line)
                site = get_site(parsed.telescope)
                nights = _iter_run_nights(parsed)
                window_start, window_end = nights[0], nights[-1]
                key = _source_identifier(parsed, window_start, window_end)

                if key in seen_keys:
                    self.stderr.write(
                        f'Line {line_num}: source_identifier {key!r} already claimed by line '
                        f'{seen_keys[key]} -- skipping (line text: {line.strip()!r})'
                    )
                    skipped_collision += 1
                    continue
                seen_keys[key] = line_num

                # NF-08 (35-REVIEW.md): narrowed to the ONE lookup WR-09 added this catch
                # for. The broad `except` below used to wrap this AND the ~80-line
                # write-plus-reconcile call beneath it, so a KeyError raised deep inside
                # the reconciler or the projector (a ZoneInfoNotFoundError from a malformed
                # Observatory.timezone subclasses KeyError) was swallowed and reported as
                # `Line N: 'some-key' (line text: ...)` -- a message naming neither the
                # module nor the stage that actually failed.
                try:
                    run_status = _CLASSICAL_RUN_STATUS[parsed.status]
                except KeyError as exc:
                    self.stderr.write(f'Line {line_num}: unknown classical status {exc} (line text: {line.strip()!r})')
                    run_skipped += 1
                    continue

                observation_details = f'Status: {parsed.status}\nSource line: {line.strip()}'
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

                if dry_run:
                    existing = CampaignRun.objects.filter(source_identifier=key).first()
                    action = preview_campaign_run_action(existing, fields)

                    if existing is not None:
                        reconcile_result = reconcile_run(existing, dry_run=True)
                        night_created += reconcile_result.created
                        night_updated += reconcile_result.updated
                        night_unchanged += reconcile_result.unchanged
                        night_retired += reconcile_result.retired
                        night_rekeyed += reconcile_result.rekeyed
                        night_blocked += reconcile_result.blocked
                        night_skipped += reconcile_result.skipped_nights
                    else:
                        # No row exists yet: a first-time dry run predicts night counts from
                        # the window length rather than from a sun-event computation.
                        night_created += len(nights)

                    # WR-02 (35-REVIEW.md, 35-14-PLAN.md): folded only here, after BOTH
                    # preview_campaign_run_action() and reconcile_run(existing, dry_run=True)
                    # have returned -- mirroring the real branch below, which increments only
                    # after its transaction.atomic() block has returned. Folding this BEFORE
                    # the reconcile call (the pre-fix shape) double-counted a line whose
                    # preview reconcile raised into both a run_created/run_updated/
                    # run_unchanged bucket AND run_skipped (via the except clauses further
                    # down), so a raising preview reported `skipped` alone in real mode but
                    # `skipped` PLUS one of the other three on the preview.
                    if action == 'created':
                        run_created += 1
                    elif action == 'updated':
                        run_updated += 1
                    else:
                        run_unchanged += 1
                else:
                    # NF-08 (35-REVIEW.md): write_and_reconcile_campaign_run() has no
                    # transaction boundary of its own -- without this, a reconcile_run()
                    # exception left the CampaignRun row it had just written committed
                    # while the line was counted under run_skipped, so the summary reported
                    # a line as skipped when a run row had in fact been created.
                    with transaction.atomic():
                        result = write_and_reconcile_campaign_run({'source_identifier': key}, fields)
                    if result.action == 'created':
                        run_created += 1
                    elif result.action == 'updated':
                        run_updated += 1
                    else:
                        run_unchanged += 1
                    night_created += result.reconcile.created
                    night_updated += result.reconcile.updated
                    night_unchanged += result.reconcile.unchanged
                    night_retired += result.reconcile.retired
                    night_rekeyed += result.reconcile.rekeyed
                    night_blocked += result.reconcile.blocked
                    night_skipped += result.reconcile.skipped_nights
            except ZoneInfoNotFoundError as exc:
                # NF-21 (35-REVIEW.md): a dedicated clause ahead of the (ValueError,
                # Observatory.DoesNotExist) catch below -- ZoneInfoNotFoundError subclasses
                # KeyError, not ValueError (35-VERIFICATION.md L234), so clause ORDER is what
                # decides which handler sees it; without this clause it escaped both and
                # aborted the whole import. `site` is bound at L244 by get_site(), two
                # statements into this same try, and ZoneInfoNotFoundError can only
                # originate downstream of it (inside write_and_reconcile_campaign_run() /
                # reconcile_run()), so `site` is always bound when this clause runs -- do
                # not "fix" this into a defensive getattr, it would hide a real bug instead.
                self.stderr.write(
                    f'Line {line_num}: invalid Observatory.timezone {site.timezone!r} for site '
                    f'{site.short_name!r} (obscode {site.obscode!r}): {exc} (line text: {line.strip()!r})'
                )
                run_skipped += 1
                continue
            except (ValueError, Observatory.DoesNotExist) as exc:
                self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
                run_skipped += 1
                continue

        prefix = 'Done (dry run).' if dry_run else 'Done.'
        self.stdout.write(
            f'{prefix} lines processed: {lines_processed}, '
            f'created: {run_created}, '
            f'updated: {run_updated}, '
            f'unchanged: {run_unchanged}, '
            f'skipped: {run_skipped}, '
            f'skipped_collision: {skipped_collision}'
        )
        self.stdout.write(
            f'{prefix} nights -- created: {night_created}, '
            f'updated: {night_updated}, '
            f'unchanged: {night_unchanged}, '
            f'retired: {night_retired}, '
            f'rekeyed: {night_rekeyed}, '
            f'blocked: {night_blocked}, '
            f'skipped: {night_skipped}'
        )
        return
