import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """The single idempotent sweep that projects and refreshes calendar events for every
    ``CampaignRun`` (RECON-01), replacing the retired ``backfill_range_calendar_events``
    command and the per-gap backfill pattern generally (RECON-09). All projection math lives
    in ``solsys_code.campaign_reconciler`` -- this command only loops it. Unlike the retired
    command, which imported the private ``_project_calendar_event`` out of the views module,
    this one imports a public function from a dedicated pure module.
    """

    help = (
        'Sweep every CampaignRun through the shared reconciler, projecting and refreshing '
        'its calendar events. --dry-run reports what would change without writing anything.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be reconciled without writing any CalendarEvent rows.',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Loop reconcile_run() over every CampaignRun and report the D-05 summary.

        Returns:
            str | None: None on completion.
        """
        dry_run = options['dry_run']

        # Deliberately unfiltered: RECON-01 says "every run, regardless of window length,
        # source, or site-resolution state", and reconcile_run()'s own _skip_reason() guard
        # (including its 'not approved' gate) is the single place that decides what does not
        # project -- this command must never grow a second, divergent copy of that rule.
        runs = CampaignRun.objects.all().select_related('site', 'campaign').order_by('pk')

        created = updated = unchanged = blocked = 0
        skipped_count = 0
        skipped_nights = 0
        detached = 0
        detach_declined = 0
        retired = 0
        rekeyed = 0
        legacy_deleted = 0
        failed_count = 0
        run_count = 0

        for run in runs:
            run_count += 1
            try:
                result = reconcile_run(run, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001 -- the only catch point, D-06
                logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
                self.stderr.write(f'Run pk={run.pk}: reconcile failed ({exc}) -- skipping')
                failed_count += 1
                continue

            if result.skipped_reason is not None:
                self.stderr.write(f'Run pk={run.pk}: skipped ({result.skipped_reason})')
                skipped_count += 1
                continue

            created += result.created
            updated += result.updated
            unchanged += result.unchanged
            blocked += result.blocked
            skipped_nights += result.skipped_nights
            detached += result.detached
            detach_declined += result.detach_declined
            retired += result.retired
            rekeyed += result.rekeyed
            legacy_deleted += result.legacy_deleted
            if result.retired:
                # D-05/D-07 (Phase 35): a linked record's placed or observed block now
                # occupies the night, so its allocation event is gone -- not a failure.
                self.stdout.write(
                    f'Run pk={run.pk}: {result.retired} night(s) retired -- now covered by a real observation'
                )
            if result.rekeyed:
                self.stdout.write(
                    f'Run pk={run.pk}: {result.rekeyed} legacy RUN:-keyed night(s) re-keyed into ALLOC: in place'
                )
            if result.legacy_deleted:
                self.stdout.write(
                    f'Run pk={run.pk}: {result.legacy_deleted} leftover per-night event(s) deleted -- these were '
                    'left over from the retired per-night key form, and this run now keeps a single '
                    'whole-window entry'
                )
            if result.blocked:
                self.stderr.write(f'Run pk={run.pk}: {result.blocked} event(s) blocked -- owned by someone else')
            if result.skipped_nights:
                # Normal, expected convergence (D-01/ANNOT-01): the night's entry already
                # comes from another writer attributed to this run -- not a failure.
                self.stdout.write(f'Run pk={run.pk}: {result.skipped_nights} night(s) skipped -- covered elsewhere')
            if result.detached:
                # WR-10: name both causes this counter counts -- a night superseded by
                # another attributed entry, or events left over from a key family this run
                # no longer belongs to. After 33-10 Task 1 a released row never carried a
                # human confirmation, so this no longer claims a stamp was cleared.
                self.stderr.write(
                    f'Run pk={run.pk}: {result.detached} event(s) released back into the attribution queue -- '
                    'superseded by another attributed entry, or left over from a key family this run no longer '
                    'belongs to'
                )
            if result.detach_declined:
                self.stderr.write(
                    f'Run pk={run.pk}: {result.detach_declined} superseded entr'
                    f"{'y' if result.detach_declined == 1 else 'ies'} left attributed -- a person confirmed "
                    'them, and an automated sweep never clears a human confirmation'
                )

        if dry_run:
            self.stdout.write(
                f'Done (dry run). runs: {run_count}, '
                f'would_create: {created}, '
                f'would_update: {updated}, '
                f'would_leave_unchanged: {unchanged}, '
                f'skipped: {skipped_count}, '
                f'failed: {failed_count}, '
                f'blocked: {blocked}, '
                f'skipped_nights: {skipped_nights}, '
                f'would_detach: {detached}, '
                f'detach_declined: {detach_declined}, '
                f'would_retire: {retired}, '
                f'would_rekey: {rekeyed}, '
                f'would_delete_legacy: {legacy_deleted}'
            )
        else:
            self.stdout.write(
                f'Done. runs: {run_count}, '
                f'created: {created}, '
                f'updated: {updated}, '
                f'unchanged: {unchanged}, '
                f'skipped: {skipped_count}, '
                f'failed: {failed_count}, '
                f'blocked: {blocked}, '
                f'skipped_nights: {skipped_nights}, '
                f'detached: {detached}, '
                f'detach_declined: {detach_declined}, '
                f'retired: {retired}, '
                f'rekeyed: {rekeyed}, '
                f'legacy_deleted: {legacy_deleted}'
            )
        return
