from typing import Any

from django.core.management.base import BaseCommand, CommandParser

# Site code -> telescope label, mirroring solsys_code/telescope_runs.py:SITES naming
# convention (e.g. 'FTS'). Values are [ASSUMED] per RESEARCH.md Assumptions Log A1/A2
# (web-search only, not yet confirmed against real ObservationRecord.parameters data
# for this project's LCO proposal) — confirm against real records before relying on
# this mapping in production.
SITE_TELESCOPE_MAP = {
    'coj': 'FTS',
    'ogg': 'FTN',
}


class Command(BaseCommand):
    """Sync LCO queue ObservationRecords to the FOMO calendar as CalendarEvents."""

    help = 'Sync LCO queue ObservationRecords for a proposal to CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--proposal',
            type=str,
            required=True,
            help='LCO proposal code to filter ObservationRecords by',
        )

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Stub handler — implemented in Task 2.

        Returns:
            str | None: None on completion.
        """
        pass
