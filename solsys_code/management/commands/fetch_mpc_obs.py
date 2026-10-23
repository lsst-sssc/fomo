import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from django.core.management.base import BaseCommand, CommandError, CommandParser

from solsys_code.utils import ades_dataframe_to_ades_data, write_psv

MPC_GET_OBS_URL = 'https://data.minorplanetcenter.net/api/get-obs'


class Command(BaseCommand):
    """Write an object's MPC observations to an ADES PSV file."""

    help = (
        'Fetch all MPC observations of an object from the MPC Observations API (as used by MPC Explorer) '
        'and write them to an ADES PSV file, e.g. for find_orb'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments"""
        parser.add_argument(
            'desig',
            type=str,
            help='Designation as the MPC knows it, e.g. "2P", "P/2026 T1", "2000 LE29" or "A11InDq" (not "2P/Encke")',
        )
        parser.add_argument(
            '--output',
            '-o',
            type=Path,
            default=None,
            help='Output PSV file (default: <desig>.psv in the current directory)',
        )
        parser.add_argument(
            '--keep-deprecated',
            action='store_true',
            help='Keep observations the MPC has flagged as deprecated',
        )
        parser.add_argument(
            '--timeout',
            type=float,
            default=180,
            help='Timeout for the API request in seconds (objects with many observations take a while)',
        )
        return super().add_arguments(parser)

    def handle(self, *args: Any, **options: Any) -> None:
        """Fetch the observations and write the PSV file"""
        desig = options['desig']
        output = options['output'] or Path(re.sub(r'[^\w.-]+', '_', desig) + '.psv')

        try:
            response = requests.get(
                MPC_GET_OBS_URL,
                json={'desigs': [desig], 'output_format': ['ADES_DF']},
                timeout=options['timeout'],
            )
        except requests.RequestException as e:
            raise CommandError(f'MPC Observations API request failed: {e}') from e
        if not response.ok:
            try:
                message = response.json().get('message', response.text)
            except ValueError:
                message = response.text
            raise CommandError(f'MPC Observations API returned {response.status_code} for {desig!r}: {message}')

        results = response.json()
        if not results or not results[0].get('ADES_DF'):
            raise CommandError(f'No observations found for {desig!r}')

        # Every field comes back as a string; keep it that way so the original precision is preserved
        ades_df = pd.DataFrame(results[0]['ADES_DF'], dtype=str)
        ades_data = ades_dataframe_to_ades_data(ades_df, drop_deprecated=not options['keep_deprecated'])
        write_psv(ades_data, output)
        self.stdout.write(f'Wrote {len(ades_data["observations"])} observations of {desig} to {output}')
