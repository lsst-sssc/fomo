"""D-23: backfill Observatory.timezone from lat/lon coordinates for every row still blank.

26-DECISION.md's "Timezone gap found during this spike" section asked Phase 27 to close
this before the Phase 29 reconciler ships, since it needs Observatory.timezone for
site-local-night key derivation. Named target: Observatory E10 (Siding Spring), whose blank
timezone was measured against the real dev DB during the Phase 26 spike.

Derived rule, not a hand-enumerated row list (same discipline as D-13's telescope_class
backfill): every Observatory with a blank timezone AND both lat and lon set gets its
timezone derived via timezonefinder's TimezoneFinder.timezone_at(), reusing
solsys_code_observatory.utils._get_timezone_finder() -- the same lazily-constructed,
module-cached finder MPCObscodeFetcher.to_observatory() already uses (quick task
260716-h8c). A coordinate with no timezone polygon (open ocean) leaves timezone blank
rather than fabricating a guess, exactly as to_observatory() already does. Rows with a
timezone already set are authoritative and are never overwritten. Rows with null
coordinates (space-based/geocentric obscodes such as 274, 289, C51, 500) are skipped and
stay blank -- correct, they have no ground location.
"""

import logging

from django.db import migrations

logger = logging.getLogger(__name__)


def backfill_observatory_timezone(apps, schema_editor):
    """Derive and save Observatory.timezone for every row with coordinates but no timezone."""
    from solsys_code.solsys_code_observatory.utils import _get_timezone_finder

    Observatory = apps.get_model('solsys_code_observatory', 'Observatory')
    finder = _get_timezone_finder()

    for obs in Observatory.objects.filter(timezone='', lat__isnull=False, lon__isnull=False):
        tz_name = finder.timezone_at(lat=obs.lat, lng=obs.lon)
        if not tz_name:
            # Open ocean or similar -- no timezone polygon at these coordinates. Leave
            # blank rather than fabricate a guess (mirrors to_observatory()'s CR-01
            # resolve-fails-gracefully / stays-retryable behavior).
            logger.warning(
                'backfill_observatory_timezone: no timezone polygon for obscode=%s at lat=%s lon=%s -- left blank',
                obs.obscode,
                obs.lat,
                obs.lon,
            )
            continue
        obs.timezone = tz_name
        obs.save(update_fields=['timezone'])
        logger.info(
            'backfill_observatory_timezone: obscode=%s -> timezone=%s',
            obs.obscode,
            tz_name,
        )


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code_observatory', '0002_observatory_timezone_seed'),
    ]

    operations = [
        migrations.RunPython(backfill_observatory_timezone, reverse_code=migrations.RunPython.noop),
    ]
