# Hand-authored (not makemigrations-generated) -- one-way data migration, no schema change.
# Style follows 0011_backfill_campaignrun_telescope_class.py and
# solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py.
#
# D-06 (26-CONTEXT.md:94): telescope_class records WHY there is no site -- it is a
# PERMANENT, correct campaign-level fact, not a placeholder that means "site resolution
# failed". A non-blank telescope_class is an ANSWER to that question, not a resolution
# failure needing staff review. Every live writer is corrected in the same plan (quick task
# 260730-jty) to never flag a class-carrying row again; this migration unflags the rows that
# were already written with the old, incorrect rule before the fix landed.
#
# Measured effect on the dev DB at authoring time: 4 rows (pk 26 JUICE/SPACE, pk 29 LCO 1m0,
# pk 30 LCO 2m0, pk 37 Generic 1m0 robotic telescope -- all with blank site_raw) come back
# site_needs_review=False with their telescope_class values intact. The filter below is a
# DERIVED rule (site_needs_review=True AND non-blank telescope_class), never a hand-
# enumerated pk list, so this migration stays correct against any database, not just the
# dev DB it was authored against.
#
# One-way only (no real reverse supplied), matching 0004/0005/0011 precedent.
#
# WR-06 (per 0003's own precedent): this migration needs no helper at all, so it imports NO
# live application code -- the derived rule is a plain queryset filter against the
# historical model fetched via apps.get_model().

import logging

from django.db import migrations

logger = logging.getLogger(__name__)


def unflag_class_wide_campaign_run_site_review(apps, schema_editor):
    """Clear site_needs_review for every CampaignRun that carries a telescope_class.

    D-06: a class-carrying row is never a genuine site-resolution failure, so it never
    belongs in the staff "Sites Needing Review" queue. Only ``site_needs_review`` is
    written -- ``site``, ``telescope_class``, and ``approval_status`` are never touched by
    this migration.
    """
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')

    affected = CampaignRun.objects.filter(site_needs_review=True).exclude(telescope_class='')

    # Log the affected pks and class values at INFO before writing, so a replay of this
    # migration against an unfamiliar database leaves an audit trail of exactly which rows
    # it touched.
    for run in affected.order_by('pk'):
        logger.info(
            'Unflagging CampaignRun pk=%s (telescope_class=%r) -- site_needs_review True -> False',
            run.pk,
            run.telescope_class,
        )

    affected.update(site_needs_review=False)


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0011_backfill_campaignrun_telescope_class'),
    ]

    operations = [
        migrations.RunPython(unflag_class_wide_campaign_run_site_review, reverse_code=migrations.RunPython.noop),
    ]
