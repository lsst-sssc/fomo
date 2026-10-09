# Hand-authored (not makemigrations-generated) per 26-DECISION.md Criterion 4 /
# 27-RESEARCH.md Pattern 4's migration-precedent discussion and Pattern 3/Pitfall 2.
# One-way only (no real reverse supplied), matching this project's own 0004/0005 precedent
# -- neither supplies a real reverse for its data migration either.
#
# D-13/D-20: telescope_class is backfilled by a derived rule (calendar_utils.
# derive_telescope_class), never a hand-enumerated pk list, so this migration stays
# correct against any database, not just the dev DB it was authored against.
#
# D-14 negative constraint (27-RESEARCH.md Pitfall 3): this migration does not touch
# either existing CampaignRun partial UniqueConstraint.

import logging

from django.db import migrations

logger = logging.getLogger(__name__)


def backfill_campaign_run_telescope_class(apps, schema_editor):
    """D-13: derive telescope_class for every site-less CampaignRun, writing only when the
    derived value is non-blank.

    A row with no class signal takes no write at all -- the field default is already ''
    and blank-plus-flagged (site_needs_review=True) is what a genuine resolution failure
    correctly looks like (D-13). The rejected pk=31 row (D-15) needs no special case: it is
    site-less, so it enters this loop, derives blank (its site_raw 'X05' has no aperture-
    class or no-obscode-space signal), and takes no write.
    """
    # apps.get_model() -- the historical/frozen model state, never the live
    # solsys_code.models.CampaignRun, per this project's own 0004/0005 precedent
    # (27-RESEARCH.md Pitfall 2).
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')

    # Function-local import: derive_telescope_class takes only primitives (D-20), so
    # importing the plain helper here is safe -- it is Pitfall 2's one documented
    # exception, and it is exactly why D-20 requires the helper to take primitives rather
    # than a model instance: this migration must not couple to a model that keeps changing
    # through Phases 28-29.
    from solsys_code.calendar_utils import derive_telescope_class

    # site-resolved rows are deliberately out of scope: telescope_class describes a run
    # with no specific site, and this is the same "no resolved site" gate the importer
    # applies at its own call site (import_campaign_csv, Plan 27-06).
    for run in CampaignRun.objects.filter(site__isnull=True).order_by('pk'):
        telescope_class = derive_telescope_class(
            site_raw=run.site_raw, telescope_instrument=run.telescope_instrument
        )
        if not telescope_class:
            continue
        logger.info(
            'Backfilling CampaignRun pk=%s telescope_class=%r (telescope_instrument=%r, site_raw=%r)',
            run.pk,
            telescope_class,
            run.telescope_instrument,
            run.site_raw,
        )
        run.telescope_class = telescope_class
        run.save(update_fields=['telescope_class'])


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0010_campaignrun_source_telescope_class_campaignrunobservation'),
    ]

    operations = [
        migrations.RunPython(backfill_campaign_run_telescope_class, reverse_code=migrations.RunPython.noop),
    ]
