# Hand-authored per 26-DECISION.md Criterion 4 / 27-RESEARCH.md Pattern 4 (not the
# makemigrations-autodetected file -- read only to confirm the dependency list and field
# kwargs Django would emit, then hand-written so this phase's migration ordering is
# deliberate). Load-bearing facts this ordering depends on:
#
# 1. The two field-adding operations below must precede migration 0011's RunPython backfill
#    step -- a backfill cannot run before its column exists.
# 2. `source` needs NO RunPython step at all: a single static field default ('legacy') is
#    what backfills every pre-milestone row, exactly as 26-DECISION.md Criterion 4 locked
#    (Criterion 1: the source vocabulary and its constraint interaction).
# 3. The model-creation operation for CampaignRunObservation has no field-value dependency
#    on the two fields above -- it is ordered after them here purely for readability, not
#    correctness.
#
# D-14 negative constraint (27-RESEARCH.md Pitfall 3): no operation in this file adds,
# alters or removes a UniqueConstraint on CampaignRun. Neither `source` nor
# `telescope_class` joins unique_campaign_run_resolved_window or
# unique_campaign_run_tbd_natural_key -- the Phase 26 lock stands with executed evidence
# (27-RESEARCH.md Pattern 1).

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0009_calendareventmeta_run'),
        ('tom_observations', '0016_alter_facility_options'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='source',
            field=models.CharField(
                choices=[
                    ('web', 'Web submission'),
                    ('classical_file', 'Classical run file'),
                    ('lco_queue', 'LCO queue'),
                    ('gemini_queue', 'Gemini queue'),
                    ('csv_import', 'CSV import'),
                    ('legacy', 'Legacy (pre-v2.2)'),
                ],
                default='legacy',
                max_length=20,
                verbose_name='Ingest source',
            ),
        ),
        migrations.AddField(
            model_name='campaignrun',
            name='telescope_class',
            field=models.CharField(
                blank=True,
                choices=[
                    ('2m0', '2m0 class allocation'),
                    ('1m0', '1m0 class allocation'),
                    ('0m4', '0m4 class allocation'),
                    ('SPACE', 'Space observatory with no MPC code'),
                ],
                default='',
                max_length=10,
                verbose_name='Telescope class allocation',
            ),
        ),
        migrations.CreateModel(
            name='CampaignRunObservation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('confirmed_at', models.DateTimeField(blank=True, null=True, verbose_name='Confirmed at')),
                (
                    'confirmed_by',
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='confirmed_campaign_run_observations',
                        to=settings.AUTH_USER_MODEL,
                        verbose_name='Confirmed by',
                    ),
                ),
                (
                    'observation_record',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='campaign_run_links',
                        to='tom_observations.observationrecord',
                        verbose_name='Observation record',
                    ),
                ),
                (
                    'run',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='observation_links',
                        to='solsys_code.campaignrun',
                        verbose_name='Campaign run',
                    ),
                ),
            ],
            options={
                'constraints': [
                    models.UniqueConstraint(fields=('observation_record',), name='unique_campaign_run_observation_record')
                ],
            },
        ),
    ]
