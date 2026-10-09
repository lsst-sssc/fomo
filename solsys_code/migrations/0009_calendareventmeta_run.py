# Hand-authored per 26-DECISION.md Criterion 4 / 27-RESEARCH.md Pattern 4: kept in its own
# migration, separate from the 0008 rename, so it stays atomic and reviewable -- a rename
# regression and a new-field regression can never be confused for each other.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0008_rename_calendareventtelescopelabel_calendareventmeta'),
    ]

    operations = [
        migrations.AddField(
            model_name='calendareventmeta',
            name='run',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='calendar_event_metas',
                to='solsys_code.campaignrun',
                verbose_name='Owning campaign run',
            ),
        ),
    ]
