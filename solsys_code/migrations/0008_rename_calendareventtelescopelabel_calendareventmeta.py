# Hand-authored (not autodetected) per 26-DECISION.md Criterion 4 and
# 27-RESEARCH.md Pattern 4: non-interactive `makemigrations` autodetection cannot tell a
# class rename apart from a delete-plus-create, and this model's `event` field is its
# actual primary key (a OneToOneField declared with `primary_key=True`) -- a
# `DeleteModel`/`CreateModel` pair, which is what autodetection would produce for this
# rename, would drop and recreate the table and destroy the 11 real `is_verified` rows it
# holds. `RenameModel` instead preserves the table (and every row in it) in place.

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0007_campaignrun_contact_public_opt_in'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='CalendarEventTelescopeLabel',
            new_name='CalendarEventMeta',
        ),
    ]
