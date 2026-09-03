"""Spike 001b: does a FOMO-owned Django ``post_save`` receiver on ObservationRecord fire on every save a
projector cares about — including the schedule-only saves TOM's own hook skips?

Run from the repo root::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/001-b-trigger-django-post-save/spike.py', run_name='__main__')"

Every database write happens inside a transaction that is rolled back at the end, so the real
KEY2026B-004 records are left exactly as they were. Forensic log: ``forensic-log.json``.
"""

import json
import os
from datetime import timedelta

from django.db import transaction
from django.db.models.signals import post_save
from django.utils import timezone
from tom_observations.facilities.lco import LCOFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
PROPOSAL = 'KEY2026B-004'
LOG: list[dict] = []
FIRED: list[dict] = []


def log(category: str, **fields) -> None:
    LOG.append({'ts': timezone.now().isoformat(), 'category': category, **fields})


def receiver(sender, instance, created, raw, update_fields, **kwargs):
    """Record every post_save Django emits for ObservationRecord; never touches the database."""
    FIRED.append(
        {
            'pk': instance.pk,
            'observation_id': instance.observation_id,
            'status': instance.status,
            'created': created,
            'raw': raw,
            'update_fields': None if update_fields is None else sorted(update_fields),
            'scheduled_start': None if instance.scheduled_start is None else instance.scheduled_start.isoformat(),
        }
    )


def scenario(name: str, expect_fire: bool | None, action) -> dict:
    mark = len(FIRED)
    action()
    fired = FIRED[mark:]
    result = {
        'scenario': name,
        'fired': bool(fired),
        'fire_count': len(fired),
        'expected_fire': expect_fire,
        'matches_expectation': (expect_fire is None) or (bool(fired) == expect_fire),
        'calls': fired,
    }
    log('scenario', **result)
    return result


def main() -> None:
    post_save.connect(receiver, sender=ObservationRecord, weak=False, dispatch_uid='spike-001b')
    log('setup', signal='post_save', sender='ObservationRecord')
    results = []
    try:
        with transaction.atomic():
            queued = ObservationRecord.objects.filter(
                facility='LCO', parameters__proposal=PROPOSAL, status='PENDING', scheduled_start__isnull=True
            ).order_by('pk')
            rec = queued.first()
            rec2 = queued.exclude(pk=rec.pk).first()
            log('fixture', queued_only_count=queued.count(), rec=rec.observation_id, rec2=rec2.observation_id)
            now = timezone.now()

            def s1():
                rec.scheduled_start = now
                rec.scheduled_end = now + timedelta(minutes=19)
                rec.save()

            results.append(scenario('S1 schedule-only change, status unchanged', expect_fire=True, action=s1))

            def s2():
                rec.status = 'COMPLETED'
                rec.save()

            results.append(scenario('S2 status change PENDING->COMPLETED', expect_fire=True, action=s2))

            def s3():
                ObservationRecord.objects.create(
                    target=NonSiderealTargetFactory(),
                    facility='LCO',
                    observation_id='spike-001b-created',
                    status='PENDING',
                    parameters={'proposal': PROPOSAL},
                )

            results.append(scenario('S3 creation', expect_fire=True, action=s3))

            original_status = LCOFacility.get_observation_status

            def fake_status(self, observation_id):
                return {'state': rec2.status, 'scheduled_start': now, 'scheduled_end': now + timedelta(minutes=19)}

            def s4():
                LCOFacility.get_observation_status = fake_status
                try:
                    LCOFacility().update_observation_status(rec2.observation_id)
                finally:
                    LCOFacility.get_observation_status = original_status

            results.append(scenario('S4 updatestatus path, block placed, state unchanged', expect_fire=True, action=s4))

            # S5 — a queryset .update() bypasses save() entirely: documents why a sweep backstop is still needed.
            def s5():
                ObservationRecord.objects.filter(pk=rec2.pk).update(scheduled_end=now + timedelta(minutes=25))

            results.append(scenario('S5 queryset.update() bypass', expect_fire=False, action=s5))

            # S6 — a save with update_fields: does the receiver still see it and learn which fields changed?
            def s6():
                rec2.scheduled_end = now + timedelta(minutes=30)
                rec2.save(update_fields=['scheduled_end'])

            results.append(scenario('S6 save(update_fields=[scheduled_end])', expect_fire=True, action=s6))

            transaction.set_rollback(True)
            log('teardown', rolled_back=True)
    finally:
        post_save.disconnect(receiver, sender=ObservationRecord, dispatch_uid='spike-001b')

    print(f'{"scenario":58} fired  expected  ok')
    for r in results:
        print(f'{r["scenario"]:58} {str(r["fired"]):6} {str(r["expected_fire"]):9} {"✓" if r["matches_expectation"] else "✗"}')
    summary = {
        'scenarios': len(results),
        'fired': sum(r['fired'] for r in results),
        'matched_expectation': sum(r['matches_expectation'] for r in results),
        'schedule_only_fires': results[0]['fired'],
        'updatestatus_placed_block_fires': results[3]['fired'],
        'queryset_update_fires': results[4]['fired'],
        'update_fields_seen_in_s6': results[5]['calls'][0]['update_fields'] if results[5]['calls'] else None,
    }
    print('summary:', json.dumps(summary))
    with open(os.path.join(SPIKE_DIR, 'forensic-log.json'), 'w') as fh:
        json.dump({'summary': summary, 'events': LOG}, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
