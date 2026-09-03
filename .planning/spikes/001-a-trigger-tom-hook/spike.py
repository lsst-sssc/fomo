"""Spike 001a: does TOM's ``observation_change_state`` hook fire on the saves a projector cares about?

Run from the repo root (Django must be set up, hence ``manage.py shell``)::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/001-a-trigger-tom-hook/spike.py', run_name='__main__')"

Every database write happens inside a transaction that is rolled back at the end, so the real
KEY2026B-004 records are left exactly as they were. The forensic log is written to
``forensic-log.json`` next to this file.
"""

import json
import os
import sys
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPIKE_DIR)

import hook_receiver  # noqa: E402  (needs SPIKE_DIR on sys.path first)
from tom_observations.facilities.lco import LCOFacility  # noqa: E402
from tom_observations.models import ObservationRecord  # noqa: E402
from tom_targets.tests.factories import NonSiderealTargetFactory  # noqa: E402

PROPOSAL = 'KEY2026B-004'
LOG: list[dict] = []


def log(category: str, **fields) -> None:
    LOG.append({'ts': timezone.now().isoformat(), 'category': category, **fields})


def fired_since(mark: int) -> list[dict]:
    return hook_receiver.FIRED[mark:]


def scenario(name: str, expect_fire: bool | None, action) -> dict:
    """Run one scenario, report whether the hook fired, and whether that matched expectation."""
    mark = len(hook_receiver.FIRED)
    action()
    fired = fired_since(mark)
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
    original_hooks = dict(settings.HOOKS)
    settings.HOOKS['observation_change_state'] = 'hook_receiver.receiver'
    log('setup', hook=settings.HOOKS['observation_change_state'])

    results = []
    with transaction.atomic():
        queued = ObservationRecord.objects.filter(
            facility='LCO', parameters__proposal=PROPOSAL, status='PENDING', scheduled_start__isnull=True
        ).order_by('pk')
        rec = queued.first()
        rec2 = queued.exclude(pk=rec.pk).first()
        log('fixture', queued_only_count=queued.count(), rec=rec.observation_id, rec2=rec2.observation_id)
        now = timezone.now()

        # S1 — the common narrowing step: block placed, status unchanged (PENDING -> PENDING).
        def s1():
            rec.scheduled_start = now
            rec.scheduled_end = now + timedelta(minutes=19)
            rec.save()

        results.append(scenario('S1 schedule-only change, status unchanged', expect_fire=False, action=s1))

        # S2 — status transition on the same record.
        def s2():
            rec.status = 'COMPLETED'
            rec.save()

        results.append(scenario('S2 status change PENDING->COMPLETED', expect_fire=True, action=s2))

        # S3 — creation (what backfill_lco_observations / a submission does).
        def s3():
            ObservationRecord.objects.create(
                target=NonSiderealTargetFactory(),
                facility='LCO',
                observation_id='spike-001a-created',
                status='PENDING',
                parameters={'proposal': PROPOSAL},
            )

        results.append(scenario('S3 creation', expect_fire=True, action=s3))

        # S4 — the real updatestatus path: facility.update_observation_status() with the portal
        # reporting a placed block and an unchanged state. Monkeypatched so no network is used.
        original_status = LCOFacility.get_observation_status

        def fake_status(self, observation_id):
            return {'state': rec2.status, 'scheduled_start': now, 'scheduled_end': now + timedelta(minutes=19)}

        def s4():
            LCOFacility.get_observation_status = fake_status
            try:
                LCOFacility().update_observation_status(rec2.observation_id)
            finally:
                LCOFacility.get_observation_status = original_status

        results.append(scenario('S4 updatestatus path, block placed, state unchanged', expect_fire=False, action=s4))

        # S5 — the updatestatus path when the state DOES change (block observed).
        def fake_status_done(self, observation_id):
            return {'state': 'COMPLETED', 'scheduled_start': now, 'scheduled_end': now + timedelta(minutes=19)}

        def s5():
            LCOFacility.get_observation_status = fake_status_done
            try:
                LCOFacility().update_observation_status(rec2.observation_id)
            finally:
                LCOFacility.get_observation_status = original_status

        results.append(scenario('S5 updatestatus path, state PENDING->COMPLETED', expect_fire=True, action=s5))

        transaction.set_rollback(True)
        log('teardown', rolled_back=True)

    settings.HOOKS.clear()
    settings.HOOKS.update(original_hooks)

    print(f'{"scenario":58} fired  expected  ok')
    for r in results:
        print(f'{r["scenario"]:58} {str(r["fired"]):6} {str(r["expected_fire"]):9} {"✓" if r["matches_expectation"] else "✗"}')
    summary = {
        'scenarios': len(results),
        'fired': sum(r['fired'] for r in results),
        'matched_expectation': sum(r['matches_expectation'] for r in results),
        'schedule_only_fires': results[0]['fired'],
        'updatestatus_placed_block_fires': results[3]['fired'],
    }
    print('summary:', json.dumps(summary))
    with open(os.path.join(SPIKE_DIR, 'forensic-log.json'), 'w') as fh:
        json.dump({'summary': summary, 'events': LOG}, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
