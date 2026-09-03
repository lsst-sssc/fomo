"""Spike 002 point 4: a record save re-projects its own event through post_save — no sweep.

Run AFTER sweep.py, from the repo root::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/signal_demo.py', run_name='__main__')"

The record mutations and the resulting event updates run inside a transaction that is rolled
back, so the real record and its event are left as the sweep produced them.
"""

import json
import os
import sys
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPIKE_DIR)

import projector  # noqa: E402
from tom_calendar.models import CalendarEvent  # noqa: E402
from tom_observations.models import ObservationRecord  # noqa: E402

PROPOSAL = 'KEY2026B-004'
LOG: list[dict] = []


def snapshot(url: str) -> dict:
    e = CalendarEvent.objects.get(url=url)
    return {'title': e.title, 'start': e.start_time.isoformat(), 'end': e.end_time.isoformat(), 'modified': e.modified.isoformat()}


def main() -> None:
    projector.connect()
    try:
        with transaction.atomic():
            rec = (
                ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL, status='PENDING', scheduled_start__isnull=True)
                .select_related('target')
                .order_by('pk')
                .first()
            )
            url = projector.event_url(rec, projector.facility_for(rec))
            s0 = snapshot(url)
            LOG.append({'step': 'queued (from sweep)', 'observation_id': rec.observation_id, **s0})

            now = timezone.now().replace(microsecond=0)
            rec.scheduled_start = now
            rec.scheduled_end = now + timedelta(minutes=19)
            rec.save()  # status still PENDING -> TOM's hook is silent; post_save fires
            s1 = snapshot(url)
            LOG.append({'step': 'placed (schedule-only save)', **s1})

            rec.status = 'COMPLETED'
            rec.save()
            s2 = snapshot(url)
            LOG.append({'step': 'observed (status save)', **s2})

            transaction.set_rollback(True)
    finally:
        projector.disconnect()

    s3 = snapshot(url)
    LOG.append({'step': 'after rollback', **s3})

    narrowed = (s1['start'], s1['end']) == (now.isoformat(), (now + timedelta(minutes=19)).isoformat())
    summary = {
        'observation_id': rec.observation_id,
        'queued_span': (s0['start'], s0['end']),
        'placed_span': (s1['start'], s1['end']),
        'placed_title': s1['title'],
        'observed_title': s2['title'],
        'narrowed_without_sweep': narrowed,
        'title_moved_queued_to_scheduled': s0['title'].startswith('[QUEUED]') and s1['title'].startswith('[SCHEDULED]'),
        'title_clean_when_observed': not s2['title'].startswith('['),
        'restored_after_rollback': (s3['start'], s3['end'], s3['title']) == (s0['start'], s0['end'], s0['title']),
    }
    for row in LOG:
        print(f'{row["step"]:32} {row["title"][:60]:60} {row["start"]} -> {row["end"]}')
    print('summary:', json.dumps(summary))
    with open(os.path.join(SPIKE_DIR, 'forensic-log-signal.json'), 'w') as fh:
        json.dump({'summary': summary, 'steps': LOG}, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
