"""Spike 004: watch the 74 PENDING KEY2026B-004 records narrow for real, over real nights.

Two modes, chosen automatically:

* **baseline** (no ``baseline.json`` yet): snapshot every record's status, stage, schedule and
  base-layer event span. No database writes.
* **recheck** (``baseline.json`` exists): run spike 002's sweep (the backstop — the post_save
  receiver is not installed in FOMO's apps, so a separate ``updatestatus`` process cannot
  trigger it), then diff every record against the baseline and report what narrowed. Appends
  to ``recheck-history.json``. The sweep is the only write, and only where a record changed.

Between the two, run TOM's own status refresh — no spike code involved::

    python manage.py updatestatus          # polls every non-terminal LCO record (74 here), one portal call each

Then::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/004-live-narrowing-updatestatus/recheck.py', run_name='__main__')"
"""

import json
import os
import sys
from collections import Counter

from django.utils import timezone

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SPIKE_DIR), '002-observation-projector'))

import projector  # noqa: E402
from tom_calendar.models import CalendarEvent  # noqa: E402
from tom_observations.models import ObservationRecord  # noqa: E402

PROPOSAL = 'KEY2026B-004'
BASELINE = os.path.join(SPIKE_DIR, 'baseline.json')
HISTORY = os.path.join(SPIKE_DIR, 'recheck-history.json')


def snapshot() -> dict[str, dict]:
    out = {}
    records = ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL).select_related('target')
    for r in records:
        fac = projector.facility_for(r)
        e = CalendarEvent.objects.filter(url=projector.event_url(r, fac)).first()
        out[r.observation_id] = {
            'target': r.target.name,
            'status': r.status,
            'stage': projector.stage_for(r, fac),
            'scheduled_start': None if r.scheduled_start is None else r.scheduled_start.isoformat(),
            'scheduled_end': None if r.scheduled_end is None else r.scheduled_end.isoformat(),
            'event_start': None if e is None else e.start_time.isoformat(),
            'event_end': None if e is None else e.end_time.isoformat(),
            'event_title': None if e is None else e.title,
            'span_hours': None if e is None else round((e.end_time - e.start_time).total_seconds() / 3600, 2),
        }
    return out


def main() -> None:
    now = timezone.now().isoformat()
    if not os.path.exists(BASELINE):
        snap = snapshot()
        with open(BASELINE, 'w') as fh:
            json.dump({'captured_at': now, 'records': snap}, fh, indent=2)
        stages = Counter(v['stage'] for v in snap.values())
        print(f'baseline captured at {now}: {len(snap)} records, stages {dict(stages)}')
        print('next: run `python manage.py updatestatus` on a later night, then re-run this script.')
        return

    with open(BASELINE) as fh:
        base = json.load(fh)['records']
    sweep = projector.project_queryset(ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL))
    snap = snapshot()

    transitions: Counter = Counter()
    narrowed, advanced, expired, examples = 0, 0, 0, []
    for oid, cur in snap.items():
        old = base.get(oid)
        if old is None:
            transitions['new-record'] += 1
            continue
        if old['stage'] != cur['stage']:
            transitions[f'{old["stage"]} -> {cur["stage"]}'] += 1
            if old['stage'] == 'queued' and cur['stage'] in ('placed', 'observed'):
                narrowed += 1
            if old['stage'] == 'placed' and cur['stage'] == 'observed':
                advanced += 1
            if cur['stage'] == 'terminal-negative':
                expired += 1
            if len(examples) < 8:
                examples.append({'observation_id': oid, 'target': cur['target'], 'from': old['stage'], 'to': cur['stage'],
                                 'span_hours': [old['span_hours'], cur['span_hours']], 'title': cur['event_title']})
        elif (old['event_start'], old['event_end']) != (cur['event_start'], cur['event_end']):
            transitions[f'{cur["stage"]} (times moved)'] += 1

    entry = {
        'rechecked_at': now,
        'baseline_captured_at': json.load(open(BASELINE))['captured_at'],
        'sweep_counters': sweep['counters'],
        'stages_now': dict(Counter(v['stage'] for v in snap.values())),
        'transitions': dict(transitions),
        'narrowed_queued_to_block': narrowed,
        'advanced_placed_to_observed': advanced,
        'newly_terminal_negative': expired,
        'examples': examples,
    }
    history = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    history.append(entry)
    with open(HISTORY, 'w') as fh:
        json.dump(history, fh, indent=2, default=str)

    print(f'recheck at {now} (baseline {entry["baseline_captured_at"]})')
    print(f'sweep: {sweep["counters"]}')
    print(f'stages now: {entry["stages_now"]}')
    print(f'transitions: {entry["transitions"]}')
    print(f'narrowed queued->block: {narrowed}   advanced placed->observed: {advanced}   newly expired/cancelled: {expired}')
    for ex in examples:
        print(f'  {ex["target"]:>7} {ex["observation_id"]}: {ex["from"]} -> {ex["to"]}  span {ex["span_hours"][0]}h -> {ex["span_hours"][1]}h  {ex["title"][:70]}')


if __name__ == '__main__':
    main()
