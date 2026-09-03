"""Spike 002 sweep: project every KEY2026B-004 record twice and verify points 1-3.

Run from the repo root::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/002-observation-projector/sweep.py', run_name='__main__')"

This one WRITES (persists) CalendarEvent rows on purpose, so the result can be looked at in
FOMO's calendar page. ``cleanup.py`` removes exactly those rows again.
"""

import json
import os
import sys
import time
from collections import Counter

from django.utils import timezone

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPIKE_DIR)

import projector  # noqa: E402
from tom_calendar.models import CalendarEvent  # noqa: E402
from tom_observations.models import ObservationGroup, ObservationRecord  # noqa: E402

PROPOSAL = 'KEY2026B-004'
LOG: list[dict] = []


def log(category: str, **fields) -> None:
    LOG.append({'ts': timezone.now().isoformat(), 'category': category, **fields})


def main() -> None:
    records = ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL)
    n_records = records.count()
    urls = {projector.event_url(r, projector.facility_for(r)) for r in records}
    before = CalendarEvent.objects.filter(url__in=urls).count()
    run_total = CalendarEvent.objects.filter(url__startswith='RUN:').count()
    log('baseline', records=n_records, existing_events_in_namespace=before, run_namespace_events=run_total)

    runs = []
    for i in (1, 2):
        t0 = time.perf_counter()
        result = projector.project_queryset(records)
        elapsed = time.perf_counter() - t0
        runs.append({'run': i, 'seconds': round(elapsed, 2), **{k: v for k, v in result.items() if k != 'rows'}})
        log('sweep', run=i, seconds=round(elapsed, 2), counters=result['counters'], per_stage=result['per_stage'])
        if i == 1:
            rows = result['rows']

    # ---- point 1: exactly one event per record, stage-correct span ----------------------------
    events = {e.url: e for e in CalendarEvent.objects.filter(url__in=urls)}
    one_each = len(events) == n_records == len(urls)
    span_ok, span_bad = 0, []
    marked_terminal, unmarked_terminal = 0, []
    for r in records.select_related('target'):
        fac = projector.facility_for(r)
        e = events.get(projector.event_url(r, fac))
        if e is None:
            span_bad.append((r.observation_id, 'no event'))
            continue
        stage = projector.stage_for(r, fac)
        if stage in ('placed', 'observed'):
            expected = (r.scheduled_start, r.scheduled_end)
        else:
            from solsys_code.calendar_utils import record_time_window

            expected = record_time_window(r)
        if (e.start_time, e.end_time) == expected:
            span_ok += 1
        else:
            span_bad.append((r.observation_id, stage, str(e.start_time), str(expected[0])))
        if stage == 'terminal-negative':
            if e.title.startswith('['):
                marked_terminal += 1
            else:
                unmarked_terminal.append(r.observation_id)
    log('point1', one_event_per_record=one_each, span_ok=span_ok, span_bad=span_bad[:10], terminal_marked=marked_terminal, terminal_unmarked=unmarked_terminal)

    # ---- point 2: series identity on every multi-member group's events -----------------------
    groups = ObservationGroup.objects.filter(observation_records__in=records).distinct()
    group_report = []
    for g in groups:
        members = list(g.observation_records.all())
        titles = [events[projector.event_url(m, projector.facility_for(m))].title for m in members]
        with_series = sum(1 for t in titles if f'· {g.name} ' in t)
        group_report.append({'group': g.name, 'members': len(members), 'titled_with_series': with_series, 'sample': titles[0]})
    log('point2', groups=len(group_report), report=group_report)

    # ---- point 3: idempotency ------------------------------------------------------------------
    second = runs[1]['counters']
    idempotent = second.get('created', 0) == 0 and second.get('updated', 0) == 0
    log('point3', second_run_counters=second, idempotent=idempotent)

    # ---- untouched namespaces ------------------------------------------------------------------
    run_after = CalendarEvent.objects.filter(url__startswith='RUN:').count()
    log('isolation', run_namespace_before=run_total, run_namespace_after=run_after, unchanged=run_total == run_after)

    stage_counts = Counter(r['stage'] for r in rows)
    print(f'records: {n_records}   events in namespace before/after: {before} -> {len(events)}')
    print(f'run 1: {runs[0]["counters"]}  ({runs[0]["seconds"]}s)')
    print(f'run 2: {runs[1]["counters"]}  ({runs[1]["seconds"]}s)   idempotent: {idempotent}')
    print(f'stages: {dict(stage_counts)}')
    print(f'point 1  one event per record: {one_each}   stage-correct spans: {span_ok}/{n_records}   terminal marked: {marked_terminal}, unmarked: {len(unmarked_terminal)}')
    print(f'point 2  groups: {len(group_report)}')
    for g in group_report:
        print(f'         {g["members"]:>2} members, {g["titled_with_series"]:>2} titled with series — e.g. {g["sample"]}')
    print(f'point 3  idempotent re-run: {idempotent}')
    print(f'RUN: namespace untouched: {run_total == run_after} ({run_total} -> {run_after})')
    summary = {
        'records': n_records, 'events': len(events), 'one_event_per_record': one_each, 'span_ok': span_ok,
        'span_bad': len(span_bad), 'terminal_marked': marked_terminal, 'terminal_unmarked': len(unmarked_terminal),
        'groups': len(group_report), 'groups_fully_titled': sum(1 for g in group_report if g['titled_with_series'] == g['members']),
        'idempotent': idempotent, 'run_namespace_untouched': run_total == run_after, 'stages': dict(stage_counts), 'runs': runs,
    }
    with open(os.path.join(SPIKE_DIR, 'forensic-log-sweep.json'), 'w') as fh:
        json.dump({'summary': summary, 'events': LOG, 'rows': rows}, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
