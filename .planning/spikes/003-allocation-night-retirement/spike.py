"""Spike 003: an allocation's per-night event retires when a real observation links to that night.

Run AFTER spike 002's sweep (it relies on the record's base-layer event existing)::

    python manage.py shell -c "import runpy; runpy.run_path('.planning/spikes/003-allocation-night-retirement/spike.py', run_name='__main__')"

Everything runs inside a transaction that is rolled back: no CampaignRun, link, meta row or
ALLOC: event persists, and the base-layer event from spike 002 is left untouched.

What "allocation" means here (decisions D2/D3): a CampaignRun with campaign=None — the shape
plan 32-01 Task 1 made legal — standing for "we intend to observe target X at site Y on these
nights". The spike owns a tiny allocation projector that writes one sunset-to-sunrise event per
night under its own ``ALLOC:{run.pk}:{night}`` key (never ``RUN:``), and a handoff rule: a
night that has a linked ObservationRecord keeps the record's event and retires the allocation's.
Attribution of the base event to the run is expressed ONLY through CalendarEventMeta.run — no
CalendarEvent field is written by the campaign side.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone as dt_timezone

from django.db import transaction
from django.utils import timezone

SPIKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SPIKE_DIR), '002-observation-projector'))

import projector  # noqa: E402  (spike 002's base-layer projector, read-only reuse)
from tom_calendar.models import CalendarEvent  # noqa: E402
from tom_observations.models import ObservationRecord  # noqa: E402

from solsys_code.calendar_utils import insert_or_create_calendar_event, record_time_window  # noqa: E402
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation  # noqa: E402
from solsys_code.solsys_code_observatory.models import Observatory  # noqa: E402
from solsys_code.telescope_runs import sun_event  # noqa: E402

PROPOSAL = 'KEY2026B-004'
SITE_OBSCODE = 'K92'  # Sutherland-LCO B: a real LCO 1 m site with an Observatory row in this DB
LOG: list[dict] = []


def log(category: str, **fields) -> None:
    LOG.append({'ts': timezone.now().isoformat(), 'category': category, **fields})


# ---- the spike's allocation projector ---------------------------------------------------------

def alloc_url(run: CampaignRun, night: date) -> str:
    return f'ALLOC:{run.pk}:{night.isoformat()}'


def nights_of(run: CampaignRun) -> list[date]:
    n = (run.window_end - run.window_start).days + 1
    return [run.window_start + timedelta(days=i) for i in range(n)]


def night_of_record(record: ObservationRecord) -> date:
    """The night a record belongs to: the UTC date its current span starts on (spike simplification —
    the build must use the site-local date, cf. the 32-01 must-have about Chilean/Australian sites)."""
    return record_time_window(record)[0].date()


def linked_nights(run: CampaignRun) -> set[date]:
    return {night_of_record(link.observation_record) for link in run.observation_links.select_related('observation_record')}


def sunset_sunrise(site: Observatory, night: date) -> tuple[datetime, datetime]:
    a, b = sun_event(site, night, kind='sun')
    a, b = a.to_datetime(timezone=dt_timezone.utc), b.to_datetime(timezone=dt_timezone.utc)
    return (a, b) if a <= b else (b, a)


def project_allocation(run: CampaignRun) -> dict[str, int]:
    """One sunset->sunrise event per night WITHOUT a linked observation; retire the rest.

    Returns counters: created/updated/unchanged for kept nights, retired for deleted events.
    The base-layer record events are never written here; attribution goes through
    CalendarEventMeta.run only.
    """
    counters = {'created': 0, 'updated': 0, 'unchanged': 0, 'retired': 0}
    taken = linked_nights(run)
    for night in nights_of(run):
        url = alloc_url(run, night)
        if night in taken:
            deleted, _ = CalendarEvent.objects.filter(url=url).delete()
            counters['retired'] += int(deleted > 0)
            continue
        start, end = sunset_sunrise(run.site, night)
        fields = {
            'title': f'[PLANNED] {run.target.name} {run.telescope_instrument} (allocation, run #{run.pk})',
            'description': f'Allocation night {night.isoformat()} at {run.site.name} (obscode {run.site.obscode})\nSpike: 003-allocation-night-retirement',
            'start_time': start,
            'end_time': end,
            'telescope': run.telescope_instrument.split('/')[0],
            'instrument': run.telescope_instrument.split('/')[-1],
            'proposal': PROPOSAL,
            'target_list': None,
        }
        _event, action = insert_or_create_calendar_event({'url': url}, fields)
        counters[action] += 1
    # attribution (annotate, never adopt): link the base events of linked records to the run via meta only
    for link in run.observation_links.select_related('observation_record'):
        rec = link.observation_record
        base = CalendarEvent.objects.filter(url=projector.event_url(rec, projector.facility_for(rec))).first()
        if base is not None:
            meta, _ = CalendarEventMeta.objects.get_or_create(event=base)
            if meta.run_id != run.pk:
                meta.run = run
                meta.save(update_fields=['run'])
    return counters


def alloc_events(run: CampaignRun) -> list[str]:
    return sorted(CalendarEvent.objects.filter(url__startswith=f'ALLOC:{run.pk}:').values_list('url', flat=True))


def snapshot_base(rec: ObservationRecord) -> dict:
    e = CalendarEvent.objects.get(url=projector.event_url(rec, projector.facility_for(rec)))
    meta = CalendarEventMeta.objects.filter(event=e).first()
    return {'title': e.title, 'start': e.start_time.isoformat(), 'end': e.end_time.isoformat(), 'modified': e.modified.isoformat(),
            'meta_run': None if meta is None else meta.run_id}


def main() -> None:
    with transaction.atomic():
        rec = (
            ObservationRecord.objects.filter(facility='LCO', parameters__proposal=PROPOSAL, status='PENDING', scheduled_start__isnull=True)
            .select_related('target').order_by('pk').first()
        )
        night = night_of_record(rec)
        site = Observatory.objects.get(obscode=SITE_OBSCODE)
        run = CampaignRun.objects.create(
            campaign=None,
            target=rec.target,
            telescope_instrument='1m0/1M0-SCICAM-SINISTRO',
            site=site,
            site_raw=site.name,
            window_start=night - timedelta(days=1),
            window_end=night + timedelta(days=1),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.LCO_QUEUE,
            source_identifier=f'SPIKE003:{rec.observation_id}',
        )
        log('fixture', record=rec.observation_id, night=str(night), run=run.pk, window=[str(run.window_start), str(run.window_end)], site=site.obscode)
        base0 = snapshot_base(rec)
        run_before = CalendarEvent.objects.filter(url__startswith='RUN:').count()

        c1 = project_allocation(run)
        e1 = alloc_events(run)
        log('step1 project allocation (no link)', counters=c1, alloc_events=e1)

        link = CampaignRunObservation.objects.create(run=run, observation_record=rec, confirmed_at=timezone.now())
        c2 = project_allocation(run)
        e2 = alloc_events(run)
        base2 = snapshot_base(rec)
        log('step2 link record, re-project', counters=c2, alloc_events=e2, base_event=base2)

        c3 = project_allocation(run)
        log('step3 re-project again (idempotent?)', counters=c3, alloc_events=alloc_events(run))

        link.delete()
        c4 = project_allocation(run)
        e4 = alloc_events(run)
        log('step4 unlink, re-project (reversible?)', counters=c4, alloc_events=e4)

        run_after = CalendarEvent.objects.filter(url__startswith='RUN:').count()
        transaction.set_rollback(True)

    base_final = snapshot_base(rec)
    summary = {
        'record': rec.observation_id,
        'night': str(night),
        'allocation_nights': 3,
        'events_before_link': len(e1),
        'events_after_link': len(e2),
        'retired_on_link': c2['retired'],
        'retired_night_is_record_night': alloc_url(run, night) in e1 and alloc_url(run, night) not in e2,
        'base_event_fields_unchanged_by_campaign_side': (base0['title'], base0['start'], base0['end'], base0['modified']) == (base2['title'], base2['start'], base2['end'], base2['modified']),
        'base_event_attributed_via_meta_run': base2['meta_run'] == run.pk,
        'idempotent_reproject': c3 == {'created': 0, 'updated': 0, 'unchanged': 2, 'retired': 0},
        'reversible_on_unlink': len(e4) == 3 and c4['created'] == 1,
        'run_namespace_untouched': run_before == run_after,
        'rolled_back_clean': base_final['meta_run'] is None and CampaignRun.objects.filter(source_identifier=f'SPIKE003:{rec.observation_id}').count() == 0
        and CalendarEvent.objects.filter(url__startswith='ALLOC:').count() == 0,
    }
    for row in LOG:
        print(f'{row["category"]:40} {json.dumps({k: v for k, v in row.items() if k not in ("ts", "category")}, default=str)[:150]}')
    print('summary:', json.dumps(summary))
    with open(os.path.join(SPIKE_DIR, 'forensic-log.json'), 'w') as fh:
        json.dump({'summary': summary, 'events': LOG}, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
