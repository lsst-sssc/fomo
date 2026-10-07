"""One-off repair: retire the three testing campaigns left in the live dev database (quick task 261002-l04).

What goes: TargetLists #4 '3I/ATLAS (demo)' and #5 '3I/ATLAS leading-comment demo' (left by demo-notebook runs on
2026-07-30) and #10 'WR06 tmp campaign' (the Phase 35 WR-06 review leftover), together with their 14 CampaignRuns, the
24 CalendarEvents those runs project (and the 24 CalendarEventMeta companion rows that go with them), and demo
Target #143 with its 2 TargetNames, 3 TargetExtras and its one membership row in list #4.
TargetList #3 '3I/ATLAS' is the real campaign and is NOT touched. Run #45 ('Uma Unresolved') is the only row behind the
"Sites needing review" banner on /campaigns/approval-queue/, so the banner goes with it.

The 24 events are deleted DELIBERATELY: 23 'ALLOC:' events and 1 'RUN:' event (pk 109, 'RUN:37': run 37 is class-wide,
so the reconciler keys its whole-window event as RUN:{pk}). The project rule is that nothing deletes RUN:/ALLOC:/
facility-url events casually. This script does it by literal pk, each event with the run it belongs to, in the same
transaction as the run. Pre-flight proves each event's url equals its literal and that its companion row attributes
it to that run.

261002-dsa must have been applied first. Event 334 (a stray with a companion row attributed to run 68) is that script's
to delete. Deleting run 68 here would null 334's companion row and break dsa's own pre-flight, so this script refuses
unless event 334 and its companion row are already gone.

Delete order, and why it is forced:
  1. the 24 events, by literal pk and url. CampaignRun has a pre_delete receiver that deletes a run's RUN:/ALLOC: events
     on its own; deleting the events first means exactly the verified 24 go and the receiver then finds nothing.
  2. the 14 runs. TargetList.campaign_runs is PROTECT, so runs must go before their lists.
  3. target 143. CampaignRun.target is SET_NULL, so the runs must already be gone. tom_targets then runs guardian's
     clean_orphan_obj_perms(), which deletes EVERY orphan object permission in the database. Pre-flight requires there
     are none, and the whole-table check requires both guardian tables to be unchanged.
  4. the three lists. CalendarEvent.target_list is SET_NULL, so the events must already be gone.
The receivers' own deletes never show up in QuerySet.delete() counts, so the script also compares whole-table counts of
20 watched models before and after, inside the transaction. Any difference rolls everything back.

This script is for ONE database only: the pks are literals. Pre-flight checks that the database still looks the way it
did when the script was written, and writes nothing if any check fails.

How to run it (always feed the file with `<`, never a pipe; `manage.py shell` ignores a pipe that is not yet readable):

    Dry run (the committed default; prints the table and the totals, writes nothing):
        python manage.py shell < \\
            .planning/quick/261002-l04-retire-the-three-testing-campaigns-targe/retire_testing_campaigns.py

    Real run (edit a COPY, never this file):
        sed 's/^DRY_RUN = True$/DRY_RUN = False/' \\
            .planning/quick/261002-l04-retire-the-three-testing-campaigns-targe/retire_testing_campaigns.py \\
            > /tmp/retire_testing_campaigns_real.py
        python manage.py shell < /tmp/retire_testing_campaigns_real.py

When to run it: between cron ticks (after a `=== FOMO unattended run END` banner in /var/log/fomo/unattended.log), or
with cron paused, and after taking a fresh online backup as the rollback:

    sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup 'src/fomo_db_20261002_pre_l04.sqlite3'"

A `database is locked` error inside the transaction means nothing was written; retry after the next END banner.
To point it at a throwaway copy instead of the live database, set FOMO_DATABASE_PATH for that one command.

Expected output of the real run (the first line is the database it is acting on; check it):

    asserted: 3 target lists, 14 runs, 24 events, 1 target
    will cascade: 24 CalendarEventMeta, 2 TargetName, 3 TargetExtra, 1 TargetList membership
    261002-dsa already applied: event 334 absent
    TargetList #3 '3I/ATLAS' (kept): runs 30, events 26, targets 1
    approved runs needing site review: [45]
    Total removed orphan object permissions instances: 0      (guardian's own log line)
    deleted: 24 events (+24 CalendarEventMeta), 14 runs, 1 target
        (+2 TargetName, 3 TargetExtra, 1 TargetList membership), 3 target lists      (printed as one line)
    whole-table changes as expected: 20/20 models
    TargetList #3 unchanged: runs 30, events 26, targets 1
    approved runs needing site review: [45] -> []
"""

from django.apps import apps
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from tom_calendar.models import CalendarEvent
from tom_targets.models import Target, TargetList

from solsys_code.models import CalendarEventMeta, CampaignRun

DRY_RUN = True

TARGET_LIST_NAMES = {4: '3I/ATLAS (demo)', 5: '3I/ATLAS leading-comment demo', 10: 'WR06 tmp campaign'}
TARGET_LIST_MEMBERS = {4: [143], 5: [], 10: []}

# The real campaign: must exist and must be referenced by nothing in scope.
KEPT_TARGET_LIST = (3, '3I/ATLAS')

# The demo target (pk, name, type) and the rows that cascade with it (accessor name -> row count).
DEMO_TARGET = (143, '3I/ATLAS (demo target)', 'NON_SIDEREAL')
DEMO_TARGET_CASCADES = {'aliases': 2, 'targetextra_set': 3}

# pk -> (campaign_id, contact_person, source, target_id)
EXPECTED_RUNS = {
    32: (4, 'Ada Test', 'csv_import', 143),
    33: (4, 'Ben Sample', 'csv_import', 143),
    34: (4, 'Cy Fixture', 'csv_import', 143),
    35: (4, 'Dee Approx', 'csv_import', 143),
    36: (4, 'Eli Blank', 'csv_import', 143),
    37: (4, 'Fay Review', 'csv_import', 143),
    38: (4, 'Gia Range', 'csv_import', 143),
    39: (4, 'Ike Pending', 'csv_import', 143),
    40: (5, 'Demo Contact One', 'csv_import', None),
    41: (5, 'Demo Contact Two', 'csv_import', None),
    42: (4, 'Grace Lifecycle', 'legacy', None),
    43: (4, 'Hal Lifecycle', 'legacy', None),
    45: (4, 'Uma Unresolved', 'csv_import', 143),
    68: (10, '', 'legacy', None),
}

# pk -> (run_pk, url). Run 38's nights are site-local dates, one event per night, 2025-08-01 to 2025-08-15.
EXPECTED_EVENTS = {
    104: (32, 'ALLOC:32:2025-07-04'),
    105: (33, 'ALLOC:33:2025-07-04'),
    106: (34, 'ALLOC:34:2025-07-06'),
    107: (35, 'ALLOC:35:2025-07-16'),
    108: (36, 'ALLOC:36:2025-07-06'),
    109: (37, 'RUN:37'),
    110: (38, 'ALLOC:38:2025-08-01'),
    111: (38, 'ALLOC:38:2025-08-02'),
    112: (38, 'ALLOC:38:2025-08-03'),
    113: (38, 'ALLOC:38:2025-08-04'),
    114: (38, 'ALLOC:38:2025-08-05'),
    115: (38, 'ALLOC:38:2025-08-06'),
    116: (38, 'ALLOC:38:2025-08-07'),
    117: (38, 'ALLOC:38:2025-08-08'),
    118: (38, 'ALLOC:38:2025-08-09'),
    119: (38, 'ALLOC:38:2025-08-10'),
    120: (38, 'ALLOC:38:2025-08-11'),
    121: (38, 'ALLOC:38:2025-08-12'),
    122: (38, 'ALLOC:38:2025-08-13'),
    123: (38, 'ALLOC:38:2025-08-14'),
    124: (38, 'ALLOC:38:2025-08-15'),
    125: (40, 'ALLOC:40:2025-08-20'),
    126: (41, 'ALLOC:41:2025-08-21'),
    357: (68, 'ALLOC:68:2025-07-04'),
}

# Event 334 belongs to 261002-dsa; it must already be gone.
DSA_STRAY_EVENT_PK = 334

# Per-step delete counts (zero entries dropped), as measured in a rolled-back trial on a copy.
EXPECTED_DELETE_COUNTS = {
    'events': {'tom_calendar.CalendarEvent': 24, 'solsys_code.CalendarEventMeta': 24},
    'runs': {'solsys_code.CampaignRun': 14},
    'target': {
        'tom_targets.BaseTarget': 1,
        'tom_targets.TargetName': 2,
        'tom_targets.TargetExtra': 3,
        'tom_targets.TargetList_targets': 1,
    },
    'target_lists': {'tom_targets.TargetList': 3},
}

# Whole-table count change for every watched model, so a signal receiver's side deletes cannot go unseen.
EXPECTED_TABLE_CHANGES = {
    'tom_calendar.CalendarEvent': -24,
    'solsys_code.CalendarEventMeta': -24,
    'solsys_code.CampaignRun': -14,
    'tom_targets.BaseTarget': -1,
    'tom_targets.TargetName': -2,
    'tom_targets.TargetExtra': -3,
    'tom_targets.TargetList': -3,
    'tom_targets.TargetList_targets': -1,
    'tom_calendar.EventTodo': 0,
    'solsys_code.CalendarEventDismissal': 0,
    'solsys_code.CampaignRunObservation': 0,
    'solsys_code.ObservationRecordDismissal': 0,
    'tom_targets.PersistentShare': 0,
    'tom_observations.ObservationRecord': 0,
    'tom_dataproducts.DataProduct': 0,
    'tom_dataproducts.ReducedDatum': 0,
    'guardian.UserObjectPermission': 0,
    'guardian.GroupObjectPermission': 0,
    'admin.LogEntry': 0,
    'django_comments.Comment': 0,
}


def preflight():
    """Check that the database still looks the way it did when this script was written.

    Collects every problem rather than stopping at the first, and every problem names the pk or relation involved.

    Returns:
        A pair (failures, rows): a list of failure strings (empty when every check passed) and a list of
        (kind, pk, name, detail) rows for the report table (42 rows when nothing is missing).
    """
    failures = []
    rows = []
    run_pks = sorted(EXPECTED_RUNS)
    event_pks = sorted(EXPECTED_EVENTS)
    list_pks = sorted(TARGET_LIST_NAMES)
    demo_pk, demo_name, demo_type = DEMO_TARGET
    kept_pk, kept_name = KEPT_TARGET_LIST
    expected_event_run = {pk: run_pk for pk, (run_pk, _url) in EXPECTED_EVENTS.items()}

    # (a) the real campaign exists.
    kept = TargetList.objects.filter(pk=kept_pk).first()
    if kept is None or kept.name != kept_name:
        failures.append(
            f'target list {kept_pk}: expected the kept campaign {kept_name!r}, found {kept and kept.name!r}'
        )

    # (b) the three lists exist with their literal names and members.
    lists = TargetList.objects.in_bulk(list_pks)
    for pk in list_pks:
        target_list = lists.get(pk)
        if target_list is None:
            failures.append(f'target list {pk}: is missing')
            rows.append(('target list', pk, 'MISSING', ''))
            continue
        if target_list.name != TARGET_LIST_NAMES[pk]:
            failures.append(f'target list {pk}: name is {target_list.name!r}, expected {TARGET_LIST_NAMES[pk]!r}')
        members = sorted(target_list.targets.values_list('pk', flat=True))
        if members != sorted(TARGET_LIST_MEMBERS[pk]):
            failures.append(f'target list {pk}: members are {members}, expected {sorted(TARGET_LIST_MEMBERS[pk])}')
        list_run_pks = [run_pk for run_pk in run_pks if EXPECTED_RUNS[run_pk][0] == pk]
        rows.append(('target list', pk, target_list.name, f'runs {list_run_pks}; members {members}'))

    # (c) the runs.
    runs = CampaignRun.objects.in_bulk(run_pks)
    for pk in run_pks:
        run = runs.get(pk)
        campaign_id, contact_person, source, target_id = EXPECTED_RUNS[pk]
        if run is None:
            failures.append(f'run {pk}: is missing')
            rows.append(('run', pk, 'MISSING', ''))
            continue
        actual = (run.campaign_id, run.contact_person, run.source, run.target_id)
        if actual != EXPECTED_RUNS[pk]:
            failures.append(
                f'run {pk}: (campaign, contact_person, source, target) is {actual}, expected {EXPECTED_RUNS[pk]}'
            )
        run_event_pks = [event_pk for event_pk in event_pks if expected_event_run[event_pk] == pk]
        rows.append(
            (
                'run',
                pk,
                run.contact_person,
                f'{run.source}; list {run.campaign_id}; target {run.target_id}; events {run_event_pks}',
            )
        )
    in_lists = set(CampaignRun.objects.filter(campaign_id__in=list_pks).values_list('pk', flat=True))
    if in_lists != set(run_pks):
        failures.append(
            f'runs in the three lists are {sorted(in_lists)}, expected {run_pks} '
            f'(an extra run would also block the list delete, which is PROTECT)'
        )
    on_demo = set(CampaignRun.objects.filter(target_id=demo_pk).values_list('pk', flat=True))
    expected_on_demo = {pk for pk in run_pks if EXPECTED_RUNS[pk][3] == demo_pk}
    if on_demo != expected_on_demo:
        failures.append(f'runs pointing at target {demo_pk} are {sorted(on_demo)}, expected {sorted(expected_on_demo)}')

    # (d) relations off the runs: nothing but the companion rows, and the companion rows are what we think.
    for relation in CampaignRun._meta.related_objects:
        accessor = relation.get_accessor_name()
        if accessor == 'calendar_event_metas':
            continue
        count = relation.related_model._default_manager.filter(**{f'{relation.field.name}__in': run_pks}).count()
        if count:
            failures.append(f'{count} {accessor} row(s) hang off the runs and would be cascaded silently')
    metas = list(CalendarEventMeta.objects.filter(run_id__in=run_pks))
    actual_event_run = {meta.event_id: meta.run_id for meta in metas}
    if actual_event_run != expected_event_run:
        failures.append(
            f'companion rows (event -> run) are {sorted(actual_event_run.items())}, '
            f'expected {sorted(expected_event_run.items())}'
        )
    for meta in metas:
        if (
            meta.confirmed_by_id is not None
            or meta.observation_record_id is not None
            or meta.observation_group_id is not None
        ):
            failures.append(
                f'event {meta.event_id}: companion row is confirmed or linked '
                f'(confirmed_by={meta.confirmed_by_id}, observation_record={meta.observation_record_id}, '
                f'observation_group={meta.observation_group_id})'
            )

    # (e) the events.
    events = CalendarEvent.objects.in_bulk(event_pks)
    for pk in event_pks:
        run_pk, url = EXPECTED_EVENTS[pk]
        event = events.get(pk)
        if event is None:
            failures.append(f'event {pk}: is missing')
            rows.append(('event', pk, 'MISSING', ''))
            continue
        if event.url != url:
            failures.append(f'event {pk}: url is {event.url!r}, expected {url!r}')
        if event.target_list_id != EXPECTED_RUNS[run_pk][0]:
            failures.append(
                f'event {pk}: target_list_id is {event.target_list_id}, expected {EXPECTED_RUNS[run_pk][0]}'
            )
        rows.append(('event', pk, event.title, f'{event.url}; run {run_pk}'))
    for relation in CalendarEvent._meta.related_objects:
        accessor = relation.get_accessor_name()
        if accessor == 'telescope_label_meta':
            continue
        count = relation.related_model._default_manager.filter(**{f'{relation.field.name}__in': event_pks}).count()
        if count:
            failures.append(f'{count} {accessor} row(s) hang off the events and would be cascaded silently')
    in_list_events = set(CalendarEvent.objects.filter(target_list_id__in=list_pks).values_list('pk', flat=True))
    if in_list_events != set(event_pks):
        failures.append(f'events in the three lists are {sorted(in_list_events)}, expected {event_pks}')

    # (f) the run pre_delete receiver's reach: each run's RUN:/ALLOC: namespace holds exactly its expected events.
    for run_pk in run_pks:
        namespace = (
            CalendarEvent.objects.filter(url=f'RUN:{run_pk}')
            | CalendarEvent.objects.filter(url__startswith=f'RUN:{run_pk}:')
            | CalendarEvent.objects.filter(url__startswith=f'ALLOC:{run_pk}:')
        )
        found = sorted(namespace.values_list('pk', flat=True))
        wanted = [pk for pk in event_pks if expected_event_run[pk] == run_pk]
        if found != wanted:
            failures.append(f'run {run_pk}: its RUN:/ALLOC: namespace holds events {found}, expected {wanted}')

    # (g) the demo target.
    target = Target.objects.filter(pk=demo_pk).first()
    if target is None:
        failures.append(f'target {demo_pk}: is missing')
        rows.append(('target', demo_pk, 'MISSING', ''))
    else:
        if target.name != demo_name or target.type != demo_type:
            failures.append(
                f'target {demo_pk}: (name, type) is {(target.name, target.type)}, expected {(demo_name, demo_type)}'
            )
        target_lists = sorted(target.targetlist_set.values_list('pk', flat=True))
        if target_lists != [4]:
            failures.append(f'target {demo_pk}: is in lists {target_lists}, expected [4]')
        alias_names = sorted(target.aliases.values_list('name', flat=True))
        extra_count = target.targetextra_set.count()
        rows.append(
            (
                'target',
                demo_pk,
                target.name,
                f'{target.type}; aliases {alias_names}; {extra_count} extras; lists {target_lists}',
            )
        )
        for relation in Target._meta.related_objects:
            if relation.many_to_many:
                continue
            accessor = relation.get_accessor_name()
            if accessor == 'campaign_runs':
                continue
            count = relation.related_model._default_manager.filter(**{relation.field.name: demo_pk}).count()
            if count != DEMO_TARGET_CASCADES.get(accessor, 0):
                failures.append(
                    f'target {demo_pk}: {count} {accessor} row(s), expected {DEMO_TARGET_CASCADES.get(accessor, 0)}'
                )

    # (h) generic references (admin log, comments, object permissions) to anything in scope.
    scope = (
        (CalendarEvent, event_pks),
        (CalendarEventMeta, event_pks),
        (CampaignRun, run_pks),
        (Target, [demo_pk]),
        (TargetList, list_pks),
    )
    generic = (
        ('admin.LogEntry', 'object_id'),
        ('django_comments.Comment', 'object_pk'),
        ('guardian.UserObjectPermission', 'object_pk'),
        ('guardian.GroupObjectPermission', 'object_pk'),
    )
    for label, column in generic:
        model = apps.get_model(label)
        for scoped_model, pks in scope:
            content_type = ContentType.objects.get_for_model(scoped_model)
            count = model._default_manager.filter(
                content_type=content_type, **{f'{column}__in': [str(pk) for pk in pks]}
            ).count()
            if count:
                failures.append(f'{count} {label} row(s) reference {scoped_model._meta.label} objects in scope')

    # (i) no orphan object permissions: deleting the target makes tom_targets delete every one in the database.
    for label in ('guardian.UserObjectPermission', 'guardian.GroupObjectPermission'):
        orphans = [p.pk for p in apps.get_model(label)._default_manager.all() if p.content_object is None]
        if orphans:
            failures.append(
                f'{len(orphans)} orphan {label} row(s) exist (pks {orphans[:10]}); deleting target {demo_pk} '
                f'would make tom_targets delete every orphan object permission in the database'
            )

    # (j) 261002-dsa must have been applied first.
    if (
        CalendarEvent.objects.filter(pk=DSA_STRAY_EVENT_PK).exists()
        or CalendarEventMeta.objects.filter(event_id=DSA_STRAY_EVENT_PK).exists()
    ):
        failures.append(
            f'event {DSA_STRAY_EVENT_PK} (or its companion row) still exists: apply '
            f'.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py first '
            f'(261002-dsa), or deleting run 68 would null its companion row'
        )

    # (k) the kept list is referenced by nothing in scope.
    if any(EXPECTED_RUNS[pk][0] == kept_pk for pk in run_pks):
        failures.append(f'an expected run belongs to the kept target list {kept_pk}')
    if CalendarEvent.objects.filter(pk__in=event_pks, target_list_id=kept_pk).exists():
        failures.append(f'an expected event belongs to the kept target list {kept_pk}')
    if target is not None and target.targetlist_set.filter(pk=kept_pk).exists():
        failures.append(f'target {demo_pk} is a member of the kept target list {kept_pk}')

    return failures, rows


def table_counts():
    """Count the rows of every watched model.

    Returns:
        A dict from model label to its whole-table row count, for every key of EXPECTED_TABLE_CHANGES.
    """
    return {label: apps.get_model(label)._default_manager.count() for label in EXPECTED_TABLE_CHANGES}


def kept_list_snapshot():
    """Snapshot the kept target list so a real run can prove it left it untouched.

    Returns:
        A tuple (name, run pks, event pks, member pks) for the kept target list, the pk lists sorted.
    """
    kept_pk = KEPT_TARGET_LIST[0]
    target_list = TargetList.objects.filter(pk=kept_pk).first()
    return (
        target_list.name if target_list else None,
        sorted(CampaignRun.objects.filter(campaign_id=kept_pk).values_list('pk', flat=True)),
        sorted(CalendarEvent.objects.filter(target_list_id=kept_pk).values_list('pk', flat=True)),
        sorted(target_list.targets.values_list('pk', flat=True)) if target_list else [],
    )


def _needs_review():
    """Return the sorted pks of approved runs flagged for site review (the approval-queue banner's rows)."""
    return sorted(
        CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.APPROVED, site_needs_review=True
        ).values_list('pk', flat=True)
    )


def _snapshot_line(snapshot):
    """Format a kept_list_snapshot() result as 'runs <n>, events <n>, targets <n>'."""
    return f'runs {len(snapshot[1])}, events {len(snapshot[2])}, targets {len(snapshot[3])}'


def main():
    """Run pre-flight, then (unless DRY_RUN) delete everything in one transaction and verify the outcome."""
    print(f'database: {settings.DATABASES["default"]["NAME"]}')
    print('mode: DRY RUN (nothing will be written)' if DRY_RUN else 'mode: DELETE')

    failures, rows = preflight()
    print('kind | pk | name | detail')
    for kind, pk, name, detail in rows:
        print(f'{kind} | {pk} | {name} | {detail}')
    if failures:
        for failure in failures:
            print(f'PRE-FLIGHT FAILED: {failure}')
        raise SystemExit(f'{len(failures)} pre-flight problem(s) found; nothing was written.')

    kept_pk, kept_name = KEPT_TARGET_LIST
    before_kept = kept_list_snapshot()
    print('asserted: 3 target lists, 14 runs, 24 events, 1 target')
    print('will cascade: 24 CalendarEventMeta, 2 TargetName, 3 TargetExtra, 1 TargetList membership')
    print(f'261002-dsa already applied: event {DSA_STRAY_EVENT_PK} absent')
    print(f"TargetList #{kept_pk} '{kept_name}' (kept): {_snapshot_line(before_kept)}")
    review_before = _needs_review()
    print(f'approved runs needing site review: {review_before}')

    if DRY_RUN:
        print('DRY RUN: nothing deleted. Run a copy with DRY_RUN = False to delete.')
        return

    event_pks = sorted(EXPECTED_EVENTS)
    event_urls = [EXPECTED_EVENTS[pk][1] for pk in event_pks]
    run_pks = sorted(EXPECTED_RUNS)
    list_pks = sorted(TARGET_LIST_NAMES)

    with transaction.atomic():
        # Re-check inside the transaction so a change since the first check cannot slip through.
        failures, _ = preflight()
        if failures:
            raise RuntimeError(f'pre-flight failed inside the transaction, nothing deleted: {failures}')
        counts_before = table_counts()

        steps = (
            ('events', CalendarEvent.objects.filter(pk__in=event_pks, url__in=event_urls)),
            ('runs', CampaignRun.objects.filter(pk__in=run_pks)),
            ('target', Target.objects.filter(pk=DEMO_TARGET[0], name=DEMO_TARGET[1])),
            ('target_lists', TargetList.objects.filter(pk__in=list_pks)),
        )
        for step, queryset in steps:
            _, per_model = queryset.delete()
            counts = {label: n for label, n in per_model.items() if n}
            if counts != EXPECTED_DELETE_COUNTS[step]:
                raise RuntimeError(
                    f'step {step!r}: unexpected delete counts {counts}, '
                    f'expected {EXPECTED_DELETE_COUNTS[step]}; rolled back'
                )

        counts_after = table_counts()
        differences = [
            f'{label}: changed by {counts_after[label] - counts_before[label]}, expected {expected}'
            for label, expected in EXPECTED_TABLE_CHANGES.items()
            if counts_after[label] - counts_before[label] != expected
        ]
        if differences:
            raise RuntimeError(f'whole-table counts differ, rolled back: {differences}')

    print(
        'deleted: 24 events (+24 CalendarEventMeta), 14 runs, 1 target '
        '(+2 TargetName, 3 TargetExtra, 1 TargetList membership), 3 target lists'
    )
    print(f'whole-table changes as expected: {len(EXPECTED_TABLE_CHANGES)}/{len(EXPECTED_TABLE_CHANGES)} models')

    remaining = {
        'events': list(CalendarEvent.objects.filter(pk__in=event_pks).values_list('pk', flat=True)),
        'runs': list(CampaignRun.objects.filter(pk__in=run_pks).values_list('pk', flat=True)),
        'target lists': list(TargetList.objects.filter(pk__in=list_pks).values_list('pk', flat=True)),
        'target': list(Target.objects.filter(pk=DEMO_TARGET[0]).values_list('pk', flat=True)),
    }
    if any(remaining.values()):
        raise SystemExit(f'rows still present after delete: {remaining}')

    after_kept = kept_list_snapshot()
    if after_kept != before_kept:
        raise SystemExit(f'TargetList #{kept_pk} changed: before {before_kept}, after {after_kept}')
    print(f'TargetList #{kept_pk} unchanged: {_snapshot_line(after_kept)}')
    print(f'approved runs needing site review: {review_before} -> {_needs_review()}')


main()
