"""One-off repair: retire the ten orphan calendar events of intent-review setup step 5 (quick task 261002-dsa).

Why they can go: pks 44-52 are the hand-entered 2026-07-22 Didymos NTT/Magellan nights. Setup step 4
(`load_telescope_runs Didymos_runs --campaign 'Didymos 2026'`) recreated every one of them, to the second, as the
`ALLOC:76/77/78:*` allocation nights, so each night now appears twice. pk 334 (`tmp`) is a stray. It has one
unconfirmed companion row (a CalendarEventMeta attributed to run 68), and that row is deleted along with it.
Run 68, its TargetList and its own `ALLOC:68:2025-07-04` event (pk 357) are NOT touched.

This script is for ONE database only: the pks are literals. Pre-flight checks that the database still looks the way
it did when the script was written, and writes nothing if any check fails.

How to run it (always feed the file with `<`, never a pipe; `manage.py shell` ignores a pipe that is not yet readable):

    Dry run (the committed default; prints the table and the totals, writes nothing):
        python manage.py shell < \\
            .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py

    Real run (edit a COPY, never this file):
        sed 's/^DRY_RUN = True$/DRY_RUN = False/' \\
            .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py \\
            > /tmp/retire_orphan_events_real.py
        python manage.py shell < /tmp/retire_orphan_events_real.py

When to run it: between cron ticks (after a `=== FOMO unattended run END` banner in /var/log/fomo/unattended.log) and
after taking a fresh online backup as the rollback:

    sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup 'src/fomo_db_20261002.sqlite3'"

A `database is locked` error inside the transaction means nothing was written; retry after the next END banner.
To point it at a throwaway copy instead of the live database, set FOMO_DATABASE_PATH for that one command.

Expected output of the real run (the first line is the database it is acting on; check it):

    asserted: 10 (9 superseded by ALLOC nights, 1 stray)
    companion rows that will cascade: 1 (pk 334 -> run 68)
    blank-url events in this database: 10
    deleted: 10
    cascaded: 1 CalendarEventMeta (pk 334's companion row)
    ALLOC nights unchanged: 9/9
    blank-url events in this database: 0
"""

from django.conf import settings
from django.db import transaction
from tom_calendar.models import CalendarEvent

from solsys_code.models import CalendarEventMeta

DRY_RUN = True

ORPHAN_PKS = [44, 45, 46, 47, 48, 49, 50, 51, 52, 334]

# Each hand-entered night and the ALLOC: night (from setup step 4) that supersedes it.
EXPECTED_ALLOC_URLS = {
    44: 'ALLOC:76:2026-07-09',
    45: 'ALLOC:76:2026-07-10',
    46: 'ALLOC:76:2026-07-11',
    47: 'ALLOC:76:2026-07-12',
    48: 'ALLOC:77:2026-07-17',
    49: 'ALLOC:77:2026-07-18',
    50: 'ALLOC:78:2026-07-18',
    51: 'ALLOC:78:2026-07-19',
    52: 'ALLOC:78:2026-07-20',
}

# The stray, and the one companion row (CalendarEventMeta) it is known to carry.
STRAY_PK = 334
STRAY_TITLE = 'tmp'
STRAY_META_RUN_PK = 68

# Fields an orphan may carry only if the ALLOC: night carries the same value (or the orphan leaves it empty).
CARRIED_FIELDS = ('title', 'user', 'proposal', 'telescope', 'instrument', 'target_list_id')

# The only rows the delete may touch: the ten events plus pk 334's companion row.
EXPECTED_DELETE_COUNTS = {'tom_calendar.CalendarEvent': 10, 'solsys_code.CalendarEventMeta': 1}


def preflight():
    """Check that the database still looks the way it did when this script was written.

    Collects every problem rather than stopping at the first.

    Returns:
        A pair (failures, rows): a list of failure strings (empty when every check passed) and a list of
        (pk, title, span, superseded-by) rows for the report table.
    """
    failures = []
    rows = []

    # (a) every orphan exists, (b) with a blank url.
    events = CalendarEvent.objects.in_bulk(ORPHAN_PKS)
    for pk in ORPHAN_PKS:
        event = events.get(pk)
        if event is None:
            failures.append(f'pk {pk}: event is missing')
            rows.append((pk, 'MISSING', '', ''))
            continue
        if event.url != '':
            failures.append(f'pk {pk}: url is not blank ({event.url!r})')

    # (c) companion rows: none for 44-52, and exactly the known unconfirmed run-68 row for pk 334.
    metas = list(CalendarEventMeta.objects.filter(event_id__in=ORPHAN_PKS))
    if len(metas) != 1:
        failures.append(
            f'expected exactly 1 companion row (pk {STRAY_PK}), found {len(metas)}: {[m.event_id for m in metas]}'
        )
    for meta in metas:
        if meta.event_id != STRAY_PK:
            failures.append(f'pk {meta.event_id}: unexpected companion row')
        elif (
            meta.run_id != STRAY_META_RUN_PK
            or meta.observation_record_id is not None
            or meta.observation_group_id is not None
            or meta.confirmed_by_id is not None
        ):
            failures.append(
                f'pk {meta.event_id}: companion row is not the expected unconfirmed run-{STRAY_META_RUN_PK} row '
                f'(run={meta.run_id}, observation_record={meta.observation_record_id}, '
                f'observation_group={meta.observation_group_id}, confirmed_by={meta.confirmed_by_id})'
            )

    # (d) every other reverse relation (todos, dismissals, anything added later) must be empty for the ten.
    for relation in CalendarEvent._meta.related_objects:
        accessor = relation.get_accessor_name()
        if accessor == 'telescope_label_meta':
            continue
        count = relation.related_model._default_manager.filter(**{f'{relation.field.name}__in': ORPHAN_PKS}).count()
        if count:
            failures.append(f'{count} {accessor} row(s) hang off the orphans and would be cascaded silently')

    # (e) each of 44-52 is superseded to the second by exactly one ALLOC: night that carries everything it does.
    for pk, url in EXPECTED_ALLOC_URLS.items():
        event = events.get(pk)
        if event is None:
            continue
        span = f'{event.start_time.isoformat()} -> {event.end_time.isoformat()}'
        matches = list(CalendarEvent.objects.filter(url=url))
        if len(matches) != 1 or matches[0].pk in ORPHAN_PKS:
            failures.append(f'pk {pk}: expected exactly one {url} event, found {[m.pk for m in matches]}')
            rows.append((pk, event.title, span, f'{url} (NOT FOUND)'))
            continue
        alloc = matches[0]
        rows.append((pk, event.title, span, f'{url} (event {alloc.pk})'))
        if event.start_time != alloc.start_time or event.end_time != alloc.end_time:
            failures.append(
                f'pk {pk}: span differs from {url} '
                f'({event.start_time.isoformat()} -> {event.end_time.isoformat()} vs '
                f'{alloc.start_time.isoformat()} -> {alloc.end_time.isoformat()})'
            )
        for name in CARRIED_FIELDS:
            mine, theirs = getattr(event, name), getattr(alloc, name)
            if mine not in ('', None) and mine != theirs:
                failures.append(f'pk {pk}: {name} {mine!r} is not carried by {url} ({theirs!r})')
        alloc_lines = alloc.description.splitlines()
        for line in event.description.splitlines():
            # The loader appends ' [<proposal id>]' to the source line when the run has a proposal (the NTT nights),
            # so a line is also carried when an ALLOC: line is that line plus the bracketed proposal tag.
            carried = any(a == line or a.startswith(f'{line} [') for a in alloc_lines)
            if line.strip() and not carried:
                failures.append(f'pk {pk}: description line {line!r} is not in the description of {url}')

    # (f) the stray is the stray we looked at.
    stray = events.get(STRAY_PK)
    if stray is not None:
        if stray.title != STRAY_TITLE:
            failures.append(f'pk {STRAY_PK}: title is {stray.title!r}, expected {STRAY_TITLE!r}')
        rows.append(
            (
                STRAY_PK,
                stray.title,
                f'{stray.start_time.isoformat()} -> {stray.end_time.isoformat()}',
                f'stray (companion row attributed to run {STRAY_META_RUN_PK} cascades)',
            )
        )

    rows.sort(key=lambda row: row[0])
    return failures, rows


def alloc_snapshot():
    """Snapshot the nine ALLOC: nights so a real run can prove it left them untouched.

    Returns:
        A dict from each ALLOC: url that exists to (pk, start_time, end_time, title, description, modified).
    """
    snapshot = {}
    for event in CalendarEvent.objects.filter(url__in=list(EXPECTED_ALLOC_URLS.values())):
        snapshot[event.url] = (
            event.pk,
            event.start_time,
            event.end_time,
            event.title,
            event.description,
            event.modified,
        )
    return snapshot


def main():
    """Run pre-flight, then (unless DRY_RUN) delete the ten events in one transaction and verify the outcome."""
    print(f'database: {settings.DATABASES["default"]["NAME"]}')
    print('mode: DRY RUN (nothing will be written)' if DRY_RUN else 'mode: DELETE')

    failures, rows = preflight()
    print('pk | title | UTC span | superseded by')
    for pk, title, span, superseded_by in rows:
        print(f'{pk} | {title} | {span} | {superseded_by}')
    if failures:
        for failure in failures:
            print(f'PRE-FLIGHT FAILED: {failure}')
        raise SystemExit(f'{len(failures)} pre-flight problem(s) found; nothing was written.')

    print('asserted: 10 (9 superseded by ALLOC nights, 1 stray)')
    print(f'companion rows that will cascade: 1 (pk {STRAY_PK} -> run {STRAY_META_RUN_PK})')
    print(f'blank-url events in this database: {CalendarEvent.objects.filter(url="").count()}')

    before = alloc_snapshot()
    if DRY_RUN:
        print('DRY RUN: nothing deleted. Run a copy with DRY_RUN = False to delete.')
        return

    with transaction.atomic():
        # Re-check inside the transaction so a change since the first check cannot slip through.
        failures, _ = preflight()
        if failures:
            raise RuntimeError(f'pre-flight failed inside the transaction, nothing deleted: {failures}')
        # url='' means this can never reach a RUN:/ALLOC:/facility-url event.
        _, per_model = CalendarEvent.objects.filter(pk__in=ORPHAN_PKS, url='').delete()
        counts = {label: n for label, n in per_model.items() if n}
        if counts != EXPECTED_DELETE_COUNTS:
            raise RuntimeError(f'unexpected delete counts {counts}, expected {EXPECTED_DELETE_COUNTS}; rolled back')

    print('deleted: 10')
    print("cascaded: 1 CalendarEventMeta (pk 334's companion row)")

    remaining = list(CalendarEvent.objects.filter(pk__in=ORPHAN_PKS).values_list('pk', flat=True))
    if remaining:
        raise SystemExit(f'orphans still present after delete: {remaining}')

    after = alloc_snapshot()
    differences = [url for url in EXPECTED_ALLOC_URLS.values() if before.get(url) != after.get(url)]
    if differences or len(after) != len(EXPECTED_ALLOC_URLS):
        raise SystemExit(f'ALLOC nights changed or went missing: {differences}')
    print(f'ALLOC nights unchanged: {len(after)}/{len(EXPECTED_ALLOC_URLS)}')
    print(f'blank-url events in this database: {CalendarEvent.objects.filter(url="").count()}')


main()
