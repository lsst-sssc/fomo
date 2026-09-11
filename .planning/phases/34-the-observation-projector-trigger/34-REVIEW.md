---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T00:00:00Z
depth: deep
scope: incremental (changes since commit 2239d0a — gap-closure plans 34-05 / 34-06, gap G-34-2)
files_reviewed: 4
files_reviewed_list:
  - solsys_code/calendar_utils.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_observation_projector_signals.py
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
findings:
  critical: 3
  warning: 11
  info: 0
  total: 14
status: issues_found
---

# Phase 34: Code Review Report (incremental — G-34-2 gap closure)

**Reviewed:** 2026-09-11
**Depth:** deep
**Files Reviewed:** 4
**Status:** issues_found

## Summary

This incremental review covers the four files changed by plans 34-05 and 34-06 since the
previously-committed review (2239d0a). The substantive change is `coerce_schedule_datetime()`
in `solsys_code/calendar_utils.py` plus `record_time_window()`'s both-populated branch routing
through it, the new unit/signal tests, and the demo notebook's `FOMO_DATABASE_PATH` scratch-copy
guard and SCHED-06 baseline-write guard.

Verified independently during this review:

- `ruff check` / `ruff format --check` are clean on all three Python files.
- `python manage.py test solsys_code.tests.test_calendar_utils solsys_code.tests.test_observation_projector_signals`
  → 73 tests, OK.
- The raise-vs-return-None contract **does** hold across the cross-module call chain:
  `record_time_window()` → `event_fields_for()` (`observation_projector.py:284`) →
  `project_record()`'s `except Exception` (`observation_projector.py:371-373`) → `'unprojectable'`,
  with `receiver_on_record_save()` (`observation_projector.py:593-600`) as a second layer. A
  raised `ValueError` cannot abort the triggering `save()`.
- `settings.TIME_ZONE = 'UTC'` / `USE_TZ = True` (`src/fomo/settings.py:172,178`), so the
  docstring's naive-string parity argument between the receiver and Django's own persistence is
  correct.

What the change does **not** hold up under is the "both paths produce the same window" invariant it
claims. Three defects block: the coercion never normalises an aware non-UTC value to UTC despite
its name, its `Returns:` contract, and the literal `Window (UTC):` label it feeds (CR-01); the
sibling parsing branch in the same function still uses `datetime.fromisoformat()`, which rejects
the `Z`-suffixed `parameters['start']` values that real records in this repo's own dev database
carry, on Python 3.10 — a runtime CI tests (CR-02); and the notebook's new scratch-copy guard
publishes the scratch copy's live post-sweep state under the "SCHED-06 baseline" heading, so the
committed evidence artifact now contradicts the committed baseline JSON it is supposed to be
diffed against (CR-03).

## Narrative Findings (AI reviewer)

### Critical Issues

#### CR-01: `coerce_schedule_datetime()` never converts an aware non-UTC value to UTC, so the event description renders the wrong wall clock under a literal "Window (UTC)" label

**File:** `solsys_code/calendar_utils.py:504-506` (behaviour), `:460,478-483` (contract),
consumed at `solsys_code/observation_projector.py:288-292`

**Issue:** The function is named `coerce_schedule_datetime(... ) -> "aware UTC datetime"` and its
`Returns:` block opens with *"otherwise a timezone-aware **UTC** datetime"* — then contradicts
itself two sentences later with *"an already-aware datetime is returned as-is"*. The body does the
latter:

```python
if value.tzinfo is None:
    return value.replace(tzinfo=dt_timezone.utc)
return value          # <- any non-UTC offset survives untouched
```

`test_aware_datetime_is_returned_with_value_and_tzinfo_unchanged`
(`test_calendar_utils.py:459-464`) and `test_non_utc_offset_form_equals_the_same_instant`
(`:448-451`) deliberately pin this pass-through, so it is intentional behaviour, not an oversight
in one branch. Verified live:

```
>>> coerce_schedule_datetime('2026-09-18T03:14:00-04:00')
datetime.datetime(2026, 9, 18, 3, 14, tzinfo=timezone(timedelta(days=-1, seconds=72000)))
```

That value flows straight into `observation_projector.py:288-292`:

```python
f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime("%Y-%m-%dT%H:%M:%S")}'
```

`strftime` on an aware datetime renders its **local** fields, not UTC. The description therefore
reads `Window (UTC): 2026-09-18T03:14:00` for an instant that is actually `07:14:00Z` — a
four-hour lie in operator-facing text, while `CalendarEvent.start_time` holds the correct instant.

Worse, this defeats the exact invariant the change was written to establish. The sweep re-fetches
the record and gets a UTC-normalised datetime from the ORM, so it renders `07:14:00`; the receiver
rendered `03:14:00`. `_update_or_unchanged()` (`calendar_utils.py:564`) compares `description` by
string equality, so every alternation of receiver-save and sweep flips the description back and
forth — *"the receiver and the sweep would write different spans to the same event forever"* is
precisely what the docstring promises this change prevents.

Reachability: the LCO/SOAR portal returns `Z`, so today's production path is UTC-only. But
`BaseRoboticObservationFacility.update_observation_status()` (`tom_observations/facility.py:555-565`)
assigns whatever any facility's `get_observation_status()` returns, the helper advertises itself as
handling a schedule value "whatever shape it arrives in", and a test explicitly exercises the
`-04:00` form. The contract is the bug, and it is one line to fix.

**Fix:**

```python
    if value.tzinfo is None:
        return value.replace(tzinfo=dt_timezone.utc)
    return value.astimezone(dt_timezone.utc)
```

and amend the `Returns:` block to drop *"an already-aware datetime is returned as-is"*. Update
`test_aware_datetime_is_returned_with_value_and_tzinfo_unchanged` to assert the instant is
preserved and `result.tzinfo is dt_timezone.utc` (rename it accordingly), and add a regression test
that `event_fields_for()`'s description renders `07:14:00` for a `-04:00` input.

---

#### CR-02: `record_time_window()`'s parameters branch uses `datetime.fromisoformat()`, which rejects the `Z`-suffixed values real records carry, on Python 3.10

**File:** `solsys_code/calendar_utils.py:539-540` (also duplicated at
`solsys_code/observation_projector.py:281-282`)

**Issue:** The branch left untouched by this change is:

```python
start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
```

Two defects, both avoidable with the very parser 34-05 just introduced in the sibling branch:

1. **`Z` is not accepted before Python 3.11.** `datetime.fromisoformat()` only gained `Z`/general
   ISO-8601 support in 3.11; CLAUDE.md and `.github/workflows/` declare Python **3.10**–3.12 as
   supported and CI-tested. I read this repo's own developer database read-only and confirmed real
   `parameters['start']` values in both shapes:

   ```
   trailing chars of parameters["start"]: {'Z', '0'}
   4282342 '2026-07-20T00:00:00Z' '2026-07-20T23:59:59Z'
   ```

   `backfill_lco_observations.py:280` writes the portal's raw window string through verbatim
   (`parameters['start'] = windows[0]['start']`), so this is the normal ingest path, not a corner
   case. On Python 3.10 every such record raises `ValueError` → `project_record()` logs
   `unprojectable` → the calendar event is never drawn or is left stale. That is the same
   user-visible failure mode as G-34-2 itself, just on a different runtime, and this change walked
   past it.

2. **`.replace(tzinfo=utc)` clobbers a real offset.** For any offset-bearing value it rewrites the
   instant rather than converting it (`'…T03:14:00-04:00'` becomes `03:14Z`, four hours wrong). The
   inline comment justifies this with "parameters['start']/['end'] are naive ISO strings
   (Pitfall 3)" — the dev-DB sample above shows that premise is already false.

**Fix:** route both branches through the same parser, and convert rather than overwrite:

```python
    if record.scheduled_start is None and record.scheduled_end is None:
        # parameters['start']/['end'] are usually naive ISO strings, but the portal-sourced
        # ingest path (backfill_lco_observations) stores 'Z'-suffixed ones; a naive value is
        # read as UTC, an offset-bearing one is converted.
        start_time = coerce_schedule_datetime(record.parameters['start'])
        end_time = coerce_schedule_datetime(record.parameters['end'])
```

Note the `Raises: KeyError` contract is preserved (the `record.parameters[...]` lookups still
raise), and `ValueError` is still raised for an unparseable value. Apply the same change to
`observation_projector.py:281-282`. Add a test with `parameters={'start': '2026-07-20T00:00:00Z', …}`
— today's fixtures only ever use `.isoformat()` output (`+00:00`), which is why this is uncaught.

---

#### CR-03: the notebook's SCHED-06 section publishes scratch-copy state under the "SCHED-06 baseline" heading, contradicting the committed baseline JSON it exists to be diffed against

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — SCHED-06 baseline
cell (the `if SCRATCH_DB_OVERRIDE is None: … else: …` block) and the per-record table cell that
follows it

**Issue:** The new `else` branch correctly refrains from *writing* the baseline file, but it still
builds `baseline_records`/`stage_tally` from the **scratch copy's live rows** — which, by the time
this cell runs, have already been mutated by the takeover sweep cells above it. It then prints that
data under the heading *"SCHED-06 baseline: the pending KEY2026B-004 records (D-20)"*, labelled
with the *committed file's* `captured_at`. The committed notebook and the committed JSON now
disagree about the same nominal thing:

| | committed notebook output | committed `…sched06-baseline.json` |
|---|---|---|
| record count | 52 | 74 |
| stage tally | `{'placed': 19, 'queued': 33}` | `{'placed': 18, 'queued': 56}` |
| `4378026` | `placed`, `2026-09-13T04:14:00 → 04:30:50` | `queued`, `2026-09-13T04:14:00 → 2026-09-14T09:14:00` |
| `4378029` | `placed`, `2026-09-18T07:14:00 → 07:30:50` | `queued`, `2026-09-18T07:14:00 → 2026-09-19T12:14:00` |

Both are stamped `captured_at='2026-09-11T04:44:59.526430+00:00'`. The published notebook table is
already-narrowed post-sweep state presented as the pre-`updatestatus` baseline — which is the exact
opposite of what the surrounding prose claims ("the evidence spike 004's PARTIAL verdict needs to
close … with no sweep involved"). UAT Test 4's documented procedure (diff the table against the
JSON) now produces a diff on every record for reasons unrelated to real observing nights, so the
evidence is not merely stale, it is actively misleading. CLAUDE.md treats the paired notebook as
part of the deliverable, not optional polish.

The inline comment asserting *"Both branches build `baseline_records`/`captured_at`/`stage_tally`
identically so the next cell's per-table works untouched either way"* is the root of the mistake:
the two branches share a code path but not a meaning.

**Fix:** on a scratch-routed run, render the **committed file's** records, not the copy's, and say
so in the heading of the output:

```python
if SCRATCH_DB_OVERRIDE is None:
    captured_at = timezone.now().isoformat()
    baseline_payload = {...}
    with open(SCHED06_BASELINE_PATH, 'w') as fh:
        json.dump(baseline_payload, fh, indent=2)
    ...
else:
    with open(SCHED06_BASELINE_PATH) as fh:
        existing_baseline = json.load(fh)
    captured_at = existing_baseline['captured_at']
    scratch_records, baseline_records = baseline_records, existing_baseline['records']
    print('Routed to a scratch copy -- showing the COMMITTED baseline below, not this copy.')
    print(f'(the scratch copy currently holds {len(scratch_records)} pending records, '
          f'already mutated by the sweep cells above; they are deliberately not shown)')

stage_tally = Counter(rec['stage'] for rec in baseline_records.values())
```

(Recomputing `stage_tally` after the branch keeps it consistent with whichever `baseline_records`
was selected — it is currently computed *before* the branch, which is why the scratch tally leaks
out.) Then re-execute and re-commit the notebook.

---

### Warnings

#### WR-01: the diagnostic `ValueError` message `coerce_schedule_datetime()` builds is never logged anywhere

**File:** `solsys_code/calendar_utils.py:500,503`; consumed at
`solsys_code/observation_projector.py:372` and `:489`

**Issue:** The `Raises:` docstring justifies raising over returning `None` with *"Raising instead
keeps the record a D-13 unprojectable one — **visible in the log** and the sweep counters"*, and
`test_unparseable_string_raises_value_error` (`test_calendar_utils.py:477-481`) asserts the message
names the rejected value *"so the message stays diagnostic"*. But both production catch sites log
only the exception **class name**:

```python
logger.warning('unprojectable observation_id=%r: %s', record.observation_id, type(exc).__name__)
```

An operator debugging a stale event sees `unprojectable observation_id='4378026': ValueError` and
nothing about which field, which value, or that a schedule string was the cause — indistinguishable
from the `parameters['start']` `ValueError` and from a `title_for()` failure. The carefully-worded
message is dead weight.

**Fix:** include the message at the two catch sites (it contains only a timestamp string, no
credentials — unlike `resolve_placement_block()`'s deliberately-silent handler, which must stay as
it is):

```python
logger.warning('unprojectable observation_id=%r: %s: %s', record.observation_id, type(exc).__name__, exc)
```

---

#### WR-02: `coerce_schedule_datetime()` duplicates `_parse_datetime_value()` with a silently different contract

**File:** `solsys_code/calendar_utils.py:460-506` vs
`solsys_code/management/commands/backfill_lco_observations.py:81-107`

**Issue:** `_parse_datetime_value()` already existed in this package and does the same job —
"parse a portal ISO-8601 value into an aware UTC datetime, `parse_datetime()` first, attach UTC when
naive". The new helper is a second implementation whose contract diverges in three ways with no
cross-reference in either docstring:

| | `_parse_datetime_value` | `coerce_schedule_datetime` |
|---|---|---|
| unparseable | returns `None` | raises `ValueError` |
| falsy (`''`, `0`) | returns `None` | `''` raises; `0` raises |
| bare ISO date | explicit `date.fromisoformat` fallback | accidental (see WR-03) |

`calendar_utils.py`'s own module docstring states its purpose is so consumers *"can share a single
implementation"* — this change adds a near-twin instead. The divergent failure contracts are the
real hazard: a future reader who moves a call from one to the other silently swaps "degrade to
`None`" for "abort and mark unprojectable".

**Fix:** either have `_parse_datetime_value()` delegate
(`try: return coerce_schedule_datetime(value) except ValueError: return None`, with its own
date-only fallback retained), or at minimum add a "see also" note to both docstrings naming the
other and why the failure contracts differ.

---

#### WR-03: a bare ISO date string is silently accepted as midnight, and the docstring documents neither the acceptance nor the risk

**File:** `solsys_code/calendar_utils.py:498-501`

**Issue:** Django 5.2's `parse_datetime()` delegates to `datetime.fromisoformat()`, which accepts a
date-only string. Verified:

```
>>> coerce_schedule_datetime('2026-09-18')
datetime.datetime(2026, 9, 18, 0, 0, tzinfo=datetime.timezone.utc)
```

The `Args:` block says the value is "a portal ISO-8601 `str`" and the `Raises:` block covers only
"cannot parse". A facility returning a date-only `scheduled_start`/`scheduled_end` therefore draws a
zero-length or midnight-anchored block on the calendar instead of being routed to `unprojectable`,
which is the outcome the "never returns `None` for an unusable value" reasoning explicitly exists to
avoid. Since this is undocumented and untested, it is also indistinguishable from a regression if
Django's parser behaviour shifts.

**Fix:** decide and pin it. Either document + test it (matching `_parse_datetime_value`'s explicit,
deliberate date-only support), or reject it:

```python
        parsed = parse_datetime(value)
        if parsed is None:
            raise ValueError(f'Unparseable schedule datetime string: {value!r}')
        if len(value) <= 10:  # bare ISO date -- not a block boundary
            raise ValueError(f'Schedule value is a date, not a datetime: {value!r}')
```

---

#### WR-04: the non-`str`/non-`datetime` raise branch and the end-to-end `unprojectable` contract are both untested

**File:** `solsys_code/tests/test_calendar_utils.py:434-481`

**Issue:** `TestCoerceScheduleDatetime` covers eight cases but not `elif not isinstance(value, datetime)`
(`calendar_utils.py:502-503`) — an entire raise branch of new code with zero coverage. More
importantly, the `Raises:` docstring's whole justification ("keeps the record a D-13 unprojectable
one … with the record's own save never aborted and its existing event left untouched") is asserted
only in prose. Nothing tests that an unparseable `scheduled_start` on a real save produces
`('unprojectable', 'ValueError')`, leaves `CalendarEvent` untouched, and does not raise — even
though `test_observation_projector_signals.py` already has all the fixtures to do it.

**Fix:** add to `TestCoerceScheduleDatetime`:

```python
    def test_non_string_non_datetime_value_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Unusable schedule datetime value'):
            coerce_schedule_datetime(1758000000)
        with self.assertRaisesRegex(ValueError, 'Unusable schedule datetime value'):
            coerce_schedule_datetime(date(2026, 9, 18))
```

and to `TestUpdateObservationStatusPath`, a facility fake returning
`'scheduled_start': 'not-a-timestamp'` asserting the save does not raise, the event's span is
unchanged from the `[Q]` window, and `'unprojectable'` is logged.

---

#### WR-05: `record_time_window()`'s return annotation is inconsistent with the helper it now calls

**File:** `solsys_code/calendar_utils.py:509,542-543`

**Issue:** `record_time_window(...) -> tuple[datetime, datetime]` now assigns from
`coerce_schedule_datetime(...) -> datetime | None`. It is correct at runtime (the `elif` guard has
already proven both fields non-`None`), but the annotations no longer agree, so any future type
check on this module reports a false positive here and a reader cannot tell whether `None` is
reachable.

**Fix:** either add an overload/assert, or narrow at the call site:

```python
    elif record.scheduled_start is not None and record.scheduled_end is not None:
        # both fields are non-None here, so the coercion cannot return None
        start_time = cast(datetime, coerce_schedule_datetime(record.scheduled_start))
        end_time = cast(datetime, coerce_schedule_datetime(record.scheduled_end))
```

---

#### WR-06: the notebook reads `existing_baseline` but discards its `records`, and two comments/prose blocks now state the opposite of what the code does

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — SCHED-06 baseline
cell and the "What happens next" markdown cell

**Issue:** Three related quality defects around CR-03:

1. `existing_baseline = json.load(fh)` is read, then only `existing_baseline['captured_at']` is
   used — `['records']` and `['record_count']` are loaded and thrown away. Dead read.
2. The inline comment claims *"Both branches build `baseline_records`/`captured_at`/`stage_tally`
   identically"* — they do not (see CR-03); `captured_at` means "now" in one branch and "months
   ago" in the other, and `baseline_records` is a different corpus.
3. "What happens next" still reads *"the SCHED-06 baseline cell above **overwrites**
   `…sched06-baseline.json` in place when it re-runs, so compare … with `git diff` on that file
   **(which this run has just overwritten)**"*. For the committed run it did *not* overwrite it.
   The appended paragraph mentions the exception several sentences later, but the parenthetical
   above it is now plainly false in the artifact a reader is looking at.

**Fix:** apply CR-03's fix (which consumes `existing_baseline['records']`), delete the "identically"
claim, and change the parenthetical to *"(which an un-routed run overwrites; a scratch-routed run
leaves it alone — see below)"*.

---

#### WR-07: the scratch branch opens the baseline JSON with no existence guard, unlike every other precondition in the notebook

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — SCHED-06 baseline
cell, `with open(SCHED06_BASELINE_PATH) as fh:`

**Issue:** Every other precondition in this notebook fails with an explanatory `RuntimeError`
(`No manage.py at …`, `No developer database at … run 'python manage.py migrate' first`). This one
raises a bare `FileNotFoundError` naming an absolute path with no hint that the file is a committed
artifact, aborting the notebook mid-execution and leaving the reader to guess. The file *is*
currently committed (`a87f5f8`), so this is latent rather than live — but it is the one path that
only triggers for someone deliberately following the new scratch-copy instructions.

**Fix:**

```python
    if not SCHED06_BASELINE_PATH.exists():
        raise RuntimeError(
            f'No committed SCHED-06 baseline at {SCHED06_BASELINE_PATH}. A scratch-routed run reads '
            f'the baseline but never writes it -- run this notebook once un-routed first.'
        )
```

---

#### WR-08: the "don't write to the developer database" guard can be satisfied by a relative path that still opens the developer database

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — Django setup cell

**Issue:** The guard is

```python
assert resolved_db_name == SCRATCH_DB_OVERRIDE and resolved_db_name != str(dev_db_path), ...
```

`resolved_db_name` is `os.getenv('FOMO_DATABASE_PATH')` verbatim (`settings.py:134`), so the first
conjunct is a tautology whenever the variable is set; the only real check is the string
inequality against an absolute `dev_db_path`. A relative override such as
`FOMO_DATABASE_PATH=../../src/fomo_db.sqlite3` passes the guard (different string) while SQLite
opens the developer database itself — and the very next cells run a **non-transactional, real
takeover sweep** against it. The comment says "An absolute `FOMO_DATABASE_PATH` override", but
nothing enforces it. Secondarily, `assert` is stripped under `python -O`, which is a poor mechanism
for a data-safety guard even in a notebook.

**Fix:** compare resolved paths, and raise rather than assert:

```python
    resolved_override = Path(resolved_db_name).resolve()
    if resolved_override == dev_db_path.resolve():
        raise RuntimeError(
            f'FOMO_DATABASE_PATH={SCRATCH_DB_OVERRIDE!r} resolves to the developer database '
            f'({dev_db_path}); point it at a scratch copy.'
        )
```

---

#### WR-09: `SCRATCH_DB_OVERRIDE` is a cross-cell global consumed eleven cells later

**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — Django setup cell
(definition) and SCHED-06 baseline cell (use)

**Issue:** The SCHED-06 cell's branch — the one thing standing between a scratch run and clobbering
the committed evidence file — depends on a name bound in the first code cell. Re-running just the
SCHED-06 cell in a fresh kernel raises `NameError`; re-running it in a kernel where the setup cell
ran *without* the override set, after the operator exported it in a terminal, silently takes the
**write** branch. Given the failure mode is "destroy the evidence artifact", the guard should be
self-contained.

**Fix:** re-read the environment in the SCHED-06 cell rather than relying on the earlier binding:

```python
SCRATCH_DB_OVERRIDE = os.environ.get('FOMO_DATABASE_PATH') or None  # re-read: this guard is self-contained
```

---

#### WR-10: `test_in_memory_instance_with_portal_iso_strings_returns_aware_utc_pair` does not test an in-memory instance

**File:** `solsys_code/tests/test_calendar_utils.py:533-554`

**Issue:** The test's name and docstring say "the post-save-instance case, **not a database row**",
but it uses `ObservationRecord.objects.create(...)` — a real INSERT that also fires the projector's
`post_save` receiver (which then fails with `InstrumentExtractionError` and is swallowed, since
`parameters={'proposal': 'TEST'}` carries no instrument signal). The test only passes because Django
happens not to refresh assigned field values back from the database after `save()`; that is an
implementation detail, and the incidental signal traffic is noise the assertion does not control.

**Fix:** construct the instance without touching the database, which is exactly the contract the
test claims to pin:

```python
        record = ObservationRecord(
            target=self.target,
            user=self.user,
            facility='LCO',
            observation_id='555555',
            status='COMPLETED',
            parameters={'proposal': 'TEST'},
            scheduled_start=start.isoformat().replace('+00:00', 'Z'),
            scheduled_end=end.isoformat().replace('+00:00', 'Z'),
        )
```

(The class can then stay a `TestCase` for its other methods, or this method can move to the
`SimpleTestCase` above it.)

---

#### WR-11: the signal tests monkeypatch a shared class attribute instead of using `patch.object`, and triplicate the same fixture

**File:** `solsys_code/tests/test_observation_projector_signals.py:94-108, 124-134, 149-162`

**Issue:** All three tests in `TestUpdateObservationStatusPath` do:

```python
original_get_status = LCOFacility.get_observation_status
...
LCOFacility.get_observation_status = fake_get_observation_status
try:
    ...
finally:
    LCOFacility.get_observation_status = original_get_status
```

This mutates a class shared process-wide and restores it by hand. The module already imports
`unittest.mock.patch`, and every other patch in this file (`:315`, `:322`, `:327`, `:383`) uses the
context-manager form; this is the only place that hand-rolls it. The two new tests added by 34-05/34-06
copied the pattern rather than fixing it, so the near-identical 12-line fixture now appears three
times, differing only in the block times and whether the fake returns strings or datetimes. A
restore missed by a future edit leaks a fake `get_observation_status` into every subsequent test in
the process — a silent, order-dependent failure.

**Fix:** use `patch.object` and factor the fake into a helper:

```python
    def _run_updatestatus(self, block_start, block_end, *, as_strings: bool):
        def fake(_self, _observation_id):
            fmt = (lambda d: d.isoformat().replace('+00:00', 'Z')) if as_strings else (lambda d: d)
            return {'state': 'PENDING', 'scheduled_start': fmt(block_start), 'scheduled_end': fmt(block_end)}

        with patch.object(LCOFacility, 'get_observation_status', fake):
            LCOFacility().update_observation_status(self.record.observation_id)
```

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (incremental — diff base 2239d0a)_
