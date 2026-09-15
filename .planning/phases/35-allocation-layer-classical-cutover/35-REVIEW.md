---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-15T19:01:09Z
depth: deep
iteration: 7
diff_base: 33664050abfbb37ffa72e51e996c234972d4be3e
files_reviewed: 9
files_reviewed_list:
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_load_telescope_runs.py
findings:
  critical: 1
  warning: 3
  info: 7
  total: 11
status: issues_found
---

# Phase 35: Code Review Report (iteration 7 — third gap-closure round)

**Reviewed:** 2026-09-15T19:01:09Z
**Depth:** deep
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Round 3 (plans 35-16, 35-17, 35-18) was given an explicitly bounded scope by the third
verification pass. All four scope items were checked against the real source, not against
the SUMMARY claims:

| Round-3 scope item | Verdict |
|---|---|
| (1) REVERT the half-null fallback entirely (subtractive, not another patch) | **GENUINE REVERT.** `_raise_if_set_window_inverted()` is back to a two-parameter signature; the `existing` parameter, both `existing.start_time`/`existing.end_time` fallback expressions and the WR-01 apology comment are deleted (`git diff 3366405..HEAD -- solsys_code/allocation_projector.py`). `grep -rn '_raise_if_set_window_inverted'` shows only two call sites, both two-argument (lines 750, 780). No trace of round-2 logic remains. Docstrings and both call-site comments were rewritten to claim silence, not parity. |
| (1a) Preserve round 2's one correct change — the both-null `and` short-circuit | **PRESERVED** (line 359: `if run.night_start_utc is None and run.night_end_utc is None: return`), rather than reverting all the way to the pre-35-13 `or`. Behaviourally the revert is now *exactly equivalent* to the pre-35-13 `or` form, because the surviving `if start is None or end is None: return` at line 366 subsumes it — see IN-01. |
| (2) Narrow the loader's over-claiming dry-run parity statements | **DONE and accurate.** `--dry-run` help text, the create-arm comment, the WR-02 fold comment, the runbook loader section, and the loader notebook's parity cell prose + printed conclusion are all arm-scoped now. Independently reproduced: the new `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure` asserts `dry=(1,1,0,0,0)` vs `real=(1,0,0,0,1)` and passes. |
| (3) Documentation caveat + pinning test for the staff-edit-revert (NOT a code change) | **Code untouched** (verified: the CR-01 predicate `existing_source_line != source_line` at line 445 is byte-identical to what round 2 left; only the `else:` branch's *reason string* gained a trailing clause). Caveat added to the module docstring and the runbook; pinning test `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` added and passing. **But the caveat's field enumeration is wrong — see WR-02.** |
| (4) Fix the reason-vocabulary gap | **DONE.** `_REASON_LABELS[_DUPLICATE_IDENTITY]` now states both causes; both runbook occurrences updated. The reconciler notebook's *executed output* carries the new two-cause string verbatim (cells 9 and 11), and every `iopub` timestamp in both notebooks moved from `16:52` to `18:46` — a real re-execution, not a stale cached run. |

**The new finding is the one the brief asked about: yes, there is a fourth sub-case, and
subtraction was not sufficient for it.** CR-01 below is a *real-run* data defect in the same
bug class, reproduced three ways against a real Django test database. Round 3's own new
docstring states the falsifying premise — "nulling a previously-set field leaves that
field's old operator value sitting in the stored event, not a sunset or sunrise" — and then
applies it only to the *preview*. Nobody checked whether the same fact breaks the *real*
re-mint decision. It does.

### Verification method

- Ran the three affected test modules: `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs` — **128 tests, OK**, 78.6 s.
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` — both **Passed** (D-07 gate clean).
- Wrote and executed a standalone probe against a real migrated Django test database (`DiscoverRunner.setup_databases()`), driving `reconcile_run()` end to end with a real `Observatory` (obscode 809, `America/Santiago`) and real `sun_event()` calls. Three fixtures, output quoted in CR-01.
- Read both notebooks as JSON (execution counts 1..18 / 1..16, every code cell carries output) and grepped the *executed outputs*, not the source cells, for the new reason label.
- Traced `_may_write(None, run)` → `True` (so `existing.pk` at line 639 is safe), `record_time_window()`'s raise conditions against `retired_nights()`'s guards, and `adopt_event_into_run()`'s return contract against both of its call sites.

---

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

---

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: clearing a sub-night field to null never re-mints the night — the allocation event keeps the old operator boundary forever, silently reported as `unchanged`

**File:** `solsys_code/allocation_projector.py:393-404` (`_span_needs_remint()`), consumed at `solsys_code/allocation_projector.py:736`

**Issue:**

`_span_needs_remint()` only ever compares a **set** sub-night field against the stored
boundary:

```python
if run.night_start_utc is None and run.night_end_utc is None:
    return False                                    # <-- both-null: never re-mints
night_span = _night_span_utc(run, night)
if run.night_start_utc is not None and existing.start_time != ...:   # <-- null start: not checked
    return True
if run.night_end_utc is not None and existing.end_time != ...:       # <-- null end: not checked
    return True
return False
```

This is correct for the *drift* case D-13 is actually about (a null field means "use the sun
event", and the sun event moves by fractions of a second between runs — don't rewrite for
that). It is **wrong for an operator clearing a previously-set field**, which is a real
semantic change, not drift: the stored boundary in that case is the operator's own *old*
`night_start_utc`/`night_end_utc` value, not a sunset or sunrise. Round 3's own new
`_raise_if_set_window_inverted()` docstring says exactly this — and then uses it only to
justify staying silent on the *preview*. The real path has the same hole and nothing
catches it.

The run row and the calendar it owns end up permanently disagreeing, with **no counter, no
log line and no exception** — `reconcile_run()` returns `unchanged=1`.

Reproduced three ways (real test DB, La Silla / `America/Santiago`, night 2026-07-09, true
sunset/sunrise `22:06:35` / `11:29:46` UTC):

```
=== PROBE A: set/set -> both nulled (operator reverts to "full night") ===
  after first mint:   2026-07-09 23:00:00+00:00 -> 2026-07-10 05:00:00+00:00
  dry-run result:     ReconcileResult(created=0, updated=0, unchanged=1, ... )
  real result:        ReconcileResult(created=0, updated=0, unchanged=1, ... )
  after null revert:  2026-07-09 23:00:00+00:00 -> 2026-07-10 05:00:00+00:00   <-- STALE
  true sunset/sunrise: 2026-07-09 22:06:35  2026-07-10 11:29:46

=== PROBE B: half-null (start set, end null) -> start also nulled ===
  after first mint:   2026-07-09 23:00:00+00:00 -> 2026-07-10 11:29:46+00:00
  real result:        ReconcileResult(created=0, updated=0, unchanged=1, ... )
  after null revert:  2026-07-09 23:00:00+00:00 -> 2026-07-10 11:29:46+00:00   <-- STALE start

=== PROBE C: set/set -> start nulled only, end left unchanged ===
  after first mint:   2026-07-09 23:00:00+00:00 -> 2026-07-10 05:00:00+00:00
  real result:        ReconcileResult(created=0, updated=0, unchanged=1, ... )
  after null revert:  2026-07-09 23:00:00+00:00 -> 2026-07-10 05:00:00+00:00   <-- STALE start
```

PROBE C is the nastiest shape: the *other* field is set and still matches, so the second
`if` is reached, evaluates False, and the function returns "no re-mint needed" even though
one of the two boundaries the run declares has genuinely changed.

**Reachability.** `CampaignRunAdmin` (`solsys_code/admin.py:132-166`) sets neither `fields`
nor `exclude` and lists only `approval_status` as read-only, so `night_start_utc` /
`night_end_utc` are fully editable in the Django admin. Clearing one of them *is* the
documented way to say "this run uses the whole night" (`models.py:265`, `D-04`'s null
convention). The very next `reconcile_campaign_runs` sweep then reports the run as
`unchanged` and leaves a calendar night claiming, e.g., 23:00–05:00 when the run now says
22:06–11:29. Not reachable via `load_telescope_runs` re-import, because dropping the window
token changes `_source_identifier()` and mints a *different* run — admin/API edits are the
live path.

Note this is materially worse than the WR-01 preview gap the round-3 scope accepted: this
one is the *real* run, it writes nothing and warns nothing, and there is no follow-up pass
that can ever discover it.

**Fix:** `_span_needs_remint()` must also fire when a field is null but the stored boundary
is provably *not* sun-derived. The cheapest sound version is to treat "field is null" as
re-mintable only when the stored boundary equals the sun event — which requires exactly one
`sun_event()` call, on a transition that is rare and is a genuine operator change rather
than D-13's astropy drift. Gate it so drift alone cannot trigger a rewrite:

```python
def _span_needs_remint(run: CampaignRun, night, existing: CalendarEvent) -> bool:
    night_span = _night_span_utc(run, night)
    if run.night_start_utc is not None and existing.start_time != _time_of_day_to_datetime(
        run.night_start_utc, night, night_span
    ):
        return True
    if run.night_end_utc is not None and existing.end_time != _time_of_day_to_datetime(
        run.night_end_utc, night, night_span
    ):
        return True
    # A null field means "use the sun event". D-13 forbids rewriting for drift, so only
    # re-mint when the stored boundary is not the sun event even to the nearest minute --
    # i.e. it is a stale operator value left behind when the field was cleared.
    if run.night_start_utc is None or run.night_end_utc is None:
        sunset, sunrise = sun_event(run.site, night, kind='sun')
        tol = timedelta(minutes=1)
        if run.night_start_utc is None and abs(
            existing.start_time - sunset.to_datetime(timezone=dt_timezone.utc)
        ) > tol:
            return True
        if run.night_end_utc is None and abs(
            existing.end_time - sunrise.to_datetime(timezone=dt_timezone.utc)
        ) > tol:
            return True
    return False
```

If reintroducing `sun_event()` on the update path is judged to breach D-13 outright, the
alternative is to make the transition detectable without astropy — e.g. persist the
resolved `night_start_utc`/`night_end_utc` the night was minted from (a new
`CalendarEventMeta` column, or a second structured line in the event description alongside
the existing `Dark window (-15 deg, UTC): ` line) and compare against *that*. Either way the
current silent-stale outcome must not ship, and whichever route is taken needs a test for
all three probe shapes above (set/set→null/null, half-null→null/null, set/set→half-null),
since each reaches the defect through a different branch.

---

## Warnings

### WR-01: after the revert, `--dry-run` cannot detect *any* half-null inverted span — and the half-night classical line is exactly that shape, while the operator runbook never mentions the limitation

**File:** `solsys_code/allocation_projector.py:321-368`, `750`, `780`; `docs/runbooks/telescope_runs_calendar.rst`

**Issue:** The revert is correct as scoped — round 2's stored-boundary substitute produced a
false positive (PROBE-P1) without fixing the false negative (PROBE-P6), so removing it was
right. But it widens the false negative from *one* half-null sub-shape to *all* of them,
and the shape it now misses is the one `_window_token_to_time()` produces for an ordinary
classical schedule line: `1130-EoN` and `BoN-0230` both yield exactly one set field and one
null field.

This is not hypothetical. The round-3-rewritten
`test_dry_run_of_a_half_null_remint_inverted_window_stays_silent_while_the_real_run_raises`
pins it: `reconcile_run(run, dry_run=True)` returns cleanly, and the immediately following
`reconcile_run(run)` raises `Computed an inverted allocation-night span ... start=2026-07-10
11:30:00+00:00 >= end=2026-07-10 11:29:46+00:00` (this error appears in the test-run log
above). Operator-visible consequence, traced through both consuming commands:

- `reconcile_campaign_runs --dry-run` reports `failed: 0`; the real pass catches the
  `ValueError` at `reconcile_campaign_runs.py:64` and reports `failed: 1`.
- `load_telescope_runs --dry-run` previews the line; the real pass catches it at the
  `(ValueError, Observatory.DoesNotExist)` clause and reports `skipped: 1`.

Round 3 narrowed the runbook's loader-section parity claim for the *create-arm* gap and
added a `.. note::` for the cutover re-run gotcha — but `grep -in 'invert|half-null|half-night|1130-EoN|BoN-0230' docs/runbooks/telescope_runs_calendar.rst` returns **nothing**. The
limitation is documented only in code comments and test docstrings, i.e. only where
operators never look, while the runbook's own always-dry-run-first instruction implies a
protection that does not exist for this shape.

**Fix:** Two parts. (a) Add the limitation to the runbook's loader and sweep sections next
to the arm-scoped parity note round 3 already added, in the same shape:

```rst
A line or run whose sub-night window is HALF-set (exactly one of ``night_start_utc`` /
``night_end_utc`` -- what a ``1130-EoN`` or ``BoN-0230`` schedule line produces) is the one
case ``--dry-run`` cannot check for an inverted span: the missing boundary is a sun event
the preview deliberately does not compute. Such a line can preview cleanly and then be
reported under ``skipped`` (loader) or ``failed`` (sweep) by the real pass. A set/set window
IS checked on both passes.
```

(b) If genuine parity is wanted later, the only sound route is the same provenance work
CR-01 needs — recording whether a stored boundary really is sun-derived. Both test
docstrings already say this; the runbook should too.

### WR-02: the round-3 "Re-run gotcha" caveat under-enumerates the fields it re-applies, and its two copies disagree with each other — `site_needs_review` (which drives a staff queue) is missing from both

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:69-73` and `docs/runbooks/telescope_runs_calendar.rst:989-1000`

**Issue:** Scope item (3) was "add a documentation caveat". The caveat that was added is
inaccurate, in a way that matters: `fields` at
`cutover_classical_allocations.py:489-504` has **14** keys, and both prose copies enumerate
fewer, differently.

| Field written by `fields` | Module docstring | Runbook note |
|---|---|---|
| `source` | listed | **missing** |
| `approval_status` | listed | **missing** |
| `run_status`, `campaign`, `target`, `site`, `site_raw`, `window_start`, `window_end`, `night_start_utc`, `night_end_utc`, `observation_details` | listed | listed |
| `site_needs_review` | **missing** | **missing** |
| `telescope_instrument` | **missing** | **missing** |

`site_needs_review` is not cosmetic: `campaign_views.py:221` filters the staff "Sites
Needing Review" queue on `site_needs_review=True`, and `campaign_tables.py:198` renders off
it. A matching-marker cutover re-run unconditionally writes `False`, so a run a staff member
deliberately re-flagged for review silently leaves that queue — the exact class of
"post-import staff edit does not survive" the caveat exists to warn about, and the one
instance it forgets to name. A caveat that lists nine of eleven staff-editable fields is
worse than none: an operator who reads it will reasonably infer that the two unlisted ones
are safe.

**Fix:** Make the two copies identical and complete, and derive them from the same place.
Replace both enumerations with:

```
every field this command writes -- ``source``, ``approval_status``, ``run_status``,
``campaign``, ``target``, ``site``/``site_raw``, ``site_needs_review``,
``telescope_instrument``, ``window_start``/``window_end``, both sub-night fields and
``observation_details`` -- is re-applied from the schedule line on every invocation.
Note ``site_needs_review`` in particular: a run a staff member re-flagged for the "Sites
Needing Review" queue is reset to ``False`` by the next cutover run.
```

and extend `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line`
to also set `site_needs_review=True` before the call and assert it is `False` afterwards, so
the pinning test actually covers the field the caveat forgot.

### WR-03: the cutover discards `adopt_event_into_run()`'s refusal return value, where its sibling call site checks it — a refused adoption would leave an event re-keyed into `ALLOC:` but unattributed, counted as a success

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:649`

**Issue:**

```python
rekeyed_event, _action = update_calendar_event_key_and_fields(event, url, rekey_fields)
adopt_event_into_run(rekeyed_event, run)          # <-- bool return discarded
# ...
claimed_nights.add(night)
group_rekeyed += 1
```

`adopt_event_into_run()` is documented to return `False`, **writing nothing**, when the
companion row already points at a different run (`campaign_utils.py:23-28`). The
allocation projector's own call site honours that contract
(`allocation_projector.py:518`: `if not campaign_utils.adopt_event_into_run(event, run):` →
log + `blocked += 1`). Here the same call's answer is thrown away, so a refusal produces an
event whose `url` has already been rewritten to `ALLOC:{run_pk}:{night}` — no longer in the
blank-url candidate set, so never re-examined by a later run — with no
`CalendarEventMeta.run`, counted under `events re-keyed` and reported to the operator as a
success. That is precisely the outcome D-18 ("what it cannot explain, it reports and leaves
completely untouched") exists to prevent.

The pre-filter at lines 511-517 rejects any event whose `meta.run_id is not None`, so today
this is unreachable — but "unreachable via the current pre-filter" is exactly the kind of
invariant this phase has already had broken twice (WR-09's `_CLASSICAL_RUN_STATUS` assertion
was added for the same reason). A one-time, effectively irreversible production migration
should not depend on a caller-side filter to keep a documented refusal path from silently
corrupting a row.

**Fix:** Honour the return value inside the per-event savepoint, so a refusal rolls the
re-key back and is reported like every other unexplainable event:

```python
rekeyed_event, _action = update_calendar_event_key_and_fields(event, url, rekey_fields)
if not adopt_event_into_run(rekeyed_event, run):
    # Re-raising inside the savepoint rolls the re-key back, so the event stays
    # byte-identical (D-18) and is reported rather than silently half-converted.
    raise _ForeignAttributionError(
        f'event was attributed to a different CampaignRun between the pre-filter and the re-key'
    )
```

with a matching `except _ForeignAttributionError` clause ahead of the broad
`except Exception`, routing to the existing `_FOREIGN_ATTRIBUTION` reason.

---

## Info

### IN-01: the both-null early return in `_raise_if_set_window_inverted()` is now redundant, and the docstring describes only the check that supersedes it

**File:** `solsys_code/allocation_projector.py:359-366`

**Issue:** After the revert, `if run.night_start_utc is None and run.night_end_utc is None:
return` (line 359) can never be the deciding branch: any path that reaches it with a null
field also hits `if start is None or end is None: return` at line 366. The function is now
behaviourally identical to the pre-35-13 `or` form, with one extra `_night_span_utc()` call
and one extra `_time_of_day_to_datetime()` call on the half-null path. Keeping round 2's
`and` was harmless, but it leaves two returns expressing one rule, and the docstring
("Checks a span only when BOTH ... are set") describes the line-366 check while the
line-359 one goes unexplained — a future reader may assume the two differ.

**Fix:** Either collapse to the single `or` early return, or add one line to the docstring
noting the both-null return is a fast path that line 366 would otherwise cover.

### IN-02: the cutover still hardcodes `'ALLOC:'` instead of importing `ALLOC_URL_NAMESPACE` (carried forward, still open)

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:690`

**Issue:** `CalendarEvent.objects.filter(url__startswith='ALLOC:')` uses a string literal
while the module already imports four other names from `allocation_projector`. Reported in
iteration 6 (IN-02); unchanged.

**Fix:** Import `ALLOC_URL_NAMESPACE` alongside the existing four and use it.

### IN-03: an identity key is claimed before the group's transaction commits (carried forward, still open)

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:539` vs `560`

**Issue:** `seen_keys[key] = source_line` runs before `with transaction.atomic():`. A
group-level rollback (the `except Exception` at line 670) undoes every write but leaves the
key claimed, so a later group sharing that key is reported under `duplicate_identity`
naming a line that converted nothing. Reported in iteration 6 (IN-03); unchanged.

**Fix:** Move the assignment to just after the `runs_created += group_created` fold at line
666, alongside the other post-commit bookkeeping NF-05 already moved there.

### IN-04: no executed notebook cell exercises the `existing_source_line is None` refusal — the exact branch CR-01 inverted (carried forward, still open)

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (cell 5, markdown only)

**Issue:** `grep` of the executed outputs finds the *mismatched*-marker message ("already
claimed ... for a different Source line", cell 11) but never the *no-marker* message ("with
no recoverable 'Source line:' marker"). The no-marker branch — the one CR-01 inverted from
permissive to refusing, and whose remedy text round 3 just extended — appears in the
notebook only as prose in a markdown cell. Reported in iteration 6 (IN-01); unchanged.

**Fix:** Add a throwaway fixture cell (same self-cleaning pattern as the NF-14 cell 9) with a
claimant whose `observation_details` has no `Source line:` marker, and print the resulting
stderr line.

### IN-05: the loader notebook restores the mutated `Observatory.timezone` without `try/finally` (carried forward, still open)

**File:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` cells 16 and 18

**Issue:** Both cells set `ntt.timezone = 'America/Santigo'`, call `call_command(...)`, then
restore. If the command raises, the typo persists for every later cell in the same run and
the failure cascades into unrelated sections. (Iteration 6's IN-04 framed this as mutating
the shared dev database; that part is inaccurate — cell 1's output confirms a per-run scratch
copy under `/tmp/fomo-notebook-db-*`, removed by the final cell. The ordering fragility
stands.)

**Fix:** Wrap each in `try: ... finally: ntt.timezone = original; ntt.save(update_fields=['timezone'])`.

### IN-06: `logger` is defined and never used in the cutover command

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:149`

**Issue:** `logger = logging.getLogger(__name__)` has zero call sites (`grep -c 'logger\.'` → 0).
Ruff does not flag an unused module-level assignment, so this survives the D-07 gate. Every
operator-facing message in this command goes through `self.stdout`/`self.stderr`, which is
the right choice for a management command — the logger is leftover scaffolding.

**Fix:** Delete the `logger` line and the now-unused `import logging`.

### IN-07: the reason breakdown is printed in insertion order on stdout but sorted order in the `CommandError`, despite the constants being declared "in report order"

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:153`, `696-697`, `701`

**Issue:** The constant block is commented "D-18's named reason vocabulary, **in report
order**", but `for category, count in reason_counts.items()` (line 696) iterates a
`defaultdict` in *first-seen* order, while the `CommandError` breakdown (line 701) uses
`sorted(reason_counts.items())`. So the same run's two operator-facing summaries can list
the same reasons in three different orders, none of which is the declared vocabulary order.
Cosmetic, but this command's whole contract is that its reports are what the operator acts
on, and a notebook caller (`call_command()`) sees only the `CommandError` message.

**Fix:** Iterate both in the declared order, e.g.
`for category in _REASON_LABELS: count = reason_counts.get(category); if count: ...`, and
build the `CommandError` breakdown from the same loop.

---

_Reviewed: 2026-09-15T19:01:09Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
