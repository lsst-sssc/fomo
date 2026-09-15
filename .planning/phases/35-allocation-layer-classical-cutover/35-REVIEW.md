---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-15T00:00:00Z
depth: deep
iteration: 5
prior_review: 35-REVIEW.md (git show 8a393cf)
diff_base: 8a393cf75608981aef6199de136044bb8df6ed08
files_reviewed: 16
files_reviewed_list:
  - CLAUDE.md
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
  - docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/observation_projector.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_observation_projector_signals.py
prior_findings:
  total: 9
  closed: 7
  partially_closed: 2
  still_open: 0
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 35: Code Review Report (iteration 5 — re-review of the `8a393cf..HEAD` gap-closure round)

**Reviewed:** 2026-09-15
**Depth:** deep
**Files Reviewed:** 16
**Status:** issues_found

## Summary

This re-reviews the four gap-closure plans 35-08 through 35-11 (commits `868caa6`,
`9e555eb`, `0eceeb5`, `eb79263`, `0b7599f`, `45b38a7`, `33f0a61`, `11b14da`, `4cee1f1`,
`5eb2718`), plus the F-34-1 fix `24875bf` that also landed in this range, against the nine
iteration-4 findings.

**Counting convention:** the `findings:` block counts only NEW findings (`CR-01`,
`WR-01`..`WR-04`, `IN-01`..`IN-03`). No iteration-4 finding is carried forward by its old
id — seven are genuinely closed, and the two that are only *partially* closed have their
surviving half re-filed under a new id so a number never stands for a claim that is now
partly true.

The batch is the strongest so far on the mechanical findings: NF-22, NF-23, NF-24 and NF-25
are cleanly closed, and NF-24's notebook work is real regenerated output rather than prose
(the loader demo's new skip-path cell prints both stderr lines; the projector demo proves
attribution on the creating save; the reconciler demo runs the cutover a *second* time and
asserts the winning run byte-identical). 215 tests pass; both ruff gates pass; the working
tree was not modified by this review.

But the loop's dominant failure mode recurred twice more, in both of the same two places:

- **NF-19 (BLOCKER) is only partially closed.** The new guard reads the database, which
  closes both harms the review reproduced. It then reads the claimant's identity out of
  `CampaignRun.observation_details` — a **free-text field that `CampaignRunAdmin` leaves
  fully editable** and that `import_campaign_csv` and the campaign submission form both
  write — and treats "no recoverable `Source line:`" as **permission to proceed**. A staff
  member replacing a classical run's details with an ops note silently disarms the guard.
  Reproduced: `run_status` planned → cancelled, the staff note destroyed, three events
  re-keyed, **exit 0, `unexplained: 0`, empty stderr** — verbatim NF-19 case 2. Re-filed as
  **CR-01 (BLOCKER)**.
- **NF-20 is only partially closed.** `_raise_if_set_window_inverted()` returns early when
  *either* sub-night field is null, but `_span_needs_remint()` — the branch that calls it —
  returns True for a **half-null** run. Reproduced on a realistic `1130-EoN` half-night
  edit: dry run returns `created=1, retired=1` with no error, the immediately following real
  run raises `ValueError: Computed an inverted allocation-night span…`. That is NF-10's
  original sentence, unchanged, in its third consecutive iteration. Re-filed as **WR-01**.

### Prior-finding verification: 7 closed, 2 partially closed, 0 still open

| Prior finding | Verdict |
|---|---|
| **NF-19** cutover identity-key guard invocation-scoped, not database-scoped (BLOCKER) | **Partially closed.** `cutover_classical_allocations.py:414-426` now queries `CampaignRun.objects.filter(source_identifier=key).first()` before any write, on both passes. Both harms the review reproduced are gone, verified by the four new tests in `TestDatabaseScopedIdentityGuard` and by the reconciler notebook's own executed second-invocation cell (`run_status` `'planned'` before and after, `duplicate_identity=3`, no `key_collision`). `IN-02`'s `seen_keys` move landed too. **But the guard's `not in (None, source_line)` predicate makes a claimant with no recoverable `Source line:` marker permissive**, and `observation_details` is editable in the Django admin — re-filed as **CR-01 (BLOCKER)**. |
| **NF-20** dry-run inversion guard covers only the create path (WARNING) | **Partially closed.** `_raise_if_set_window_inverted()` (`allocation_projector.py:321-353`) is a genuine shared helper, called from both `_mint_fields()` caller branches (`:732`, `:756`), and `test_dry_run_of_a_remint_inverted_window_also_raises` pins the set/set re-mint case. **But the helper's `if run.night_start_utc is None or run.night_end_utc is None: return` early-out leaves the half-null shape uncovered, and `_span_needs_remint()` reaches that shape** — re-filed as **WR-01**. |
| **NF-21** malformed `Observatory.timezone` aborts the whole loader (WARNING) | **Closed.** `load_telescope_runs.py:338-353` adds a dedicated `except ZoneInfoNotFoundError` clause *ahead* of the `(ValueError, Observatory.DoesNotExist)` clause — clause order is correct, and `site` is provably bound (`get_site()` is the second statement in the same `try` and cannot itself raise this class). Verified by execution: a three-line file with a typo'd timezone on line 2 reports `skipped: 2`, processes line 3, and leaves no `CampaignRun` behind for either bad line. The new `TestMalformedTimezoneSkipsOneLine` test and the loader notebook's cell 16 both pin it with real output. **A dry-run/real counter divergence on this same path survives — WR-02.** |
| **NF-22** blocked legacy takeover event double-counted (WARNING) | **Closed.** `allocation_projector.py:690-698` claims `legacy_urls_claimed` before the `_may_write()` check, symmetric with the retired branch (`:651`). `TestTakeoverBlockedCountedOnce` asserts `result.blocked == 1` and exactly one log record for the row, and that the legacy event and its foreign attribution both survive. I traced the downstream consumer (`_stale_dated_events()` `:615-616`) and confirmed the exclusion cannot over-delete: the excluded row is shape (d), which that function leaves alone anyway. |
| **NF-23** `_detach_stale_family_events()` 3-tuple annotation vs 4 returns (WARNING) | **Closed.** `campaign_reconciler.py:678` is now `-> tuple[int, int, int, int]`, matching the `Returns:` docstring (`:757`), the `return` (`:817`) and the caller's 4-way unpack (`:883`). **The third copy of the *other* NF-17 contract was missed — WR-03.** |
| **NF-24** paired notebooks for the two changed modules not updated (WARNING) | **Closed, and better than asked.** `load_telescope_runs_demo.ipynb` gains a "Per-line skip paths" section whose executed output shows both stderr lines and `skipped: 2`, with assertions that line 3 still converted and neither bad line left a run row. `project_observation_calendar_demo.ipynb` gains a creating-save attribution cell with real output (`CalendarEventMeta.run_id: 74`, `allocation nights left: 2`), run inside a deliberately-rolled-back `atomic()` against the real dev database and using `NonSiderealTargetFactory` per CLAUDE.md. `CLAUDE.md:133-135` also maps `allocation_projector.py` — the phase's central module, which had no paired notebook at all — into `reconcile_campaign_runs_demo.ipynb`, and the breach history entry is added. The `sched06-baseline.json` rewrite (74 → 50 records) is a deliberate, separately-documented discharge of the SCHED-06 step owed since `34-UAT.md`, not collateral damage. |
| **NF-25** un-actionable `duplicate_identity` remedy (WARNING) | **Closed.** Both message sites (`cutover_classical_allocations.py:396-398`, `:421-424`) and all three runbook passages (`:939-949`, `:1454-1461`, `:1494-1503`) now say "edit the affected events' description `Source line:` text … in the Django admin", and the runbook explicitly contrasts it with `load_telescope_runs`'s schedule-file remedy ("the remedy is NOT the same"). The notebook's committed output shows the new text. |
| **IN-01** comment hard-coding line numbers (INFO) | **Closed.** `test_cutover_classical_allocations.py:559-564` now cites `rekeyed_event.url` / `delete_legacy_pk` by identifier. |
| **IN-02** identity key claimed before convertibility is known (INFO) | **Closed as specified.** `seen_keys[key] = source_line` moved to `:501`, after the campaign-mismatch, status-lookup and all-events-foreign checks. **One residual branch — IN-03.** |

### Verification method

`python manage.py test solsys_code.tests.test_allocation_projector
solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations
solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals`
— **215 tests, all passing.** `pre-commit run ruff --all-files` and `pre-commit run
ruff-format --all-files` — **both Passed.**

CR-01, WR-01, WR-02 and WR-04 are backed by **executed probes** against a real Django test
database (probe modules written under `solsys_code/tests/`, run, then deleted;
`git status --short` confirms **no source file was modified by this review**). Reproduced
facts, not inferences:

- A pre-existing `CampaignRun` holding the derived key whose `observation_details` is a
  staff free-text note: `Done. candidates: 3, groups: 1, runs created: 0, updated: 1,
  events re-keyed: 3, unexplained: 0`, **exit 0**, `run_status` `planned` → `cancelled`,
  `observation_details` `'Rescheduled per PI request; see ticket OPS-4412.'` →
  `'Status: cancelled\nSource line: NTT EFOSC2 cancelled 9-12 July'` (CR-01).
- Half-null run (`night_start_utc=11:30`, `night_end_utc=None`) at La Silla:
  `reconcile_run(dry_run=True)` → `ReconcileResult(created=1, retired=1, …)` and no error;
  `reconcile_run(run)` → `ValueError: Computed an inverted allocation-night span for run
  pk=1 night=2026-07-09: start=2026-07-10T11:30:00+00:00 >= end=2026-07-10T11:29:46+00:00`
  (WR-01).
- `load_telescope_runs --dry-run` over an existing run with a typo'd timezone:
  `Done (dry run). lines processed: 1, created: 0, updated: 0, unchanged: 1, skipped: 1`
  against the real run's `unchanged: 0, skipped: 1` — one line, two outcomes (WR-02).
- A genuine DB-level failure inside the `campaign_run_links` lookup is swallowed with no
  savepoint; on SQLite the caller's next query still succeeds (`{'count': 0}`), so the
  exposure is backend-dependent — see WR-04 for the honest scope (WR-04).

### Positives worth recording

The NF-22 fix is the model of what this loop should produce: a one-line move, a regression
test that asserts both the counter *and* the log-record count (the actual symptom), and a
comment naming the sibling branch it is now symmetric with. The NF-19 test class is the
first in this phase to test a *second invocation* rather than a single pass, and its four
cases (merge / differing line / same line / no marker) are the right partition — the bug in
CR-01 is that the fourth case's assertion pins the wrong outcome, not that the case was
missed. `TestMalformedTimezoneSkipsOneLine` correctly asserts the *following* line still
processed, which is the real invariant rather than the error message. And the loader
notebook's skip-path cell restores the mutated `Observatory.timezone` immediately after the
demonstration, so later sections still resolve NTT — a failure mode a less careful
regeneration would have shipped.

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: NF-19's database-scoped guard derives its authority from an admin-editable free-text field, and treats "no `Source line:` marker" as permission — a staff edit re-opens the silent merge, with exit 0

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:402-426`
(specifically the `not in (None, source_line)` predicate at `:417`); contract asserted at
`:46-51`, `:663-664`, and `docs/runbooks/telescope_runs_calendar.rst:939-949`, `:1501-1503`
**Severity:** BLOCKER

**Issue:** The fix correctly moved the guard from an in-process `dict` to a database lookup.
It then resolves *who* the database claimant is by re-parsing the claimant's own
`observation_details`:

```python
existing_run = CampaignRun.objects.filter(source_identifier=key).first()
if existing_run is not None:
    existing_source_line = _extract_source_line(existing_run.observation_details)
    if existing_source_line not in (None, source_line):
        _mark_unexplained(events, _DUPLICATE_IDENTITY, ...)
        continue
```

`None` is on the permissive side of that predicate, and the comment above it states the
rationale: *"a database row with no recoverable `Source line:` marker has nothing to
disagree with, so it is treated as the SAME line rather than rejected."* That reasoning is
inverted for a one-time, destructive migration: a row with no marker is precisely the row
this command **cannot prove it owns**, and the write it then performs is a full
find-and-update of every dispatch-deciding field on an APPROVED run.

`observation_details` is not an internal field. `CampaignRunAdmin` (`solsys_code/admin.py:132`)
sets only `readonly_fields = ['approval_status']`, so **`observation_details` is freely
editable in the Django admin for every `CampaignRun`, classical ones included** — and this
command's own remedy text sends operators into the admin to edit exactly these rows. It is
also written from an arbitrary CSV column by `import_campaign_csv.py:321` and from a
`forms.CharField(widget=forms.Textarea)` on the campaign submission form
(`campaign_forms.py:65`). Any of those clears the marker.

Reproduced (probe, executed; a classical run created by a prior import, whose details a
staff member replaced with an ops note, and the stranded blank-url events of the *cancelled*
counterpart line):

```
PROBE-A exit_nonzero: False
PROBE-A stdout: Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0,
                events re-keyed: 3, unexplained: 0
PROBE-A run_status now: cancelled (was planned)
PROBE-A observation_details now: 'Status: cancelled\nSource line: NTT EFOSC2 cancelled 9-12 July'
```

Exit 0, `unexplained: 0`, empty stderr. The run's real-world lifecycle state is flipped, the
operator's free-text note is destroyed, `target` is set to `None` and `campaign`,
`window_start`/`window_end`, `site` and `site_raw` are all overwritten from a line that is
*not* the line that created the run (`fields` at `:451-466`). The three events are then
re-keyed onto it and titled `[CANCELLED] NTT EFOSC2`. This is the same class of harm NF-19
was filed for, and nothing tells the operator it happened.

The command's own test suite pins this outcome as correct:
`test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` asserts the
conversion proceeds — but it constructs the claimant with `observation_details=''` and an
`allocation`-shaped line, so no field visibly changes and the destruction is invisible to
the assertion.

Three documentation statements now assert the guarantee unconditionally and are false for
this path: the module docstring's *"This guarantee holds on every invocation, not only the
first, because the check reads the database"* (`:46-48`); the `CommandError`'s new
*"rewrites no existing `CampaignRun` (NF-19, 35-REVIEW.md)"* (`:663-664`), which is printed
verbatim in the reconciler notebook's committed output; and the runbook's *"this holds on
the first invocation and on every re-run"* (`:939-942`).

**Fix:** invert the default — refuse what the command cannot prove it owns, and give the
unprovable case its own actionable reason rather than folding it into `duplicate_identity`:

```python
existing_run = CampaignRun.objects.filter(source_identifier=key).first()
if existing_run is not None:
    existing_source_line = _extract_source_line(existing_run.observation_details)
    if existing_source_line != source_line:
        # CR-01: a claimant whose stored Source line: is absent (an admin edit, an
        # import_campaign_csv 'Observation Details' column, a submission-form note) is a
        # claimant this command cannot prove it owns -- refuse and report, never
        # find-and-update. `observation_details` is admin-editable, so its absence is not
        # evidence of agreement.
        reason = (
            f'CampaignRun pk={existing_run.pk} already claims {key!r} '
            + (
                'for a different Source line'
                if existing_source_line is not None
                else "with no recoverable 'Source line:' marker in its observation_details, so "
                'this command cannot prove the run came from this schedule line'
            )
            + "; restore or correct that run's observation_details 'Source line:' text in the "
            'Django admin so it matches, or disambiguate the two lines, then re-run'
        )
        _mark_unexplained(events, _DUPLICATE_IDENTITY, reason)
        continue
```

Replace `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` with its
inverse, and add the destructive-case regression the current suite is missing: a claimant
with `run_status=PLANNED` and a non-marker `observation_details`, a group whose line is the
`cancelled` counterpart, asserting a non-zero exit **and** that `run_status`,
`observation_details` and `target` are byte-identical afterwards. Then correct the three
documentation statements above to name the marker requirement.

---

## Warnings

### WR-01: NF-20's shared guard skips the half-null sub-night shape that `_span_needs_remint()` reaches — a `1130-EoN` operator edit still previews clean and raises on the real run

**File:** `solsys_code/allocation_projector.py:348-349` (the early-out), reached from `:732`
and `:756`; contrast with `_span_needs_remint()` at `:378-379`
**Severity:** WARNING

**Issue:** `_raise_if_set_window_inverted()` refuses to check anything unless **both**
sub-night fields are set:

```python
if run.night_start_utc is None or run.night_end_utc is None:
    return
```

Its docstring justifies that as *"the same null-field convention `_span_needs_remint()`
itself already uses"*. The two conventions are **not** the same. `_span_needs_remint()`
short-circuits only when **both** fields are null (`:378`, `and`, not `or`); for a half-null
run it checks the one set field and can return True — routing the night into the re-mint
branch, whose `if dry_run:` short-circuit then calls a guard that declines to look. The real
run's `_mint_fields()` → `night_bounds()` resolves the set end from `zoneinfo` and the null
end from `sun_event()`, and those two can be inverted.

This is not an exotic shape: a half-night classical line (`1130-EoN`, `BoN-0230`) produces
exactly one set field and one null one via `_window_token_to_time()`.

Reproduced (probe, executed; La Silla, one night, valid `23:00`/null minted first, then the
start edited to `11:30` — after that night's 11:29:46 sunrise):

```
PROBE-D sunset/sunrise UTC: 2026-07-09 22:06:35.918  2026-07-10 11:29:46.816
PROBE-D minted span: 2026-07-09 23:00:00+00:00 -> 2026-07-10 11:29:46+00:00
PROBE-D dry run returned (no error): ReconcileResult(created=1, retired=1, ...)
PROBE-D real run RAISED: Computed an inverted allocation-night span for run pk=1
        night=2026-07-09: start=2026-07-10T11:30:00+00:00 >= end=2026-07-10T11:29:46+00:00.
```

`reconcile_campaign_runs --dry-run` reports `would_retire: 1, would_create: 1`; the real
sweep reports the run under `failed`. That is NF-10's original sentence, verbatim, for the
third iteration running — and the create-branch comment at `:750-754` now asserts the
opposite without a null-field qualifier (*"so the two passes cannot drift apart on this
check"*).

**Fix:** on the **re-mint** branch the missing boundary is already available with no astropy
call — it is the stored boundary on `existing`, which was minted from the same deterministic
`sun_event()` for the same site and night:

```python
def _raise_if_set_window_inverted(run: CampaignRun, night, existing: CalendarEvent | None = None) -> None:
    if run.night_start_utc is None and run.night_end_utc is None:
        return
    night_span = _night_span_utc(run, night)
    start = (
        _time_of_day_to_datetime(run.night_start_utc, night, night_span)
        if run.night_start_utc is not None
        else (existing.start_time if existing is not None else None)
    )
    end = (
        _time_of_day_to_datetime(run.night_end_utc, night, night_span)
        if run.night_end_utc is not None
        else (existing.end_time if existing is not None else None)
    )
    # WR-01: a null field on the CREATE path has no stored counterpart and genuinely needs
    # sun_event(), which D-13 forbids on a preview -- that one case stays unchecked, and
    # the docstrings must say so rather than claiming full parity.
    if start is None or end is None:
        return
    _raise_if_inverted(run, night, start, end)
```

Call it as `_raise_if_set_window_inverted(run, night, existing)` from the re-mint branch and
`_raise_if_set_window_inverted(run, night)` from the create branch. Add the half-null twin of
`test_dry_run_of_a_remint_inverted_window_also_raises` (the probe above is the fixture), and
narrow the `:750-754` comment and the helper docstring to say the create-path half-null case
is the one shape that remains unpreviewable.

### WR-02: `load_telescope_runs --dry-run` counts one line twice when the preview's own `reconcile_run()` raises — the dry run and the real run disagree about the totals

**File:** `solsys_code/management/commands/load_telescope_runs.py:294-316` (counters at
`:297-302`, the raising call at `:305`), handled at `:338-353` and `:354-357`
**Severity:** WARNING

**Issue:** The dry-run branch increments `run_created`/`run_updated`/`run_unchanged` from
`preview_campaign_run_action()` **before** calling `reconcile_run(existing, dry_run=True)`.
If that call raises — NF-21's `ZoneInfoNotFoundError`, `sun_event()`'s own `ValueError` for a
blank timezone or a polar site, or WR-01's inverted-span `ValueError` — the per-line handler
then adds `run_skipped += 1` for the same line. The real branch does not have this shape:
`:325-330` increments only after the `transaction.atomic()` block has returned, so a failure
there yields `skipped` alone.

Reproduced (probe, executed; a line whose run already exists, site timezone then mistyped):

```
PROBE-B first (real) pass: Done. lines processed: 1, created: 1, updated: 0, unchanged: 0, skipped: 0
PROBE-B dry run  : Done (dry run). lines processed: 1, created: 0, updated: 0, unchanged: 1, skipped: 1
PROBE-B real run : Done. lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1
```

One line, `unchanged: 1 + skipped: 1` in the preview against `skipped: 1` for real — so
`created + updated + unchanged + skipped` no longer equals `lines processed` on the preview,
and an operator reading the dry run believes a run will be left untouched when in fact the
line will be dropped. NF-21's own regression test (`TestMalformedTimezoneSkipsOneLine`)
exercises only the real path, which is why this survived the fix.

**Fix:** mirror the real branch — compute the action, run the preview reconcile, and fold the
counters only once both have succeeded:

```python
if dry_run:
    existing = CampaignRun.objects.filter(source_identifier=key).first()
    action = preview_campaign_run_action(existing, fields)
    if existing is not None:
        reconcile_result = reconcile_run(existing, dry_run=True)   # may raise -> handled below
        night_created += reconcile_result.created
        ...
    else:
        night_created += len(nights)
    # WR-02: folded only now, so a raising preview reports `skipped` alone -- the same
    # single outcome the real branch reports for the identical failure.
    if action == 'created':
        run_created += 1
    elif action == 'updated':
        run_updated += 1
    else:
        run_unchanged += 1
```

Add a dry/real parity assertion to `TestMalformedTimezoneSkipsOneLine` (same fixture, both
modes, identical `created/updated/unchanged/skipped` tuple).

### WR-03: the third copy of the `claimed_legacy_urls` contract was left behind — `_stale_dated_events()`, the function that actually performs the exclusion, still documents the pre-NF-09 meaning

**File:** `solsys_code/campaign_reconciler.py:600-606`; contrast with the two corrected
copies at `:746-756` and `solsys_code/allocation_projector.py:576-595`
**Severity:** WARNING

**Issue:** NF-17 corrected two descriptions of what `claimed_legacy_urls` holds and why
excluding it matters. There are three. The Arg docstring on `_stale_dated_events()` — the
function whose `stale_dated.exclude(url__in=claimed_legacy_urls)` at `:615-616` *is* the
exclusion — still carries the original, now-wrong text:

```
claimed_legacy_urls: legacy ``RUN:{pk}:{date}`` urls the allocation projector's own
    per-night loop already decided the fate of THIS call (a takeover re-key or a
    retirement delete) -- excluded here so a ``dry_run`` preview never
    double-counts the SAME url under both ``rekeyed``/``retired`` and
    ``legacy_deleted``.
```

Both halves are stale. NF-09 widened the set to include blocked and human-declined urls, and
**NF-22 (closed in this very batch) widened it again** to include the blocked takeover url —
so the parenthetical enumeration is missing two of the four outcomes. And the "so a
`dry_run` preview never double-counts" clause is the exact claim NF-17 was filed to correct:
the two sibling copies now both state that the exclusion is *load-bearing in real mode* for a
blocked or declined url. A reader of the function that does the work gets the superseded
contract; a reader of its caller gets the current one.

**Fix:** replace `:600-606` with the same wording the two corrected copies use, and add the
blocked-takeover outcome NF-22 introduced:

```
claimed_legacy_urls: legacy ``RUN:{pk}:{date}`` urls the allocation projector's own
    per-night loop already decided the fate of THIS call AT ALL -- a takeover re-key, a
    retirement delete, a block (either branch), or a human-confirmed decline. Excluded
    here so the SAME single decision is never reported twice: a no-op in real mode for a
    re-keyed or deleted url (it has already left the ``RUN:`` namespace), but LOAD-BEARING
    in real mode for a blocked or declined one, which is by definition not written and is
    still sitting in this namespace right now (NF-09/NF-17/NF-22, 35-REVIEW.md). Empty for
    a container-dispatched run, which never takes over a legacy night at all.
```

### WR-04: F-34-1's swallowed database error has no savepoint — the module's own `project_record()` docstring explains at length why that is the thing that makes such a catch safe

**File:** `solsys_code/observation_projector.py:647-658`; contrast with `:339-352` and
`:365-370`
**Severity:** WARNING

**Issue:** The new guard catches a database error and continues:

```python
try:
    links = list(instance.campaign_run_links.select_related('run'))
except Exception as exc:  # noqa: BLE001 -- F-34-1/TRIG-02: a query fault here must
    # never abort the caller's save, same guarantee project_record() gets above.
    ...
    links = []
```

The comment's claim — *"same guarantee `project_record()` gets above"* — is not accurate.
`project_record()`'s guarantee does not come from its `except`; it comes from the
`transaction.atomic()` savepoint at `:365`, and that module already spends twelve lines
(`:339-352`) explaining precisely why: *"a database error … has to propagate all the way out
of the `with` block before this function's own `try` catches it, so `Atomic.__exit__` sees
the exception still in flight and rolls back to the savepoint rather than committing it."*
The new lookup has no `with transaction.atomic():` at all, so on a backend where a failed
statement poisons the surrounding transaction (PostgreSQL — which CLAUDE.md names as the
production target: *"SQLite3 has concurrent write limitations; production deployments should
migrate to PostgreSQL"*), swallowing the error hands the caller a transaction whose next
statement raises `InFailedSqlTransaction`/`TransactionManagementError`. The save is not
actually protected; the failure is only relocated and made harder to attribute.

Honest scope: on this project's current SQLite backend the probe showed no failure —
`{'count': 0}`, the caller's next query succeeded. So the exposure is backend-dependent, not
reproducible here today. What *is* reproducible here is that the regression test cannot
detect it either way: `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection`
patches `campaign_run_links` with a `SimpleNamespace` whose `select_related` raises a
Python-constructed `OperationalError` that never reaches the database — so it proves the
`except` clause exists, not that the transaction survives. The same gap applies to the
pre-existing per-link `except` at `:664-675` and to `allocation_projector.py:871-880`/`:952-961`.

**Fix:** give the lookup the savepoint the module's own convention requires, with the `except`
outside the `with` as `project_record()` documents:

```python
try:
    with transaction.atomic():
        links = list(instance.campaign_run_links.select_related('run'))
except Exception as exc:  # noqa: BLE001 -- WR-04: the savepoint, not the except, is what
    # makes this safe; catching inside the `with` would commit the broken block instead.
    logger.warning(...)
    links = []
```

Correct the comment's "same guarantee `project_record()` gets" to name the savepoint as the
mechanism, and either extend the same treatment to the three sibling catches or note in each
why it is unnecessary there.

---

## Info

### IN-01: the cutover re-queries `existing_run` inside the group transaction, shadowing the NF-19 lookup it already performed

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:414` and `:524`
**Issue:** The dry-run branch re-runs `CampaignRun.objects.filter(source_identifier=key).first()`
and rebinds `existing_run`, a name already bound 110 lines above by the NF-19 guard. The two
lookups are identical, so the second is a redundant query and a shadowed name that invites a
future reader to assume the guard's result is being reused when it is not.
**Fix:** delete `:524` and use the outer `existing_run` binding; if a fresh read is wanted for
race safety, rename it (`claimant_run` for the guard, `existing_run` for the write) so the two
roles stay distinguishable.

### IN-02: the cutover hardcodes `'ALLOC:'` instead of the `ALLOC_URL_NAMESPACE` constant it already imports the module's helpers from

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:648`
**Issue:** `CalendarEvent.objects.filter(url__startswith='ALLOC:')` is a bare literal in the
summary line, while the module imports `allocation_night_url`/`allocation_night_title` from
`solsys_code.allocation_projector`, which defines `ALLOC_URL_NAMESPACE = 'ALLOC:'` for exactly
this purpose. A namespace rename would silently make the final summary count zero.
**Fix:** import `ALLOC_URL_NAMESPACE` alongside the existing four names and use
`url__startswith=ALLOC_URL_NAMESPACE`.

### IN-03: IN-02's fix leaves one branch where an unconvertible group still claims the identity key

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:501` relative to
`:521-641`
**Issue:** `seen_keys[key] = source_line` now runs after the campaign-mismatch, status-lookup
and all-events-foreign checks — but still *before* the group's `try: with
transaction.atomic():` block. A group whose transaction rolls back entirely (the `except` at
`:628`, which marks every event `_OTHER` and writes nothing) keeps the key claimed, so a
sibling group sharing that key is reported under `duplicate_identity` naming a line that
converted nothing — the residual of the same operator-confusion IN-02 described.
**Fix:** move `seen_keys[key] = source_line` to just after the `runs_created += group_created`
fold at `:624-627`, i.e. only once the group's writes have committed.

---

_Reviewed: 2026-09-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (re-review, iteration 5)_
