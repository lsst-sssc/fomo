---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-16T16:30:00Z
depth: deep
iteration: 9
diff_base: 61f0df04f275912a2ec28d84fd72270b872ba237
files_reviewed: 6
files_reviewed_list:
  - solsys_code/allocation_projector.py
  - solsys_code/models.py
  - solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py
  - solsys_code/tests/test_allocation_projector.py
  - docs/runbooks/telescope_runs_calendar.rst
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
findings:
  critical: 2
  warning: 8
  info: 4
  total: 14
status: issues_found
---

# Phase 35: Code Review Report (iteration 9 — fifth gap-closure round)

**Reviewed:** 2026-09-16T16:30:00Z
**Depth:** deep
**Files Reviewed:** 6
**Status:** issues_found

## Summary

**All three of iteration 8's criticals are genuinely closed**, verified line-by-line against
the current source rather than against the SUMMARY claims:

| Iteration-8 finding | Verdict |
|---|---|
| **CR-01** (re-mint delete had no human-confirmation guard) | **CLOSED.** `_remint_decline_reason()` (`allocation_projector.py:598-649`) runs at `:987`, *before* either counter moves and before the `dry_run` short-circuit, and is a pure read (`_clearable_declined_and_unattributed()` + a companion-row staff-state check), so dry-run and real-run parity holds. Probe 8/probe 9's shapes are covered by seven tests. The deliberate divergences the plan recorded (no foreign-ownership arm; decline rather than preserve) both check out: `_may_write(existing, run)` at `:883` provably routes a foreign-owned night to `blocked` before `:987` is reached. |
| **CR-02** (token omitted the site) | **CLOSED for every shape that reaches the token.** `_sub_night_provenance_token()` (`:443-446`) now emits `v2\|{site_id}\|{start}\|{end}`; `_span_needs_remint()` `:566` is a version-prefix test, so `None`, `''` and the pre-release `'none\|none'` form all correctly read as unrecorded and re-resolve once through the bounded legacy branch — no data migration, D-15 respected. Migration 0020 widens the column 32→64 and matches `models.py:130-132` exactly. Probe 1 (La Silla → Siding Spring) now re-mints. **But the token is never consulted at all for a run with BOTH sub-night fields set** — see WR-05. |
| **CR-03** (delete before the failure point) | **CLOSED, both halves.** `_mint_fields()` is computed at `:1033`, before `existing.delete()` at `:1041`; the delete/create/link/record group is wrapped in `transaction.atomic()` at `:1034`, scoped to one night. Probe 6's shape and a mid-block `RuntimeError` each have their own test. There is still no outer transaction in `reconcile_run()` — correct, and consistent with the "a sweep still commits the nights it has already finished" rationale. |

Also closed: **IN-02** (column widened to 64, with a worst-case-token width test read off
the model field rather than hardcoded) and **IN-04** (CLAUDE.md's pairing map now maps
`allocation_projector.py` → `reconcile_campaign_runs_demo.ipynb`, and the notebook carries
two new real, executed demos for CR-01 and CR-02). **WR-03** and **WR-04** are *partially*
closed and are carried forward below with only their residue.

**This round opened two new critical defects of its own, both in the same blind spot.** The
guard 35-20 added is correct about *not destroying* the night, but it also stops the night
being *refreshed* — a declined night's `title`/`description`/`target_list` are frozen
forever, so a `mark_cancelled` never reaches it (CR-04). And after seven iterations of
auditing delete paths one at a time, the one `ALLOC:`-event delete that still has **no**
UAT-2026-09-09 Option B guard is the retirement branch's own `existing.delete()` at `:940`
— the guard sitting eight lines above it protects only the *legacy* `RUN:` event, not the
allocation night itself (CR-05). Iteration 8 asserted that branch was already guarded; it
is not.

### Verification method

- Re-read `allocation_projector.py` in full (1290 lines) plus every cross-module symbol it
  depends on: `campaign_reconciler._may_write()`, `_clearable_and_declined()`,
  `_clearable_declined_and_unattributed()`, `reconcile_run()`'s dispatch,
  `management/commands/reconcile_campaign_runs.py`'s per-run loop and message block, and
  `campaign_views._resolve_site()` / `_message_reconcile_side_effects()` /
  `CalendarEventMetaAdmin.save_model()` / `get_readonly_fields()`.
- Re-ran iteration 8's probes as traces against the new code: probe 1 (site change), probes
  8/9 (confirmed + staff-state re-mint), probe 6 (inverted re-mint), probe 2/3 (dry-run cost
  and dry-run raise).
- Re-derived `_time_of_day_to_datetime()`'s candidate selection by hand for the Chile→Sydney
  set/set case to establish exactly which site corrections step 1 catches and which it does
  not (WR-05).
- Checked the new tests for coverage holes by fixture shape: every site-change test and the
  notebook's site-correction demo use `_make_run()`'s default **null/null** sub-night pair.
- Read the notebook's stored outputs (real executed output, `execution_count` 1-20, no
  placeholder cells) and the full runbook diff.
- No file other than this REVIEW.md was written; no test suite was run (per instructions).
- Iteration-7 deferrals recorded as `user_deferred:` in `35-VERIFICATION.md` are not
  re-raised.

## Structural Findings (fallow)

No structural pre-pass was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-04: a declined re-mint also declines the *non-destructive* refresh — the night's title, description and campaign are frozen forever, so `mark_cancelled` never reaches it

**File:** `solsys_code/allocation_projector.py:987-1012` (specifically the `continue` at
`:1012`)

**Issue:** The new decline branch does not merely skip the delete/create pair — it
`continue`s out of the per-night loop entirely, past the plain-update path at `:1077-1100`
that writes `title`, `description` (with the preserved dark-window line) and `target_list`.
Those three fields are not destructive rewrites and have nothing to do with the boundary
change the guard is refusing; they are the mechanism by which a staff action reaches an
allocation night at all. `allocation_night_description()`'s own docstring (`:151-169`) states
this explicitly:

> "reused deliberately so a staff ``mark_cancelled``/``mark_weather_failure`` action reaches
> allocation nights the same way it reaches container events."

That property is now broken for any declined night. And the decline is not transient:
nothing on this path records provenance, so `_span_needs_remint()` keeps returning True on
every subsequent sweep, and the night is declined — and therefore skipped — again, forever.

Trace (all four steps are ordinary staff actions):

1. Staff confirms an `ALLOC:` night's attribution (reachable via
   `CalendarEventMetaAdmin.save_model()`, `admin.py:373+`, which stamps `confirmed_by` on a
   run-link transition for any companion row), **or** sets `is_verified=False` on it (not in
   either surface's `readonly_fields`).
2. An operator corrects the run's `site`, or edits a sub-night window field →
   `_span_needs_remint()` returns True.
3. `_remint_decline_reason()` returns `'confirmed'`/`'staff_state'` → `detach_declined += 1`,
   `continue`.
4. The run is later marked CANCELLED. `allocation_night_title()` would now return
   `'[CANCELLED] NTT EFOSC2'` — but the night is declined again at step 3 and the title is
   never written. The calendar shows a cancelled run's night as an ordinary observing night,
   indefinitely, and the sweep reports `created: 0, updated: 0, unchanged: 0` for it (the
   notebook's own cell-37 output shows exactly that counter triple).

This is the same defect class iteration 7's CR-01 and iteration 8's CR-02 were rated
Critical for — admin-reachable, silent, permanent stale calendar data — reintroduced one
field over.

**Fix:** decline the destructive half only; fall through to the ordinary update path instead
of `continue`ing. Replace the `continue` at `:1012` with a flag that skips the re-mint block
and lets execution reach `:1077`:

```python
decline_reason = _remint_decline_reason(run, existing)
if decline_reason is not None:
    ...                                   # existing logging
    totals['detach_declined'] += 1
    active_urls.add(url)
    remint_declined = True                # do NOT continue
else:
    remint_declined = False

if not remint_declined:
    totals['retired'] += 1
    ...                                   # the whole existing re-mint block
    continue
# falls through to the plain-update path: title/description/target_list only,
# start_time/end_time untouched -- exactly what the decline promised to preserve.
```

Note the counter consequence to decide deliberately: with the fall-through, a declined night
reports `detach_declined: 1` **and** `updated: 1`/`unchanged: 1`. That is more honest than
today's "counted nowhere but `detach_declined`", but it must be stated in the runbook's
counter section and pinned by a test (`test_declined_night_still_receives_a_cancelled_title`).

---

### CR-05: the retirement branch deletes the allocation night itself with no human-confirmation guard — the guard directly above it protects only the legacy `RUN:` event

**File:** `solsys_code/allocation_projector.py:889-944` (specifically `:938-940`)

**Issue:** Iteration 8's CR-01 listed "the retired branch (`:802-819`, added as CR-03 in an
earlier iteration)" among the paths that already apply the UAT-2026-09-09 Option B rule.
Re-reading the branch line by line shows that is false. The guard block at `:899-937`
operates exclusively on `legacy_event` (the `RUN:{pk}:{night}` row) — its own comment says so
("the legacy RUN:{pk}:{night} event this retirement would also delete gets the SAME two
guards"). The allocation night itself is deleted unconditionally four lines later:

```python
if not dry_run:
    if existing is not None:
        existing.delete()          # <-- no _clearable_declined_and_unattributed() anywhere
    if legacy_deletable:
        legacy_event.delete()      # <-- guarded
```

The only gate `existing` passed is `_may_write(existing, run)` at `:883`, and `_may_write()`
returns True for `meta.run_id == run.pk` **regardless of `confirmed_by`** — which is precisely
the shape CR-03/CR-04 and this round's CR-01 exist to protect. `CalendarEventMeta.event` is
`OneToOneField(primary_key=True, on_delete=CASCADE)` (`models.py:59-65`), so the delete takes
`confirmed_by`, `confirmed_at`, `observation_record`, `observation_group` and `is_verified`
with it, with no warning log, no `detach_declined`, and `retired: 1` reported as ordinary
work.

Reachability is the same two-step as CR-04's, and it is *easier* than the re-mint shape
because it needs no boundary edit at all:

1. Staff confirms the `ALLOC:` night's attribution on the standalone admin page
   (`CalendarEventMetaAdmin.save_model()` stamps `confirmed_by`/`confirmed_at`).
2. Anyone links an `ObservationRecord` with a placed/observed block on that night — which
   also fires `receiver_on_run_observation_save()` (`:1159`) immediately, so no sweep is
   needed. `night in retired` → the confirmed row is destroyed.

The module is internally inconsistent about this exact decision: the D-14 convergence step
(`:1114-1131`) declines a confirmed delete for a **window shrink** — a night equally
"genuinely going away" — and has two dedicated tests
(`TestFinalConvergenceGuard.test_window_shrink_never_deletes_a_night_human_confirmed_to_this_run`).
The retirement branch has one test, `TestRetirePathLegacyEventGuard`, and it asserts only
about `legacy_event`. No test anywhere confirms an `ALLOC:` night and then retires it.

**Fix:** apply the same two-way split the re-mint branch just received, to `existing` itself:

```python
if night in retired:
    retired_urls.add(url)
    ...                                    # existing legacy_event guard block, unchanged
    existing_deletable = True
    if existing is not None:
        deletable_ids, confirmed_declined = _clearable_declined_and_unattributed(
            run, CalendarEvent.objects.filter(pk=existing.pk)
        )
        if existing.pk not in deletable_ids:
            logger.warning(
                'Allocation retire declined: night pk=%s is human-confirmed to run pk=%s '
                '-- an automated retirement never destroys it.', existing.pk, run.pk,
            )
            totals['detach_declined'] += 1
            existing_deletable = False
            active_urls.add(url)           # keep D-14 convergence off it
    if not dry_run:
        if existing is not None and existing_deletable:
            existing.delete()
        if legacy_deletable:
            legacy_event.delete()
    if existing_deletable:
        totals['retired'] += 1
    continue
```

Add `TestRetirePathAllocationEventGuard.test_retiring_a_night_never_deletes_a_human_confirmed_alloc_event`
(confirm the `ALLOC:` companion row, then `_link_record()` a placed block on that night;
assert the event and the stamp both survive, `retired == 0`, `detach_declined == 1`), plus a
dry-run parity case. Decide and document whether `is_verified=False`/an
`observation_record` link should also decline here (`_remint_decline_reason()`'s rule 2) or
whether only `confirmed_by` applies — the re-mint branch and this branch answering that
question differently is defensible, but it has to be stated, not left to the reader to infer
from two different call sites.

## Warnings

### WR-01 (carried forward from iteration 8, unaddressed): `--dry-run` still pays one `sun_event()` call per unrecorded night on every invocation, and the branch comment still says the opposite

**File:** `solsys_code/allocation_projector.py:1014-1017` (the comment), `:571` + `:593-594`
(the behaviour), `:533-534` (the docstring's cost bound)

**Issue:** Verbatim unchanged this round. The comment still reads "Both halves are skipped
under dry_run (no `sun_event()` call either), so a dry-run preview and a real run agree on
the same pair of counters", while `_span_needs_remint(..., dry_run=True)` reaches `:571` and
calls `sun_event()` for every unrecorded night, and `:593` deliberately skips the recording —
so the next dry run pays the same cost again. The docstring's "at most one
`sun_event(kind='sun')` call per unrecorded night, **once ever**" (`:533-534`) is still
unqualified and still holds only in real mode. Now materially larger than in iteration 8: the
version-prefix test (CR-02's fix) makes **every** pre-release token unrecorded too, so the
first post-upgrade `--dry-run` pays this for every existing `ALLOC:` night, and pays it again
on every repeat preview.

**Fix:** as in iteration 8 — correct the `:1014-1017` comment to say the re-mint *write* is
skipped under `dry_run` while the unrecorded-provenance *resolution* still runs, qualify the
`:533-534` bound with "in real mode; a `--dry-run` preview repeats the call because it may
not record what it proves", and pin the per-dry-run call count with a test.

---

### WR-02 (carried forward from iteration 8, unaddressed — with one correction to its original impact claim): a read-only `--dry-run` preview can still raise `sun_event()`'s `ValueError`

**File:** `solsys_code/allocation_projector.py:571`; conflicting rationale at `:1051-1071`

**Issue:** Unchanged. The create branch still carries the WR-03 comment explaining why
`_mint_fields()` must not be called under `dry_run` ("...could raise `sun_event()`'s own
`ValueError` (e.g. a blank `Observatory.timezone`) on what the module's own docstring
documents as a read-only preview"), while `:571` does exactly that on the legacy branch. The
module contradicts itself in two comments 480 lines apart.

**Correction to iteration 8's own text, made while re-verifying it:** WR-02 claimed "one bad
`Observatory.timezone` now aborts the whole preview sweep for every run after it". That is
**wrong** — `reconcile_campaign_runs.py:62-68` wraps each `reconcile_run()` call in a per-run
`try/except Exception` that logs, writes `Run pk=N: reconcile failed (...) -- skipping` and
increments `failed_count`. The blast radius is one run per bad site, not the sweep. The
finding stands (a preview that used to complete now fails, and the staff `_resolve_site()`
path at `campaign_views.py:724-733` swallows it into a generic "use Resolve to retry"
message that can never succeed), but at reduced severity.

**Fix:** as in iteration 8 — either state the trade-off at `:571` and amend the `:1051-1071`
comment, or degrade a preview to "cannot decide, report `unchanged`" on `ValueError` when
`dry_run` is set.

---

### WR-03 (carried forward from iteration 8, partially closed): the `retired` enumeration and deploy note landed; the `rekeyed` paragraph's stability promise did not

**File:** `docs/runbooks/telescope_runs_calendar.rst:1104-1108`

**Issue:** Plan 35-22 closed most of this: `retired` now enumerates five reasons
(`:1079-1102`), the `detach_declined` paragraph describes both meanings (`:1142-1167`), and
the post-upgrade deploy note WR-04(b) asked for is present (`:1169-1178`). Not closed: the
`rekeyed` paragraph still tells operators

> "``rekeyed`` counts a night carried across from the old, retired ``RUN:{pk}:{date}`` key
> form into the current ``ALLOC:{pk}:{night}`` form, in place -- same primary key, same
> start/end time, just re-keyed."

The re-key path (`:967-979`) deliberately does **not** record provenance, so on the *next*
sweep that same night enters `_span_needs_remint()`'s legacy branch (`:569-595`) and is
re-minted — new primary key, new boundaries — whenever its carried-over boundary sits more
than a minute from the computed sun event. The promise is true of the re-key itself and false
one sweep later, which is exactly when an operator checks.

**Fix:** append to that paragraph: *"The 'same primary key, same start/end time' guarantee
covers the re-key itself. A re-keyed night carries no mint provenance, so the next sweep
resolves it once under ``retired`` reason (5) and may re-mint it at the computed sun
event — see the deploy note below."*

---

### WR-04 (carried forward from iteration 8, partially closed): the deploy note landed; the unverifiable premise and the odd-one-out fixture convention did not

**File:** `solsys_code/allocation_projector.py:67-76` (`_UNRECORDED_PROVENANCE_TOLERANCE`),
`solsys_code/tests/test_cutover_classical_allocations.py:434-454`

**Issue:** Part (b) is done — the runbook's post-upgrade deploy note (`:1169-1178`) now tells
operators to run `--dry-run` first and reads a non-zero `would_retire` as the one-time audit.
Parts (a) and (c) are untouched: the constant's comment (`:67-76`) is byte-identical and
still names no retired writer and no audit query an operator could run to check the premise
before the sweep, and `_make_three_night_group()`'s round-hour convention is still the
odd one out relative to the one test that was rebuilt around real `sun_event()`-derived
boundaries. The premise remains uncheckable from the tree (`grep -rn 'sun_event'
solsys_code/management/commands/` still returns nothing).

**Fix:** as in iteration 8 — (a) name the retired writer and the release it belonged to in
the constant's comment, or give the operator a one-off audit query; (c) either propagate the
realistic boundaries into `_make_three_night_group()` or comment there why the round-hour
convention is still acceptable for the ~30 tests that never run a sweep.

---

### WR-05: "a site correction re-mints" is false for a run with both sub-night fields set — step 2 returns before the token is ever read, and every test and the notebook demo use the null/null shape

**File:** `solsys_code/allocation_projector.py:557-558`; claims at
`docs/runbooks/telescope_runs_calendar.rst:1084-1090`, `35-21-PLAN.md:24`,
`solsys_code/tests/test_allocation_projector.py:1997-2023` and `:2230-2244`

**Issue:** `_span_needs_remint()` short-circuits before the provenance comparison whenever
both sub-night fields are set:

```python
if run.night_start_utc is not None and run.night_end_utc is not None:
    return False          # <-- the token, and therefore the site, is never consulted
```

So CR-02's widened token governs only the null/null and half-null shapes. For a fully-set
run, a site correction produces one of two outcomes, neither of them the documented one:

- **Same timezone** (e.g. one Chilean site to another): `_night_span_utc()` is unchanged,
  both step-1 comparisons match, `return False`. No re-mint, reported `unchanged`. The
  operator-set boundaries are legitimately unchanged — but the event's **description keeps
  the old site's `Dark window (-15 deg, UTC): ...` line forever**, because the plain-update
  path reuses it verbatim via `preserved_dark_window_line()` (`:1078`). That line is the one
  site-derived field on the update path, and nothing ever refreshes it.
- **Different timezone** (the documented La Silla → Siding Spring case): I traced
  `_time_of_day_to_datetime()` by hand for 23:00/05:00 moved from `America/Santiago` to
  `Australia/Sydney` on 2026-07-09 — the end comparison flips (stored `07-10T05:00Z` vs.
  newly resolved `07-09T05:00Z`), so step 1 returns True and the branch re-mints, but
  `night_bounds()` then resolves `start=07-09T23:00Z >= end=07-09T05:00Z` and
  `_raise_if_inverted()` raises. The run is reported `failed` by the command, and
  `campaign_views._resolve_site()` (`:724-733`) catches it into "use Resolve to retry" —
  a retry that can never succeed, with `site_needs_review` left set.

Meanwhile the runbook now states the opposite as fact ("a correction to the run's ``site``
... so the night is deleted and re-created fresh, at the corrected site's real
sunset/sunrise"), and 35-21's own success criterion claims "re-mints **every one of its
nights**". The gap is invisible to the test suite because `_make_run()` defaults
`night_start_utc`/`night_end_utc` to null, and **both** site tests
(`TestSiteChangeRemints`, `TestMintInputInvariant.test_changing_the_site_remints`) and the
notebook's site-correction demo (cell 35, on `classical_run`) use that default. A classical
run with a fixed `1130-0530` window — the very thing these fields exist for — is the untested
shape.

**Fix:** pick one and make the docs match:
1. Refresh the dark-window line on the update path when the token's site component differs
   (costs one `sun_event(kind='dark')` call on exactly that transition, not on an idempotent
   sweep, so D-13 and `TestNoSunEventRecompute` are unaffected); or
2. Let a site change re-mint even for a set/set run by moving the `:557-558` short-circuit
   *after* the version/token comparison; or
3. Document the limitation honestly in both the runbook and
   `_span_needs_remint()`'s docstring step 2 ("a fully-set sub-night pair pins both
   boundaries, so a site correction changes only the dark-window line, which is not
   refreshed").
In all three cases, add a set/set fixture to `TestSiteChangeRemints`, and pin the
cross-timezone inverted-span outcome so it is a decision rather than a surprise.

---

### WR-06: `detach_declined`'s two operator-facing messages still say "left attributed -- someone had already confirmed them", which is false for the new `staff_state` decline

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:126-131`;
`solsys_code/campaign_views.py:467-472`

**Issue:** 35-20 routed a second, unrelated meaning into `detach_declined` (a re-mint the
sweep declined to perform, possibly for `is_verified=False` or an `observation_record` link
with no human confirmation anywhere). The runbook (`:1142-1167`) and the notebook (cell 31)
were both updated to describe both meanings. The two surfaces an operator actually reads were
not:

```
Run pk=7: 1 superseded entry left attributed -- a person confirmed them, and an
automated sweep never clears a human confirmation
```

```python
messages.info(request, f'{result.detach_declined} superseded entries left attributed '
                       '-- someone had already confirmed them.')
```

For a `'staff_state'` decline both clauses are false: nothing was "left attributed" (no
attribution was at stake) and nobody confirmed anything. For a `'confirmed'` re-mint decline
the first clause is still wrong — a boundary correction was skipped, not an attribution
retained. Worse, the runbook now tells the operator there IS a remedy for a declined re-mint
("clear the confirmation or the link ... and re-run the sweep"), while the message tells them
a human decision was respected and there is nothing to do. This is the same defect NF-16 was
raised for one counter over ("'blocked's message reads 'owned by someone else' — false twice
over for this shape").

**Fix:** widen both messages to name both causes without claiming either, e.g. *"N calendar
decision(s) declined -- a person's confirmation, an observation link, or an unverified row
outranks this automated sweep; see the runbook's ``detach_declined`` section"*. If the two
meanings need to stay distinguishable to an operator, split the counter instead
(`detach_declined` / `remint_declined`) — which also removes CR-04's counter ambiguity.

---

### WR-07: a declined night with unrecorded provenance re-computes `sun_event()` on every sweep, forever — the documented "once ever" cost bound and D-13's astropy-free idempotent sweep no longer hold in real mode either

**File:** `solsys_code/allocation_projector.py:571`, `:579-595`, `:987-1012`; docstring bound
at `:533-534`

**Issue:** `_span_needs_remint()` only records provenance on the **within-tolerance** path
(`:593-594`). A night that resolves *stale* returns True at `:592` without recording — and
with CR-01's new guard, a stale night carrying staff state is now **declined** instead of
re-minted, so it never gets a fresh token from the re-mint path either. The result is a
permanent state in which every single sweep:

- calls `sun_event(run.site, night, kind='sun')` (real astropy work, in real mode, on an
  otherwise idempotent sweep — the thing D-13 and `TestNoSunEventRecompute` exist to prevent),
- emits an `Allocation unrecorded-provenance night ...` warning,
- emits an `Allocation re-mint declined ...` warning,
- and reports `detach_declined: 1`.

The docstring's stated bound — "at most one `sun_event(kind='sun')` call per unrecorded
night, once ever" — is now false in real mode, not only under `--dry-run` (WR-01). On a
first post-upgrade sweep every legacy night is unrecorded, so any of them carrying a
confirmation or `is_verified=False` joins this permanent loop.

**Fix:** qualify the `:533-534` bound to name both escapes (a dry run, and a night whose
re-mint is declined), and consider recording the *resolved* token on the declined path so
the night stops being re-resolved — but only together with CR-04's fall-through, since
recording provenance on a night whose boundaries were NOT re-minted would re-create exactly
the false-provenance claim round 2 was reverted for. The safe version is a separate
"resolution attempted, declined" marker, or simply accepting the cost and saying so.

---

### WR-08: `is_verified` is now a load-bearing veto on automated re-mints, but the model docstring still documents it as a vestigial field with no current writer or reader

**File:** `solsys_code/models.py:22-31` and `:69-71`;
`solsys_code/allocation_projector.py:647`

**Issue:** `_remint_decline_reason()` gives `is_verified=False` a new, permanent, production
consequence: an allocation night with that flag can never be corrected by an automated
sweep again (see WR-07 for what that costs). `models.py`'s WR-06 paragraph — the place a
reader goes to find out what the field means — still says the opposite:

> "as of Phase 34, no writer in this codebase sets ``is_verified=False`` any more ... it is
> not currently reachable by re-running any sweep or receiver. The two ``calendar.html``
> template branches keyed on ``is_verified == False`` are consequently unreachable..."

Two concrete consequences of overloading it rather than adding an explicit field:
(a) a *historical* `is_verified=False` row (the docstring says such rows exist from before
Phase 34) silently acquires veto power it was never given deliberately; (b) `is_verified` is
the one companion-row field **not** in either admin surface's `readonly_fields`, so any staff
user can freeze a night against automated correction from a checkbox labelled "Whether the
telescope label was live-verified against the LCO API" — with no hint at that surface that
this is what they are doing.

**Fix:** at minimum, extend the WR-06 paragraph and the field's `verbose_name`/help text to
state the second meaning, and say so in the runbook's `detach_declined` section (which
currently says "an unverified companion row" without explaining where that comes from).
Better: give the re-mint veto its own explicit concept rather than borrowing a field
documented as meaningless, or drop the `is_verified` arm and keep only the two link fields
plus `confirmed_by`.

## Info

### IN-01 (carried forward from iteration 8, mitigated but not fixed): `null=True` + `blank=True` on a `CharField` still leaves `''` as a third state

**File:** `solsys_code/models.py:130-132`,
`solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py:16`

The field is still `CharField(max_length=64, null=True, blank=True)`. CR-02's version-prefix
test materially **defuses** the consequence iteration 8 described: `''.startswith('v2|')` is
False, so an empty string now reads as unrecorded, resolves once through the legacy branch and
self-heals by recording a real token — and
`test_empty_string_token_reads_as_unrecorded_and_resolves_once` pins exactly that. The
three-state ambiguity itself remains (`NULL`, `''`, token), for a field no form is allowed to
bind.

**Fix:** drop `blank=True` (nothing needs to submit it) in a follow-up schema-only migration,
or leave it and note in the field comment that `''` is treated identically to `NULL` by
design.

---

### IN-03 (carried forward from iteration 8, unaddressed): the class docstring still says "no row at all means verified by documented default" without noting the projector materialises rows purely to record provenance

**File:** `solsys_code/models.py:12-57` (the sentence at `:16-18`),
`solsys_code/allocation_projector.py:449-464`

`_record_sub_night_provenance()` still uses `update_or_create(event=event, ...)`, and
`_span_needs_remint()`'s legacy branch (`:593-594`) now calls it for any night whose
provenance reads as unrecorded — which, after CR-02's version test, includes every
pre-release row in the database. So the "no row at all" shape is converted to "row with
`run IS NULL`" on the first post-upgrade sweep, at scale. Verified harmless (`_may_write()`
and `_clearable_declined_and_unattributed()` treat the two shapes identically), but the
class docstring was extensively rewritten this round for CR-02 and still does not mention it.

**Fix:** one sentence in the class docstring: "the allocation projector may materialise a row
solely to record `minted_sub_night_window`, so the absence of a row is not evidence that no
projection has touched the event."

---

### IN-05: `active_urls.add(url)` in the decline branch is a no-op, under a comment asserting it is load-bearing

**File:** `solsys_code/allocation_projector.py:1008-1011`

```python
# Load-bearing, not cosmetic: without this, the D-14 convergence step at the
# bottom of this function deletes the very night this guard just refused to
# delete.
active_urls.add(url)
```

`active_urls.add(url)` already ran at `:946`, unconditionally, for every night that is not
retired — and the decline branch is only reachable from `:981`, well past it. The set add is a
no-op and the rationale is false as written. It is harmless today, but a comment that claims
a line is protecting against deletion is exactly the kind of statement a later refactor
trusts instead of re-deriving. (If CR-05's fix adds the same call to the *retired* branch,
that one genuinely will be load-bearing — which makes the distinction worth getting right
now.)

**Fix:** delete the line and the comment, or keep the line as a deliberate belt-and-braces
and reword the comment to "redundant with `:946`; kept so this branch does not depend on a
distant caller's bookkeeping".

---

### IN-06: two of `_remint_decline_reason()`'s three staff-state arms are unreachable for `ALLOC:` nights outside a test

**File:** `solsys_code/allocation_projector.py:647`;
`solsys_code/tests/test_allocation_projector.py:1782-1841`

`observation_record` and `observation_group` are admin-readonly on both surfaces
(`admin.py:335-341`, `:119-125`) and are written only by the observation projector, which
owns facility-url events, never `ALLOC:`-keyed ones — `_sync_observation_attribution()`
(`:721-788`) is explicitly scoped out of this namespace. So on an allocation night those two
fields can only be set by a direct ORM `.update()`, which is exactly how both tests reach
them. The plan calls this "covered for the same reason at no extra cost", which is fair; it
is recorded here only so a future reader does not mistake two passing tests for evidence that
the production path exercises those arms.

**Fix:** none required. Optionally note in the docstring's rule 2 that the two link arms are
defensive, and `is_verified` is the only production-reachable one (which the docstring
already half-says).

---

_Reviewed: 2026-09-16T16:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
