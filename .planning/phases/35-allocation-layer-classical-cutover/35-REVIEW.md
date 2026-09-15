---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-15T00:00:00Z
depth: deep
iteration: 6
prior_review: 35-REVIEW.md (git show 894c982)
diff_base: 894c982c62c4c78655ee0959eac5ad2ed2a2d704
files_reviewed: 10
files_reviewed_list:
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_load_telescope_runs.py
prior_findings:
  total: 4
  closed: 2
  partially_closed: 2
  still_open: 0
  deferred: 1
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 35: Code Review Report (iteration 6 — re-review of the `894c982..HEAD` second gap-closure round)

**Reviewed:** 2026-09-15
**Depth:** deep
**Files Reviewed:** 10
**Status:** issues_found

## Summary

This re-reviews plans 35-12 through 35-15 (commits `e0f1f66`, `478852d`, `479c529`,
`ffd5174`, `2448264`, `909a67a`, `9bb2b64`, `25f41de`, `e8015e3`) against iteration 5's one
BLOCKER and three attempted warnings. WR-04 (`observation_projector.py`'s savepoint-less
swallowed database error) was deliberately deferred this round as advisory — its absence is
**not** treated as a regression here.

**The blocker is genuinely closed this time.** CR-01's predicate inversion holds on *both*
the real and the `--dry-run` path, refuses the no-marker claimant, leaves the run
byte-identical, and the test that previously pinned the defect has been replaced with its
inverse plus the destructive-case regression the suite was missing. Reproduced by probe:
`Done (dry run). candidates: 3, groups: 1, runs created: 0, updated: 0, unchanged: 0, events
re-keyed: 0, unexplained: 3` with `CommandError`, `run_status` still `planned`, the staff
note intact, all three events still `url=''`.

**But the loop's dominant failure mode recurred again, in the same two places.** Both
"partially closed" iteration-5 findings are partially closed *again*, and one of the two
fixes introduced a failure mode that did not exist before it:

- **WR-01's fix is half a fix and half a new bug.** `_raise_if_set_window_inverted()` now
  falls back to `existing.start_time`/`existing.end_time` for the null sub-night field, on
  the stated premise that the stored boundary "was minted from the same deterministic
  `sun_event()` for the same site and night". That premise is **false whenever the
  now-null field was previously SET** — the stored boundary is then the old operator value,
  not a sun event. Reproduced in **both** directions on a realistic `2300-EoN` → `BoN-2230`
  edit: the dry run raises `ValueError` for a night the real run creates cleanly (**new**
  false alarm), and — with the stale boundary on the other side of sunset — the dry run
  returns clean for a night the real run refuses (**WR-01's original symptom, verbatim,
  fourth iteration running**). Re-filed as **WR-01**.
- **WR-02's fix closes the arm it was filed against and leaves its twin open.** The
  existing-run arm now folds after the preview reconcile, and the probe confirms
  `(0,0,0,1)` on both passes. The **create** arm (`existing is None`) never calls
  `reconcile_run()` at all, so a brand-new line whose real reconcile raises still previews
  `created: 1, skipped: 0` against the real run's `created: 0, skipped: 1`. That is WR-02's
  own sentence — "an operator reading the dry run believes a run will be created when in
  fact the line will be dropped" — surviving in the sibling branch, and the loader
  notebook's freshly committed output prints `the preview never disagrees with the real
  run` directly underneath it. Re-filed as **WR-03**.

A third, separate harm surfaced while probing CR-01's *surviving* permissive branch: a
claimant whose marker **matches** is still find-and-updated, so a staff `run_status`
edit made after the import is silently reverted with **exit 0** — and the runbook's new
CR-01 remedy paragraph instructs the operator to restore the marker and re-run, which is
exactly the action that triggers it (**WR-02**, new).

### Prior-finding verification: 2 closed, 2 partially closed, 1 deferred

| Prior finding | Verdict |
|---|---|
| **CR-01** cutover identity guard permissive on a missing `Source line:` marker (BLOCKER) | **Closed.** `cutover_classical_allocations.py:426-445` is now `if existing_source_line != source_line:` with a two-branch reason string. Verified by probe on **both** passes (real pass: `TestDatabaseScopedIdentityGuard` tests 4 and 5; `--dry-run`: my own PROBE-P3 — `unexplained: 3`, `duplicate_identity=3`, `CommandError`, `run_status='planned'`, `observation_details` still `'Rescheduled per PI request; see ticket OPS-4412.'`, all three events still `url=''`). No third sub-case survives: `existing_source_line` is `str \| None`, `source_line` is always a non-`None` `str` (the groups dict is keyed on it), so `None` now lands on the refusing side; `source_identifier` carries a `UniqueConstraint` (`models.py:413-417`) so `.first()` cannot disagree with `get_or_create()`'s match; and only `load_telescope_runs` and this command ever write `source_identifier`, both via `f'…\nSource line: {line.strip()}'`, so the benign cutover-after-import ordering still matches byte-for-byte (verified against the pre-rewrite loader at `git show 6ec5955^`). The three named doc statements (module docstring `:45-62`, `CommandError` `:683-690`, runbook `:946-968`) all now carry the marker precondition. **The matching-marker branch's own destructive update is a separate, new finding — WR-02.** |
| **WR-01** dry-run inversion guard skips the half-null shape (WARNING) | **Partially closed, and regressed in the other direction.** The shape the new test pins (`night_start_utc` set, `night_end_utc` null, stored end = a real sunrise) is genuinely fixed — my PROBE-P2 confirms dry and real now raise the **identical** message. But the stored-boundary fallback is wrong whenever the null field was previously set. Re-filed as **WR-01**. |
| **WR-02** loader dry/real counter double-count (WARNING) | **Partially closed.** `load_telescope_runs.py:312-326` folds after the preview reconcile; PROBE-P5's existing-run arm reproduces `(0,0,0,1)` on both passes, matching the new `test_dry_run_and_real_run_report_the_same_counters_for_a_skipped_line`. The create arm still diverges. Re-filed as **WR-03**. |
| **WR-03** stale third copy of the `claimed_legacy_urls` contract (WARNING) | **Closed.** `campaign_reconciler.py:600-607` now carries the same four-outcome, real-mode-load-bearing wording as its two siblings (`:750-760`, `allocation_projector.py:595-617`). I checked for a fourth copy: `campaign_reconciler.py:842-844` is a neutral inline comment that enumerates no outcomes and repeats no superseded claim — correctly left alone. |
| **WR-04** `observation_projector.py` savepoint-less swallowed DB error | **Deferred by design this round** (advisory). Not re-verified, not counted as a regression. |
| **IN-01** shadowed `existing_run` re-query | **Closed.** `cutover_classical_allocations.py:542-548` reuses the guard's binding; the redundant query is gone and the reason is commented. |

### Verification method

`python manage.py test solsys_code.tests.test_allocation_projector
solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations
solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_observation_projector_signals`
— **219 tests, all passing.** `pre-commit run ruff --all-files`, `pre-commit run
ruff-format --all-files` and `pre-commit run sphinx-build --all-files` — **all Passed.**

WR-01, WR-02 and WR-03 are backed by **executed probes** against a real Django test
database (a probe module written under `solsys_code/tests/`, run, then deleted;
`git status --short` confirms **no source file was modified by this review**). Reproduced
facts, not inferences:

```
PROBE-P1 (WR-01, false positive) -- La Silla, one night, minted 2300-EoN then edited to BoN-2230:
  sunset/sunrise UTC      : 2026-07-09 22:06:35.917  2026-07-10 11:29:46.816
  minted span             : 2026-07-09 23:00:00+00:00 -> 2026-07-10 11:29:46+00:00
  dry run RAISED          : Computed an inverted allocation-night span for run pk=1
                            night=2026-07-09: start=2026-07-09T23:00:00+00:00 >=
                            end=2026-07-09T22:30:00+00:00.
  real run returned       : ReconcileResult(created=1, retired=1, ...)
  event after real run    : (2026-07-09 22:06:35+00:00, 2026-07-09 22:30:00+00:00)   <- valid
  DIVERGENCE              : True

PROBE-P6 (WR-01, false negative) -- same site, minted 2100-EoN then edited to BoN-2130:
  minted span             : 2026-07-09 21:00:00+00:00 -> 2026-07-10 11:29:46+00:00
  dry run returned        : ReconcileResult(created=1, retired=1, ...)   <- no error
  real run RAISED         : ... start=2026-07-09T22:06:35+00:00 >= end=2026-07-09T21:30:00+00:00
  DIVERGENCE              : True

PROBE-P2 (WR-01, the shape the new test pins) -- stored boundary IS sun-derived:
  dry RAISED / real RAISED, identical message.  PARITY: True

PROBE-P3 (CR-01, --dry-run path):
  Done (dry run). candidates: 3, groups: 1, runs created: 0, updated: 0, unchanged: 0,
                  events re-keyed: 0, unexplained: 3
  CommandError raised; run_status 'planned'; details 'Rescheduled per PI request; see
  ticket OPS-4412.'; urls ['', '', '']

PROBE-P4 (WR-02) -- claimant whose marker MATCHES, run_status set to CANCELLED by staff:
  exit_nonzero            : False
  Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0,
        events re-keyed: 3, unexplained: 0
  run_status after        : planned     (was cancelled)

PROBE-P5 (WR-03) -- brand-new line, NTT timezone typo'd, no existing CampaignRun:
  dry : Done (dry run). lines processed: 1, created: 1, updated: 0, unchanged: 0, skipped: 0
  real: Done.           lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1
  AGREE: False ; CampaignRun rows in db afterwards: 0
```

### Positives worth recording

The CR-01 fix is the best work this phase has produced. It is not just a predicate flip: the
reason string branches so each of the two causes gets its own operator action, the module
docstring, the `CommandError` text and the runbook all name the precondition instead of
restating a guarantee, and — the part previous rounds kept missing — the *test that pinned
the defect was inverted rather than deleted*, with
`test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` added to assert
the three fields the old fixture's `observation_details=''` had made invisible. The
`TestDryRunAndRealRunAgree` fixture change (adding a matching marker so the group still
reaches the per-event preconditions) shows the author traced which *other* tests the new
refusal would silently reroute, instead of chasing a red bar.

Both notebooks are genuine regenerated output, not hand patches: `reconcile_campaign_runs_demo`
carries execution counts 1..18 with no nulls and five occurrences of the corrected
`CommandError` text (`updates an existing CampaignRun only when that run's stored Source
line: matches…`) and zero of the superseded `rewrites no existing` clause;
`load_telescope_runs_demo` carries 1..16 with the new WR-02 parity cell printing the real
`(0, 0, 0, 1)` on both passes. The runbook's new paragraph even volunteers the honest
sentence most doc fixes would omit — *"This is reachable through the very Django-admin edit
this paragraph itself asks the operator to perform."*

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

None. Iteration 5's CR-01 is closed (see the verification table and PROBE-P3 above).

## Warnings

### WR-01: the new half-null fallback assumes the stored boundary is sun-derived — false whenever the null field was previously set, so the dry run now raises on valid nights *and* still misses inverted ones

**File:** `solsys_code/allocation_projector.py:357-375` (the fallback at `:360-369`; the
premise stated at `:338-344`), reached from `:756` (re-mint branch)
**Severity:** WARNING

**Issue:** The fix reaches the half-null shape by reading the missing boundary off the
stored event:

```python
start = (
    _time_of_day_to_datetime(run.night_start_utc, night, night_span)
    if run.night_start_utc is not None
    else (existing.start_time if existing is not None else None)
)
```

Its justification is stated twice, in the docstring (`:338-344`) and at the call site
(`:751-755`): *"the boundary already stored on the re-mint branch's own night, which was
minted from the same deterministic `sun_event()` for the same site and night"*. That holds
only if the field was null **at mint time too**. The re-mint branch is reached precisely
because the run's sub-night fields **changed**, and one of the changes an operator can make
is nulling a previously-set field — `2300-EoN` → `BoN-2230`, i.e. "start at sunset instead
of 23:00, end at 22:30 instead of sunrise". After that edit `existing.start_time` is
`23:00`, an old operator value, and the guard compares it against a boundary the real run
will never pair it with.

Both directions are reachable, and both are reproduced above:

- **False positive (new, introduced by this fix):** stale stored boundary *after* sunset.
  PROBE-P1 — dry run raises `ValueError: … start=2026-07-09T23:00:00+00:00 >=
  end=2026-07-09T22:30:00+00:00`; the real run creates the night `22:06:35 → 22:30`
  without complaint. Before this round the guard simply returned early for a half-null run,
  so a preview that *aborts a night the real run handles* is a behaviour this fix
  introduced. `reconcile_campaign_runs --dry-run` reports the run under `failed:`
  (`reconcile_campaign_runs.py:64-67` contains it per-run), and `load_telescope_runs
  --dry-run` folds the line into `skipped` — the exact WR-02-shaped preview/real
  disagreement, now pointing the other way. An operator following the runbook's "always run
  `--dry-run` first" will go and "correct" data that is already correct.
- **False negative (WR-01's original symptom, unchanged):** stale stored boundary *before*
  sunset. PROBE-P6 — dry run returns `ReconcileResult(created=1, retired=1, …)` with no
  error; the real run raises `… start=2026-07-09T22:06:35+00:00 >=
  end=2026-07-09T21:30:00+00:00`.

`test_dry_run_of_a_half_null_remint_inverted_window_also_raises` cannot see either case: it
keeps `night_end_utc` null from mint through edit, so its `existing.end_time` really *is*
the sunrise, which is the one sub-shape the fallback is sound for.

**Fix:** only trust the stored boundary when it provably came from a sun event — i.e. when
the field was null at mint time as well as now. The run row does not record that, but the
event's own description does: `_mint_fields()` writes the dark-window line, and the stored
boundary is sun-derived exactly when the corresponding sub-night field is null *and* the
night was minted under the same null. The cheapest correct option is to stop guessing and
narrow the guard to what it can prove, restoring parity by *silence* rather than by a wrong
answer:

```python
    # WR-01 (iteration 6): only fall back to a stored boundary when it is provably the
    # sun-derived one -- i.e. when the event was minted while this same field was null.
    # A field the operator has just NULLED leaves its old value in start_time/end_time,
    # which the real run will replace with sun_event(); comparing against it produces a
    # preview that disagrees with the real run in BOTH directions (PROBE-P1/P6).
    start = _time_of_day_to_datetime(run.night_start_utc, night, night_span) if run.night_start_utc is not None else None
    end = _time_of_day_to_datetime(run.night_end_utc, night, night_span) if run.night_end_utc is not None else None
    if start is None or end is None:
        return
```

…and, if the half-null re-mint case is worth previewing at all, do it by recording the mint
provenance rather than inferring it — e.g. store the resolved boundaries' origin in the
description alongside the dark-window line, or accept one `sun_event()` call on the
*re-mint* path only (D-13 forbids recomputing an **unchanged** night's boundary; this night
is by definition changing and is about to be deleted and re-minted anyway, so the call is
not the drift-rewrite D-13 bans). Whichever is chosen, add both PROBE-P1 and PROBE-P6 as
regression tests — the existing half-null test passes under the current bug.

### WR-02: the cutover's surviving matching-marker branch silently reverts a post-import staff edit with exit 0 — and the new CR-01 remedy text routes the operator straight into it

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:424-445`
(the permissive branch is the fall-through at `:445`), write at `:470-485` / `:551`;
remedy text at `:436-443` and `docs/runbooks/telescope_runs_calendar.rst:959-968`
**Severity:** WARNING

**Issue:** CR-01's guard proves **provenance** ("this run was created from this schedule
line"), and the code then treats that as **currency** ("so every field on it may be
overwritten from the line"). A `CampaignRun` that *did* come from this line, and whose
`run_status` a staff member has since changed in the admin, is find-and-updated back to the
line's status with no report beyond `runs updated: 1`.

Reproduced (PROBE-P4; claimant's `observation_details` = `'Status: allocation\nSource line:
NTT EFOSC2 allocation 9-12 July'` — a *matching* marker — with `run_status` set to
`CANCELLED` by staff):

```
P4 exit_nonzero: False
P4 stdout: Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0,
           events re-keyed: 3, unexplained: 0
P4 run_status after (was CANCELLED): planned
```

The `fields` dict at `:470-485` also unconditionally writes `target: None`, `campaign`,
`site`, `site_raw`, `window_start`/`window_end` and `observation_details` — so the staff
member's free-text note goes too, on a one-time migration, with exit 0 and empty stderr.
This is the same harm class CR-01 was filed for; CR-01 removed the *unprovable* half of it
and left the provable half untouched, which is defensible as find-or-update semantics but
is nowhere stated as a caveat.

What makes it worth a finding rather than a note is the interaction with the new remedy
text this very round added. The `duplicate_identity` no-marker message (`:436-443`) and the
runbook (`:959-968`) both tell the operator: *"restore or correct that run's
`observation_details` 'Source line:' text in the Django admin so it matches … then
re-run"*. An operator who follows that instruction converts a refused (safe) claimant into
a matching (silently-overwritten) one. The documentation's own worked remedy is the trigger.

**Fix:** either narrow the write, or say so loudly. Narrowing is cheap and matches D-18's
"report what you cannot explain" posture — compare the line-derived fields against the
claimant before writing and report a divergence instead of overwriting it:

```python
if existing_run is not None and existing_source_line == source_line:
    drifted = [f for f, v in fields.items() if f != 'observation_details' and getattr(existing_run, f) != v]
    if drifted:
        _mark_unexplained(
            events,
            _RUN_DRIFT,   # new named reason: "claimant disagrees with its own schedule line"
            f'CampaignRun pk={existing_run.pk} came from this Source line but its '
            f'{", ".join(drifted)} no longer match it -- an edit made after the import would be '
            'silently reverted; reconcile the run in the Django admin (or re-import the line), then re-run',
        )
        continue
```

If instead the overwrite is intended, add the caveat to the module docstring, the
`CommandError` and the runbook remedy paragraph in the same sentence that sends the
operator to the admin ("restoring the marker also makes the next run re-apply the schedule
line's `run_status`, `window` and `campaign` to that row"), and add PROBE-P4 as a test that
pins the intended outcome explicitly rather than leaving it unasserted.

### WR-03: WR-02's parity fix covers only the `existing is not None` arm — a brand-new line still previews `created: 1` and really runs `skipped: 1`, under a notebook cell that prints the opposite

**File:** `solsys_code/management/commands/load_telescope_runs.py:294-326` (the create arm
at `:307-310`); claim at `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
code cell 9 (`cells[18]`), and `docs/runbooks/telescope_runs_calendar.rst:82-89`
**Severity:** WARNING

**Issue:** The dry-run branch calls the raising `reconcile_run(existing, dry_run=True)`
**only** when a run already exists. When it does not, it predicts from the window length
(`night_created += len(nights)`) and folds `run_created += 1` — nothing in that arm can
fail. The real branch runs `write_and_reconcile_campaign_run()` for the same line, whose
reconcile *can* raise (NF-21's `ZoneInfoNotFoundError`, `sun_event()`'s `ValueError` for a
blank timezone or a polar site, WR-01's create-path inverted span), and reports the line
under `skipped`.

Reproduced (PROBE-P5; NTT timezone typo'd, **no** pre-existing `CampaignRun` — the exact
state `TestMalformedTimezoneSkipsOneLine.setUpTestData` seeds):

```
dry : Done (dry run). lines processed: 1, created: 1, updated: 0, unchanged: 0, skipped: 0
real: Done.           lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1
AGREE: False ; CampaignRun rows in db afterwards: 0
```

The four counters still sum to `lines processed` on both passes, so the runbook's new
paragraph (`:82-89`) is literally true — but its neighbouring sentence, and the notebook
cell committed one commit later, both promise more than that. The notebook's executed
output ends with:

```
Both passes agree: (0, 0, 0, 1) -- the preview never disagrees with the real run.
```

printed from a fixture deliberately chosen to take the `existing is not None` arm. That is
the strongest claim in the phase's operator-facing documentation, and PROBE-P5 falsifies it
with the same command, the same site and the same typo'd timezone.

**Fix:** give the create arm the same failure surface the real run has, so the two passes
can only disagree when nothing can be known:

```python
if existing is not None:
    reconcile_result = reconcile_run(existing, dry_run=True)
    ...
else:
    # WR-03 (iteration 6): the real branch runs the SAME reconcile for a brand-new run, and
    # a raising reconcile there yields `skipped` alone. Preview it against a transient,
    # rolled-back row so the create arm cannot report `created` for a line the real pass drops.
    with transaction.atomic():
        probe_run, _action = insert_or_create_campaign_run({'source_identifier': key}, fields)
        reconcile_result = reconcile_run(probe_run, dry_run=True)
        transaction.set_rollback(True)
    night_created += reconcile_result.created
    ...
```

(If a transient row is judged too invasive for a preview, the alternative is to narrow the
claim: drop the notebook cell's final `print` to the invariant it actually demonstrates,
and state in the runbook that a *brand-new* line's preview cannot predict a reconcile
failure.) Either way, extend `TestMalformedTimezoneSkipsOneLine` with the no-existing-run
variant — its own `setUpTestData` already builds the fixture, and no test currently runs
`--dry-run` over it.

### WR-04: the runbook's `duplicate_identity` reason vocabulary and troubleshooting "Cause" paragraph still describe only the group-vs-group collision — the new claimant-marker cause is missing from both

**File:** `docs/runbooks/telescope_runs_calendar.rst:933-937` (the reason vocabulary the
rest of the runbook calls "the full reason vocabulary") and `:1498-1500` (the
troubleshooting Cause list); contrast with the corrected remedy paragraphs at `:946-968`
and `:1511-1521`
**Severity:** WARNING

**Issue:** 35-15 corrected the three passages `35-VERIFICATION.md` gap 2 named (L938-949,
L1497, L1503) — I verified each: `grep` finds zero occurrences of `left untouched either
way` or `rewrites no existing`, no hedge words, and both remedy paragraphs now branch on
the marker. But `duplicate_identity` acquired a **second, structurally different cause**
this round, and the two places that *define* the reason were not touched:

```
**``duplicate_identity``** -- a second GROUP (a second, distinct
``Source line:`` string) whose derived run identity key is the same as an
earlier group's, because the key ignores the schedule line's status word
```

```
or the reason is ``duplicate_identity`` -- this event's own group's
``Source line:`` resolves to the same run identity key as an earlier
group's, because the key ignores the line's status word
```

Both are false for the CR-01 branch, where there is only **one** group and the claimant is
a pre-existing `CampaignRun` row. PROBE-P3's stderr for that case reads *"CampaignRun
pk=1 already claimed 'CLASSICAL:NTT:EFOSC2:…' with no recoverable 'Source line:' marker…"*
— an operator who takes the runbook's cause definition at face value goes looking for a
second schedule line that does not exist. The `_REASON_LABELS` entry in the command itself
(`cutover_classical_allocations.py:180`) has the same single-cause wording.

**Fix:** state both causes wherever the reason is defined:

```
**``duplicate_identity``** -- either a second GROUP (a second, distinct
``Source line:`` string) whose derived run identity key is the same as an
earlier group's, because the key ignores the schedule line's status word;
or a ``CampaignRun`` row that already holds the derived identity key whose
own stored ``observation_details`` ``Source line:`` is absent or differs
from the group's line, so this command cannot prove the run came from it.
```

Mirror the same two-cause sentence into the `:1498-1500` Cause list and into
`_REASON_LABELS[_DUPLICATE_IDENTITY]`, whose current text ("a second Source line resolves
to the same run identity key as an earlier group") is printed verbatim above every
per-event stderr line, including the claimant-marker ones.

---

## Info

### IN-01: no executed notebook cell covers the branch CR-01 actually inverted

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (cells 9-11)
**Issue:** 35-15's own key-decision records that the notebook's `duplicate_identity` demo
uses two groups with genuinely *differing* markers, which exercises the branch CR-01 left
unchanged; the pre-existing `pk=334` row is a `no_source_line` case on a different code
path. So the phase's one BLOCKER fix ships with corrected `CommandError` text in the
committed output but **no executed demonstration of the behaviour change itself** — the
committed notebook would look identical if the predicate were reverted.
**Fix:** add one cell that seeds a claimant with a non-marker `observation_details` and runs
the cutover, showing the refusal and the byte-identical run — the same fixture
`test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` already builds.

### IN-02: the cutover still hardcodes `'ALLOC:'` instead of the constant it could import (carried forward, still open)

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:671`
**Issue:** Unchanged from iteration 5's IN-02. `CalendarEvent.objects.filter(url__startswith='ALLOC:')`
is a bare literal in the final summary line, while `allocation_projector` defines
`ALLOC_URL_NAMESPACE = 'ALLOC:'` and this module already imports four names from it. The
reconciler notebook (cell 4) imports the constant properly, so the two disagree on style
for the same string.
**Fix:** import `ALLOC_URL_NAMESPACE` alongside the existing four and use it.

### IN-03: an identity key is still claimed before the group's transaction commits (carried forward, still open)

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:520` relative to
`:540-664`
**Issue:** Unchanged from iteration 5's IN-03. `seen_keys[key] = source_line` runs after the
convertibility checks (IN-02's fix) but still *before* `try: with transaction.atomic():`. A
group whose transaction rolls back wholesale (the `except` at `:651`, which marks every
event `_OTHER` and writes nothing) keeps the key claimed, so a sibling group sharing that
key is reported under `duplicate_identity` naming a line that converted nothing.
**Fix:** move the assignment to just after the `events_rekeyed += group_rekeyed` fold at
`:647-650`, i.e. only once the group's writes have committed.

### IN-04: the loader notebook's new parity cell mutates the shared dev database without a `finally`

**File:** `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, code cell 9
(`cells[18]`), added by 35-14
**Issue:** The cell writes `ntt.timezone = 'America/Santigo'` to the real developer database,
runs two `call_command()` invocations, and restores the original value with a plain
statement afterwards. Neither call is expected to raise today (the loader catches
`ZoneInfoNotFoundError` per line), but the notebook is re-executed by hand during
regeneration and any failure between the two writes — an interrupted kernel, an edit to the
schedule line, a future loader change that lets the error escape — leaves obscode `809`
with a typo'd timezone in the shared dev database, which every later cell and every other
notebook then resolves against. 35-11's own skip-path cell has the same shape.
**Fix:** wrap the mutation in `try: … finally: ntt.timezone = original; ntt.save(...)`, or
run the whole cell inside a deliberately-rolled-back `transaction.atomic()` the way
`project_observation_calendar_demo.ipynb`'s attribution cell already does.

---

_Reviewed: 2026-09-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (re-review, iteration 6)_
