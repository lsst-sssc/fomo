---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-14T00:00:00Z
depth: deep
iteration: 4
prior_review: 35-REVIEW.md (git show adf524f)
files_reviewed: 11
files_reviewed_list:
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/observation_projector.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - solsys_code/tests/test_observation_projector_signals.py
prior_findings:
  total: 12
  closed: 10
  partially_closed: 2
  still_open: 0
findings:
  critical: 1
  warning: 6
  info: 2
  total: 9
status: issues_found
---

# Phase 35: Code Review Report (iteration 4 — re-review of the nine `adf524f..HEAD` fix commits)

**Reviewed:** 2026-09-14
**Depth:** deep
**Files Reviewed:** 11
**Status:** issues_found

## Summary

This re-reviews the nine fix commits `b4b5b9e`, `8d22999`, `aabaea7`, `86be342`, `30112a0`,
`d57b461`, `c689eef`, `1f7b514`, `80cefed`, which `35-REVIEW-FIX.md` claims close all 12
iteration-3 findings with none skipped.

**Counting convention:** the `findings:` block counts only NEW findings (`NF-19`..`NF-25`,
`IN-01`, `IN-02`). No iteration-3 finding is carried forward — ten are genuinely closed and
two (NF-14, NF-10) are *partially* closed, with the surviving half re-filed under a new id
so it never shares a number with a claim that is now partly true.

The fix quality is materially better than iteration 3's: NF-05, NF-16, NF-17, NF-18, NF-11,
NF-12 and NF-13 are cleanly closed and I verified the behavioural ones by execution, not by
reading the diff. But the loop's dominant failure mode recurred **three more times**: NF-15's
fix opened a mirror-image double count (NF-22), NF-08's narrowing traded a misleading message
for a whole-command abort (NF-21), and NF-10's dry-run guard closed the create path while
leaving the *re-mint* path — the more likely operator scenario — with exactly the defect
NF-10 described (NF-20). The headline problem is NF-14: its guard is **invocation-scoped**
while the harm it prevents is **database-scoped**, so the remedy the command's own
`CommandError` prints ("then re-run this command — it is safe to repeat") walks straight
into the silent merge NF-14 exists to stop (NF-19, BLOCKER).

### Prior-finding verification: 10 closed, 2 partially closed, 0 still open

| Prior finding | Verdict |
|---|---|
| **NF-14** cutover dry run exits 0 on a fixture the real run rejects; no identity-key guard; silent merge; wrong reason (BLOCKER) | **Partially closed.** Within a single invocation the `seen_keys` guard works exactly as specified: `cutover_classical_allocations.py:387-395` rejects the second group under `_DUPLICATE_IDENTITY` before either pass attempts a write, `claimed_by_key[key]` replaces the per-group set, and my probe confirms dry-run and real-run summaries are byte-identical (`groups: 2, runs created: 1, events re-keyed: 3, unexplained: 3 (duplicate_identity=3)`, both non-zero exit). The run's fields stay group A's. **But the guard lives only in an in-memory dict for the current process.** Probe: a second invocation — the action the `CommandError` message itself instructs — silently merged group B into group A's run (`run_status` planned→cancelled, `observation_details` replaced) and reported `key_collision=3`, the precise wrong reason NF-14 named. A pre-existing `CampaignRun` row holding the key (the normal state after `load_telescope_runs` imported the counterpart line) is unguarded on the FIRST invocation too, and exits **zero**. Re-filed as **NF-19 (BLOCKER)**; the un-actionable remedy text is **NF-25**. |
| **NF-15** foreign-attributed stale `RUN:{pk}:{date}` neither counted nor logged (WARNING) | **Closed.** `_stale_dated_events()` (`campaign_reconciler.py:614-631`) now computes `foreign = stale_dated.count() - writable_dated.count()`, logs it, returns it as a third element, and `reconcile_run()` folds it into `blocked` on both the real and dry-run branches. The shape is no longer silent. **But the fold double-counts one sibling shape** (**NF-22**) and the helper's own type annotation was left describing a 3-tuple while it now returns 4 (**NF-23**). |
| **NF-16** human-confirmed legacy-retire decline reported as "blocked — owned by someone else" (WARNING) | **Closed.** Verified by execution: `reconcile_run()` over a retired night whose legacy `RUN:{pk}:{date}` event is `confirmed_by`-stamped to *this* run now returns `blocked=0, detach_declined=1, retired=1`, the legacy row survives, and a second sweep reports the identical tuple (idempotent). `reconcile_run()`'s `result._replace(detach_declined=result.detach_declined + detach_declined, ...)` (`campaign_reconciler.py:900`) correctly **adds** rather than overwrites, so the projector's contribution is not dropped — the latent bug the review named is avoided. `_reconcile_container()` always returns `detach_declined=0`, so the addition is a no-op on that branch. |
| **NF-17** `claimed_legacy_urls` docstring claimed a no-op exclusion (WARNING) | **Closed.** Both copies corrected and now agree: `campaign_reconciler.py:749-756` and `allocation_projector.py:546-560` both state "no-op for a re-keyed or deleted url, load-bearing for a blocked or declined one". |
| **NF-18** `_check_event_night()` docstring denied the mutation (WARNING) | **Closed.** `cutover_classical_allocations.py:245-247` now reads "Read here; the CALLER adds the night after its own write (or preview) succeeds…", the review's own suggested text. |
| **NF-05** group savepoint rolled back writes but not counters; re-marked already-marked events (WARNING) | **Closed.** Verified by execution: with a group-level exception raised after the run write (patched `ZoneInfo`), the summary reports `runs created: 0, events re-keyed: 0, unexplained: 3 (other=3)` against **0** `CampaignRun` rows and 0 `ALLOC:` events in the database, and each of the three events is reported exactly once. Both halves of the finding are gone — the `group_created`/`group_updated`/`group_unchanged`/`group_rekeyed` locals fold in only after the `with` block exits (`:583-586`), and `already_marked` (`:594`) suppresses the duplicate marking. |
| **NF-10** dry run hides `night_bounds()`'s inversion (WARNING) | **Partially closed.** `_raise_if_inverted()` (`allocation_projector.py:286-318`) is a genuine shared guard, and the dry-run **create** path now calls it over the same two datetimes (`:712-717`). The new `test_dry_run_of_a_brand_new_inverted_window_also_raises` passes. **But `_mint_fields()` has a second caller-path the fix did not touch:** the `_span_needs_remint()` re-mint branch (`:679-691`) still does `if dry_run: continue` before `_mint_fields()`. Probe: on an *existing* night whose operator edited `night_start_utc`/`night_end_utc` into an inverted pair, `reconcile_run(dry_run=True)` returns `created=1, retired=1` with no error while the immediately following real sweep raises `ValueError: Computed an inverted allocation-night span…` — the identical divergence, in the branch an operator edit is *most* likely to reach. Re-filed as **NF-20**. |
| **NF-11** vacuous `1 + 1 == 2` assertion (WARNING) | **Closed.** `test_cutover_classical_allocations.py:527-531` — the three lines are gone, replaced by a comment. (The comment hard-codes line numbers; **IN-01**.) |
| **NF-12** local `writable_events` shadowed the ownership helper (WARNING) | **Closed.** Renamed to `unattributed_events` at all four sites (`:442`, `:448`, `:459`, `:519`, `:541`). |
| **NF-13** broken sentence in the WR-11 runbook paragraph (WARNING) | **Closed.** `docs/runbooks/telescope_runs_calendar.rst:913-915` now reads "…whose events agree on their campaign, whose events are not already attributed to a different run, and whose derived observing nights are not already claimed." |
| **NF-04** observation event unattributed on the save that creates it (WARNING) | **Closed in code.** `receiver_on_record_save()` (`observation_projector.py:626-636`) now runs `project_record()` first, guarded by `if instance.facility in PROJECTED_FACILITIES`, with `action = stage = None` seeded so the D-11 loop still runs for a non-projected facility (WR-01's point survives) and the debug line is suppressed when no projection happened. `write_event_meta()` never touches `CalendarEventMeta.run`, so the reorder cannot fight the attribution bridge. The new regression test pins the exact scenario. **Paired notebook not updated — NF-24.** |
| **NF-08** loader's widened `except` tuple wrapped the whole reconcile call (WARNING) | **Closed as specified, with a regression.** The `KeyError` catch is now a dedicated three-line `try` around `_CLASSICAL_RUN_STATUS[parsed.status]` (`load_telescope_runs.py:258-271`), and wrapping `write_and_reconcile_campaign_run()` in `transaction.atomic()` (`:317-322`) genuinely fixes the committed-row-counted-as-skipped half — verified: after an abort, `CampaignRun.objects.count() == 0`. **But removing `KeyError` from the outer tuple means `ZoneInfoNotFoundError` (a `KeyError` subclass) now escapes the per-line handler entirely** and aborts the whole command — **NF-21**. **Paired notebook/runbook not updated — NF-24.** |

### Verification method

`python manage.py test solsys_code.tests.test_allocation_projector
solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations
solsys_code.tests.test_observation_projector_signals` — **181 tests, all passing.**

Findings NF-19, NF-20, NF-21, NF-22 and the NF-05/NF-16 closure verdicts are backed by
**executed probes** against a real Django test database (a probe module written under
`solsys_code/tests/`, run, then deleted; `git status --short` confirms **no source file was
modified by this review**). Reproduced facts, not inferences:

- `cutover_classical_allocations` run twice over two status-only-colliding groups: run 1
  reports `duplicate_identity=3` and leaves run pk=1 at `run_status='planned'` /
  `Status: allocation`; run 2 reports `runs updated: 1, key_collision=3` and leaves run pk=1
  at `run_status='cancelled'` / `Status: cancelled`, while its three `ALLOC:` events still
  carry group A's `'NTT EFOSC2'` titles (NF-19).
- With a pre-existing `CampaignRun` holding the key, the first invocation reports
  `runs created: 0, updated: 1, events re-keyed: 3, unexplained: 0` and **exits zero**,
  having silently flipped that run's status (NF-19).
- `reconcile_run(run, dry_run=True)` over a re-mint-needed night with an inverted operator
  window returns `created=1, retired=1` and raises nothing; `reconcile_run(run)` raises
  `ValueError: Computed an inverted allocation-night span for run pk=1 night=2026-07-09` (NF-20).
- `load_telescope_runs <file>` against an `Observatory` whose `timezone` is a typo
  (`'America/Santigo'`) exits with an **uncaught** `ZoneInfoNotFoundError`, empty stdout,
  empty stderr, line 2 of the file never processed (NF-21).
- `reconcile_run(run)` over ONE legacy `RUN:{pk}:{night}` event attributed to a different
  run returns `blocked=2` and emits two log lines for that single row (NF-22).
- The NF-05 group-rollback probe reports `runs created: 0` against 0 database rows with each
  event marked once; the NF-16 probe reports `blocked=0, detach_declined=1` and is idempotent.

### Positives worth recording

NF-05's fix is the cleanest in the batch — it fixed both halves of the finding (counters and
duplicate marking) with the minimal correct change, and the rollback probe shows summary and
database agreeing exactly. NF-16's `result._replace(detach_declined=result.detach_declined +
detach_declined, ...)` is exactly right and the fixer explicitly reasoned about the
overwrite-vs-add trap in a comment rather than discovering it later. NF-04's reorder is
correct and the regression test (make the record unprojectable, then supply instrument and
block in one save) is a genuinely adversarial fixture rather than a restatement of the fix.
The `TestDuplicateIdentityKeyAcrossGroups` dry/real parity test asserts summary equality
rather than hand-picked counters, which is the right shape — it just does not cover a second
invocation.

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### NF-19: the cutover's identity-key guard is invocation-scoped while the collision is database-scoped — the re-run the command itself prescribes silently merges the second group, and a DB-resident claimant is unguarded and exits zero

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:344`, `:387-395`,
`:483-487`; contract asserted at `:44-50` and `:80-86`, and at
`docs/runbooks/telescope_runs_calendar.rst:934-940`, `:1436-1446`
**Severity:** BLOCKER

**Issue:** `seen_keys` is a plain `dict` local to one `handle()` call. It records only which
identity keys **this process** has claimed. But the thing it is protecting —
`insert_or_create_campaign_run({'source_identifier': key}, fields)` at `:487` — matches
against the **database**, where a claimant can already exist from an earlier invocation of
this command or from `load_telescope_runs`. Two consequences, both reproduced:

**(1) The documented remedy triggers the harm.** The `CommandError` the command raises ends
with *"then re-run this command -- it is safe to repeat"*, and the runbook repeats it
(`:1481-1484`). On that re-run, group A's events are no longer blank-url, so group A is no
longer a candidate at all — `seen_keys` is empty, group B claims the key unopposed, and
`insert_or_create_campaign_run()` find-and-updates group A's run:

```
run 1: Done. candidates: 6, groups: 2, runs created: 1, events re-keyed: 3, unexplained: 3
       unexplained (duplicate_identity): 3
       run pk=1 -> run_status='planned',  observation_details starts 'Status: allocation'

run 2: Done. candidates: 3, groups: 1, runs created: 0, updated: 1, events re-keyed: 0, unexplained: 3
       unexplained (key_collision): 3
       pk=4: night 2026-07-09 url is already held by CalendarEvent pk=1
       run pk=1 -> run_status='cancelled', observation_details starts 'Status: cancelled'
       ALLOC: event titles still ['NTT EFOSC2', 'NTT EFOSC2', 'NTT EFOSC2']
```

That is verbatim the harm NF-14 was filed for — a run whose status contradicts its own
calendar entries, reported under `key_collision`, whose documented action ("find the
duplicate row … and delete it or re-attribute it") is wrong. The only thing the fix changed
is that the operator now has to press enter twice.

**(2) A pre-existing run is unguarded on the FIRST pass, and the command exits zero.** When
`load_telescope_runs` has already imported the `allocation` line (creating run R), and the
stranded blank-url events belong to the `cancelled` line, `seen_keys` is empty and nothing
stops the merge:

```
Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0,
      events re-keyed: 3, unexplained: 0
pre-existing run status now: cancelled   (was planned)
pre-existing run details now: Status: cancelled
```

Exit 0, no stderr, no reason, an APPROVED run's real-world lifecycle state silently flipped,
and three events re-keyed onto it. This is worse than case (1) because nothing tells the
operator it happened.

The module docstring now states the opposite as settled fact — *"reported (never silently
merged into the earlier group's run)"* (`:46-47`) — as does the runbook: *"the SECOND group
is never merged into the first group's run … the first group's run and events are converted
and left untouched either way"* (`:940-947`). Both are false on the second invocation.

**Fix:** make the guard read the database, not just the in-process dict — the key's claimant
is recoverable from the run's own stored `Source line:`, which is exactly what this command
already re-parses:

```python
if key in seen_keys:
    _mark_unexplained(events, _DUPLICATE_IDENTITY, ...)
    continue

# NF-19: a claimant from an EARLIER invocation (or from load_telescope_runs) is just as
# much a collision as one in this run's own seen_keys -- the harm is a find-or-update
# against the database, so the guard has to look there too.
existing_run = CampaignRun.objects.filter(source_identifier=key).first()
if existing_run is not None and _extract_source_line(existing_run.observation_details) not in (
    None,
    source_line,
):
    _mark_unexplained(
        events,
        _DUPLICATE_IDENTITY,
        f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: CampaignRun pk={existing_run.pk} already holds '
        f'{key!r} for a different Source line; disambiguate the two lines before converting',
    )
    continue
seen_keys[key] = source_line
```

(`_extract_source_line()` already parses the same `Source line: ` marker that
`observation_details` carries, so this needs no new parsing.) Add a test that runs the
command **twice** over the two-group fixture and asserts the run's `run_status` and
`observation_details` are unchanged by the second pass and that the reported reason stays
`duplicate_identity`, plus a test for the pre-existing-run case asserting a non-zero exit.
Correct the two docstring/runbook sentences that now assert the un-held guarantee, and the
runbook's "the first group's run and events are converted and left untouched either way".

---

## Warnings

### NF-20: NF-10's dry-run inversion guard covers the create path only — the `_span_needs_remint()` re-mint path, the one an operator window edit actually reaches, still hides the identical `ValueError`

**File:** `solsys_code/allocation_projector.py:679-691` (contrast with the fixed `:712-717`)
**Severity:** WARNING

**Issue:** `_mint_fields()` — the only caller of `night_bounds()`, where `_raise_if_inverted()`
lives — is reached from **two** branches of the per-night loop, not one. The fix added the
guard to the `existing is None` create branch and left the re-mint branch untouched:

```python
if existing is not None and _span_needs_remint(run, night, existing):
    totals['retired'] += 1
    totals['created'] += 1
    if dry_run:
        continue                      # <- still short-circuits before any boundary check
    existing.delete()
    event, _action = insert_or_create_calendar_event({'url': url}, fields=_mint_fields(run, night))
```

This is the *more* reachable half. `_span_needs_remint()` returns True precisely when an
operator has changed `night_start_utc`/`night_end_utc` on a run whose nights are already
minted — which is exactly the edit that can invert them. Reproduced (probe, executed; La
Silla, one night, valid `23:00`/`05:00` minted first, then edited to `09:00`/`23:00`):

```
PROBE-P2 initial real reconcile: ReconcileResult(created=1, ...)
PROBE-P2 minted span: 2026-07-09 23:00:00+00:00 -> 2026-07-10 05:00:00+00:00
PROBE-P2 dry run after inverting the window:
        ReconcileResult(created=1, retired=1, ..., skipped_reason=None)     <- no error
PROBE-P2 real run RAISED ValueError: Computed an inverted allocation-night span for run
        pk=1 night=2026-07-09: start=2026-07-10T09:00:00+00:00 >= end=2026-07-09T23:00:00+00:00.
```

`reconcile_campaign_runs --dry-run` reports `would_retire: 1, would_create: 1`; the real
sweep reports the run under `failed`. The preview still cannot show the operator the one
condition that stops the run from projecting — NF-10's own sentence, unchanged, for this
branch.

**Fix:** hoist the same check the create branch now performs into a tiny helper and call it
from both, so a third caller of `_mint_fields()` cannot reopen this a third time:

```python
def _raise_if_set_window_inverted(run: CampaignRun, night) -> None:
    """The astropy-free half of night_bounds()'s guard, for the dry-run paths that skip
    _mint_fields() (NF-10/NF-20). A null field's boundary needs sun_event() and stays
    unchecked, matching _span_needs_remint()'s own null-field convention."""
    if run.night_start_utc is None or run.night_end_utc is None:
        return
    night_span = _night_span_utc(run, night)
    _raise_if_inverted(
        run,
        night,
        _time_of_day_to_datetime(run.night_start_utc, night, night_span),
        _time_of_day_to_datetime(run.night_end_utc, night, night_span),
    )
```

then `if dry_run: _raise_if_set_window_inverted(run, night); continue` in **both** branches.
Add the re-mint twin of `test_dry_run_of_a_brand_new_inverted_window_also_raises` — mint a
valid night, invert the run's window, and assert the dry run and the real run raise the
identical error.

### NF-21: NF-08's narrowing turns a mistyped `Observatory.timezone` into an unhandled abort of the whole loader — the outcome the module's own WR-09 comment and the runbook both name as the thing to avoid

**File:** `solsys_code/management/commands/load_telescope_runs.py:337`, contrast with the
module's own `:206-211` and `docs/runbooks/telescope_runs_calendar.rst:1381-1394`
**Severity:** WARNING

**Issue:** Removing `KeyError` from `except (ValueError, KeyError, Observatory.DoesNotExist)`
did narrow the misleading message, but it also removed the only handler for
`zoneinfo.ZoneInfoNotFoundError`, which **subclasses `KeyError`**:

```
>>> issubclass(ZoneInfoNotFoundError, KeyError)
True
```

`Observatory.timezone` is a plain unvalidated `CharField(max_length=64)`
(`solsys_code_observatory/models.py:66`) — no choices, no validator — and the runbook's own
"Observatory missing timezone" section tells operators to type an IANA name into it by hand.
A **blank** value raises `ValueError` and is still caught; a **typo** raises
`ZoneInfoNotFoundError` from `project_allocation()`'s `ZoneInfo(run.site.timezone)` and now
escapes the per-line handler entirely. Reproduced (probe, executed; two-line schedule file,
`timezone='America/Santigo'`):

```
PROBE-P8 UNCAUGHT ZoneInfoNotFoundError : 'No time zone found with key America/Santigo'
PROBE-P8 stdout so far: ''
PROBE-P8 stderr so far: ''
PROBE-P8 CampaignRun rows: 0
```

Line 2 is never processed, no summary line is printed, and the operator gets a bare
traceback. That violates the invariant the runbook states for this exact command —
*"**one bad row never aborts the whole run.** A problem with a single line or record is
logged and skipped, and the command continues to the end, reporting a summary count"*
(`:1382-1385`) — and it is word-for-word the outcome `load_telescope_runs.py:206-211`'s own
WR-09 comment says the import-time assert exists to prevent: *"an uncaught KeyError that
aborts the whole import/cutover mid-run, after partial commits, with no reason report."*
NF-08 asked for a narrower catch, not for no catch; `cutover_classical_allocations` keeps its
per-group/per-event `except Exception` catch-all for the same class of failure, so the two
sibling commands now disagree about whether one bad site aborts the batch.

**Fix:** keep the narrow `KeyError` catch where NF-08 put it, and give the write/reconcile
call its own handler that reports the real stage instead of swallowing it into the parse
bucket:

```python
                    with transaction.atomic():
                        result = write_and_reconcile_campaign_run({'source_identifier': key}, fields)
                except ZoneInfoNotFoundError as exc:
                    self.stderr.write(
                        f'Line {line_num}: Observatory {site.short_name!r} (obscode={site.obscode}) has an '
                        f'invalid IANA timezone {site.timezone!r}: {exc} (line text: {line.strip()!r})'
                    )
                    run_skipped += 1
                    continue
```

(placed as its own `except` clause on the per-line `try`, ahead of the
`(ValueError, Observatory.DoesNotExist)` clause, since `ZoneInfoNotFoundError` is not a
`ValueError`). Add a regression test asserting the command completes, processes the following
line, and reports `skipped: 1` for a site with a malformed timezone.

### NF-22: NF-15's `foreign` fold double-counts a legacy `RUN:{pk}:{date}` event blocked on the takeover path — `blocked=2` for one row, with two log lines

**File:** `solsys_code/allocation_projector.py:655-664` (the missing claim),
`solsys_code/campaign_reconciler.py:622-631`, `:897-902`
**Severity:** WARNING

**Issue:** `project_allocation()` claims a legacy url into `legacy_urls_claimed` in two
places, and the two are **asymmetric**. The retired branch claims it *before* the ownership
check (`:616`, NF-09's fix). The takeover branch claims it *after* (`:664`), so the
`_may_write()`-False path `continue`s at `:663` without ever claiming:

```python
if existing is None and legacy_event is not None:
    if not _may_write(legacy_event, run):
        logger.warning('Allocation blocked: legacy event pk=%s is not owned by run pk=%s.', ...)
        totals['blocked'] += 1
        continue                      # <- returns BEFORE the claim on the next line
    legacy_urls_claimed.add(legacy_url)
```

That was harmless while `_stale_dated_events()` silently dropped the foreign shape. NF-15
made it stop dropping it — so the same row is now counted a second time, as `foreign`, and
folded into `blocked` again at `campaign_reconciler.py:898`. Reproduced (probe, executed; one
`RUN:{pk}:2026-07-09` event whose companion row points at another run, one-night window):

```
PROBE-P1 result: ReconcileResult(created=0, updated=0, unchanged=0, blocked=2, ...)
PROBE-P1 legacy still exists: True
log: Allocation blocked: legacy event pk=1 is not owned by run pk=1.
log: Reconcile found 1 leftover per-night event(s) for run pk=1 attributed to a different run.
```

`reconcile_campaign_runs` renders this as `Run pk=N: 2 event(s) blocked -- owned by someone
else` for a single row — the same class of defect NF-09 was filed for, re-opened by the fix
for its sibling.

**Fix:** make the takeover branch claim the url on every decision, exactly as the retired
branch already does:

```python
if existing is None and legacy_event is not None:
    # NF-22: claim the url the moment this loop decides the legacy event's fate AT ALL --
    # including a block -- so the downstream foreign count (NF-15) cannot report the same
    # single decision a second time under the same `blocked` total.
    legacy_urls_claimed.add(legacy_url)
    if not _may_write(legacy_event, run):
        ...
```

Add a regression test asserting `blocked == 1` (not 2) and exactly one log line for a legacy
`RUN:{pk}:{date}` event attributed elsewhere on the takeover path — no existing test covers
this shape, which is why 181 green tests did not catch it.

### NF-23: `_detach_stale_family_events()`'s return annotation still says a 3-tuple while the function returns four values

**File:** `solsys_code/campaign_reconciler.py:678`
**Severity:** WARNING

**Issue:** The NF-15 fix added `foreign_blocked` to both the return statement (`:817`) and
the `Returns:` docstring (`:759`, correctly `tuple[int, int, int, int]`), but left the
signature annotation behind:

```python
def _detach_stale_family_events(
    run: CampaignRun, active_urls: set[str], claimed_legacy_urls: frozenset[str] = frozenset()
) -> tuple[int, int, int]:          # <- returns 4
```

The module's docstring and its annotation now state different contracts for the same
function, and the only caller (`reconcile_run():883`) unpacks four. Neither ruff nor the test
suite checks annotations, so nothing will catch this until a reader or a type checker trusts
the wrong one — and this is exactly the pair of half-updated descriptions NF-17 was filed for
one iteration ago, in the same module.

**Fix:** `) -> tuple[int, int, int, int]:`

### NF-24: the paired notebooks for the two out-of-scope modules the fixer changed were not updated, contrary to CLAUDE.md's paired-docs rule

**File:** `solsys_code/management/commands/load_telescope_runs.py` (changed by `d57b461`) with
no `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` change;
`solsys_code/observation_projector.py` (changed by `30112a0`) with no
`docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` change
**Severity:** WARNING

**Issue:** CLAUDE.md's paired-docs rule maps `load_telescope_runs.py` →
`load_telescope_runs_demo.ipynb` and `observation_projector.py` →
`project_observation_calendar_demo.ipynb`, and scopes the trigger to a change in a module's
*behavior* — "not pure refactors or typo fixes". Both of these are behavioral, not cosmetic:

- **NF-08** introduced a new per-line stderr message (`Line N: unknown classical status …`),
  a new transaction boundary, and — as NF-21 documents — changed which exception classes are
  reported per-line versus aborting the batch. The demo notebook's cells 10 and 12 print
  `stderr (skipped lines)` verbatim, i.e. the exact surface this change alters, and the
  runbook's "Per-line / per-record skip-and-log behaviour" bullet for this command
  (`docs/runbooks/telescope_runs_calendar.rst:1386-1394`) is now inaccurate for the
  malformed-timezone case.
- **NF-04** changed the order of two signal-triggered writes so that campaign attribution now
  lands on the creating save rather than a later one — the precise behaviour the D-11 trigger
  demo exists to show.

`35-REVIEW-FIX.md`'s own verification section asserts the opposite ("No other fix in this
pass changed … `observation_projector.py`'s *documented, demonstrated* behavior in a way that
would make an existing runbook sentence or notebook cell's output inaccurate"). The runbook
sentence above is one counterexample. Note this is the fourth logged instance of this rule
being missed (CLAUDE.md's own breach history lists `260619-f7u`, `260620-v9x`, `260726-kdp`).

**Fix:** add a cell to `load_telescope_runs_demo.ipynb` exercising the new
unknown-status/invalid-timezone skip path and its stderr text, and a cell to
`project_observation_calendar_demo.ipynb` (or the D-11 trigger section of whichever notebook
covers it) asserting the attribution lands on the creating save; regenerate both with
`jupyter nbconvert --to notebook --execute --inplace`. Update the runbook's
`load_telescope_runs` skip-and-log bullet once NF-21 is resolved.

### NF-25: the `duplicate_identity` remedy the command prints and the runbook repeats cannot be carried out for this command — it has no schedule file, and the notebook's own committed output shows the advice already satisfied

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:391-392`,
`docs/runbooks/telescope_runs_calendar.rst:940-947`, `:1443-1446`, `:1475-1479`
**Severity:** WARNING

**Issue:** The reason string tells the operator to *"add a bracketed proposal token to one of
the two lines"*, and the runbook says the same three times. That remedy is correct for
`load_telescope_runs`, which reads a file the operator can edit. `cutover_classical_allocations`
**needs no schedule file** — it is the command's stated selling point (`:18-22`): every fact
comes from re-parsing each stranded event's own `description`. Editing the schedule file
changes nothing the cutover will ever read; the operator must edit the `Source line:` text of
every affected `CalendarEvent` row in the admin. The runbook's bare "then re-run the command"
compounds this, because NF-19 shows the un-resolved re-run is the destructive path.

The paired notebook's own committed output demonstrates the advice failing on itself — both
demo lines already carry a bracketed token, and the command still prints the instruction to
add one:

```
pk=354 ('NTT EFOSC2'): a second Source line resolves to the same run identity key as an earlier
group: 'NTT EFOSC2 allocation 9-12 July [NF-14-demo]' already claimed
'CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-11:BoN:EoN:NF-14-demo';
add a bracketed proposal token to one of the two lines
```

**Fix:** name the action this command can actually act on:

```python
f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: {seen_keys[key]!r} already claimed {key!r}; '
"edit this group's events' description 'Source line:' in the admin to carry a DIFFERENT "
'bracketed [proposal] token from the earlier group, then re-run'
```

and correct the three runbook passages to say the same — the schedule-file remedy belongs
only to the `load_telescope_runs` half of that troubleshooting section.

---

## Info

### IN-01: the comment that replaced NF-11's vacuous assertion hard-codes line numbers that will rot

**File:** `solsys_code/tests/test_cutover_classical_allocations.py:527-531`
**Issue:** *"one re-keyed (asserted above at line 507, `rekeyed_event.url`), one deleted
(asserted above at line 512, `delete_legacy_pk`)"* — a comment whose accuracy depends on two
absolute line numbers in its own file, which the next edit above it invalidates silently.
**Fix:** cite the assertion by the identifier alone (`rekeyed_event.url` / `delete_legacy_pk`),
which is already unambiguous within the test.

### IN-02: the cutover's identity-key guard runs before the checks that decide whether the claiming group is convertible at all

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:387-460`
**Issue:** `seen_keys[key] = source_line` is set at `:395`, ahead of the `_CAMPAIGN_MISMATCH`
check (`:397-400`), the `_CLASSICAL_RUN_STATUS` lookup (`:410-414`) and the all-events-foreign
`continue` (`:459-460`). A group that is itself unconvertible therefore permanently claims the
key and causes a convertible sibling group to be reported under `duplicate_identity`, naming a
line that converted nothing. Harmless for correctness (the two lines do collide either way),
but the operator message points at the wrong line to fix first.
**Fix:** move `seen_keys[key] = source_line` down to just before the group's `try:
with transaction.atomic():` block, i.e. after the group is known to have something to write.

---

_Reviewed: 2026-09-14_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (re-review, iteration 4)_
