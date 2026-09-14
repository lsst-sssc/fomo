---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-13T00:00:00Z
depth: deep
iteration: 3
prior_review: 35-REVIEW.md (git show 5b2bd38)
files_reviewed: 9
files_reviewed_list:
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/models.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_cutover_classical_allocations.py
prior_findings:
  total: 13
  closed: 5
  partially_closed: 1
  still_open: 7
findings:
  critical: 1
  warning: 11
  info: 0
  total: 12
status: issues_found
---

# Phase 35: Code Review Report (iteration 3 — re-review after the ti1/ti3/rmd fixes)

**Reviewed:** 2026-09-13
**Depth:** deep
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This is a re-review of the 11 commits from quick tasks `260913-ti1`, `260913-ti3` and
`260913-rmd`, which claimed to close iteration 2's NF-01, NF-02, NF-03, NF-06, NF-07 and
NF-09.

**Counting convention (differs slightly from iteration 2, stated here so downstream
consumers are not misled):** the `findings:` block counts **every finding this review
reports as open** — the 7 iteration-2 findings that are still open (which keep their
original `NF-xx` ids) plus the 5 new findings introduced or newly reached by these fixes
(`NF-14`..`NF-18`). Iteration 2 counted only its new findings; folding the carried-forward
ones in here keeps the frontmatter total honest, since the still-open priors were not
re-numbered.

### Prior-finding verification: 5 closed, 1 partially closed, 7 still open

| Prior finding | Verdict |
|---|---|
| NF-01 stale `ALLOC:`/`RUN:` events writable-but-unattributed (BLOCKER) | **Closed.** `_clearable_declined_and_unattributed()` (`campaign_reconciler.py:457-510`) is a genuine total partition over shapes (a)/(b)/(c-this-run), and is wired into all four call sites the finding named (`project_allocation()`'s convergence, its CR-03 retire branch, `_stale_dated_events()`, `_stale_allocation_events()`). The reverse-`OneToOneField` join means the two halves cannot double-count, and `.exclude(telescope_label_meta__confirmed_by__isnull=False)` correctly retains the no-companion-row rows (verified by the four new shape-(a)/(b) tests, all passing). One residual class the finding did **not** name — a *foreign*-attributed stale `RUN:{pk}:{date}` event — is still the forbidden third outcome: new finding NF-15. |
| NF-02 cutover `--dry-run` exits zero on a run the real pass rejects (BLOCKER) | **Partially closed.** The per-event half is genuinely fixed: `_check_event_night()` (`cutover_classical_allocations.py:182-240`) is the single home of all three preconditions, both branches call it, the window now comes from `fields[...]` on the dry-run side, and `_WINDOW_MISMATCH` got its own reason name in the module docstring, `_REASON_LABELS` and the runbook. But the *group*-level divergence is untouched, and a dry run still exits 0 over a fixture the immediately following real run rejects — reproduced below, new BLOCKER **NF-14**. Worse, the fix wrote the now-false parity claim into the module docstring (`:68-74`) and the runbook (`:936-942`) as settled fact. |
| NF-03 site-direction rule does not generalize (BLOCKER) | **Closed.** `_site_runs_behind_utc()`'s sign-of-offset boolean is gone; `_night_span_utc()` + the nearest-candidate `_time_of_day_to_datetime()` (`allocation_projector.py:169-239`) resolve each boundary against the site's own local-18:00→+12h span, with no hour threshold anywhere. Hand-verified correct for UTC-10, -4, +2, +5:30, +6, +10 and +14, including the half-hour case, and DST-correct (the 12 h is added to the zone-carrying local datetime before `astimezone()`). `night_bounds()` and `_span_needs_remint()` share one span, so the create and re-mint paths agree. The false two-case taxonomy in `models.py:266-277` was replaced with the correct three-band one. |
| NF-04 observation event unattributed on the save that creates it (WARNING) | **Still open — not addressed.** `solsys_code/observation_projector.py` is unchanged since `5b2bd38` (`git diff --stat` confirms it is not in the change set). |
| NF-05 group savepoint rolls back writes but not counters (WARNING) | **Still open — not addressed**, and now *amplified* by the NF-02 fix. Reproduced below. |
| NF-06 `_may_write()` vs `writable_allocation_events()` disagree (WARNING) | **Closed.** `_may_write()` (`campaign_reconciler.py:248-286`) now also accepts `ALLOC:{run.pk}:` as a self-owned namespace, with a correct trailing colon on the prefix. Traced every caller: the widening reaches only `project_allocation()`'s per-night lookup (the intended fix); `_reconcile_container()` and the legacy-event checks pass `RUN:`-keyed urls only, so nothing else changes behaviour. The new `test_may_write_agrees_with_both_queryset_twins_for_every_shape` pins it. |
| NF-07 runbook counter definitions stale (WARNING) | **Closed.** Both `retired` (`:1011-1025`) and `legacy_deleted` (`:1033-1050`) were rewritten to the multi-cause form, the "keeps a single whole-window entry" clause is now explicitly gated on the run actually being container-dispatched, and the paired notebook cell (`reconcile_campaign_runs_demo.ipynb`) carries the matching correction — the CLAUDE.md paired-docs rule is satisfied for this change. |
| NF-08 `load_telescope_runs` widened `except` tuple (WARNING) | **Still open — not addressed.** `solsys_code/management/commands/load_telescope_runs.py` is unchanged since `5b2bd38`. |
| NF-09 declined legacy event counted twice (WARNING) | **Closed as specified.** `legacy_urls_claimed.add(legacy_url)` moved to line 585, before the `_may_write()` branch, so a blocked/declined legacy event is claimed and the downstream `_stale_dated_events()` no longer sees it (`test_retiring_a_night_never_deletes_a_human_confirmed_legacy_event` now asserts `detach_declined == 0`). The double-count is gone — but the surviving counter is the *wrong* one: new finding **NF-16**. The fix also invalidated a docstring it did not update: **NF-17**. |
| NF-10 dry run hides the `night_bounds()` inversion (WARNING) | **Still open — not addressed.** `allocation_projector.py:664-666` still short-circuits the dry-run create path before any boundary validation. Reproduced below. |
| NF-11 vacuous assertion in the cutover sequence test (WARNING) | **Still open — not addressed.** `test_cutover_classical_allocations.py:531-533` still reads `rekeyed_count = 1; legacy_deleted_count = 1; self.assertEqual(rekeyed_count + legacy_deleted_count, 2)`. |
| NF-12 local `writable_events` shadows the ownership-helper name (WARNING) | **Still open — not addressed.** `cutover_classical_allocations.py:383` still binds a plain list to `writable_events`. |
| NF-13 broken sentence in the WR-11 runbook edit (WARNING) | **Still open — not addressed.** `docs/runbooks/telescope_runs_calendar.rst:912-914` still reads "…whose events agree on their campaign, and whose events are not already attributed to a different run, are not already claimed on a colliding night." — the same comma splice with a dangling subject. |

### Verification method

`python manage.py test solsys_code.tests.test_allocation_projector
solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations`
— **147 tests, all passing.** `pre-commit run ruff --files <the 7 changed Python files>` —
clean.

Four of the findings below are backed by executed probe tests run against a real Django
test database (probe modules written under `solsys_code/tests/`, run, then deleted; `git
status --short` confirms **no source file was modified by this review**). Reproduced facts,
not inferences:

- `cutover_classical_allocations --dry-run` reports `runs created: 2, events re-keyed: 6,
  unexplained: 0` and **exits zero**; the immediately following real run reports `runs
  created: 1, updated: 1, events re-keyed: 3, unexplained: 3 (key_collision=3)` and exits
  non-zero — over the exact fixture the runbook itself names (NF-14).
- A stale `RUN:{pk}:{date}` event attributed to a different run survives every sweep with
  `blocked=0, detach_declined=0, legacy_deleted=0, retired=0` and **no log line at all**
  (NF-15).
- After a group-level rollback the cutover prints `runs created: 1` with zero cutover runs
  in the database, and reports one event **twice**, under two different reasons (NF-05).
- `reconcile_run(run, dry_run=True)` reports `created=1` for a run whose immediately
  following real sweep raises `ValueError: Computed an inverted allocation-night span …`
  (NF-10).

### Positives worth recording

The NF-03 rewrite is the strongest work in this batch: it replaced a two-case rule with a
derivation from the site's own night, removed the hour threshold entirely rather than
adding a third branch to it, and correctly does wall-clock-then-convert so a DST shift
inside the night is handled rather than assumed away. The NF-01 helper's disjointness
argument (reverse `OneToOneField` ⇒ the two halves cannot overlap) is correct and load-bearing,
and the `stray_confirmed` branch closes the narrower hole a naive fix would have re-opened.
The NF-02 helper's decision to pass the window in as parameters — rather than read it off
`run` — is exactly the right shape for the parity it is trying to guarantee, and is why the
per-event half of that finding is genuinely closed. The NF-07 fix correctly updated both the
runbook and the paired pre-executed notebook in the same commit.

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### NF-14: the cutover's dry run still exits zero on a fixture the real run rejects — NF-02's fix shared the per-event checks but not the group-level ones, and the module now asserts a parity it does not have

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:307-511` (the
group loop), contract newly asserted at `:68-74` and `docs/runbooks/telescope_runs_calendar.rst:936-942`
**Severity:** BLOCKER

**Issue:** `_check_event_night()` made the **per-event** preconditions identical on both
paths. The divergence that remains is one level up: two different `source_line` strings can
map to the **same** `_source_identifier`, because that key
(`load_telescope_runs.py:57-88`) is built from telescope, instrument, window and the two
sub-night tokens and deliberately ignores `parsed.status`. The cutover groups by the raw
`source_line` string, so those are two *groups* that resolve to one *run* — and the two
passes then disagree on everything:

- **Dry run:** `claimed_nights` is created fresh per group (`:435`), and for group 2
  `existing_run` is `None` (nothing was written), so `_check_event_night()`'s third
  precondition — the only cross-group check there is — is skipped at `:235`. Group 2's
  nights look free. Exit 0.
- **Real run:** group 1's events have already been re-keyed and committed inside the group
  savepoint, so group 2's url probe finds them. Exit non-zero, three `key_collision`s.

Reproduced (probe, executed; six blank-url events, three on
`'NTT EFOSC2 allocation 9-12 July'` and three on `'NTT EFOSC2 cancelled 9-12 July'` — the
exact status-only-difference collision the runbook's own example line names):

```
key A: CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-11:BoN:EoN
key B: CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-11:BoN:EoN

PROBE-4 DRY RUN EXITED ZERO
  Done (dry run). candidates: 6, groups: 2, runs created: 2, updated: 0, unchanged: 0,
  events re-keyed: 6, unexplained: 0

PROBE-4 real CommandError: 3 event(s) could not be explained and were left untouched
                           (key_collision=3).
  Done. candidates: 6, groups: 2, runs created: 1, updated: 1, unchanged: 0,
  events re-keyed: 3, unexplained: 3
  pk=4: night 2026-07-09 url is already held by CalendarEvent pk=1
```

This is word-for-word the condition the ti3 commit added to the module docstring as closed:

> `--dry-run` applies every per-event precondition the real pass applies … so the two passes
> agree on the re-key count, the unexplained count, every per-reason count and the exit
> status. That agreement is what makes the promise in the preceding sentence true rather
> than aspirational: **a dry run can no longer exit 0 over a fixture the immediately
> following real run rejects.**

It is also asserted in the runbook (`:936-942`) and is the stated premise of the new
`TestDryRunAndRealRunAgree` class — which passes only because its fixtures never contain
two groups sharing an identity key.

Two further defects on the same path, both strictly worse than the parity gap:

1. **The cutover has no identity-key collision guard at all**, unlike
   `load_telescope_runs`, which explicitly refuses the second line
   (`load_telescope_runs.py:248-255`, `seen_keys` / `skipped_collision`). The runbook states
   the opposite as fact for this command
   (`docs/runbooks/telescope_runs_calendar.rst:1404-1425`): *"``load_telescope_runs`` (and,
   on the legacy cutover path, ``cutover_classical_allocations``) matches a schedule line to
   its ``CampaignRun`` by a deterministic key … **The skipped line is never silently merged
   into the first**; it is reported and dropped until the collision is resolved."* The probe
   shows it **is** silently merged: the surviving run pk=1 ends with
   `run_status='cancelled'` and `observation_details` beginning `Status: cancelled`, while
   its three `ALLOC:` events were re-keyed from the *allocation* group and still carry that
   group's titles — a run whose status contradicts its own calendar entries, with nothing
   printed about it. Which group wins depends on `dict` insertion order, i.e. on the lowest
   `CalendarEvent.pk` in each group.
2. **The printed reason is wrong for this cause.** The operator is told
   `key_collision`, whose documented action is "find the duplicate row in the Django admin
   … and delete it or re-attribute it". Deleting those rows is exactly the wrong action —
   they are a legitimate second schedule line, and the correct action is the one the
   runbook already documents for `load_telescope_runs`: add a bracketed proposal token to
   disambiguate.

**Fix:** give the cutover the same guard its sibling command has, and make the dry run
carry cross-group state so the two passes see the same world:

```python
_DUPLICATE_IDENTITY = 'duplicate_identity'
_REASON_LABELS[_DUPLICATE_IDENTITY] = (
    'a second Source line resolves to the same run identity key as an earlier group'
)

seen_keys: dict[str, str] = {}          # source_identifier -> the source_line that claimed it
claimed_by_key: dict[str, set[date]] = defaultdict(set)   # cross-group, keyed by run identity

for source_line, events in groups.items():
    ...
    if key in seen_keys:
        _mark_unexplained(
            events, _DUPLICATE_IDENTITY,
            f'{_REASON_LABELS[_DUPLICATE_IDENTITY]}: {seen_keys[key]!r} already claimed '
            f'{key!r}; add a bracketed proposal token to one of the two lines',
        )
        continue
    seen_keys[key] = source_line
    ...
    claimed_nights = claimed_by_key[key]   # NOT a fresh set per group
```

Using `claimed_by_key[key]` (rather than a per-group set) makes the dry run detect the
collision even while `run is None`, which is the whole parity gap; the `seen_keys` guard
then makes the *real* run stop silently overwriting the first group's run fields. Also
count a previewed `created` only once per identity key, so `runs created: 2` for one row
cannot recur. Add a test that seeds two groups with one identity key and asserts the dry
run and the real run agree on `runs created`, `events re-keyed`, every reason count and the
exit status — the assertion `TestDryRunAndRealRunAgree` already makes, over a fixture that
can actually break it.

---

## Warnings

### NF-15: NF-01's total partition stops one shape short — a *foreign*-attributed stale `RUN:{pk}:{date}` event is still D-16's forbidden third outcome, and unlike its sibling path it is not even counted

**File:** `solsys_code/campaign_reconciler.py:555-599` (`_stale_dated_events`),
`solsys_code/campaign_reconciler.py:742-775` (`_detach_stale_family_events`), contrast with
`solsys_code/allocation_projector.py:699-716`
**Severity:** WARNING

**Issue:** `_stale_dated_events()` passes `stale_dated` — derived from `owned_events(run)`,
i.e. **namespace identity only** — into `_clearable_declined_and_unattributed()`. That
helper is a total partition over shapes (a)/(b)/(c-this-run), but `stale_dated` also
contains shape (d): an event in *this* run's `RUN:{pk}:{date}` namespace whose companion row
attributes it to a **different** run. Shape (d) is excluded from `_clearable_and_declined()`
(`run_id=run.pk`) and from the unattributed half (`run` is not null), so it lands in
neither — and `_detach_stale_family_events()` has no foreign counter at all.

The contrast with the sibling path is the tell: `project_allocation()`'s convergence, fixed
in the very same commit, computes `foreign_stale_count = stale_qs.count() -
writable_stale.count()`, logs a warning naming it, and folds it into `blocked`
(`allocation_projector.py:699-712`). `_detach_stale_family_events()` got no equivalent, so
its foreign shape is silent rather than merely untouched.

Reproduced (probe, executed; a `RUN:{pk}:2026-06-01` event outside the run's window, with a
companion row pointing at another run):

```
PROBE-1 result: ReconcileResult(created=1, updated=0, unchanged=0, blocked=0,
                skipped_nights=0, detached=0, detach_declined=0, retired=0, rekeyed=0,
                legacy_deleted=0)
PROBE-1 still exists: True
PROBE-1 log lines: []          # no warning emitted at all
PROBE-1 second sweep: ReconcileResult(..., unchanged=1, all other counters 0)
```

Leaving the row alone is the *correct policy* (T-29-19 — a human attribution outranks a
sweep). The defect is that the event is in a key family this phase retires entirely, so no
code path will ever revisit it — run B can never touch it, because it is in run A's
namespace — and the operator is never told it exists. That is exactly the
"permanently-orphaned events" outcome `models.py`'s cascade docstring and
`ReconcileResult.legacy_deleted`'s "no third outcome" contract both say the design exists to
prevent.

**Fix:** mirror the sibling path — report the foreign shape rather than deleting it:

```python
def _stale_dated_events(run, active_urls, claimed_legacy_urls=frozenset()) -> tuple[list[int], int, int]:
    _stale_bare, stale_dated = _split_stale_owned_events(run, active_urls)
    if claimed_legacy_urls:
        stale_dated = stale_dated.exclude(url__in=claimed_legacy_urls)
    writable_dated = stale_dated.filter(
        Q(telescope_label_meta__isnull=True)
        | Q(telescope_label_meta__run__isnull=True)
        | Q(telescope_label_meta__run=run)
    )
    foreign = stale_dated.count() - writable_dated.count()
    deletable, declined = _clearable_declined_and_unattributed(run, writable_dated)
    return deletable, declined, foreign
```

and in `_detach_stale_family_events()` / `reconcile_run()`'s dry-run twin, fold `foreign`
into `blocked` with the same warning `project_allocation()` already logs, so
`len(deletable) + declined + foreign == stale_dated.count()` holds here too — the invariant
the `project_allocation()` fix wrote into its own comment. Add a regression test for a
foreign-attributed `RUN:{pk}:{date}` asserting a non-zero counter and a log line.

### NF-16: NF-09's fix traded a double-count for a false operator message — a human-confirmed legacy decline is now reported *only* as "blocked — owned by someone else"

**File:** `solsys_code/allocation_projector.py:585-606`,
`solsys_code/management/commands/reconcile_campaign_runs.py:110-111`,
pinned by `solsys_code/tests/test_allocation_projector.py:415-422`
**Severity:** WARNING

**Issue:** NF-09's remedy (claim the legacy url whatever the decision) landed as written and
does remove the double-count. But the counter that survives is `totals['blocked']`
(`:606`), and `reconcile_campaign_runs` renders that as:

```
Run pk=N: 1 event(s) blocked -- owned by someone else
```

For this case that sentence is false twice over: the event is in **this** run's own
`RUN:` namespace, and it is `confirmed_by`-stamped to **this** run — nobody else owns it. The
counter that carries the true explanation, `detach_declined` ("superseded entr{y,ies} left
attributed — a person confirmed…"), is now deliberately zero, and the new test asserts that
zero as the intended contract (`self.assertEqual(result.detach_declined, 0)`), so the
mis-categorisation is now pinned rather than latent. This is the same class of defect NF-06
was filed for — a counter whose operator-facing text contradicts the condition that produced
it.

**Fix:** keep the single-count invariant, but route the decision to the counter whose
message is true. `project_allocation()` already returns a `ReconcileResult`, which carries
`detach_declined`:

```python
elif confirmed_declined:
    logger.warning(
        'Allocation retire declined: legacy event pk=%s is human-confirmed to run pk=%s '
        '-- an automated retirement never clears it.', legacy_event.pk, run.pk,
    )
    totals['detach_declined'] += 1      # not 'blocked' -- nobody else owns it
```

(`totals` will need `'detach_declined': 0` seeded, and `reconcile_run()`'s
`result._replace(detach_declined=...)` at `campaign_reconciler.py:841` must **add** to the
branch's value rather than overwrite it — overwriting is itself a latent bug the moment any
branch sets that field.) Leave the `_may_write()`-False case under `blocked`, where the
message is accurate. Update the two tests' expectations accordingly.

### NF-17: the NF-09 fix invalidated `claimed_legacy_urls`' own contract, which still says the real-mode exclusion is a no-op

**File:** `solsys_code/campaign_reconciler.py:715-718`, and the mirroring clause at
`solsys_code/allocation_projector.py:525-529`
**Severity:** WARNING

**Issue:** `_detach_stale_family_events()`'s `claimed_legacy_urls` docstring still states:

> real-mode is unaffected either way since a claimed legacy url has already left the
> ``RUN:`` namespace in the database by the time this function runs.

That was true when the set only ever held re-keyed or deleted urls. After the NF-09 fix the
set also holds urls that were **blocked** or **declined** — rows that, by definition, were
*not* written and are still sitting in the `RUN:` namespace when `_stale_dated_events()`
runs. The exclusion is therefore load-bearing in real mode now, not a no-op: it is the only
thing preventing the second count. The `project_allocation()` return-value docstring was
updated for NF-09 (`:519-534`) but its caller-side twin was not, so the module's two
descriptions of the same set now contradict each other — and the stale one is the sentence a
future maintainer would rely on when deciding the exclusion is safe to drop.

**Fix:** replace the clause with the real rule, e.g. *"in real mode the exclusion is a no-op
for a re-keyed or deleted url (it has already left the `RUN:` namespace) and load-bearing for
a blocked or declined one (it has not) — dropping it would restore NF-09's double count."*

### NF-18: `_check_event_night()`'s docstring denies the mutation its own parity depends on

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:221-222`
**Severity:** WARNING

**Issue:**

```
claimed_nights: nights already claimed by an earlier event in this same group;
    mutated by neither this function nor its callers.
```

Both callers mutate it — `claimed_nights.add(night)` at `:466` (dry run) and `:499` (real).
The mutation is the mechanism by which the in-run collision check works at all, and the
placement of that `.add()` relative to the per-event savepoint is a documented correctness
requirement two lines above it (`:495-498`). In the one helper whose entire purpose is to
make the two branches provably agree, a docstring that denies the state-carrying behaviour
of the parameter that carries the parity state is the worst place for this error: a reader
trusting it would conclude the set is read-only and that hoisting or sharing it is safe —
which is, as it happens, exactly the change NF-14 needs.

**Fix:** `claimed_nights: nights already claimed by an earlier event in this same group.
Read here; the CALLER adds the night after its own write (or preview) succeeds, so a failed
event leaves the night free for a later one.`

### NF-05 (carried forward): the cutover's group savepoint rolls back the writes but not the counters — and the NF-02 fix widened the double-reporting

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:421-426`,
`:467`, `:500`, `:507-511`
**Severity:** WARNING

**Issue:** Unchanged from iteration 2 and now worse. `runs_created`/`runs_updated`/
`runs_unchanged` and `events_rekeyed` are still plain ints incremented **inside** the group
`with transaction.atomic()`; the group-level `except` at `:507` rolls back every write and
leaves them at their post-write values.

The second half — `_mark_unexplained(events, ...)` at `:510` re-marking the **whole** group,
including events the loop already marked — got strictly worse with the NF-02 fix, because
there are now three per-event categories that can be marked inside the same atomic block
(`_WINDOW_MISMATCH`, `_KEY_COLLISION`, `_OTHER`) instead of one pre-loop category.

Reproduced (probe, executed; one event foreign-attributed, then a group-level failure after
the run write):

```
PROBE-3 stdout: Done. candidates: 3, groups: 1, runs created: 1, updated: 0, unchanged: 0,
                events re-keyed: 0, unexplained: 4
  unexplained (foreign_attribution): 1
  unexplained (other): 3
PROBE-3 stderr: pk=3: already attributed to a different CampaignRun
                pk=1: RuntimeError: group-level boom
                pk=2: RuntimeError: group-level boom
                pk=3: RuntimeError: group-level boom      <-- pk=3 reported twice
PROBE-3 cutover runs actually in db: 0                    <-- summary says "runs created: 1"
PROBE-3 alloc events: 0
```

`unexplained: 4` for a three-event group, and a one-time production migration's summary
claiming a run it did not create.

**Fix:** as in iteration 2 — accumulate the group's counters into locals and fold them into
the totals only after the `with` block exits successfully, and mark only the events not
already marked:

```python
already_marked = {e.pk for e, _c, _r in unexplained}
_mark_unexplained([e for e in events if e.pk not in already_marked], _OTHER,
                  f'{type(exc).__name__}: {exc}')
```

### NF-10 (carried forward): the dry-run short-circuit still hides the one failure mode `night_bounds()` raises

**File:** `solsys_code/allocation_projector.py:657-667`, `:282-297`
**Severity:** WARNING

**Issue:** Unchanged. `_mint_fields()` remains the only caller of `night_bounds()`, and the
dry-run create path still returns before it. The NF-03 rewrite did **not** remove the
inversion class — it only removed the *site-geometry* cause; an operator-entered
`night_start_utc > night_end_utc` still raises.

Reproduced (probe, executed; La Silla, one-night window, `night_start_utc=23:00`,
`night_end_utc=22:00`):

```
PROBE-2 dry-run: ReconcileResult(created=1, ..., skipped_reason=None)
PROBE-2 real run raised ValueError: Computed an inverted allocation-night span for run
        pk=1 night=2026-07-09: start=2026-07-09T23:00:00+00:00 >= end=2026-07-09T22:00:00+00:00.
```

`reconcile_campaign_runs --dry-run` reports `would_create: 1`; the real sweep reports the run
under `failed`. The preview cannot show the operator the one condition that stops the run
from projecting.

**Fix:** as in iteration 2, but against the new API — keep `_mint_fields()` skipped and
validate the set boundaries with the astropy-free span the NF-03 rewrite already provides:

```python
if dry_run:
    if run.night_start_utc is not None and run.night_end_utc is not None:
        span = _night_span_utc(run, night)
        if _time_of_day_to_datetime(run.night_start_utc, night, span) >= _time_of_day_to_datetime(
            run.night_end_utc, night, span
        ):
            raise ValueError(...)   # the same message night_bounds() raises
    totals['created'] += 1
    continue
```

### NF-04 (carried forward, out of this review's file scope): WR-01's re-ordering leaves the observation event unattributed on the save that creates it

**File:** `solsys_code/observation_projector.py:610-626`, `solsys_code/allocation_projector.py:406-473`
**Severity:** WARNING

**Issue:** Not addressed — `observation_projector.py` is unchanged since `5b2bd38`. The
attribution bridge still runs before `project_record()`, so on the save that first creates
the observation event `_sync_observation_attribution()`'s
`CalendarEvent.objects.filter(url=...).first()` returns `None` and the adoption is silently
skipped, leaving the event that takes over the retired night with no campaign attribution
until a later save. See iteration 2's NF-04 for the executed reproduction and the fix block.

### NF-08 (carried forward, out of this review's file scope): the classical loader's widened `except` tuple wraps the whole reconcile call

**File:** `solsys_code/management/commands/load_telescope_runs.py:316-321`
**Severity:** WARNING

**Issue:** Not addressed — `load_telescope_runs.py` is unchanged since `5b2bd38`. Note that
this one is now *more* reachable: NF-03's fix made `ZoneInfo(run.site.timezone)` a per-night
call inside `_night_span_utc()`, and `ZoneInfoNotFoundError` subclasses `KeyError`, so a
malformed `Observatory.timezone` is still swallowed by that ~80-line handler and reported as
`Line N: 'Some/Zone' (line text: …)`. See iteration 2's NF-08 for the fix block.

### NF-11 (carried forward): a vacuous assertion in the cutover sequence-contract test

**File:** `solsys_code/tests/test_cutover_classical_allocations.py:531-533`
**Severity:** WARNING

**Issue:** Not addressed. `rekeyed_count = 1; legacy_deleted_count = 1;
self.assertEqual(rekeyed_count + legacy_deleted_count, 2)` still asserts `1 + 1 == 2` under a
comment claiming to verify three-group reconciliation, inside `TestCutoverSequenceContract`.

**Fix:** assert the real counters from the reconciles the test performs, or delete the three
lines — the preceding assertions already prove both outcomes against the database.

### NF-12 (carried forward): the cutover's local `writable_events` collides with the codebase's ownership-helper name

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:383-389`
**Severity:** WARNING

**Issue:** Not addressed. `writable_events` is still bound to a plain `list` of blank-url
events built from a different rule than `campaign_reconciler.writable_events(run)` — a name
this batch made *more* prominent, since `_may_write()`/`writable_events()`/
`writable_allocation_events()` are now documented as one three-way predicate family and the
module already imports underscore-named helpers across module boundaries.

**Fix:** rename to `unattributed_events` or `convertible_events`.

### NF-13 (carried forward): the WR-11 runbook edit's broken sentence

**File:** `docs/runbooks/telescope_runs_calendar.rst:912-914`
**Severity:** WARNING

**Issue:** Not addressed. The sentence defining what the one-time production migration
converts still reads "…whose events agree on their campaign, and whose events are not
already attributed to a different run, are not already claimed on a colliding night." — a
comma splice with no conjunction and a dangling subject. The paragraph gained a
`window_mismatch` clause immediately below it in this batch, so the file was edited without
the adjacent breakage being noticed.

**Fix:** `… whose events agree on their campaign, whose events are not already attributed to
a different run, and whose derived observing nights are not already claimed.`

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep (re-review, iteration 3)_
