---
phase: 35-allocation-layer-classical-cutover
fixed_at: 2026-09-14T06:22:00Z
review_path: .planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
---

# Phase 35: Code Review Fix Report

**Fixed at:** 2026-09-14T06:22:00Z
**Source review:** .planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md
(iteration 3 — re-review after the `260913-ti1`/`260913-ti3`/`260913-rmd` fixes)
**Iteration:** 1

**Summary:**
- Findings in scope: 12 (1 critical/BLOCKER — NF-14; 11 warning — NF-04, NF-05, NF-08,
  NF-10, NF-11, NF-12, NF-13, NF-15, NF-16, NF-17, NF-18)
- Fixed: 12
- Skipped: 0

All fixes were made in an isolated git worktree/branch (`gsd-reviewfix/35-606165`),
fast-forward-merged onto `issue37-telescope-runs-calendar`. Every commit passed
`pre-commit run ruff`/`ruff-format` and the project's own pre-commit test-suite gate.
NF-04 and NF-08 were flagged in the review as "out of this review's file scope"
(`solsys_code/observation_projector.py`, `solsys_code/management/commands/load_telescope_runs.py`)
but were fixed anyway per the task's explicit instruction that file scope and fix scope
are not the same thing.

## Fixed Issues

### NF-14 (BLOCKER): the cutover's dry run still exits zero on a fixture the real run rejects — no identity-key collision guard, silent merge, wrong reason label

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`,
`solsys_code/tests/test_cutover_classical_allocations.py`,
`docs/runbooks/telescope_runs_calendar.rst`,
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
**Commit:** `aabaea7`
**Applied fix:** `_source_identifier()` deliberately ignores the schedule line's status
word, so two `Source line:` groups differing only in status (e.g. an `allocation` line
and a `cancelled` line for the same telescope/instrument/window) resolved to the same run
identity key. The cutover grouped candidates by the raw `source_line` string, so those
were two GROUPS mapping to one RUN: the second group's `insert_or_create_campaign_run()`
found-and-updated the first group's already-created row, silently overwriting its fields,
and its own events were then rejected under `key_collision` — a reason whose documented
remedy (delete/re-attribute the duplicate row) is wrong for a legitimate second schedule
line. Added a `load_telescope_runs.py`-style `seen_keys` guard: the moment a group's
identity key is claimed, a second group sharing it is marked `duplicate_identity` and
skipped entirely, on both `--dry-run` and the real pass, before either ever attempts a
write for it. `claimed_nights` is now scoped to the identity key (`claimed_by_key[key]`)
rather than a fresh `set()` per group, so the dry run's night-level check would also catch
the collision even if the `seen_keys` guard were ever bypassed by a future refactor.
Updated the module docstring, the reason vocabulary, and the operator runbook (both the
"always dry-run first" parity claim and the identity-key-collision section, which wrongly
asserted this guard already existed on the cutover path — this fix also closed **NF-13**'s
broken sentence in the same paragraph, since both live in the same edited block). Added a
self-contained, self-cleaning demonstration cell to the paired
`reconcile_campaign_runs_demo.ipynb` notebook (verified via a real `jupyter nbconvert
--execute --inplace` run against a scratch copy of the developer database — confirmed the
new cell reproduces the exact `duplicate_identity=3` collision and that its cleanup
restores the notebook's downstream end-state assertions unchanged). Added
`TestDuplicateIdentityKeyAcrossGroups` (two tests: the real-run non-merge assertion, and a
dry-run/real-run parity assertion) to `test_cutover_classical_allocations.py`.

### NF-15: a foreign-attributed leftover `RUN:{pk}:{date}` event was silently dropped — no counter, no log line

**Files modified:** `solsys_code/campaign_reconciler.py`, `solsys_code/tests/test_campaign_reconciler.py`
**Commit:** `c689eef`
**Applied fix:** `_stale_dated_events()` passed `stale_dated` — derived from namespace
identity alone — into `_clearable_declined_and_unattributed()`, a total partition over
shapes (a)/(b)/(c-this-run). But `stale_dated` also contains shape (d): an event in this
run's own `RUN:{pk}:{date}` namespace whose companion row attributes it to a DIFFERENT
run. Shape (d) matched neither half of the partition, so it was left alone (correctly —
a human attribution outranks a sweep) but silently, with no counter and no log line, in a
key family this phase retires entirely. `_stale_dated_events()` now computes the same
`foreign` count `project_allocation()`'s mirror `ALLOC:` convergence step already
computes, logs a warning naming it, and returns it as a third tuple element;
`_detach_stale_family_events()` and `reconcile_run()`'s dry-run twin fold it into the
run's own `blocked` total. Extended the existing
`test_foreign_attribution_is_neither_deleted_nor_detached` regression test to assert
`blocked==1`, the warning log line (via `assertLogs`), and idempotency across a second
sweep.

### NF-16: a human-confirmed legacy-retire decline was reported under `blocked` — a false "owned by someone else" message

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `1f7b514`
**Applied fix:** NF-09's earlier fix correctly stopped double-counting a declined legacy
retirement, but the surviving counter was `totals['blocked']`, which
`reconcile_campaign_runs` renders as "blocked — owned by someone else". That message is
false twice over for this shape: the event is in THIS run's own `RUN:` namespace, and
`confirmed_by`-stamped to THIS run — nobody else owns it. Routed the decision to
`totals['detach_declined']` instead, whose rendered message ("a person confirmed them…")
is the true one; seeded `'detach_declined': 0` into `project_allocation()`'s totals dict.
Updated the one test pinning the old (wrong) categorization
(`test_retiring_a_night_never_deletes_a_human_confirmed_legacy_event`) to assert
`blocked==0, detach_declined==1`.

### NF-17: `claimed_legacy_urls`' docstring still claimed the real-mode exclusion is a no-op, after NF-09 made it load-bearing

**Files modified:** `solsys_code/campaign_reconciler.py`, `solsys_code/allocation_projector.py`
**Commit:** `1f7b514` (same commit as NF-16, since NF-16's fix is exactly what makes this
docstring's claim newly false in a way worth documenting precisely)
**Applied fix:** NF-09 widened `claimed_legacy_urls` to also hold blocked/declined urls
(not written), but both of this set's docstrings — the caller's in
`campaign_reconciler.py` and the mirroring return-value docstring in
`allocation_projector.py` — still claimed the caller's exclusion is a no-op in real mode
"either way". That is true only for a re-keyed or deleted url; it is load-bearing (not a
no-op) for a blocked or declined one, since that row is, by definition, still sitting in
the `RUN:` namespace when the caller's convergence step runs. Fixed both copies of the
claim to state the real rule (no-op for re-keyed/deleted, load-bearing for blocked/declined).

### NF-18: `_check_event_night()`'s docstring denied the mutation its own parity depends on

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`
**Commit:** `b4b5b9e`
**Applied fix:** The docstring claimed `claimed_nights` is "mutated by neither this
function nor its callers", but both callers mutate it via `claimed_nights.add(night)`
right after their own write (or preview) succeeds — the mechanism the in-run collision
check depends on. Replaced with the review's suggested correction: "Read here; the CALLER
adds the night after its own write (or preview) succeeds, so a failed event leaves the
night free for a later one."

### NF-05: the cutover's group savepoint rolled back writes but not counters, and re-marked already-explained events

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`
**Commit:** `8d22999`
**Applied fix:** `runs_created`/`runs_updated`/`runs_unchanged`/`events_rekeyed` were
incremented inside the group's `with transaction.atomic()` block as the outer totals
themselves, so a group-level rollback undid every write but left the counters at their
post-write values. Routed the increments through group-local counters (`group_created`,
etc.) folded into the outer totals only once the `with` block exits successfully. Also
stopped the group-level `except` from re-marking every event in the group
unconditionally — an event already rejected per-event (e.g. foreign attribution, decided
before the savepoint even opened) was reported and counted a second time, under a second,
misleading reason. Now marks only events not already present in `unexplained`.

### NF-14 (cross-reference): see above — reject a second cutover group sharing an earlier group's run identity key.

### NF-11: a vacuous assertion in the cutover sequence-contract test

**Files modified:** `solsys_code/tests/test_cutover_classical_allocations.py`
**Commit:** `86be342`
**Applied fix:** `TestCutoverSequenceContract` asserted `1 + 1 == 2` over two local
literals — exercising no production code, under a comment claiming to verify three-group
reconciliation. Both outcomes it claimed to cover (the re-keyed legacy night, the deleted
legacy night) are already pinned against the database by the preceding assertions in the
same test, so the no-op assertion was removed rather than duplicated (per the review's
own stated alternative fix).

### NF-12: the cutover's local `writable_events` collided with the codebase's ownership-helper name

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`
**Commit:** `86be342` (same commit as NF-11, both small mechanical fixes in the same file)
**Applied fix:** Renamed the local `writable_events` list (built from "has no companion
row pointing at any run", never namespace-identity) to `unattributed_events`, which names
what it actually holds and no longer collides with
`campaign_reconciler.writable_events(run)`, the canonical queryset-level ownership helper
referenced throughout the reconciler and this module's own docstrings.

### NF-04 (carried forward, out of review's file scope): the observation event was unattributed on the save that created it

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/tests/test_observation_projector_signals.py`
**Commit:** `30112a0`
**Applied fix:** WR-01's earlier fix moved the D-11 linked-run re-project step above the
base projection's own facility guard, to stay facility-independent even for a facility the
base guard didn't (yet) handle. But on the save that FIRST CREATES (or re-keys) a
record's own event, `project_allocation()`'s attribution bridge looked that event up by
url BEFORE it existed, silently skipped the adoption, and left the event that takes over
a retired night with no campaign attribution until a later save or sweep. Reordered so
base projection (`project_record()`) runs first, then the D-11 step — the D-11 step only
depends on `instance.campaign_run_links`, never on the base projection's action/stage or
facility, so its own facility-independence (WR-01) is unchanged by this reordering; it
also now runs even if `project_record()` raises (matching the review's suggested fix,
consistent with D-11's "neither failure may abort the caller's save" contract). Added
`test_placing_the_block_attributes_the_records_own_event_on_the_creating_save`, which
makes a record unprojectable at creation (no instrument signal), then supplies both the
instrument and the placed block in one save, asserting the attribution lands on that
exact save.

### NF-08 (carried forward, out of review's file scope): the classical loader's widened `except` tuple wrapped the whole reconcile call

**Files modified:** `solsys_code/management/commands/load_telescope_runs.py`
**Commit:** `d57b461`
**Applied fix:** WR-09's `except` tuple widening (`KeyError`) wrapped the entire per-line
body — including `write_and_reconcile_campaign_run()` and `reconcile_run()` — not just the
`_CLASSICAL_RUN_STATUS[parsed.status]` lookup it was added for. Narrowed the `KeyError`
catch to that one lookup (a dedicated `try`/`except KeyError` right where it happens), and
wrapped `write_and_reconcile_campaign_run()` (which has no transaction boundary of its
own) in `transaction.atomic()` — without it, a `reconcile_run()` exception left the
`CampaignRun` row it had just written committed while the line was counted under
`run_skipped`.

### NF-10: the dry-run create path hid the one failure mode `night_bounds()` raises

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `80cefed`
**Applied fix:** `_mint_fields()` is the only caller of `night_bounds()`, where CR-06's
inversion guard lives; the dry-run create path skipped `_mint_fields()` entirely (WR-03)
to avoid a wasted `sun_event()` call, but that also hid the one failure mode an
operator-set (not site-derived) `night_start_utc`/`night_end_utc` pair can raise. Extracted
`night_bounds()`'s inversion check into a shared `_raise_if_inverted()` helper so both the
real create path and the dry-run preview call the exact same guard over the exact same two
datetimes. When both sub-night fields are set, the dry-run path now resolves them via the
same `zoneinfo`-only span `night_bounds()` itself uses (no `sun_event()` call) and raises
identically; a null field's boundary depends on the sun event and stays unchecked,
matching `_span_needs_remint()`'s own null-field convention. Added
`test_dry_run_of_a_brand_new_inverted_window_also_raises`, reproducing the sibling Sydney
fixture on a brand-new night and asserting the dry run and the real run raise the
identical error.

### NF-13 (cross-reference): the WR-11 runbook edit's broken sentence

**File:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `aabaea7` (fixed as part of NF-14's runbook edit, since both findings touch the
same paragraph)
**Applied fix:** "…whose events agree on their campaign, and whose events are not already
attributed to a different run, are not already claimed on a colliding night." → "…whose
events agree on their campaign, whose events are not already attributed to a different
run, and whose derived observing nights are not already claimed." — exactly the review's
suggested correction.

## Skipped Issues

None — all 12 in-scope findings were fixed.

## Verification

All fixes were verified in the isolated `gsd-reviewfix/35-606165` worktree
(`/home/tlister/git/fomo_devel/.claude/worktrees/rf-35-606165-1789363316`):

- **Tier 1 (always):** every modified file was re-read after editing to confirm the fix
  text was present and surrounding code intact.
- **Tier 2 (preferred):** `python -c "import ast; ast.parse(...)"` syntax-checked every
  modified `.py` file before each commit; `pre-commit run ruff --files ...` was run (and
  passed clean) on every commit's changed files, and `pre-commit run ruff-format
  --all-files` / `pre-commit run ruff --all-files` both passed clean across the whole tree
  at the end.
- **Test verification:** after each fix, the directly-relevant Django test module(s) were
  run and confirmed green before committing. After all 9 commits (covering all 12
  findings), a full regression pass ran clean:
  - `python manage.py test solsys_code.tests.test_allocation_projector
    solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_cutover_classical_allocations`
    — **150 tests, all passing** (up from the review's own 147, +3 new regression tests
    from this fix pass net of the removed NF-11 vacuous assertion).
  - The full project test-gate label set (every `solsys_code`/`solsys_code_observatory`
    test module except the known-segfaulting `test_views.TestEphemeris`) — **all passing**
    (exit code 0), matching `.planning/config.json`'s configured `test_command`.
  - `pre-commit`'s own bundled test-suite gate (which runs the project's full `manage.py
    test` invocation) also passed on every one of the 9 commits.
- **Notebook re-execution:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
  was regenerated via a real `jupyter nbconvert --to notebook --execute --inplace` run
  (against a scratch copy of the developer database, matching the notebook's own
  documented isolation contract) after inserting NF-14's new demonstration cell — no cell
  raised, and the notebook's own pre-existing end-state assertions (cells asserting zero
  date-bearing `RUN:` nights, byte-identical facility events, etc.) still pass unchanged,
  confirming the new cell's self-cleanup left no residue.
- **Environment note:** all test runs, gate checks, and the notebook re-execution above ran
  inside the isolated worktree (`.claude/worktrees/rf-35-606165-1789363316`), not the main
  checkout — the worktree shares the repository's installed environment (editable install,
  dependencies, `~/.cache/sorcha/` SPICE kernels) but is a separate git working tree on its
  own temporary branch, fast-forwarded onto `issue37-telescope-runs-calendar` at cleanup.
  Two generated, gitignored artifacts (`src/fomo/_version.py`, `src/fomo_db.sqlite3`) had
  to be copied from the main checkout into the worktree to make the test suite and the
  paired notebook runnable there; neither is tracked by git, and both remain
  worktree-local (removed along with the worktree at cleanup).

No paired-docs update was skipped: NF-14 (and NF-13, in the same paragraph) required and
received a runbook correction and a real, executed notebook demonstration cell — see the
NF-14 entry above. No other fix in this pass changed `campaign_reconciler.py`'s,
`allocation_projector.py`'s, `cutover_classical_allocations.py`'s, or
`observation_projector.py`'s *documented, demonstrated* behavior in a way that would make
an existing runbook sentence or notebook cell's output inaccurate (checked directly against
each affected section of `docs/runbooks/telescope_runs_calendar.rst` and against the
notebook's recorded output for the developer-database fixture, which does not exercise the
NF-15/NF-16/NF-04/NF-08/NF-10 scenarios).

## Human Verification Recommended

None of these 12 fixes are classified by REVIEW.md as a logic-error-class finding in the
sense that would warrant `"fixed: requires human verification"` over plain `"fixed"` — each
is either a counter/categorization routing fix, a docstring correction, a missing guard
(with an exhaustive collision check mirroring an existing sibling pattern), or a signal
re-ordering whose correctness is pinned by a new, real regression test asserting the exact
scenario the review reproduced. That said, two are worth a human's attention given their
reach:

- **NF-14**: the identity-key collision guard changes production behavior for a one-time
  migration command (`cutover_classical_allocations`) that has likely already been run
  once against the real developer database (per the paired notebook's "worked example").
  A human should confirm no PRODUCTION database currently has two classical schedule lines
  sharing an identity key before the corrected command is next run there — the fix's own
  behavior (reporting `duplicate_identity` rather than silently merging) is safe either
  way (nothing is deleted or merged), but an operator seeing a NEW `duplicate_identity`
  count on a re-run that previously reported success should not be alarmed: it means this
  fix newly detects a pre-existing collision the unfixed command silently merged before.
- **NF-04**: the signal-ordering change in `observation_projector.py` is exercised by a
  new regression test reproducing the exact review-cited scenario, but signal-ordering
  changes are inherently sensitive to call-order assumptions elsewhere in the codebase; a
  human should confirm no other code path depends on the D-11 linked-run step running
  BEFORE base projection (none was found in this review).

---

_Fixed: 2026-09-14T06:22:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
