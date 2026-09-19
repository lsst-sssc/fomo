---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
fixed_at: 2026-09-19T15:00:36Z
review_path: .planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md
iteration: 1
findings_in_scope: 15
fixed: 15
skipped: 0
status: all_fixed
---

# Phase 37: Code Review Fix Report

**Fixed at:** 2026-09-19T15:00:36Z
**Source review:** .planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 15 (Critical: 3, Warning: 12 — Info findings excluded per `fix_scope`)
- Fixed: 15
- Skipped: 0

All in-scope findings were fixed. Two findings carry a **narrower scope than the review
suggested**, documented below with reasoning rather than silently applied — see CR-02's
rollup-caching note and WR-07's `telescope_runs.py` note. One finding (CR-02) surfaced a
necessary trade-off against an existing, deliberately-designed performance test, which was
resolved by updating that test's expectation rather than reverting the fix — documented
below as well.

## Fixed Issues

### CR-01: Tally cache key is blind to link creation/deletion

**Files modified:** `solsys_code/campaign_tally.py`, `solsys_code/tests/test_campaign_tally.py`
**Commit:** `26b778d`
**Applied fix:** `link_counts_for_runs()` now also returns a distinct-link-count and a
linked-record-id watermark (`link_version`, `Max('observation_record_id')`).
`build_tally_cache_key()` folds `records_count`/`link_version` into the cache key alongside
`records_version`, so a link create, a delete, or a delete-then-create pair that leaves the
count unchanged all produce a new key. `get_or_compute_tally()`/`tallies_for_runs()` updated
to pass the new fields. Added three regression tests reproducing the exact reported bugs:
linking a pre-existing record whose `modified` is older than the run's current max, and
removing a non-newest link — both previously left the cached tally unchanged.

### CR-02: Calendar `[U]` marker and table's unused count could disagree for up to an hour

**Files modified:** `solsys_code/campaign_tally.py`, `solsys_code/tests/test_campaign_tally.py`,
`solsys_code/tests/test_campaign_views.py`
**Commits:** `26b778d` (fix, combined with CR-01 per the review's own note that the two share
a root cause and touch the same functions), `6f0ae3f` (test follow-up)
**Applied fix:** `get_or_compute_tally()` and `tallies_for_runs()` now cache only the five
link/night-count tally fields; the three `unused_*` fields are recomputed **live, on every
call, including a cache hit**, from `is_unused_allocation_night()` — the same rule
`unused_night_decoration()` already evaluates live for the calendar's `[U]` marker. This
makes the table's Progress column and the calendar agree by construction, with no TTL wait.
Added regression tests (`test_unused_count_is_live_even_on_a_cache_hit`, both call sites)
that flip `run_status` to `CANCELLED` after warming the cache and assert the unused figure
moves on the very next call.

**Discovered trade-off (documented, not silently applied):** running `unused_*` live on
every call, even a cache hit, costs two bounded queries per rendered table row (one
allocation-event lookup, one proposal-allocation existence check). An existing test,
`test_page_query_count_does_not_grow_with_additional_cached_rows`, encoded a "zero marginal
query cost on an already-warmed page" contract that this fix structurally cannot satisfy —
you cannot have both "zero marginal cost" and "always live, never stale." Since the cost is
bounded by page size (25 rows after WR-04, not by the number of linked records), this is an
acceptable trade-off for closing a Critical, publicly-visible staleness bug. The test was
rewritten (commit `6f0ae3f`) to assert the new bounded per-row delta (exactly 2 queries) so a
future regression that makes the cost scale with *linked records* instead of *rendered rows*
is still caught.

**Scope note (deliberately narrower than the review's Fix section):**
`campaign_rollup()`/`get_or_compute_rollup()` (the campaign-list roll-up strip, used by
`CampaignListView`) were **left untouched**. Applying the same live-unused split there would
require calling `tallies_for_runs()` even on a rollup cache hit, which directly conflicts
with an existing, explicitly-titled test:
`test_campaign_list_query_count_bound_with_three_campaigns` (`test_campaign_views.py`,
docstring cites "D-10/T-37-19") asserts that a cached rollup adds **zero** further queries on
the anonymous, high-traffic campaign list page. That bound exists specifically because this
page is reachable anonymously and was already flagged by WR-05 as an amplification risk
(`O(campaigns × runs)` on a cold cache) — relaxing it the way CR-02 relaxed the per-row table
test would reintroduce exactly the amplification WR-05 fixes. The residual gap CR-02
describes ("`campaign_rollup()`'s summary strip disagree[ing] with the sum of the rows
rendered directly beneath it") therefore still exists at the roll-up-strip level only, bounded
to `TALLY_CACHE_TTL_SECONDS` (1 hour), same as before this fix. This needs a human product
decision (accept the existing rollup staleness bound, or relax the T-37-19 query bound) before
it can be closed — flagging for follow-up rather than picking one side silently.

### CR-03: `facility_for()` unguarded on public tally paths

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/campaign_tally.py`,
`solsys_code/campaign_gap.py`, `solsys_code/tests/test_campaign_tally.py`,
`solsys_code/tests/test_campaign_gap.py`, `solsys_code/tests/test_observation_projector.py`
**Commit:** `99588c5`
**Applied fix:** Added `observation_projector.facility_for_or_none()`, returning `None`
instead of raising `ImportError` for an unconfigured facility name. Switched the two real
call sites the orchestrator confirmed (`campaign_tally.night_counts_for_run()` and
`campaign_gap.observation_claimed_dates()`) to use it —
`night_counts_for_run()` now skips an unclassifiable record entirely (contributes to no
night set, same as an INCONSISTENT/QUEUED record today);
`observation_claimed_dates()` folds it into the existing `site_unknown_count` data-quality
signal. Per the orchestrator's correction, `calendar_display_extras.py` contains no direct
`facility_for()` call — its `run_tally()` tag reaches the fixed code transitively through
`get_or_compute_tally()`, so no third call site needed changing. Added regression tests with
`facility='NOT_A_CONFIGURED_FACILITY'` at all three affected layers (the new
`facility_for_or_none()` helper itself, `night_counts_for_run()`, and
`observation_claimed_dates()`).

**Landed in the same commit as WR-02** (below) — both are adjacent edits inside
`night_counts_for_run()`'s per-record loop and could not be cleanly separated at the hunk
level.

### WR-01: `ApprovalQueueTable`'s always-unavailable `Progress` column

**Files modified:** `solsys_code/campaign_tables.py`, `solsys_code/tests/test_campaign_approval.py`
**Commit:** `3c14424`
**Applied fix:** Added `'progress'` to `ApprovalQueueTable.Meta.exclude`, alongside the
existing `weather`/`observation_outcome`/`publication_plans` exclusions. Added a regression
test asserting `'progress'` is absent from `ApprovalQueueTable`'s columns while still present
on the base `CampaignRunTable`.

### WR-02: `night_counts_for_run()`'s `.only()` omitted `parameters`

**Files modified:** `solsys_code/campaign_tally.py` (commit `99588c5`, combined with CR-03 —
see above)
**Applied fix:** Added `'parameters'` to the `.only()` field list and corrected the
comment above it, matching `campaign_gap.observation_claimed_dates()`'s existing discipline.
Covered by the existing `test_record_time_window_raising_is_skipped_never_aborts` test, which
already exercises the `parameters`-fallback branch this fix restores field access to.

### WR-03: `campaign_rollup()`'s `.only()` triggered deferred-field queries

**Files modified:** `solsys_code/campaign_tally.py`, `solsys_code/tests/test_campaign_tally.py`
**Commit:** `900dfd3`
**Applied fix:** Added `select_related('site')` and named `run_status`/`site__timezone`/
`site__obscode` in `.only()`. Added a query-count regression test
(`test_only_does_not_trigger_deferred_field_queries`) pinning the roll-up's cost for one run
at 5 queries; verified by temporarily reverting the fix that it was 7 before.

### WR-04: Campaign table computed tallies for the whole filtered queryset, not the rendered page

**Files modified:** `solsys_code/campaign_views.py` (commit `05359bb`, combined with WR-05
below — both bound query fan-out on adjacent public pages and were reviewed together)
**Applied fix:** `get_table_kwargs()` now slices `self.object_list` to the same page window
`RequestConfig` itself resolves (the `'page'` query param), left unevaluated so it stays a
single `LIMIT`/`OFFSET` subquery — no extra query over the previous unbounded version. An
out-of-range/non-integer page number degrades to page 1's pks (`RequestConfig` separately
clamps the actually-rendered page).

### WR-05: `CampaignListView` looped roll-ups over an unpaginated, anonymous campaign list

**Files modified:** `solsys_code/campaign_views.py`, `solsys_code/tests/test_campaign_views.py`,
`src/templates/campaigns/campaign_list.html`
**Commit:** `05359bb`
**Applied fix:** Added `paginate_by = 100` to `CampaignListView` (bounding worst-case fan-out
without ever hiding a campaign's tally behind a cache-hit gate, which would have violated
TALLY-01's "visible to any visitor" guarantee), minimal Bootstrap4 pagination controls in the
template, and explicit `.order_by('name')` on the queryset (pagination needs a deterministic
order; Django's own `UnorderedObjectListWarning` surfaced once pagination was added). Added
tests covering pagination triggering and that a second page's campaigns remain reachable.

### WR-06: `unused_night_decoration()` had no `is_publicly_visible` gate

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`,
`solsys_code/tests/test_calendar_display_extras.py`
**Commit:** `b5da9db`
**Applied fix:** Added the same `run is None or not run.is_publicly_visible` gate
`campaign_decoration()`/`run_tally()` already use. Added a regression test with a
`PENDING_REVIEW` run's elapsed allocation night, asserting the decoration is now `None`.

### WR-07: Proposal code interpolated into a credentialed URL without quoting/validation

**Files modified:** `solsys_code/proposal_allocation.py`, `solsys_code/tests/test_proposal_allocation.py`
**Commit:** `e3bd492`
**Applied fix:** Added `_PROPOSAL_CODE_RE` (a conservative charset check covering every real
LCO/SOAR/ESO proposal-code shape already in this codebase's fixtures, e.g. the ESO-style
`'0110.C-0234'`), validated before the request is built, and `quote()`d into the URL path.
Added tests for path-traversal-like input, `?`/`#` characters, an over-long code, a blank
code, and that a valid ESO-style code both passes validation and reaches a correctly-quoted
URL.

**Scope note (deliberately narrower than the review's Fix section):** the review's suggested
fix also proposed rejecting an over-long/ill-formed token in
`telescope_runs._resolve_proposal()` (the classical-file ingestion grammar) so it "never
reaches the DB." This was **not applied** — `_resolve_proposal()` is a live, real-schedule-file
parser, and tightening its accepted grammar without dedicated testing against real classical
schedule files risked rejecting a legitimately-formatted proposal code the operator has
actually used. The sink-level fix in `proposal_allocation.py` (validate immediately before
the credentialed request is built) closes the actual security/DB-overflow risk without that
blast radius. Flagging the source-level hardening as a follow-up for a human to scope against
real schedule-file examples.

### WR-08: `unused_hours_for()` summed across every semester; stale rows never pruned

**Files modified:** `solsys_code/proposal_allocation.py`, `solsys_code/tests/test_proposal_allocation.py`
**Commit:** `2c8601e`
**Applied fix:** `unused_hours_for()` now accepts an optional `semester` parameter, defaulting
to the proposal's own alphabetically most-recent stored semester (semester strings sort
correctly as plain text). `store_proposal_allocations()` now prunes any stored row for the
`proposal_code` whose `(semester, instrument_type, allocation_type)` key is absent from the
current portal response. Added tests for cross-semester summation (only the most recent is
summed), explicit-semester override, no-rows-in-requested-semester, pruning on a later
response, and pruning never touching a different proposal's rows.

### WR-09: Bare marker literals survived in `observation_projector.py` and `calendar.html`

**Files modified:** `solsys_code/status_vocabulary.py`, `solsys_code/observation_projector.py`,
`solsys_code/templatetags/calendar_display_extras.py`,
`src/templates/tom_calendar/partials/calendar.html`, `solsys_code/tests/test_calendar_display_extras.py`
**Commit:** `dea1d3e`
**Applied fix:** Both `observation_projector.py` fallback literals (`'[F]'`, `'[?]'`) now
source from `status_vocabulary.MARKER`. Added a `'filterable'` key to
`status_vocabulary.LEGEND` (`True` only for the `UNUSED` entry) and switched
`calendar.html`'s click-to-filter legend swatch to branch on `entry.filterable` instead of
comparing `entry.marker == '[U]'`. Per the orchestrator's correction, this was scoped as a
small tidy (source the fallback + comparison from `status_vocabulary`), not a structural
refactor — `observation_projector.py:203` (a docstring, not code) was left untouched. Added a
regression test asserting exactly one legend entry (`[U]`) is filterable.

### WR-10: `ProposalTimeAllocationAdmin` left add/delete unguarded

**Files modified:** `solsys_code/admin.py`, `solsys_code/tests/test_admin.py`
**Commit:** `6d85e03`
**Applied fix:** Added `has_add_permission()`/`has_delete_permission()`, both returning
`False` unconditionally. Added both unit-level tests (calling the methods directly) and
HTTP-level tests — the latter needed correcting mid-fix: this project's
`tom_common.middleware.Raise403Middleware` turns every 403 into a 302 redirect to the login
page, so the HTTP-level tests assert that redirect rather than a raw 403 status code (a
project-wide convention discovered while verifying, not a defect in the fix).

### WR-11: Gap page's "Claimed nights" list not bounded by the requested date range

**Files modified:** `solsys_code/campaign_gap.py`, `solsys_code/tests/test_campaign_gap.py`
**Commit:** `91f1968`
**Applied fix:** `_compute_gap()`'s `'claimed_dates'`/`'observation_claimed_dates'` result
keys are now bounded to `[start, end]` at the point the result dict is built; `gap_dates`
keeps reading from the unbounded `claimed` set, unchanged (an out-of-range claim still
correctly removes a night from the observable set). Added a regression test with a claimed
run night and a claimed observation night both far outside the requested range, asserting
both are excluded from the display lists while `claimed_dates()` itself (the underlying,
intentionally-unbounded function) still reports them.

### WR-12: Runbook still documented retired `[CANCELLED]`/`[WEATHERED]` prefixes as current

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `3a19846`
**Applied fix:** Replaced the two stale bracket-word prefixes with the current `[C]`/`[W]`
markers in the declined-retirement note. Grepped the rest of the runbook for the same
strings — the only other occurrence (the "One-time title change (Phase 37)" note) is
correctly phrased in the past tense describing the pre-Phase-37 behavior, so it was left
unchanged.

## Skipped Issues

None — all 15 in-scope findings were fixed. Two findings (CR-02, WR-07) were fixed with a
narrower scope than the review's suggested code, documented above with reasoning rather than
silently applied; one finding (CR-02) required updating an existing test's expectation to
reflect a necessary trade-off, also documented above.

## Verification

Every fix was verified against its directly affected Django test module(s)
(`python manage.py test solsys_code.tests.test_<module>`) after being applied, per this
agent's iterative workflow — all passed before each commit. All verification ran in the main
working tree (per `workflow.use_worktrees` being overridden to off for this task by explicit
orchestrator instruction — no isolated worktree was created), so the results are reproducible
directly from this checkout.

A final combined run across every test module touched by this fix pass
(`test_campaign_tally`, `test_campaign_gap`, `test_observation_projector`,
`test_campaign_approval`, `test_campaign_views`, `test_calendar_display_extras`,
`test_proposal_allocation`, `test_admin`, `test_status_vocabulary`, `test_calendar_template`)
was started before this report was written; **see the orchestrator's own full-suite run
(`python manage.py test solsys_code --exclude-tag=ephemeris_segfault`) for the authoritative
final result** — per this agent's closeout discipline, close-out is not blocked on a
long-running suite run when per-fix verification has already passed.

Quality gates (`pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`)
ran automatically on every commit via the repository's pre-commit hooks and passed (one
ruff-format auto-reformat was applied and re-committed during the WR-08 commit; captured in
that commit's content, not left uncommitted).

---

_Fixed: 2026-09-19T15:00:36Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
