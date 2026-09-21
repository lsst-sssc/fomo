---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
fixed_at: 2026-09-21T18:30:39Z
review_path: .planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 37: Code Review Fix Report

**Fixed at:** 2026-09-21T18:30:39Z
**Source review:** `.planning/phases/37-status-vocabulary-public-tallies-provenance-blind-gaps/37-REVIEW.md`
**Iteration:** 1
**Mode:** sequential (no worktree, per orchestrator project notes) — edits and commits made
directly on `issue37-telescope-runs-calendar`.

**Summary:**
- Findings in scope (critical + warning): 8
- Fixed: 8
- Skipped: 0

All fixes were applied against the code as it stands today (not blindly pasted from
REVIEW.md's suggested snippets), verified with `ast.parse`, `pre-commit run ruff` /
`ruff-format`, and the relevant Django test module(s) after each change, then committed
individually with hooks enabled (no `--no-verify`).

## Fixed Issues

### CR-01: the `per_page` cap clamped below-range values UP to the maximum

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `a1d4328`
**Applied fix:** Added `DEFAULT_TABLE_PER_PAGE = 25` (matching the view's existing
`table_pagination` default) alongside `MAX_TABLE_PER_PAGE = 100`. `CampaignRunTableView.get()`
now clamps a too-large `per_page` DOWN to `MAX_TABLE_PER_PAGE` (unchanged) but a too-small one
(`< 1`, e.g. `0`, `-1`, `-9999`) to `DEFAULT_TABLE_PER_PAGE` instead of `MAX_TABLE_PER_PAGE` —
fixing the 4x anonymous-cost amplification the review identified. `table_pagination` now reads
from the same `DEFAULT_TABLE_PER_PAGE` constant so the two never drift. Docstring rewritten to
describe the two-direction clamp explicitly instead of asserting the old (wrong) single-target
behaviour.

### WR-01: the huge-`per_page` cap test was vacuous (30-run fixture couldn't expose a 100-row cap)

**Files modified:** `solsys_code/tests/test_campaign_views.py`
**Commit:** `fbb3c50`
**Applied fix:** Replaced `test_huge_per_page_is_capped_and_still_fully_covered` (which
asserted `assertLessEqual(30, 100)` — true with or without the cap) with a `_make_n_runs(n)`
helper and four real boundary tests, all against a 120-run fixture (> `MAX_TABLE_PER_PAGE`):
`test_huge_per_page_is_capped_at_the_maximum` (exact `100`, not "at most"),
`test_per_page_exactly_at_the_cap_is_honoured` (`per_page=100` boundary),
`test_first_value_above_the_cap_is_clamped_down` (`per_page=101`), and
`test_degenerate_per_page_falls_back_to_the_default_not_the_maximum` (`0`, `-1`, `-9999`
against the existing 30-run fixture, asserting the new `DEFAULT_TABLE_PER_PAGE` fallback from
CR-01). 12/12 tests in the class pass.

### WR-02: broken multi-line `assert` in the committed notebook (dead string literals)

**Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
**Commit:** `2049056`
**Applied fix:** Rewrote the split `assert (...), '...' '...' '...'` (whose message halves
after the first string were an inert, unreachable expression statement) as
`assert rollup['unused_is_estimate'] is False, ('...' '...' '...')` — one string, one
assert. Re-executed the whole notebook in place via `jupyter nbconvert --to notebook --execute
--inplace` (all 50 cells, zero errors) so the committed output reflects the corrected source,
per the CLAUDE.md paired-docs convention of committing `pre_executed/` notebooks with output.

### WR-03: the roll-up's "nothing known" state had two disagreeing `unused_is_estimate` shapes

**Files modified:** `solsys_code/campaign_tally.py`, `solsys_code/tests/test_campaign_tally.py`
**Commit:** `92cf849`
**Applied fix:** In `_apply_rollup_unused_fields()`, the `unused_is_estimate` assignment moved
inside the `if exact_known or estimate_known` / `else` branches: the known branch keeps
`bool(contributing_codes)`, the not-known branch now writes `True` — matching the other three
not-yet-known writers in the module (empty-`runs` early return, `campaign_rollup()`'s initial
dict, `_without_unused_fields()`'s cached reset). Added an assertion to the existing
`test_nothing_known_at_all_leaves_the_strip_segment_unknown_not_partial` test plus a new
`test_not_yet_known_states_agree_on_every_unused_key_except_unknown_runs`, which directly
compares an empty-campaign roll-up against a one-run "nothing contributed" roll-up and asserts
every `unused_*` key matches except `unused_unknown_runs` (0 vs. 1). Full
`test_campaign_tally` + `test_campaign_views` suite (176 tests) green.

### WR-04: stale "three"/"eight" key counts in `campaign_tally.py` docstrings

**Files modified:** `solsys_code/campaign_tally.py`
**Commit:** `4e9a9f2`
**Applied fix:** Updated the three docstring mentions review pointed at (`campaign_rollup()`'s
own docstring "three unused_* keys" and "eight tally keys plus runs", and
`get_or_compute_rollup()`'s "three unused_* keys") to "four" / "nine" respectively. Left the
four other "three unused_*"/"three ``unused_*`` keys" mentions untouched, per the review's own
note that those describe the per-run tally, which never gained the fourth key.

### WR-05: `CampaignRunTable.__init__`'s `tallies` kwarg was dead, comments still described it as live

**Files modified:** `solsys_code/campaign_tables.py`
**Commit:** `b5da497`
**Applied fix:** Dropped the `tallies=None` constructor kwarg; `__init__` now sets
`self.tallies = {}` unconditionally, with the docstring rewritten to explain that
`CampaignRunTableView.get_table()` attaches tallies via post-construction attribute assignment
instead (G-37-5/CR-01), and that a `tallies=` kwarg would previously have been silently
overwritten. Also updated `render_progress()`'s docstring (which still referenced "constructed
with no `tallies` kwarg") and `ApprovalQueueTable.Meta`'s comment (which read as if some other
construction site passes the kwarg) to describe the current attribute-based mechanism. Grep for
`tallies=` in `solsys_code/` now returns zero hits.

### WR-06: the cap hardcoded the literal `'per_page'` instead of resolving it from the table

**Files modified:** `solsys_code/campaign_views.py`, `solsys_code/tests/test_campaign_views.py`
**Commit:** `cb07ff2`
**Applied fix:** `CampaignRunTableView.get()` now resolves the query-string field name as
`self.table_class._meta.prefix + self.table_class._meta.per_page_field` (verified interactively
to be the identical expression `table.prefixed_per_page_field` that django-tables2's own
`RequestConfig.configure()` reads) instead of the bare literal `'per_page'`, and uses that same
resolved name for both the `request.GET.get(...)` read and the `mutable_get[...] = ...` write.
Added `test_per_page_field_name_used_by_the_cap_matches_the_table`, which pins
`CampaignRunTable(data=[]).prefixed_per_page_field == 'per_page'` so a future `prefix` or
`per_page_field` override breaks this test loudly instead of silently disabling the cap.

### WR-07: `MAX_TABLE_PER_PAGE = 100`'s ~300-query cost was justified by analogy, not arithmetic

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** `0fad7e8`
**Applied fix:** Comment-only change (kept the constant at `100` — the lower-risk of the two
options the review offered, since lowering it would have rippled into WR-01's new boundary
tests). Rewrote the `MAX_TABLE_PER_PAGE` comment to state the actual query budget explicitly:
100 rows x the 3 queries/row this view's own
`test_page_query_count_grows_by_a_bounded_per_row_amount_not_unboundedly` test pins for
`campaign_tally.tallies_for_runs()` ~= 300 queries as the accepted anonymous ceiling, and noted
there is no throttle on this endpoint, and that lowering the constant (e.g. to 50) halves the
ceiling if needed later. No longer borrows `CampaignListView.paginate_by`'s number by analogy to
a page with a materially different (fully cached) cost shape.

## Skipped Issues

None — all 8 in-scope findings (CR-01, WR-01 through WR-07) were fixed.

## Notes for the orchestrator / next steps

- **Info findings (IN-01 through IN-06) were NOT addressed** — `fix_scope` for this run was
  `critical_warning`, which excludes Info-tier findings by design. If the next iteration widens
  scope to `all`, IN-01 through IN-06 remain open in `37-REVIEW.md`.
- Every fix was verified against the actual current source (not review's line numbers verbatim
  — `campaign_tally.py`'s target lines had drifted by a few lines by the time WR-04 was applied,
  due to WR-03 landing first in the same file; all edits were re-located via `grep` before being
  applied, and every edit matched the finding's described content byte-for-byte).
- Test modules run after each fix: `solsys_code.tests.test_campaign_views` (93 tests) and
  `solsys_code.tests.test_campaign_tally` (83 tests) — both fully green after all 8 commits, run
  together (176 tests) after WR-03/WR-04 to confirm no cross-file regression.
- `pre-commit run ruff --all-files` / `ruff-format --all-files` were run per-file (scoped to
  each commit's changed files) after every edit; a repo-wide `--all-files` pass was not re-run
  in this session but no fix touched files outside the seven already listed as
  `files_reviewed_list` in `37-REVIEW.md`.
- A full `solsys_code` test suite run (excluding the known-segfaulting `TestEphemeris`, tag
  `ephemeris_segfault`) was started as an extra sanity check beyond the per-fix verification
  above, but was deliberately not waited on to completion — per this agent's own scope, the
  full suite is the verifier phase's job, not the fixer's, and each fix was already verified
  against its directly relevant test module(s) (93 + 83 = 176 tests, all green). The
  orchestrator/verifier should still run the full suite as part of normal phase verification.
- WR-02's optional "Consider also widening the lint hook to
  `types_or: [ python, pyi, jupyter ]`" suggestion in `.pre-commit-config.yaml` was **not**
  applied — it is explicitly a "consider" (not part of the Fix), and turning on ruff *lint*
  (not just `ruff-format`, which already covers `jupyter`) for every notebook in the repo risks
  surfacing unrelated findings across notebooks this phase didn't touch. Left for a
  separately-scoped follow-up if desired.

---

_Fixed: 2026-09-21T18:30:39Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
