---
phase: 34-the-observation-projector-trigger
fixed_at: 2026-09-11T16:10:00Z
review_path: .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
iteration: 3
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 34: Code Review Fix Report

**Fixed at:** 2026-09-11T16:10:00Z
**Source review:** .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
**Iteration:** 3 (final -- no re-review follows this pass)

**Prior pass.** This is the third automated fix pass over Phase 34's code review. The
first pass (iteration 1) closed 5 findings from `34-REVIEW.md` iteration 2's re-review
(CR-01, WR-01, WR-02, WR-03, WR-04), committed as `c68ec3e`/`a8b6263`/`65fa5c7`/`9913f58`.
The second pass (iteration 2) closed 4 findings from iteration 3's re-review, reusing the
same IDs under different content (CR-01, WR-01, WR-02, WR-03), committed as
`3dfa45c`/`00c3c0b`/`a432094`/`427d99c`. Both passes' full reports are preserved in git
history; this document replaces the iteration-2 report on disk with a fresh report for the
current (fourth review / third fix) pass only, carrying forward both prior recaps below for
traceability.

**Iteration 1 recap (for traceability).** Commits `c68ec3e`, `a8b6263`, `65fa5c7`, `9913f58`
closed: WR-02+WR-03 (the savepoint moved into `project_record()`, restructured so the
`except` sits outside the `with transaction.atomic():` block, `c68ec3e`); CR-01 (the real
write branch of `project_queryset()` reads `project_record()`'s own return value instead of
the preview's guess, `a8b6263`); WR-04 (`--proposal` fails closed on an all-empty-segment
value instead of silently sweeping everything, `65fa5c7`); WR-01 (the series-decoration tag
gates its un-attributed case on the viewer being authenticated, `9913f58`).

**Iteration 2 recap (for traceability).** The iteration-3 review found the iteration-1 pass
left the `--dry-run` half of the CR-01 fix disagreeing with the real run (a regression the
fix itself introduced), the WR-01 fix's gate still leaking the group name for an
attributed-and-approved event, plus two new findings (WR-02: the `'ogg'` obscode bridge;
WR-03: the m2m receiver's unreachable, undocumented `try`). Commits `3dfa45c` (CR-01: dry
run now detects the one write failure it can see without writing -- a duplicate calendar-
event url -- and counts it `unprojectable`, matching the real run; docstring and runbook
narrowed to an honest lower-bound claim), `00c3c0b` (WR-01: the anonymous-viewer gate made
unconditional and first, so it can no longer be bypassed by an approved-and-attributed
run), `a432094` (WR-02: removed `'ogg'`/`'sor'` from the site-keyed `LCO_SITE_CODE_TO_OBSCODE`
table, which was wrong for `'ogg'`'s two telescopes, and added a new LABEL-keyed
`OBSERVED_TELESCOPE_OBSCODES` table instead), `427d99c` (WR-03: the m2m receiver's `try`
given the same "second, outer layer of defence" documentation as its `post_save` sibling,
and made to read and log `project_record()`'s own return value).

**This pass (iteration 3) covers only the current `34-REVIEW.md`'s three findings** (WR-01,
WR-02, WR-03 -- 0 critical this round; note these IDs are reused from the review's own
per-pass numbering and do not correspond 1:1 to the iteration-1 or iteration-2 findings of
the same name). It does not re-litigate or duplicate the recaps above.

**Summary:**
- Findings in scope: 3 (WR-01, WR-02, WR-03 -- the `critical_warning` fix scope; IN-01
  through IN-09 were left untouched per scope, as instructed)
- Fixed: 3
- Skipped: 0

**Verification environment.** Every fix below was edited, linted, and test-run inside an
isolated git worktree this run created
(`.claude/worktrees/rf-34-1285076-1789141533`, branch `gsd-reviewfix/34-1285076`), on top of
`issue37-telescope-runs-calendar`; the worktree's cleanup (fast-forward, `git worktree
remove`, temp-branch delete, sentinel removal) runs after this report is written, per this
agent's own transactional cleanup protocol. `pre-commit run ruff --files` and
`pre-commit run ruff-format --files` are clean on every Python file this pass touched
(`solsys_code/management/commands/project_observation_calendar.py`,
`solsys_code/campaign_attribution.py`, `solsys_code/calendar_utils.py`,
`solsys_code/tests/test_campaign_attribution.py`). `python manage.py test` over the four
modules this task specified (`solsys_code.tests.test_campaign_attribution`,
`test_campaign_attribution_views`, `test_project_observation_calendar`,
`test_observation_projector`) runs **176 tests green in 7.1 s**, with the narrower
`test_campaign_attribution` + `test_calendar_utils` pair also re-run standalone (84 tests
green) immediately after the WR-02 commit. The `src/fomo/_version.py` build artifact needed
for `manage.py` to import at all (gitignored, `setuptools_scm`-generated) was copied from
the main checkout into the worktree purely so tests could run -- not a source change, not
committed.

## Fixed Issues

### WR-03: `project_observation_calendar.py`'s module docstring still claimed "one documented exception" to dry-run/real-run agreement

**Files modified:** `solsys_code/management/commands/project_observation_calendar.py`
**Commit:** `ff80c87`
**Applied fix:** the iteration-2 CR-01 fix (`3dfa45c`) rewrote the "one documented exception"
sentence in `project_queryset()`'s own docstring (`observation_projector.py:410-433`) and in
the runbook (`telescope_runs_calendar.rst:183-198`) to "two exceptions", but missed this
command module's own copy of the identical sentence -- the file an operator reading
`--help`-adjacent source lands on first. Reworded the module docstring to state both
exceptions explicitly: the one-time observed-site lookup a dry run never performs, and the
fact that a dry run's `unprojectable` count is only a lower bound on the real sweep's
(it can only detect a write failure it can see without writing). Also dropped the bare
`(WR-02)` review-ID citation the sentence carried, which resolved only against `.planning/`
and is not shipped (IN-04's own concern, addressed here incidentally since this docstring
was already being touched).

### WR-02: the WR-02 fix (iteration 2) made `_extract_lco_site_code()`'s observed-label branch unreachable for every in-repo caller, and left the D-07 regression test's docstring crediting it as the mechanism under test

**Files modified:** `solsys_code/campaign_attribution.py`, `solsys_code/calendar_utils.py`,
`solsys_code/tests/test_campaign_attribution.py`
**Commit:** `36c7eae`
**Applied fix:** traced the three observed labels (`'FTN'`, `'FTS'`, `'SOAR'`) through
`_extract_lco_site_code()`'s `OBSERVED_TELESCOPE_SITE_CODES.get()` consultation and
confirmed the review's finding: `telescope_match_score()` only ever calls
`_extract_lco_site_code()` once its own step-1 condition
(`observed_obscode is not None and run.site_id is not None`) has already failed, which
happens for an observed label only when `run.site_id is None` -- and `run.site_id is not
None` is also required by the very next check (`:373` pre-fix) that would have used this
consultation's result. So the consultation's return value could never survive to change an
outcome, for any of the three labels, in any state. Removed the consultation from
`_extract_lco_site_code()`, leaving the function's genuine remaining job (the
SITECODE-CLASS site-code split, still needed for e.g. `'COJ-1m0'`/`'COJ-2m0'`), and removed
the now-unused `OBSERVED_TELESCOPE_SITE_CODES` import from `campaign_attribution.py`.
Confirmed via `find_referencing_symbols`-equivalent grep that `OBSERVED_TELESCOPE_SITE_CODES`
itself (defined in `calendar_utils.py`) is still genuinely exercised elsewhere -- its own
regression test in `test_calendar_utils.py` (`TestTelescopeLabelResolutionHelpers`) verifies
the table's own content independent of `campaign_attribution.py` -- so left the table in
place, but corrected its comment in `calendar_utils.py`, which claimed
`campaign_attribution.py`'s telescope-match signal still needed it (it no longer does).
Updated `_extract_lco_site_code()`'s own docstring to describe its narrower remaining
behaviour and point a caller genuinely needing one of the three labels' classical site code
at `calendar_utils.OBSERVED_TELESCOPE_SITE_CODES` directly instead. Fixed
`test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level`'s docstring,
which credited the removed consultation as "exactly the asymmetric regression this bridge
... closes" -- it now names the actual mechanism, `telescope_match_score()`'s own
LABEL-keyed `OBSERVED_TELESCOPE_OBSCODES` step, and states plainly that
`_extract_lco_site_code()` plays no part in the outcome (it returns `None` for all three
labels now). No test assertions changed -- the existing test continues to pass because it
was already exercising `OBSERVED_TELESCOPE_OBSCODES`, as the review found.

**Notebook impact (checked per this pass's instructions, not silently ignored).**
`docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`'s D-07 cell (cell 22) imports
`_extract_lco_site_code` directly and asserts its return equals
`OBSERVED_TELESCOPE_SITE_CODES`'s site code for each of `'FTN'`/`'FTS'`/`'SOAR'`. That
assert would now fail (`AssertionError`, not a silent difference) on its first iteration if
the cell is ever re-executed, since the function now returns `None` for all three labels
instead of their site codes -- this is a real, new consequence of this commit, and was not
worked around by keeping the dead branch alive or promoting it to a public alias (both
options this pass's own instructions offered): the dead-code removal was judged the more
honest fix, since a "thin documented alias" would just re-introduce the same
no-live-consumer smell WR-02 exists to close. This breakage is folded into WR-01's own
notebook caveat below (see WR-01) and into the outstanding full-notebook-re-execution
follow-up both findings now point at -- not fixed by re-executing or hand-editing the
notebook's code cell, per this task's explicit constraint against doing either.

### WR-01: `campaign_lifecycle_demo.ipynb` cell 22's committed output is now unproducible from its own source -- the prior caveat undercounted the drift and this pass's own WR-02 fix made it worse

**Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (markdown
cell 21 only -- prose, not re-executed)
**Commit:** `c7f78ee`
**Applied fix:** the caveat paragraph `a432094` (iteration 2) added to cell 21 named two
stale print lines and asserted the `telescope_match_score()` call below the loop was
"unaffected" by the drift it described -- but that call's own evidence string was itself one
of the stale lines (a reviewer-caught omission), and this pass's own WR-02 commit above
changes the picture further: the loop's site-code prints for `'FTN'`/`'SOAR'` are now *also*
wrong (previously they still showed `'ogg'`/`'sor'`; now `_extract_lco_site_code()` returns
`None` for all three labels), and the loop's own `assert resolved_site_code ==
expected_site_code` now raises `AssertionError` on its very first iteration if re-executed
-- so the cell would not just print different values, it would crash before reaching
`FTS`/`SOAR` or the `telescope_match_score()` call at all. Rewrote the caveat to: name all
four stale/would-not-print lines instead of two; explain precisely why the loop now raises
instead of merely differing; quote what `telescope_match_score()`'s evidence string actually
produces for `'FTN'` today (`orphan observed telescope 'FTN' resolves to obscode F65,
matching the run's site obscode F65`) against what the committed, stale output shows
(`orphan LCO site code 'ogg' resolves to obscode F65, ...`); and state explicitly that the
resolved obscode (`F65`) and match result (`TELESCOPE_MATCH_SITE`) are unchanged -- only the
wording of how they were derived differs. Also lightly amended the cell's intro paragraph
(two lines), which flatly asserted `_extract_lco_site_code()` "now checks
`calendar_utils.OBSERVED_TELESCOPE_SITE_CODES` first, closing that gap" -- no longer true
after this same pass's WR-02 commit -- to say it "originally closed that gap" and point at
the caveat for what has since changed, rather than leave a now-false claim standing
unqualified next to a caveat that only addressed a different paragraph.

Per this pass's explicit instructions, the notebook was **not** re-executed and its code
cell/output were **not** hand-edited -- only the markdown prose in cell 21 changed. The
caveat's outstanding follow-up, stated with the exact command:

```
jupyter nbconvert --to notebook --execute --inplace \
  docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
```

should be paired with rewriting the D-07 loop itself before that re-execution, since running
it as currently written will raise `AssertionError` rather than produce a clean new output
-- the caveat names the rewrite (demonstrate the bridge through
`OBSERVED_TELESCOPE_OBSCODES` directly, dropping the `_extract_lco_site_code` import, which
also closes IN-05's private-helper-import anti-pattern) as part of the same follow-up. JSON
validity was verified with `python -c "import json;json.load(open(...))"` after every edit
to this file; the diff is scoped to the one markdown cell (16 insertions, 3 deletions) with
no other cell touched.

## Remaining (out of scope this pass, unchanged from `34-REVIEW.md`)

These are Info-severity findings from the current review. `fix_scope` for this pass was
`critical_warning`, so none of these were touched. Listed here for visibility only:

- **IN-01** -- `write_event_meta()` silently reverts any admin-set `is_verified=False` on
  every projection; not documented, not made read-only on the admin inline.
- **IN-02** -- the LCO/SOAR shared-URL test pins the url but not which record ends up owning
  the companion `CalendarEventMeta` row (last-writer-wins, untested/undocumented as such).
- **IN-03** -- `event_form.html:109`'s comment still references the retired
  `sync_lco_observation_calendar` command.
- **IN-04** -- review-finding IDs (`WR-*`, `CR-*`, etc.) are embedded throughout shipped
  source, templates and tests; they resolve only against `.planning/`, which is not shipped.
  (Note: this pass's own WR-03 fix happened to remove one such citation as a side effect of
  the docstring rewrite it was already making; the rest are unchanged.)
- **IN-05** -- the demo notebook imports a private cross-module helper
  (`_extract_lco_site_code`) the codebase's own docs call out as an anti-pattern. This pass's
  WR-01/WR-02 fixes make this import doubly stale (it now returns `None` for all three
  labels the cell iterates over) but did not remove it, per the constraint against
  hand-editing the notebook's code cell -- the same rewrite recommended in WR-01's caveat
  closes this finding too, as the outstanding follow-up.
- **IN-06** -- `CalendarEvent.url` is the one externally-sourced 200-character column the
  truncation-warning comment does not mention as deliberately unbounded.
- **IN-07** -- `[F]` is the shared legend label for both `FAILURE_LIMIT_REACHED` and
  `NOT_ATTEMPTED`, reading as "Failed" for both on the public calendar.
- **IN-08** -- the dry-run `unprojectable` path (added by the iteration-2 CR-01 fix) is the
  only one of four `unprojectable`-counting sites in `project_queryset()` that logs nothing.
- **IN-09** -- the runbook's ring-color description groups an Inconsistent record entry with
  the Queued ring; the code actually gives it the terminal ring (the tag's own docstring
  already states this correctly; only the runbook prose is wrong).

## Notes and Follow-ups

- **Commit grouping.** Each of the three findings landed as its own dedicated commit
  (`ff80c87` WR-03, `36c7eae` WR-02, `c7f78ee` WR-01), applied in that order (smallest/
  most-isolated first) so that WR-01's notebook caveat could accurately describe the
  post-WR-02 state of the code rather than needing a fourth "fix the caveat again" commit.
- **This is the final iteration.** Per this pass's own instructions, no further automated
  re-review follows. The Info findings above and WR-01's own outstanding
  rewrite-then-re-execute follow-up are the complete list of what remains open on Phase 34
  after this pass.
- All three commits (`ff80c87`, `36c7eae`, `c7f78ee`) were individually verified against
  the task's specified test modules; the final combined run (all four modules together)
  reports 176 tests green in 7.1 s.

---

_Fixed: 2026-09-11T16:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 3_
