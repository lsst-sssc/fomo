---
phase: 30-v2-2-tech-debt-cleanup
reviewed: 2026-09-01T00:00:00Z
depth: deep
files_reviewed: 9
files_reviewed_list:
  - solsys_code/campaign_attribution.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_import_campaign_csv.py
  - docs/runbooks/telescope_runs_calendar.rst
  - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  - CLAUDE.md
  - pyproject.toml
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 30: Code Review Report

**Reviewed:** 2026-09-01T00:00:00Z
**Depth:** deep
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This phase makes four independent changes: (1) `campaign_attribution.py` gains a shared
`_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES` constant that excludes REJECTED `CampaignRun`s
at both `_eligible_runs_for_event()` and `_eligible_runs_for_record()`; (2)
`import_campaign_csv.py` widens the `telescope_class` re-import preservation guard from a
blanking-only check to a general "differs from stored" check; (3) `campaign_reconciler.py`
gets a docstring-only repair of dead-symbol references; (4) `pyproject.toml`/`CLAUDE.md`
pin `ruff==0.2.1` and route the documented lint command through `pre-commit`.

Items (1), (3) and (4) verify clean: the REJECTED exclusion is enforced identically at both
gates via one shared constant, `is_offered_candidate()` re-derives from the database through
those same gates (confirmed by `test_is_offered_candidate_refuses_a_rejected_run` and by
direct code trace) so a crafted POST naming a rejected run's pk cannot bypass it, the
standing `_eligible_runs_for_record` target-FK-equality prohibition is untouched, the
reconciler diff is prose-only (verified against `git show`), and the ruff/pre-commit pin is
internally consistent with `.pre-commit-config.yaml`.

Item (2), the widened `preserve_telescope_class` guard, is a genuine strict superset of the
old blanking-only guard (confirmed by trace and by the new
`TestReImportTelescopeClassPreservation` test class), but the widening exposes -- without
introducing -- a real interaction bug: `site_needs_review` is computed from the row's
*pre-preservation* `telescope_class` value rather than the value that actually lands in the
database once the preservation guard pops the field. This produces exactly the
"contradictory triple" (`site` resolved / `telescope_class` non-blank / `site_needs_review`
True) that a sibling guard (`preserve_site`) and its own dedicated test
(`test_telescope_class_not_derived_for_preserved_site`, "Case 9") were written specifically
to prevent -- but only in the mirror direction. See WR-01 below.

## Warnings

### WR-01: `site_needs_review` computed from the pre-preservation `telescope_class`, reproducing the "contradictory triple" bug the codebase explicitly guards against elsewhere

**File:** `solsys_code/management/commands/import_campaign_csv.py:277-281, 298-300, 309, 399-410`

**Issue:**

`needs_review` (line 309) is computed from the *local* `telescope_class` variable before the
`preserve_telescope_class` guard (lines 298-300, 399-410) has a chance to pop `telescope_class`
back out of `fields`:

```python
telescope_class = (
    derive_telescope_class(site_raw=site_raw, telescope_instrument=telescope_instrument)
    if site is None and not preserve_site
    else ''
)

preserve_telescope_class = (
    existing is not None and existing.telescope_class and telescope_class != existing.telescope_class
)
...
needs_review = site_resolution_failed and not telescope_class
```

When an existing row already carries a non-blank `telescope_class` (site-less, previously
derived or hand-corrected) and this row's freshly-resolved `site` differs from `None` while
`site_resolution_failed` is `True` -- which happens whenever `resolve_site()` falls through to
its tier-3 placeholder (a Site Code that is 1-4 chars but matches no local `Observatory` and no
MPC obscode) -- the *local* `telescope_class` is computed as `''` (because `site is not None`
now), so:

* `preserve_telescope_class` fires (`'' != '1m0'`), `telescope_class` is popped from `fields`,
  and the database correctly **retains** the old non-blank value.
* `site` is **not** popped (`preserve_site` requires `existing.site_id is not None`, which is
  false here -- this row was previously site-less specifically *because* it carried a
  `telescope_class`), so `fields['site']` writes the newly-created tier-3 placeholder
  `Observatory` onto the row.
* `needs_review = site_resolution_failed(True) and not telescope_class(local, '' -> True)` ==
  `True`, and since `'site_needs_review'` was never popped from `fields`, this `True` value is
  written to the database.

The resulting row is `site=<placeholder Observatory>`, `telescope_class='1m0'` (correctly
preserved), `site_needs_review=True` -- the exact triple `test_telescope_class_not_derived_for_preserved_site`'s
docstring calls "the contradictory triple ... produces" and that the `site is None and not
preserve_site` guard on the `telescope_class` computation was written specifically to prevent
*for the mirror direction* (site preserved, so telescope_class must not be freshly derived).
No equivalent protection exists for this direction (telescope_class preserved, so
`site_needs_review` must not be freshly computed from the *local*, about-to-be-discarded
`telescope_class`).

Per `models.py`'s own D-06 comment: "A class-carrying run (telescope_class non-blank) is never
flagged, because the class already answers 'why is there no site'." This row violates that
invariant. It also inflates the command's own `site_needs_review:` summary counter (line 420's
`resulting_needs_review` only accounts for `'site_needs_review' in fields`, not for
`telescope_class` having been separately preserved), so the printed summary line itself is
wrong, not just the database row. The bug is reachable via the narrower "still unresolved"
path too (not just the tier-3-placeholder path): any re-import where the CSV's
`Telescope / Instrument` text this time derives a *different* non-blank-or-blank class than
the value already stored produces the same local-vs-final mismatch whenever
`site_resolution_failed` is simultaneously `True`.

This is not introduced by this phase's widening (the pre-existing blanking-only guard --
`if existing.telescope_class and not telescope_class:` -- could already trigger the same
local/final mismatch), but the widened guard is exactly the code this phase's task asked to be
checked for correctness, no test in `test_import_campaign_csv.py` covers this interaction (the
closest test, `test_telescope_class_never_blanked_by_reimport`, uses a *genuine* site
resolution where `site_resolution_failed` is `False`, so `needs_review` comes out `False`
either way and the bug never surfaces there), and the runbook's "Re-import gotcha" note does
not mention this edge case either.

**Fix:** compute `needs_review` from the value that will actually end up on the row, the same
way `resulting_needs_review` (line 420) already does for `site_needs_review`:

```python
resulting_telescope_class = existing.telescope_class if preserve_telescope_class else telescope_class
needs_review = site_resolution_failed and not resulting_telescope_class
```

and add a regression test mirroring `test_telescope_class_not_derived_for_preserved_site`
("Case 9") but for this direction: an existing site-less, class-carrying row, re-imported with
a Site Code cell that fails to resolve to a genuine `Observatory` (either the length-guard path
or a fresh tier-3 placeholder), asserting `site_needs_review` stays `False` and, in the
placeholder case, that `site` does not silently pick up a newly-fabricated placeholder
`Observatory` alongside a preserved `telescope_class` (the model's own class-wide invariant is
that such a run should stay `site=None` permanently, per the `models.py:242-254` docstring's
"a class-wide campaign legitimately keeps site=None forever").

## Info

### IN-01: `is_offered_candidate('record', ...)` REJECTED-exclusion has no direct test

**File:** `solsys_code/tests/test_campaign_attribution.py:393-401`

**Issue:** `TestApprovalStatusGate.test_is_offered_candidate_refuses_a_rejected_run` only
exercises `is_offered_candidate('event', ...)`. Code trace confirms the `'record'` branch
(`campaign_attribution.py:825-832`) routes through `candidates_for_record()` ->
`_eligible_runs_for_record()`, which applies the identical
`_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES` exclusion, so there is no evidence of an actual
functional gap -- but this phase's own stated concern was specifically "cannot be bypassed via
a crafted POST through `is_offered_candidate()`," and that claim is currently unverified for
the record path by any test. `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`'s
new rejected-run demo cell (cell 22) also only exercises the event path.

**Fix:** add a `test_is_offered_candidate_refuses_a_rejected_run_for_record` mirroring the
existing event-path test, asserting `is_offered_candidate('record', record.pk,
rejected_run.pk)` returns `None` while the identical `approved_run` returns a real candidate.

---

_Reviewed: 2026-09-01T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
