---
phase: quick-260705-l1v
verified: 2026-07-05T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Quick Task 260705-l1v: Fix Approval Queue Site-Visibility Gap — Verification Report

**Task Goal:** Fix approval-queue site-visibility gap: show `site_raw` in the pending
`CampaignRun` approval queue (currently blank for public submissions because
`site`/`site_needs_review` are unset until approval), and stop
`CampaignRunDecisionView.post()`'s automatic `resolve_site()` call from silently
fabricating a placeholder `Observatory` row for unresolvable free-text site names
(e.g. `'DCT'`, `'LCO-B'`) on approve.

**Verified:** 2026-07-05
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The pending approval queue shows the submitted site text (`site_raw`) for every pending run that has one, even though `site_needs_review` is `False` pre-approval. | VERIFIED | `solsys_code/campaign_tables.py:111-137` `render_site()` now falls back to `site_raw` whenever `site__short_name` is empty and `site_raw` is non-empty, independent of `site_needs_review`. Confirmed by `TestApprovalQueueSiteVisibility.test_pending_unresolved_site_shows_site_raw` (site=None, site_raw='DCT', site_needs_review=False → cell contains 'DCT') — test PASSES (see below). `ApprovalQueueTable(CampaignRunTable)` inherits `render_site`, so the approval queue view (`campaign_views.py` `pending_table`/`decided_table` built from `ApprovalQueueTable`) gets the fix. |
| 2 | Approving a run whose `site_raw` matches no real Observatory (tier 1) and no MPC obscode (tier 2) leaves `site=None` + `site_needs_review=True` and creates NO placeholder Observatory row. | VERIFIED | `campaign_views.py:302` calls `resolve_site(run.site_raw, create_placeholder=False)`; `campaign_utils.py:166-169` returns `(None, True)` and skips tier 3's `Observatory.objects.create(...)` when `create_placeholder` is False. Confirmed by `TestApprovalSiteResolution.test_approving_unresolvable_free_text_site_creates_no_placeholder_observatory` (MPC query patched to raise, approve POST → `run.site is None`, `run.site_needs_review is True`, `Observatory.objects.count() == 0`) — test PASSES. |
| 3 | The CSV import path (`resolve_site` default behavior) still creates a tier-3 placeholder Observatory as before — unchanged. | VERIFIED | `solsys_code/management/commands/import_campaign_csv.py:135` calls `resolve_site(site_raw)` positionally (no `create_placeholder` kwarg), so the new keyword-only parameter defaults to `True` and tier 3 is unaffected. Confirmed by `TestApprovalSiteResolution.test_resolve_site_default_still_creates_placeholder_observatory` (`resolve_site('DCT')` → placeholder Observatory created with `obscode='DCT'`, `Observatory.objects.count() == 1`) — test PASSES. |
| 4 | Approval still succeeds (`approval_status=APPROVED`) even when the site cannot be resolved — site failure never blocks approval (D-07). | VERIFIED | `campaign_views.py` approval-path logic unchanged around the `resolve_site` call — no early return/exception path added on unresolved site. Confirmed by same test as truth #2: `run.approval_status == APPROVED` after approving an unresolvable free-text site. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_tables.py` | `render_site` fallback change | VERIFIED | `render_site()` (lines 111-137) rewritten: falls back to `site_raw` whenever `site__short_name` empty and `site_raw` non-empty; two distinct presentations (muted-italic "pending review" vs. warning-triangle "failed") gated on `site_needs_review`. Uses `format_html`, not f-strings, for HTML. |
| `solsys_code/campaign_utils.py` | `resolve_site(create_placeholder=...)` opt-out | VERIFIED | `resolve_site(site_code_raw: str, *, create_placeholder: bool = True)` (line 85). Tier 1/2 logic unchanged; tier 3 gated by `if not create_placeholder: return None, True` (lines 166-169) before the `Observatory.objects.create(...)` block. Docstring updated with `Args:` entry for the new param. |
| `solsys_code/campaign_views.py` | Approval call site opt-out | VERIFIED | Line 302: `resolve_site(run.site_raw, create_placeholder=False)`. Comment above (lines 296-301) explains D-07/CAL-01 rationale. Surrounding approve/reject/calendar-projection logic untouched. |
| `solsys_code/tests/test_campaign_approval.py` | New DB-backed tests for both defects + CSV non-regression | VERIFIED | `TestApprovalQueueSiteVisibility` (3 tests) and `TestApprovalSiteResolution` (3 tests) added; all 6 pass (see spot-check below). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `CampaignRunTable.render_site` | rendering fallback | falls back to `site_raw` whenever `site__short_name` empty (not gated solely on `site_needs_review`) | WIRED | Confirmed by reading the diff and by the 3 `TestApprovalQueueSiteVisibility` tests exercising all three branches (pending/blank/failed). |
| `ApprovalQueueTable` | `CampaignRunTable.render_site` | class inheritance (`class ApprovalQueueTable(CampaignRunTable)`, `campaign_tables.py:146`) | WIRED | `ApprovalQueueTable` does not override `render_site`, so the approval-queue view (which builds `pending_table`/`decided_table` from `ApprovalQueueTable` in `campaign_views.py`) inherits the fixed fallback. |
| `CampaignRunDecisionView.post` | `resolve_site(..., create_placeholder=False)` | direct call at `campaign_views.py:302` | WIRED | Grep-confirmed keyword argument present at the approval call site; not present at the CSV-import call site (`import_campaign_csv.py:135`, positional-only), so `create_placeholder` defaults `True` there as intended. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| New tests (both defects) pass | `python manage.py test solsys_code.tests.test_campaign_approval -v2` | 23/23 tests OK | PASS |
| Full app suite stays green (no regression) | `python manage.py test solsys_code` | 332/332 tests OK | PASS |
| Lint/format clean on touched files | `ruff check` + `ruff format --check` on the 4 modified files | "All checks passed!" / "4 files already formatted" | PASS |

### Anti-Patterns Found

None. Grep for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER|placeholder|not yet implemented` across the 4 modified files surfaces only legitimate uses of the word "placeholder" as part of the intended `create_placeholder` parameter name/docstring and the pre-existing tier-3 "placeholder Observatory" terminology — not debt markers.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SUBMIT-03 | 260705-l1v | Staff can review/approve pending runs (visibility of site_raw is part of reviewability) | SATISFIED | `render_site` fallback fix + passing `TestApprovalQueueSiteVisibility` tests |
| D-07 | 260705-l1v (design decision from Phase 16, not a REQUIREMENTS.md ID) | Site resolution never blocks approval; free-text site captured pre-resolution | SATISFIED | `TestApprovalSiteResolution.test_approving_unresolvable_free_text_site_creates_no_placeholder_observatory` proves approval still succeeds with site unresolved |

No orphaned requirements: this quick task's `requirements: [SUBMIT-03, D-07]` frontmatter matches the two concerns actually addressed; D-07 is a documented design decision (Phase 16 CONTEXT.md) rather than a formal REQUIREMENTS.md ID, consistent with its usage elsewhere in the project's planning docs.

### Human Verification Required

None. All must-haves are directly testable via DB-backed unit/integration tests, and those tests exist, pass, and correctly exercise the asserted behaviors (not just presence/wiring — e.g. the placeholder-fabrication truth is proven both via the HTTP approval flow and a direct `resolve_site()` unit call with an `Observatory.objects.count()` assertion).

### Gaps Summary

No gaps. Both defects described in the task goal are fixed with proof:

1. **Visibility gap** — `render_site()` no longer requires `site_needs_review=True` to show `site_raw`; it now shows `site_raw` whenever the site is unresolved and `site_raw` is present, with a non-alarming "pending review" presentation distinct from the "resolution failed" presentation. Verified end-to-end through `ApprovalQueueTable`'s inheritance of `render_site` from `CampaignRunTable`.
2. **Placeholder fabrication** — `resolve_site()` gained a keyword-only `create_placeholder` parameter (default `True`, preserving the CSV-import path exactly), and the approval endpoint explicitly opts out (`create_placeholder=False`), so approving an unresolvable free-text site no longer creates a fake `Observatory` row while still allowing the run to be approved (D-07).

All 6 new tests plus the full 332-test `solsys_code` suite pass; `ruff check` and `ruff format --check` are clean on the touched files.

---

_Verified: 2026-07-05_
_Verifier: Claude (gsd-verifier)_
