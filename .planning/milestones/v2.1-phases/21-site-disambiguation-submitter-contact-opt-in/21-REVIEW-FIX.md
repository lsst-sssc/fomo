---
phase: 21-site-disambiguation-submitter-contact-opt-in
fixed_at: 2026-07-11T15:47:47Z
review_path: .planning/phases/21-site-disambiguation-submitter-contact-opt-in/21-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 21: Code Review Fix Report

**Fixed at:** 2026-07-11T15:47:47Z
**Source review:** .planning/phases/21-site-disambiguation-submitter-contact-opt-in/21-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (2 critical, 2 warning; IN-01 excluded by `fix_scope: critical_warning`)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### CR-01: Fuzzy-matched site candidates (name/short_name/old_names) cannot actually resolve on approve

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** 19fc96e
**Applied fix:** In `CampaignRunDecisionView.post()`, the submitted `site_selection` text is
now looked up in `build_site_candidates()`'s `{display_string: obscode}` pool before being
passed to `resolve_site()`. This mirrors the same candidate pool `ApprovalQueueTable.render_site()`
uses to build the datalist, so an exact match on a name/short_name/old_names candidate now
maps back to its real obscode; a value that was never a candidate (including a genuinely
typed obscode) passes through unchanged via `dict.get(selection, selection)`.

### CR-02: "Create new Observatory" `?next=`/`?obscode=` round-trip is dropped by the real form

**Files modified:** `src/templates/solsys_code_observatory/observatory_create.html`
**Commit:** 3b21750
**Applied fix:** Added a hidden `<input type="hidden" name="next" value="{{ request.GET.next }}">`
field to the create-observatory form, rendered only when `request.GET.next` is present. Since
`get_success_url()` already reads `self.request.POST.get('next')` as a fallback, the real
browser-submitted POST now carries the query param through, restoring the round-trip back to
the approval queue. The `next` value is still validated by `url_has_allowed_host_and_scheme()`
in `get_success_url()`, so the open-redirect protection is preserved.

### WR-01: `build_site_candidates()` can still raise despite its "never raises" contract

**Files modified:** `solsys_code/campaign_utils.py`
**Commit:** a752fba
**Applied fix:** Broadened the `except` clause in `build_site_candidates()` to also catch
`AttributeError`, which is what `_flatten_mpc_candidates()`'s `.items()`/`.get()` calls raise
if the MPC bulk endpoint ever returns a non-dict shape (list, `None`, etc.). The bulk-fetch
failure now degrades gracefully to the local-only candidate pool in that case too, instead of
propagating uncaught into `ApprovalQueueView.get_context_data()`.

### WR-02: `get_initial()` pre-fills a `max_length=3` field with arbitrary-length `site_raw` text

**Files modified:** `solsys_code/solsys_code_observatory/views.py`
**Commit:** b5a039c
**Applied fix:** `CreateObservatory.get_initial()` now only pre-fills `obscode` from
`?obscode=` when the raw value is exactly 3 characters (matching
`CreateObservatoryForm.obscode`'s `min_length=max_length=3`), leaving the field blank
otherwise (e.g. for a full site name like `'Siding Spring Observatory'`) so staff aren't
shown a field that already looks "filled in" but is guaranteed invalid.

## Skipped Issues

None — all in-scope findings were fixed.

---

_Fixed: 2026-07-11T15:47:47Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
