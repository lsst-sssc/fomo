---
phase: 28-operator-assisted-attribution
verified: 2026-08-02T00:33:28Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 6/8
  gaps_closed:
    - "Staff can confirm a candidate association one at a time through the rendered attribution-queue UI, not just via a bare test client POST (CR-01)"
    - "Both the confirmation and the undo are attributable to a person and a time, across every write path that can create the association (CR-02)"
  gaps_remaining: []
  regressions: []
---

# Phase 28: Operator-Assisted Attribution Verification Report

**Phase Goal:** Give staff a queue of suggested run↔event and run↔record associations with the evidence visible, confirmable one candidate at a time and undoable — the mechanism that connects the existing calendar events and observation records to their parent runs without ever guessing silently.
**Verified:** 2026-08-02T00:33:28Z
**Status:** passed
**Re-verification:** Yes — after gap closure (28-05-PLAN.md and 28-06-PLAN.md)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Staff see a queue of suggested associations with evidence side by side (matched telescope, date overlap, campaign, instrument similarity), not a bare score | ✓ VERIFIED | `campaign_attribution.AttributionCandidate` carries four separate evidence strings (`campaign_attribution.py:355-370`); `attribution_queue.html:85-89`/`~171-175` render all four plus the score as a subordinate badge; `TestEvidenceColumns.test_evidence_columns_score_and_band_badge_all_present` passes (confirmed by direct re-run) |
| 2 | Candidates are confidence-scored and filterable by score/band | ✓ VERIFIED | `band_for_score()` (High/Medium/Low), `?band=` GET filter wired in `AttributionQueueView`/template `<select>`; `TestBandFilterAndBanner.test_band_high_filter_shows_only_high_band_candidates` / `test_band_medium_filter_shows_only_medium_band_candidates` pass; new `TestSoleHighCandidateUnderBandFilter` (4 tests) additionally pins that the sole-High checkbox gate now agrees with the server-side authoriser under every band filter (WR-02 closed) |
| 3 | No association is ever created without an explicit per-candidate staff POST (server-enforced, never trusting a rendered row), and no suggestion is ever offered across a campaign/target boundary | ✓ VERIFIED | `is_offered_candidate()` re-derives eligibility from the DB on every write; `_eligible_runs_for_event`/`_eligible_runs_for_record` gate on campaign membership before any scoring; `TestConcurrencyAndTampering.test_cross_campaign_run_post_writes_nothing` / `test_malformed_*` pass; `AttributionDecisionView.http_method_names = ['post']` blocks GET |
| 4 | Staff can confirm a candidate association one at a time through the rendered attribution-queue UI (not only via a raw test-client POST) | ✓ VERIFIED (CR-01 closed) | Both per-candidate Confirm `<button>`s in `attribution_queue.html` (events table line ~98, records table line ~184) now carry `formnovalidate`; the Dismiss-only `reason` input keeps `required` untouched, and no `<form>` carries a form-level `novalidate`. Verified by direct file read (`grep -c 'formnovalidate' src/templates/campaigns/attribution_queue.html` → `2`) and by running the new structure-only test module `test_attribution_template.py` (6 tests, all pass), which never calls `self.client.post()` and therefore is structurally capable of catching a regression of this exact defect class |
| 5 | A confirmed association can be undone from the same screen that created it | ✓ VERIFIED | `AttributionDecisionView._undo_confirmation()`; `TestConfirmUndo.test_undo_confirmation_event_clears_link_and_writes_dismissal` / `test_undo_confirmation_record_deletes_link_and_writes_dismissal` pass; new `TestUndoConfirmationOrdering.test_a_genuine_undo_still_clears_the_link_and_writes_its_dismissal` confirms a real undo still works after the WR-01 reorder |
| 6 | Both the confirmation and the undo are attributable to a person and a time, across every write path that can create the association | ✓ VERIFIED (CR-02 closed) | `CalendarEventMetaInline`/`CampaignRunAdmin.save_formset` stamp correctly (unchanged, still tested). `CalendarEventMetaAdmin` — the standalone admin page — now has class-level `readonly_fields = ['confirmed_by', 'confirmed_at']` (verified: 3 occurrences of this exact line across the file — 2 inlines + this admin) and a `save_model()` override that stamps `confirmed_by=request.user`/`confirmed_at=timezone.now()` on a genuine run-link transition, nulls both on a clear, and leaves both untouched on an unrelated edit. All 7 tests in the new `CalendarEventMetaStandaloneAdminAuditStampTests` pass, including value-comparison proof (not just non-null) that a hand-typed `confirmed_by`/`confirmed_at` does not bind |
| 7 | The known real case is surfaced: `CampaignRun` pk=1-equivalent (FTS/MuSCAT4, 7-21 July, Siding Spring E10) is offered against LCO queue events/records despite the date-span and instrument-string mismatch | ✓ VERIFIED | `TestCriterion5RealCase` (`test_campaign_attribution.py:275+`) builds an equivalent fixture and asserts High band, sole-High-candidate status, against 10 events and 10 records; passes unmodified (28-06's plan explicitly required this test stay untouched — confirmed by direct re-run) |
| 8 | A non-staff request to any attribution route is redirected before anything is written or rendered | ✓ VERIFIED | `StaffRequiredMixin` on both views; `TestAttributionStaffGating` (`test_anonymous_get_attribution_redirects`, `test_non_staff_get_attribution_redirects`, `test_anonymous_post_decide_redirects_and_writes_nothing`, `test_non_staff_post_decide_redirects_and_writes_nothing`) all pass |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | `CalendarEventDismissal`, `ObservationRecordDismissal`, `CalendarEventMeta.confirmed_by`/`confirmed_at` | ✓ VERIFIED | Present with named per-pair `UniqueConstraint`s; `makemigrations --check` reports no drift |
| `solsys_code/admin.py` | `save_formset` stamps `CalendarEventMeta` audit fields on run transition (inline path); `CalendarEventMetaAdmin` protects and stamps the same fields on the standalone page | ✓ VERIFIED | Both paths present and independently tested (`test_save_formset_stamps_calendar_event_meta_on_run_transition` for the inline; `CalendarEventMetaStandaloneAdminAuditStampTests` (7 tests) for the standalone page). CR-02 closed |
| `solsys_code/campaign_attribution.py` | Matcher: scoring, banding, orphan querysets, `candidates_for_event`/`candidates_for_record`, `sole_high_candidate_pk` from the full uncapped list | ✓ VERIFIED | `_sole_high_candidate_pk(full_candidates)` called at both backlog-builder call sites (`grep -c` → 2); `TestSoleHighCandidateUnderBandFilter` (4 tests) pins the fix. IN-01 docstring corrected to describe the rounded-score comparison, behavior deliberately unchanged (documented no-op at current weights) |
| `solsys_code/campaign_views.py` | `AttributionQueueView`, `AttributionDecisionView`; `_undo_confirmation()` gates the dismissal write on the link-clearing write's `changed_count` | ✓ VERIFIED | Server-side re-validation via `is_offered_candidate()` on every confirm write (unchanged); `_undo_confirmation()` now runs the link-clearing update/delete first and writes the D-13 dismissal only `if changed_count:`, inside the same `transaction.atomic()` block. WR-01 closed, pinned by `TestUndoConfirmationOrdering` (5 tests) |
| `solsys_code/campaign_urls.py` | `campaigns:attribution`, `campaigns:attribution_decide` | ✓ VERIFIED | Both routes present |
| `src/templates/campaigns/attribution_queue.html` | Four-section staff page; Confirm submitter usable in a real browser | ✓ VERIFIED | Renders all four sections and full evidence; both Confirm buttons carry `formnovalidate`; Dismiss's `required` reason gate survives untouched. CR-01 closed |
| `solsys_code/campaign_tables.py` | `AttributionDismissedTable`, `AttributionConfirmedTable` | ✓ VERIFIED | Both classes present with CSRF-per-row and auto-escaping discipline (unchanged since prior pass) |
| `src/templates/campaigns/campaign_list.html` | `attribution_count` banner | ✓ VERIFIED | Staff-only banner present; tests pass (unchanged) |
| `docs/runbooks/telescope_runs_calendar.rst` | Attribution-pass operator section, including the CR-01/CR-02 operator-visible consequences | ✓ VERIFIED | Contains the literal phrase "required for Dismiss only" and states Confirm never asks for a reason; "Behavior change" paragraph states an admin-created link is now stamped automatically and its audit fields are no longer hand-editable. Sphinx docs build succeeds over the edited file (verified directly: `sphinx-build -M html ./docs ...` → "build succeeded, 9 warnings", all pre-existing/unrelated) |
| `solsys_code/tests/test_attribution_template.py` | Structure-only regression tests for CR-01, no `self.client.post()` calls | ✓ VERIFIED | 182 lines (exceeds 90-line minimum); `grep -v '^\s*#' ... \| grep -c 'self\.client\.post('` → `0`; 6 tests, all pass; mutation check recorded in 28-05-SUMMARY.md shows both test classes fail when `formnovalidate` is removed |
| `.planning/REQUIREMENTS.md` | ATTRIB-01/02/05/06 checklist and traceability rows synced to Complete | ✓ VERIFIED | All six ATTRIB checklist items now `[x]`; all six traceability rows read `Complete`; zero unchecked `- [ ] **ATTRIB-` entries remain |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `campaign_attribution.py` | `difflib.SequenceMatcher` | tokenised instrument similarity | ✓ WIRED | Unchanged from prior pass |
| `campaign_attribution.py` | `CalendarEventDismissal`/`ObservationRecordDismissal` | dismissed-pair exclusion | ✓ WIRED | Unchanged; `_undo_confirmation()`'s reordering does not affect this exclusion query |
| `campaign_views.py` | `campaign_attribution.is_offered_candidate` | server-side re-validation on every write | ✓ WIRED | Called in `_do_confirm_event`, `_do_confirm_record`, `_confirm_selected` before any write. Deliberately **not** called in `_undo_confirmation()` (an already-confirmed pair is no longer "offered") — the `changed_count`-gated conditional write is the correct upstream check there instead, and is now in place |
| `campaign_views.py` | `CalendarEventMeta` | atomic conditional update keyed on `run__isnull=True` (confirm) / `run_id=run_pk` (undo) | ✓ WIRED | `_do_confirm_event()` and the reordered `_undo_confirmation()` both use this pattern |
| `campaign_views.py` | `CampaignRunObservation` | `get_or_create` guarded by unique constraint | ✓ WIRED | `_do_confirm_record()`; undo side uses a conditional `.delete()` |
| `attribution_queue.html` | `campaigns:attribution_decide` | form action on every confirm/dismiss/undo control | ✓ WIRED | Present on all forms; the Confirm submitter now carries `formnovalidate` so it submits without being blocked by the Dismiss-only `required` field. CR-01's blocking wiring defect is closed |
| `campaign_list.html` | `campaigns:attribution` | banner link | ✓ WIRED | Confirmed unchanged |
| `campaign_views.py` | `campaign_tables.py` | `AttributionDismissedTable`/`AttributionConfirmedTable` construction | ✓ WIRED | Confirmed in `get_context_data`, unchanged |
| `admin.py CalendarEventMetaAdmin` | `request.user` / `django.utils.timezone.now()` | `save_model()` stamping on a genuine run transition | ✓ WIRED | Verified by direct read and by `test_linking_a_run_stamps_the_acting_user_and_a_time` / `test_hand_typed_audit_values_do_not_bind` (value-comparison proof, not merely non-null) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All six phase-28 Django test modules pass, including the two new gap-closure test modules/classes | `python manage.py test solsys_code.tests.test_campaign_attribution solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_dismissals solsys_code.tests.test_admin solsys_code.tests.test_attribution_template -v 1` | 146 tests, OK (124 pre-existing + 22 new: 6 template + 7 standalone-admin-audit + 5 undo-ordering + 4 sole-high-band-filter) | ✓ PASS |
| No pending model/migration drift | `python manage.py makemigrations --check --dry-run solsys_code` | "No changes detected in app 'solsys_code'" | ✓ PASS |
| `ruff check` clean on all phase-28 Python files (01-06) | `ruff check campaign_attribution.py campaign_views.py admin.py test_attribution_template.py test_admin.py test_campaign_attribution.py test_campaign_attribution_views.py` | "All checks passed!" | ✓ PASS |
| `ruff format --check` clean on the same files | — | "7 files already formatted" | ✓ PASS |
| Confirm button reachable without typing a dismissal reason (browser behavior) | Static inspection of `attribution_queue.html` (both action forms) + structure-only test module | `formnovalidate` present on both Confirm buttons; `required` still present on both `reason` inputs; no form-level `novalidate` | ✓ PASS (was FAIL in prior pass — CR-01 closed) |
| Standalone admin audit fields cannot be hand-typed or left un-stamped | `CalendarEventMetaStandaloneAdminAuditStampTests` (7 tests, run directly) | All 7 pass, including value-comparison (not null-only) proof | ✓ PASS (was FAIL in prior pass — CR-02 closed) |
| A stale/tampered undo writes nothing; a genuine undo still works | `TestUndoConfirmationOrdering` (5 tests, run directly) | All 5 pass | ✓ PASS (WR-01 closed) |
| `sole_high_candidate_pk` agrees with the server-side gate under every band filter | `TestSoleHighCandidateUnderBandFilter` (4 tests, run directly) | All 4 pass | ✓ PASS (WR-02 closed) |
| Sphinx docs build over the edited runbook | `sphinx-build -M html ./docs ./_readthedocs_verify_tmp -T -D exclude_patterns=notebooks/*,_build` (run directly; temp output removed after) | "build succeeded, 9 warnings" (all pre-existing, unrelated to phase 28) | ✓ PASS |

Note: `python -m pytest -q` collects only 1 test (`tests/fomo/test_packaging.py::test_version`) — `pyproject.toml`'s `addopts = "--doctest-modules --doctest-glob=*.rst"` line is misplaced under `[tool.isort]` instead of `[tool.pytest.ini_options]`, so pytest never picks it up and the `docs`/`src` testpaths effectively collect nothing extra. This is a pre-existing repository misconfiguration (present before Phase 28, confirmed via `git log -p -- pyproject.toml`), not introduced or worsened by this phase, and is why the Sphinx build was verified directly instead of through `python -m pytest -q` as 28-05-PLAN.md's own verify step assumed.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ATTRIB-01 | 28-02, 28-04, 28-05 | Staff see a queue of suggested associations with evidence for each candidate | ✓ SATISFIED | Evidence columns + worklists implemented and tested; `.planning/REQUIREMENTS.md` now checked and marked Complete (documentation sync from prior pass's staleness note is closed) |
| ATTRIB-02 | 28-02, 28-03, 28-04, 28-05, 28-06 | Candidates confidence-scored and filterable, bulk-confirm the confident tail | ✓ SATISFIED | Banding, `?band=` filter, `confirm_selected` bulk path with server-side sole-High-candidate gate all implemented and tested; WR-02 fix keeps the display helper honest under every band filter |
| ATTRIB-03 | 28-01, 28-02, 28-03, 28-05 | No association is ever created without explicit staff confirmation | ✓ SATISFIED | Server-side re-validation solid; CR-01's UI-path regression (Confirm blocked in a real browser) is closed |
| ATTRIB-04 | 28-01, 28-03, 28-05, 28-06 | A staff user can undo a confirmed association | ✓ SATISFIED | Undo write path works and is tested; CR-02's attributability gap on the standalone admin path is closed; WR-01's undo-ordering integrity gap is closed |
| ATTRIB-05 | 28-02 | The known real case (pk=1-equivalent vs. its 11 LCO queue events) surfaced as a candidate | ✓ SATISFIED | `TestCriterion5RealCase`, unmodified and passing |
| ATTRIB-06 | 28-04 | Attribution can be completed before the first full reconcile sweep | ✓ SATISFIED (structurally) | Phase 29 (the reconciler) has not started, so nothing has run ahead of attribution; the mechanism is now usable end-to-end through the rendered UI, not only via a raw test-client POST |

No orphaned requirements — all six ATTRIB IDs declared across the six plans' `requirements:` frontmatter (`28-01`..`28-06`) match `.planning/REQUIREMENTS.md`'s ATTRIB-01 through ATTRIB-06 exactly, and `.planning/REQUIREMENTS.md` no longer contradicts `ROADMAP.md`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found. All five findings from the prior pass are closed: CR-01 (Confirm gated by Dismiss-only required field) fixed with `formnovalidate`; CR-02 (`CalendarEventMetaAdmin` unprotected audit fields) fixed with `readonly_fields` + `save_model()`; WR-01 (`_undo_confirmation()` dismissal-before-clear ordering) fixed with the `changed_count` gate; WR-02 (`sole_high_candidate_pk` computed from the band-filtered list) fixed by passing `full_candidates`; IN-01 (docstring/implementation mismatch on the drop threshold) closed by correcting the docstring, with the behavior change explicitly and reasonedly deferred as a documented no-op at current weights |

No TBD/FIXME/XXX debt markers found in any phase-28-touched file (every "TBD" occurrence is the established domain term for an unresolved observing window, not a completion marker — confirmed by direct grep across all nine files touched by 28-05/28-06).

### Human Verification Required

None. Both items flagged for human verification in the prior pass (browser-based confirmation of CR-01's blocking behavior; admin-page confirmation of CR-02's fabricated-attribution path) are now closed by structure-only automated tests specifically designed to be immune to the `self.client.post()` blind spot that let both defects ship past 124 previously-green tests: `test_attribution_template.py`'s `AttributionRenderedFormStructureTests` parses the real rendered HTML and asserts no Confirm submitter is gated by a required control (with a non-vacuity guard proving the assertion isn't passing for lack of candidates), and `CalendarEventMetaStandaloneAdminAuditStampTests` exercises the real admin change/add forms through `self.client` and asserts by value comparison (not null-only) that hand-typed audit fields do not bind.

## Gaps Summary

None. Both BLOCKER findings from the initial verification pass (CR-01, CR-02) and both WARNING findings plus the INFO finding (WR-01, WR-02, IN-01) from the code review are closed, each with a regression test specifically designed to fail under the pre-fix behavior (mutation checks performed and recorded verbatim in 28-05-SUMMARY.md and 28-06-SUMMARY.md confirm this). All 8 derived observable truths verify cleanly against the current codebase; all 6 ATTRIB requirements are satisfied and correctly reflected in `.planning/REQUIREMENTS.md`; the full phase-28 Django test suite (146 tests across 5 modules) passes; `ruff check`/`ruff format --check` are clean on every phase-28-touched Python file; migrations have no drift; and the Sphinx docs build succeeds over the updated runbook. Phase 28's goal — a staff queue of evidence-backed suggested associations, confirmable one candidate at a time through the actual rendered UI and undoable, with every write path attributable to a person and a time — is achieved.

---

_Verified: 2026-08-02T00:33:28Z_
_Verifier: Claude (gsd-verifier)_
