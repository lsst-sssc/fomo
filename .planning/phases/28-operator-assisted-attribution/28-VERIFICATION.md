---
phase: 28-operator-assisted-attribution
verified: 2026-08-01T20:12:47Z
status: gaps_found
score: 6/8 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Staff can confirm a candidate association one at a time through the rendered attribution-queue UI, not just via a bare test client POST"
    status: failed
    reason: "28-REVIEW.md CR-01: the per-candidate <form> wraps Confirm, the free-text 'reason' input, and Dismiss together. 'reason' carries the HTML5 required attribute; the Confirm <button> has no formnovalidate and the <form> has no novalidate. In any real browser, clicking Confirm is blocked by native constraint validation until the operator types something into the Dismiss-only reason box -- even though the server-side confirm handlers (_do_confirm_event/_do_confirm_record) never read reason at all. Invisible to the test suite because self.client.post() bypasses browser-side constraint validation, so all 124 passing tests give no signal on this path."
    artifacts:
      - path: "src/templates/campaigns/attribution_queue.html"
        issue: "Lines 91-103 (events table) and 177-189 (records table): shared <form> with reason[required] and no formnovalidate on the Confirm submit button"
    missing:
      - "Add formnovalidate to the Confirm button (or otherwise stop the required reason field from gating the Confirm submitter), verified with a check that does not rely on Django's test client (constraint validation is a browser-side behavior, not something self.client.post() exercises)"
  - truth: "Both the confirmation and the undo are attributable to a person and a time, across every write path that can create the association -- not only the path the test suite exercises"
    status: failed
    reason: "28-REVIEW.md CR-02: CalendarEventMetaInline (used from CampaignRunAdmin) correctly marks confirmed_by/confirmed_at readonly and CampaignRunAdmin.save_formset stamps them on a genuine run transition -- that path is solid and tested (test_save_formset_stamps_calendar_event_meta_on_run_transition). But CalendarEventMetaAdmin, the STANDALONE admin page that 27.1-02's own get_readonly_fields() docstring calls 'the primary staff surface for hand-linking a run to an event,' only freezes the event pk. It never adds confirmed_by/confirmed_at to readonly_fields and has no save_model() override to stamp them. A staff user linking a run through this primary surface can leave both fields blank (an association with no attribution at all) or hand-type an arbitrary confirmed_by/confirmed_at (fabricated attribution). This is also a direct miss against 28-01-PLAN.md's own must-have truth 'An admin-created event-to-run link is stamped with confirmed_by/confirmed_at, exactly as an admin-created observation link already is' -- that must-have was delivered only for the inline/save_formset path, not for the standalone admin page. No test in CalendarEventMetaStandaloneAdminPkFreezeTests or elsewhere exercises the audit fields on this page."
    artifacts:
      - path: "solsys_code/admin.py"
        issue: "CalendarEventMetaAdmin (lines 279-318): get_readonly_fields() only freezes 'event'; no save_model() override stamps confirmed_by/confirmed_at on a run None->not-None transition"
    missing:
      - "Add confirmed_by/confirmed_at to CalendarEventMetaAdmin.get_readonly_fields() (mirroring the inline) and a save_model() override that stamps confirmed_by=request.user/confirmed_at=timezone.now() on a genuine run transition, matching CampaignRunAdmin.save_formset()'s existing pattern -- plus a regression test analogous to test_save_formset_stamps_calendar_event_meta_on_run_transition but exercising this standalone admin page"
---

# Phase 28: Operator-Assisted Attribution Verification Report

**Phase Goal:** Give staff a queue of suggested run<->event and run<->record associations with the evidence visible, confirmable one candidate at a time and undoable — the mechanism that connects the existing calendar events and observation records to their parent runs without ever guessing silently.
**Verified:** 2026-08-01T20:12:47Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Staff see a queue of suggested associations with evidence side by side (matched telescope, date overlap, campaign, instrument similarity), not a bare score | ✓ VERIFIED | `campaign_attribution.AttributionCandidate` carries four separate evidence strings (`campaign_attribution.py:355-370`); `attribution_queue.html:85-89` renders all four plus the score as a subordinate badge; `TestEvidenceColumns.test_evidence_columns_score_and_band_badge_all_present` passes |
| 2 | Candidates are confidence-scored and filterable by score/band | ✓ VERIFIED | `band_for_score()` (High/Medium/Low), `?band=` GET filter wired in `AttributionQueueView`/template select; `TestBandFilterAndBanner.test_band_high_filter_shows_only_high_band_candidates` and `test_band_medium_filter_shows_only_medium_band_candidates` pass |
| 3 | No association is ever created without an explicit per-candidate staff POST (server-enforced, never trusting a rendered row), and no suggestion is ever offered across a campaign/target boundary | ✓ VERIFIED | `is_offered_candidate()` re-derives eligibility from the DB on every write; `_eligible_runs_for_event`/`_eligible_runs_for_record` gate on campaign membership before any scoring; `TestConcurrencyAndTampering.test_cross_campaign_run_post_writes_nothing` and `test_malformed_*` pass; `AttributionDecisionView.http_method_names = ['post']` blocks GET |
| 4 | Staff can confirm a candidate association one at a time through the rendered attribution-queue UI (not only via a raw test-client POST) | ✗ FAILED | **28-REVIEW.md CR-01**: the Confirm button shares a `<form>` with the Dismiss-only `reason` input, which carries the HTML5 `required` attribute; the Confirm button has no `formnovalidate` and the form has no `novalidate`. In a real browser, clicking Confirm is blocked by native constraint validation until text is typed into "Why doesn't this candidate match?" — invisible to `self.client.post()`-based tests, all of which bypass browser-side validation entirely |
| 5 | A confirmed association can be undone from the same screen that created it | ✓ VERIFIED | `AttributionDecisionView._undo_confirmation()`; `TestConfirmUndo.test_undo_confirmation_event_clears_link_and_writes_dismissal` / `test_undo_confirmation_record_deletes_link_and_writes_dismissal` pass |
| 6 | Both the confirmation and the undo are attributable to a person and a time, across every write path that can create the association | ✗ FAILED | **28-REVIEW.md CR-02**: `CalendarEventMetaInline`/`CampaignRunAdmin.save_formset` stamp `confirmed_by`/`confirmed_at` correctly and are tested (`test_save_formset_stamps_calendar_event_meta_on_run_transition`) — but `CalendarEventMetaAdmin`, the standalone admin page its own docstring calls "the primary staff surface for hand-linking a run to an event" (27.1-02), never adds those fields to `readonly_fields` and has no `save_model()` stamping override, so a link created there can carry no attribution, or a fabricated one. This is also a direct miss against 28-01-PLAN.md's own must-have ("An admin-created event-to-run link is stamped with confirmed_by/confirmed_at, exactly as an admin-created observation link already is") |
| 7 | The known real case is surfaced: `CampaignRun` pk=1-equivalent (FTS/MuSCAT4, 7-21 July, Siding Spring E10) is offered against LCO queue events/records despite the date-span and instrument-string mismatch | ✓ VERIFIED | `TestCriterion5RealCase` (`test_campaign_attribution.py:275+`) builds an equivalent fixture and asserts High band, sole-High-candidate status, against 10 events and 10 records; passes |
| 8 | A non-staff request to any attribution route is redirected before anything is written or rendered | ✓ VERIFIED | `StaffRequiredMixin` on both views; `TestAttributionStaffGating` (`test_anonymous_get_attribution_redirects`, `test_non_staff_get_attribution_redirects`, `test_anonymous_post_decide_redirects_and_writes_nothing`, `test_non_staff_post_decide_redirects_and_writes_nothing`) all pass |

**Score:** 6/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | `CalendarEventDismissal`, `ObservationRecordDismissal`, `CalendarEventMeta.confirmed_by`/`confirmed_at` | ✓ VERIFIED | Present with named per-pair `UniqueConstraint`s (`unique_calendar_event_dismissal_pair`, equivalent for records) |
| `solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py` | Schema for the above | ✓ VERIFIED | `makemigrations --check` reports no drift; migration content matches model diff |
| `solsys_code/admin.py` | `save_formset` stamps `CalendarEventMeta` audit fields on run transition | ⚠️ PARTIAL | Stamping present and tested for the `CampaignRunAdmin`/inline path only; the standalone `CalendarEventMetaAdmin` page has no equivalent stamping (CR-02, see gap 6 above) |
| `solsys_code/campaign_attribution.py` | Matcher: scoring, banding, orphan querysets, `candidates_for_event`/`candidates_for_record` | ✓ VERIFIED | 776 lines, all described functions present and covered by `test_campaign_attribution.py` |
| `solsys_code/campaign_views.py` | `AttributionQueueView`, `AttributionDecisionView` | ✓ VERIFIED | Present; server-side re-validation via `is_offered_candidate()` on every write, confirmed by reading the write paths |
| `solsys_code/campaign_urls.py` | `campaigns:attribution`, `campaigns:attribution_decide` | ✓ VERIFIED | Both routes present |
| `src/templates/campaigns/attribution_queue.html` | Four-section staff page | ⚠️ PARTIAL | Renders all four sections and full evidence, but the Confirm control is non-functional in a real browser (CR-01) |
| `solsys_code/campaign_tables.py` | `AttributionDismissedTable`, `AttributionConfirmedTable` | ✓ VERIFIED | Both classes present with CSRF-per-row and auto-escaping discipline |
| `src/templates/campaigns/campaign_list.html` | `attribution_count` banner | ✓ VERIFIED | Staff-only banner present; `TestBandFilterAndBanner.test_staff_campaign_list_banner_shows_count_and_link` / `test_anonymous_campaign_list_banner_shows_neither` pass |
| `docs/runbooks/telescope_runs_calendar.rst` | Attribution-pass operator section | ✓ VERIFIED | "How do I attribute existing calendar events and observation records to a run?" section present, describes worklists, evidence columns, dismissal semantics, Phase 29 done signal |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `campaign_attribution.py` | `difflib.SequenceMatcher` | tokenised instrument similarity | ✓ WIRED | `instrument_similarity()` uses `SequenceMatcher` on both whole strings and tokens |
| `campaign_attribution.py` | `CalendarEventDismissal`/`ObservationRecordDismissal` | dismissed-pair exclusion | ✓ WIRED | `candidates_for_event`/`candidates_for_record` filter `dismissed_run_ids` derived from these models |
| `campaign_views.py` | `campaign_attribution.is_offered_candidate` | server-side re-validation on every write | ✓ WIRED | Called in `_do_confirm_event`, `_do_confirm_record`, `_confirm_selected` before any write |
| `campaign_views.py` | `CalendarEventMeta` | atomic conditional update keyed on `run__isnull=True` | ✓ WIRED | `_do_confirm_event()` uses exactly this pattern |
| `campaign_views.py` | `CampaignRunObservation` | `get_or_create` guarded by unique constraint | ✓ WIRED | `_do_confirm_record()` |
| `attribution_queue.html` | `campaigns:attribution_decide` | form action on every confirm/dismiss/undo control | ✓ WIRED | Present on all forms, but see CR-01 — the Confirm submitter inside the shared form is blocked client-side before the request is even sent |
| `campaign_list.html` | `campaigns:attribution` | banner link | ✓ WIRED | Confirmed |
| `campaign_views.py` | `campaign_tables.py` | `AttributionDismissedTable`/`AttributionConfirmedTable` construction | ✓ WIRED | Confirmed in `get_context_data` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 28's four Django test modules pass | `python manage.py test solsys_code.tests.test_campaign_attribution solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_dismissals solsys_code.tests.test_admin` | 124 tests, OK | ✓ PASS (but does not exercise CR-01 — see truth 4) |
| No pending model/migration drift | `python manage.py makemigrations --check --dry-run solsys_code` | "No changes detected in app 'solsys_code'" | ✓ PASS |
| `ruff check` clean on phase-28 Python files | `ruff check campaign_attribution.py campaign_views.py campaign_tables.py admin.py models.py test_campaign_attribution*.py test_attribution_dismissals.py` | "All checks passed!" | ✓ PASS |
| `ruff format --check` clean | `ruff format --check` on the same files | "5 files already formatted" | ✓ PASS |
| Confirm button reachable without typing a dismissal reason (browser behavior) | Static inspection of `attribution_queue.html:91-103`/`177-189` (HTML5 constraint-validation semantics; not exercisable via `self.client.post()`) | `reason[required]` with no `formnovalidate`/`novalidate` anywhere in the shared form | ✗ FAIL — confirms CR-01 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ATTRIB-01 | 28-02, 28-04 | Staff see a queue of suggested associations with evidence for each candidate | ✓ SATISFIED | Evidence columns + worklists implemented and tested; REQUIREMENTS.md traceability table still lists this "Pending" — documentation not synced to delivery, not a code gap |
| ATTRIB-02 | 28-02, 28-03, 28-04 | Candidates confidence-scored and filterable, bulk-confirm the confident tail | ✓ SATISFIED | Banding, `?band=` filter, `confirm_selected` bulk path with server-side sole-High-candidate gate all implemented and tested |
| ATTRIB-03 | 28-01, 28-02, 28-03 | No association is ever created without explicit staff confirmation | ✓ SATISFIED (server-side) / ⚠️ UI regression | Server-side re-validation solid; CR-01 blocks the confirm UI path in a real browser (see truth 4) |
| ATTRIB-04 | 28-01, 28-03 | A staff user can undo a confirmed association | ✓ SATISFIED (except attributability gap) | Undo write path works and is tested; attributability guarantee broken for the standalone admin path (CR-02, truth 6) |
| ATTRIB-05 | 28-02 | The known real case (pk=1-equivalent vs. its 11 LCO queue events) surfaced as a candidate | ✓ SATISFIED | `TestCriterion5RealCase` |
| ATTRIB-06 | 28-04 | Attribution can be completed before the first full reconcile sweep | ✓ SATISFIED (structurally) | Phase 29 (the reconciler) has not started, so nothing has run ahead of attribution; the mechanism itself is usable end-to-end apart from CR-01's UI friction |

**Note on REQUIREMENTS.md staleness:** the requirements checklist and traceability table (`.planning/REQUIREMENTS.md` lines 47-52, 105-110) still mark ATTRIB-01, ATTRIB-02, ATTRIB-05, and ATTRIB-06 as unchecked/"Pending", even though ROADMAP.md marks Phase 28 complete (4/4 plans) and ties all six ATTRIB requirements to it. This is a documentation-sync gap, not a code gap — the underlying functionality for all four is present and tested — but it should be corrected so REQUIREMENTS.md reflects reality.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/templates/campaigns/attribution_queue.html` | 91-103, 177-189 | Confirm submitter sharing a form with a `required` field meant only for Dismiss | 🛑 Blocker | CR-01 — blocks the primary "Confirm" interaction in a real browser |
| `solsys_code/admin.py` | 279-318 | `CalendarEventMetaAdmin` does not protect/stamp `confirmed_by`/`confirmed_at` | 🛑 Blocker | CR-02 — the phase's own "primary staff surface" for hand-linking a run to an event can create unattributed or fabricated-attribution associations |
| `solsys_code/campaign_views.py` | 1441-1485 | `_undo_confirmation()` writes the dismissal side-effect before confirming the link-clearing update actually matched a row | ⚠️ Warning | 28-REVIEW.md WR-01 — a stale/malformed `run_pk` resubmit can permanently dismiss a pair that was never actually confirmed. Edge case (requires a non-current `run_pk`), not routine operation; does not block the phase goal today |
| `solsys_code/campaign_attribution.py` | 621-638, 669, 703 | `_sole_high_candidate_pk()` computed from the band-filtered list, not the full uncapped list its docstring promises | ⚠️ Warning | 28-REVIEW.md WR-02 — currently inert (no visibly-wrong checkbox renders today) but is a latent trap for the next consumer of `sole_high_candidate_pk` |
| `solsys_code/campaign_attribution.py` | 539-541, 598-600 | Candidate filtering compares the *rounded* display score (`round(score, 2)`), not the raw weighted sum, against the `<= 0.0` drop threshold | ℹ️ Info | 28-REVIEW.md IN-01 — a genuine sub-0.005 nonzero score could be dropped as if it had no evidence at all; unlikely at the current weights |

No TBD/FIXME/XXX debt markers found in any phase-28-touched file (occurrences of the string "TBD" are the established domain term for an unresolved observing window, not a completion marker).

### Human Verification Required

### 1. Confirm the browser-blocking behavior of CR-01 firsthand

**Test:** Open the attribution queue page as a staff user in an actual browser (not the Django test client), locate any candidate row, and click "Confirm" without typing anything into the "Why doesn't this candidate match?" box.
**Expected:** The candidate should be confirmed immediately (the server-side handler for `action=confirm` never reads `reason`).
**Why human:** This is exactly the class of defect the review flagged as invisible to `self.client.post()`-based tests — HTML5 constraint validation is a browser behavior, not something exercised by the existing automated suite. A human (or a browser-automation tool such as Playwright/Selenium) is needed to observe the actual blocked submission.

### 2. Confirm a run-link through the standalone `CalendarEventMeta` admin page and inspect the audit fields

**Test:** As a superuser, go to `/admin/solsys_code/calendareventmeta/<pk>/change/` for a row with `run` unset, set `run` to some `CampaignRun`, and save — without touching `confirmed_by`/`confirmed_at`. Then re-open the row.
**Expected:** Given CR-02's finding, `confirmed_by` and `confirmed_at` will likely remain blank (or accept whatever was typed/left from a prior edit) rather than being auto-stamped or protected.
**Why human:** Confirms the exploit path end-to-end through the live admin UI, complementing the static code-reading evidence already gathered.

## Gaps Summary

Phase 28 delivers a substantial, well-tested attribution mechanism: the matcher (`campaign_attribution.py`) is thorough and its scoring/banding/boundary-gate logic holds up under direct reading and its own acceptance test (`TestCriterion5RealCase`); server-side re-validation on every write (`is_offered_candidate()`) is real, not just claimed; staff gating, double-submit protection, dismissal-reason enforcement, and the campaign/target boundary gate are all genuinely tested and correct. 6 of 8 derived truths — including both halves of ROADMAP criterion 5 (the pk=1-equivalent real case) and both halves of criterion 3 (server-side confirmation integrity and the boundary gate) — verify cleanly.

However, two code-review findings (28-REVIEW.md CR-01, CR-02) are goal-blocking, not code-quality nits, and neither has been fixed (no `28-REVIEW-FIX.md` exists yet):

- **CR-01** breaks the phase goal's own wording — "confirmable one candidate at a time" — for real staff use: the rendered Confirm button is blocked by the browser's native HTML5 validation on a field (`reason`) that is only meaningful for Dismiss. Every one of the 124 passing Django tests uses `self.client.post()`, which bypasses browser-side constraint validation entirely, so this defect produced zero test failures despite disabling the primary interaction.
- **CR-02** breaks ROADMAP criterion 4's attributability guarantee ("both the confirmation and the undo are attributable to a person and a time") for the standalone `CalendarEventMetaAdmin` page — which 27.1-02's own docstring calls "the primary staff surface for hand-linking a run to an event." It also directly misses 28-01-PLAN.md's own must-have truth that an admin-created event-to-run link is stamped with `confirmed_by`/`confirmed_at` "exactly as an admin-created observation link already is" — that guarantee was delivered for the `CampaignRunAdmin`/inline path only.

Both are narrow, well-understood, single-file fixes (the review report includes concrete patches), but as shipped they mean a staff member using the rendered UI cannot straightforwardly confirm a candidate, and a staff member using the documented "primary" admin surface can create an association with no attribution at all or a fabricated one — undermining exactly the two guarantees (explicit per-candidate confirmation being usable, and attributability) this phase exists to add. Recommend a closure plan addressing both CR-01 and CR-02 (and, while in the area, considering WR-01's ordering fix) before Phase 29 relies on attribution having been completable through the intended UI.

---

_Verified: 2026-08-01T20:12:47Z_
_Verifier: Claude (gsd-verifier)_
