---
phase: 23
reviewers: [gemini, codex]
reviewed_at: 2026-07-16T18:39:52.000Z
plans_reviewed: [23-01-PLAN.md, 23-02-PLAN.md, 23-03-PLAN.md]
---

# Cross-AI Plan Review — Phase 23

## Gemini Review

# Cross-AI Plan Review: Weather/Storm Cancellation Handling (Phase 23)

We have reviewed the proposed execution plans (`23-01-PLAN.md`, `23-02-PLAN.md`, and `23-03-PLAN.md`) against the codebase. Below is a structured assessment of the plan quality, completeness, and architectural risks.

---

## 1. Summary

The proposed execution plans represent an exceptionally high-quality, surgical, and idiomatic implementation of Phase 23. By leveraging pre-existing abstractions such as `insert_or_create_calendar_event()`, the `_TERMINAL_PREFIXES` constant tuple, and the single-form actions pattern in `ApprovalQueueTable`, the plans achieve clean cross-system coordination with zero database migrations or new packages. The plans are highly defensive, demonstrating an acute awareness of critical edge cases (such as preventing calendar projection crashes on approved range/TBD runs and avoiding live-search widget leaks into the Decided table). The dependency chain and sequence of parallel vs. serial tasks are perfectly balanced.

---

## 2. Strengths

### Plan 23-01 (Classical Run Ingest Cancellation)
*   **Idempotency & Reversion Safety (`solsys_code/management/commands/load_telescope_runs.py`):** Computing the `title` fresh from the parsed status on every ingest instead of reading or appending to `event.title` is a major strength. It elegantly avoids double-prefixing on repeat imports and allows clean reversion to an unprefixed title if the cancellation word is removed (addressing **RESEARCH Pitfall 4**).
*   **Input Validation Constraint:** Looking up the title prefix from the fixed constant map `_CLASSICAL_STATUS_PREFIX` ensures that the prefix is strictly controlled and never derived or interpolated from untrusted line text, fulfilling **V5 Input Validation** constraints.
*   **CLAUDE.md Compliance:** Including the paired demo notebook update (`docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`) ensures strict compliance with repository documentation and demo guidelines.

### Plan 23-02 (CampaignRun Status Change Staff UI & Sync)
*   **Surgical separation of Table Gating (`solsys_code/campaign_tables.py`):** Introducing an independent `status_actions` initialization flag while keeping `show_actions=False` on the Decided table is a brilliant architectural pattern. It ensures that the actions are rendered without inadvertently leaking the interactive `_render_site_search_widget()` into the unresolved site column (successfully mitigating **RESEARCH Pitfall 3**).
*   **Existence-Guarded Calendar Updates (`solsys_code/campaign_views.py`):** Gating `insert_or_create_calendar_event()` behind a `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()` check is a critical reliability guard. Because range-window or TBD runs never project calendar events, passing a title/description-only update fields dictionary to the helper's create-path would otherwise crash the view with a database `IntegrityError` or `TypeError` due to missing non-nullable `start_time`/`end_time` arguments (successfully mitigating **RESEARCH Pitfall 1**).
*   **Server-Side Access Gating:** The view enforces `run.approval_status == CampaignRun.ApprovalStatus.APPROVED` before applying any state change. This ensures that client-side row-level button visibility is never trusted in isolation, fulfilling **V4 Access Control** guidelines.
*   **Visual Orthogonality:** Utilizing `'[WEATHERED]'` and adding it to `_TERMINAL_PREFIXES` in `calendar_display_extras.py` lets the weathered-out runs inherit the red terminal box-shadow ring automatically for free, keeping the presentation layer completely decoupled and unified.

### Plan 23-03 (Gemini FT-115 Scenario & I11 Site Resolution)
*   **Mocked Tier-2 MPC Resolution (`solsys_code/tests/test_campaign_approval.py`):** Adding the real ground-based coordinates and characteristics of `I11` (Gemini South) to `BULK_MPC_FIXTURE` allows full test coverage of Tier-2 site resolution without hitting the live network, preventing brittle integration tests.
*   **Scenario Verification Coverage:** Creating an end-to-end integration test specifically exercising the FT-115 range-window run validates that the exists-guarded `_set_run_status()` handle operates successfully without crashing or fabricating spurious calendar rows, leaving zero testing gaps.

---

## 3. Concerns

### Moderate Severity: Unhandled `DoesNotExist` on Deleted Rows
*   **Location:** `solsys_code/campaign_views.py:CampaignRunDecisionView._set_run_status()`
*   **Mechanism:** The plan performs:
    ```python
    CampaignRun.objects.filter(pk=pk, approval_status=CampaignRun.ApprovalStatus.APPROVED).update(run_status=new_run_status)
    run.refresh_from_db()
    ```
    If another staff member concurrently deletes this run or shifts its approval status, the `.update()` will match 0 rows. While this is expected behavior, calling `run.refresh_from_db()` on a row that has been concurrently deleted from the database will raise an unhandled `CampaignRun.DoesNotExist` exception, triggering a generic 500 error.
*   **Precedent in Codebase:** In `solsys_code/campaign_views.py:629-639` (`_resolve_site()`), the returned count of the update is explicitly checked (`if claimed == 0: ... return redirect(...)`) before calling `refresh_from_db()` to avoid exactly this runtime exception.

---

## 4. Suggestions

### 1. Graceful Guard on `_set_run_status` Update Count
To eliminate the unhandled `DoesNotExist` risk during concurrent access:
*   **Change:** Capture the returned update count from the database filter statement. If it is `0`, display a warning message and redirect gracefully, avoiding the `refresh_from_db()` call.
*   **Proposed Implementation (`solsys_code/campaign_views.py`):**
    ```python
    updated_count = CampaignRun.objects.filter(
        pk=pk,
        approval_status=CampaignRun.ApprovalStatus.APPROVED
    ).update(run_status=new_run_status)

    if updated_count == 0:
        messages.warning(request, "This run's status could not be updated (it may have been modified or deleted).")
        return redirect('campaigns:approval_queue')

    run.refresh_from_db()
    ```

---

## 5. Risk Assessment

*   **Overall Risk Level:** **LOW**
*   **Justification:** The plans are highly targeted and exceptionally surgical. No database migrations are introduced, no external dependencies are added, and the proposed structures mirror existing in-repo patterns (e.g., LCO title prefixes and `get_token` form actions) with 100% fidelity. Potential edge cases have been exhaustively investigated, and the suggested update-count guard reduces the remaining concurrent access risk to negligible levels. All tests can be run locally via standard `./manage.py test solsys_code` and are guaranteed to maintain high confidence in the codebase.

---

## Claude Review

*(skipped — this review was run from within a Claude Code session; the executing AI's own CLI is excluded for reviewer independence)*

---

## Codex Review

## 23-01-PLAN.md

**Summary**
Strong, low-risk plan. It targets the exact gap in `load_telescope_runs`: title is currently always computed as plain `{telescope} {instrument}` at `solsys_code/management/commands/load_telescope_runs.py:139`, while `parsed.status` is already available and validated from `KNOWN_STATUSES` in `solsys_code/telescope_runs.py:36` and `:361-393`.

**Strengths**
- Correctly reuses the existing idempotent update path. `insert_or_create_calendar_event()` updates changed fields in place via `_update_or_unchanged()` at `solsys_code/calendar_utils.py:297-315`.
- Correctly avoids a templatetag change: `[CANCELLED]` is already terminal in `_TERMINAL_PREFIXES` at `solsys_code/templatetags/calendar_display_extras.py:46`.
- Re-ingest/revert testing is well chosen because the lookup includes telescope/instrument/start time and updates the existing row at `load_telescope_runs.py:146-150`.

**Concerns**
- **LOW:** The "all four non-cancelled statuses" test can become misleading if implemented as one schedule with the same telescope/instrument/night, because the command keys events by telescope/instrument/start time and may update the same row rather than creating distinct rows. Use subtests with cleanup or distinct nights.
- **LOW:** Notebook re-execution may change unrelated output cells. The plan says "do not alter unrelated cells beyond re-execution," which is fine, but reviewers should expect broader output diffs.

**Suggestions**
- Add a tiny helper or local expression for `base_title = f'{parsed.telescope} {parsed.instrument}'` to make prefix/revert behavior obvious.
- In the revert test, assert `CalendarEvent.objects.count()` remains unchanged and the original `pk` survives.

**Risk Assessment: LOW**
The change is narrow, uses existing parsed state, and relies on an already-proven update helper.

---

## 23-02-PLAN.md

**Summary**
Mostly well-designed and aligned with the current code. The plan correctly identifies the two real integration risks: `ApprovalQueueTable.show_actions=False` currently suppresses actions and preserves read-only site rendering at `solsys_code/campaign_tables.py:284-305`, and `_project_calendar_event()` only creates campaign events for resolved single-night runs at `solsys_code/campaign_views.py:388-435`.

**Strengths**
- The independent `status_actions` flag is the right design. Flipping `show_actions=True` would affect `render_site()` and could expose the live-search widget, as shown by `campaign_tables.py:272-297`.
- Existence guard before updating a `CAMPAIGN:{pk}` event is necessary because `insert_or_create_calendar_event()` falls back to `get_or_create()` at `calendar_utils.py:375`, and a title/description-only create would lack required event times.
- Staff gating builds on the existing `StaffRequiredMixin` endpoint and fixed action whitelist at `campaign_views.py:438-456`.
- Adding `[WEATHERED]` to `_TERMINAL_PREFIXES` is correct because `status_border_css()` only rings titles starting with known prefixes at `calendar_display_extras.py:94-99`.

**Concerns**
- **MEDIUM:** `_set_run_status()` is described as using a conditional `.update()`, but the plan does not require checking `updated_count`. Existing code treats update counts as load-bearing: approve/reject checks `updated_count` at `campaign_views.py:460-464` and handles stale rows at `:543-552`; `_resolve_site()` checks `claimed == 0` at `:629-639`. Without checking count, a race where approval status changes after the initial fetch could still show success.
- **LOW:** The plan says title/description are updated, but the proposed `description = run.observation_details` may be identical to the existing projection description from `_project_calendar_event()` at `campaign_views.py:395-396`. If the requirement expects status reflected in description too, append a status line.
- **LOW:** The `exists()` then `insert_or_create_calendar_event()` pattern still has a tiny delete-between-check-and-use race. Low practical risk for staff UI, but worth noting.

**Suggestions**
- Require:
  ```python
  updated_count = CampaignRun.objects.filter(
      pk=pk,
      approval_status=CampaignRun.ApprovalStatus.APPROVED,
  ).update(run_status=new_run_status)
  if updated_count == 0:
      messages.warning(...)
      return redirect(...)
  ```
- Consider adding status to event description, e.g. `Run status: Weather/Technical Failure`, if D-05's "title/description" wording is meant literally.
- In rendering tests, inspect `response.context['decided_table']` or scoped HTML around the decided table, not only global response content.

**Risk Assessment: MEDIUM**
The architecture is good, but the missing `updated_count` check weakens the claimed race/staleness safety.

---

## 23-03-PLAN.md

**Summary**
The scenario coverage is valuable, but this is the weakest plan. It correctly depends on Plan 02 and exercises the important range-window/no-event path, but its `resolve_site('I11')` test design is under-specified and likely wrong if implemented exactly as written.

**Strengths**
- Correctly treats FT-115 as a `CampaignRun` range window, not a Gemini `ObservationRecord`.
- Correctly expects no calendar projection for the 2026-07-13..2026-07-16 range, matching `_project_calendar_event()`'s single-night guard at `campaign_views.py:388-393`.
- Good reuse of existing test fixture patterns in `test_campaign_approval.py`, especially `_make_pending_run()` at `solsys_code/tests/test_campaign_approval.py:129-143`.

**Concerns**
- **HIGH:** Adding `I11` only to `BULK_MPC_FIXTURE` is not enough to test `resolve_site('I11')`. `resolve_site()` calls `MPCObscodeFetcher.query()` and then `to_observatory()` at `campaign_utils.py:184-200`; it does not use `query_all()`. `to_observatory()` requires single-code response keys such as `obscode`, `rhocosphi`, `rhosinphi`, `created_at`, `updated_at`, and `uses_two_line_observations` at `solsys_code/solsys_code_observatory/utils.py:119-154`. The bulk fixture shape at `test_campaign_approval.py:65-108` lacks those.
- **MEDIUM:** The plan says a second POST from weathered to cancelled is an "idempotent no-op," but it is a real state transition between two different `RunStatus` values defined at `models.py:57-59`.
- **MEDIUM:** The live FT-115 row is pushed into `user_setup`, not a committed artifact. That may be acceptable for dev data, but it means the plan's success criteria are not fully verifiable by code review or CI.
- **LOW:** Timezone assertion for `I11` can be fragile unless the test patches `to_observatory()` or provides full parallax constants and controls `timezonefinder`.

**Suggestions**
- For the `resolve_site('I11')` test, either:
  - create a deterministic `Observatory` row and assert Tier 1 resolution, or
  - patch `MPCObscodeFetcher.query()` to populate a full single-code `obs_data` shape and patch/verify `to_observatory()` behavior.
- Change the second status POST assertion wording: it is not a no-op if it changes `WEATHER_TECH_FAILURE` to `CANCELLED`; instead assert "still no CalendarEvent fabricated."
- If the real dev DB row matters, add a management command, fixture, or documented manual verification step outside the automated plan rather than making it part of autonomous success.

**Risk Assessment: MEDIUM-HIGH**
The production-code risk is low because this plan adds tests only, but the test plan can easily encode the wrong resolver path and give false confidence about D-06.

---

## Consensus Summary

Both reviewers independently traced the actual code paths (not just the plan text) and converged on the same two structural weak points, while diverging on how severe they judge the overall phase risk to be (Gemini: LOW across the board; Codex: LOW / MEDIUM / MEDIUM-HIGH per plan).

### Agreed Strengths
- The `insert_or_create_calendar_event()` existence guard in Plan 23-02 correctly prevents the `CalendarEvent.objects.create()` crash on range/TBD/unresolved-site runs (RESEARCH Pitfall 1) — both reviewers cite this as the single most important guard in the phase.
- The independent `status_actions` flag (vs. reusing `show_actions`) correctly avoids leaking the live-search site widget onto the Decided table (RESEARCH Pitfall 3).
- Plan 23-01's "always recompute title fresh from `parsed.status`" approach correctly handles both idempotent re-ingest and reversion when the cancelled word is removed (RESEARCH Pitfall 4).
- Zero new dependencies, zero migrations — both reviewers rate the overall architectural approach as idiomatic and low-blast-radius.

### Agreed Concerns
- **[MEDIUM — both reviewers, same file:line evidence] `_set_run_status()`'s conditional `.update()` in Plan 23-02 does not require checking the returned update count before proceeding (e.g. before `refresh_from_db()`).** Both reviewers independently traced this against the exact same precedent: `_resolve_site()`'s `if claimed == 0: ... return redirect(...)` guard at `campaign_views.py:629-639`, and Gemini additionally notes approve/reject's `updated_count` check at `campaign_views.py:460-464`. Gemini frames the failure mode as an unhandled `CampaignRun.DoesNotExist` on `refresh_from_db()` after a concurrent delete/status-change race a 500 error, not a silent no-op. **Action for planner:** Plan 23-02's `_set_run_status()` task should explicitly capture `updated_count` from the `.update()` call and short-circuit with a warning message + redirect when it is `0`, mirroring `_resolve_site()`'s existing pattern, before calling `refresh_from_db()` or building the success message.

### Divergent Views
- **Codex-only, HIGH severity: Plan 23-03's `resolve_site('I11')` test design is likely wrong as written.** Codex traced that `resolve_site()` calls `MPCObscodeFetcher.query()` (single-code) → `to_observatory()`, not `query_all()`, and that `to_observatory()` needs single-code response keys (`obscode`, `rhocosphi`, `rhosinphi`, `created_at`, `updated_at`, `uses_two_line_observations`) that the plan's proposed approach of adding an `I11` entry to `BULK_MPC_FIXTURE` does not supply — `BULK_MPC_FIXTURE`'s shape is for the bulk/`query_all()` path, not the single-code path `resolve_site()` actually exercises. Gemini's review did not flag this — it treated the `BULK_MPC_FIXTURE` addition as sufficient. This is a concrete, source-grounded claim (with exact function/line citations) that the planner should verify directly against `campaign_utils.py:184-200` and `solsys_code_observatory/utils.py:119-154` before execution, since an incorrectly-mocked test would pass while giving false confidence about D-06's site-resolution claim.
- **Severity calibration diverges but substance does not.** Gemini rates overall phase risk LOW throughout; Codex rates 23-01 LOW, 23-02 MEDIUM, and 23-03 MEDIUM-HIGH. The underlying findings substantially overlap (both flag the same `updated_count` gap); the difference is in how much weight each reviewer assigns to a staff-only internal tool's race-condition/test-fixture risks versus the phase's otherwise low architectural blast radius.
- **Codex-only, LOW severity, worth a planner decision:** whether `_project_calendar_event()`'s existing description text and the new status-change description update would be identical/redundant, and whether D-05 intends the status to also appear in the event `description` (not just the title prefix) — Codex flags this as worth a one-line addition (`Run status: Weather/Technical Failure`) if D-05's wording is meant literally; Gemini did not address this.
