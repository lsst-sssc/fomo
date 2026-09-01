# Phase 29 — UI Review

**Audited:** 2026-08-05
**Baseline:** Abstract 6-pillar standards
**Screenshots:** Not captured (no dev server running; code-only audit)

---

## Executive Summary

Phase 29 ("The Reconciler") is a **backend-only milestone** that creates an idempotent calendar reconciler to project `CampaignRun` state onto existing `tom_calendar` event rows (titles, descriptions). **No new HTML templates, CSS, or JavaScript were introduced.** The phase's user-visible footprint is entirely the **generated content** (event titles/descriptions) displayed on the existing `/calendar/` page and event-detail pop-up — not new UI components.

As a result:
- **Pillar 1 (Copywriting)** is the only pillar with material substance (the reconciler generates event titles, descriptions, and skip-reason messages).
- **Pillars 2-5 (Visuals, Color, Typography, Spacing)** are N/A — no new UI components exist to audit.
- **Pillar 6 (Experience Design)** is partially relevant — the backend workflow is more consistent, but user-facing interaction patterns are unchanged.

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | Practical skip reasons and event titles; functional but not polished |
| 2. Visuals | 4/4 | No new components created; existing tom_calendar UI unchanged |
| 3. Color | 4/4 | No new color usage introduced |
| 4. Typography | 4/4 | No new typography introduced |
| 5. Spacing | 4/4 | No new spacing introduced |
| 6. Experience Design | 3/4 | Backend workflow improvement (automatic reconciliation); limited new visible behavior |

**Overall: 21/24**

---

## Top 3 Priority Fixes

1. **Polish event title copywriting** — Titles like `"{CAMPAIGN}: {TELESCOPE}"` are functional but terse. Consider: `"Campaign {NAME} · {TELESCOPE}"` or `"{TELESCOPE} observation ({CAMPAIGN})"` for better scannability on a crowded calendar.

2. **Clarify skip-reason UX** — Messages like `"window_end before window_start"` and `"missing telescope/instrument"` are technical. Operators would benefit from more actionable phrasing: e.g., `"Window dates invalid (end before start)"` or `"No telescope/instrument assigned"`.

3. **Add event-creation feedback in calendar UI** — Staff actions (approve, resolve_site, etc.) now trigger automatic reconciliation, but there's no visual confirmation on the calendar that events were created. Consider a flash message or calendar-event count indicator post-action.

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

**Audit Method:** Code review of title/description generation and skip-reason strings.

**Findings:**

1. **Event titles** (`campaign_reconciler.event_title()`):
   - Format: `"{CAMPAIGN_NAME}: {TELESCOPE_INSTRUMENT}"` + optional `"(window START..END)"` + optional status prefix (`[CANCELLED]` or `[WEATHERED]`)
   - **Issue**: Terse, no article or phrasing variation. All titles follow identical pattern.
   - **Evidence**: `solsys_code/campaign_reconciler.py:144-153`
   - **Impact**: Calendar displays many similar-looking titles; no visual hierarchy or context beyond raw name concatenation.
   - **Example**: `"3I/ATLAS: FTN/MuSCAT3 (window 2025-07-04..2025-07-04)"` — functional but not scannable.

2. **Event descriptions** (`campaign_reconciler.event_description()`):
   - Includes `run.observation_details` + optional status line (`"Run status: {status}"`).
   - **Positive**: Reuses existing `observation_details` field, good content reuse.
   - **Issue**: Status line only added for terminal statuses (`CANCELLED`, `WEATHERED`), not for approved/resolved runs. Leaves most events without status context.
   - **Evidence**: `solsys_code/campaign_reconciler.py:156-161`
   - **Impact**: Operators must click into the event to confirm it's `APPROVED` or resolve its site status; no inline indication.

3. **Skip-reason messages** (user-facing in `reconcile_campaign_runs --dry-run` output):
   - Strings: `"not approved"`, `"missing telescope/instrument"`, `"TBD window"`, `"window_end before window_start"`, `"unresolved site"`
   - **Positive**: Exact, non-generic; no "error occurred" patterns.
   - **Issue**: Technical phrasing; some are abbreviations (`"TBD"`) without explanation. Operators need to understand each reason to resolve it.
   - **Evidence**: `solsys_code/campaign_reconciler.py:174-184`
   - **Severity**: LOW — these are operator-facing only (not end-user-facing), and the sweep operates in batch mode, so clarity is less critical than correctness.

4. **Command summary output**:
   - Format: `"Done. runs: 44, created: 63, updated: 1, unchanged: 0, skipped: 8, failed: 0, blocked: 0"`
   - **Positive**: Clear counter names, consistent with existing `import_campaign_csv` format.
   - **Minor issue**: No friendly "success" framing (e.g., "All calendar events synced.").

**Score Justification**: Copywriting is functional and non-generic, meeting basic UX expectations. However, it's terse and lacks polish — no variation in phrasing, no conversational tone, no friendly status context. This is typical operational/backend copy: clear to engineers, but not optimized for operator experience.

---

### Pillar 2: Visuals (4/4)

**Audit Method:** File inventory for new HTML templates or visual components.

**Findings:**

- No new HTML templates created or modified in Phase 29.
- No new CSS files added.
- No new visual components exist.
- The existing `tom_calendar` UI renders the generated titles and descriptions unchanged.
- Event styling (borders, colors, icons) remains under `tom_calendar`'s control.

**Score Justification**: No new components = no new visual issues. The existing UI continues unchanged. Score: 4/4 (no new work, no new problems).

---

### Pillar 3: Color (4/4)

**Audit Method:** Grep for new color definitions; check for hardcoded colors in generated content.

**Findings:**

- No new color variables or hardcoded `#`, `rgb()`, or `hsl()` values added to Phase 29 code.
- The existing `RUN_STATUS_CALENDAR_PREFIX` dict (moved from `campaign_views.py`) uses only text prefixes (`[CANCELLED]`, `[WEATHERED]`), no color tags.
- Accent color usage for terminal-status events relies on existing `tom_calendar` CSS rules (`calendar_display_extras._TERMINAL_PREFIXES`), unchanged.

**Score Justification**: No new color scope or usage; existing color scheme applies unchanged. Score: 4/4.

---

### Pillar 4: Typography (4/4)

**Audit Method:** Grep for font-size, font-weight, or text-styling changes.

**Findings:**

- No new typography classes or inline styles added.
- Generated text (titles, descriptions) flows directly into existing `tom_calendar` event DOM, using inherited font sizes and weights.
- Status prefixes (`[CANCELLED]`, `[WEATHERED]`) are plain text, no formatting.

**Score Justification**: No new typography introduced; existing font hierarchy applies. Score: 4/4.

---

### Pillar 5: Spacing (4/4)

**Audit Method:** Grep for margin/padding changes; check template layout modifications.

**Findings:**

- No new spacing classes, padding adjustments, or layout changes in Phase 29.
- The reconciler generates text content; layout (margins, gaps between events, card padding) is controlled by existing `tom_calendar` CSS.

**Score Justification**: No new spacing introduced. Score: 4/4.

---

### Pillar 6: Experience Design (3/4)

**Audit Method:** Code review of workflow changes and state-handling improvements.

**Findings:**

1. **Automatic reconciliation on staff actions** (RECON-08):
   - **Positive**: Staff actions (`approve`, `resolve_site`, `mark_cancelled`, `mark_weather_failure`) now call `campaign_reconciler.reconcile_run()` automatically instead of relying on a separate backfill command.
   - **Positive**: Idempotent (`running it twice changes nothing` — RECON-01 proven).
   - **Positive**: Deterministic (all four staff actions use the same reconciler logic, no divergence).
   - **Evidence**: `solsys_code/campaign_views.py` staff-action calls to `reconcile_run()`; `solsys_code/tests/test_campaign_approval.py` validates idempotency and event creation across approval workflows.
   - **Impact**: Operators no longer need to remember to run a separate backfill command; calendar state stays in sync automatically with run status.

2. **No visual feedback on event creation**:
   - **Issue**: When a staff member approves a run, the reconciliation happens silently. The calendar doesn't immediately show the new event without a page refresh.
   - **Severity**: MEDIUM — creates confusion: staff take an action but don't see an immediate result.
   - **Workaround**: Staff can refresh the calendar page manually or wait for their browser's auto-refresh.
   - **Evidence**: `solsys_code/campaign_views.py:approve()` calls `reconcile_run()` and returns a success message, but the template (`approval_queue.html`) does not trigger a page reload or calendar refresh.

3. **Error handling on event-creation failures**:
   - **Positive**: `approve()` swallows projection failures (run stays approved; event creation is not required to succeed). `resolve_site()` retains the `site_needs_review=True` flag if projection fails, allowing retry.
   - **Positive**: Asymmetric failure handling matches existing pattern (D-04).
   - **Concern**: Operators see no indication that calendar projection failed if `approve()` succeeds but `reconcile_run()` raises an exception (e.g., timezone parsing error).
   - **Evidence**: `solsys_code/campaign_views.py:approve()` does not log or display reconciliation failure details.

4. **Dry-run workflow**:
   - **Positive**: `reconcile_campaign_runs --dry-run` reports what would be created/updated/unchanged without writing, allowing operators to validate before sweeping.
   - **Positive**: Second dry-run after a real sweep correctly reports `would_create: 0, would_update: 0` (idempotency proof).
   - **Evidence**: Phase 29 plans 29-01 through 29-06 all test dry-run parity.

5. **Skip-reason reporting**:
   - **Positive**: The batch command itemizes skips by run PK and reason (e.g., `"Run pk=4: skipped (TBD window)"`) to stderr, allowing operators to see what was not reconciled and why.
   - **Positive**: All skip reasons are actionable (approval, telescope assignment, window dates, site resolution).
   - **Issue**: No guidance on how to resolve each skip reason (e.g., "TBD window" does not suggest "edit the `window_start`/`window_end` fields").

**Score Justification**: Workflow improvements are solid (automatic reconciliation, idempotency, dry-run support), but the user-facing experience lacks polish (no visual feedback on success, no guidance on failures, no error handling hints). This is typical operational software: it works reliably but doesn't guide the operator. Score: 3/4.

---

## Files Audited

**Backend modules (reconciler core, staff-action rewires):**
- `solsys_code/campaign_reconciler.py` (144 lines of core logic + title/description builders)
- `solsys_code/campaign_views.py` (staff actions rewired to call `reconcile_run()`)
- `solsys_code/calendar_utils.py` (no-churn helpers for event updates)

**Management command:**
- `solsys_code/management/commands/reconcile_campaign_runs.py` (dry-run, summary reporting)

**Database:**
- `solsys_code/models.py` (`CampaignRun.Source.ESO_QUEUE` added per real-data need)
- `solsys_code/migrations/0014_alter_campaignrun_source.py`

**Documentation & tests:**
- `docs/runbooks/telescope_runs_calendar.rst` (runbook updated; no new UI patterns)
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (demo notebook, not UI code)
- All test files in `solsys_code/tests/test_campaign_reconciler.py`, `test_campaign_approval.py`, `test_admin.py` (no template assertions)

**No HTML, CSS, or JavaScript files modified.**

---

## Summary

Phase 29 successfully delivers a reliable, idempotent backend reconciler. The copywriting (event titles, descriptions, skip reasons) is functional and non-generic, meeting minimum UX standards. However, the phase does not introduce new UI components, interaction patterns, or visual polish — it is purely a backend infrastructure improvement that generates content for an existing UI.

**Recommendation**: Phase 29 is production-ready as-is. The three priority fixes above are enhancements for future phases, not blockers. If operator experience is a priority, consider:
- A light UI refresh for event-title formatting (Pillar 1).
- Flash messages or toast notifications when staff actions trigger reconciliation (Pillar 6).
- Inline status indicators on calendar events (Pillar 1/6).

---

*Phase: 29-the-reconciler*
*Audited: 2026-08-05*
