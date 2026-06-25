---
phase: 08-telescope-label-verification-sidecar
verified: 2026-06-25T13:44:25Z
status: human_needed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Open /calendar/ in a browser with at least one fallback-labeled event present (e.g. seed one via the demo notebook's fallback fixture, or run sync_lco_observation_calendar against a record that times out) and look at the calendar grid."
    expected: "The fallback event's block shows a visibly dashed border distinct from neighboring solid-bordered events, at normal viewing zoom, without needing to open the event or read its title."
    why_human: "Automated tests assert the exact CSS string '2px dashed rgba(0, 0, 0, 0.65)' and the tooltip substring 'estimate' are present in the rendered HTML — they prove the markup exists and is scoped correctly, but they cannot judge whether the dashed border is subjectively perceptible as a visual cue distinguishable from the existing [QUEUED] solid border treatment and the default solid/borderless verified style, side by side on an actual rendered page."
  - test: "Hover over the same fallback-labeled event block in the browser."
    expected: "The browser's native title tooltip appears within ~1 second, showing the plain-language sentence explaining the label is an unverified, coarse fallback (not the LCO API outcome itself)."
    why_human: "The test suite asserts the title= attribute's substring is present in the HTML response; it does not (and cannot, without a browser automation tool) confirm the native tooltip actually renders on hover in a real browser."
---

# Phase 8: Telescope Label Verification Sidecar Verification Report

**Phase Goal:** Operators can tell, directly in the calendar UI, whether a synced event's telescope label was live-verified against the LCO API or fallback-guessed, without reading title text.
**Verified:** 2026-06-25T13:44:25Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Roadmap Phase 8 Success Criteria (the contract) plus PLAN frontmatter must-haves, merged:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | After running `sync_lco_observation_calendar`, every synced `CalendarEvent` has an associated `CalendarEventTelescopeLabel` row correctly marked verified or fallback, matching `telescope_api_failed`. | ✓ VERIFIED | `sync_lco_observation_calendar.py:635-637` calls `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': not telescope_api_failed})` inside the per-record loop, immediately after the existing `get_or_create`. `test_display_01_verified_record_creates_sidecar_row_is_verified_true` and `test_display_01_fallback_record_creates_sidecar_row_is_verified_false` both pass (confirmed by direct test run, 49/49 green). |
| 2 | Events created by `load_telescope_runs` (classical schedule) have no sidecar row and render as "verified" by documented default — no behavior change to that command. | ✓ VERIFIED | `grep` confirms zero references to `CalendarEventTelescopeLabel`/`telescope_label` in `load_telescope_runs.py` (file untouched). `test_display_01_no_sidecar_row_for_classically_scheduled_event` asserts `CalendarEventTelescopeLabel.objects.count() == 0` and that `event.telescope_label_meta` raises `DoesNotExist`; passes. Template uses `== False` (not `|default:False`) so a missing row + an explicit `True` both fall to the verified/unstyled branch — confirmed by reading `calendar.html`. |
| 3 | On the calendar page, a fallback-labeled event is visually distinguishable (border/badge) from a verified one, discoverable without opening the event or reading its title. | ✓ VERIFIED (markup) / see human verification | `calendar.html` lines 160-162 (all-day) and 174-182 (timed) both add `border: 2px dashed rgba(0, 0, 0, 0.65);` only when `event.telescope_label_meta.is_verified == False`. `test_dashed_border_count_matches_fallback_event_count_only` asserts the marker count equals exactly the fallback-event day-cell occurrence count (3), proving verified/no-row events get zero occurrences. Markup-level proof is solid; actual visual perceptibility in a rendered browser is a human-judgment item (see below). |
| 4 | Hovering a fallback-labeled event shows a tooltip with the verification detail (not just the visual cue alone). | ✓ VERIFIED (markup) / see human verification | Both branches add `title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0)."` on the same outer (hoverable) div as the dashed border — not a child span (Pitfall 4 honored). `test_fallback_events_get_dashed_border_and_tooltip` asserts the `estimate` substring is present. Native-tooltip-on-hover behavior in an actual browser is a human-judgment item. |
| 5 | Re-running the sync command on unchanged records does not create duplicate sidecar rows and does not churn `CalendarEvent.modified` (existing no-churn contract preserved). | ✓ VERIFIED | `test_display_01_rerun_on_unchanged_record_no_duplicate_sidecar_row` runs sync twice on an unchanged record and asserts `CalendarEventTelescopeLabel.objects.count() == 1` after the second run and that `CalendarEvent.pk` is unchanged; passes. The sidecar write is a standalone `update_or_create` statement, never folded into `fields`/`changed` (confirmed by reading lines 605-637) — `CalendarEvent.modified`'s own no-churn contract (asserted by the pre-existing Phase 7 test `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged`) is untouched by this phase's change since the sidecar lives in a separate table/statement. |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | Defines `CalendarEventTelescopeLabel` with `event` (OneToOneField, pk, related_name='telescope_label_meta', CASCADE) and `is_verified` (BooleanField, default=True), both with `verbose_name` | ✓ VERIFIED | Read directly — matches spec exactly, plus Google-style docstring and `__str__`. |
| `solsys_code/migrations/0001_calendareventtelescopelabel.py` | Generated (not hand-written) first migration for the app | ✓ VERIFIED | Generated-by-Django header, `initial = True`, single `CreateModel` op matching the model fields. `./manage.py makemigrations solsys_code --check --dry-run` reports "No changes detected" (run live, exit 0). |
| `sync_lco_observation_calendar.py` write site | Calls `CalendarEventTelescopeLabel.objects.update_or_create` after the existing `get_or_create` | ✓ VERIFIED | Confirmed at lines 630-637, standalone statement, inside the per-record loop. |
| `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | Executed cell(s) demonstrating the sidecar write | ✓ VERIFIED | 3 new cells (verified/fallback/no-row) with real executed output: `PASS: verified record has a sidecar row with is_verified=True`, `PASS: fallback-labeled record has a sidecar row with is_verified=False`, `PASS: classically-scheduled event has no CalendarEventTelescopeLabel row`. |
| `src/templates/tom_calendar/partials/calendar.html` | Dashed-border + tooltip branches in both all-day and timed loops | ✓ VERIFIED | `is_verified == False` appears exactly twice; `2px dashed rgba(0, 0, 0, 0.65)` appears exactly twice; tooltip substring present in both. |
| `solsys_code/tests/test_calendar_template.py` | New test, renders calendar via Client, asserts dashed-border + tooltip markers | ✓ VERIFIED | 3 tests, all pass; covers fallback/verified/no-row across both render branches and the silenced-`DoesNotExist` (no-500) path. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `telescope_api_failed` (popped at sync line 606) | `is_verified=not telescope_api_failed` on sidecar row | `update_or_create` at line 635-637 | ✓ WIRED | Direct read of the source line; matches the must-have exactly. |
| `OneToOneField related_name='telescope_label_meta'` | `calendar.html` read | `event.telescope_label_meta.is_verified == False` in both render loops | ✓ WIRED | Confirmed in template; the `== False` idiom additionally protects the missing-row case from raising. |
| `event.telescope_label_meta.is_verified == False` | dashed border + `title=` attribute | Conditional branch on the outer (hoverable) div, both loops | ✓ WIRED | Same div carries both the style and the title attribute — Pitfall 4 (ship together) honored in both branches. |
| missing sidecar row / `is_verified=True` | falls through to unstyled/verified branch | Django's silenced `ObjectDoesNotExist` + `== False` comparison | ✓ WIRED | `test_calendar_renders_200_including_no_sidecar_row_events` proves no 500 for a no-row event; `test_dashed_border_count_matches_fallback_event_count_only` proves no spurious dashed border. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Migration is up to date | `python manage.py makemigrations solsys_code --check --dry-run` | "No changes detected in app 'solsys_code'", exit 0 | ✓ PASS |
| Phase 8 + adjacent test files pass | `python manage.py test solsys_code.tests.test_calendar_template solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_load_telescope_runs -v1` | 49/49 OK | ✓ PASS |
| Full `solsys_code` app suite green | `python manage.py test solsys_code -v1` | 138/138 OK | ✓ PASS |
| `reverse('calendar:calendar')` resolves | Python one-liner via Django shell context | Returns `/calendar/` | ✓ PASS |
| Quality gates clean on phase-touched files | `ruff check .` / `ruff format --check .` (repo-wide) | 2 pre-existing notebook issues + pre-existing `settings.py`/`.planning/quick/` formatting gaps, none in phase-08-touched files; confirmed identical against the pre-Phase-8 revision (`b89734b`) and `settings.py`'s last-touch commit (Phase 5, `adc5a61`) | ✓ PASS (no regressions introduced) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| DISPLAY-01 | 08-01-PLAN.md | Sidecar model + migration + sync write, no row for classical events | ✓ SATISFIED | Model, migration, write-site, and 4 tests (verified/fallback/no-churn/no-row) all confirmed live and passing. |
| DISPLAY-02 | 08-02-PLAN.md | Visual cue (dashed border) distinguishing fallback from verified | ✓ SATISFIED | Both render branches add the dashed border only for `is_verified == False`; rendering test confirms scoping. |
| DISPLAY-03 | 08-02-PLAN.md | Hover tooltip with verification detail | ✓ SATISFIED | `title=` attribute present on the same hoverable div in both branches; rendering test confirms tooltip substring present only for fallback events. |

No orphaned requirements: REQUIREMENTS.md's Traceability table maps DISPLAY-01/02/03 to Phase 8 only, and both phase plans together declare exactly these three IDs (`requirements: [DISPLAY-01]` in 08-01, `requirements: [DISPLAY-02, DISPLAY-03]` in 08-02) — full coverage, no gaps, no extras.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/templates/tom_calendar/partials/calendar.html` | 158-165 (all-day loop) | `[QUEUED]` title-prefix check is evaluated *before* `is_verified == False` in an `if/elif/else` chain, so a hypothetical event that is both `[QUEUED]` and fallback-labeled would render as `[QUEUED]` grey, losing the dashed-border/tooltip cue (timed loop has no such precedence conflict). | ℹ️ INFO (not a blocker for this phase) | Not currently reachable: `sync_lco_observation_calendar.py` line 472 hardcodes `telescope_api_failed=False` whenever `record.scheduled_start is None` (the exact condition that produces the `[QUEUED]` prefix), so today's only writer can never produce this combination — confirmed by direct code read. The plan text itself (`08-02-PLAN.md` line 81-82) explicitly scopes this out: "a queued + fallback event renders as `[QUEUED]` grey this phase — reconciling the two is Phase 9's concern." `08-REVIEW.md` (WR-01) flags this as a Warning, not Critical, for the same reason. No must-have in either plan's frontmatter or the ROADMAP success criteria requires the queued+fallback combination to be handled this phase. Recorded here for visibility, not as a phase-blocking gap. |

No debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) found in any of the 7 phase-modified files. No empty-implementation or hardcoded-stub patterns found in the new model, migration, sync-command write, or template branches.

### Human Verification Required

1. **Visual perceptibility of the dashed border on a rendered calendar page**
   - **Test:** Open `/calendar/` in a browser with at least one fallback-labeled event present (seed via the demo notebook's fallback fixture, or run `sync_lco_observation_calendar` against a record whose API call times out).
   - **Expected:** The fallback event's block shows a visibly dashed border distinct from neighboring solid-bordered/unstyled events, perceivable at a normal glance without opening the event or reading its (often truncated) title.
   - **Why human:** Automated tests prove the exact CSS literal (`2px dashed rgba(0, 0, 0, 0.65)`) and tooltip substring are present in the HTML and correctly scoped to fallback events only — that is necessary but not sufficient evidence for the phase goal's actual claim ("operators can tell... without reading title text"), which is a perceptual/UX claim a grep cannot adjudicate.

2. **Native tooltip appears on hover in a real browser**
   - **Test:** Hover over a fallback-labeled event block on the rendered calendar page.
   - **Expected:** The browser's native `title=` tooltip appears, showing the plain-language "estimate... could not be verified against the LCO API... coarse fallback label" sentence.
   - **Why human:** The test suite confirms the `title=` attribute and its content are present in the response HTML; it cannot exercise actual browser hover behavior.

### Gaps Summary

No blocking gaps. All 5 ROADMAP Phase 8 success criteria and all 3 requirement IDs (DISPLAY-01/02/03) are backed by real, passing, behaviorally-meaningful tests — not stubs. The model, migration, sync-command write, and template branches were all read directly and match the plan's must-haves verbatim. The full `solsys_code` test suite (138 tests) and the phase-specific subset (49 tests) both pass; `ruff check .`/`ruff format --check .` introduce no new issues beyond pre-existing, confirmed-predating ones. The only open item from code review (WR-01, the `[QUEUED]`-precedence ordering inconsistency between the all-day and timed branches) is explicitly out of scope for this phase per the plan's own text and is not currently reachable given the only writer's behavior — it is recorded as an info-level note for Phase 9 awareness, not a gap.

The reason this report's status is `human_needed` rather than `passed` is the visual/UX nature of success criteria 3 and 4 (DISPLAY-02/03): they are about an operator's *ability to perceive* a cue and *see* a tooltip on a rendered page, which automated HTML-string assertions support but cannot fully certify. Per the verification process's gate rules, any phase whose goal includes "discoverable... without reading title text" or "hovering... shows a tooltip" routes to human verification even when all programmatic checks pass.

---

*Verified: 2026-06-25T13:44:25Z*
*Verifier: Claude (gsd-verifier)*
