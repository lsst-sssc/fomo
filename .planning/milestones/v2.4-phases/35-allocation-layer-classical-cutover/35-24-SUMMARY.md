---
phase: 35-allocation-layer-classical-cutover
plan: 24
subsystem: allocation-projector
tags: [allocation-projector, calendar-event, provenance-token, sha256-fingerprint, tdd, gap-closure]

# Dependency graph
requires:
  - phase: 35 (plan 35-23)
    provides: "remint_declined counter split, declined-remint fall-through, retirement-branch guard"
provides:
  - "A v3 provenance token carrying a site-position fingerprint (lat/lon/altitude/timezone), closing the round-5 verifier's escalated decision: an in-place Observatory correction now re-mints"
  - "_site_position_fingerprint() and _site_provenance_differs() -- two new pure-read helpers"
  - "A dark-window-line refresh on the plain-update path for a fully-set sub-night pair whose site position moved (WR-05)"
  - "A qualified, honest Cost bound: statement (WR-07) and finished WR-08 documentation (model docstring, is_verified verbose_name/help_text, _remint_decline_reason() cross-reference)"
affects: ["35-25 (wave 3: operator runbook + reconcile_campaign_runs_demo.ipynb regeneration, still owes this plan's and 35-23's behaviour changes)"]

# Actuals (#2632)
actuals:
  tokens: 21078
  tasks: 3
  commits: 5
plan_head_before: d75cf4d542a98f4d9d867c3487999b6e668b2e26

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fixed-width fingerprint over unbounded position data: _site_position_fingerprint() truncates a SHA-256 digest to 16 hex characters, so the worst-case token width is a constant regardless of the underlying lat/lon/altitude magnitude -- the worst-case-width test needed no extreme coordinate values, only a real Observatory row for the fingerprint half to resolve against."
    - "Component-wise token comparison replaces whole-string equality once a token carries more than one boundary-relevant input: _span_needs_remint() step 3 now distinguishes 'an input moved' (fingerprint-only difference, routes to resolution) from 'the boundary moved' (sub-night/site_id difference, routes to re-mint) -- the same distinction _site_provenance_differs() reuses for the dark-window-only refresh."
    - "A stated, tested exception to a locked decision, at the call site: the dark-window refresh's comment quotes D-13's clause verbatim, states the transition-only exception's bound, and names the test (TestNoSunEventRecompute) that proves the exception is bounded rather than trusting the comment alone."

key-files:
  created: []
  modified:
    - solsys_code/models.py
    - solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py
    - solsys_code/allocation_projector.py
    - solsys_code/tests/test_allocation_projector.py

key-decisions:
  - "The escalated decision (35-VERIFICATION.md 'Human Verification Required' #1): the site component of the provenance token is promoted from identity alone (site_id) to identity-plus-position (site_id + a 16-hex-character SHA-256 fingerprint of lat/lon/altitude/timezone). site_id is KEPT, not replaced -- it keeps a stored token legible in a log line and preserves the documented behaviour that a site SWAP re-mints, at the accepted cost of a redundant re-mint when two Observatory rows share byte-identical position."
  - "A fingerprint-only difference routes to _span_needs_remint() step 4's existing resolution branch (one real sun_event() call, tolerance-compared) rather than to an outright re-mint. An input moving is not the same fact as a boundary moving; a one-metre altitude correction must not destroy and re-create every night at that site."
  - "WR-05: a fully-set sub-night pair never reaches the token at all (step 2's short-circuit, unmoved) -- so the ONLY site-derived field a correction can still reach is the event's stored dark-window line, refreshed by a new, narrowly-scoped exception to D-13 on the plain-update path, bounded to exactly one sun_event(kind='dark') call per night per site correction and never in a preview."
  - "WR-07: the 'once ever' cost-bound claim is qualified rather than kept: a declined-and-unrecorded night resolves once PER SWEEP, indefinitely (accepted, bounded, with the rejected false-provenance alternative named); a --dry-run preview repeats on every invocation (WR-01, separately open, not claimed fixed)."
  - "WR-08: is_verified's model docstring, verbose_name, help_text and _remint_decline_reason()'s own docstring now all state the re-mint veto AND that it does not veto a retirement (plan 35-23's CR-05 decision) -- consistently, in the words plan 35-25's runbook section must match verbatim."
  - "The cross-timezone set/set case (WR-05's third named case) was RUN, not assumed: moving a Chile-fixed 23:00/05:00 UTC sub-night window to Sydney inverts the resolved span for this fixture, matching the review's own hand-trace exactly. No code change was needed or made (prohibition 4) -- the existing CR-03 compute-before-destroy ordering already keeps the night's event intact when night_bounds() raises."

requirements-completed: []  # See 'Requirements gate' note below -- all 4 are shared with plan 35-25 (not yet executed) and blocked by requirements.ready-ids.

coverage:
  - id: D1
    description: "The escalated decision closed: an in-place Observatory lat/lon/altitude/timezone correction, with run.site never reassigned, now re-mints every night already projected at that site to the corrected position's real sun_event() values (new primary key, retired==1/created==1) instead of reading as unchanged forever. Reproduces the round-5 verifier's exact probe transcript as the fixture's provenance."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestObservatoryCorrectionRemints (1 test)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A position correction that does not move the sun events beyond the one-minute tolerance does NOT re-mint -- it resolves once, re-records a current-format v3 token, and reports unchanged; a site SWAP between two identically-positioned Observatory rows still re-mints on site_id, the accepted cost of keeping site_id in the token."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSiteChangeRemints, TestMintInputInvariant (3 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "WR-05 closed: a same-timezone site correction on a fully-set sub-night run refreshes the event's dark-window line and records a current token, at zero astropy cost on the next sweep and zero in a preview (with the one-count preview-over-reports-updated divergence pinned by its own test); the cross-timezone case's actual outcome (an inverted-span ValueError, night's event surviving) is run and pinned, not assumed."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSetWindowSiteCorrection (6 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "WR-07 closed as a bounded, tested acceptance: a declined-and-unrecorded night resolves exactly once per sweep across two consecutive sweeps (one sun_event(kind='sun') call, both warnings, each time), and no provenance token is ever recorded for boundaries that were not re-minted."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestDeclinedNightResolutionCostIsBounded (1 test)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The format contract's edges are named tests, not assumed conventions: a v2|-format token and a current-version token with the wrong part count both read as unrecorded and resolve through the identical bounded branch; the worst-case token (now including the fingerprint) measures 61 characters against the field's max_length=128."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestProvenanceTokenFormat (9 tests)"
        status: pass
    human_judgment: false
  - id: D6
    description: "WR-08 closed on the model and in the projector: minted_sub_night_window's max_length widened to 128 via a schema-only migration; is_verified's verbose_name/help_text state the re-mint veto and its retirement non-veto; CalendarEventMeta's class docstring and _remint_decline_reason()'s own docstring both cross-reference the field's full meaning."
    requirement: ALLOC-01
    verification:
      - kind: other
        ref: "python manage.py makemigrations solsys_code --check --dry-run (no drift); grep of the exact verbose_name/help_text strings in the generated migration"
        status: pass
    human_judgment: false
  - id: D7
    description: "ALLOC-02's site-local night promise and ALLOC-04's classical-loader idempotency (test_load_telescope_runs.py, unedited) hold at a corrected site position, and ALLOC-05's cutover sequencing is unaffected -- pinned by the seven-module regression gate rather than asserted."
    requirement: ALLOC-02
    verification:
      - kind: integration
        ref: "python manage.py test over the 7 named modules: 324 tests, OK"
        status: pass
    human_judgment: false

# Metrics
duration: ~75min
completed: 2026-09-16
status: complete
---

# Phase 35 Plan 24: Site-Position Fingerprint, WR-05 Dark-Window Refresh, and the Bounded WR-07/WR-08 Documentation Summary

**A `v3` provenance token now carries a 16-character SHA-256 fingerprint of the site's `lat`/`lon`/`altitude`/`timezone` alongside `site_id`, so an in-place `Observatory` correction re-mints instead of reading as `unchanged` forever -- closing the round-5 verifier's escalated decision plus WR-05, WR-07 and WR-08.**

## Performance

- **Duration:** ~75 min
- **Started:** 2026-09-16T18:10Z (approx)
- **Completed:** 2026-09-16T19:25Z (approx)
- **Tasks:** 3 (1 tracer + 2 TDD)
- **Files modified:** 4

## Accomplishments

- **The escalated decision closed** (35-VERIFICATION.md "Human Verification Required" #1; 35-UAT.md test 4): `_PROVENANCE_TOKEN_VERSION` bumped `v2` -> `v3`; new `_site_position_fingerprint(run)` returns `'none'` for a null site or the first 16 hex characters of a SHA-256 digest over `repr(lat), repr(lon), repr(altitude), repr(timezone)`; `_sub_night_provenance_token()` now returns a five-part token (`version|site_id|fingerprint|start|end`); `_span_needs_remint()`'s step 3 is a component-wise comparison -- a sub-night or `site_id` difference re-mints immediately, a fingerprint-only difference falls through to step 4's existing resolution branch (one `sun_event()` call, tolerance-compared) instead of an outright re-mint.
- **`TestObservatoryCorrectionRemints` reproduces the verifier's probe exactly** and proves it now re-mints: editing the SAME `Observatory` row's `lat`/`lon`/`altitude`/`timezone` in place (never reassigning `run.site`) produces `retired==1/created==1`, a new primary key, and both boundaries equal to the corrected position's live `sun_event()` values.
- **WR-05 closed:** new `_site_provenance_differs(run, existing)` (pure read, returns False for any untrusted token) lets `project_allocation()`'s plain-update path refresh a fully-set run's dark-window line when the site's recorded provenance proves it moved -- the ONE site-derived field a correction can still reach on that path, since the boundaries themselves are pinned by step 2's short-circuit. Bounded to one `sun_event(kind='dark')` call per night per site correction, never in `--dry-run`, with a pinned one-count preview-over-reports-`updated` divergence and a pinned cross-timezone outcome (an inverted-span `ValueError`, the night's event surviving).
- **WR-07 closed as a bounded, tested acceptance:** the `Cost bound:` docstring paragraph now names both escapes from "once ever" -- a declined night resolves once per sweep, indefinitely (accepted, with the rejected false-provenance alternative stated), and `--dry-run` repeats every invocation (WR-01, separately open).
- **WR-08 closed on the model and in the projector:** `is_verified`'s `verbose_name`/`help_text` and the class docstring state the re-mint veto and its retirement non-veto; `_remint_decline_reason()`'s own docstring cross-references the model. The runbook half is plan 35-25's.
- **Migration `0021_alter_calendareventmeta_is_verified_and_more.py`** -- exactly the filename this plan predicted -- contains two schema-only `AlterField` operations and no `RunPython`.

## Task Commits

Each task committed atomically. Task 1 is a tracer (implementation + test in one commit, per its `type="tracer"` declaration -- not TDD). Tasks 2 and 3 are TDD (`tdd="true"`) and produced RED -> GREEN pairs:

1. **Task 1 (tracer): carry the site's position in the token** -- `8e3a60b` (feat)
2. **Task 2 RED: add failing `TestSetWindowSiteCorrection`** -- `8c54266` (test)
2. **Task 2 GREEN: refresh the dark-window line for a fully-set site correction** -- `71ce9a5` (feat)
3. **Task 3 RED: widen the format contract's tests, pin the bounded declined-night cost** -- `e4cb63a` (test)
3. **Task 3 GREEN: qualify the cost bound, finish WR-08's documentation** -- `e8ee1ce` (feat)

**Plan metadata:** (this commit, docs)

## Files Created/Modified

- `solsys_code/models.py` -- `minted_sub_night_window` widened to `max_length=128`; `is_verified`'s new `verbose_name`/`help_text` (exact strings below); class docstring's two relevant paragraphs corrected.
- `solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py` -- two schema-only `AlterField` operations, no `RunPython`.
- `solsys_code/allocation_projector.py` -- `_PROVENANCE_TOKEN_VERSION` bumped to `v3`; `_site_position_fingerprint()` and `_site_provenance_differs()` (new); `_sub_night_provenance_token()` and `_span_needs_remint()` rewritten/extended; `project_allocation()`'s plain-update path gains the dark-window refresh; `_remint_decline_reason()`'s docstring gains its WR-08 sentence.
- `solsys_code/tests/test_allocation_projector.py` -- `TestObservatoryCorrectionRemints`, `TestSetWindowSiteCorrection` (6 tests), `TestDeclinedNightResolutionCostIsBounded`, extensions to `TestProvenanceTokenFormat` (2 new cases + worst-case-width fixture fix) and `TestMintInputInvariant` (1 new case + tripwire extension); Rule 1 fixes to `TestSiteChangeRemints`'s two hardcoded `'v2|'` literals.

## `is_verified`'s exact shipped strings (for plan 35-25's runbook section to match verbatim)

**`verbose_name`:**
```
Whether the telescope label was live-verified against the LCO API (unchecking also vetoes an automated re-mint)
```

**`help_text`:**
```
Setting this False permanently prevents the allocation projector from correcting the boundaries of this night: an automated re-mint is declined and reported under remint_declined. It does not prevent the night being retired when a linked observation places a block on it. See the runbook section on remint_declined.
```

## RED Failure Evidence (TDD)

**Task 1 (tracer, not TDD -- no RED/GREEN gate applies).** The "before" evidence for `TestObservatoryCorrectionRemints` is the round-5 verifier's own probe transcript, reproduced verbatim in the test's docstring and confirmed by this plan's own run against the corrected code (same timestamps, now on the OTHER side of the fix):

```
PROBE token= v2|1|none|none
PROBE before start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE result= ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0, ...)
PROBE after  start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE true corrected sunset/sunrise= 2026-07-09 07:20:39  2026-07-09 20:57:12
PROBE same pk? True
```
Running `TestObservatoryCorrectionRemints` against the *fixed* tree logs exactly the same "stored boundary ... disagrees beyond tolerance" warning at the same timestamps, then proceeds to re-mint (`retired=1/created=1`, new pk) -- proving the fix closes precisely the gap the verifier found.

**Task 2 RED (`TestSetWindowSiteCorrection`, verified against the tree as committed by Task 1, before Task 2's implementation):**
```
test_same_timezone_correction_on_a_set_window_run_refreshes_the_dark_window_line:
  AssertionError: 0 != 1   (result.updated)
test_dry_run_parity_and_zero_astropy_cost_for_the_correction:
  AssertionError: 0 != 1   (dry_result.updated)
test_preview_may_over_report_updated_by_one_on_a_site_correction:
  AssertionError: 0 != 1   (dry_result.updated)
```
The other three cases (idempotence, the null/half-null control, and the cross-timezone pinned-outcome case) already held pre-fix -- they are controls proving the OTHER paths are unaffected, not RED cases for this feature.

**Task 3 RED (`TestDeclinedNightResolutionCostIsBounded`, the `TestProvenanceTokenFormat` extensions, and `TestMintInputInvariant`'s new case):** all 14 pass immediately against the tree as it stood after Tasks 1 and 2. This is an *expected* "unexpected GREEN," not a process miss: Task 3's own `<behavior>` list is fully implemented by Tasks 1 and 2 already; Task 3's remaining, genuinely new work is the docstring truth-telling committed separately as the GREEN half (`e8ee1ce`). The three pre-existing `TestProvenanceTokenFormat` cases this task's commit also fixes (`test_null_token...`, `test_empty_string_token...`, `test_pre_release_token...`) WERE genuinely RED after Task 1's version bump -- their hardcoded `'v2|'` literals failed with `AssertionError: False is not true` until corrected to `'v3|'` (Rule 1 deviation, documented below).

## Worst-Case Token Measurement

Constructed directly (`_sub_night_provenance_token()` on an unsaved `CampaignRun` with a real `Observatory` row carrying `pk=999999999`, a microsecond-valued sub-night pair):
```
v3|999999999|5884a60fe2946a56|23:59:59.999999|00:00:00.999999
```
**Length: 61 characters**, against `CalendarEventMeta._meta.get_field('minted_sub_night_window').max_length == 128`. The position fingerprint is a FIXED 16-hex-character width regardless of the underlying coordinate magnitude (`_site_position_fingerprint()`'s own docstring), so this measurement is stable for any real `Observatory` row, not just this one.

## Cross-Timezone Set/Set Case -- Observed Outcome

Run against the real tree, not assumed (per prohibition 4, no code change was made or needed):

```
Allocation night_bounds inverted for run pk=1 night=2026-07-09: start=2026-07-09 23:00:00+00:00 >= end=2026-07-09 05:00:00+00:00 (night_start_utc=23:00:00, night_end_utc=05:00:00).
```

Moving a fully-set `23:00`/`05:00` UTC sub-night window from La Silla (`America/Santiago`) to Siding Spring (`Australia/Sydney`) resolves BOTH boundaries onto the SAME UTC date for Sydney's night-span band (offset > +6, entirely inside its own UTC date), with the resolved end (`05:00`) before the resolved start (`23:00`) -- an inverted span. `night_bounds()`'s guard raises from inside `_mint_fields()`, called BEFORE `existing.delete()` in the re-mint branch (plan 35-20's CR-03 compute-before-destroy ordering), so the failed `reconcile_run()` call leaves the night's existing event intact. This matches 35-REVIEW.md's own hand-trace exactly. Pinned by `TestSetWindowSiteCorrection.test_a_set_window_moved_across_timezones_has_a_pinned_outcome`.

## Declined-Night Bounded-Cost Measurement (WR-07)

Two consecutive real `reconcile_run()` calls on a confirmed, unrecorded-provenance, stale-beyond-tolerance night:

| Sweep | `sun_event()` calls | Staleness warnings | Decline warnings | `remint_declined` | `minted_sub_night_window` after |
|-------|---------------------|---------------------|-------------------|--------------------|----------------------------------|
| 1 | 1 | 1 | 1 | 1 | still `None` |
| 2 | 1 | 1 | 1 | 1 | still `None` |

Confirms: the repetition is a bounded, per-sweep constant (never zero, never two), and no provenance token is ever recorded for boundaries that were not re-minted.

## D-13 Narrowing (Named Deviation from a Locked Decision's Literal Terms)

D-13's clause, quoted verbatim (its own words): *"`sun_event()` (both `'sun'` and `'dark'`) runs only for a night being created or re-minted."*

The comment shipped at the refresh call site (`solsys_code/allocation_projector.py`, inside `project_allocation()`'s plain-update `else:` branch):

```
# D-13 (verbatim): "`sun_event()` (both `'sun'` and `'dark'`) runs only for
# a night being created or re-minted." This call is the ONE stated exception
# to that clause. The night reaching this line is neither created nor
# re-minted; the call fires ONLY on the transition where a recorded
# current-format token proves the site component moved AND both sub-night
# fields are set; it is bounded to exactly one call per night per site
# correction, because this same sweep records the current token below (see
# the comment on that write); and it never fires in a preview (the `dry_run`
# half of this condition) or on an idempotent sweep (the flag is False
# whenever the site component has not moved). What is NOT narrowed: D-13's
# purpose -- no `sun_event()` call on an idempotent re-reconcile of an
# existing night, the todo
# `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`
# asked for -- is untouched, and `TestNoSunEventRecompute` staying green
# unedited is what proves it. WR-05 is the reason this exception exists at
# all: for a fully-set pair the boundaries are correctly pinned, so the
# dark-window line is the only site-derived field a correction can still
# reach.
```

Measured bound, from `TestSetWindowSiteCorrection`: **one `sun_event(kind='dark')` call on the transition sweep** (`test_same_timezone_correction_on_a_set_window_run_refreshes_the_dark_window_line`), **zero calls on the immediately following sweep** (`test_idempotent_reconcile_after_the_refresh_makes_zero_sun_event_calls`, `mock_sun_event.assert_not_called()`), and **zero calls in a `--dry-run` preview** of the same correction (`test_dry_run_parity_and_zero_astropy_cost_for_the_correction`).

**`TestNoSunEventRecompute` and `TestSubNightWindow` were left unedited** -- confirmed by `git diff` against both classes showing no changes -- which is the constraint that decides whether this narrowing is the right fix: D-13's astropy budget survives the `v3` bump, the fingerprint, and the refresh, exactly as before.

## Decisions Made

See `key-decisions` in frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 -- stale version literal] `TestSiteChangeRemints`'s two `'v2|'` token-prefix assertions**
- **Found during:** Task 1's own full-module regression check, immediately after the `v2`->`v3` bump.
- **Issue:** `_PROVENANCE_TOKEN_VERSION` is bumped by this plan's own Task 1 (the whole point of the escalated fix), so a hardcoded `'v2|'` prefix check on a freshly-recorded token fails -- `AssertionError: False is not true`. `TestSiteChangeRemints` is named in prohibition 9's "must stay green unedited" list, but a version-literal assertion is inherently a casualty of a version bump that list's own siblings (`TestProvenanceTokenFormat`) are explicitly permitted to update for.
- **Fix:** both literals updated `'v2|'` -> `'v3|'`. The class's actual behaviour under test (a site swap re-mints) is unchanged and still asserted identically.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`
- **Verification:** `TestSiteChangeRemints` passes; full seven-module regression (324 tests) OK.
- **Committed in:** `8e3a60b` (Task 1 commit)

**2. [Rule 1 -- stale version literals, three cases] `TestProvenanceTokenFormat`'s unrecorded-value cases**
- **Found during:** Task 1's full-module regression check (same root cause as deviation 1).
- **Issue:** `test_null_token_reads_as_unrecorded_and_resolves_once`, `test_empty_string_token_reads_as_unrecorded_and_resolves_once` and `test_pre_release_token_reads_as_unrecorded_and_resolves_once` each asserted the freshly-recorded token started with `'v2|'`.
- **Fix:** all three updated to `'v3|'`. Prohibition 9 explicitly names this class's "unrecorded-value cases" as the one exception permitted to be EXTENDED (never weakened) for the new version marker -- this is exactly that extension, done as part of Task 3's own remit rather than left stranded from Task 1.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`
- **Verification:** all three pass; `TestProvenanceTokenFormat` (9 tests) OK.
- **Committed in:** `e4cb63a` (Task 3 RED commit)

**3. [Rule 1 -- broken fixture] `test_worst_case_token_fits_within_the_declared_max_length`**
- **Found during:** Task 1's full-module regression check.
- **Issue:** `_sub_night_provenance_token()` now reads `run.site` (not just `run.site_id`) to build the fingerprint. The pre-existing fixture set `worst_case_run.site_id = 999999999` directly on an unsaved `CampaignRun` with no matching `Observatory` row; the fingerprint helper's `run.site` dereference raised `Observatory.DoesNotExist` (setting `.site_id` directly, rather than assigning `.site`, does not leave a stale cached relation object behind in this Django version to paper over the missing row).
- **Fix:** the fixture now creates a REAL `Observatory` row with an explicit huge primary key (`pk=999999999`, valid for an integer PK at create time) and assigns it via `CampaignRun(site=worst_case_site, ...)`. Since the position fingerprint is a fixed 16-hex-character width regardless of the underlying coordinates, this changes nothing about what the test measures.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`
- **Verification:** the test passes; measured worst-case length 61 <= `max_length` 128.
- **Committed in:** `e4cb63a` (Task 3 RED commit)

No fixture in `test_cutover_classical_allocations.py` broke (confirmed by the full seven-module regression passing unedited); that file required no Rule 1 repair this round.

---

**Total deviations:** 3 auto-fixed (all Rule 1, all direct consequences of the deliberate `v2`->`v3` version bump). **Impact on plan:** none beyond the necessary literal/fixture corrections -- no scope creep, no behaviour change to any class besides the corrected assertions themselves.

## Issues Encountered

None beyond the deviations above.

## Requirements Gate

`requirements.ready-ids` reports **0/4 ready** for `ALLOC-01`, `ALLOC-02`, `ALLOC-04`, `ALLOC-05`: all four are shared with plan 35-25 (wave 3, not yet executed), which also declares them in its own `requirements` frontmatter. Per the shared-ID gate (#2388), none are marked complete by this plan -- they will be marked once 35-25's own `update_requirements` step runs and finds all declaring plans finished.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

**The round is NOT complete at the end of this plan.** Plan 35-25 (wave 3) still owes the operator runbook's site-correction/`remint_declined`-qualification sections and a re-executed `reconcile_campaign_runs_demo.ipynb` for both this plan's behaviour changes and plan 35-23's -- until it lands, both committed artifacts describe the pre-`v3` behaviour. `docs/` was correctly left untouched by this plan (prohibition 7), and `solsys_code/admin.py`, `solsys_code/campaign_reconciler.py`, `solsys_code/campaign_views.py`, `solsys_code/management/` and `src/fomo_db.sqlite3` were all correctly left untouched (prohibitions 3, 5, 6, 8) -- verified by the clean `git status --short` above.

**Stated residuals, not gaps** (unchanged from the plan's own `<verification>` block): (a) WR-01's dry-run repetition is untouched and explicitly not claimed fixed; (b) WR-02's preview-raises-`ValueError` surface is unchanged and deliberately not widened; (c) a 16-character truncated digest could in principle collide, costing a missed re-mint on one position change, which the next genuine change still catches -- stated in `_site_position_fingerprint()`'s own docstring; (d) the operator runbook and the committed demo notebook still describe the pre-`v3` behaviour until plan 35-25 lands.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-16*

## Self-Check: PASSED

- All 5 key files (4 modified + this SUMMARY) confirmed present on disk.
- All 5 task/RED/GREEN commit hashes (`8e3a60b`, `8c54266`, `71ce9a5`, `e4cb63a`, `e8ee1ce`) confirmed in `git log`.
- Re-ran every plan-level `<acceptance_criteria>` and `<verification>` item: all pass (see body above for measured evidence).
- Seven-module regression: 324 tests, OK (>= plan 35-23's recorded 313).
- Both ruff hooks: Passed. `makemigrations --check --dry-run`: no drift. `git status --short` over the prohibited paths: empty.
