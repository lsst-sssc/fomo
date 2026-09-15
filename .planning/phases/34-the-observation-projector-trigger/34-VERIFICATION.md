---
phase: 34-the-observation-projector-trigger
verified: 2026-09-15T01:56:16Z
status: passed
score: 14/14 must-haves verified
covered_files:

  - ".planning/REQUIREMENTS.md"
  - ".planning/debug/resolved/34-updatestatus-receiver-attributeerror.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-01-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-01-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-02-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-02-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-03-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-03-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-04-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-04-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-05-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-05-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-06-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-06-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-07-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-07-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-UAT.md"
  - "CLAUDE.md"
  - "docs/notebooks.rst"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json"
  - "docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/apps.py"
  - "solsys_code/calendar_utils.py"
  - "solsys_code/campaign_attribution.py"
  - "solsys_code/management/commands/project_observation_calendar.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/templatetags/calendar_display_extras.py"
  - "solsys_code/tests/test_calendar_display_extras.py"
  - "solsys_code/tests/test_calendar_template.py"
  - "solsys_code/tests/test_calendar_utils.py"
  - "solsys_code/tests/test_campaign_attribution.py"
  - "solsys_code/tests/test_observation_projector.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_project_observation_calendar.py"
  - "solsys_code/tests/test_projector_demo_notebook.py"
  - "solsys_code/views.py"
  - "src/templates/tom_calendar/partials/calendar.html"
  - "src/templates/tom_calendar/partials/event_form.html"

covered_digest: "v1:sha256:e1be38552a6f58271368bdab4777f23f1f7d771d12241cf0577de9bd2542195d"
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: passed
  previous_score: 14/14
  trigger: "content-fingerprint re-stale only. Commit `24875bf` (`fix(35)`) touched `solsys_code/observation_projector.py`, which this phase's report covers, so the 2026-09-15T00:35:15Z digest no longer matched. The commit is Phase 35 scope -- it closes the advisory this phase's prior pass raised and routed to Phase 35."
  gaps_closed: []
  gaps_remaining: []
  regressions: []
  advisories_closed:
    - "Prior advisory 1 -- `receiver_on_record_save()` evaluated `instance.campaign_run_links.select_related('run')` in the `for` header outside any `try`, so a DB fault there escaped the receiver and propagated out of `ObservationRecord.save()`. Commit `24875bf` resolves the lookup into a plain list inside its own `try` (`observation_projector.py:647-658`), logging and degrading to no linked runs. Closed with a wired named regression test, `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection`, run green in isolation this pass."
gaps: []
deferred:

  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 34 scope note: 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Phase 37 = 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02). Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
advisory:

  - finding: "Wording correction to the prior pass's truth-5 evidence, not a code finding. The 2026-09-15T00:35:15Z report wrote 'longest observed is 16 characters'. Measured live this pass, the longest facility-url event title is 35 characters (e.g. `[O] FTS Didymos COJ 2026 Field #02`). PROJ-06's actual tested contract is narrower and is met: `test_marker_and_token_within_first_16_characters` asserts the marker and telescope token both appear within `title[:16]`, so the meaningful part survives a truncated month cell -- it never asserted a 16-character total title."
    category: other
    reason: "Recorded so a future reader measuring MAX(LENGTH(title)) does not conclude PROJ-06 regressed. Not caused by commit `24875bf` -- the developer database is byte-identical to the prior pass (`stat` `1789431410 1232896` unchanged), so these titles are the same rows the prior pass measured; only the prior report's summarising phrase was loose. Truth 5 is VERIFIED against the contract the test pins."
    evidence_status: "measured -- live read-only SQL, MAX(LENGTH(title)) = 35 over the 159 facility-url events; test body read at solsys_code/tests/test_observation_projector.py:154-200"
  - finding: "Carried forward unchanged from the prior pass: the debug doc's `CalendarEvent.modified == ObservationRecord.modified` signature for records 4378332 / 4378046 no longer holds live -- the 2026-09-14 operator sweep re-titled both with their resolved site token and bumped `CalendarEvent.modified` to 2026-09-14 23:06/23:07."
    category: other
    reason: "A dated Evidence entry (point-in-time), independently corroborated by the 22:39Z pass ~25 minutes before the sweep ran. The substantive half survives: both events still span their own receiver-derived observed blocks. The operator deliberately spent that baseline. Untouched by this pass's commit; re-stated only so it is not rediscovered."
    evidence_status: "carried forward -- database byte-identical to the prior pass, so the prior measurement stands unchanged"
  - finding: "Carried forward unchanged from the prior pass: the corrected debug doc's summary sentence reads, in isolation, as if the 2026-09-14 sweep followed the 0018 migration; the first sweep attempt in fact preceded it."
    category: other
    reason: "The entry's own parenthetical states the correct two-attempt sequence and the timestamps confirm it. Ambiguous phrasing in a summary line, not a factual error. No action required. Untouched by this pass's commit."
    evidence_status: "carried forward -- no artifact in this chain changed since the prior pass"
---

# Phase 34: The Observation Projector & Trigger — Verification Report (third re-verification)

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-15T01:56:16Z (HEAD `24875bf`, branch `issue37-telescope-runs-calendar`)
**Status:** passed (14/14 truths verified; 0 gaps; 0 human items outstanding)
**Re-verification:** Yes — fingerprint re-stale from a single Phase 35 commit. Supersedes the 2026-09-15T00:35:15Z report.

## What this pass was asked to establish

Exactly one commit landed since the prior pass touched a file this phase's report covers:

```
24875bf fix(35): F-34-1 -- guard the campaign_run_links lookup itself in
        receiver_on_record_save, not just the per-link reproject call
```

Phase 34's content fingerprint covers `solsys_code/observation_projector.py`, so the report went stale even though the change belongs to Phase 35. The question for this pass is narrow: **does this commit regress anything Phase 34 established?**

**Answer: no — and it strictly improves one of them.** The commit is the fix for the advisory the *previous* Phase 34 pass raised and routed to Phase 35. It closes that advisory with a wired regression test. Every Phase 34 must-have still holds, re-measured rather than cited.

## The commit under review

`git diff e3303e6..HEAD` touches three files: the prior VERIFICATION.md and two source files. The source change, read in full rather than taken from the message:

```python
# before                                     # after
                                             try:
                                                 links = list(instance.campaign_run_links.select_related('run'))
                                             except Exception as exc:  # noqa: BLE001 -- F-34-1/TRIG-02
                                                 logger.warning('linked-run lookup failed for observation_id=%r: %s', ...)
                                                 links = []

for link in instance.campaign_run_links      for link in links:
        .select_related('run'):
    if link.run is None: ...                     if link.run is None: ...
```

Three properties matter for Phase 34, and all three check out against the file as it stands (`observation_projector.py:631-672`):

1. **The success path is unchanged.** `list(qs)` evaluates the same queryset the `for` header already evaluated, then iterates the same rows in the same order. No filter, no ordering, no field selection changed. The only difference is that the rows are materialised before the loop instead of during it.
2. **Phase 34's own block is untouched and still runs first.** `project_record(instance)` is still called at line 637 inside its own `try` (lines 636-643), *before* the linked-run step. The base projection — the thing every Phase 34 must-have rests on — commits before any of this.
3. **TRIG-02's guarantee gets stronger, not weaker.** The prior pass demonstrated that a DB fault in the link query escaped the receiver and aborted `ObservationRecord.save()`. That hole is now closed, so the receiver as a whole honours "logged, never aborts the caller's save" rather than only its projector block.

The new test is a genuine negative regression test, not a tautology: it patches `campaign_run_links` to raise `OperationalError`, then asserts `record.save()` does **not** raise, that the record's status persisted, that the record's *own* calendar event still exists, and that the warning was logged. Run in isolation this pass:

```
$ python manage.py test solsys_code.tests.test_observation_projector_signals.TestLinkedRunReproject\
  .test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection
Ran 1 test in 5.165s
OK
```

That it is meaningful is established independently: the prior pass reproduced this exact failure deterministically on a scratch copy (`save() RAISED OperationalError`) using the same mechanism the test now pins.

## Goal Achievement

### Observable Truths

Truths 1-14 carry forward from the 2026-09-15T00:35:15Z pass. Under re-verification mode the prior pass's must-haves are reused verbatim; because the prior pass closed every item, this pass ran regression checks across all fourteen, with full re-verification concentrated on truths 1-3 (the ones the changed file could plausibly affect). The regression checks were live measurements and test runs, not citations.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Re-measured live this pass: **159 LCO/SOAR records, 159 facility-url events, 0 duplicate non-blank urls, 0 orphan facility-url events**. Status × marker cross-tab reproduces the prior pass exactly (`COMPLETED`→`[O]` 76, `CANCELED`→`[C]` 6, `WINDOW_EXPIRED`→`[X]` 26, `FAILURE_LIMIT_REACHED`→`[F]` 1, `PENDING`→`[Q]` 38 / `[S]` 12; sum 159) with **zero terminal-status records carrying a `[Q]`/`[S]` marker**. Sweep over a current-state copy: `unprojectable: 0` across all 159. |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | **Strengthened by this commit.** `apps.py` receiver wiring unchanged (all five `dispatch_uid`s intact). Behavioural re-run: the 8-module phase suite passes at **343 tests** (342 + the new regression test), `OK (skipped=1)` — including `test_updatestatus_narrows_the_event_with_no_command_run`, `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`. The never-abort guarantee now additionally holds for the adjacent link query — the gap the prior pass raised as advisory 1 is closed and pinned. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | Re-run against a scratch copy of the current developer database with the *changed* projector code: `Done (dry run). failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero` — byte-matching the prior pass. Namespace isolation re-measured live: `RUN:` urls **72**, `ALLOC:` urls **0**, blank-url **10** — unchanged. `TestDryRun`, `TestFailureIsolation`, `TestNamespaceIsolation` pass. |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ✓ VERIFIED | Archival evidence unchanged: 34-UAT.md Test 3 `result: pass`, independently corroborated from the live database by the 22:39Z pass. Re-proved at integration level against today's code by `test_updatestatus_narrows_the_event_with_no_command_run`, and at corpus level by `unprojectable: 0` over all 159 records. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `test_marker_and_token_within_first_16_characters` passes; test body read this pass — it asserts marker and telescope token both fall within `title[:16]` across all nine stage cases. Live titles conform (`[O] TFN-1m0 11P`, `[O] COJ-1m0 220P`). See advisory 1: the prior report's phrase "longest observed is 16 characters" was loose — full titles run to 35 characters — but the contract the test pins is the first-16 rule, and it holds. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Re-checked this pass: `manage.py help \| grep -c sync_lco_observation_calendar` → **0**; `management/commands/` holds `project_observation_calendar.py` + `sync_gemini_observation_calendar.py` and no LCO sync; `pre_executed/` holds only the two surviving notebooks; runbook carries **0** references to the retired command. |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | Templates unchanged (not in the commit's diff); `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label` pass. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | `calendar_display_extras.py` unchanged; `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. `_TERMINAL_PREFIXES` still covers both vocabularies. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | `event_form.html` unchanged; `TestObservationSeriesDecoration` and `test_modal_query_count_does_not_grow_with_group_size` pass. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✓ VERIFIED | Notebook and `sched06-baseline.json` byte-unchanged: `git status --porcelain` on both paths is empty, and the baseline's sha256 is still `453ae2ba…d353859c`. Guard module passes. The commit under review does not trigger a new paired-docs obligation — see "CLAUDE.md paired-docs check" below. |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | `write_event_meta()` unchanged; the four pinning tests pass. Live corroboration: every one of the 159 titles is `'[marker] <token> <target>'` with no group-derived text. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | Sweep against the current corpus reports `site_lookups: 0` — the read-back half, re-demonstrated with the changed code in place. The resolve half was exercised for real by the 2026-09-14 sweep (16 fresh lookups). `test_campaign_attribution.py` passes; both files unchanged. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ✓ VERIFIED | Declared `verification: backstop`; closed by directly observed operator behaviour in 34-UAT.md Test 1 (`result: pass`) — two overlapping `updatestatus` runs, 0 `AttributeError`, 0 `unprojectable`, 0 `OperationalError`. UAT artifact unchanged since the prior pass. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | Pinned by the real 2026-09-14 pre-migration sweep incident (an interrupted sweep in production that left all 14 processed records correct), by 34-UAT.md Test 3, and by the notebook's 33 → 0 convergence. Re-confirmed this pass: the dry run over the current corpus converges at `updated: 0`. |

**Score:** 14/14 truths verified (0 present-but-behavior-unverified, 0 abstained, 0 failed)

Truth 2 is behaviour-dependent — it asserts a cancellation/never-abort invariant that presence checks cannot see. It is marked VERIFIED on behavioural evidence: `test_raising_projector_does_not_block_a_save` and the newly added `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection` were both run green this pass, the latter in isolation by name.

### Regression check — did the commit break anything?

| Surface | Check | Verdict |
|---|---|---|
| Source diff scope | `git diff --stat e3303e6..HEAD` | ✓ Exactly 2 source files (`observation_projector.py`, `test_observation_projector_signals.py`) + the prior VERIFICATION.md. Nothing else. |
| Changed-code success path | Read `observation_projector.py:631-672` | ✓ `project_record()` still first and still guarded; link iteration semantically identical. |
| Phase 34 test suite | 8 modules, one run | ✓ `Ran 343 tests … OK (skipped=1)` — 342 prior + the 1 new regression test. Same outcome, no new skips or failures. |
| Consumers of the changed receiver | `test_allocation_projector_signals` + `test_allocation_projector` | ✓ `Ran 73 tests … OK`. The D-11 linked-run path the commit edits is unbroken for its own owner. |
| Sweep convergence with changed code | `--dry-run` on a current-state scratch copy | ✓ `created: 0, updated: 0, unchanged: 159, unprojectable: 0`. |
| Live corpus | read-only SQL | ✓ 159/159, 0 duplicates, 0 orphans, cross-tab identical to the prior pass. |
| Foreign namespaces | read-only SQL | ✓ `RUN:` 72, `ALLOC:` 0, blank-url 10 — unchanged. |
| Retired command | `manage.py help \| grep -c` | ✓ `0`. |
| Phase 34 docs artifacts | `git status --porcelain` + sha256 | ✓ Notebook and baseline byte-unchanged; baseline sha256 still `453ae2ba…d353859c`. |
| Working tree | `git status --porcelain solsys_code/ src/ docs/` | ✓ Only Phase 35's `reconcile_campaign_runs_demo.ipynb` dirty — same as the prior pass. |
| Requirements text | `git log --since` on `.planning/REQUIREMENTS.md` | ✓ No commits; all eleven mapped lines present and unchanged. |
| Lint / format gate | `pre-commit run ruff` / `ruff-format` on both changed files | ✓ Both `Passed`. |
| Developer DB | `stat -c '%Y %s'` | ✓ `1789431410 1232896` — byte-identical to the prior pass, and untouched by this verification. |

### CLAUDE.md paired-docs check

`solsys_code/observation_projector.py` is in CLAUDE.md's notebook map (paired with `project_observation_calendar_demo.ipynb`), so a behaviour change to it would oblige a notebook update in the same change. It does not apply here, and the reasoning is specific rather than convenient:

- The rule's trigger is a **behaviour** change of the listed kind — "new extraction logic, new parameters, new fixture shapes — not pure refactors or typo fixes". This commit adds no extraction logic, no parameter, and no fixture model; it wraps an already-evaluated query in the same guard every other step in the function already has.
- The notebook demonstrates the sweep's takeover and convergence. It contains **0** occurrences of `campaign_run_links`, and nothing in its executed output can change — confirmed empirically, since the sweep re-run this pass with the changed code produces the identical counter line.
- The runbook's only never-abort prose (lines 1384, 1413) is about the *sweep's* per-record failure isolation, not the receiver's link lookup. The change makes the existing documented guarantee more true, not stale.

No paired-docs obligation is breached.

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose vs. terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Both vocabularies already paint the correct ring. |

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | Prior report's truth-5 phrase "longest observed is 16 characters" is loose — live max title length is 35 chars. PROJ-06's tested contract is marker+token within `title[:16]`, and it holds. | other | A wording correction to a superseded report, not a code finding. Database byte-identical to the prior pass, so no data changed; recorded so a future `MAX(LENGTH(title))` measurement is not misread as a regression. |
| 2 | Carried forward: the debug doc's `CalendarEvent.modified == ObservationRecord.modified` signature for 4378332 / 4378046 no longer holds live. | other | A dated Evidence entry, corroborated 25 minutes before the sweep overwrote it. Substantive half (observed-block spans) survives. Untouched by this commit. |
| 3 | Carried forward: the corrected debug doc's summary sentence reads as if the sweep followed the migration; the first attempt preceded it. | other | The entry's own parenthetical states the correct sequence. Ambiguous phrasing, not a factual error. Untouched by this commit. |

**Prior advisory 1 is CLOSED, not carried forward.** The unguarded `campaign_run_links` lookup that the 2026-09-15T00:35:15Z pass reproduced deterministically is fixed in `24875bf` and pinned by a named regression test that passes. That advisory was routed to Phase 35; Phase 35 actioned it.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + three signal receivers | ✓ VERIFIED | **Changed this pass** (`24875bf`) and re-read in full at the changed region. Base projection still first and guarded; link lookup now guarded too. Imported by `apps.py`, the sweep command, `allocation_projector.py`, the template tags and 10 test modules. |
| `solsys_code/tests/test_observation_projector_signals.py` | Receiver behaviour pinned | ✓ VERIFIED | **Changed this pass** — one added negative regression test, run green by name. Module total 32. |
| `solsys_code/apps.py` | `ready()` wires the receivers | ✓ VERIFIED | Unchanged; all five `dispatch_uid`s intact (3 observation projector + 2 allocation projector). |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` used by both `record_time_window()` branches | ✓ VERIFIED | Unchanged; `TestCoerceScheduleDatetime` + `TestRecordTimeWindow` pass. |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep, zero required args | ✓ VERIFIED | Unchanged. Executed `--dry-run` against a current-state scratch copy with the changed projector: `updated: 0, unchanged: 159`. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series decoration | ✓ VERIFIED | Unchanged; all three tags still called from the two templates. |
| `solsys_code/campaign_attribution.py` | `OBSERVED_TELESCOPE_OBSCODES` bridge | ✓ VERIFIED | Unchanged; its test module passes. |
| `solsys_code/allocation_projector.py` | D-11 re-project target of the changed loop | ✓ VERIFIED | Unchanged; its two test modules pass (73 tests) against the changed receiver. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo with a real takeover | ✓ VERIFIED | Byte-unchanged; guard passes; no new pairing obligation from this commit. |
| `…/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical to committed state | ✓ VERIFIED | sha256 `453ae2ba…d353859c`, unchanged. |
| `solsys_code/tests/test_projector_demo_notebook.py` | Repo-level guard over the committed notebook | ✓ VERIFIED | Unchanged; passes within the suite run. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section, no stale sync content | ✓ VERIFIED | 0 references to the retired command; no prose invalidated by the commit. |
| `src/fomo_db.sqlite3` | SCHED-06 evidence database | ✓ VERIFIED | `1789431410 1232896` — byte-identical to the prior pass. Untouched by this verification (read-only URIs; sweep ran against a scratchpad copy). |
| `solsys_code/tests/*` (8 phase modules) | Behaviour pinned | ✓ VERIFIED | `Ran 343 tests … OK (skipped=1)`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` | ✓ WIRED | Unchanged code path, re-proved by the passing signals module (32 tests). |
| `SolsysCodeConfig.ready()` | the receivers | `.connect(weak=False, dispatch_uid=…)` | ✓ WIRED | Unchanged; five `dispatch_uid`s present. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | `unchanged: 159` on a corpus written by both writers is the proof the two agree. |
| sweep `pre_fields_hook` | `record.save()` → receiver → `project_record()` | one-time observed-site resolution | ✓ WIRED | `site_lookups: 0` on the read-back pass with the changed code. |
| `receiver_on_record_save()` | `allocation_projector.reproject_allocation_if_dispatched()` | `list(instance.campaign_run_links.select_related('run'))` under its own `try`, then per-link `try` | ✓ WIRED (guard now complete) | Upgraded from ⚠️ at the prior pass. Both the lookup and the per-link call are guarded; `test_allocation_projector_signals` (73 tests with `test_allocation_projector`) passes. |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` | ✓ WIRED | Display-time only; unchanged. |
| committed notebook evidence | every `manage.py test` run | `test_projector_demo_notebook.py` | ✓ WIRED | Guard runs in the default suite. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` | Yes — 159 live events each spanning their own stage-appropriate block | ✓ FLOWING |
| `telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']`, else `coarse_telescope_label()` | Yes — resolved tokens read back with `site_lookups: 0` | ✓ FLOWING |
| `stage_for()` → marker | marker letter | `record.status` | Yes — status × marker cross-tab is one-to-one across all 159 rows, no mismatched pair | ✓ FLOWING |
| `project_queryset()` counters | `action` | `preview_calendar_event_action(before, fields)` | Yes — discriminates rather than blanket-counting (`updated: 14` pre-sweep historically, `updated: 0` now) | ✓ FLOWING |
| `receiver_on_record_save()` linked-run loop | `links` → `link.run` | `list(instance.campaign_run_links.select_related('run'))` | Yes (tested, both success and fault paths); dormant live (0 `ALLOC:` events) | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant | Intentionally fixed (documented) | ✓ FLOWING (by design) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-34 test modules pass (8 modules, one run) | `python manage.py test solsys_code.tests.test_observation_projector …test_projector_demo_notebook` | `Ran 343 tests in 69.659s` / `OK (skipped=1)` | ✓ PASS |
| **New regression test passes in isolation** | `python manage.py test …TestLinkedRunReproject.test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection` | `Ran 1 test in 5.165s` / `OK` | ✓ PASS |
| **Changed receiver's own consumers unbroken** | `python manage.py test solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_allocation_projector` | `Ran 73 tests in 231.962s` / `OK` | ✓ PASS |
| **Sweep converges with the changed projector** | `FOMO_DATABASE_PATH=<scratch copy> python manage.py project_observation_calendar --dry-run` | `failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero` | ✓ PASS |
| One event per record, no duplicates, no orphans (live) | read-only SQL | `159 records / 159 facility-url events / 0 duplicate urls / 0 orphan events` | ✓ PASS |
| Zero terminal-status records carry a queued/scheduled marker (live) | read-only SQL, status × marker cross-tab | `0` | ✓ PASS |
| Foreign namespaces intact (live) | read-only SQL | `RUN:` 72, `ALLOC:` 0, blank-url 10 | ✓ PASS |
| Retired command really gone | `python manage.py help \| grep -c sync_lco_observation_calendar` | `0` | ✓ PASS |
| Baseline JSON byte-identical | `sha256sum` | `453ae2ba…d353859c` | ✓ PASS |
| Lint gate on changed files | `pre-commit run ruff --files …` | `Passed` | ✓ PASS |
| Format gate on changed files | `pre-commit run ruff-format --files …` | `Passed` | ✓ PASS |
| Developer DB untouched by this verification | `stat -c '%Y %s'` before/after | `1789431410 1232896` both times | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Requirements Coverage

Every requirement ID declared in this phase's plan frontmatter, cross-referenced against `.planning/REQUIREMENTS.md` (which has no commits since the prior pass — all eleven mapped lines verified present and unchanged this pass).

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1 — live: 159/159, 0 duplicates, 0 orphans |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truths 1, 2, 4 |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Cross-tab: `WINDOW_EXPIRED`→`[X]` 26, `CANCELED`→`[C]` 6, `FAILURE_LIMIT_REACHED`→`[F]` 1, none mismarked |
| PROJ-04 (title-stem clause) | **not declared in any plan's `requirements:`** | Series identity carried by real FKs, not text in the title | ✓ SATISFIED but ⚠️ **ORPHANED** | REQUIREMENTS.md L17 maps the title-stem clause to Phase 34, yet no plan frontmatter claims `PROJ-04` and the ROADMAP's phase requirement list omits it. Re-checked against all 7 plans' frontmatter this pass — still absent. Delivered anyway (truths 9, 11). Traceability gap in the plans, not a delivery gap — carried forward unchanged for the fourth pass. |
| PROJ-05 | 34-01, 34-02, 34-03, 34-07 | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation`; live `RUN:` 72 / `ALLOC:` 0 / blank-url 10 intact |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 — `test_marker_and_token_within_first_16_characters` passes across all nine stage cases. See advisory 1 for the prior report's loose phrasing. |
| SCHED-06 | 34-04, 34-05, 34-06, 34-07 | A user watches a record narrow over real nights with no command | ✓ SATISFIED | Truth 4; UAT Test 3 `pass`; baseline database byte-identical this pass |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; `apps.py` unchanged; `TestUpdateObservationStatusPath` |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED **(strengthened)** | Truth 2. The never-abort guarantee now covers the whole receiver, not only its projector block: `test_raising_projector_does_not_block_a_save` **and** the new `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection` both pass. The prior pass's known hole in this requirement's neighbourhood is closed. |
| TRIG-03 | 34-02, 34-04, 34-07 | Sweep command with `--dry-run`, failure isolation, and a paired pre-executed demo notebook | ✓ SATISFIED | Truth 3 — sweep re-run with the changed code converges at `updated: 0` |
| ANNOT-03 | 34-02, 34-04, 34-07 | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✓ SATISFIED | Truth 6 |

**All 10 declared requirement IDs accounted for.** One orphan (`PROJ-04`) mapped to Phase 34 in REQUIREMENTS.md but unclaimed by any plan — delivered, flagged for traceability only.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/observation_projector.py` | — | Debt markers (`TBD`/`FIXME`/`XXX`) in the changed file | ✓ **NONE** | Grepped explicitly: zero matches. The two `# noqa: BLE001` comments are justified in place against TRIG-02 and D-11, not suppression without reason. |
| `solsys_code/tests/test_observation_projector_signals.py` | — | Debt markers in the changed file | ✓ **NONE** | Zero matches. |
| `solsys_code/observation_projector.py` | 640 | Previously ⚠️→📋 Advisory: unguarded queryset evaluation in the D-11 loop header | ✓ **RESOLVED** | Fixed by `24875bf`; pinned by a passing named regression test. Was the prior pass's advisory 1. |
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | Pre-existing (`a8613bc8`, 2025-07-23); `views.py` not modified since the prior `verified:` timestamp, so new-scope under the re-verification evidence gate and not blocking. Classified identically by all four passes. |
| `docs/runbooks/telescope_runs_calendar.rst` | 1099 | `` ``TBD window`` `` | ℹ️ Info | Not a debt marker — a documented skip-reason *value* the reconciler emits, listed beside `not approved` / `unresolved site`. No unfinished-work semantics. |

Under the re-verification evidence gate, the only files modified since the prior `verified:` timestamp are `observation_projector.py` and `test_observation_projector_signals.py`; both were scanned at full scope and are clean of debt markers. No blocker was found on either, so the gate's evidence requirement is not exercised this pass.

### Human Verification Required

**None.** No truth is present-but-behavior-unverified, no truth abstained, and no item requires human observation. The two human items from the 2026-09-14T22:39:20Z report were closed and re-measured at the 00:35:15Z pass; nothing since has reopened them.

### Gaps Summary

**No gaps. No regressions. 14/14 truths verified.**

This pass existed for a mechanical reason — Phase 34's content fingerprint covers `observation_projector.py`, so a Phase 35 commit to that file re-staled a report that had already passed. The substantive question was whether that commit regressed anything Phase 34 established, and it did not.

I did not take the commit message's word for what changed. The diff is two source files; I read the changed region of `observation_projector.py` in full and confirmed the three properties Phase 34 depends on: `project_record()` is still the first thing the receiver does and is still inside its own guard, the linked-run iteration is semantically identical on the success path (`list(qs)` over the same queryset, same order, same fields), and nothing in Phase 34's own block was touched. Then I re-ran rather than reasoned: the 8-module phase suite at 343 tests (342 + the one added), the changed code's own consumers in `test_allocation_projector*` at 73 tests, and the sweep against a scratch copy of the current developer database, which converges at `created: 0, updated: 0, unchanged: 159` — byte-matching the prior pass with the new code in place.

The live corpus is unchanged and I re-measured it rather than citing it: 159 records, 159 facility-url events, zero duplicates, zero orphans, and a status × marker cross-tab that is one-to-one across every row, with zero terminal-status records carrying a queued or scheduled marker. The foreign namespaces the projector must never touch (`RUN:` 72, `ALLOC:` 0, blank-url 10) are intact. The developer database is byte-identical to the prior pass (`1789431410 1232896`) and was not written by this verification.

The commit is better than neutral for this phase. The previous pass raised one evidenced advisory — the `campaign_run_links` lookup was evaluated in the `for` header outside any `try`, so a database fault there escaped the receiver and aborted `ObservationRecord.save()`, which it had done for real during the 2026-09-14 pre-migration sweep. That advisory was routed to Phase 35, and this is Phase 35 actioning it. The lookup is now resolved under its own guard, and the new test is a real negative regression test rather than a tautology: it forces `OperationalError` from the lookup and asserts the save does not raise, the status persists, the record's own event still exists, and the warning is logged. It passes in isolation. TRIG-02's never-abort guarantee therefore now holds across the whole receiver rather than only its projector block, and the key link to `allocation_projector` moves from "wired, guard incomplete" to fully guarded.

One correction I owe to the record, found by measuring rather than copying: the prior report's evidence for truth 5 said "longest observed is 16 characters". That is not what the data shows — the longest facility-url title is 35 characters. The contract the test actually pins is narrower and is met: marker and telescope token both within `title[:16]`, so a truncated month cell still shows the meaningful part. The database is byte-identical to the prior pass, so this is a loose phrase in a superseded report, not a regression — but I have recorded it as an advisory so the next reader who runs `MAX(LENGTH(title))` does not mistake it for one.

I also checked the CLAUDE.md paired-docs rule rather than assuming it was inapplicable, since `observation_projector.py` is in the notebook map. It is not triggered: the change adds no extraction logic, parameter, or fixture model; the notebook contains no `campaign_run_links` content and its executed output provably cannot change, since the sweep re-run with the new code produces the identical counter line; and the runbook's only never-abort prose is about the sweep's per-record isolation, which this makes more true rather than stale.

The phase goal — one writer for observation-backed nights, drawn and kept current with no operator command — holds in the codebase and in the live database today.

---

_Verified: 2026-09-15T01:56:16Z_
_Verifier: Claude (gsd-verifier)_
