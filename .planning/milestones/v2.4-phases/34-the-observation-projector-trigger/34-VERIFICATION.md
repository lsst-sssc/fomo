---
phase: 34-the-observation-projector-trigger
verified: 2026-09-15T02:14:56Z
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

covered_digest: "v1:sha256:f55d4734974beb09447b9be5b754c70c2c24adfe71a782749fa93b6e7cad0638"
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: passed
  previous_score: 14/14
  trigger: "content-fingerprint re-stale only. Commit `5b66202` (`docs(34)`) changed three planning documents -- `34-01-PLAN.md`, `34-03-PLAN.md` (both in this phase's covered_files) and `.planning/ROADMAP.md` -- to close the PROJ-04 traceability orphan this report has flagged for four consecutive passes. No source file changed: `git diff 588cf22..HEAD -- solsys_code/ src/ docs/` is empty."
  gaps_closed: []
  gaps_remaining: []
  regressions: []
  traceability_closed:
    - "PROJ-04 ORPHAN CLOSED. The prior four passes recorded PROJ-04 as `✓ SATISFIED but ⚠️ ORPHANED` -- REQUIREMENTS.md L106 maps its shared-title-stem clause to Phase 34, but no plan's `requirements:` frontmatter claimed it and the ROADMAP's Phase 34 requirements line omitted it. Commit `5b66202` adds `PROJ-04` to `34-01-PLAN.md` (write side) and `34-03-PLAN.md` (display side) and to the ROADMAP line. Verified this pass by programmatic three-way reconciliation: the union of all seven plans' frontmatter, the ROADMAP Phase 34 requirements line, and the set of REQUIREMENTS.md traceability rows mapping to Phase 34 are now the *identical* 11-element set, with `orphans: NONE` and `unmapped: NONE`. The attribution is independently corroborated by 34-VALIDATION.md (written 2026-09-12, predating the fix), which already mapped PROJ-04 to task 34-01-01 in plan 01 and task 34-03-02 in plan 03 -- exactly the two plans the commit amends."
  advisories_closed: []
gaps: []
deferred:

  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 34 scope note (unchanged by this commit -- the diff touches only the Requirements line): 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Phase 37 = 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02). Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
advisory:

  - finding: "Carried forward from the prior pass: PROJ-06's tested contract is 'marker and telescope token both within `title[:16]`', not a 16-character total title. Live maximum facility-url title length is 35 characters (e.g. `[O] FTS Didymos COJ 2026 Field #02`)."
    category: other
    reason: "Restated so a future reader measuring MAX(LENGTH(title)) does not conclude PROJ-06 regressed. The 2026-09-15T00:35:15Z report's phrase 'longest observed is 16 characters' was loose; truth 5's evidence below now states the contract correctly. Untouched by this pass's commit -- the developer database is byte-identical (`stat` `1789431410 1232896`), so these are the same rows."
    evidence_status: "carried forward -- database byte-identical to the prior pass, so the prior measurement stands unchanged"
  - finding: "Carried forward unchanged: the debug doc's `CalendarEvent.modified == ObservationRecord.modified` signature for records 4378332 / 4378046 no longer holds live -- the 2026-09-14 operator sweep re-titled both with their resolved site token and bumped `CalendarEvent.modified` to 2026-09-14 23:06/23:07."
    category: other
    reason: "A dated Evidence entry (point-in-time), independently corroborated by the 22:39Z pass ~25 minutes before the sweep ran. The substantive half survives: both events still span their own receiver-derived observed blocks. The operator deliberately spent that baseline. Untouched by this pass's commit, which changes no source and no data."
    evidence_status: "carried forward -- database byte-identical to the prior pass"
  - finding: "Carried forward unchanged: the corrected debug doc's summary sentence reads, in isolation, as if the 2026-09-14 sweep followed the 0018 migration; the first sweep attempt in fact preceded it."
    category: other
    reason: "The entry's own parenthetical states the correct two-attempt sequence and the timestamps confirm it. Ambiguous phrasing in a summary line, not a factual error. No action required. Untouched by this pass's commit."
    evidence_status: "carried forward -- no artifact in this chain changed since the prior pass"
---

# Phase 34: The Observation Projector & Trigger — Verification Report (fourth re-verification)

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-15T02:14:56Z (HEAD `5b66202`, branch `issue37-telescope-runs-calendar`)
**Status:** passed (14/14 truths verified; 0 gaps; 0 human items outstanding; **0 traceability orphans — first pass with a clean requirement ledger**)
**Re-verification:** Yes — fingerprint re-stale from a planning-doc-only commit. Supersedes the 2026-09-15T01:56:16Z report.

## What this pass was asked to establish

One commit landed since the prior pass:

```
5b66202 docs(34): fix PROJ-04 traceability orphan -- claim it in 34-01/34-03
        plan frontmatter and the ROADMAP phase 34 requirements line
```

Two questions, both narrow:

1. **Is PROJ-04 now correctly claimed and traceable, with all 11 requirement IDs accounted for and no orphans?** — Yes, established by programmatic three-way reconciliation below.
2. **Did anything else drift?** — No. The commit is planning-doc-only; I confirmed that by diff rather than by its message.

## The commit under review

I did not take the commit message's word for its scope. `git diff --stat 588cf22..HEAD` is three files, three changed lines, all `.planning/`:

```
 .planning/ROADMAP.md                                     | 2 +-
 .planning/phases/34-.../34-01-PLAN.md                    | 2 +-
 .planning/phases/34-.../34-03-PLAN.md                    | 2 +-
 3 files changed, 3 insertions(+), 3 deletions(-)
```

Every change is one requirement list gaining `PROJ-04` in sorted position:

```diff
-**Requirements**: PROJ-01, PROJ-02, PROJ-03, PROJ-05, …   (ROADMAP)
+**Requirements**: PROJ-01, PROJ-02, PROJ-03, PROJ-04, PROJ-05, …
-requirements: [PROJ-01, PROJ-02, PROJ-03, PROJ-05, PROJ-06, TRIG-01, TRIG-02]   (34-01)
+requirements: [PROJ-01, PROJ-02, PROJ-03, PROJ-04, PROJ-05, PROJ-06, TRIG-01, TRIG-02]
-requirements: [PROJ-06, PROJ-03, PROJ-05]                 (34-03)
+requirements: [PROJ-06, PROJ-03, PROJ-04, PROJ-05]
```

**Source drift is zero, measured not assumed:** `git diff --name-only 588cf22..HEAD -- solsys_code/ src/ docs/` returns nothing. The working tree carries the same single unrelated dirty file as the prior pass (Phase 35's `reconcile_campaign_runs_demo.ipynb`). The developer database is byte-identical (`1789431410 1232896`), and the phase's docs artifacts are clean with the baseline JSON still at sha256 `453ae2ba…`.

## PROJ-04: is the claim substantively true, or just asserted?

A plan can be *made* to claim a requirement by editing one line. That would satisfy a naive ledger check while meaning nothing. Three independent things had to hold, and all three do.

**1. An independent document already attributed PROJ-04 to exactly these two plans.** `34-VALIDATION.md` was written 2026-09-12, two days before this fix, so it cannot have been shaped by it. It maps:

| Task | Plan | Requirement |
|---|---|---|
| 34-01-01 | **01** | `PROJ-04 (title stem)` |
| 34-03-02 | **03** | `PROJ-04 (title stem)`, PROJ-05 |

The commit amends plans 01 and 03 — precisely the two plans the pre-existing validation matrix names. The claim matches independent prior attribution rather than being reverse-engineered to close a flag.

**2. The write side really is in plan 01's delivered code.** `observation_projector.py:315-332` (`write_event_meta()`) writes exactly the `CalendarEventMeta` carrier fields PROJ-04 names — `is_verified` / `observation_record` / `observation_group` — after first clearing any stale one-to-one claim from a companion row. `series_group_for()` (line 237) resolves the group deterministically by lowest pk. This is the FK carrier PROJ-04 demands in place of spike 002's title-suffix stopgap.

**3. The display side really is in plan 03's delivered code.** `calendar_display_extras.observation_series_decoration()` (line 615) renders "night n of N" *at request time* from `meta.observation_group` / `meta.observation_record`, and is wired into `src/templates/tom_calendar/partials/event_form.html:148`. It returns `None` when either link is absent (line 670) — read-only, nothing stored.

**Behavioural proof run this pass, not cited:**

```
$ python manage.py test solsys_code.tests.test_observation_projector.TestMetaLinks \
                        solsys_code.tests.test_calendar_event_meta_links
Ran 12 tests in 2.418s — OK
```

`TestMetaLinks` includes `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description` — the test that pins both halves of PROJ-04 at once: the group link is written to the FK, and the group name is kept *out* of the title and description.

### Three-way requirement reconciliation

Computed programmatically over the three source documents rather than read by eye:

| Source | Count | IDs |
|---|---|---|
| Union of all 7 plans' `requirements:` frontmatter | **11** | ANNOT-03, PROJ-01…06, SCHED-06, TRIG-01…03 |
| ROADMAP Phase 34 `**Requirements**:` line | **11** | *identical set* |
| REQUIREMENTS.md traceability rows mapping to Phase 34 | **11** | *identical set* |

```
orphans  (mapped to P34 but unclaimed by any plan): NONE
unmapped (claimed by a plan but not P34 in REQUIREMENTS): NONE
three-way identical: True
```

Per-plan claims after the fix:

| Plan | Requirements |
|---|---|
| 34-01 | PROJ-01, PROJ-02, PROJ-03, **PROJ-04**, PROJ-05, PROJ-06, TRIG-01, TRIG-02 |
| 34-02 | ANNOT-03, PROJ-01, PROJ-05, TRIG-03 |
| 34-03 | PROJ-03, **PROJ-04**, PROJ-05, PROJ-06 |
| 34-04 | ANNOT-03, SCHED-06, TRIG-03 |
| 34-05 | PROJ-02, SCHED-06, TRIG-01, TRIG-02 |
| 34-06 | PROJ-02, SCHED-06, TRIG-01, TRIG-02 |
| 34-07 | ANNOT-03, PROJ-05, SCHED-06, TRIG-03 |

REQUIREMENTS.md's split-requirement note (L106, L139-141) is respected: PROJ-04's carrier-field clause stays with Phase 33 and its shared-title-stem clause with Phase 34. The fix claims only Phase 34's half; it does not annex Phase 33's.

## Goal Achievement

### Observable Truths

Truths 1-14 carry forward from the 2026-09-15T01:56:16Z pass. Because no source file, test file, notebook, runbook or database byte changed since that pass, the correct treatment under re-verification mode is a regression check rather than a from-scratch re-measurement — and the *precondition for carrying prior measurements forward was itself verified*, not assumed: the diff over `solsys_code/ src/ docs/` is empty and the database `stat` is byte-identical. On top of that I re-ran **228 tests green** across the phase's modules this pass rather than resting entirely on the prior run.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Re-run this pass: `test_observation_projector` + `test_observation_projector_signals` → `Ran 81 tests … OK`, including the one-event-per-record and shared-URL cases. Live corpus measurements carry forward legitimately — the database is byte-identical to the pass that measured them (159 records / 159 facility-url events / 0 duplicates / 0 orphans; status × marker cross-tab one-to-one; `unprojectable: 0`). |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | Behaviour-dependent (never-abort / no-churn invariants), so verified on behavioural evidence re-run this pass, not on presence: the 81-test projector+signals run passes, covering `test_updatestatus_narrows_the_event_with_no_command_run`, `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`, and `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection` (the F-34-1 fix from `24875bf`). `apps.py` unchanged with all 5 `dispatch_uid`s intact. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | Sweep command file unchanged; the prior pass's dry run against a current-state scratch copy (`created: 0, updated: 0, unchanged: 159, unprojectable: 0`) carries forward because both the code and the database are byte-identical. Namespace isolation unchanged (`RUN:` 72, `ALLOC:` 0, blank-url 10). |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ✓ VERIFIED | 34-UAT.md Test 3 `result: pass`, artifact unchanged. Re-proved at integration level this pass by the passing `TestUpdateObservationStatusPath` within the 81-test run. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `test_marker_and_token_within_first_16_characters` passes within this pass's projector run. The contract it pins — marker **and** telescope token both inside `title[:16]`, across all nine stage cases — holds. Full titles run longer (live max 35 chars); see advisory 1, which exists so that is not misread as a regression. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Re-checked live this pass: `manage.py help \| grep -c sync_lco_observation_calendar` → **0**; `management/commands/` holds only `project_observation_calendar.py` + `sync_gemini_observation_calendar.py`; runbook carries **0** references to the retired command. |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | Re-run this pass: `test_calendar_display_extras` + `test_calendar_template` → 135 tests pass, including `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label`. Templates unchanged. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | Same 135-test run: `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. `_TERMINAL_PREFIXES` still covers both vocabularies. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | **PROJ-04 display half — focus of this pass.** `observation_series_decoration()` read at `calendar_display_extras.py:615-707` (request-time read, returns `None` without both links); wired at `event_form.html:148`. `TestObservationSeriesDecoration` and `test_modal_query_count_does_not_grow_with_group_size` pass in the 135-test run. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✓ VERIFIED | Notebook, baseline JSON and runbook all clean in `git status --porcelain`; baseline sha256 still `453ae2ba…`. No paired-docs obligation arises — this commit changes no module behaviour at all (see below). |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | **PROJ-04 write half — focus of this pass.** `write_event_meta()` read at `observation_projector.py:315-332`: writes only `is_verified`/`observation_record`/`observation_group`. Pinned by `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description` and `test_record_in_two_groups_links_the_lowest_pk_group`, run green by name this pass (12-test run). |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | `campaign_attribution.py` unchanged; prior pass's `site_lookups: 0` read-back carries forward on a byte-identical database and codebase. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ✓ VERIFIED | Declared `verification: backstop`; closed by directly observed operator behaviour in 34-UAT.md Test 1 (`result: pass`) — two overlapping `updatestatus` runs, 0 `AttributeError`, 0 `unprojectable`, 0 `OperationalError`. UAT artifact unchanged. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | Pinned by the real 2026-09-14 pre-migration sweep incident, by 34-UAT.md Test 3, and by the notebook's 33 → 0 convergence. Artifacts unchanged. |

**Score:** 14/14 truths verified (0 present-but-behavior-unverified, 0 abstained, 0 failed)

Truth 2 is behaviour-dependent — it asserts never-abort and no-churn invariants that presence checks cannot see. It is marked VERIFIED on behavioural evidence re-run this pass (the 81-test projector + signals module run), not on symbol presence.

### Regression check — did anything drift?

| Surface | Check | Verdict |
|---|---|---|
| Commit scope | `git diff --stat 588cf22..HEAD` | ✓ 3 planning docs, 3 lines. Nothing else. |
| **Source drift** | `git diff --name-only 588cf22..HEAD -- solsys_code/ src/ docs/` | ✓ **Empty** — no source, template, notebook or runbook byte changed. |
| Requirement ledger | programmatic three-way set reconciliation | ✓ 11 = 11 = 11, `orphans: NONE`, `unmapped: NONE`, sets identical. |
| PROJ-04 write side | read `observation_projector.py:315-332` + named tests | ✓ FK carrier written; `Ran 12 tests … OK`. |
| PROJ-04 display side | read `calendar_display_extras.py:615-707` + template line 148 | ✓ Request-time render, wired; covered in the 135-test run. |
| Core projector behaviour | `test_observation_projector` + `…_signals` | ✓ `Ran 81 tests … OK`. |
| Calendar display behaviour | `test_calendar_display_extras` + `test_calendar_template` | ✓ 135 tests pass. |
| Receiver wiring | `grep -c dispatch_uid solsys_code/apps.py` | ✓ `5` — unchanged. |
| Retired command | `manage.py help \| grep -c` | ✓ `0`. Runbook references: `0`. |
| Phase docs artifacts | `git status --porcelain` + sha256 | ✓ Clean; baseline still `453ae2ba…`. |
| Developer DB | `stat -c '%Y %s'` | ✓ `1789431410 1232896` — byte-identical; untouched by this verification. |
| ROADMAP success criteria | diff inspection | ✓ All 5 SCs and the Phase 37 scope note unchanged; only the Requirements line moved. |

**Test total run this pass: 228 green, 0 failures.** (135 + 12 + 81. One loader error occurred on my first invocation because I guessed a non-existent class name `TestSeriesGroupLink`; the real class is `TestMetaLinks`. That was my error, not a code failure — the 135 real tests in that same run all passed, and the corrected invocation is recorded above.)

### CLAUDE.md paired-docs check

Not triggered, and the reasoning is concrete rather than convenient: the rule's trigger is a **behaviour** change to a mapped module. This commit changes **no module at all** — the diff over `solsys_code/`, `src/` and `docs/` is empty. Three planning documents gained one requirement ID each. No notebook or runbook can go stale from a requirements-list edit, and the notebook's executed output is provably untouched (byte-identical file, byte-identical baseline hash). No paired-docs obligation is breached.

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose vs. terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note (verified unchanged by this commit): "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Both vocabularies already paint the correct ring. |

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | Carried forward: PROJ-06's tested contract is marker+token within `title[:16]`, not a 16-char total title; live max is 35 chars. | other | Restated so a future `MAX(LENGTH(title))` measurement is not misread as a regression. Truth 5's evidence above now states the contract correctly. Database byte-identical — no data changed. |
| 2 | Carried forward: the debug doc's `CalendarEvent.modified == ObservationRecord.modified` signature for 4378332 / 4378046 no longer holds live. | other | A dated Evidence entry, corroborated 25 minutes before the 2026-09-14 sweep overwrote it. Substantive half (observed-block spans) survives. Untouched by this commit. |
| 3 | Carried forward: the corrected debug doc's summary sentence reads as if the sweep followed the migration; the first attempt preceded it. | other | The entry's own parenthetical states the correct sequence. Ambiguous phrasing, not a factual error. Untouched by this commit. |

No new advisories. Prior advisory 1 (the unguarded `campaign_run_links` lookup) remains **closed** — fixed in `24875bf`, pinned by `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection`, which passed again inside this pass's 81-test signals run.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + receivers; PROJ-04 FK carrier write | ✓ VERIFIED | Unchanged since prior pass. `write_event_meta()` re-read at 315-332 this pass for the PROJ-04 claim: writes only the three meta fields, clearing stale companion claims first. 81 tests green. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, PROJ-04 request-time series decoration | ✓ VERIFIED | Unchanged. `observation_series_decoration()` re-read at 615-707; 135 tests green. |
| `src/templates/tom_calendar/partials/event_form.html` | Renders the series block | ✓ VERIFIED | Unchanged; tag invoked at line 148. |
| `solsys_code/tests/test_observation_projector.py` | Pins PROJ-04 write side | ✓ VERIFIED | `TestMetaLinks` run green by name this pass. |
| `solsys_code/tests/test_calendar_event_meta_links.py` | Pins the meta link model | ✓ VERIFIED | Run green this pass (within the 12-test run). |
| `solsys_code/apps.py` | `ready()` wires the receivers | ✓ VERIFIED | Unchanged; 5 `dispatch_uid`s intact. |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` | ✓ VERIFIED | Unchanged. |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep | ✓ VERIFIED | Unchanged; `manage.py help` lists it, LCO sync absent. |
| `solsys_code/campaign_attribution.py` | `OBSERVED_TELESCOPE_OBSCODES` bridge | ✓ VERIFIED | Unchanged. |
| `solsys_code/allocation_projector.py` | D-11 re-project target | ✓ VERIFIED | Unchanged. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo | ✓ VERIFIED | Byte-unchanged (clean `git status`). |
| `…/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical | ✓ VERIFIED | sha256 `453ae2ba…`, unchanged. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section | ✓ VERIFIED | Clean; 0 references to the retired command. |
| `.planning/phases/34-…/34-01-PLAN.md` | **Claims PROJ-04 (write side)** | ✓ VERIFIED | **Changed this pass.** Frontmatter now `[PROJ-01, PROJ-02, PROJ-03, PROJ-04, PROJ-05, PROJ-06, TRIG-01, TRIG-02]`; matches 34-VALIDATION.md's task 34-01-01 mapping and the delivered `write_event_meta()`. |
| `.planning/phases/34-…/34-03-PLAN.md` | **Claims PROJ-04 (display side)** | ✓ VERIFIED | **Changed this pass.** Frontmatter now `[PROJ-06, PROJ-03, PROJ-04, PROJ-05]`; matches 34-VALIDATION.md's task 34-03-02 mapping and the delivered series decoration. |
| `.planning/ROADMAP.md` | Phase 34 requirements line includes PROJ-04 | ✓ VERIFIED | **Changed this pass.** 11 IDs, matching the plan union exactly. Success criteria and scope note untouched. |
| `src/fomo_db.sqlite3` | SCHED-06 evidence database | ✓ VERIFIED | `1789431410 1232896` — byte-identical. Untouched by this verification. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` | ✓ WIRED | Unchanged; re-proved by the 81-test projector + signals run. |
| `SolsysCodeConfig.ready()` | the receivers | `.connect(weak=False, dispatch_uid=…)` | ✓ WIRED | 5 `dispatch_uid`s present. |
| `project_record()` | `CalendarEventMeta.observation_group` | `write_event_meta()` → `series_group_for(record)` | ✓ WIRED | **PROJ-04 write link**, re-read and test-proved this pass (12 tests). |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` at template line 148 | ✓ WIRED | **PROJ-04 display link**, re-read and test-proved this pass (135 tests). Display-time only — nothing stored. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | Unchanged; prior convergence evidence valid on a byte-identical corpus. |
| `receiver_on_record_save()` | `allocation_projector.reproject_allocation_if_dispatched()` | guarded `list(instance.campaign_run_links…)` then per-link `try` | ✓ WIRED | Fully guarded since `24875bf`; its regression test passed again in this pass's signals run. |
| committed notebook evidence | every `manage.py test` run | `test_projector_demo_notebook.py` | ✓ WIRED | Guard unchanged and present in the default suite. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `write_event_meta()` | `observation_group` | `series_group_for(record)` → `ObservationGroup.objects.filter(observation_records=record).order_by('pk').first()` | Yes — a real FK to a real group row, not text; proved by `TestMetaLinks` | ✓ FLOWING |
| `observation_series_decoration()` | `index` / `size` / `group_name` | `meta.observation_group.observation_records.all()` at request time | Yes — returns `None` when links absent, real n-of-N when present | ✓ FLOWING |
| `event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` | Yes — 159 live events spanning stage-appropriate blocks | ✓ FLOWING |
| `telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']`, else `coarse_telescope_label()` | Yes — read back with `site_lookups: 0` | ✓ FLOWING |
| `stage_for()` → marker | marker letter | `record.status` | Yes — status × marker cross-tab one-to-one across all 159 rows | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant | Intentionally fixed (documented) | ✓ FLOWING (by design) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| **PROJ-04 write side pinned** | `python manage.py test …test_observation_projector.TestMetaLinks …test_calendar_event_meta_links` | `Ran 12 tests in 2.418s` / `OK` | ✓ PASS |
| **PROJ-04 display side + legend/rings pinned** | `python manage.py test …test_calendar_display_extras …test_calendar_template` | 135 tests pass | ✓ PASS |
| **Core projector + receiver behaviour** | `python manage.py test …test_observation_projector …test_observation_projector_signals` | `Ran 81 tests in 42.788s` / `OK` | ✓ PASS |
| Requirement ledger has no orphans | programmatic 3-way set reconciliation | `orphans: NONE`, `unmapped: NONE`, `identical: True` | ✓ PASS |
| Source drift since prior pass | `git diff --name-only 588cf22..HEAD -- solsys_code/ src/ docs/` | (empty) | ✓ PASS |
| Retired command really gone | `python manage.py help \| grep -c sync_lco_observation_calendar` | `0` | ✓ PASS |
| Receiver wiring intact | `grep -c dispatch_uid solsys_code/apps.py` | `5` | ✓ PASS |
| Baseline JSON byte-identical | `sha256sum` | `453ae2ba…` | ✓ PASS |
| Developer DB untouched | `stat -c '%Y %s'` | `1789431410 1232896` | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Requirements Coverage

All eleven IDs declared in this phase's plan frontmatter, cross-referenced against `.planning/REQUIREMENTS.md`.

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1 — 159/159, 0 duplicates, 0 orphans |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truths 1, 2, 4 |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Cross-tab: `WINDOW_EXPIRED`→`[X]` 26, `CANCELED`→`[C]` 6, `FAILURE_LIMIT_REACHED`→`[F]` 1, none mismarked |
| **PROJ-04** (Phase 34's shared-title-stem clause) | **34-01 (write), 34-03 (display)** | Series identity carried by real FKs on `CalendarEventMeta`, not text in the title | ✓ SATISFIED — **orphan CLOSED** | **Now claimed by both delivering plans and by the ROADMAP line.** Three-way reconciliation identical, `orphans: NONE`. Attribution corroborated by 34-VALIDATION.md's independent task mapping (34-01-01, 34-03-02). Delivery re-proved this pass: truths 9 and 11, `write_event_meta()` at `observation_projector.py:315-332`, `observation_series_decoration()` at `calendar_display_extras.py:615`, 12 named tests green. Phase 33 retains the carrier-field clause per the documented split. |
| PROJ-05 | 34-01, 34-02, 34-03, 34-07 | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation`; `RUN:` 72 / `ALLOC:` 0 / blank-url 10 intact |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 — `test_marker_and_token_within_first_16_characters` passes. See advisory 1 on the contract's exact wording. |
| SCHED-06 | 34-04, 34-05, 34-06, 34-07 | A user watches a record narrow over real nights with no command | ✓ SATISFIED | Truth 4; UAT Test 3 `pass`; evidence database byte-identical |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; 5 `dispatch_uid`s; `TestUpdateObservationStatusPath` |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED | Truth 2 — both never-abort tests pass in this pass's 81-test run |
| TRIG-03 | 34-02, 34-04, 34-07 | Sweep with `--dry-run`, failure isolation, paired pre-executed demo notebook | ✓ SATISFIED | Truth 3; notebook byte-unchanged with guard in the default suite |
| ANNOT-03 | 34-02, 34-04, 34-07 | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✓ SATISFIED | Truth 6 — `manage.py help` count 0, runbook references 0 |

**All 11 requirement IDs accounted for. Zero orphans, zero unmapped — the ledger is clean for the first time in this phase's verification history.**

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `34-01-PLAN.md`, `34-03-PLAN.md` | — | Debt markers (`TBD`/`FIXME`/`XXX`) in files changed this pass | ✓ **NONE** | Grepped explicitly: zero matches in both. |
| `.planning/ROADMAP.md` | 98, 250, 303, 320, 347 | `TBD` occurrences | ℹ️ Info | **Not debt markers.** Domain vocabulary and structure: "Range/TBD Import" is a phase name, "TBD/range run" is a real scheduling concept in this codebase, and "**Plans**: TBD" marks phases 36/37 as not yet planned. All pre-existing; the commit changed only line 209 (the Requirements line), so none is in this change's scope. |
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | Pre-existing (`a8613bc8`, 2025-07-23); `views.py` not modified since the prior `verified:` timestamp, so new-scope under the re-verification evidence gate and non-blocking. Classified identically by all five passes. |
| `docs/runbooks/telescope_runs_calendar.rst` | 1099 | `` ``TBD window`` `` | ℹ️ Info | A documented skip-reason *value* the reconciler emits, beside `not approved` / `unresolved site`. No unfinished-work semantics. |

Under the re-verification evidence gate, the only files modified since the prior `verified:` timestamp are the three planning documents; all three were scanned and the two plan files are clean. No blocker was found, so the gate's evidence requirement is not exercised this pass.

### Human Verification Required

**None.** No truth is present-but-behavior-unverified, no truth abstained, and no item requires human observation.

### Gaps Summary

**No gaps. No regressions. 14/14 truths verified. Zero traceability orphans.**

This pass existed to confirm one thing and to make sure confirming it broke nothing else. Phase 34's content fingerprint covers `34-01-PLAN.md` and `34-03-PLAN.md`, so a planning-doc commit re-staled a report that had already passed.

The substantive question was whether PROJ-04 is now genuinely traceable rather than merely flagged-as-fixed. I treated "a plan now claims the requirement" as the weakest possible evidence, since that is one line of text anyone can add, and looked for reasons the claim might be hollow. I found three reasons it is not. First, `34-VALIDATION.md` — written 2026-09-12, two days before the fix, so it cannot have been written to justify it — already attributed PROJ-04 to task 34-01-01 in plan 01 and task 34-03-02 in plan 03, which are exactly the two plans the commit amends. Second, the write side is real code in plan 01's deliverable: `write_event_meta()` writes the `observation_record` and `observation_group` foreign keys, which is precisely the FK carrier PROJ-04 demands in place of spike 002's title-suffix stopgap. Third, the display side is real code in plan 03's deliverable: `observation_series_decoration()` renders "night n of N" at request time from that FK and is wired into `event_form.html:148`. I then ran the tests that pin both halves rather than citing them — twelve green, including the one test that asserts the group link is written to the FK *and* the group name stays out of the title and description.

The ledger itself I reconciled programmatically rather than by eye, across all three documents that have to agree: the union of all seven plans' frontmatter, the ROADMAP's Phase 34 requirements line, and the REQUIREMENTS.md rows mapping to Phase 34. All three are now the identical eleven-element set, with no orphans and nothing claimed that REQUIREMENTS.md does not map here. The documented Phase 33 / Phase 34 split of PROJ-04 is respected — Phase 34 claims its title-stem clause only, not Phase 33's carrier fields.

For the rest, I established the precondition for carrying prior measurements forward instead of assuming it: the diff over `solsys_code/`, `src/` and `docs/` is empty, the developer database is byte-identical, and the notebook and baseline JSON are clean with an unchanged hash. On a byte-identical codebase and corpus, the prior pass's live measurements are the same measurements. Even so I re-ran 228 tests across the phase's modules — the core projector and signals suite at 81, the calendar display and template suite at 135, and the PROJ-04 pair at 12 — all green. One loader error in my first invocation was my own bad class-name guess, not a code failure, and I have recorded it rather than quietly re-running.

The phase goal — one writer for observation-backed nights, drawn and kept current with no operator command — holds in the codebase, in the live database, and now in the requirement ledger too.

---

_Verified: 2026-09-15T02:14:56Z_
_Verifier: Claude (gsd-verifier)_
