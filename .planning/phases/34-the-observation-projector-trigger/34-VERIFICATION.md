---
phase: 34-the-observation-projector-trigger
verified: 2026-09-15T00:35:15Z
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

covered_digest: "v1:sha256:d06d1cc24e344641a2efea13f0d47158ac3028bbde3a34f0a0b0de81a1783254"
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: human_needed
  previous_score: 14/14
  trigger: "closure check on the two human items the 2026-09-14T22:39:20Z pass raised -- finding F-34-1 (a Phase 34 debug artifact claiming an overnight updatestatus-only run that never happened) and the operator decision on whether to spend the SCHED-06 baseline by running the backstop sweep"
  gaps_closed:
    - "F-34-1 half (a) -- the debug doc's false claim. `.planning/debug/resolved/34-updatestatus-receiver-attributeerror.md` was corrected in commit `e3303e6`: the original Evidence entry is preserved and explicitly RETRACTED, `next_action` carries a dated CORRECTION, and `verification.signal_real_path` is amended. Every corrected factual claim was re-measured against `src/fomo_db.sqlite3` this pass and all of them hold."
    - "F-34-1 half (b) -- the 14 legacy-stale events. Verified CLEARED in the live developer database: a status x marker cross-tab over all 159 facility-url events returns ZERO terminal-status records carrying a `[Q]`/`[S]` marker (COMPLETED -> `[O]` 76, CANCELED -> `[C]` 6, WINDOW_EXPIRED -> `[X]` 26, FAILURE_LIMIT_REACHED -> `[F]` 1, PENDING -> `[Q]` 38 / `[S]` 12). Each of the 14 named observation_ids now carries an `[O]` event over its own observed block."
    - "Convergence independently reproduced this pass: `project_observation_calendar --dry-run` against a scratch copy of the CURRENT developer database reports `failed: 0 | LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0` -- byte-matching the result the corrected debug doc claims for its post-migration re-run, and down from `updated: 14` at the prior pass."
    - "Migration `0018_campaignrun_night_window_fields` confirmed applied: `django_migrations` row `solsys_code|0018_campaignrun_night_window_fields|2026-09-15 00:16:40.283886`. Migration file read: purely additive -- two nullable `TimeField`s (`night_start_utc`, `night_end_utc`) on `CampaignRun`, no data migration, no alteration of any existing column."
  gaps_remaining: []
  regressions: []
gaps: []
deferred:

  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 34 scope note: 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Phase 37 = 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02). Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
advisory:

  - finding: "NEW, EVIDENCED: `receiver_on_record_save()` evaluates `instance.campaign_run_links.select_related('run')` at the `for` statement, OUTSIDE any `try`. A database error raised by that query escapes the receiver and propagates out of `ObservationRecord.save()`. This is the Phase 35 D-11 linked-run loop, not Phase 34's projector block, but it sits inside the receiver Phase 34 wired."
    category: architectural
    reason: "Deterministically reproduced this pass on a scratch database copy: with `campaign_run_links` made to raise `OperationalError('no such column: solsys_code_campaignrun.night_start_utc')`, `record.save()` RAISED `OperationalError` rather than absorbing it. This is exactly the real-world condition the corrected debug doc records for the pre-migration sweep attempt on 2026-09-14. Phase 35's own regression test `test_linked_run_reproject_raising_does_not_abort_the_records_own_save_or_projection` does NOT cover it -- it patches `project_allocation` to raise, which is inside the per-link `try`; the link QUERY itself is unguarded. NOT a Phase 34 must-have failure: TRIG-02 and truth 2 both scope the guarantee to 'a projector error', and the projector block IS guarded (`test_raising_projector_does_not_block_a_save` passes). The phase goal is provably unaffected -- base projection runs and commits BEFORE the failing step, which is why the 14 events were corrected on 2026-09-14 while the D-11 step errored. Recommended fix belongs to Phase 35's scope (ALLOC-03): move the `for` header inside a guard, or wrap the queryset evaluation in `list(...)` under its own `try`. Phase 35 has no VERIFICATION.md yet, so this is routed there rather than closed here."
    evidence_status: "reproduced -- scratch-copy probe: save() raised OperationalError; corroborated by the 2026-09-14 pre-migration sweep incident recorded in the corrected debug doc"
  - finding: "The debug doc's corrected Evidence entry states in the present tense that records 4378332 / 4378046 'each carry an [O] event over their own observed block with CalendarEvent.modified == ObservationRecord.modified to the second'. That timestamp-equality signature no longer holds live: the 2026-09-14 operator sweep re-titled both with their resolved site token (`[O] 1m0 11P` -> `[O] TFN-1m0 11P`, `[O] 1m0 10P` -> `[O] LSC-1m0 10P`) and bumped `CalendarEvent.modified` to 2026-09-14 23:06/23:07."
    category: other
    reason: "Not an inaccuracy -- it is a dated Evidence entry (`timestamp: 2026-09-14`, `checked:`), i.e. a point-in-time observation, and it was independently corroborated from the live database by the prior verification pass at 22:39Z, ~25 minutes before the sweep ran at 23:06Z. The substantive half of the evidence SURVIVES in the database today: both events still span their own observed blocks (4378332: `2026-09-12 01:06:46 -> 01:26:04`), which is the receiver-produced narrowing itself; only the title token and the modified stamp were later overwritten. The operator deliberately spent this baseline, which is precisely what human item 2 of the prior report asked them to decide. Noted so a future reader does not re-derive the equality check and conclude the doc is wrong."
    evidence_status: "measured -- live SQL on 4378332 / 4378046 this pass"
  - finding: "The corrected debug doc's summary sentence ('cleared ... by a real project_observation_calendar sweep on 2026-09-14, after applying the then-pending 0018 migration') reads, in isolation, as if the sweep followed the migration. The database timestamps show the FIRST sweep attempt preceded it."
    category: other
    reason: "The entry's own parenthetical already states the correct two-attempt sequence, and the timestamps confirm it exactly: 14 events modified 2026-09-14 23:06:34-23:07:19 (first, pre-migration attempt -- corrections landed), migration applied 2026-09-15 00:16:40, post-migration re-run wrote nothing (`updated: 0`), DB mtime 2026-09-15 00:16:50. Ambiguous phrasing in a summary line, not a factual error. No action required."
    evidence_status: "measured -- django_migrations row + CalendarEvent.modified distribution + file mtime, all three mutually consistent"
---

# Phase 34: The Observation Projector & Trigger — Verification Report (second re-verification)

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-15T00:35:15Z (HEAD `e3303e6`, branch `issue37-telescope-runs-calendar`)
**Status:** passed (14/14 truths verified; 0 gaps; 0 human items outstanding)
**Re-verification:** Yes — closure check on the two human items raised by the 2026-09-14T22:39:20Z report. Supersedes that report.

## What this pass was asked to establish

The prior pass found no regression from Phase 35 and confirmed 14/14 must-haves, but landed on `human_needed` because of finding **F-34-1**: a Phase 34 debug artifact claimed an overnight `updatestatus`-only run had narrowed all 33 previously-stale LCO events, when the database showed no such run had occurred and 14 of the 33 were still stale.

Both items have since been actioned. This pass verifies the claimed closure **against the live database and the code**, not against the commit messages.

**Answer: F-34-1 is genuinely closed on both halves, and nothing else drifted.** No source file changed since the prior pass — `git diff --stat 31ec92c..HEAD` touches exactly two Markdown files (the debug doc and the prior VERIFICATION.md), and `git status --porcelain solsys_code/ src/templates/` is empty. One new, evidenced finding is raised as an advisory and routed to Phase 35.

## F-34-1 closure — the primary question

### Half (a): is the corrected debug doc accurate?

Every factual claim in the corrected doc was re-measured. All of them hold.

| Claim in the corrected doc | Measurement this pass | Verdict |
|---|---|---|
| "no `updatestatus` run occurred after 2026-09-12 … as of the re-verification" | The prior report's own dated measurements (`MAX(modified)` = `2026-09-12 22:10:48.773592`, mtime `1789251048`) are preserved in the superseded report and were not re-derivable after the sweep. Consistent with the 09-11/09-12/09-14 modification distribution seen today (131 / 12 / 16). | ✓ Accurate (dated) |
| "14 of the 33 were still stale as of the re-verification" | The 14 named observation_ids all carry `CalendarEvent.modified` of **2026-09-14 23:06-23:07** — i.e. they were last written by the 09-14 sweep, having been untouched since 2026-09-11. Exactly 14, not 15: `4378041` is `WINDOW_EXPIRED`/`[X]`, correct all along, and was never in the stale set. | ✓ Accurate |
| "two individual records (4378332, 4378046) show genuine receiver-alone narrowing" | Both still span their own observed blocks (`4378332`: `2026-09-12 01:06:46 → 01:26:04`). See advisory 2 — the `modified`-equality half has since been overwritten by the operator sweep, as designed. | ✓ Accurate as a dated entry |
| "the other 14 … could never be reached by `updatestatus` (terminal states excluded, `facility.py:573`)" | Corroborated structurally: all 14 are `COMPLETED`; the sweep — not the receiver — is what resolved their observed-site token. | ✓ Accurate |
| "cleared … by a real `project_observation_calendar` sweep on 2026-09-14" | **Independently corroborated by a signature only the sweep can produce.** All 14 titles now carry a resolved site prefix (`[O] COJ-1m0 220P`, `[O] LSC-1m0 10P`, `[O] CPT-1m0 10P`, `[O] TFN-1m0 11P`). The D-07 observed-site lookup lives in the sweep's `pre_fields_hook` — the `post_save` receiver never performs it. Before the sweep these titles read `[Q]`/`[S] 1m0 …` with no site token. The 45-second spread of the 14 `modified` stamps (23:06:34 → 23:07:19, 2-5 s apart) is the network-bound site lookup pacing. | ✓ Accurate, and mechanism-corroborated |
| "after applying the then-pending `0018_campaignrun_night_window_fields` migration … `OperationalError` in the D-11 step on the first attempt … corrections had already landed via `receiver_on_record_save()`'s own `project_record()` call, which runs before that step" | Code read confirms the ordering exactly: `observation_projector.py:626-651` runs `project_record(instance)` first (guarded), then the `campaign_run_links` loop. The loop's queryset does `select_related('run')`, so pre-migration it selects `night_start_utc`/`night_end_utc` and raises. **Reproduced deterministically** on a scratch copy — see advisory 1. | ✓ Accurate, and mechanism-reproduced |
| "Post-migration re-run: `failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159`" | **Reproduced byte-for-byte this pass** against a scratch copy of the current developer database. | ✓ Accurate |
| "Zero `[Q]`/`[S]`-marked events remain among COMPLETED LCO records" | Status × marker cross-tab returns an empty set for terminal-status × `[Q]`/`[S]`. | ✓ Accurate |
| "latest event `modified` is `2026-09-14T23:07:19Z`" | `SELECT MAX(modified) FROM tom_calendar_calendarevent` → `2026-09-14 23:07:19.066718`. | ✓ Accurate to the microsecond |

The doc also handles the retraction correctly: the original false Evidence entry is **preserved with an explicit `RETRACTED (2026-09-14, Phase 34 re-verification)` block** rather than silently deleted, `next_action` carries a dated `CORRECTION`, and `verification.signal_real_path` is amended in place. The audit trail is intact.

### Half (b): are the 14 legacy-stale events actually gone?

Read-only SQL against `src/fomo_db.sqlite3`, status cross-tabulated against event marker over all 159 facility-url events:

| Record status | Marker | Count |
|---|---|---|
| COMPLETED | `[O]` | 76 |
| CANCELED | `[C]` | 6 |
| WINDOW_EXPIRED | `[X]` | 26 |
| FAILURE_LIMIT_REACHED | `[F]` | 1 |
| PENDING | `[Q]` | 38 |
| PENDING | `[S]` | 12 |

**Zero terminal-status records carry a queued or scheduled marker.** Every row of that table is the marker the projector is supposed to paint for that status, and the tally sums to exactly 159 — one event per record, no residue, no misclassification. Against the prior pass (`[Q]`:42 `[S]`:22 `[O]`:62) the delta is exactly −4 `[Q]`, −10 `[S]`, +14 `[O]`: the 14 events, and only those 14, moved.

**F-34-1 is closed.**

### Migration 0018

```
solsys_code|0018_campaignrun_night_window_fields|2026-09-15 00:16:40.283886
```

Applied. The migration file was read rather than trusted: two `AddField` operations, both `TimeField(blank=True, null=True)` on `CampaignRun`, no data migration, no `AlterField`, no `RemoveField`. Purely additive, exactly as described, and it touches no model this phase's must-haves rest on.

### Database safety

No write command was issued against `src/fomo_db.sqlite3` by this verification. Every inspection used `file:…?mode=ro`; both the dry-run sweep and the receiver probe ran against scratch copies under `FOMO_DATABASE_PATH`. `stat -c '%Y %s'` returned `1789431410 1232896` before and after every command in this pass.

## Goal Achievement

### Observable Truths

Truths 1-14 carry forward from the 2026-09-14T22:39:20Z pass. No source file changed since (`git diff --stat 31ec92c..HEAD` = 2 Markdown files; working tree clean under `solsys_code/`, `src/templates/`, and Phase 34's `docs/` artifacts), so the failed-item/passed-item optimization applies: F-34-1's truths got full re-verification, the rest got regression checks. The regression checks were live measurements, not citations.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Re-measured live: `lco_soar_records = 159`, `facility_url_events = 159`, **zero duplicate non-blank urls**, **zero facility-url events without a record link**. The status × marker cross-tab above is stronger than the prior pass's flat marker tally — it proves not just that the markers sum to 159 but that **every one of them is the correct marker for its record's status**, which is the property F-34-1 was the last violation of. Today's dry-run sweep over the whole real corpus: `unprojectable: 0` across all 159. |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | `apps.py` receiver wiring unchanged (0 commits since prior pass). Behavioural re-run: the full 342-test phase suite passes, including `test_updatestatus_narrows_the_event_with_no_command_run`, `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`. The projector block's guard is intact and verified — see advisory 1 for the separate, Phase 35-owned gap in the *link-query* evaluation, which this truth's wording ("a projector error") does not cover and which does not affect calendar currency. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | **Strengthened this pass.** The prior pass could only show the sweep *predicting* 14 updates. The sweep has now actually run against the real corpus, and today's dry run reports `created: 0, updated: 0, unchanged: 159` — the convergence property demonstrated on live data rather than on a copy. Namespace isolation re-proved after that real write pass: `RUN:` **72**, blank-url **10**, `ALLOC:` **0** — identical to the Phase 34 notebook figures and to the prior pass, across a sweep that wrote 16 facility-url events. `TestDryRun`, `TestFailureIsolation`, `TestNamespaceIsolation` pass. |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ✓ VERIFIED | Evidence unchanged and now archival: 34-UAT.md Test 3 `result: pass`; independently corroborated from the live database by the prior verification pass (committed, `0793233`) at 22:39Z on a database no sweep had ever run against — records `4378332` / `4378046` with `CalendarEvent.modified == ObservationRecord.modified` to the second. Both events still span their own receiver-derived observed blocks today. The operator subsequently, and deliberately, spent that baseline by running the sweep (the prior report's human item 2) — see advisory 2. Re-proved at integration level with today's code by `test_updatestatus_narrows_the_event_with_no_command_run` and at corpus level by `unprojectable: 0` over all 159 records. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `test_marker_and_token_within_first_16_characters` passes. Live titles after the sweep, now with resolved site tokens: `'[O] TFN-1m0 11P'`, `'[O] COJ-1m0 220P'`, `'[O] CPT-1m0 10P'` — longest observed is 16 characters, still well inside the cap. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Re-checked: `manage.py help \| grep -c sync_lco_observation_calendar` → **0**; `management/commands/` holds only `project_observation_calendar.py` + `sync_gemini_observation_calendar.py`; `pre_executed/` holds only the two surviving notebooks; runbook carries **0** references to the retired command. |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | Templates unchanged (0 commits); `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label` pass. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | `calendar_display_extras.py` unchanged; `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. `_TERMINAL_PREFIXES` still covers both vocabularies. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | `event_form.html` unchanged; `TestObservationSeriesDecoration` and `test_modal_query_count_does_not_grow_with_group_size` pass. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✓ VERIFIED | Notebook and `sched06-baseline.json` byte-unchanged: `git log` on both paths shows no commit since the prior pass (`8757750` / `a87f5f8`), `git status --porcelain docs/` shows only Phase 35's `reconcile_campaign_runs_demo.ipynb`, and the baseline's sha256 still equals `git show a87f5f8:<path>` (`453ae2ba…d353859c`). Guard passes (5 ok, 1 branch-correct skip). |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | `write_event_meta()` unchanged; the four pinning tests pass. Live corroboration: every one of the 159 titles is `'[marker] <token> <target>'` with no group-derived text. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | **Strengthened this pass.** The prior pass showed `site_lookups: 0` on records whose sites were already stored. The 2026-09-14 sweep exercised the *other* half — it performed 16 fresh lookups, stored them, and today's dry run over the same corpus reports `site_lookups: 0` again. The once-only contract is now demonstrated across a real resolve-then-read-back cycle on live data. `test_campaign_attribution.py` passes; both files unchanged. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ✓ VERIFIED | Declared `verification: backstop`; closed by directly observed operator behaviour in 34-UAT.md Test 1 (`result: pass`) — two overlapping `updatestatus` runs, 0 `AttributeError`, 0 `unprojectable`, 0 `OperationalError`. Unchanged since the prior pass. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | **Strengthened this pass by a real incident.** The 2026-09-14 pre-migration sweep attempt *was* an interrupted sweep — it failed partway through the D-11 step on every record — and the live database shows every already-processed record left with a correct event (all 14 correct), with the post-migration re-run converging to `updated: 0` and no repair step. That is truth 14's exact claim, observed in production rather than simulated. Also pinned by 34-UAT.md Test 3 and the notebook's 33 → 0 convergence. |

**Score:** 14/14 truths verified (0 present-but-behavior-unverified, 0 abstained, 0 failed)

### Regression check — did anything drift since the prior pass?

| Surface | Check | Verdict |
|---|---|---|
| All source files | `git diff --stat 31ec92c..HEAD` | ✓ Two `.md` files only (debug doc, prior VERIFICATION.md). **No code changed.** |
| Working tree | `git status --porcelain solsys_code/ src/templates/` | ✓ Empty. |
| Phase 34 docs artifacts | `git status --porcelain docs/` | ✓ Only Phase 35's `reconcile_campaign_runs_demo.ipynb` is dirty; both Phase 34 artifacts clean and byte-identical. |
| Phase 34 test suite | 8 modules, one run | ✓ `Ran 342 tests in 15.259s … OK (skipped=1)` — identical count and outcome to the prior pass. |
| Foreign namespaces | live SQL after a real sweep | ✓ `RUN:` 72, blank-url 10, `ALLOC:` 0 — unchanged. |
| Requirements text | `.planning/REQUIREMENTS.md` | ✓ All eleven mapped lines present and unchanged. |

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose vs. terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Both vocabularies already paint the correct ring. |

### Advisory (New Scope)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | **NEW, EVIDENCED** — `receiver_on_record_save()` evaluates `instance.campaign_run_links.select_related('run')` outside any `try`; a DB error there escapes the receiver and propagates out of `ObservationRecord.save()`. Reproduced deterministically; occurred for real on 2026-09-14. | architectural | Phase 35's D-11 code, not Phase 34's projector block. TRIG-02 and truth 2 both scope the guarantee to "a projector error", and that block IS guarded. Calendar currency is provably unaffected — base projection commits before the failing step. Routed to Phase 35 (no VERIFICATION.md there yet). Does not falsify a Phase 34 must-have. |
| 2 | The debug doc's `CalendarEvent.modified == ObservationRecord.modified` signature for 4378332 / 4378046 no longer holds live — the 2026-09-14 sweep re-titled both and bumped the stamps. | other | A dated Evidence entry, corroborated by the prior verification 25 minutes before the sweep. The substantive half (observed-block spans) survives. The operator spent the baseline deliberately, as the prior report's human item 2 invited. Noted for future readers. |
| 3 | The corrected doc's summary sentence reads as if the sweep followed the migration; the first attempt preceded it. | other | The entry's own parenthetical states the correct sequence and the timestamps confirm it. Ambiguous phrasing, not a factual error. No action required. |

Advisories 1-3 are **new-scope relative to this re-verification round** — `solsys_code/observation_projector.py` was not git-modified since the prior `verified:` timestamp and none of these is a carried-forward gap — so under the re-verification evidence gate none of them reverts a completed must-have. Advisory 1 is nonetheless recorded with deterministic evidence so it can be actioned rather than rediscovered.

#### Advisory 1 — reproduction

```python
# against a scratch COPY under FOMO_DATABASE_PATH; the dev DB was not touched
class Boom:
    def select_related(self, *a, **k):
        raise OperationalError('no such column: solsys_code_campaignrun.night_start_utc')

ObservationRecord.campaign_run_links = property(lambda self: Boom())
record.save()
# -> PROBE-RESULT: save() RAISED -> OperationalError : no such column: ...
```

`observation_projector.py:626-651` — `project_record()` runs inside its own `try` (lines 628-636); the D-11 loop's per-link `try` (lines 643-651) is **inside** the loop body, so the `for` header's queryset evaluation at line 640 is unguarded. Suggested fix, Phase 35 scope: `for link in list(...)` under its own `try`, or hoist the query above the loop inside a guard.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + three signal receivers | ✓ VERIFIED | 0 commits and no working-tree change since the prior pass. Imported by `apps.py`, the sweep command, `allocation_projector.py` and 4 test modules. See advisory 1 for the Phase 35 sub-surface. |
| `solsys_code/apps.py` | `ready()` wires the three receivers | ✓ VERIFIED | Unchanged; all three `dispatch_uid`s intact. |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` used by both `record_time_window()` branches | ✓ VERIFIED | Unchanged; `TestCoerceScheduleDatetime` + `TestRecordTimeWindow` pass. |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep, zero required args | ✓ VERIFIED | Unchanged. Executed `--dry-run` against a current-state scratch copy this pass: `updated: 0, unchanged: 159`. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series decoration | ✓ VERIFIED | Unchanged; all three tags still called from the two templates. |
| `solsys_code/campaign_attribution.py` | `OBSERVED_TELESCOPE_OBSCODES` bridge | ✓ VERIFIED | Unchanged; its test module passes. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo with a real takeover | ✓ VERIFIED | Byte-unchanged; guard passes. |
| `…/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical to committed state | ✓ VERIFIED | sha256 `453ae2ba…d353859c` = `git show a87f5f8:<path>`. |
| `solsys_code/tests/test_projector_demo_notebook.py` | Repo-level guard over the committed notebook | ✓ VERIFIED | Unchanged; 5 pass + 1 branch-correct skip. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section, no stale sync content | ✓ VERIFIED | 0 references to the retired command. |
| `solsys_code/migrations/0018_campaignrun_night_window_fields.py` | Additive, applied | ✓ VERIFIED | Two nullable `TimeField`s on `CampaignRun`; applied `2026-09-15 00:16:40`. Not a Phase 34 artifact — verified because F-34-1's closure depended on it. |
| `src/fomo_db.sqlite3` | SCHED-06 evidence database | ✓ RECONCILED (was ⚠️ NOTED) | mtime moved `2026-09-12T22:10Z` → `2026-09-15T00:16:50Z` — the operator sweep and the migration. **F-34-1's residue is gone.** Untouched by this verification (`1789431410 1232896` before and after every command). |
| `solsys_code/tests/*` (8 phase modules) | Behaviour pinned | ✓ VERIFIED | `Ran 342 tests in 15.259s … OK (skipped=1)`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` | ✓ WIRED | Re-proved by the passing signals module and, live, by the 2026-09-14 sweep incident in which the base projection wrote all 16 events *before* the D-11 step failed. |
| `SolsysCodeConfig.ready()` | the three receivers | `.connect(weak=False, dispatch_uid=…)` | ✓ WIRED | Unchanged. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | The sweep's `unchanged: 159` against a corpus written by both the receiver and the sweep is the proof the two writers agree. |
| sweep `pre_fields_hook` | `record.save()` → receiver → `project_record()` | one-time observed-site resolution | ✓ WIRED | Demonstrated live: 16 site tokens resolved, stored, and read back with `site_lookups: 0` on the next pass. |
| `receiver_on_record_save()` | `allocation_projector.reproject_allocation_if_dispatched()` | `instance.campaign_run_links.select_related('run')`, per-link `try` | ⚠️ WIRED, guard incomplete | Reached and functional; the *link query* is unguarded (advisory 1). Dormant on the live corpus (1 `CampaignRunObservation` row, 0 `ALLOC:` events). |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` | ✓ WIRED | Display-time only. |
| committed notebook evidence | every `manage.py test` run | `test_projector_demo_notebook.py` | ✓ WIRED | Guard runs in the default suite. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` | Yes — live: each of the 14 repaired events spans its own observed block, e.g. `4378021` `2026-09-04 16:18:07 → 16:34:57` | ✓ FLOWING |
| `telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']`, else `coarse_telescope_label()` | Yes — 16 tokens resolved on 09-14 (`COJ`/`LSC`/`CPT`/`TFN`), then read back with `site_lookups: 0` | ✓ FLOWING |
| `stage_for()` → marker | marker letter | `record.status` | Yes — the status × marker cross-tab is one-to-one across all 159 rows, no mismatched pair | ✓ FLOWING |
| `project_queryset()` counters | `action` | `preview_calendar_event_action(before, fields)` | Yes — discriminates, not a blanket count: `updated: 14` before the sweep, `updated: 0` after | ✓ FLOWING |
| `receiver_on_record_save()` linked-run loop | `link.run` | `instance.campaign_run_links` | Yes (tested); dormant live (1 link, 0 `ALLOC:` events) | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant | Intentionally fixed (documented) | ✓ FLOWING (by design) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-34 test modules pass (8 modules, one run) | `python manage.py test solsys_code.tests.test_observation_projector …test_projector_demo_notebook` | `Ran 342 tests in 15.259s` / `OK (skipped=1)` | ✓ PASS |
| **Sweep converges on the post-fix real corpus** | `FOMO_DATABASE_PATH=<scratch copy> python manage.py project_observation_calendar --dry-run` | `failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero` | ✓ PASS |
| **Zero terminal-status records carry a queued/scheduled marker (live)** | read-only SQL, status × marker cross-tab | empty result set; COMPLETED → `[O]` 76 | ✓ PASS |
| **Migration 0018 applied (live)** | `SELECT … FROM django_migrations` | `solsys_code\|0018_campaignrun_night_window_fields\|2026-09-15 00:16:40.283886` | ✓ PASS |
| **Latest event modified matches the doc's claim** | `SELECT MAX(modified) FROM tom_calendar_calendarevent` | `2026-09-14 23:07:19.066718` | ✓ PASS |
| One event per record, no duplicates, no orphans (live) | read-only SQL | `159 records / 159 facility-url events / 0 duplicate non-blank urls / 0 unlinked facility-url events` | ✓ PASS |
| Foreign namespaces intact after a real sweep (live) | read-only SQL | `RUN: 72`, `ALLOC: 0`, blank-url `10` | ✓ PASS |
| Retired command really gone | `python manage.py help \| grep -c sync_lco_observation_calendar` | `0` | ✓ PASS |
| Sweep registered with its flags | `python manage.py project_observation_calendar --help` | `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]` | ✓ PASS |
| Baseline JSON byte-identical to `a87f5f8` | `sha256sum` vs `git show` | both `453ae2ba…d353859c` | ✓ PASS |
| **Receiver link-query guard (negative probe)** | scratch-copy probe forcing `campaign_run_links` to raise | `save() RAISED OperationalError` — advisory 1 | ✗ FAIL (advisory, Phase 35 scope) |
| Developer DB untouched by this verification | `stat -c '%Y %s'` before/after every command | `1789431410 1232896` both times | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Requirements Coverage

Every requirement ID declared in this phase's plan frontmatter, cross-referenced against `.planning/REQUIREMENTS.md`.

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1 — live: 159/159, 0 duplicates, 0 orphans |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truths 1, 2, 4; the 14 repaired events each span their own observed block |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Cross-tab: `WINDOW_EXPIRED` → `[X]` 26, `CANCELED` → `[C]` 6, `FAILURE_LIMIT_REACHED` → `[F]` 1, none mismarked |
| PROJ-04 (title-stem clause) | **not declared in any plan's `requirements:`** | Series identity carried by real FKs, not text in the title | ✓ SATISFIED but ⚠️ **ORPHANED** | REQUIREMENTS.md L18/L106 maps the title-stem clause to Phase 34, yet no plan frontmatter claims `PROJ-04` and the ROADMAP's phase requirement list omits it. Delivered anyway (truths 9, 11). Traceability gap in the plans, not a delivery gap — carried forward unchanged for the third pass. |
| PROJ-05 | 34-01, 34-02, 34-03, 34-07 | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation`; live `RUN: 72` / `ALLOC: 0` / blank-url `10` intact across a real 16-event sweep |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 — longest live title 16 chars even with the resolved site token |
| SCHED-06 | 34-04, 34-05, 34-06, 34-07 | A user watches a record narrow over real nights with no command | ✓ SATISFIED | Truth 4. **The debug artifact that previously contradicted this is now corrected, and every corrected claim was re-measured against the database this pass.** |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; `apps.py` unchanged; `TestUpdateObservationStatusPath` |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED | Truth 2; `test_make_request_is_never_called_during_a_record_save`, `test_raising_projector_does_not_block_a_save`. The projector block's guard holds. Advisory 1 records a Phase 35-owned gap in the adjacent link query — outside this requirement's "a projector error" wording, and it does not affect calendar currency. |
| TRIG-03 | 34-02, 34-04, 34-07 | Sweep command with `--dry-run`, failure isolation, and a paired pre-executed demo notebook | ✓ SATISFIED | Truth 3 — **the sweep has now performed its designed backstop role on live data**, clearing exactly the residue `updatestatus` structurally cannot reach, then converging to `updated: 0` |
| ANNOT-03 | 34-02, 34-04, 34-07 | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✓ SATISFIED | Truth 6 |

**All 10 declared requirement IDs accounted for.** One orphan (`PROJ-04`) mapped to Phase 34 in REQUIREMENTS.md but unclaimed by any plan — delivered, flagged for traceability only.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | Pre-existing (`a8613bc8`, 2025-07-23); `views.py` has 0 commits since the prior pass. Outside the regression window; classified identically by all three passes. |
| `docs/runbooks/telescope_runs_calendar.rst` | 1099 | `` ``TBD window`` `` | ℹ️ Info | Not a debt marker — a documented skip-reason *value* the reconciler emits, listed beside `not approved` / `unresolved site`. No unfinished-work semantics. |
| `.planning/debug/resolved/34-updatestatus-receiver-attributeerror.md` | Evidence / Resolution | Previously ⚠️ Warning (F-34-1) | ✓ **RESOLVED** | The false claim is retracted in place with a dated `RETRACTED` block, `next_action` carries a `CORRECTION`, and `signal_real_path` is amended. Every corrected claim re-measured against the database this pass; all hold. Zero debt markers in the file. |
| `src/fomo_db.sqlite3` (data) | — | Previously ℹ️ Info (F-34-1's 14 stale events) | ✓ **RESOLVED** | Cleared by the operator sweep on 2026-09-14. Cross-tab shows zero terminal-status records with a queued/scheduled marker; dry run reports `updated: 0`. |
| `solsys_code/observation_projector.py` | 640 | Unguarded queryset evaluation in the D-11 loop header | ⚠️ Warning → 📋 Advisory | New-scope, file unmodified since the prior `verified:` timestamp, not a carried-forward gap. Evidenced and reproduced; routed to Phase 35. Does not falsify a Phase 34 must-have. See advisory 1. |

### Human Verification Required

**None.** Both items from the 2026-09-14T22:39:20Z report are closed and were re-measured rather than accepted:

1. *Reconcile the debug resolution with the developer database* — **closed.** The doc was corrected in `e3303e6`; every corrected claim independently re-measured this pass; the retraction preserves the original entry.
2. *Decide whether to spend the SCHED-06 evidence and clear F-34-1's 14 legacy events* — **closed.** The operator ran the sweep. The 14 events are repaired, the foreign namespaces are intact, and the corpus converges.

Advisory 1 is a defect report, not a human-verification item: it is fully measured, requires no human observation, and belongs to Phase 35's scope.

### Gaps Summary

**No gaps. No regressions. 14/14 truths verified. F-34-1 closed.**

The prior pass's only reason for withholding a pass was F-34-1 — a Phase 34 debug artifact asserting an overnight `updatestatus`-only run that the database showed had never happened, and 14 stale events it claimed had been repaired. Both halves are now closed, and I verified the closure rather than reading it.

The doc was corrected properly: the false Evidence entry is preserved and explicitly retracted rather than deleted, and a dated replacement entry states what actually happened. I re-measured every factual claim in that replacement against `src/fomo_db.sqlite3` — the modified-date distribution, the 14 observation_ids, the terminal-state exclusion reasoning, the migration, the sweep result, the latest event timestamp — and all of them hold, one of them (`2026-09-14T23:07:19Z`) to the microsecond. The claim I found most worth testing was the mechanism: that a *sweep* rather than the receiver cleared those 14. The database settles it independently of the narrative — all 14 titles now carry a resolved observatory site token (`COJ`, `LSC`, `CPT`, `TFN`) that only the sweep's `pre_fields_hook` can produce, and their `modified` stamps are spread 2-5 seconds apart across a 45-second window, the pacing of a network-bound site lookup. The receiver never performs that lookup.

The residue itself is gone, and I checked it with a stronger test than the prior pass used. Rather than tallying markers, I cross-tabulated every record's status against its event's marker: `COMPLETED` → `[O]` 76, `CANCELED` → `[C]` 6, `WINDOW_EXPIRED` → `[X]` 26, `FAILURE_LIMIT_REACHED` → `[F]` 1, `PENDING` → `[Q]` 38 / `[S]` 12, and nothing else — 159 rows, every marker correct for its status, which is the property F-34-1 was the last violation of. A dry-run sweep over a copy of the current database reports `created: 0, updated: 0, unchanged: 159`, and the foreign namespaces the projector must never touch (`RUN:` 72, blank-url 10, `ALLOC:` 0) came through a real 16-event write pass untouched.

Two must-haves are actually *better* evidenced than before. Truth 12's once-only site-resolution contract had only been shown in its read-back half; the 09-14 sweep exercised the full resolve-store-read-back cycle on live data. Truth 14's interrupted-sweep property had been shown against a copy; the pre-migration sweep attempt was a genuine interrupted sweep in production, and it left every already-processed record with a correct event and converged on re-run with no repair step.

Nothing drifted. No source file changed since the prior pass — the diff is two Markdown files — and the 342-test phase suite returns the same count and the same clean result.

One new finding is raised and I want it read rather than buried. `receiver_on_record_save()` evaluates `instance.campaign_run_links.select_related('run')` in the `for` header, outside any `try`, so a database error from that query escapes the receiver and propagates out of `ObservationRecord.save()`. I reproduced it deterministically on a scratch copy, and it is exactly what happened for real during the pre-migration sweep attempt. Phase 35's own regression test for this area patches `project_allocation` to raise, which is inside the per-link `try` — the link query itself has no coverage. I am not treating this as a Phase 34 failure, and the reasoning is specific rather than generous: TRIG-02 and truth 2 both scope the guarantee to "a projector error", the projector block is guarded and passes its test, the code is Phase 35's D-11 addition, and the phase goal is provably untouched because base projection commits before the failing step — which is why the 14 events were repaired on 09-14 *while* that step was erroring. It belongs to Phase 35, which has no verification report yet, so it is recorded here with its reproduction so it can be fixed rather than rediscovered.

The phase goal — one writer for observation-backed nights, drawn and kept current with no operator command — holds in the codebase and in the live database today.

---

_Verified: 2026-09-15T00:35:15Z_
_Verifier: Claude (gsd-verifier)_
