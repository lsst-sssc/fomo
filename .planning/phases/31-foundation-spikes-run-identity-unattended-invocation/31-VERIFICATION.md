---
phase: 31-foundation-spikes-run-identity-unattended-invocation
verified: 2026-09-02T21:05:00Z
status: passed
score: 41/41 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 34/34
  previous_verified: 2026-09-02T18:55:45Z
  trigger: "UAT gap G-31-3 (major) raised at 31-UAT.md test 3; closed by gap-closure plan 31-06"
  gaps_closed:
    - "G-31-3 — Gemini presented as a facility with real queue read-back in 31-DECISION.md and docs/design/run_identity_and_unattended_invocation_spike.rst; SOAR not named as the facility that actually has one"
  gaps_remaining: []
  regressions: []
  human_items_resolved:
    - "UAT test 1 (host identity) — pass"
    - "UAT test 2 (container image existence) — pass"
    - "UAT test 4 (six judgment-tier prohibitions from plans 31-01..31-05) — pass"
  human_items_new:
    - "Two judgment-tier prohibitions introduced by plan 31-06 (never human-resolved)"

decision_coverage:
  honored: 8
  total: 8
  not_honored: []
prohibitions_flagged: 2
human_verification:

  - test: "Accept (or reject) plan 31-06's prohibition: 'MUST NOT change any of the four verdicts (SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHED-07). This correction qualifies the evidence framing behind the SCHEMA-02 per-adapter tables; it does not reopen a settled decision, and no recommended field, constraint, key formula or invocation shape changes.' Non-authoritative verifier judgment: NOT violated. Evidence: `git diff --name-status ad6fd76..19146ee` touches 5 files, none under `solsys_code/` or `src/`; the complete set of deleted lines across both committed artifacts is 10, every one of which was re-added with its original text intact plus an appended dated qualification; all four verdict anchors are byte-present at HEAD (`null=True, blank=True`; `max_length=500` + `unique_campaign_run_source_identifier` + `source_identifier__isnull=False` at 31-DECISION.md:663-680; '**not sufficient**' at :806; `/usr/bin/flock -n /var/lock/fomo/<command-name>.lock` at :927), and the four `### SCHEMA-0x` / `### Scheduling track (SCHED-07)` verdict headings are unchanged in count and order."
    expected: "Operator confirms no settled decision was reopened — the correction is evidence framing only, as scoped."
    why_human: "Declared `verification: judgment`. Judgment-tier prohibitions are a documented soft-gate and are never silently passed by the verifier, however strong the mechanical evidence."
  - test: "Accept (or reject) plan 31-06's prohibition: 'MUST NOT delete or silently rewrite an earlier finding. Each corrected passage keeps its original claim visible and gains a dated qualification beside it, following the correction idiom this document already uses.' Non-authoritative verifier judgment: NOT violated. Evidence: the three deleted 31-DECISION.md lines are the Block (E) Gemini row, the SCHEMA-02 per-ingest-path Gemini row, and the `LCO_QUEUE/GEMINI_QUEUE/CLASSICAL_FILE` bullet — each replaced by the identical original sentence plus a '**Corrected 2026-09-02:**' clause, matching the document's existing 'Correction, recorded during the code-review fix pass' idiom. Nothing was removed without replacement."
    expected: "Operator confirms the correction idiom was followed and no earlier finding was quietly erased."
    why_human: "Declared `verification: judgment`. Whether a rewrite is 'silent' is a reading judgment, not a grep."
warnings:

  - id: W-A
    item: "`.planning/ROADMAP.md:155` still reads `**Plans**: 6 plans — 5 executed, 1 gap-closure plan pending`, and `:180` still carries `- [ ] 31-06-PLAN.md` unchecked, although 31-06 executed and committed (commits 6553526, d7dd012, da50ed6, 19146ee)."
    impact: "Bookkeeping only. ROADMAP.md is outside 31-06's declared `files_modified` and checkbox/plan-count updates are the phase-seal orchestrator's job, not the gap-closure plan's. Flagged so the seal step does not skip it."
  - id: W-B
    item: "`.planning/REQUIREMENTS.md:12-13` (SCHEMA-01, SCHEMA-02) still describe the target as a 'non-campaign LCO/Gemini queue observation' and 'Gemini observation ID', carrying the same facility framing G-31-3 corrected in the two committed artifacts. Likewise `ROADMAP.md:184` (Phase 32 goal) and `:193` (Phase 32 success criterion 3) still name Gemini as the second facility proving the pattern generalises."
    impact: "Not a Phase 31 gap. G-31-3 names exactly two artifacts and both are corrected; 31-06 explicitly and reasonably scoped requirements/roadmap edits out (a gap-closure plan does not rewrite the milestone contract). The consequence is recorded twice on surfaces a Phase 32 planner reads — `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` (listed at STATE.md:225) and 31-DECISION.md:1105-1125. Deferred to Phase 32, see Deferred Items."
  - id: W-C
    item: "`docs/design/run_identity_and_unattended_invocation_spike.rst:7-9` states facility read-back 'exists for the LCO path only'. SOAR is inside the LCO path (`SOARFacility(LCOFacility)`, handled inside `sync_lco_observation_calendar`), so the sentence is literally true, but a cold reader meets 'LCO only' 48 lines before the correction block that names SOAR."
    impact: "Cosmetic. Disambiguated at :26 ('LCO/SOAR queue observation') and fully at :57-59. No factual error."
  - id: W-D
    item: "Carried forward from the initial verification (W-5): `31-DECISION.md:195` retains one absolute developer path (`RESOLVED_DB_NAME=/home/tlister/git/fomo_devel/tmp/31-spike-db-copy.sqlite3`) inside the constraint-probe transcript."
    impact: "Informational only — this repository's own checkout path, not host infrastructure or a credential. Re-scanned during this verification: zero email addresses and zero 32+ hex strings in either committed artifact."
  - id: W-E
    item: "`31-06-SUMMARY.md` records the Sphinx result as 'build succeeded, 12 warnings'; a clean build run during this verification reported 'build succeeded, 6 warnings'."
    impact: "None. The difference is build-directory state (incremental build against the repo's `_readthedocs/` tree vs. a clean scratch tree), not a regression. The load-bearing facts hold either way: the build succeeds, none of the warnings reference the spike page, and the page renders."
  - id: W-F
    item: "Carried forward from the initial verification (W-3, W-4): IN-03's hardcoded LCO identity literal in the git-excluded probe script, and the 31-REVIEW-FIX.md logic-verification caveat (probe-script logic changed without re-running the probes)."
    impact: "Unchanged by this gap closure. Both concern git-excluded `tmp/` scripts, not the committed deliverable. Initial verification independently resolved W-3 in the deliverable's favour against the real dev DB."
deferred:

  - truth: "CampaignRun.Source gains a SOAR_QUEUE value and its migration"
    addressed_in: "Phase 32"
    evidence: "Phase 32 owns all three adapters (ROADMAP.md:184-187, requirements ADAPT-01..05); `models.py:108-138`'s own Source docstring records that the queue values have no producer until ADAPT-01..03 rewires the adapters, so the value has nothing behind it until Phase 32 builds one. Hand-forward recorded at `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` and 31-DECISION.md:1113-1114."
  - truth: "ADAPT-03 and Phase 32 success criterion 3 re-target SOAR rather than Gemini"
    addressed_in: "Phase 32"
    evidence: "ROADMAP.md:193 is Phase 32's own success criterion; rewriting it is Phase 32's planning input, not a Phase 31 gap-closure edit. Recorded at 31-DECISION.md:1115-1118 and in the pending todo."
  - truth: "Phase 33's outcome propagation carries an explicit Gemini-infeasibility caveat"
    addressed_in: "Phase 33"
    evidence: "ROADMAP.md:199 Phase 33 (Outcome Propagation & Window Narrowing) owns OUTCOME-01..04. Recorded at 31-DECISION.md:1119-1122 and in the pending todo."
---

# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation — Verification Report

**Phase Goal:** Settle, before any code lands, how a non-campaign queue observation gets a persistent `CampaignRun` identity and how the unattended jobs will actually be invoked on the real host.
**Verified:** 2026-09-02T21:05:00Z
**Status:** human_needed
**Re-verification:** Yes — after gap closure (G-31-3, plan 31-06)

## Verification posture

Re-verification mode. Plans 31-01..31-05 were verified 34/34 on 2026-09-02T18:55:45Z; those truths receive a regression check here (existence + anchor sanity), while the gap-closure work receives full three-level verification. The four specific questions the orchestrator asked were each answered against the codebase rather than against `31-06-SUMMARY.md`:

- the **factual basis of the correction** was re-derived from the installed `tom_observations` source tree and this repository's own commands, not read back from the debug session or the decision doc;
- the **"verdicts unchanged"** claim was tested by extracting *every deleted line* from the gap-closure commit range, not by trusting the summary's structural gate;
- the **investigation-only** claim was tested with `git diff --name-status` over the commit range plus `git status --porcelain -- solsys_code src`;
- the **Sphinx build** was executed here, twice, into a clean scratch tree.

## Goal Achievement

### Gap G-31-3 closure — the question asked

G-31-3 named exactly two artifacts and five line ranges in each. Line numbers drifted under editing, so each was located by anchor text.

| Artifact | UAT-named place | Corrected? | Evidence at HEAD |
|---|---|---|---|
| `31-DECISION.md` | :246 — Block (E) per-ingest-path Gemini row | ✓ | Now :252. Original cell text preserved; gains "**Corrected 2026-09-02:** the deeper reason no such row exists is not missing test data — it is that `GEMFacility` has no facility read-back at all (`gemini.py:490-492`, `gemini.py:506-507`) … This is a submission-echo path, not a facility read-back." |
| `31-DECISION.md` | :268 — the `Tag:` paragraph grouping Gemini with classical as constructed-input | ✓ | Now :274-280. States the two rows are constructed for *different reasons* — classical "does not have a real sample run through it yet", Gemini "cannot have a real counterpart at all, because the Gemini facility class exposes no read method that returns real state". |
| `31-DECISION.md` | :710 — SCHEMA-02 verdict per-ingest-path table (the row Phase 32 transcribes) | ✓ | Now :723. Every column value unchanged; "Where the value comes from" gains the synthesis explanation and "This is a submission-echo key, not a facility-read identifier, unlike the LCO row above." |
| `31-DECISION.md` | :759 — Phase 32 guidance bullet | ✓ | Now :772-776. "**Corrected 2026-09-02:** … SOAR, not Gemini, is the facility with real queue read-back, and Gemini's read-back is structurally absent (it is a submission-echo path only) rather than merely unimplemented." |
| `31-DECISION.md` | :773 — closing `Tag:` paragraph | ✓ | Now :793-798. Same distinction restated at the point of use. |
| `31-DECISION.md` | — (new) | ✓ | Consolidated correction section at :1043-1125, plus a pointer to it in the document header at :26-30 so a reader meets the correction before the tables. |
| design page | :5 — opening paragraph | ✓ | Now :4-9: "The identity scheme covers all three paths; the facility read-back that would let FOMO learn about an observation it did not itself submit exists for the LCO path only (see the correction below, before the Decisions tables)." |
| design page | :23 — Background | ✓ | Now :26-27: "a routine LCO/SOAR queue observation, a Gemini ToO submission **replayed onto the calendar**, or a classically scheduled night". |
| design page | :95-96 — per-ingest-path value row | ✓ | Now :109-112, with "**corrected 2026-09-02:** this key is FOMO's own synthesized string, echoing FOMO's own prior submission, not an identifier obtained from the facility". |
| design page | :112 — cardinality sentence | ✓ | Now :128-130: "per submitted Gemini Target-of-Opportunity **record** (**corrected 2026-09-02:** not per facility-side observation — FOMO never sees a facility-side Gemini observation)". |
| design page | :188 — credential naming convention | ✓ | Now :206-208, adding that SOAR authenticates through the same LCO portal. |
| design page | — (new) | ✓ | Correction block at :52-60 placed **above** both Decisions tables, naming `GEMFacility`'s stubs and `SOARFacility`'s real inherited portal read path; Future-scope entry at :269-273. |

**Adversarial residual scan.** Grepping both artifacts for any surviving un-caveated "Gemini queue / Gemini sync / queue visibility / see into" framing returns only the two lines that *state the correction itself* (31-DECISION.md:27 and :1058). The design page's remaining "three" occurrences all refer to *ingest paths* or *sync commands*, both of which are accurate — three ingest paths do exist and the identity scheme does cover all three; only read-back was ever the false claim.

### The correction's factual basis — independently re-derived

Every source claim the correction rests on was re-checked here against the installed tree at `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/`, not taken from the debug session.

| Claim in the correction | Independent result |
|---|---|
| `GEMFacility.get_observation_url()` returns `''`, real URL commented out | ✓ Confirmed verbatim — `return ''` with `# return PORTAL_URL + '/requests/' + observation_id` above it |
| `GEMFacility.get_observation_status()` returns a fixed dict regardless of argument | ✓ Confirmed verbatim — `return {'state': '', 'scheduled_start': None, 'scheduled_end': None}` |
| `sync_gemini_observation_calendar` reads only FOMO's own local rows | ✓ `sync_gemini_observation_calendar.py:40` = `ObservationRecord.objects.filter(facility='GEM')` |
| That command never imports `GEMFacility` and makes no outbound call | ✓ Grep for `GEMFacility`, `requests.`, `http` across the file returns **nothing** |
| Its identity key is locally synthesized | ✓ `:150` = `url = f'GEM:{prog}/{record.observation_id}'`, exactly as cited |
| `SOARFacility` subclasses `LCOFacility` at `soar.py:240` | ✓ `240:class SOARFacility(LCOFacility):` |
| The inherited read path is a live portal GET at `ocs.py:1548` | ✓ `get_observation_status()` issues `make_request('GET', urljoin(portal_url, f'/api/requests/{observation_id}'))` |
| SOAR is already a distinct facility inside the LCO sync command | ✓ `sync_lco_observation_calendar.py:289` = `facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}`; `:298` filters `facility__in=['LCO', 'SOAR']` |
| `CampaignRun.Source` declares `GEMINI_QUEUE` and no `SOAR_QUEUE` | ✓ `models.py:108-138` — `WEB, CLASSICAL_FILE, LCO_QUEUE, GEMINI_QUEUE, ESO_QUEUE, CSV_IMPORT, LEGACY`; no SOAR member |
| The limitation was recorded once in v1.5 and lost | ✓ `.planning/milestones/v1.5-REQUIREMENTS.md:43` = "Live Gemini ODB status polling \| `GEMFacility.get_observation_status()` is a stub returning empty state", plus `:36` GEM-GPP-02 naming the constructed key |

The correction is not merely plausible — every citation in it resolves.

### Verdict integrity — "were any of the four decisions altered?"

Tested by extracting the complete set of deleted lines across the gap-closure commit range for both committed artifacts (`git diff ad6fd76..19146ee -- <artifacts> | grep '^-'`). **Ten** lines were deleted in total; all ten were re-added with their original claim intact plus an appended dated qualification. None is a verdict statement.

| Verdict | Anchor at HEAD | Status |
|---|---|---|
| SCHEMA-01 — nullable FK | `31-DECISION.md:567` heading; `null=True, blank=True` present, unchanged | ✓ UNCHANGED |
| SCHEMA-02 — `source_identifier` + constraint | `:660` heading; `CharField(max_length=500, null=True, blank=True)` at :663, `condition=models.Q(source_identifier__isnull=False)` / `name='unique_campaign_run_source_identifier'` at :679-680 | ✓ UNCHANGED |
| SCHEMA-03 — classical tolerance match | `:800` heading; "**not sufficient**" at :806 | ✓ UNCHANGED |
| SCHED-07 — cron + `flock -n` | `:353` evidence heading; invocation shape `/usr/bin/flock -n /var/lock/fomo/<command-name>.lock` at :927 | ✓ UNCHANGED |

All four `### SCHEMA-0x` / `### Scheduling track (SCHED-07)` headings survive in count and order, all below `## Recommendation` (`:565`). 31-06 was, as scoped, evidence-framing-only.

### Investigation-only — "was anything under solsys_code/ or src/ touched?"

| Check | Command | Result |
|---|---|---|
| Working tree clean under both trees | `git status --porcelain -- solsys_code src` | empty |
| Gap-closure commit range touches no source | `git diff --name-status ad6fd76..19146ee` | 5 files: `STATE.md`, `31-06-SUMMARY.md`, `31-DECISION.md`, the new pending todo, the `.rst` — **zero** under `solsys_code/` or `src/` |
| Whole phase touches no source | `git diff --name-only <phase-range> -- solsys_code src` | empty |
| No migration added | `git diff --name-only ad6fd76..19146ee \| grep -i migration` | empty |
| Nothing from git-excluded `tmp/` committed | `git ls-files tmp/` | empty |
| Disposable DB copy removed | `ls tmp/31-spike-db-copy.sqlite3` | No such file |
| Real dev DB fingerprint unchanged | `stat -c '%s %Y' src/fomo_db.sqlite3` | `1208320 1788271090` — identical to the value recorded at plan 31-01 and at the initial verification |

### Plan 31-06 Must-Have Truths (full verification)

| # | Truth | Status | Evidence |
|---|---|---|---|
| P6.1 | SCHEMA-02 per-adapter tables distinguish a facility read-back path from a submission-echo path | ✓ VERIFIED | Both per-adapter tables (`31-DECISION.md:252` Block (E), `:723` verdict table) carry the distinction in the Gemini row itself, in one consistent vocabulary ("submission-echo" / "facility read-back"). The LCO row is described as a real portal read; the classical row's constructed status is distinguished by *reason* at :274-280 and :793-798. |
| P6.2 | Every place offering the Gemini path as evidence carries the caveat *at that place*, not only in a correction section elsewhere | ✓ VERIFIED | All five 31-DECISION.md touch points and all five design-page places carry an inline dated qualification — see the closure table above. This truth is the one most at risk of being satisfied by a single appended section; it is not: the correction section is *additional* to the five in-place edits, and each in-place edit is self-contained. |
| P6.3 | Both artifacts name SOAR as the facility with a real, API-backed queue read path, with source evidence | ✓ VERIFIED | `31-DECISION.md:1081-1089` names `SOARFacility` (`soar.py:240`), the inherited `get_observation_status()` (`ocs.py:1548`) live portal GET, and the existing `sync_lco_observation_calendar.py:289/298` handling — all four citations independently resolved above. Design page `:57-59` carries the same claim. |
| P6.4 | All four verdicts unchanged | ✓ VERIFIED | Verdict-integrity table above, derived from the complete deleted-line set rather than a heading count. |
| P6.5 | A Phase 32/33 planner who never opens 31-UAT.md still finds the consequence through the standard pending-todo surface, with a debug-session pointer | ✓ VERIFIED | `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` exists (5080 bytes) naming `SOAR_QUEUE` (:52, :58), ADAPT-03 (:60-61) and the debug session (:15, :83); `STATE.md:225-228` carries a matching bullet in the Pending Todos list alongside the four pre-existing entries; `.planning/debug/gemini-vs-soar-facility-scope.md` exists with `status: diagnosed`. Both key links resolve. |
| P6.6 | Phase 31 remains investigation-only: no file under `solsys_code/` or `src/`, no migration | ✓ VERIFIED | Investigation-only table above, verified at three levels (working tree, gap-closure range, whole-phase range). |
| P6.7 | Sphinx still builds with the corrected page; neither artifact carries a credential value or a contact address | ✓ VERIFIED | `python -m sphinx -b html docs <clean-tree>` → `build succeeded, 6 warnings`; none of the 6 references the spike page (they are `fomo._version` autoapi, two `docs/autoapi/fomo/urls/index.rst` docutils warnings, and an unrelated ESO notebook toctree warning). `design/run_identity_and_unattended_invocation_spike.html` produced (31529 bytes) containing `SOARFacility` and 3 rendered "corrected 2026-09-02" qualifications. Credential scan over both artifacts: 0 email matches, 0 matches for 32+ hex strings. |

### Regression check — plans 31-01..31-05 (previously 34/34)

| Group | Regression probe | Result |
|---|---|---|
| P1.1-P1.5, P1.B (31-01) | SCHEMA-01/02 evidence anchors (`NULL_CAMPAIGN_ROWS=`, `FINGERPRINT_BEFORE`, `unique_campaign_run_resolved_window`, `GUARD_DISPOSABLE_COPY`) | ✓ 19 occurrences intact; real-DB fingerprint still `1208320 1788271090`; `## Findings` at :32 |
| P2.1-P2.4, P2.B (31-02) | `## Recommendation` at :565, `### SCHEMA-01` at :567, `### SCHEMA-02` at :660, field + constraint declarations at :663-680 | ✓ Intact. Previous warning W-2 (stale `max_length=255` in `31-02-SUMMARY.md`) is now **resolved** — grep returns nothing; fixed by commit 84635d0 |
| P3.1-P3.4, P3.B (31-03) | `### SCHEMA-03` at :800, "**not sufficient**" at :806, `KNOWN_STATUSES` reconciliation (5 occurrences) | ✓ Intact |
| P4.1-P4.6, P4.B (31-04) | `### Scheduling track (SCHED-07)` at :353, `FINK_CREDENTIAL` / `_notify_staff` / `flock from util-linux` (10 occurrences), invocation shape at :927 | ✓ Intact |
| P5.1-P5.6 (31-05) | Design page present and corrected, toctree entry `docs/design/design.rst:48`, Sphinx build green, no source change, `git ls-files tmp/` empty, disposable copy absent, credential scan clean | ✓ Intact — all re-run, not carried forward |

Previous warning W-1 (31-DECISION.md's status header reading "In progress" and announcing a nonexistent `## Durable summary` section) is also **resolved**: the header now reads "**Status:** Complete." and describes the real structure.

**Score:** 41/41 truths verified (34 regression-checked from plans 31-01..31-05, 7 fully verified from plan 31-06); 0 present-but-behaviour-unverified.

### Roadmap Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| SC1 | Nullable-FK decision + non-campaign identity, backed by executable dev-DB evidence | ✓ VERIFIED | Unchanged by the correction; anchors re-checked (`:567`, `null=True, blank=True`, dev-DB fingerprint identical). |
| SC2 | Write-time identity field + constraint, per-adapter value, non-collision with both partial constraints | ✓ VERIFIED | `:660-742` intact. The correction qualifies *why* the Gemini row's value is constructed; it does not remove the value, the source line, or the non-collision result, all of which the criterion asks for. |
| SC3 | Explicit classical tolerance-match sufficiency answer with the failing case named | ✓ VERIFIED | `:800-870` untouched by 31-06. |
| SC4 | Unattended-invocation mechanism against the real host's constraints | ✓ VERIFIED | `:872-1016` untouched by 31-06; host-identity confirmation was resolved by UAT test 1 (pass). |
| SC5 | Decisions readable outside `.planning/`; test suite unchanged because no source behaviour changed | ✓ VERIFIED | Design page corrected and still wired at `docs/design/design.rst:48`; Sphinx build re-run green here; zero files under `solsys_code/` or `src/` across the whole phase including gap closure. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `31-DECISION.md` | Corrected framing at each touch point + consolidated correction section; `contains: GEMFacility` | ✓ VERIFIED | 1126 lines. `GEMFacility` present (:252, :1064-1073, :1100). All five touch points corrected in place; correction section at :1043-1125; header pointer at :26-30. |
| `docs/design/run_identity_and_unattended_invocation_spike.rst` | Same distinction on the published page; `contains: SOAR` | ✓ VERIFIED | 273 lines. `SOAR` present at :26, :57, :206-208, :270. Correction block sits *above* both Decisions tables, not buried in Future scope. Renders to HTML with the corrections visible. |
| `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` | Durable hand-forward; `contains: SOAR_QUEUE` | ✓ VERIFIED | Created (5080 bytes); `SOAR_QUEUE` at :52/:58; all three consequences recorded. |
| `.planning/STATE.md` | Pending-todo bullet | ✓ VERIFIED | :225-228, matching the format of the four existing entries. |
| `docs/design/design.rst` | Toctree entry (regression) | ✓ VERIFIED | :48 unchanged. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `31-DECISION.md` | `.planning/debug/gemini-vs-soar-facility-scope.md` | Correction section's evidence pointer naming the debug session and the UAT gap id | ✓ WIRED | `:1124-1125` — "Evidence pointer: `.planning/debug/gemini-vs-soar-facility-scope.md` (diagnosed session), gap `G-31-3` in `31-UAT.md`." Target exists, `status: diagnosed`. |
| pending todo | `.planning/debug/gemini-vs-soar-facility-scope.md` | Todo's evidence pointer | ✓ WIRED | Todo :15 and :83 both cite the session by path. Target exists. |
| `.planning/STATE.md` | pending todo | Bullet in Pending Todos list | ✓ WIRED | `STATE.md:225` cites the exact filename; file exists at that path. |
| `31-DECISION.md` | `solsys_code/models.py` | Proposed constraint stated against the two existing constraints (regression) | ✓ WIRED | `unique_campaign_run_resolved_window` / `unique_campaign_run_tbd_natural_key` cited; both still declared in `CampaignRun.Meta`. |
| `31-DECISION.md` | `src/fomo/settings.py` | Credential verdict cites the `FINK_CREDENTIAL` env-var pattern (regression) | ✓ WIRED | Pattern present in both. |
| `31-DECISION.md` | `solsys_code/campaign_views.py` | `_notify_staff` named as the layer Phase 34 extends (regression) | ✓ WIRED | Present in both. |
| `docs/design/design.rst` | design page | Toctree (regression) | ✓ WIRED | `:48`. |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|---|---|---|---|
| Documentation builds with the corrected page | `python -m sphinx -b html docs <clean-tree>` | `build succeeded, 6 warnings` | ✓ PASS |
| Corrected page actually renders | `ls <build>/design/run_identity_and_unattended_invocation_spike.html` | 31529 bytes | ✓ PASS |
| Corrections survive into rendered HTML | `grep "SOARFacility" / "corrected 2026-09-02"` on the HTML | `SOARFacility` present; 3 correction markers | ✓ PASS |
| No Sphinx warning references the new page | warning list inspected | 6 warnings, all pre-existing and unrelated | ✓ PASS |
| Source tree untouched | `git status --porcelain -- solsys_code src` | empty | ✓ PASS |
| No migration added | `git diff --name-only ad6fd76..19146ee \| grep -i migration` | empty | ✓ PASS |
| Real dev DB unmodified | `stat -c '%s %Y' src/fomo_db.sqlite3` | `1208320 1788271090` (unchanged) | ✓ PASS |
| No credential/email in committed artifacts | email + 32-hex regex scan on both | 0 / 0 | ✓ PASS |

Django test suite not run: no file under `solsys_code/` or `src/` changed anywhere in this phase, so there is no behaviour for it to exercise. (Per the project's own gotcha, `manage.py test` also pays the ~1.6 GB SPICE-kernel import cost and `test_views.TestEphemeris` segfaults — running it here would produce no new evidence at high cost.)

### Prohibitions

| # | Statement (abbreviated) | Tier | Verifier judgment | Disposition |
|---|---|---|---|---|
| 31-01 a | No finding recorded as evidence-backed unless its probe ran | judgment | Not violated | ✓ Human-resolved — UAT test 4, pass |
| 31-01 b | No write to the real dev database | judgment | Not violated (fingerprint identical) | ✓ Human-resolved — UAT test 4, pass |
| 31-02 a | No identity verdict silently covering only some ingest paths | judgment | Not violated — and materially *strengthened* by 31-06, which is what an honest labelling of the Gemini path looks like | ✓ Human-resolved — UAT test 4, pass |
| 31-03 a | Illustrative doc lines not treated as a real sample | judgment | Not violated | ✓ Human-resolved — UAT test 4, pass |
| 31-04 a | Container/AWS scopes not marked settled on a developer-shell probe | judgment | Not violated | ✓ Human-resolved — UAT test 4, pass |
| 31-05 a | No live credential value copied into a committed artifact | judgment | Not violated (scan clean) | ✓ Human-resolved — UAT test 4, pass |
| **31-06 a** | **MUST NOT change any of the four verdicts** | **judgment** | **Not violated** — deleted-line analysis + all four verdict anchors byte-present | ⚠️ **FLAGGED — unverified-prohibition, human review recommended** |
| **31-06 b** | **MUST NOT delete or silently rewrite an earlier finding** | **judgment** | **Not violated** — all 10 deleted lines re-added with original text plus a dated qualification | ⚠️ **FLAGGED — unverified-prohibition, human review recommended** |
| 31-06 c | MUST NOT do Phase 32's work (no new Source value, no migration, no source edit, no ADAPT-03/ROADMAP rewrite) | automated | Not violated | ✓ VERIFIED by deterministic evidence — `git diff --name-status` over the range shows 5 files, none under `solsys_code/`/`src/`, none a migration, and neither `ROADMAP.md` nor `REQUIREMENTS.md` among them |

The two flagged items are `verification: judgment`, which is a documented soft-gate: the verifier records a **non-authoritative** verdict and never silently passes them. Both are listed under Human Verification below. (Prohibition 31-06 c declares a non-standard tier value `automated` rather than `test`; it was nonetheless checked deterministically at verification time with the command and output recorded above, so it is not fail-closed-flagged.)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| SCHEMA-01 | 31-01, 31-02 | Spike settles whether `CampaignRun.campaign` becomes nullable so a non-campaign queue observation gets a persistent identity | ✓ SATISFIED | `31-DECISION.md:567-658` (Option A, nullable FK) — unchanged by gap closure. `REQUIREMENTS.md:91` marks Phase 31 / Complete. |
| SCHEMA-02 | 31-01, 31-02, **31-06** | Spike settles a `source_identifier`-style field plus its `UniqueConstraint` mapping each adapter's write-time key onto `CampaignRun` | ✓ SATISFIED | `:660-742` — field, constraint and all three per-adapter values intact, now with honest per-path evidence labelling. `REQUIREMENTS.md:92` Complete. This is the only requirement 31-06 declares. |
| SCHEMA-03 | 31-03 | Spike confirms whether the classical tolerance match is a sufficient write-time identity surface | ✓ SATISFIED | `:800-870` ("**not sufficient**", failing case named). `REQUIREMENTS.md:93` Complete. |
| SCHED-07 | 31-04 | Spike settles the scheduling mechanism against the real deployment's constraints | ✓ SATISFIED | `:872-1016` (cron + `flock -n`, credential handling, two-layer visibility, per-scope labelling). `REQUIREMENTS.md:94` Complete. |

**Orphan check:** `REQUIREMENTS.md:117` maps 4 requirements to Phase 31; the phase's plans declare exactly SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHED-07. **No orphaned requirements** — every ID mapped to Phase 31 is claimed by a plan, and every ID claimed by a plan is mapped to Phase 31.

Note (W-B): SCHEMA-01's and SCHEMA-02's *wording* in `REQUIREMENTS.md:12-13` still says "LCO/Gemini queue observation" / "Gemini observation ID". Both requirements are nonetheless satisfied as written — the decision doc does state a non-campaign identity and does show Gemini's write-time value and its non-collision. The wording is a milestone-contract concern handed forward to Phase 32, not a Phase 31 gap.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `31-DECISION.md` | 43, 94, 96, 209-210 | `TBD` | ℹ️ Info — **not** a debt marker | Domain vocabulary. `TBD` here is this project's own run-state term (`unique_campaign_run_tbd_natural_key`, "the TBD branch", "TBD population"), not an unresolved-work marker. |
| `.planning/STATE.md` | 53, 55, 175 | `TBD` | ℹ️ Info — not a debt marker | Same domain vocabulary in phase titles, plus one `**Plans**: TBD` placeholder for an unplanned future phase. |
| both committed artifacts | — | `FIXME` / `XXX` | — | None found. |

No blocker anti-patterns. No stub, orphan, hollow or disconnected artifact — this is a documentation-only phase, so Level-4 data-flow tracing does not apply; the equivalent check (does every cited source line actually resolve?) was run in full and is reported above.

### Human Verification Required

Automated verification is complete and clean: 41/41 truths verified, gap G-31-3 closed at both named artifacts, all four verdicts intact, phase confirmed investigation-only. Two items remain, both because they are declared `verification: judgment` rather than because evidence is missing.

#### 1. Accept plan 31-06's "verdicts unchanged" prohibition

**Test:** Confirm that correcting the Gemini framing did not reopen a settled decision.
**Expected:** The correction is evidence framing only — no recommended field, constraint, key formula or invocation shape changed.
**Verifier's non-authoritative judgment:** NOT violated. The gap-closure commit range deletes exactly ten lines across both committed artifacts, every one re-added with its original text plus an appended dated qualification; no verdict statement is among them. All four verdict anchors are byte-present at HEAD, and the four verdict headings are unchanged in count and order.
**Why human:** `verification: judgment` prohibitions are a documented soft-gate and are never silently passed by the verifier.

#### 2. Accept plan 31-06's "no silent rewrite" prohibition

**Test:** Read the three rewritten passages in `31-DECISION.md` (:252, :723, :772-776) and confirm each keeps its original claim visible beside the new dated qualification, in the document's existing correction idiom.
**Expected:** No earlier finding was quietly erased or restated as though it had always said the corrected thing.
**Verifier's non-authoritative judgment:** NOT violated. Each replacement preserves the original sentence verbatim and appends a "**Corrected 2026-09-02:**" clause, matching the existing "Correction, recorded during the code-review fix pass" idiom.
**Why human:** Whether a rewrite reads as "silent" is a reading judgment, not a grep.

### Gaps Summary

**No gaps.** UAT gap G-31-3 is genuinely closed, and closed better than the minimum the gap description asked for: rather than appending a single correction section and calling the artifacts fixed, 31-06 qualified the claim *in place* at all ten named touch points across both artifacts, so a reader who lands mid-document via search still meets the caveat. The correction's own factual basis — every one of the ten source citations behind it — was re-derived here from the installed `tom_observations` tree and this repository's commands, and all ten resolve.

The three consequences that reach past Phase 31 (a missing `SOAR_QUEUE` value, ADAPT-03's facility target, Phase 33's Gemini outcome-propagation infeasibility) were deliberately *not* actioned. That was the right call and is verified as such: actioning them would have required a source edit and a migration, falsifying the already-verified investigation-only must-have and breaking roadmap Success Criterion 5. They are instead recorded on two surfaces a future planner actually reads, with a pointer to the diagnosed debug session — a direct response to the debug session's own finding that this exact limitation was recorded once in v1.5 and lost.

What remains is bookkeeping (W-A: ROADMAP's plan count and the 31-06 checkbox) and inherited milestone-contract wording (W-B), neither of which is this phase's deliverable, plus two judgment-tier prohibitions awaiting the operator's explicit acceptance.

---

_Verified: 2026-09-02T21:05:00Z_
_Verifier: Claude (gsd-verifier)_
