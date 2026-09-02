---
phase: 31-foundation-spikes-run-identity-unattended-invocation
verified: 2026-09-02T18:55:45Z
status: human_needed
score: 34/34 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 8
  total: 8
  not_honored: []
prohibitions_flagged: 6
human_verification:
  - test: "Confirm the shell this phase ran its host probes in is the same machine D-01 refers to — the local Rocky 9 / WSL2 install FOMO actually runs on — and not a look-alike sandbox. `31-DECISION.md`'s interim-host evidence section quotes the kernel string, the init-system version and the crontab it found; the crontab entries reference a sibling checkout path."
    expected: "The redacted paths and cron entries are yours, so the interim-host findings (flock present, 0/3 unguarded Django management-command entries, heartbeat egress HTTP 301) stand as evidence about the real target host."
    why_human: "Only the operator can confirm host identity. Corroborating evidence found during this verification: the probe transcript's kernel string `5.14.0-687.42.1.el9_8.x86_64` matches the machine this verification is running on, which is also the machine holding the real FOMO checkout and the dev database whose fingerprint the probes recorded — but 'this is the machine FOMO is deployed on' remains a claim only the operator can settle."
    source: "31-04-SUMMARY.md Task 1 human-check (carried forward in 31-05-SUMMARY.md)"
  - test: "Confirm whether a FOMO container image exists anywhere yet, and if so where its build definition lives. This repository tracks none (independently re-confirmed: `git ls-files` shows no Dockerfile/Containerfile/compose/Helm/K8s manifest)."
    expected: "Either no FOMO image/build definition exists outside this repo — in which case confirm that whoever writes it inherits the requirement to install a cron daemon and the lock utility inside it — or one exists, in which case the container row of `31-DECISION.md`'s scope table must be re-checked against it before Phase 34 builds the scheduler entry point."
    why_human: "An image or build definition living outside this repository is unreachable from any probe run in this checkout."
    source: "31-04-SUMMARY.md Task 2 human-check (carried forward in 31-05-SUMMARY.md)"
  - test: "Read `docs/design/run_identity_and_unattended_invocation_spike.rst` as a reader who was not part of this phase. Both Decisions tables should be actionable without opening `31-DECISION.md`, and every unconfirmed scope — the container image, the AWS target, and the classical-schedule-file question — should be visible on the page itself."
    expected: "No verdict reads as more settled on the page than the evidence behind it supports; if one does, the wording is corrected before the phase seals."
    why_human: "Whether prose overstates confidence is a judgement call, not a grep."
    source: "31-05-SUMMARY.md Task 2 human-check"
  - test: "Explicitly accept (or reject) the six judgment-tier prohibitions carried by this phase's plans — see the Prohibitions table in this report. Each was assessed as NOT violated during this verification, with the evidence recorded, but every one is declared `verification: judgment` and so requires human resolution rather than an automated pass."
    expected: "Operator confirms each prohibition holds, in particular the two evidence-integrity ones (no finding recorded as evidence-backed unless the probe behind it ran; no committed artifact carries a credential value)."
    why_human: "`verification: judgment` prohibitions are a documented soft-gate — never silently passed by the verifier."
    source: "must_haves.prohibitions across plans 31-01..31-05"
warnings:
  - id: W-1
    item: "31-DECISION.md's own status header (lines 4-11) still reads `**Status:** In progress.` and announces a `## Durable summary` section that does not exist in the file (the heading inventory ends at `### SCHED-07 - unattended invocation mechanism`)."
    impact: "A Phase 32/34 reader opening the sealed deliverable sees it labelled in-progress and looks for a section that was never written. All required content IS present (under `## Recommendation`) and the durable-summary role is filled by the published docs page, so this is an accuracy defect in the document's own header, not a missing deliverable."
  - id: W-2
    item: "`31-02-SUMMARY.md` (lines 33, 86) still records `source_identifier CharField(max_length=255, ...)`, the unvalidated width that review finding WR-13 corrected. The deliverables both now say `max_length=500` (31-DECISION.md:650, docs page:79), and 31-DECISION.md explains the discrepancy explicitly."
    impact: "Low. A Phase 32 executor who reads the plan SUMMARY rather than the decision doc could transcribe the wrong width into a migration, where `max_length` is baked in at write time."
  - id: W-3
    item: "Review finding IN-03 (Info, deliberately not fixed) noted that `tmp/31_constraint_probe.py:170` hardcodes the LCO identity literal while its comment asserts dev-DB provenance, and 31-DECISION.md inherits that claim with the tag **Confirmed against real rows**."
    impact: "None to the conclusion — this verification independently queried the real dev DB read-only and found `https://observe.lco.global/requests/4247146` present exactly once in `tom_calendar_calendarevent` (10 LCO request URLs total), so the provenance claim is substantively true even though the probe did not establish it. Recorded because it is the one place a confidence tag rests on a claim the probe itself did not demonstrate."
  - id: W-4
    item: "The `## Logic-verification caveat` in 31-REVIEW-FIX.md stands: the review-fix pass changed probe-script logic (CR-02 guard rewrite, CR-03/WR-06 Block B, WR-11 baseline check) without re-running the probes. Those scripts live under git-excluded `tmp/` and are not part of the committed deliverable."
    impact: "None to this phase's committed evidence (the transcripts are the pre-fix as-run record and were verified verbatim here). Relevant only if the probes are re-run in a later phase."
  - id: W-5
    item: "31-DECISION.md:189 retains one absolute developer path (`RESOLVED_DB_NAME=/home/tlister/git/fomo_devel/tmp/31-spike-db-copy.sqlite3`) inside the constraint-probe transcript, after the WR-08 redaction pass removed venv/checkout/hostname detail elsewhere."
    impact: "Informational only — it is this repository's own checkout path, not host infrastructure or a credential. No credential, email address or long-hex secret was found in either committed artifact (scans run during this verification)."
---

# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation — Verification Report

**Phase Goal:** Settle the two structural questions this milestone cannot proceed without — how a routine, non-campaign queue observation acquires a persistent `CampaignRun` identity, and how unattended invocation actually works on the real target host — against real rows and the real deployment, so every later phase executes a decision instead of making one.
**Verified:** 2026-09-02T18:55:45Z
**Status:** human_needed
**Re-verification:** No — initial verification.

## Verification posture

This is an investigation-only spike: the *deliverable is the evidence and the decision derived from it*, so every load-bearing number in `31-DECISION.md` was re-derived independently rather than read back from the document. Specifically, this verification:

- re-queried the real dev database read-only (`file:src/fomo_db.sqlite3?mode=ro`) and reproduced every SCHEMA-01 figure;
- byte-compared the quoted probe blocks in `31-DECISION.md` against the raw git-excluded transcripts in `tmp/`;
- re-ran the blast-radius greps the document publishes and compared the site lists;
- read the real parser source to check the SCHEMA-03 failure narrative against actual control flow;
- ran the Sphinx build; and
- checked whether the code-review fixes actually landed in the current document, not whether the fix report says they did.

## Goal Achievement

### Roadmap Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| SC1 | Decision doc states whether `CampaignRun.campaign` becomes nullable and what a non-campaign queue observation's persistent identity looks like, backed by executable evidence against real dev-DB rows | ✓ VERIFIED | `31-DECISION.md:554-645` states Option A (nullable FK, `null=True, blank=True`, `on_delete` unchanged) with the exact field declaration and a single-`AlterField` migration. Independently reproduced against the real DB read-only: `TOTAL 49 / NULLCAMP 0 / RESOLVED 43 / TBD 6 / DUPWIN 4 / DUPTBD 0` and the per-source breakdown — every figure matches `31-DECISION.md:44-63` exactly. Identity = nullable campaign + `source_identifier`. |
| SC2 | Doc states the write-time identity field and its uniqueness constraint, and shows for each of the three adapters which value it writes and that it cannot collide with either existing partial constraint | ✓ VERIFIED | `31-DECISION.md:647-742`: `source_identifier CharField(max_length=500, null=True, blank=True)` + `UniqueConstraint(fields=('source_identifier',), condition=Q(source_identifier__isnull=False), name='unique_campaign_run_source_identifier')`. Per-ingest-path table names the value, the source line and the availability for all three commands. Non-collision argued by disjoint field sets AND empirically re-exercised — Block (E) re-fired both existing constraints with the field present (transcript verified verbatim). Source-line citations checked: `sync_gemini_observation_calendar.py:150`, `sync_lco_observation_calendar.py:329/341`, `load_telescope_runs.py:207-216` all match. |
| SC3 | Doc answers explicitly whether the classical tolerance-windowed match is sufficient on its own, with the failing case named if not | ✓ VERIFIED | `31-DECISION.md:775-870`: "**not sufficient**", failing case named as two distinct proposals on the same telescope + instrument + night, plus the four candidate cases each dispositioned. Grounded in one real operator file (structure independently confirmed: 3 run lines, line 1 carrying a leading `digits.alnum.digits` token). |
| SC4 | Doc states the unattended-invocation mechanism against the real target host's constraints — overlap prevention, credential handling, missed-invocation visibility | ✓ VERIFIED (host-identity confirmation routed to human) | `31-DECISION.md:872-1016`: cron + `flock -n` on the interim host; per-command lock path convention; absolute interpreter/`manage.py` paths; skip-not-queue semantics; env-var credentials with explicit "never a command-line argument"; two-layer visibility (`_notify_staff()` extension + external dead-man's switch). Host facts verified verbatim against `tmp/31-host-probe.txt`. |
| SC5 | Decisions readable outside `.planning/` via a `docs/design/` page, and the test suite is unchanged because no source behaviour changed | ✓ VERIFIED | `docs/design/run_identity_and_unattended_invocation_spike.rst` (248 lines, both verdict tables + Future scope), wired at `docs/design/design.rst:48`. Sphinx build re-run during this verification: `build succeeded, 12 warnings`, none referencing the new page; `_readthedocs/html/design/run_identity_and_unattended_invocation_spike.html` produced. `git diff` across every phase-31 execution commit touches only `docs/design/*.rst` and `.planning/**` — zero files under `solsys_code/` or `src/`. |

### Plan Must-Have Truths

| # | Truth (plan) | Status | Evidence |
|---|---|---|---|
| P1.1 | Real, dated count of `CampaignRun` rows and null-campaign rows from `src/fomo_db.sqlite3` (31-01) | ✓ VERIFIED | 49 / 0, independently reproduced today. |
| P1.2 | DB size/mtime fingerprint recorded before and after every read-only probe and stated identical (31-01) | ✓ VERIFIED | `FINGERPRINT_BEFORE=FINGERPRINT_AFTER=1208320 1788271090`; `stat -c '%s %Y'` today still returns `1208320 1788271090`. |
| P1.3 | Every non-test source file reading the campaign FK listed with file:line citations (31-01) | ✓ VERIFIED | Greps re-run: class (a) returns exactly the 5 cited sites (`models.py:352`, `campaign_reconciler.py:176`, `campaign_tables.py:467`, `:538`, `campaign_attribution.py:397`); class (b) exactly the 2 cited (`campaign_reconciler.py:261,398`); class (d) genuinely zero (the only template hit is `event_form.html` reading `campaign_id`, as the doc states). Each line spot-read and matches the quoted expression. |
| P1.4 | Verbatim probe output showing, per candidate shape, collision behaviour against both existing constraints (31-01) | ✓ VERIFIED | Blocks (A)-(E) in `31-DECISION.md:188-222` match `tmp/31-constraint-probe.txt` verbatim, including the literal `UNIQUE constraint failed: ...` strings. |
| P1.5 | Every write-probe ran against the disposable copy; real DB fingerprint unchanged (31-01) | ✓ VERIFIED | `RESOLVED_DB_NAME=.../tmp/31-spike-db-copy.sqlite3`, `GUARD_DISPOSABLE_COPY=OK`; real-DB fingerprint still unchanged (above). |
| P1.B | *(backstop)* Two non-campaign runs shown to collide or not per shape, with the `IntegrityError` (or its absence) quoted verbatim (31-01) | ✓ VERIFIED | Block B records the taken branch explicitly ("second row created (pk=60) ... did NOT collide"); Blocks C/D quote the raised `IntegrityError` text verbatim. Transcript verified against the document. |
| P2.1 | Exactly one chosen schema shape under a Recommendation heading, with both rejected shapes and the evidence that ruled each out (31-02) | ✓ VERIFIED | `31-DECISION.md:552-601`, `#### Why not the other two` cites `DUP_TELINST_WINDOW_TUPLES_IGNORING_CAMPAIGN=4` for Option B and Block (D)'s split-not-eliminate result for Option C. |
| P2.2 | Identity field name, type, nullability and exact `UniqueConstraint` definition (31-02) | ✓ VERIFIED | `31-DECISION.md:649-669`. Width corrected to the probe-validated 500 with the discrepancy explained (WR-13 fix landed). |
| P2.3 | For each of the three commands, the value written and non-collision with both existing constraints (31-02) | ✓ VERIFIED | Six-column per-ingest-path table at `31-DECISION.md:706-710`, each row carrying its own confidence tag. |
| P2.4 | If the shape leaves the FK nullable, the read sites needing a null guard are handed to Phase 32 as a named obligation (31-02) | ✓ VERIFIED | `31-DECISION.md:603-625` lists all 5 class-(a) sites in priority order plus the 2 class-(b) pass-throughs. |
| P2.B | *(backstop)* What happens to an existing non-campaign run's identity if its proposal later acquires a real campaign (31-02) | ✓ VERIFIED | `31-DECISION.md:627-635`: FK updated null → real `TargetList`; same pk, same `source_identifier`; nothing re-pointed. Also carried onto the published page's Future-scope list. |
| P3.1 | Explicit answer on classical tolerance-match sufficiency (31-03) | ✓ VERIFIED | "**not sufficient**", `31-DECISION.md:777-781`. |
| P3.2 | The concrete failing case named (31-03) | ✓ VERIFIED | Two distinct proposals, same telescope + instrument + night, no partial-night window; plus the reassignment variant. |
| P3.3 | How many real classical schedule files were obtained, stated in those words if zero (31-03) | ✓ VERIFIED | "**1 real classical schedule file** was obtained ... `CLASSICAL_SAMPLE_FILES=1`". File confirmed present at `tmp/31-classical-samples/didymos_2026_july_classical_runs.txt`, git-excluded, with exactly the described structure (3 run lines, blank separator, 5-line non-classical block, trailing blank). |
| P3.4 | For every classical run state observed, whether a proposal code was present (31-03) | ✓ VERIFIED | Per-line table at `31-DECISION.md:285-289`; only `allocation` observed, correlation explicitly labelled n=1 and non-generalisable. |
| P3.B | *(backstop)* D-07's remembered status words reconciled against `KNOWN_STATUSES` (31-03) | ✓ VERIFIED | `KNOWN_STATUSES` read from `telescope_runs.py:36` = `{'allocation','proposed','confirmed','cancelled','not confirmed'}` — matches the doc exactly; `CampaignRun.RunStatus` (`models.py:96-106`) does declare `PLANNED`/`OBSERVED` — matches the doc exactly. The narrative that line 1's rejection is *positional* was checked against real control flow: `parse_run_line` raises the leftover-token error at `telescope_runs.py:472` **before** `_resolve_telescope`, which is exactly why the observed error is `Unrecognized status 'EFOSC2'` rather than an unknown-telescope error. The document's account is correct in a way that could only come from having actually run it. |
| P4.1 | Mechanism stated against verified real-host facts, not a recommendation (31-04) | ✓ VERIFIED | See SC4. Host-identity confirmation routed to human verification. |
| P4.2 | Verbatim, dated terminal output proving flock presence and heartbeat egress, run during this phase (31-04) | ✓ VERIFIED | `flock from util-linux 2.37.4`, exit 0; heartbeat `301`, curl exit 0 — both match `tmp/31-host-probe.txt` verbatim under the stated redactions. |
| P4.3 | Overlap prevention stated concretely: invocation shape, lock-file path convention, fail-fast flag (31-04) | ✓ VERIFIED | `31-DECISION.md:898-931`: `/usr/bin/flock -n /var/lock/fomo/<command>.lock <venv>/python <checkout>/manage.py <cmd>`, one lock per command, skip-not-queue semantics stated. |
| P4.4 | Credential handling extends the env-var pattern, with an explicit "never a command argument" (31-04) | ✓ VERIFIED | `31-DECISION.md:941-965`; `FINK_CREDENTIAL_*` pattern confirmed live at `src/fomo/settings.py:309-311`. |
| P4.5 | Missed-invocation visibility as two independent layers (31-04) | ✓ VERIFIED | In-command `_notify_staff()` extension (confirmed at `campaign_views.py:326`, with the real `build_absolute_uri` at `:338` and `fail_silently=True` at `:344` — the doc's adaptation requirement is accurate) plus an external dead-man's switch. |
| P4.6 | Each deployment scope labelled separately; no scope confirmed on another scope's probe (31-04) | ✓ VERIFIED | Three-scope table at `31-DECISION.md:546-550`: interim host **Reached**; container **Unconfirmed**; AWS **Unconfirmed by construction**. Stand-in image results explicitly tagged constructed-input with "no tag upgrade is applied". |
| P4.B | *(backstop)* Whether existing crontab entries already run management commands without an overlap guard (31-04) | ✓ VERIFIED | 0 of 3 unguarded, recorded as live risk. Raw transcript confirms: three unguarded `manage.py` lines, two `flock -n` lines belonging to an unrelated project (correctly excluded), one commented-out entry (correctly excluded). |
| P5.1 | A `docs/design/` page carries both verdicts (31-05) | ✓ VERIFIED | Two `list-table` decision tables (schema/identity and scheduling) plus Background and Future scope. |
| P5.2 | Page wired into the Sphinx toctree (31-05) | ✓ VERIFIED | `docs/design/design.rst:48`. |
| P5.3 | Sphinx build succeeds with the page included (31-05) | ✓ VERIFIED | Re-run here: `build succeeded, 12 warnings`; HTML output produced for the new page. |
| P5.4 | Test suite unchanged because no file under `solsys_code/` or `src/` was modified (31-05) | ✓ VERIFIED | `git status --porcelain -- solsys_code src` empty **and** the phase's full commit range touches no path under either tree. No PLAN in the phase declared a `files_modified` entry under those trees either (all five declare only `.planning/**`, `tmp/**`, `docs/design/*.rst`) — so the claim is verified at both the plan-declaration and the executed-diff level. |
| P5.5 | Nothing under git-excluded `tmp/` committed; disposable DB copy removed (31-05) | ✓ VERIFIED | `git ls-files tmp/` empty; `tmp/31-spike-db-copy.sqlite3` absent. |
| P5.6 | Neither committed artifact contains a credential value, long hex secret or email address (31-05) | ✓ VERIFIED | Scans re-run over `31-DECISION.md` and the `.rst`: zero email matches, zero 32+ hex matches, zero venv/site-packages/conda paths, heartbeat ping-id redacted to `<ping-id>`. (One repo-checkout path remains — see W-5.) |

**Score:** 34/34 truths verified (0 present, behaviour-unverified)

### Code-review fix verification (independent of the fix report's claims)

The prompt asked specifically whether the review fixes actually resolved what they claimed. Checked in the current committed state, not in `31-REVIEW-FIX.md`:

| Finding | Claimed fix | Landed? | Evidence in current state |
|---|---|---|---|
| CR-01 / IN-06 | Remove stray `</content>` from the published page and 31-05-SUMMARY | ✓ Yes | `grep -rn '</content>' docs .planning/phases/31-*` returns only the review/fix documents discussing it. Sphinx warning count dropped 13 → 12, matching. |
| CR-04 | Stop implying the classical key mirrors the tolerance match | ✓ Yes | Docs page:96-110 now specifies a 5-minute **bucket**, states a boundary-straddling pair still splits, and flags that the probe used a `datetime.date` not a `datetime`. `31-DECISION.md:727-742` carries the same correction. |
| **CR-05** | Stop calling the new constraint "proven" | ✓ **Yes — the exact question asked** | Docs page:83-91 now scopes "proven, not just argued" to the *additive* property (which was tested) and states the new constraint's own duplicate-rejection / NULL-coexistence behaviour "is asserted from Django/SQLite partial-unique-index semantics, not independently measured by a positive/negative control in the probe; close that gap before relying on it as tested." That is an accurate description of what Block (E) did and did not do. |
| **CR-06** | Split the flock decision by deployment target | ✓ **Yes — the exact question asked** | Both artifacts now split it. `31-DECISION.md:877-888` and docs page:152-171 state that a container-local `flock` gives **no** cross-pod mutual exclusion (rolling update / restarted pod / replicas > 1 each get their own writable layer) and that the correct construct is a `CronJob` per command with `concurrencyPolicy: Forbid` + `startingDeadlineSeconds`, `flock` retained only as a same-pod guard. No universal-transfer claim survives anywhere in either artifact. |
| WR-02 | Correct "this project's own management commands" | ✓ Yes | All occurrences now attribute the three cron entries to the third-party `tom_jpl` / `tom_dataservices` plugins in a sibling checkout. Independently confirmed: `grep -rn "rundataquery\|updatescout" --include=*.py .` finds nothing in this repository. |
| WR-05 | Correct the false "never selects contact_person" claim | ✓ Yes | `31-DECISION.md:35-41` states the weaker, true guarantee (no individual value printed, only a count). |
| WR-07 | Stop overclaiming the four-collision figure's scope | ✓ Yes | Docs page:60-63 now says the recurrence rate "among today's 49 campaign-bearing rows ... is demonstrably non-zero" rather than measuring the future collision. |
| WR-08 | Redact developer-host infrastructure detail | ✓ Yes (one residual, W-5) | Hostname, kernel build string, init version, venv/checkout paths, the commented-out entry and the unrelated project's entries are all redacted in the committed quotation; the raw `tmp/31-host-probe.txt` confirms exactly what was removed and that nothing load-bearing was. |
| WR-13 | Record the validated field width | ✓ Yes | `max_length=500` in both artifacts; the stale 255 survives only in a plan SUMMARY (W-2). |
| IN-03 | *(not fixed — Info)* | n/a | Independently resolved in the deliverable's favour: the hardcoded LCO URL does exist in the real dev DB. See W-3. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `.planning/phases/31-.../31-DECISION.md` | Findings + Recommendation for SCHEMA-01/02/03 and SCHED-07 | ✓ VERIFIED | 1017 lines; `## Findings` and `## Recommendation` present with all four required verdict subsections. |
| `docs/design/run_identity_and_unattended_invocation_spike.rst` | Durable reader-facing summary of both verdicts | ✓ VERIFIED | Present, two `list-table`s, points back to `31-DECISION.md` with the archival-path caveat. |
| `docs/design/design.rst` | Toctree entry | ✓ VERIFIED | Line 48. |
| `tmp/31_dbsnapshot_probe.py`, `tmp/31-dbsnapshot.txt` | Probe + transcript | ✓ VERIFIED | Present, git-excluded, transcript matches the document. |
| `tmp/31_constraint_probe.py`, `tmp/31-constraint-probe.txt` | Probe + transcript | ✓ VERIFIED | Present, git-excluded, transcript matches the document. |
| `tmp/31-host-probe.txt`, `tmp/31-container-probe.txt` | Live host/container transcripts | ✓ VERIFIED | Present, git-excluded, match the redacted quotations. |
| `tmp/31-classical-samples/` | Real operator schedule file | ✓ VERIFIED | One file, git-excluded, structure matches the recorded description. |
| `tmp/31-spike-db-copy.sqlite3` | Disposable copy, removed at close-out | ✓ VERIFIED (absent by design) | Confirmed removed. |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| `tmp/31_dbsnapshot_probe.py` | `31-DECISION.md` | transcript quoted verbatim (`NULL_CAMPAIGN_ROWS=`) | ✓ WIRED |
| `tmp/31_constraint_probe.py` | `31-DECISION.md` | transcript quoted verbatim (`unique_campaign_run_resolved_window`) | ✓ WIRED |
| `31-DECISION.md` | `solsys_code/models.py` | proposed constraint stated against the two declared constraints | ✓ WIRED (names confirmed at `models.py:291,302`) |
| `31-DECISION.md` | `solsys_code/telescope_runs.py` | SCHEMA-03 argued against `KNOWN_STATUSES` / `ParsedRun` | ✓ WIRED (`telescope_runs.py:36`) |
| `31-DECISION.md` | `src/fomo/settings.py` | credential verdict cites `FINK_CREDENTIAL_*` | ✓ WIRED (`settings.py:309-311`) |
| `31-DECISION.md` | `solsys_code/campaign_views.py` | notification layer names `_notify_staff` | ✓ WIRED (`campaign_views.py:326`) |
| `docs/design/design.rst` | the new spike page | toctree entry | ✓ WIRED |
| the new spike page | `31-DECISION.md` | Future-scope pointer with archival caveat | ✓ WIRED |

All 8 key links across the five plans verified by `gsd-tools query verify.key-links` (`all_verified: true` for every plan) and re-checked by hand against the cited source lines.

### Data-Flow Trace (Level 4)

| Artifact | Value | Source | Produces real data | Status |
|---|---|---|---|---|
| `31-DECISION.md` SCHEMA-01 block | 49 / 0 / 43 / 6 / 4 / 0 + source breakdown | `src/fomo_db.sqlite3` | Yes — re-queried read-only during this verification, all values reproduced | ✓ FLOWING |
| `31-DECISION.md` SCHEMA-02 block | Blocks (A)-(E), 19 `PASS:` lines | `tmp/31-constraint-probe.txt` | Yes — verbatim match | ✓ FLOWING |
| `31-DECISION.md` SCHEMA-03 block | 1 file / 3 lines / 1 rejection | `tmp/31-classical-samples/*.txt` + `parse_run_line` | Yes — file structure and parser control flow both independently confirmed | ✓ FLOWING |
| `31-DECISION.md` SCHED-07 block | flock 2.37.4 / 0-of-3 / HTTP 301 | `tmp/31-host-probe.txt` | Yes — verbatim match under the stated redactions | ✓ FLOWING |
| `31-DECISION.md` container block | `NONE` + image inventory + stand-in results | `tmp/31-container-probe.txt` | Yes — verbatim match (registry hostname redacted as claimed) | ✓ FLOWING |
| docs page decision tables | Both verdicts | `31-DECISION.md` | Yes — no verdict on the page that the decision doc does not support; the page is, if anything, more hedged | ✓ FLOWING |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|---|---|---|---|
| Real dev DB matches every published SCHEMA-01 figure | read-only `sqlite3` queries via `python3` | 49 / 0 / 43 / 6 / 4 / 0, sources identical | ✓ PASS |
| Real dev DB untouched since the probes | `stat -c '%s %Y' src/fomo_db.sqlite3` | `1208320 1788271090` — identical to the recorded fingerprint | ✓ PASS |
| LCO identity value has genuine dev-DB provenance | read-only query for the exact URL | 1 exact match (10 LCO request URLs total) | ✓ PASS |
| Blast-radius class (a)/(b)/(d) site lists | the document's own published greps | 5 / 2 / 0 — exact match | ✓ PASS |
| Sphinx build with the new page | `python -m sphinx -M html ./docs ./_readthedocs ...` | `build succeeded, 12 warnings`; new page HTML produced | ✓ PASS |
| No container build file tracked | `git ls-files` search | none | ✓ PASS |
| No source tree modified by the phase | `git diff --name-only` across the phase commit range | only `docs/design/*.rst` + `.planning/**` | ✓ PASS |
| SCHEMA-03 parser narrative | read `telescope_runs.py:361-475` | leftover-token `ValueError` raised at `:472` **before** `_resolve_telescope` — exactly explains the recorded `Unrecognized status 'EFOSC2'` | ✓ PASS |
| Django test suite | not run | n/a | ? SKIP — no source file changed, so "the test suite is unchanged" is established by the diff; running it would import `ephem_utils` (~1.6 GB SPICE) and, per project notes, segfault in `TestEphemeris`. |

### Probe Execution

No `scripts/*/tests/probe-*.sh` exist in this repository and no plan declared one. The phase's "probes" are the git-excluded `tmp/*.py` evidence scripts, which are single-shot, mutate global model state, and require the disposable DB copy that close-out deliberately deleted — they are not re-runnable by design. Verification therefore compared their captured transcripts against the committed quotations byte-for-byte and independently re-derived the underlying facts from the real database and source tree, which is the stronger check.

| Probe | Command | Result | Status |
|---|---|---|---|
| `tmp/31_dbsnapshot_probe.py` | not re-run (one-shot) | transcript verified verbatim; all figures independently reproduced against the real DB | ✓ PASS (equivalent evidence) |
| `tmp/31_constraint_probe.py` | not re-run (requires the deleted disposable copy; poisons the process) | transcript verified verbatim | ✓ PASS (equivalent evidence) |
| host / container probes | not re-run | transcripts verified verbatim under the stated redactions | ✓ PASS (equivalent evidence) |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| SCHEMA-01 | 31-01, 31-02, 31-05 | Settle whether `CampaignRun.campaign` becomes nullable so a non-campaign queue observation gets a persistent identity | ✓ SATISFIED | Option A locked at a blocking human checkpoint, with the full three-way comparison, the 0/49 backfill finding and the 5-site guard obligation. |
| SCHEMA-02 | 31-01, 31-02, 31-05 | Settle a `source_identifier`-style field plus its `UniqueConstraint` and map each adapter's write-time key onto it | ✓ SATISFIED | Field, width, constraint and all three per-adapter values stated; additive coexistence measured. Honest caveat retained: the new constraint's own duplicate-rejection was not negative-tested (disclosed on the page). |
| SCHEMA-03 | 31-03, 31-05 | Confirm whether the classical tolerance-windowed match is a sufficient write-time identity surface | ✓ SATISFIED | "Not sufficient"; failing case named; two separable Phase 32 follow-on items recorded. |
| SCHED-07 | 31-04, 31-05 | Settle the scheduling mechanism against the real target deployment, including credentials and overlap prevention | ✓ SATISFIED (scope-qualified) | Verified for the interim host; correct Kubernetes construct named and explicitly carried forward as an open item; container and AWS scopes labelled unconfirmed rather than assumed. The remaining confirmations are the two host/container human-checks below. |

**Orphaned requirements:** none — `REQUIREMENTS.md` maps exactly SCHEMA-01/02/03 and SCHED-07 to Phase 31, all four already marked Complete, and all four are claimed by plans.

### Prohibitions (judgment-tier — flagged, never silently passed)

| # | Prohibition (plan) | Verifier assessment | Disposition |
|---|---|---|---|
| 1 | No finding recorded as evidence-backed unless its probe actually ran; every finding carries one of the two fixed confidence tags (31-01) | Every evidence and recommendation section ends in an explicit `Tag: **Confirmed against real rows**` / `**Constructed-input code-path check**` line, and the tags are used discriminatingly (Gemini/classical values tagged constructed, LCO tagged real). One residual: the LCO value's tag rests on a hardcoded literal in the probe — independently confirmed true here (W-3). | `unverified` — human review recommended |
| 2 | Never write to `src/fomo_db.sqlite3`; fingerprint identical before and after (31-01) | Fingerprint recorded identical, and still identical at verification time. Write-probes resolved to the disposable copy. | `unverified` — human review recommended |
| 3 | No write-time identity verdict silently covering only some ingest paths (31-02) | All three commands appear in the recommendation table, each with its own confidence tag and availability note. | `unverified` — human review recommended |
| 4 | Never treat the three illustrative doc lines as a representative sample; say so in words if no real file was obtained (31-03) | A real file was obtained and inspected; the count is stated as `CLASSICAL_SAMPLE_FILES=1` and the sample's thinness is repeatedly flagged (n=1 vs n=2 correlation explicitly called "far too thin to generalize"). | `unverified` — human review recommended |
| 5 | Never mark the container or AWS scope settled on a developer-shell probe (31-04) | Both remain **Unconfirmed**; the stand-in image check is tagged constructed-input with an explicit "no tag upgrade is applied". | `unverified` — human review recommended |
| 6 | Never copy a live credential value into a committed artifact (31-05) | Scans re-run here: zero emails, zero 32+ hex strings, zero venv/site-packages paths, heartbeat ping-id redacted. | `unverified` — human review recommended |

All six are declared `verification: judgment`, so none can be marked green automatically. Consolidated into human-verification item 4.

### Decision Coverage

`gsd-tools query check.decision-coverage-verify` → **8 of 8** trackable `31-CONTEXT.md` decisions honored by shipped artifacts; `not_honored: []`. Non-blocking gate, recorded for drift tracking.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `31-DECISION.md` | 4-11 | Stale status header: "Status: In progress" and a promised `## Durable summary` section that does not exist | ⚠️ Warning | See W-1. No content is missing; the header misdescribes the document. |
| `31-02-SUMMARY.md` | 33, 86 | `max_length=255` — the value WR-13 corrected to 500 in the deliverables | ⚠️ Warning | See W-2. |
| `31-DECISION.md` | 189 | Absolute repo-checkout path inside a quoted transcript | ℹ️ Info | See W-5. Not a credential, not host infrastructure. |
| `31-DECISION.md` | various | `TBD` occurrences | ℹ️ Info — **not** debt markers | Every hit is domain vocabulary (`unique_campaign_run_tbd_natural_key`, the TBD-window branch for runs with a null `window_start`), not an unresolved-work marker. No `FIXME`/`XXX` anywhere in either committed artifact. |
| `docs/design/*.rst` | various | `placeholder` | ℹ️ Info — **not** a stub | Domain vocabulary for the rejected placeholder-`TargetList` candidate shapes. |

No blocker anti-patterns. No stub, orphan or hollow artifact.

### Human Verification Required

**1. Confirm the probed host is the operator's real FOMO machine** *(from 31-04 Task 1)*
- **Test:** Read `31-DECISION.md`'s interim-host evidence section. Confirm the redacted crontab entries and checkout paths are yours, on the Rocky 9 / WSL2 install FOMO actually runs on.
- **Expected:** They are yours, so the interim-host findings stand as evidence about the real target host. If not, every finding in that section is about an unrelated machine and must be re-run before Phase 34 builds against it.
- **Why human:** Only the operator can settle host identity. *Corroboration found during verification:* the transcript's kernel string `5.14.0-687.42.1.el9_8.x86_64` matches the machine this verification ran on, which also holds the real FOMO checkout and the dev database whose fingerprint the probes recorded — strong, but not a substitute for your confirmation.

**2. Confirm whether a FOMO container image or build definition exists outside this repository** *(from 31-04 Task 2)*
- **Test:** Say whether a FOMO image / build definition exists anywhere, and where.
- **Expected:** If none exists, confirm that whoever writes it inherits the requirement to install a cron daemon and the lock utility inside it. If one exists, the container row of the scope table must be re-checked against it before Phase 34 builds the scheduler entry point.
- **Why human:** Independently re-confirmed that this repository tracks no container build file of any kind; anything outside this checkout is unreachable from here.

**3. Confirm the published design page reads as actionable without overstating confidence** *(from 31-05 Task 2)*
- **Test:** Read `docs/design/run_identity_and_unattended_invocation_spike.rst` cold. Both decision tables should be actionable without opening `31-DECISION.md`, and every unconfirmed scope (container image, AWS target, classical-file question) should be visible on the page itself.
- **Expected:** No verdict reads as more settled than its evidence supports.
- **Why human:** Tone and confidence calibration is a judgement call. *Verifier's own reading, offered as input not as a substitute:* the page is, if anything, more hedged than the evidence requires — it discloses the untested new-constraint behaviour, the classical bucket-boundary gap, the `datetime.date`-not-`datetime` probe caveat, the Kubernetes cross-pod flock limitation, and both unconfirmed deployment scopes, all in the decision cells rather than buried in Future scope.

**4. Accept the six judgment-tier prohibitions**
- **Test:** Review the Prohibitions table above and accept or reject each.
- **Expected:** Each holds; the two evidence-integrity ones (no unearned confidence tags; no credential value in a committed artifact) deserve the closest look, and W-3 is the one place worth a second opinion.
- **Why human:** All six are declared `verification: judgment` — a soft gate the verifier is not permitted to close on its own.

### Gaps Summary

No gaps. Every roadmap Success Criterion and every plan must-have is verified, and — unusually for a document-only deliverable — the load-bearing numbers were re-derived from the real database and the real source tree rather than read back from the document. The two evidence-integrity failures the code review found in the *substance* of the deliverable (CR-05's "proven" overclaim about the new constraint, CR-06's universal-flock-transfer claim) are genuinely repaired in the current text, not merely reported as repaired: the page now names the untested boundary explicitly, and the flock verdict is split by deployment target with the correct Kubernetes construct named.

What remains is not gaps but scope honesty plus five documentation warnings. The phase's own conclusions are correctly qualified: the interim host is reached, the container image and the AWS target are labelled unconfirmed, and the classical identity gap is named as accepted risk with two Phase 32 follow-on items rather than papered over. The status is `human_needed` solely because three deliberately deferred `<human-check>` items and six judgment-tier prohibitions require the operator's word, not because anything in the codebase or the deliverable is missing.

---

_Verified: 2026-09-02T18:55:45Z_
_Verifier: Claude (gsd-verifier)_
