---
phase: "31"
slug: "foundation-spikes-run-identity-unattended-invocation"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-01"
validated: "2026-09-02"
---

# Phase 31 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

This phase ships no source-code behavior change (Success Criterion 5: "the test suite is
unchanged because no source behaviour changed"). Precedent phases 18 and 26 added zero
`solsys_code/tests/` changes and instead treated their own investigation scripts/queries as the
evidence artifact — this phase follows the identical pattern: validation here means the
investigation evidence quoted verbatim in `31-DECISION.md`, plus a regression check that the
existing suite stays green.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` via `python manage.py test` (existing, unchanged by this phase) |
| **Config file** | none — no new test infrastructure needed |
| **Quick run command** | N/A — this phase's own deliverable is a decision doc, not test-covered code. Disposable investigation scripts run via `python manage.py shell < tmp/31_*.py` (git-excluded), mirroring `tmp/26_integrity_check.py` |
| **Full suite command** | `python manage.py test` (excluding `solsys_code.tests.test_views.TestEphemeris` — known native-ASSIST segfault, per project convention) |
| **Estimated runtime** | Same as current baseline — no new tests added |

---

## Sampling Rate

- **Per investigation step:** re-run the specific probe/query and record output verbatim in `31-DECISION.md`, exactly as `26-DECISION.md` and `18-DECISION.md` do.
- **After every plan wave / before phase gate:** `python manage.py test` (excluding `TestEphemeris`) to confirm the investigation activity left no accidental source edits behind — this phase's correctness bar for the *existing* suite is "unchanged," not "passing new tests."
- **Before `/gsd-verify-work`:** Full suite must be green; `31-DECISION.md` and the `docs/design/` page must both exist with all five roadmap success criteria addressed.
- **Max feedback latency:** N/A (no watch-mode / iterative test loop for this phase — investigation is query/read driven, not edit/test driven).

---

## Per-Task Verification Map

Reconciled against the five plans written on 2026-09-01. The substance is unchanged from this
file's draft; the plan and task ids below are the real ones, and the SCHEMA-03 track moved into
its own plan (31-03) because it opens with a blocking operator checkpoint that would otherwise
stall the schema-evidence plan.

Extended 2026-09-02 for gap-closure plan 31-06 (UAT gap G-31-3, wave 6) — three additional
document-assertion tasks correcting the Gemini/SOAR facility framing, added below in the same
row shape.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01-01 | 01 | 1 | SCHEMA-01 | T-31-01, T-31-02 | Read-only probe; no contact-field values selected | investigation script (tracer) | `python manage.py shell < tmp/31_dbsnapshot_probe.py` then assert `FINGERPRINT_UNCHANGED=PASS` and the count keys | ✅ — script written, transcript confirms `FINGERPRINT_UNCHANGED=PASS` | ✅ green |
| 31-01-02 | 01 | 1 | SCHEMA-01 | T-31-03 | N/A — read-only grep inventory | grep inventory, self-deriving | loop over `grep -rl '\.campaign\b'` non-test sources asserting each appears in `31-DECISION.md` | ✅ — real source tree | ✅ green — all read-path sites accounted for, re-verified independently at audit time |
| 31-01-03 | 01 | 1 | SCHEMA-02 | T-31-02 | Hard DB-path guard before any write; live DB fingerprint re-compared | disposable constraint-probe script | `python manage.py shell < tmp/31_constraint_probe.py` asserting `GUARD_DISPOSABLE_COPY=OK`, ≥5 `PASS:`, 0 `FAIL:` | ✅ — transcript confirms guard + 19 PASS / 0 FAIL | ✅ green |
| 31-02-01 | 02 | 2 | SCHEMA-01 | — | N/A | blocking-human decision checkpoint (one-way door, D-05) | none — checkpoint | N/A | ✅ green — human selected nullable-fk (Option A) via the orchestrator's real checkpoint gate, not auto-approved |
| 31-02-02 | 02 | 2 | SCHEMA-01 | T-31-05 | N/A | document assertion | heading + candidate-shape mention count inside the SCHEMA-01 recommendation section | ✅ | ✅ green |
| 31-02-03 | 02 | 2 | SCHEMA-02 | T-31-04, T-31-06 | Contact field named, never valued | document assertion | all three command names and all three `Source` values present inside the SCHEMA-02 section | ✅ | ✅ green |
| 31-03-01 | 03 | 3 | SCHEMA-03 | T-31-07 | Operator files stay in git-excluded `tmp/` | blocking-human action checkpoint | none — checkpoint | ✅ — operator supplied a real Didymos-campaign classical schedule file at `tmp/31-classical-samples/` | ✅ green |
| 31-03-02 | 03 | 3 | SCHEMA-03 | T-31-07, T-31-08 | No PI name, title, email or raw line quoted into the committed doc | sample inspection + document assertion | sample count reconciliation, plus a negative search for credential/email tokens in `31-DECISION.md` | ✅ — real file inspected (1 file, 3 real lines), PII/credential scan clean, re-verified independently at audit time | ✅ green |
| 31-03-03 | 03 | 3 | SCHEMA-03 | T-31-09 | N/A | document assertion | sufficiency verdict, tolerance reference and four failing-case mentions inside the SCHEMA-03 section | ✅ | ✅ green |
| 31-04-01 | 04 | 4 | SCHED-07 | T-31-10, T-31-13 | Crontab redacted at capture time, before anything is written | shell-level verification | `flock --version` (util-linux banner), redacted `crontab -l` (a `manage.py` entry), `curl -w '%{http_code}'` to the heartbeat host (1xx-5xx) | ✅ — real host | ✅ green — flock present (util-linux 2.37.4), 0/3 existing cron entries guarded, heartbeat HTTP 301 |
| 31-04-02 | 04 | 4 | SCHED-07 (container + AWS scope) | T-31-10 | Same, verified inside a real image where one exists | shell-level verification | tracked build-file search, local image inventory, then the same two probes inside whichever image branch applies | ✅ — ran against a `python:3.11-slim` stand-in (no FOMO image exists); container/AWS scopes correctly left unconfirmed | ✅ green (stand-in evidence, honestly labelled — not a gap) |
| 31-04-03 | 04 | 4 | SCHED-07 | T-31-11, T-31-12, T-31-14, T-31-15 | No credential value in any committed artifact (SCHED-10) | document assertion + negative credential search | invocation-shape and two-layer-visibility assertions, plus `! grep -REiq` for credential-shaped tokens in `31-DECISION.md` | ✅ | ✅ green |
| 31-05-01 | 05 | 5 | SCHEMA-01..03, SCHED-07 | T-31-16 | No credential, email or operator line on the published page | docs build precondition + document assertion | page structure, two decision tables, toctree entry, and a one-line-added diff assertion on `docs/design/design.rst` | ✅ | ✅ green — toctree diff confirmed exactly `1 0` (one line added, none removed) |
| 31-05-02 | 05 | 5 | Regression | T-31-17, T-31-18, T-31-19 | Disposable DB copy removed; nothing under `tmp/` tracked | Sphinx build + targeted suite + cleanliness gates | `sphinx-build -M html ./docs ./_readthedocs …` prints `build succeeded`; `python manage.py test` over six named modules ends `OK`; empty `git status -- solsys_code src`; empty `git ls-files tmp/` | ✅ — suite already exists | ✅ green — build succeeded, 143 tests OK, source tree and tmp/ clean, DB copy removed |
| 31-06-01 | 06 | 6 (gap closure, G-31-3) | SCHEMA-02 | T-31-23 | Original claims kept, dated qualification appended beside each — no earlier finding erased | document assertion (grep) | citation checks for `GEMFacility`, `gemini.py:490`/`:506`, `soar.py:240`, `ocs.py:1548`, the debug-session pointer and `G-31-3` in `31-DECISION.md`; ≥4 pre-heading / ≥6 total `submission-echo` occurrences; structural heading-order regression (4 verdict headings unchanged, ≥6 evidence subsections) | ✅ | ✅ green — all gates passed on first attempt (31-06-SUMMARY.md); independently re-derived at 31-VERIFICATION.md audit |
| 31-06-02 | 06 | 6 (gap closure, G-31-3) | SCHEMA-02 | T-31-23, T-31-24 | Same correction idiom carried to the published page; no library line numbers on a public doc | document assertion (grep) + docs-build + cleanliness gate | `SOAR`/`GEMFacility`/`SOARFacility`/read-path-distinction checks; page-structure regression grep (headings, list-tables, `31-DECISION`, ingest-path names, `flock`, unconfirmed-scope labels); empty `git status -- docs/design/design.rst`; `sphinx-build …` prints `build succeeded` | ✅ | ✅ green — build succeeded; independently re-derived at 31-VERIFICATION.md audit (W-E: warning count differs from a stale local build dir, not a regression — see 31-VERIFICATION.md) |
| 31-06-03 | 06 | 6 (gap closure, G-31-3) | SCHEMA-02 (hand-forward) | T-31-20, T-31-21, T-31-22 | No credential/contact value in any of the three touched artifacts; no edit under `solsys_code/`/`src/`; REQUIREMENTS.md/ROADMAP.md wording unchanged | document assertion + negative credential search + git status gates | new-todo grep checks (`SOAR_QUEUE`, `ADAPT-03`, `G-31-3`, debug-session pointer, `gemini.py:506`, `soar.py:240`) and STATE.md pending-list entry; empty `git status -- solsys_code src` and `-- solsys_code/migrations`; ADAPT-03/Phase-32-criterion presence checks in REQUIREMENTS.md/ROADMAP.md; three-artifact credential/contact negative regex scan | ✅ | ✅ green — all gates passed on first attempt; independently re-derived at 31-VERIFICATION.md audit (41/41 must-haves verified) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Why the regression run names modules rather than running the app suite unqualified:** project
convention (CLAUDE.md, and every v2.2 plan) forbids `python manage.py test solsys_code` on its own —
`solsys_code.tests.test_views.TestEphemeris` segfaults in native ASSIST and importing
`solsys_code.ephem_utils` triggers a very large SPICE kernel download. The six modules named in
31-05-02 cover the model whose schema this phase reasoned about, the three ingest paths whose
identity keys it inventoried, the reconciler that reads the campaign FK, and the canonical-record
migration; none of them imports the ephemeris utilities. The primary regression signal for an
investigation-only phase is the empty `git status` over `solsys_code/` and `src/`, with the test
run as an independent second signal.

---

## Wave 0 Requirements

- [ ] `tmp/31_dbsnapshot_probe.py` — disposable, git-excluded **read-only** probe against the real `src/fomo_db.sqlite3`, recording a `stat -c '%s %Y'` fingerprint before and after and printing `KEY=value` population counts. Written by task 31-01-01 (the tracer).
- [ ] `tmp/31_constraint_probe.py` — disposable, git-excluded probe script mirroring `tmp/26_integrity_check.py`'s pattern: write against a **disposable copy** of `src/fomo_db.sqlite3`, never the live file, to check all three candidate schema shapes and a candidate write-time identity field against both existing `UniqueConstraint`s. Must re-point the default connection and print `GUARD_DISPOSABLE_COPY=OK` before any write; must never touch `src/fomo/local_settings.py`, which holds a live credential on the real host. Written by task 31-01-03.
- [ ] Container-level shell check for the lock utility and outbound HTTPS reachability to the heartbeat host (Pitfall 5 in RESEARCH.md) — not a test file, a one-time verification step whose output is quoted verbatim in `31-DECISION.md`. Run by task 31-04-02. This repository tracks no container build file, so that task must first establish whether a FOMO image exists at all and label a stand-in check as a stand-in.
- [ ] No new `tests/*.py` file — this phase adds no source behavior to test.

*Existing `solsys_code/tests/` infrastructure covers all phase requirements that touch source code (none, by design) — Wave 0 here is investigation tooling, not test scaffolding.*

---

## Manual-Only Verifications

Both items below were resolved by real operator action during execution — retained here as a record of what made them manual, not as outstanding gaps.

| Behavior | Requirement | Why Manual | Resolution |
|----------|-------------|------------|-------------------|
| Classical schedule line carries a proposal code for specific run states ("planned"/"observed") | SCHEMA-03 | No real classical schedule file existed anywhere in this repository to automate against; needed operator-supplied sample or explicit "unavailable" finding | Operator supplied a real Didymos-campaign classical schedule file (`tmp/31-classical-samples/`, git-excluded). Inspected against `KNOWN_STATUSES`/`parse_run_line`; recorded in `31-DECISION.md` §SCHEMA-03. Verdict: tolerance match insufficient (two-proposals-same-night collision), `source_identifier` inherits the same blind spot — an incomplete mitigation flagged for Phase 32, not a fully closed question. |
| Container-image-level `flock`/network verification | SCHED-07 (D-03) | Required building/accessing the real FOMO container image; this repository tracks no container build file at all | No FOMO image exists to check. Ran the same two probes inside a `python:3.11-slim` stand-in instead (labelled as such, not upgraded to real-image confidence): `flock` present, `curl` absent. Container and AWS scopes remain explicitly unconfirmed in `31-DECISION.md`, per design — this phase correctly did not fabricate a confirmation for a scope it could not reach. |

Three `<human-check>` items remain open for the operator, routed here per `workflow.human_verify_mode=end-of-phase` (not Nyquist gaps — these are `<verify><human-check>` items from autonomous tasks, consolidated from 31-04-SUMMARY.md and 31-05-SUMMARY.md):

1. Confirm the shell 31-04 probed is really the operator's FOMO host (D-01's Rocky 9/WSL2 install), not a look-alike sandbox.
2. Confirm whether a FOMO container image or build definition exists anywhere outside this repository that 31-04 could not see.
3. Confirm the published `docs/design/run_identity_and_unattended_invocation_spike.rst` page reads as actionable to an outside reader and that no verdict reads more settled than the evidence in `31-DECISION.md` supports.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (both Wave 0 scripts were written and run by plan 31-01; the container-level check was run by plan 31-04, against a labelled stand-in)
- [x] No watch-mode flags
- [x] Feedback latency < N/A (investigation-only phase)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-09-02 — 0 gaps found across 17 tasks (15 automated, 2 checkpoint). All automated commands independently re-run/re-checked at audit time and confirmed passing. No `gsd-nyquist-auditor` subagent dispatch was needed — this phase adds no source-code test surface (investigation-only, per Success Criterion 5), so every requirement traces to a document/transcript assertion rather than a unit test, and all of those assertions passed on first execution with zero deviations across all six plans (including gap-closure plan 31-06).

## Validation Audit 2026-09-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed) |
| Escalated | 0 |

## Validation Audit 2026-09-02 (gap closure — plan 31-06)

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed) |
| Escalated | 0 |

Extended coverage for gap-closure plan 31-06 (UAT gap G-31-3): 3 tasks added to the Per-Task
Verification Map, all document-assertion / negative-search / git-status gates, all green on first
execution (31-06-SUMMARY.md) and independently re-derived against the installed source tree during
this phase's `31-VERIFICATION.md` audit. No new test infrastructure — this gap closure touches no
file under `solsys_code/` or `src/`. `nyquist_compliant: true` unchanged.
