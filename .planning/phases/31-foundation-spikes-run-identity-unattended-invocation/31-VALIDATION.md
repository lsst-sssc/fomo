---
phase: "31"
slug: "foundation-spikes-run-identity-unattended-invocation"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-01"
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

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01-01 | 01 | 1 | SCHEMA-01 | T-31-01, T-31-02 | Read-only probe; no contact-field values selected | investigation script (tracer) | `python manage.py shell < tmp/31_dbsnapshot_probe.py` then assert `FINGERPRINT_UNCHANGED=PASS` and the count keys | ❌ W0 — executor writes this throwaway script | ⬜ pending |
| 31-01-02 | 01 | 1 | SCHEMA-01 | T-31-03 | N/A — read-only grep inventory | grep inventory, self-deriving | loop over `grep -rl '\.campaign\b'` non-test sources asserting each appears in `31-DECISION.md` | ✅ — real source tree | ⬜ pending |
| 31-01-03 | 01 | 1 | SCHEMA-02 | T-31-02 | Hard DB-path guard before any write; live DB fingerprint re-compared | disposable constraint-probe script | `python manage.py shell < tmp/31_constraint_probe.py` asserting `GUARD_DISPOSABLE_COPY=OK`, ≥5 `PASS:`, 0 `FAIL:` | ❌ W0 — executor writes this throwaway script | ⬜ pending |
| 31-02-01 | 02 | 2 | SCHEMA-01 | — | N/A | blocking-human decision checkpoint (one-way door, D-05) | none — checkpoint | N/A | ⬜ pending |
| 31-02-02 | 02 | 2 | SCHEMA-01 | T-31-05 | N/A | document assertion | heading + candidate-shape mention count inside the SCHEMA-01 recommendation section | ✅ | ⬜ pending |
| 31-02-03 | 02 | 2 | SCHEMA-02 | T-31-04, T-31-06 | Contact field named, never valued | document assertion | all three command names and all three `Source` values present inside the SCHEMA-02 section | ✅ | ⬜ pending |
| 31-03-01 | 03 | 3 | SCHEMA-03 | T-31-07 | Operator files stay in git-excluded `tmp/` | blocking-human action checkpoint | none — checkpoint | N/A — depends on operator-supplied classical schedule file (Open Question 1, RESEARCH.md) | ⬜ pending |
| 31-03-02 | 03 | 3 | SCHEMA-03 | T-31-07, T-31-08 | No PI name, title, email or raw line quoted into the committed doc | sample inspection + document assertion | sample count reconciliation, plus a negative search for credential/email tokens in `31-DECISION.md` | ❌ — no real file exists in-repo | ⬜ pending |
| 31-03-03 | 03 | 3 | SCHEMA-03 | T-31-09 | N/A | document assertion | sufficiency verdict, tolerance reference and four failing-case mentions inside the SCHEMA-03 section | ✅ | ⬜ pending |
| 31-04-01 | 04 | 4 | SCHED-07 | T-31-10, T-31-13 | Crontab redacted at capture time, before anything is written | shell-level verification | `flock --version` (util-linux banner), redacted `crontab -l` (a `manage.py` entry), `curl -w '%{http_code}'` to the heartbeat host (1xx-5xx) | ✅ — real host | ⬜ pending |
| 31-04-02 | 04 | 4 | SCHED-07 (container + AWS scope) | T-31-10 | Same, verified inside a real image where one exists | shell-level verification | tracked build-file search, local image inventory, then the same two probes inside whichever image branch applies | ❌ W0 — container-level check, not yet run | ⬜ pending |
| 31-04-03 | 04 | 4 | SCHED-07 | T-31-11, T-31-12, T-31-14, T-31-15 | No credential value in any committed artifact (SCHED-10) | document assertion + negative credential search | invocation-shape and two-layer-visibility assertions, plus `! grep -REiq` for credential-shaped tokens in `31-DECISION.md` | ✅ | ⬜ pending |
| 31-05-01 | 05 | 5 | SCHEMA-01..03, SCHED-07 | T-31-16 | No credential, email or operator line on the published page | docs build precondition + document assertion | page structure, two decision tables, toctree entry, and a one-line-added diff assertion on `docs/design/design.rst` | ✅ | ⬜ pending |
| 31-05-02 | 05 | 5 | Regression | T-31-17, T-31-18, T-31-19 | Disposable DB copy removed; nothing under `tmp/` tracked | Sphinx build + targeted suite + cleanliness gates | `sphinx-build -M html ./docs ./_readthedocs …` prints `build succeeded`; `python manage.py test` over six named modules ends `OK`; empty `git status -- solsys_code src`; empty `git ls-files tmp/` | ✅ — suite already exists | ⬜ pending |

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

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Classical schedule line carries a proposal code for specific run states ("planned"/"observed") | SCHEMA-03 | No real classical schedule file exists anywhere in this repository to automate against; needs operator-supplied sample or explicit "unavailable" finding | Obtain a real classical schedule file sample from the operator (or record the gap explicitly per Open Question 1); inspect for a proposal-code field per `KNOWN_STATUSES` in `solsys_code/telescope_runs.py` |
| Container-image-level `flock`/network verification | SCHED-07 (D-03) | Requires building/accessing the real FOMO container image, which this session's shell is not confirmed to be | Build or exec into the container image; run `flock --version` and a `curl` to the heartbeat endpoint; quote output verbatim in `31-DECISION.md` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < N/A (investigation-only phase)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
