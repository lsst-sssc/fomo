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

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01-01 | 01 | 1 | SCHEMA-01 | — | N/A — investigation only | investigation script | `sqlite3 src/fomo_db.sqlite3 "SELECT COUNT(*) FROM solsys_code_campaignrun WHERE campaign_id IS NULL;"` | ✅ (real DB, already queried in research: 0/49) | ⬜ pending |
| 31-01-02 | 01 | 1 | SCHEMA-02 | — | Credentials/no new input surface — N/A | disposable constraint-probe script | `python manage.py shell < tmp/31_constraint_probe.py` | ❌ W0 — executor writes this throwaway script | ⬜ pending |
| 31-01-03 | 01 | 1 | SCHEMA-03 | — | N/A | manual inspection | N/A — depends on operator-supplied classical schedule file (Open Question 1, RESEARCH.md) | ❌ — no real file exists in-repo | ⬜ pending |
| 31-02-01 | 02 | 1 | SCHED-07 | T-31-01 | Credential value never appears in a log line/notification (SCHED-10) | shell-level verification | `flock --version`, `crontab -l`, `curl -sS -o /dev/null -w '%{http_code}' https://hc-ping.com/` | ✅ (already run against interim host in research) | ⬜ pending |
| 31-02-02 | 02 | 1 | SCHED-07 (container scope) | T-31-01 | Same as above, verified inside the real container image | shell-level verification | Same probes, re-run inside the FOMO container image | ❌ W0 — container-level check, not yet run | ⬜ pending |
| 31-03-01 | 03 | 2 | Regression | — | N/A | full suite | `python manage.py test` (excluding `test_views.TestEphemeris`) | ✅ — suite already exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tmp/31_constraint_probe.py` — disposable, git-excluded probe script mirroring `tmp/26_integrity_check.py`'s pattern: write against a **disposable copy** of `src/fomo_db.sqlite3`, never the live file, to check candidate `source_identifier` schemes against both existing `UniqueConstraint`s without an unexpected `IntegrityError`.
- [ ] Container-level shell check for `flock --version` / outbound HTTPS reachability to `hc-ping.com` (Pitfall 5 in RESEARCH.md) — not a test file, a one-time verification step whose output is quoted verbatim in `31-DECISION.md`.
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
