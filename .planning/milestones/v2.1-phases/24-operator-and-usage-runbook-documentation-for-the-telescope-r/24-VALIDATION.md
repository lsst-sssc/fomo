---
phase: 24
slug: operator-and-usage-runbook-documentation-for-the-telescope-r
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-18
reconstructed: 2026-07-18
---

# Phase 24 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Phase Nature

This phase is **docs-only** (mirrors Phase 18's investigation-spike precedent):
Task 1 modifies `docs/installation.rst`, Task 2 creates
`docs/runbooks/telescope_runs_calendar.rst` and wires it into `docs/index.rst`,
and Task 3 appends a Troubleshooting section — zero lines of `solsys_code/`
production code change (confirmed in 24-01-SUMMARY.md and re-confirmed at
`24-VERIFICATION.md` Truth 10 / Anti-Patterns: `git diff --name-only` touches
only the three doc files). There is nothing for `pytest` / `./manage.py test`
to assert against, so no unit-test framework applies. Verification is
therefore **structural and evidence-based** — label/heading exists via
`grep`, all five command names + both staff-action identifiers present, a
`.. list-table::` cheat-sheet exists, the toctree line is wired, the verbatim
timezone error string and its IANA fix-it are present, and no non-`@example.`
email address appears anywhere in the runbook (PII gate for security threat
T-24-01). Every one of these gates was already embedded in the plan's
`<verify><automated>` blocks (present since planning, not added by this
audit) and was live-executed successfully during plan execution
(24-01-SUMMARY.md coverage table, all `status: pass`) and independently
re-run by the verifier (24-VERIFICATION.md, 11/11 truths verified). This
audit re-ran all of them a third time against the current committed state
and confirmed they still pass (see Per-Task Verification Map).

The existing `./manage.py test solsys_code` and `pytest` suites stay green
throughout — nothing in scope touches a tested module.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | none (structural verification: `grep`, `test -f`, `sphinx-build`) |
| **Config file** | none — no pytest/Django test framework applies to this phase's deliverables |
| **Quick run command** | per-task `<automated>` grep gate (see Per-Task Verification Map) |
| **Full suite command** | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` |
| **Estimated runtime** | < 10 seconds per gate |

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` structural gate (grep/`test -f`)
- **After every plan wave:** Re-run all structural gates for the plan plus a clean `sphinx-build`
- **Before `/gsd-verify-work`:** All structural gates green; PII email regex clean; no orphan-toctree warning for the runbook page
- **Max feedback latency:** < 10 seconds (no long-running suite; `sphinx-build` is the slowest gate, a few seconds)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 24-01-01 | 01 | 1 | D-08, D-09 | T-24-02 | Onboarding subsection is generic Django/manage.py orientation only, no PII | structural + build | `grep -n '^.. _running-management-commands:' docs/installation.rst && grep -n '^Running FOMO Management Commands' docs/installation.rst && sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` | ✅ | ✅ green (re-verified during this audit) |
| 24-01-02 | 01 | 1 | D-01, D-02, D-03, D-04, D-05, D-06, D-07, D-10 | — | N/A — no dependency added (T-24-SC accepted, zero packages installed) | structural + build | `test -f docs/runbooks/telescope_runs_calendar.rst && grep -c 'How do I' docs/runbooks/telescope_runs_calendar.rst && for c in load_telescope_runs sync_lco_observation_calendar sync_gemini_observation_calendar import_campaign_csv backfill_range_calendar_events mark_cancelled mark_weather_failure; do grep -q "$c" docs/runbooks/telescope_runs_calendar.rst \|\| echo MISSING:$c; done && grep -q 'list-table' docs/runbooks/telescope_runs_calendar.rst && grep -q 'Runbooks <runbooks/telescope_runs_calendar>' docs/index.rst && sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` (no orphan-toctree warning for the page) | ✅ | ✅ green (re-verified during this audit) |
| 24-01-03 | 01 | 1 | D-11, D-12, D-13 | T-24-01 | Every troubleshooting example uses synthetic placeholder PII (Jane Doe / `@example.` only); the one real string quoted verbatim (`Observatory 'FTN' (obscode=F65) has no timezone set`) is an MPC obscode + telescope short-name, not personal data | structural + PII gate + build | `grep -n '^Troubleshooting' docs/runbooks/telescope_runs_calendar.rst && grep -F "Observatory 'FTN' (obscode=F65) has no timezone set" docs/runbooks/telescope_runs_calendar.rst && grep -q 'America/Santiago' docs/runbooks/telescope_runs_calendar.rst && grep -q 'site_needs_review' docs/runbooks/telescope_runs_calendar.rst && EMAILS=$(grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+' docs/runbooks/telescope_runs_calendar.rst \| grep -v '@example\.' \|\| true); test -z "$EMAILS" && sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` | ✅ | ✅ green (re-verified during this audit) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

None. This phase produces no shippable application code, so no test-framework
scaffolding (`tests/test_file.py`, `conftest.py`, framework install) is
needed. Existing structural gates embedded in each task's
`<verify><automated>` block (present since planning, not added by this audit)
cover all 13 phase requirements (D-01..D-13) and the one security threat
(T-24-01).

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* The semantic-accuracy
checks that are genuinely not grep-mechanizable — e.g. "is the runbook truly
organized by task-oriented headings and not a flat per-command dump" (D-06),
"do the documented failure modes match real command source behavior, not
speculation" (D-11) — were independently read-and-cross-checked once against
the actual command source files by the verifier at initial verification time
(`24-VERIFICATION.md` Truth 3, Truth 8, and the "Behavioral Spot-Checks"
table: `sync_gemini_observation_calendar.add_arguments` confirmed a no-op
`pass`, `import_campaign_csv.py`'s WR-07 target-reset comment confirmed, and
`telescope_runs.py:275`'s format string confirmed matching the quoted
verbatim error). This was a one-time editorial judgment at verification, not
a recurring gate the Nyquist sampling loop needs to re-run — the file-content
and grep gates above already re-verify structurally on every task commit.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify (all 3 tasks — no manual-only gates needed)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (all 3 tasks are structural-automated)
- [x] Wave 0 covers all MISSING references (none — no MISSING references; phase ships no application code)
- [x] No watch-mode flags
- [x] Feedback latency < 10s (grep/`test -f` gates run near-instantly; `sphinx-build` is the only slower gate, seconds not minutes)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-18 (reconstructed audit — see Validation Audit below)

---

## Validation Audit 2026-07-18

Reconstructed from `24-01-PLAN.md` (existing `<verify><automated>` structural
gates), `24-01-SUMMARY.md` (coverage table, all `status: pass`), and
`24-VERIFICATION.md` (11/11 must-have truths verified, 0 gaps). The
pre-existing `24-VALIDATION.md` on disk was still the unfilled seed template
(frontmatter `status: draft`, placeholder `{...}` table cells) — this audit
replaces it with the reconstructed contract below, per the same
docs-only/no-shippable-code precedent Phase 18 established.

All three tasks' automated gates were re-run live during this audit against
the current committed state of `docs/installation.rst`,
`docs/runbooks/telescope_runs_calendar.rst`, and `docs/index.rst`, and
confirmed still green: the `.. _running-management-commands:` label and
heading are present; all five command names and both
`mark_cancelled`/`mark_weather_failure` action identifiers appear in the
runbook; the `.. list-table::` cheat-sheet and toctree line are present; the
verbatim timezone error string, `America/Santiago`, and `site_needs_review`
are present; no non-`@example.` email address exists in the runbook; and a
fresh `sphinx-build -M html ./docs ./_readthedocs -T -E -d
./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` exits 0 with
9 pre-existing/unrelated warnings and zero orphan-toctree warning for
`runbooks/telescope_runs_calendar`.

No unit-test framework applies — this phase changes no `solsys_code/`
production code, so there is nothing for `pytest`/`./manage.py test` to
assert against.

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed — all 13 requirements (D-01..D-13) plus threat T-24-01 already had structural automated gates embedded in the plan, re-verified green) |
| Escalated | 0 |
| Manual-only (by design, not a gap) | 0 (one-time editorial cross-checks already performed at initial verification, not a recurring sampling-loop gate) |
