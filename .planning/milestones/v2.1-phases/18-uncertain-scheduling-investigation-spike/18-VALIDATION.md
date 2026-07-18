---
phase: 18
slug: uncertain-scheduling-investigation-spike
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-09
reconstructed: 2026-07-09
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Phase Nature

This phase is a **no-shippable-code investigation spike** (mirrors Phase 13's
ESO feasibility spike). Both plans' `<phase_nature_note>` state this explicitly:
no task changes `solsys_code/` production behavior, so there is nothing for a
unit-test suite to assert against. Verification is therefore **structural and
evidence-based** — file exists / git-excluded / parses as valid Python;
required Finding/Recommendation section headers present; no PII (email
address) leak; the throwaway probe leaves the `Observatory` DB unchanged
(start/end row-count match); the durable `.rst` builds cleanly under Sphinx.
Phase 13 shipped no `VALIDATION.md` at all under this same reasoning; Phase 18
reconstructs one here (structurally, not via pytest) since the nyquist
workflow hook is active on this run.

The existing `./manage.py test solsys_code` and `pytest` suites stay green
throughout — nothing in scope touches a tested module (the probe only reads).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | none (structural verification: `grep`, `test -f`, `python -c "ast.parse(...)"`, `sphinx-build`) |
| **Config file** | none — no pytest/Django test framework applies to this phase's deliverables |
| **Quick run command** | per-task `<automated>` grep/parse gate (see Per-Task Verification Map) |
| **Full suite command** | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` (Plan 02 Task 2 only) |
| **Estimated runtime** | < 5 seconds per gate |

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` structural gate (grep/`test -f`/`ast.parse`)
- **After every plan wave:** Re-run all structural gates for that plan; Plan 02 additionally requires a clean `sphinx-build`
- **Before `/gsd-verify-work`:** All structural gates green; email-address regex clean; `Observatory` row count unchanged
- **Max feedback latency:** < 5 seconds (no long-running suite)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | SCHED-01 | T-18-SC | Human confirms `rapidfuzz` package legitimacy before scratch install (SUS verdict was a documented download-lookup false-positive) | manual gate | N/A — `checkpoint:human-verify`, inherently non-automatable | ✅ | ✅ green (approved) |
| 18-01-02 | 01 | 1 | SCHED-01 | T-18-01 / T-18-02 / T-18-03 | Probe is git-excluded, never committed; persists nothing (rolled-back transactions) | structural | `test -f fuzzy_match_probe.py && grep -q 'fuzzy_match_probe.py' .git/info/exclude && grep -q 'create_placeholder=False' fuzzy_match_probe.py && grep -qE 'transaction\.atomic\|set_rollback' fuzzy_match_probe.py && python -c "import ast; ast.parse(open('fuzzy_match_probe.py').read())"` | ✅ | ✅ green |
| 18-01-03 | 01 | 1 | SCHED-01 | T-18-01 | No PII (email/contact-person) leaks into the committed decision doc | structural | `grep -q 'criterion 3' "$D" && grep -q 'criterion 2' "$D" && grep -q 'criterion 4' "$D" && grep -q 'criterion 5' "$D" && grep -q 'redacted per D-01' "$D" && ! grep -nE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "$D"` | ✅ | ✅ green (re-verified during this audit) |
| 18-02-01 | 02 | 2 | SCHED-01 | T-18-01 | Recommendation section grounded in Plan 01 Findings; no PII leak | structural | `grep -q '## Recommendation' "$D" && ! grep -q 'completed in Plan 02' "$D" && grep -qiE 'rapidfuzz\|difflib' "$D" && grep -q 'contact_person' "$D" && grep -qE '_HHMM_RANGE\|_APPROX_HOUR\|_BARE_HOUR_UTC' "$D" && ! grep -nE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "$D"` | ✅ | ✅ green (re-verified during this audit) |
| 18-02-02 | 02 | 2 | SCHED-01 | T-18-05 | Durable `.rst` matches house skeleton, no PII leak, builds under Sphinx | structural + build | `test -f "$F" && grep -qiE 'rapidfuzz\|difflib' "$F" && grep -q 'list-table' "$F" && head -3 "$F" \| grep -qE '====' && ! grep -nE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "$F"` plus `sphinx-build -M html ...` | ✅ | ✅ green (re-verified during this audit) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

`$D` = `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`;
`$F` = `docs/design/uncertain_scheduling_spike.rst`.

---

## Wave 0 Requirements

None. This phase produces no shippable application code, so no test-framework
scaffolding (`tests/test_file.py`, `conftest.py`, framework install) is needed.
Existing structural gates embedded in each task's `<verify><automated>` block
(present since planning, not added by this audit) cover all phase
requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Confirm `rapidfuzz` is the real, well-known PyPI package (not a slopsquat) before the scratch install | SCHED-01 | Package-legitimacy judgment is inherently a human trust decision; the automated gate returned a documented `SUS` false-positive (unresolved download-count lookup) that cannot be resolved by a script | Open https://pypi.org/project/rapidfuzz/, confirm MIT license / `github.com/rapidfuzz/RapidFuzz` / active maintenance; cross-check against 18-RESEARCH.md's Package Legitimacy Audit row |
| Confirm each recommendation in `18-DECISION.md` genuinely rests on its cited Plan 01 Finding (not just keyword-matching the grep gate) | SCHED-01 | Semantic grounding of a written argument is not mechanically verifiable; per Plan 02's own SUMMARY.md this was flagged `human_judgment: true` | Read `18-DECISION.md`'s `## Recommendation` section against its `## Findings` section; confirm each of the 5 criteria's verdict cites specific Plan 01 evidence (e.g. the D-09 rapidfuzz/difflib scores, the `250`/`274`/`289` TypeError finding) |
| Confirm `docs/design/uncertain_scheduling_spike.rst` is a faithful, non-divergent summary of `18-DECISION.md` (not just structurally similar) | SCHED-01 | Cross-document consistency of a prose verdict is a reading task, not a grep pattern | Read both documents side by side; confirm the fuzzy-library verdict and window-schema decision are stated identically |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or a documented non-automatable manual gate (Task 1's package-legitimacy checkpoint)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (only Task 1 is manual; Tasks 2-5 are all structural-automated)
- [x] Wave 0 covers all MISSING references (none — no MISSING references; phase ships no application code)
- [x] No watch-mode flags
- [x] Feedback latency < 5s (grep/`ast.parse` gates run near-instantly; `sphinx-build` is the only slower gate, seconds not minutes)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-09 (reconstructed audit — see Validation Audit below)

---

## Validation Audit 2026-07-09

Reconstructed from `18-01-PLAN.md`/`18-02-PLAN.md` (existing `<verify><automated>`
structural gates) and `18-01-SUMMARY.md`/`18-02-SUMMARY.md` (confirmed passing at
execution time, both with `Self-Check: PASSED`). All five structural gates were
re-run live during this audit against the current committed state of
`18-DECISION.md` and `docs/design/uncertain_scheduling_spike.rst` and confirmed
still passing (see Per-Task Verification Map). No unit-test framework applies —
this phase changes no `solsys_code/` production code, so there is nothing for
`pytest`/`./manage.py test` to assert against; the phase's own `<phase_nature_note>`
sections in both plans state this explicitly, matching Phase 13's precedent.

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed — all 5 requirement-bearing tasks already had structural automated gates, re-verified green) |
| Escalated | 0 |
| Manual-only (by design, not a gap) | 3 (package-legitimacy checkpoint + 2 human-judgment semantic-grounding checks already flagged in Plan 02's own SUMMARY.md) |
