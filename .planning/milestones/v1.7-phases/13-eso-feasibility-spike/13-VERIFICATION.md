---
phase: 13-eso-feasibility-spike
verified: 2026-07-02T09:18:37Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 13: ESO Feasibility Spike Verification Report

**Phase Goal:** Answer "can ESO/VLT observation sync work at all, and if so how?" by probing the real ESO P2 API for OB status/execution data and the headless-credential situation, then writing a decision doc recommending Bridge, Bypass, or Not Yet Feasible. No sync command is implemented this milestone; the deliverable is a written, evidence-grounded recommendation that seeds a future milestone's requirements.
**Verified:** 2026-07-02T09:18:37Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Merged from ROADMAP.md's 5 Success Criteria and both plans' `must_haves.truths` (deduplicated — plan truths restate the roadmap SCs one-for-one, plus Plan 02 adds the D-11 conditional and the durable-summary truth).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 13-DECISION.md states explicitly whether valid ESO P2 credentials were obtainable/usable, with supporting evidence (ESO-01 / Roadmap SC1) | VERIFIED | `### ESO-01` section states Paranal production credentials ARE obtainable/usable via `ESOAPI(environment='production', ...)`, evidenced by real data returned in ESO-02 (not just a bare connection flag). Also records the La Silla `production_lasilla` connection failure (`P1Error`) as a valid non-blocking finding per D-06, with a root-cause investigation (independently reproduced below). |
| 2 | At least one real p2api response for OB status/execution data captured verbatim (ESO-02 / Roadmap SC2) | VERIFIED | `getOB(4725578)` full dict (obStatus='P') and `getNightExecutions('FORS2', '2026-07-01')` (`obStatus='M'`, `grade='X'`) are pasted verbatim in `### ESO-02`, each followed by the "credential-adjacent fields redacted per D-04" note. |
| 3 | Viable (or explicitly non-viable) headless credential-sourcing path stated, accounting for ESOProfile session-bound Fernet encryption and absent `FACILITIES['ESO']` fallback (ESO-03 / Roadmap SC3) | VERIFIED | `### ESO-03` concludes "viable, with a concrete named path" (add `FACILITIES['ESO']` settings entry), explicitly contrasts with the non-viable `ESOProfile`+session path (silent `None` decryption outside a session, per PITFALLS.md). |
| 4 | Decision doc recommends exactly one of Bridge / Bypass / Not Yet Feasible, rationale tied to ESO-01..03 (ESO-04 / Roadmap SC4) | VERIFIED | `## Recommendation (ESO-04)` states a single bold verdict — **Bypass** — with three explicit paragraphs each naming ESO-01, ESO-02, and ESO-03 by name and tying the rationale to what those findings did/didn't evidence (e.g. "Nothing in the probe touched `ESOFacility.submit_observation()`... Bridge... that patch was never attempted"). |
| 5 | If feasible, doc sketches what "synced" could mean for a future command, scoped as future-milestone input (ESO-05 / Roadmap SC5) | VERIFIED | `## Future-sync sketch (ESO-05)` covers reusable landing point (`insert_or_create_calendar_event()`), synthetic key `ESO:{p2_environment}/{obId}`, banner-only vs. status-aware options, and the 12-code obStatus vocabulary — explicitly framed as "input to a future milestone's requirements... nothing here is implemented in this phase." |
| 6 | If verdict is Bridge, ESO-05 sketch includes D-11 rough effort-sizing estimate | VERIFIED (N/A correctly handled) | Verdict is Bypass, not Bridge, so D-11 does not apply. `### Not included: D-11 effort-sizing` explicitly states this rather than silently omitting it — matches the plan's conditional requirement precisely. |
| 7 | Durable summary `docs/design/eso_feasibility_spike.rst` exists alongside `telescope_runs_calendar.rst`, stating the same verdict up front | VERIFIED | File exists, follows the house skeleton (title `=`, sections `-`), opens with "This document records...", states **Bypass** as the bold "Key finding" one-liner — identical verdict to `13-DECISION.md`. Confirmed reachable from `docs/design/design.rst`'s toctree (added in follow-up commit `46e62f4`, verified below). |
| 8 | Live read-only p2api call output -> verbatim redacted evidence block, no credential-adjacent content reaches committed files (D-03/D-04 key link) | VERIFIED | Grep for password/username value patterns and `ESO_P2_PASSWORD=`/`ESO_P2_USERNAME=` across both deliverable files returns zero hits; every pasted response block is followed by the D-04 redaction note; `eso_p2_probe.py` was never staged (confirmed via `git log --all -- eso_p2_probe.py` = empty, `git check-ignore -v` = matched by `.git/info/exclude:18`) and contains zero write-style p2api calls (grep for `saveOB|deleteOB|createOB|create_observation_block|submit_observation|submit_new_observation_block` = 0 matches). |
| 9 | Committed diff for the phase contains only planning docs + the two intended deliverable files (no stray application code) | VERIFIED | `git show --stat` on all 8 phase-13 commits (`48b800d`, `7594910`, `7a52db1`, `9a3c8a3`, `7ea0974`, `63cd325`, `0e20e53`, `46e62f4`) shows only `.planning/` tracking files, `13-DECISION.md`, `docs/design/eso_feasibility_spike.rst`, and `docs/design/design.rst` (toctree entry) touched — no `solsys_code/` or other application code changed. |

**Score:** 9/9 truths verified (0 present-but-behavior-unverified)

### Independent Factual Spot-Checks (beyond trusting SUMMARY.md/13-REVIEW.md)

A prior `13-REVIEW.md` (code-review agent, deep pass) already cross-checked the `.rst`'s technical claims against installed packages. This verification independently re-ran a subset of those checks directly against the live venv rather than trusting either document:

| Claim | Command | Result |
|-------|---------|--------|
| `tom_eso==0.2.4` installed | `pip show tom_eso` | Version: 0.2.4 — confirmed |
| `ESOAPI.__init__` constructs `p1api.ApiConnection` before `p2api.ApiConnection` | `inspect.getsource(ESOAPI.__init__)` | Confirmed — `self.api1 = p1api.ApiConnection(...)` precedes `self.api2 = p2api.ApiConnection(...)` |
| `p1api.API_URL` has no `production_lasilla` key; `p2api.API_URL` does | `p1api.p1api.API_URL` / `p2api.p2api.API_URL` | Confirmed — p1api: `{'production':..., 'demo':...}` (no La Silla); p2api: includes `'production_lasilla': 'https://www.eso.org/copls/api/v1'` |
| `get_observation_status`/`get_observation_url`/`data_products` raise `NotImplementedError`; `submit_observation` is a stub | `grep -n -A3 "def get_observation_status\|def get_observation_url\|def data_products" tom_eso/eso.py` | Confirmed — all three are one-line `raise NotImplementedError` |

All four independently-reproduced technical claims match the decision doc and durable summary exactly. This corroborates the ESO-01 root-cause narrative and the ESO-04 rationale are grounded in real, verifiable facts about the installed libraries, not documentation guesswork.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` | Findings (ESO-01/02/03) + Recommendation (ESO-04) + Future-sync sketch (ESO-05), all populated with real evidence | VERIFIED | All 5 sections present, no placeholder markers (`<!-- completed in Plan 02 -->` grep returns 0 hits), content substantive (verbatim API responses, named verdict, sketch) |
| `docs/design/eso_feasibility_spike.rst` | Durable summary matching `telescope_runs_calendar.rst` skeleton, same verdict | VERIFIED | Title `=`-underlined, `Background`/`Key finding`/list-tables/`Future scope` sections present; RST parses cleanly with `docutils.core.publish_doctree`; toctree entry present in `docs/design/design.rst` |
| `eso_p2_probe.py` (intentional non-deliverable, D-09) | Git-excluded, never committed, read-only only | VERIFIED | `git check-ignore -v` matches `.git/info/exclude:18`; `git log --all -- eso_p2_probe.py` returns nothing (never staged/committed); 0 write-style p2api call names present in the file on disk |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| Live p2api response (operator-captured) | `13-DECISION.md` ESO-02 evidence blocks | Verbatim paste + D-04 redaction note | WIRED | Both `getOB()` and `getNightExecutions()` blocks present, each immediately followed by "credential-adjacent fields redacted per D-04" |
| `13-DECISION.md` ESO-01/02/03 Findings | `13-DECISION.md` ESO-04 Recommendation | Explicit named cross-reference in rationale | WIRED | Recommendation section has one paragraph per ESO-01/02/03, each naming the finding and explaining what it does/doesn't support for Bypass vs. Bridge |
| `13-DECISION.md` ESO-04 verdict | `docs/design/eso_feasibility_spike.rst` Key finding | Same verdict text ("Bypass") | WIRED | Both docs state Bypass as the bold headline verdict — no divergence |
| `docs/design/eso_feasibility_spike.rst` | `docs/design/design.rst` toctree | Sphinx `.. toctree::` entry | WIRED | Confirmed present (follow-up fix commit `46e62f4`, verified by re-reading current `design.rst`); orphan-doc warning from the code-review's `13-REVIEW.md` (WR-01) is resolved |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ESO-01 | 13-01 | Credential obtainability/usability confirmed | SATISFIED | `### ESO-01` section, evidenced by ESO-02's real data |
| ESO-02 | 13-01 | Real OB status/execution data shape captured verbatim | SATISFIED | `### ESO-02` verbatim `getOB()`/`getNightExecutions()` blocks |
| ESO-03 | 13-01 | Headless credential-sourcing path determined | SATISFIED | `### ESO-03` concludes viable with named path |
| ESO-04 | 13-02 | Single Bridge/Bypass/Not-Yet-Feasible recommendation with rationale | SATISFIED | `## Recommendation (ESO-04)` — Bypass, rationale ties to ESO-01/02/03 |
| ESO-05 | 13-02 | Future-sync sketch scoped as future-milestone input | SATISFIED | `## Future-sync sketch (ESO-05)` |

No orphaned requirements: `.planning/REQUIREMENTS.md` maps only ESO-01..05 to Phase 13; ESO-10/ESO-11 are explicitly deferred (v2, contingent on this phase's decision) and are not claimed by either plan — consistent with the phase's investigation-only scope.

**Note (non-blocking, process-ordering, not a phase-goal gap):** `.planning/REQUIREMENTS.md`'s checkboxes for ESO-01..05 are still unchecked (`- [ ]`) and its phase-mapping table (lines 44-48) still shows "Pending" rather than "Complete". `STATE.md` confirms this is expected ordering — the phase is marked `status: executed`, explicitly "awaiting phase-goal verification" before the requirements/roadmap bookkeeping is finalized. This is standard GSD phase-completion sequencing (requirements checkboxes update after verification passes), not evidence the goal wasn't achieved.

### Anti-Patterns Found

Scanned `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` and `docs/design/eso_feasibility_spike.rst` (the two committed deliverables) for TBD/FIXME/XXX/TODO/HACK/placeholder/stub markers.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `13-DECISION.md:252` | "...as a v1/simplified/**placeholder** version of shippable work." | `placeholder` (string match) | None (false positive) | This is the sketch explicitly *disclaiming* placeholder status ("nothing here... should be read as a v1/simplified/placeholder version") — not a stub marker. No action needed. |

No TBD/FIXME/XXX/HACK/TODO markers found. No unresolved debt markers. No empty-implementation patterns (N/A — no code artifacts this phase).

### Human Verification Required

None. This phase's deliverables are two prose/RST documents; the content-fidelity checks that would normally route to "human judgment" in the plan's `<human-check>` blocks (redaction completeness, verdict-to-evidence traceability, RST skeleton fidelity) were resolved directly in this verification pass via grep/read/independent library introspection (see Independent Factual Spot-Checks above), and a prior deep code-review (`13-REVIEW.md`) additionally cross-checked every factual claim against the installed packages and sibling docs, finding 0 critical issues (2 warnings, both since fixed in commit `46e62f4`, and 1 info item left open as a style nit, non-blocking). No visual/real-time/external-service behavior requires a human tester — the phase produces no runnable code.

### Gaps Summary

None. All 5 ROADMAP success criteria and all `must_haves` from both PLAN.md frontmatters are met by real, independently-verifiable content in `13-DECISION.md` and `docs/design/eso_feasibility_spike.rst`. The D-11 Bridge-effort-sizing conditional was correctly recognized as not-applicable (verdict is Bypass) and explicitly stated as such rather than silently omitted. The one prior code-review finding that mattered for durability (orphaned toctree entry) was already fixed in a follow-up commit before this verification ran. No credential leakage found in either committed file. No stray application-code changes crossed into this investigation-only phase.

---

_Verified: 2026-07-02T09:18:37Z_
_Verifier: Claude (gsd-verifier)_
