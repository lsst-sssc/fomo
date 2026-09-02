---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 03
subsystem: investigation
tags: [classical-adapter, telescope-runs, run-identity, schema-spike, decision-doc]

requires:
  - phase: 31-02
    provides: "SCHEMA-01/02 recommendation: nullable campaign FK (Option A), source_identifier field/constraint, classical adapter's default synthesized key pending this plan's findings"
provides:
  - "31-DECISION.md SCHEMA-03 evidence subsection: real classical schedule-file sample inspection (1 file, 3 lines) against parse_run_line, D-07's remembered status vocabulary reconciled against CampaignRun.RunStatus"
  - "31-DECISION.md SCHEMA-03 recommendation: the tolerance match is NOT sufficient on its own; the failing case (two proposals sharing telescope/instrument/night) named; consequence for source_identifier stated; Phase 32's two follow-on items named"
affects: [31-04, 31-05, 32]

actuals:
  tokens: 3486
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Dual-tag evidence vocabulary within a single section (Confirmed against real rows for observed parse facts, Constructed-input code-path check for the sufficiency verdict itself), used where a real sample grounds the facts but the failing case was not literally observed"

key-files:
  created: []
  modified:
    - .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md

key-decisions:
  - "Task 1 (checkpoint:human-action, blocking-human): resolved by the operator supplying a real classical schedule file (tmp/31-classical-samples/didymos_2026_july_classical_runs.txt, git-excluded) before this continuation started; no re-ask, no re-checkpoint."
  - "SCHEMA-03 verdict: the existing 5-minute telescope/instrument/start_time tolerance match is NOT sufficient as a write-time identity surface on its own. Failing case: two genuinely distinct proposals allocated the same telescope, instrument and night with no partial-night window distinguishing them collide and silently overwrite. Confidence: Constructed-input code-path check for the verdict itself (no real collision observed in the 3-line sample); Confirmed against real rows for the parse facts it rests on."
  - "SCHEMA-02 consequence: the classical source_identifier synthesized key (CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}) uses the same three fields as the tolerance match, so it inherits the identical blind spot -- compatible with plan 31-02's populate-not-blank decision (no contradiction), but an incomplete mitigation. A proposal code is not currently a reliable fallback (present in only 1 of 3 real lines, and unparseable in today's grammar where observed)."

requirements-completed: [SCHEMA-03]

coverage:
  - id: T2
    description: "Task 2: inspect the real classical schedule sample against parse_run_line, record per-status-word findings, reconcile D-07 against KNOWN_STATUSES/CampaignRun.RunStatus"
    requirement: "SCHEMA-03"
    verification:
      - kind: other
        ref: "grep -q '^#### SCHEMA-03 evidence - classical schedule-file sample inspection' 31-DECISION.md && grep -qE '(0 real classical schedule files were available|CLASSICAL_SAMPLE_FILES=1)' 31-DECISION.md"
        status: pass
      - kind: other
        ref: "sed -n '/^#### SCHEMA-03 evidence/,/^#### /p' 31-DECISION.md | grep -q KNOWN_STATUSES && ... | grep -qE 'planned|PLANNED' && ... | grep -qE 'Tag: \\*\\*(Confirmed against real rows|Constructed-input code-path check)\\*\\*'"
        status: pass
      - kind: other
        ref: "! grep -REiq 'api[_-]?key[[:space:]]*[:=]|[0-9a-f]{32,}|[[:alnum:]._%+-]+@[[:alnum:].-]+\\.[[:alpha:]]{2,}' 31-DECISION.md"
        status: pass
    human_judgment: false
  - id: T3
    description: "Task 3: write the SCHEMA-03 sufficiency verdict, address all four candidate failing cases, state the SCHEMA-02 consequence, check nullable-FK compatibility"
    requirement: "SCHEMA-03"
    verification:
      - kind: other
        ref: "grep -q '^### SCHEMA-03 - classical write-time identity surface' 31-DECISION.md; section states sufficient/not-sufficient, references the 5-minute tolerance/_START_TIME_MATCH_TOLERANCE, and names Phase 32"
        status: pass
      - kind: other
        ref: "sed -n '/^### SCHEMA-03/,/^## /p' 31-DECISION.md | grep -cE 'instrument|night convention|status changed|proposal' -ge 4 (actual: 28)"
        status: pass
    human_judgment: false

duration: ~25min
completed: 2026-09-02
status: complete
---

# Phase 31 Plan 03: Classical Adapter Write-Time Identity (SCHEMA-03) Summary

**The 5-minute telescope/instrument/start_time tolerance match is insufficient on its own — two proposals sharing a telescope/instrument/night collide and silently overwrite — and today's parser can't even extract a proposal code as a fallback (1/3 real lines carried one, and it was unparseable where seen), so the classical `source_identifier` synthesized key inherits the same gap as a documented, low-frequency risk pending a Phase 32 parser change.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2 (Task 1's checkpoint was already resolved by the operator before this continuation started — the real sample file was in place at `tmp/31-classical-samples/didymos_2026_july_classical_runs.txt`)
- **Files modified:** 1 (`31-DECISION.md`, built up across 2 commits)

## Accomplishments

- Ran all 3 real classical run lines in the operator-supplied Didymos-campaign sample through `solsys_code.telescope_runs.parse_run_line()` in a read-only Django shell (pure function, no DB touched). Found: 1 of 3 lines rejected by `ValueError` — not a status-vocabulary gap, but a positional collision, since the line places a proposal-code-shaped token before the telescope name, shifting every downstream token by one and leaving an unconsumed leftover the parser reports as "Unrecognized status 'EFOSC2'". Only status word observed anywhere in the sample: `allocation` (the other two lines default silently, per `ParsedRun`'s documented default).
- Reconciled D-07's remembered "planned"/"observed" status words against the parser's real `KNOWN_STATUSES` set (`{allocation, proposed, confirmed, cancelled, not confirmed}` — neither word is a member) versus `CampaignRun.RunStatus` (`solsys_code/models.py:96-106`, which declares `PLANNED` and `OBSERVED` verbatim). Concluded D-07's recollection belongs to the `CampaignRun.RunStatus` lifecycle vocabulary, not the classical-file vocabulary — a real, evidence-backed answer rather than an assumption.
- Wrote the SCHEMA-03 sufficiency verdict: **not sufficient**. Named the concrete failing case — two distinct proposals allocated the same telescope, instrument and night with no partial-night window differentiation — and ruled out the other three candidate cases (instrument swap: already discriminated by the lookup; night-convention shift: a static per-site config, not a data-variability risk; status-only re-issue: `status` is deliberately excluded from the lookup, so this already works correctly).
- Stated the SCHEMA-02 consequence explicitly: the classical `source_identifier` synthesized key (`CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}`) uses the identical three fields as the tolerance match, so it inherits the identical blind spot — compatible with plan 31-02's decision to populate (not blank) the field, but an incomplete mitigation, not a fix.
- Handed Phase 32 two explicit, separable follow-on items: extend `ParsedRun`/`parse_run_line`'s grammar to recognize a leading proposal-code token (needed regardless of whether it becomes part of the identity key), and decide whether to fold a reliably-extractable proposal code into both the `CalendarEvent` lookup and the `source_identifier` formula once that grammar exists.

## Task Commits

1. **Task 1: Obtain a real classical schedule file** — resolved by the operator before this continuation started (file supplied at `tmp/31-classical-samples/didymos_2026_july_classical_runs.txt`, git-excluded); no commit produced by that step itself.
2. **Task 2: Inspect the classical schedule sample and record per-status-word findings** — `a5ff9f2` (docs)
3. **Task 3: Write the SCHEMA-03 recommendation** — `836abd8` (docs)

**Plan metadata:** committed alongside this SUMMARY (see below).

## Files Created/Modified

- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` — appended `#### SCHEMA-03 evidence - classical schedule-file sample inspection` (under `## Findings`, after the SCHEMA-02 evidence subsection) and `### SCHEMA-03 - classical write-time identity surface` (under `## Recommendation`, after the SCHEMA-02 recommendation subsection)
- `tmp/31-classical-samples/didymos_2026_july_classical_runs.txt` — the operator-supplied real sample; git-excluded, not committed, confirmed via `git ls-files tmp/` printing nothing

## Decisions Made

See `key-decisions` in the frontmatter above. Summarized: the tolerance match is insufficient on its own for the two-proposals-same-night case; the classical `source_identifier` key inherits that same gap; a proposal code is not currently a reliable substitute given today's real sample and grammar; Phase 32 inherits a parser-grammar extension plus a fold-in decision, not a closed question.

## Deviations from Plan

None — plan executed exactly as written. Both automated `<verify>` blocks and all `<acceptance_criteria>` for Task 2 and Task 3 passed on the first attempt; no fix-up needed. The write-up implied a real parser/grammar change would be valuable (to accept a leading proposal-code token), but per the plan's own scoping instruction this was recorded as a Phase 32 obligation, not implemented here — no code in `solsys_code`/`src` was touched (confirmed by `git status --porcelain -- solsys_code src` printing nothing throughout).

## Issues Encountered

None. The operator-supplied sample's trailing Gemini-informational block (explicitly self-labeled "not classical-schedule format -- informational only") was excluded from all analysis per the plan's content-handling note; no PI name, proposal title, or contact detail from either the classical lines or the ignored block was quoted anywhere, verified by the whole-document PII/credential regex check (`grep -REiq 'api[_-]?key...|[0-9a-f]{32,}|...@...'`) passing with zero matches.

## Next Phase Readiness

**For plan 31-04 (SCHED-07):** no direct dependency on this plan's content beyond the phase's shared investigation-only framing; 31-04 is an independent track (scheduling mechanism against the real host).

**For plan 31-05 (durable summary + `docs/design/` page):** this plan's SCHEMA-03 verdict is compatible with 31-02's SCHEMA-01 (nullable `campaign` FK) decision — orthogonal, no interaction. It is also compatible with 31-02's SCHEMA-02 decision to populate (not blank) `source_identifier` for the classical path — no contradiction in the nullability/blank-vs-synthesized choice — but the verdict does surface an **incomplete mitigation**, not a settled answer: the synthesized key does not close the two-proposals-same-night gap, since it uses the same three fields as the tolerance match it mirrors. Plan 31-05 and Phase 32 should carry forward the two explicit follow-on items named in the recommendation (parser-grammar extension for a leading proposal-code token; the fold-in-or-accept-risk decision once that grammar exists) as open Phase 32 work, not as something this investigation resolved.

**Confidence caveat for Phase 32:** the sufficiency verdict itself is tagged **Constructed-input code-path check** (reasoned from the loader's own lookup/tolerance code, not from an observed two-proposal collision) — the underlying parse facts (1 real file, 3 real lines, 1 rejected, 1 proposal-code-shaped token in an unparseable position) are tagged **Confirmed against real rows**. Phase 32 should re-check the verdict against the next real classical schedule file it sees, specifically watching for two distinct full-night entries sharing one telescope and instrument.

---
*Phase: 31-foundation-spikes-run-identity-unattended-invocation*
*Completed: 2026-09-02*
