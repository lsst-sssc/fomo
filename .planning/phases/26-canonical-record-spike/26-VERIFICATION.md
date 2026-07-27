---
phase: 26-canonical-record-spike
verified: 2026-07-27T17:53:09Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "SPIKE-03: the doc states one canonical reconciler event-key scheme, stable across all four pipeline stages, and answers explicitly whether a class-wide run fans out one event per candidate site or produces a single class-wide event"
    status: partial
    reason: >
      The site-fanout-vs-single-class-wide-event dichotomy IS answered explicitly
      (Recommendation, Criterion 3: "a class-wide run produces a single class-wide
      event per day ... not one event per candidate site", grounded in the real
      SITE_TELESCOPE_MAP 5-site/2-site counts, independently confirmed against
      solsys_code/calendar_utils.py:36-53). But the "one canonical ... scheme,
      stable across all four pipeline stages" half is no longer true without
      qualification. A post-execution domain correction from the project owner
      (recorded in 26-DECISION.md "### Domain correction — queue windows are not
      sets of owned nights" and mirrored in docs/design/canonical_record_spike.rst)
      rescopes the locked RUN:{run_pk}:{date} key to classically-scheduled runs
      only. CampaignRun pk=1 — the flagship real-data anchor the phase goal itself
      names ("against the real dev-DB rows (CampaignRun pk=1, its 11 LCO queue
      events...)") — is an LCO *queue* run, and for queue runs the decision doc
      and durable page both state explicitly that "whether — and how — a
      queue-scheduled run should appear on the calendar at all... is a separate,
      still-open question for Phase 29." 26-03-SUMMARY.md's own "Next Phase
      Readiness" section confirms this directly: "Phase 29 ... has everything
      locked for classically scheduled runs. For queue-scheduled runs ... two
      questions are explicitly open." So for the run type the phase goal's own
      anchor example belongs to, Phase 29 does not yet "execute a decision instead
      of making one" — it inherits a second open question the ticked SPIKE-03
      requirement implies was closed. This is self-disclosed transparently in the
      committed docs (not a hidden stub), and D-05's related 80x5=400 class-wide
      fan-out figure is itself flagged in the same correction as "very likely the
      same category error one level up" — meaning even the class-wide answer that
      IS locked may need revisiting once Phase 29 resolves the queue-run question.
      ROADMAP.md's own Phase 29 success criterion 2 ("A site-resolved run shows
      one event per night... a class-wide run shows one 00:00-23:59 event per
      day...") has not been updated to reflect this correction and still reads as
      though the pre-correction per-night/per-day semantics apply universally —
      a staleness the spike explicitly flagged as a follow-up todo but did not fix
      (investigation-only scope boundary, consistent with its D-16/PROJECT.md
      precedent of flagging-not-fixing upstream docs).
    artifacts:
      - path: ".planning/phases/26-canonical-record-spike/26-DECISION.md"
        issue: "Criterion 3/SPIKE-03 section and 'Domain correction' section both state the key scheme is locked for classical runs only; queue-run projection mechanism is an open Phase 29 question"
      - path: "docs/design/canonical_record_spike.rst"
        issue: "Same rescoping mirrored in the durable page's 'Event key' row and 'Domain correction' section"
    missing:
      - "Either (a) a human decision to formally accept the SPIKE-03 rescoping (queue-run projection deferred to Phase 29, alongside the already-accepted D-11 write-strategy deferral) via a VERIFICATION.md override, updating ROADMAP.md/REQUIREMENTS.md language to say 'stable across all four pipeline stages for classically-scheduled runs' rather than unqualified; or (b) a short closure plan that produces the same kind of measured evidence for queue-run projection that D-11 produced for the write-strategy question, before Phase 29 is planned."
human_verification: []
---

# Phase 26: Canonical-Record Spike Verification Report

**Phase Goal:** Settle the identity, key-scheme, migration and attribution questions milestone questioning deliberately left open — against the real dev-DB rows (`CampaignRun` pk=1, its 11 LCO queue events, the 19 unprojected 3I/ATLAS runs), not against hypotheticals — so that every later phase executes a decision instead of making one.
**Verified:** 2026-07-27T17:53:09Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Phase Nature Note

This is an investigation-only spike (confirmed against `26-CONTEXT.md`'s Phase
Boundary and `26-DECISION.md`'s header). No `solsys_code/` code, migration, or
UI ships from this phase by design — verified true (see Teardown section below).
That is correct, not a gap. The deliverables under test are `26-DECISION.md` and
`docs/design/canonical_record_spike.rst`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SPIKE-01: decision doc fixes the `source` vocabulary and demonstrates, by executable check, `CampaignRun` pk=1 + its 11 LCO companion rows coexisting with no `IntegrityError` and no constraint change | VERIFIED | 26-DECISION.md Block A-D output (5 PASS / 0 FAIL) quoted verbatim; independently cross-checked against `solsys_code/models.py:120-160`'s actual constraint field tuples and names — exact match |
| 2 | SPIKE-02: doc maps each adapter's existing calendar-event identity key onto a run (classical, LCO, Gemini, campaign-projection) with file:line construction sites and confidence tags | VERIFIED | All four construction sites independently re-read from source (`load_telescope_runs.py:207-216`, `sync_lco_observation_calendar.py:361`, `sync_gemini_observation_calendar.py:150`, `campaign_views.py:447,485`) — lookup dicts match the doc's claims exactly |
| 3 | SPIKE-03: doc states one canonical reconciler event-key scheme, stable across all four pipeline stages, and explicitly answers the class-wide fan-out question | **PARTIAL / FAILED** | Fan-out dichotomy answered explicitly (single class-wide event, not per-site) — confirmed against real `SITE_TELESCOPE_MAP` (`calendar_utils.py:36-53`, 5 sites for `1m0`, 2 for `2m0`). But "stable across all four pipeline stages" does not hold post-domain-correction for queue-scheduled runs (incl. `CampaignRun` pk=1, the phase goal's own anchor) — see Gaps below |
| 4 | SPIKE-04: doc states migration + rename checklist naming every integration point, `related_name`-unchanged decision recorded | VERIFIED | Migration applied cleanly (31/20/11 row counts byte-identical before/after, quoted verbatim); 6-point checklist (not the original 4) measured via `./manage.py check` `ImportError` + narrow 7-module test run (177→265 tests); matches current `solsys_code/models.py` (still pre-rename, as expected — rename is Phase 27's job) |
| 5 | Decisions are durable and readable outside `.planning/` via a `docs/design/` page | VERIFIED | `docs/design/canonical_record_spike.rst` exists, is wired into `docs/design/design.rst`'s toctree (line 47), and `sphinx-build -M html ./docs ...` (the exact pre-commit invocation) succeeds with 0 warnings attributable to this file |

**Score:** 4/5 truths fully verified; 1 truth (SPIKE-03) partially verified — see Gaps.

### D-11 (write-strategy deferral) — judged separately, not counted as a gap

SPIKE-03's *write-strategy* half (adopt vs. gap-fill) was deliberately left open
for Phase 29 at the project owner's explicit direction during execution (recorded
in `26-03-SUMMARY.md`'s "Recorded plan-deviation" section). This is judged
**acceptable**: both options were fully measured against the real 15-night pk=1
window (identical calendar either way; a concrete two-writer-churn code-level
finding distinguishes them), a named trigger condition for revisiting the choice
is recorded (v2.3's adapter rewiring), and Phase 29 inherits complete evidence,
not an open investigative question. This differs materially from the SPIKE-03
key-scheme gap below, which is a scope question (does a canonical scheme exist
for queue runs at all), not a tie-break between two already-measured options.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/26-canonical-record-spike/26-DECISION.md` | Findings + Recommendation covering SPIKE-01..04, including the post-execution domain correction | VERIFIED | Present, `## Recommendation` section present, domain-correction subsection present and dated |
| `docs/design/canonical_record_spike.rst` | Durable, redaction-free summary | VERIFIED | Present, mirrors DECISION.md content including the domain correction; no PII, no "upsert" jargon (grepped clean) |
| `docs/design/design.rst` | Toctree extended | VERIFIED | `canonical_record_spike` entry present at line 47 |
| Scratch branch `spike/26-canonical-record-probe` | Deleted, unmerged | VERIFIED | `git branch -a` shows no such branch anywhere |
| `tmp/`, `local_settings.py` | Removed | VERIFIED | Neither exists on disk |
| Scratch migration `0008_scratch_canonical_record_probe.py` | Absent from every git ref | VERIFIED | `git log --all` for that path returns nothing |
| `solsys_code/` | Unmodified vs. committed state | VERIFIED | `git status --porcelain solsys_code/` is empty; `git diff --stat 77e16b5 HEAD -- solsys_code/` is empty |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `26-DECISION.md` claims (constraint names/fields) | `solsys_code/models.py:120-160` | direct source comparison | VERIFIED | `unique_campaign_run_resolved_window`, `unique_campaign_run_tbd_natural_key`, `campaign_run_window_start_end_null_together` — field tuples and conditions match exactly |
| `26-DECISION.md` SPIKE-02 claims | classical/LCO/Gemini/campaign adapter source | direct source comparison | VERIFIED | All 4 construction sites and lookup dicts match current code |
| `26-DECISION.md` D-05 (80×5=400) | `solsys_code/calendar_utils.py:36-53` `SITE_TELESCOPE_MAP` | direct source comparison | VERIFIED | 5 sites carry `1m0`, 2 carry `2m0` — exact match |
| `26-DECISION.md` E10-blank-timezone finding | `Observatory` model, obscode E10 | read-only single-field query | VERIFIED | `Observatory.objects.get(obscode='E10').timezone` returns `''` in the real dev DB, confirmed independently |
| `docs/design/canonical_record_spike.rst` | `docs/design/design.rst` toctree | grep | VERIFIED | Entry present, `sphinx-build` resolves it with no orphan warning |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Sphinx build of the durable page succeeds under the exact pre-commit invocation | `sphinx-build -M html ./docs <out> -T -E -d <doctrees> -D exclude_patterns=notebooks/*,_build` | `build succeeded, 9 warnings` (all 9 pre-existing, unrelated to `canonical_record_spike.rst` — confirmed by grepping warning output for the file name: none) | PASS |
| Constraint definitions match the doc's quoted output | `sed -n` on `solsys_code/models.py:120-160` | field tuples/names match verbatim | PASS |
| Adapter lookup dicts match the doc's quoted claims | `sed -n` on 4 adapter source files | match verbatim | PASS |
| `Observatory.timezone` for E10 is blank as claimed | read-only Django shell query | `''` | PASS |
| All phase-26 commit hashes cited in SUMMARYs actually exist | `git cat-file -e <hash>` × 12 | all resolve | PASS |
| No `solsys_code/` file differs from pre-phase commit `77e16b5` | `git diff --stat 77e16b5 HEAD -- solsys_code/` | empty | PASS |

### Probe Execution

Not applicable — this phase has no `scripts/*/tests/probe-*.sh` convention and none is declared in the plans.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| SPIKE-01 | 26-02-PLAN.md | `source` vocabulary + constraint coexistence | SATISFIED | Executable evidence, cross-checked against live model code |
| SPIKE-02 | 26-01-PLAN.md | Per-adapter identity key mapping | SATISFIED | Cross-checked against live adapter code |
| SPIKE-03 | 26-02-PLAN.md, 26-03-PLAN.md | Canonical event-key scheme + fan-out answer | **PARTIALLY SATISFIED** | Fan-out answered; key-scheme universality gap (see truth #3) |
| SPIKE-04 | 26-01-PLAN.md, 26-03-PLAN.md | Migration + attribution strategy, rename checklist | SATISFIED | Migration applied cleanly against real-copy data; 6-point checklist measured |

No orphaned requirements — all four Phase-26 requirement IDs are claimed by at least one plan and REQUIREMENTS.md's traceability table names Phase 26 for all four.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` debt markers in the committed deliverables (the one `TBD` hit in `26-DECISION.md` refers to the codebase's own `TBD`-window `CampaignRun` domain concept — an established term matching `unique_campaign_run_tbd_natural_key`'s constraint name — not a debt marker). No email addresses or `upsert` jargon in either committed document (CLAUDE.md's plain-English and PII gates both grep clean). Pre-existing repo-wide `ruff`/format drift is correctly identified in `deferred-items.md` as unrelated to this phase (confirmed: `git diff 77e16b5` empty for every flagged file).

### Human Verification Required

None. All must-haves for this phase resolve to VERIFIED or FAILED programmatically; nothing requires interactive/visual confirmation beyond what the phase itself already completed (the manual `/calendar/` load is recorded, with verbatim human-reported text, inside `26-DECISION.md`'s SPIKE-04 criterion 4(c) section).

### Gaps Summary

Three of SPIKE-01/02/03/04's four criteria (01, 02, 04) plus the durable-doc
success criterion (5) are fully and independently verifiable against the current
codebase — not just asserted in SUMMARY.md — and all hold up. Teardown is clean:
no scratch branch, no `tmp/`, no `local_settings.py`, no scratch migration in any
ref, `solsys_code/` byte-identical to its pre-phase commit. No PII, no banned
jargon.

The one real gap is narrow but consequential: **SPIKE-03's "stable across all
four pipeline stages" claim does not hold without qualification.** A domain
correction from the project owner, landing after the requirement was already
ticked complete, rescoped the locked `RUN:{run_pk}:{date}` key to
classically-scheduled runs only — and left open, for queue-scheduled runs
(including `CampaignRun` pk=1, the very row the phase goal names as its anchor
evidence), whether per-night calendar projection applies at all. This is
disclosed transparently in both `26-DECISION.md` and the durable page (not
hidden), and is structurally similar in spirit to the already-accepted D-11
write-strategy deferral — but it is a different kind of open question (scope of
applicability, not a tie-break between two fully-measured options), and it means
Phase 29 does not yet have a full "decision instead of a question" for the run
type its own flagship real-data anchor belongs to. ROADMAP.md's Phase 29 success
criterion 2 has not been updated to reflect this and still describes the
pre-correction per-night/per-day semantics unconditionally.

Recommend one of: (a) an explicit human override accepting this narrower scope,
paired with a small edit to ROADMAP/REQUIREMENTS wording so "Complete" accurately
reflects "locked for classical runs; queue-run projection deferred to Phase 29
alongside the write-strategy choice"; or (b) a short closure plan producing the
same kind of measured evidence for queue-run projection that D-11 already
produced for the write-strategy question, before Phase 27 begins consuming this
spike's output.

---

*Verified: 2026-07-27T17:53:09Z*
*Verifier: Claude (gsd-verifier)*
