---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 01
subsystem: investigation
tags: [django, campaignrun, schema-spike, uniqueconstraint, sqlite, evidence-gathering]

requires: []
provides:
  - "31-DECISION.md header, investigation-only framing, and the SCHEMA-01/SCHEMA-02 evidence sections"
  - "Dated real dev-DB CampaignRun snapshot (49 rows, 0 null-campaign, 4 pre-existing telinst/window collisions ignoring campaign)"
  - "Grep-verified campaign-FK read-path blast-radius inventory (5 sites would raise under a nullable FK, 1 hot)"
  - "Constructed-input constraint probe of all three D-05 candidate schema shapes against both existing partial UniqueConstraints"
affects: [31-02, 31-03, 31-05]

actuals:
  tokens: 9610
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Disposable-file-copy write-probe posture (Phase 26 precedent): schema_editor()-applied in-process changes against tmp/31-spike-db-copy.sqlite3, no migration file, no rollback for positive-case writes"
    - "Fixed two-tag evidence vocabulary: Confirmed against real rows / Constructed-input code-path check"
    - "Fingerprint-before/after discipline (stat -c '%s %Y') for every step touching src/fomo_db.sqlite3"

key-files:
  created:
    - .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md
    - tmp/31_dbsnapshot_probe.py
    - tmp/31-dbsnapshot.txt
    - tmp/31_constraint_probe.py
    - tmp/31-constraint-probe.txt
    - tmp/31-spike-db-copy.sqlite3
  modified: []

key-decisions:
  - "None locked yet by this plan -- SCHEMA-01/02's actual recommendation is deferred to plan 31-02's checkpoint; this plan only gathers the evidence the recommendation will rest on."
  - "Real dev-DB evidence confirms D-06's low-migration-risk premise directly: 0/49 CampaignRun rows have a null campaign FK today."
  - "Constructed-input evidence directly falsifies Option B (single shared sentinel TargetList) as a zero-risk choice: 4 real telescope_instrument/window_start/window_end tuples already recur across different real campaigns, meaning collapsing every non-campaign run onto one shared campaign value would refuse a second row for at least 4 real-shaped situations."
  - "Option A (nullable FK) trades cheapest migration for total loss of duplicate-prevention on non-campaign rows -- both existing partial UniqueConstraints stop discriminating entirely once campaign is null, confirmed with two constructed null-campaign rows that did not collide."
  - "A candidate source_identifier field is additive alongside both existing constraints (re-confirmed to still fire with the new field present) and is idempotent across all three real adapter identity shapes (real LCO url, constructed Gemini/classical values)."

requirements-completed: [SCHEMA-01, SCHEMA-02]

coverage:
  - id: D1
    description: "Dated, fingerprinted snapshot of the real CampaignRun population (SCHEMA-01), plus a grep-verified inventory of every non-test read path that dereferences the campaign FK"
    requirement: "SCHEMA-01"
    verification:
      - kind: other
        ref: "python manage.py shell < tmp/31_dbsnapshot_probe.py > tmp/31-dbsnapshot.txt; grep -q '^FINGERPRINT_UNCHANGED=PASS$' tmp/31-dbsnapshot.txt"
        status: pass
      - kind: other
        ref: "for f in $(grep -rl '\\.campaign\\b' --include='*.py' solsys_code src | grep -v '/migrations/' | grep -v '/tests/'); do grep -q \"$f\" 31-DECISION.md || echo MISSING; done"
        status: pass
    human_judgment: false
  - id: D2
    description: "All three D-05 candidate schema shapes (nullable FK / single sentinel TargetList / per-proposal placeholder TargetList) exercised for real against both existing partial UniqueConstraints on a disposable DB copy, plus a candidate source_identifier field checked for additive coexistence"
    requirement: "SCHEMA-02"
    verification:
      - kind: other
        ref: "cp src/fomo_db.sqlite3 tmp/31-spike-db-copy.sqlite3 && python manage.py shell < tmp/31_constraint_probe.py > tmp/31-constraint-probe.txt; grep -q '^GUARD_DISPOSABLE_COPY=OK$' tmp/31-constraint-probe.txt && [ \"$(grep -c '^PASS:' tmp/31-constraint-probe.txt)\" -ge 5 ] && [ \"$(grep -c '^FAIL:' tmp/31-constraint-probe.txt)\" -eq 0 ]"
        status: pass
      - kind: other
        ref: "stat -c '%s %Y' src/fomo_db.sqlite3 == FINGERPRINT_BEFORE value in tmp/31-dbsnapshot.txt (unchanged)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-01
status: complete
---

# Phase 31 Plan 01: Schema/Identity Evidence Gathering (SCHEMA-01/02) Summary

**Measured 49 real `CampaignRun` rows (0 null-campaign), found 4 pre-existing telescope/window
tuple collisions that falsify a shared-sentinel schema shape, and constructed-input-probed all
three D-05 candidate shapes against both live partial `UniqueConstraint`s on a disposable DB
copy — Option A loses all duplicate protection, Option B/C both correctly collide as designed.**

## Performance
- **Duration:** ~25min
- **Started:** 2026-09-02T04:20:00Z
- **Completed:** 2026-09-02T04:32:00Z
- **Tasks:** 3
- **Files modified:** 1 committed (`31-DECISION.md`, built up across 3 commits) + 4 disposable git-excluded `tmp/` artifacts

## Accomplishments
- Ran a read-only, fingerprinted probe against the real `src/fomo_db.sqlite3` and recorded a
  dated `CampaignRun` snapshot in `31-DECISION.md`: 49 rows, 0 with a null `campaign` FK, 43
  resolved-window / 6 TBD, a per-`Source` breakdown, and — the load-bearing new measurement —
  4 pre-existing `(telescope_instrument, window_start, window_end)` tuples that already recur
  across *different* real campaigns, ignoring `campaign` entirely.
- Produced a grep-verified inventory of every non-test, non-migration site in `solsys_code/`
  and `src/` that reads the `campaign` FK across four access classes (direct attribute read,
  pass-through assignment, queryset traversal, template read): 5 sites would raise
  `AttributeError` under a nullable FK (1 of them hot — `campaign_reconciler.event_title()`,
  called on every `reconcile_run()`), 2 survive as pass-throughs, 12 are query-time-only
  traversals, and 0 genuine template-level FK-object traversals exist anywhere.
- Wrote and ran a disposable-copy constraint probe exercising all three D-05 candidate schema
  shapes for real: Option A (nullable FK) creates two null-campaign rows sharing
  telescope/instrument/window with zero `IntegrityError` — both existing partial constraints
  silently stop discriminating. Option B (single sentinel `TargetList`) and Option C
  (per-proposal placeholder) both correctly refuse a genuine duplicate under either existing
  constraint. A candidate `source_identifier` field was added in-process, shown idempotent
  across all three real adapter identity shapes (a real LCO portal `url`, a constructed Gemini
  `GEM:` key, and a synthesized classical key), and shown not to disturb either existing
  constraint's behavior.

## Task Commits
1. **Task 1: dev-DB snapshot, real-data evidence** - `1553504` (docs)
2. **Task 2: campaign FK read-path blast-radius inventory** - `0159d6a` (docs)
3. **Task 3: candidate-shape constraint probe** - `9373ddc` (docs)

**Plan metadata:** committed alongside this SUMMARY (see below).

## Files Created/Modified
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` -
  Phase 31's decision document: header, investigation-only framing, and three evidence
  subsections (`SCHEMA-01 evidence - real dev-DB snapshot`, `SCHEMA-01 evidence - campaign FK
  read-path blast radius`, `SCHEMA-02 evidence - candidate-shape constraint probe`)
- `tmp/31_dbsnapshot_probe.py` (git-excluded) - read-only Django shell script snapshotting the
  real `CampaignRun` population with a fingerprint-before/after guard
- `tmp/31-dbsnapshot.txt` (git-excluded) - captured transcript of the above
- `tmp/31_constraint_probe.py` (git-excluded) - disposable-copy schema-editor probe exercising
  all three candidate shapes plus the candidate `source_identifier` field
- `tmp/31-constraint-probe.txt` (git-excluded) - captured transcript of the above (19 `PASS:`,
  0 `FAIL:`)
- `tmp/31-spike-db-copy.sqlite3` (git-excluded) - disposable copy of the dev DB, left in place
  for reuse by later plans; deleted by plan 31-05

## Decisions Made
None locked by this plan — SCHEMA-01/02's actual recommendation between the three candidate
shapes is deferred to plan 31-02's checkpoint, per the roadmap. This plan's job was only to
gather the evidence that recommendation will rest on. See the `key-decisions` frontmatter
field above for the concrete findings this evidence establishes.

## Deviations from Plan
None - plan executed exactly as written. All three tasks' `<verify>` commands and
`<acceptance_criteria>` passed on the first attempt with no fix-up needed.

## Issues Encountered
One self-caught issue during Task 1 authoring (not a deviation from the plan, a mid-task
correction before any verification ran): the probe script's original docstring literally
spelled out the four forbidden write-style ORM call patterns (`.save(`, `.create(`, `.update(`,
`.delete(`) as prose, which would have failed the task's own acceptance criterion ("contains no
occurrence of `.save(`...") since the criterion is a literal substring check with no
docstring/comment exemption. Rephrased the docstring to describe the same constraint without
spelling out the substrings before running the verify step — caught before any verification
attempt, so no fix-attempt budget was consumed.

## Next Phase Readiness
The evidence this plan gathered is what plan 31-02's checkpoint will decide between. Three
concrete findings feed that decision directly: (1) 0/49 rows have a null campaign today, so
migration risk for *existing* rows is confirmed zero regardless of which shape wins; (2) Option
A's cheap migration cost is offset by a real, counted 5-site read-path blast radius (1 hot) and
by total loss of duplicate-prevention for non-campaign rows; (3) Option B's collision risk is
no longer hypothetical — 4 pre-existing telescope/window tuples already recur across different
real campaigns, meaning Option B would need `source_identifier` folded into the constraint
itself (not just added alongside it) to be viable, a detail plan 31-02 should weigh explicitly.
Option C's open placeholder-lifecycle question (what happens when a proposal later gets a real
campaign) remains genuinely unresolved and is recorded in `31-DECISION.md` for 31-02 to address
if Option C is the chosen shape. `tmp/31-spike-db-copy.sqlite3` is left in place (not deleted)
for reuse by any later plan that needs the same disposable-copy posture before plan 31-05's
final cleanup. No blockers for 31-02, 31-03, or 31-04.

---
*Phase: 31-foundation-spikes-run-identity-unattended-invocation*
*Completed: 2026-09-01*

## Self-Check: PASSED

All created files verified present (`31-DECISION.md`, `tmp/31_dbsnapshot_probe.py`,
`tmp/31-dbsnapshot.txt`, `tmp/31_constraint_probe.py`, `tmp/31-constraint-probe.txt`,
`tmp/31-spike-db-copy.sqlite3`); all three task commit hashes (`1553504`, `0159d6a`,
`9373ddc`) confirmed present in `git log --oneline --all`.
