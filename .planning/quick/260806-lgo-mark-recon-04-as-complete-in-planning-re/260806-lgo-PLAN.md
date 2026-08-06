---
phase: quick-260806-lgo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/REQUIREMENTS.md
  - .planning/phases/29-the-reconciler/29-VERIFICATION.md
autonomous: true
requirements: [LGO-01, LGO-02]

must_haves:
  truths:
    - "RECON-04's checkbox in .planning/REQUIREMENTS.md is checked, so no v1 requirement in that file is left unchecked"
    - "RECON-04's traceability-table row reads Complete, so no row in that table reads Pending"
    - "The RECON-04 traceability row carries a note pointing at Phase 28's pre-existing implementation (sync_lco_observation_calendar.py's _build_event_fields()/_time_window(), promoted to calendar_utils.record_time_window()) and at what Phase 29's own scope for RECON-04 actually was"
    - "29-VERIFICATION.md's frontmatter carries a one-entry overrides list recording the human acceptance, attributed to Tim Lister at 2026-08-06T22:27:12Z"
    - "29-VERIFICATION.md's overrides_applied count agrees with the length of its overrides list"
    - "Both files still parse: the verification report's YAML frontmatter loads cleanly and the requirements table keeps its 3-column shape"
  artifacts:
    - path: ".planning/REQUIREMENTS.md"
      provides: "RECON-04 marked complete in both the checkbox list and the traceability table, with the Phase 28 provenance note"
      contains: "- [x] **RECON-04**"
    - path: ".planning/phases/29-the-reconciler/29-VERIFICATION.md"
      provides: "Frontmatter override record for the accepted RECON-04 traceability decision"
      contains: "overrides:"
  key_links:
    - from: ".planning/REQUIREMENTS.md"
      to: ".planning/phases/29-the-reconciler/29-VERIFICATION.md"
      via: "the override's must_have string naming the same RECON-04 change the requirements file now reflects"
      pattern: "RECON-04 marked Complete in REQUIREMENTS.md"
---

<objective>
Record the human decision documented in `29-VERIFICATION.md`'s Traceability Note: RECON-04 is
Complete, not Pending.

The verifier found RECON-04 unchecked in `.planning/REQUIREMENTS.md` while every other RECON-*
requirement was Complete, investigated it, and concluded it is an unupdated tracking artifact rather
than a functional gap — RECON-04's stage-3/4 behavior (a scheduled night narrowing to its
`ObservationRecord`'s window, a completed observation showing its final observed range marked
COMPLETED) is implemented by pre-existing Phase 28 code, and Phase 29's own scope for RECON-04 was
only to prove the reconciler leaves those events alone. The verifier explicitly declined to resolve
it unilaterally and asked for a human call. That call has been made: accept as-is.

Purpose: stop `.planning/REQUIREMENTS.md` from silently contradicting Phase 29's declared
completion, and leave an auditable record of who accepted the discrepancy and why.
Output: two edited planning artifacts. No source code, no tests, no behavior change.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/REQUIREMENTS.md
@.planning/phases/29-the-reconciler/29-VERIFICATION.md

Key source lines (already located — do not re-search for them):

- `.planning/REQUIREMENTS.md:36` — the only unchecked `- [ ]` line in the file
- `.planning/REQUIREMENTS.md:114` — the only table row whose Status cell reads `Pending`
- `.planning/REQUIREMENTS.md:98` — the in-file precedent for annotating a traceability Status cell:
  `Complete — key scheme settled for both classically-scheduled and queue-scheduled runs`
- `.planning/phases/29-the-reconciler/29-VERIFICATION.md:1-12` — frontmatter, currently carrying
  `overrides_applied: 0` on line 6 and no `overrides:` key
- `.planning/phases/29-the-reconciler/29-VERIFICATION.md:113-146` — the Traceability Note, including
  the verbatim suggested `overrides:` block at lines 140-146

CLAUDE.md conventions that apply here: plain English over database jargon in planning docs (no
"upsert"). The paired-docs rule does not apply — this plan touches no module under `solsys_code/`
and no page under `docs/runbooks/`.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Mark RECON-04 Complete in REQUIREMENTS.md, with the Phase 28 provenance note</name>
  <files>.planning/REQUIREMENTS.md</files>
  <action>
Make two edits, both to existing lines. Do not add, remove, or reorder any other line.

Edit 1 — line 36, the RECON-04 checkbox. Change the leading `- [ ]` to `- [x]`. Leave the
requirement statement itself byte-identical: every other requirement in this file states its
requirement with no inline annotation, and the provenance note belongs in the traceability table
(edit 2), not here.

Edit 2 — line 114, the RECON-04 traceability row. Replace the Status cell's `Pending` with a
`Complete — <note>` cell, following the in-file precedent at line 98 (SPIKE-03), which is the only
other annotated Status cell. Write the row as:

`| RECON-04 | Phase 29 — The Reconciler | Complete — the stage-3/4 narrowing-to-record and COMPLETED-marking behavior is implemented by pre-existing Phase 28 code (`sync_lco_observation_calendar.py`'s `_build_event_fields()`/`_time_window()`, promoted to `calendar_utils.record_time_window()`); Phase 29's own scope for RECON-04 was to prove the reconciler leaves those events alone, covered by `TestQueueOwnershipDoesNotTouchRecordEvents`. Accepted 2026-08-06 — see `29-VERIFICATION.md` Traceability Note |`

Keep the em-dash after `Complete` (matching line 98) and keep the row at exactly three cells — the
note lives inside the Status cell, so the pipe count must not change. Backticks inside a markdown
table cell are fine; do not introduce a literal `|` character anywhere in the note text, since that
would split the cell.

Leave the Coverage block below the table untouched: the counts (24 total, 24 mapped, 0 unmapped) are
about mapping, not completion, and are already correct.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && test "$(grep -c '^- \[ \]' .planning/REQUIREMENTS.md)" = "0" && test "$(grep -c '^- \[x\] \*\*RECON-04\*\*' .planning/REQUIREMENTS.md)" = "1" && test "$(grep -c '| Pending |' .planning/REQUIREMENTS.md)" = "0" && test "$(grep -c '^| RECON-04 |.*| Complete — ' .planning/REQUIREMENTS.md)" = "1" && test "$(grep -c 'record_time_window' .planning/REQUIREMENTS.md)" = "1" && test "$(grep -c 'TestQueueOwnershipDoesNotTouchRecordEvents' .planning/REQUIREMENTS.md)" = "1" && awk -F'|' '/^\| RECON-/ {if (NF != 5) {print "BAD CELL COUNT: " $0; exit 1}}' .planning/REQUIREMENTS.md && echo TASK1_OK</automated>
  </verify>
  <done>
`.planning/REQUIREMENTS.md` has zero unchecked requirement boxes and zero `Pending` traceability
rows. The RECON-04 row reads `Complete — ` followed by a note naming
`calendar_utils.record_time_window()` and `TestQueueOwnershipDoesNotTouchRecordEvents`, and all nine
RECON rows still have exactly three cells.
  </done>
</task>

<task type="auto">
  <name>Task 2: Record the accepted override in 29-VERIFICATION.md frontmatter</name>
  <files>.planning/phases/29-the-reconciler/29-VERIFICATION.md</files>
  <action>
Add the `overrides:` block the verifier itself drafted at lines 140-146 of this same file, filling in
the two placeholders. Insert it into the YAML frontmatter (lines 1-12) directly after the
`overrides_applied:` line, before `human_verification:`, so the two override-related keys sit
together:

```
overrides:
  - must_have: "RECON-04 marked Complete in REQUIREMENTS.md"
    reason: "RECON-04's narrowing/COMPLETED behavior is implemented by pre-existing Phase 28 code; Phase 29's own scope for RECON-04 (non-interference) is fully tested and verified. REQUIREMENTS.md's checkbox appears to be an unupdated tracking artifact, not a functional gap."
    accepted_by: "Tim Lister"
    accepted_at: "2026-08-06T22:27:12Z"
```

Reproduce the `must_have` and `reason` strings exactly as the verifier drafted them — they are the
accepted text, not a starting point to reword.

Then change `overrides_applied: 0` to `overrides_applied: 1` on the same line it currently occupies.
This is the internal-consistency half of the task: a one-entry `overrides:` list alongside a count of
zero would be a fresh contradiction of the same kind this quick task exists to remove.

Do NOT touch anything else in this file:
- Leave the `score:` line alone. Its "1 traceability inconsistency noted (WARNING)" text is an
  accurate record of what was true at verification time (2026-08-05); the new `overrides` entry is
  what records the later resolution. Do not rewrite verification history.
- Leave the prose Traceability Note at lines 113-146 exactly as written, including its suggested
  block — it is the rationale this override points back to.
- Leave the `**Status:** human_needed` line in the report body alone; it is out of scope here.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && python3 -c "
import sys, yaml, io
p='.planning/phases/29-the-reconciler/29-VERIFICATION.md'
raw=open(p).read()
assert raw.startswith('---\n'), 'frontmatter must open the file'
fm=raw.split('---\n',2)[1]
d=yaml.safe_load(fm)
o=d.get('overrides')
assert isinstance(o,list) and len(o)==1, 'expected exactly 1 override, got %r' % (o,)
e=o[0]
assert e['must_have']=='RECON-04 marked Complete in REQUIREMENTS.md', e['must_have']
assert e['accepted_by']=='Tim Lister', e['accepted_by']
assert str(e['accepted_at']).startswith('2026-08-06T22:27:12'), e['accepted_at']
assert 'pre-existing Phase 28 code' in e['reason'], e['reason']
assert d['overrides_applied']==len(o), 'overrides_applied=%r disagrees with len(overrides)=%d' % (d['overrides_applied'], len(o))
assert d['status']=='passed'
assert 'human_verification' in d
print('TASK2_OK')
"</automated>
  </verify>
  <done>
`29-VERIFICATION.md`'s frontmatter parses as YAML and contains a one-entry `overrides` list whose
`must_have` is `RECON-04 marked Complete in REQUIREMENTS.md`, attributed to Tim Lister at
`2026-08-06T22:27:12Z`, with `overrides_applied: 1` agreeing with it. `status`, `score` and
`human_verification` are unchanged.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| none crossed | Both edited files are developer-authored planning artifacts under `.planning/`. They are not served, parsed by the Django app, or reachable by any request path; no untrusted input enters this change |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-LGO-01 | Repudiation | The RECON-04 acceptance decision | mitigate | The override records `accepted_by` and `accepted_at` in frontmatter, and the requirements note back-references `29-VERIFICATION.md`'s Traceability Note, so the decision cannot later appear as an anonymous silent edit — this is the point of the task, not a side benefit |
| T-LGO-02 | Tampering | `.planning/REQUIREMENTS.md` completion state | accept | Marking a requirement complete is a tracking assertion, not a gate on any code path; the underlying implementation was independently confirmed by direct read during verification and is git-tracked and reviewable |
| T-LGO-SC | Tampering | npm/pip/cargo installs | n/a | No package-manager install occurs in this plan; no dependency is added, removed or upgraded |
</threat_model>

<verification>
Run both task gates plus one cross-file consistency check:

```bash
cd /home/tlister/git/fomo_devel

# 1. Requirements file: nothing unchecked, nothing Pending
grep -c '^- \[ \]' .planning/REQUIREMENTS.md          # expect 0
grep -c '| Pending |' .planning/REQUIREMENTS.md        # expect 0

# 2. Verification frontmatter parses and the counts agree (Task 2 gate)

# 3. Cross-file: the override's must_have names the change the requirements file now shows
grep -q 'RECON-04 marked Complete in REQUIREMENTS.md' \
  .planning/phases/29-the-reconciler/29-VERIFICATION.md \
  && grep -q '^| RECON-04 |.*| Complete — ' .planning/REQUIREMENTS.md \
  && echo CROSS_FILE_CONSISTENT

# 4. Scope guard: only the two intended files changed
git status --porcelain
```

No test suite run is required — this plan changes no Python, no template, and no notebook.
</verification>

<success_criteria>
- `.planning/REQUIREMENTS.md` line 36 reads `- [x] **RECON-04**` and is the file's only former unchecked box
- `.planning/REQUIREMENTS.md`'s RECON-04 traceability row reads `Complete — ` with a note naming Phase 28's `_build_event_fields()`/`_time_window()` → `calendar_utils.record_time_window()` and `TestQueueOwnershipDoesNotTouchRecordEvents`, and back-references `29-VERIFICATION.md`
- All nine RECON traceability rows still have exactly three cells
- `29-VERIFICATION.md`'s frontmatter is valid YAML with a one-entry `overrides` list (must_have / reason / accepted_by "Tim Lister" / accepted_at "2026-08-06T22:27:12Z") and `overrides_applied: 1`
- `29-VERIFICATION.md`'s `status`, `score`, `human_verification` and body prose are unchanged
- `git status --porcelain` shows exactly two modified files (plus this plan's own SUMMARY)
</success_criteria>

<output>
Create `.planning/quick/260806-lgo-mark-recon-04-as-complete-in-planning-re/260806-lgo-SUMMARY.md` when done
</output>
