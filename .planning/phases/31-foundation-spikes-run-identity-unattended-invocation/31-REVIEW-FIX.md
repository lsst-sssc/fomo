---
phase: 31-foundation-spikes-run-identity-unattended-invocation
fixed_at: 2026-09-02T18:10:00Z
review_path: .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-REVIEW.md
iteration: 1
findings_in_scope: 22
fixed: 22
skipped: 0
status: all_fixed
---

# Phase 31: Code Review Fix Report

**Fixed at:** 2026-09-02
**Source review:** .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 22 (6 critical + 13 warning, per `fix_scope: critical_warning`, plus
  3 Info findings — IN-05, IN-06, IN-07 — explicitly added to scope per task instructions,
  since IN-06 is the identical root cause as CR-01 and IN-05/IN-07 share the same evidence
  section as WR-01/WR-02)
- Fixed: 22
- Skipped: 0

All work happened in an isolated git worktree (`gsd-reviewfix/31-<pid>`, since
`workflow.use_worktrees` is `true`), fast-forwarded onto `issue37-telescope-runs-calendar`
after the last commit, then torn down. 14 commits landed on that branch; each ran the
project's full pre-commit suite (ruff, Sphinx build, Django unit tests) and passed.

**Scope note (this phase is investigation-only):** as instructed, no file under
`solsys_code/` or `src/` was touched. All fixes landed in one of two places:

1. **Git-tracked, committed:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
   `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`,
   and `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-05-SUMMARY.md`.
2. **Git-excluded, edited but never committed:** `tmp/31_constraint_probe.py` and
   `tmp/31_dbsnapshot_probe.py`. `tmp/` is gitignored (`.gitignore:145`) and
   `git ls-files tmp/` prints nothing both before and after this fix pass — these edits are
   local filesystem changes only, verified via Tier 1 (re-read) and Tier 2
   (`python -c "import ast; ast.parse(...)"`, both files parse clean) and are **not** part of
   any commit. Per task instructions, neither script was re-run against the database; the
   transcripts quoted in `31-DECISION.md` (produced by the pre-fix script versions) were left
   untouched as the historical record, with corrections noted inline where a fix would now
   produce a different result.

## Fixed Issues

### CR-01: Stray `</content>` tool-wrapper tag committed into the published Sphinx page

**Files modified:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-05-SUMMARY.md`
**Commit:** `6f38e93`
**Applied fix:** Deleted the trailing `</content>` line from both files (the same
tool-wrapper leak in two locations, per the task's explicit instruction to fix IN-06
alongside CR-01 in the same pass). `grep -rn '</content>'` over the worktree (excluding
`.git/`) now returns nothing.

### CR-02: Disposable-copy guard is tautological and bypassable

**File:** `tmp/31_constraint_probe.py` (gitignored — no commit)
**Applied fix:** Replaced the guard with: capture the real `DATABASES['default']['NAME']`
via `realpath` *before* any reassignment (the original bug: reading back a value the script
had just assigned itself one line earlier); resolve `COPY_PATH` repo-root-relative via
`__file__` rather than CWD-relative; abort if the copy file does not already exist (no
longer lets SQLite silently create an empty one); abort if the copy path resolves to the
real DB or is a link/samefile to it. Verified via Tier 1 re-read and Tier 2
(`ast.parse`, clean). Not re-run against the database, per task instructions — the historical
`GUARD_DISPOSABLE_COPY=OK` transcript line is unaffected by this fix (a correct guard would
have printed the same line on the run that actually happened).

### CR-03: Block B asserts nothing — both branches print `PASS`

**File:** `tmp/31_constraint_probe.py` (gitignored — no commit)
**Applied fix:** The `IntegrityError` branch (the "unexpectedly fired" case) now prints
`FAIL:` instead of `PASS:`, matching every other block's convention. Verified via Tier 1 and
Tier 2. **Not re-run** — the transcript quoted in `31-DECISION.md:191` still reads
`PASS: constraint no longer fires...` from the original (correct, as-run) behavior; only the
label on the *other*, never-taken branch was wrong, so the historical transcript's content is
unaffected by this fix, and no re-run was needed to confirm that.

### CR-04: Classical `source_identifier` key cannot reproduce the tolerance match it mirrors

**Files modified:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `3a71535`
**Applied fix:** Reworded the "Per-ingest-path value" decision cell to state the
quantisation rule the classical key needs (bucket `start_time` to a 5-minute boundary,
matching the tolerance match's granularity) instead of implying it is an exact mirror of a
tolerance match; noted a 5-minute bucket still splits a pair straddling a boundary. Flagged
that the probe exercised a `datetime.date`, not the `datetime` the loader computes, so the
round-trip result needs re-confirming with a real datetime before Phase 32 relies on it.
Added a matching correction note to `31-DECISION.md`'s SCHEMA-02 recommendation section
(same overclaim originated there too).

### CR-05: New partial unique constraint never negative-tested, but called "proven"

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `c5d6850`
**Applied fix:** Scoped the "proven, not just argued" claim to what was actually tested (the
additive property — both existing constraints still fire after adding the field) and added
that the new constraint's own duplicate-rejection/NULL-coexistence behaviour is asserted from
Django/SQLite semantics, not independently measured by a positive/negative control in the
probe. Checked `31-DECISION.md` for the same overclaim; its parallel passage
(`"confirmed empirically, not just argued"`, SCHEMA-02 section) only ever claims the two
*existing* constraints still fire — which was tested — so no `31-DECISION.md` change was
needed for this finding.

### CR-06: `flock -n` claimed to transfer unchanged from interim host to Kubernetes

**Files modified:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `11f7bf8`
**Applied fix:** Split the "Invocation mechanism" decision cell (and the corresponding
opening statement in `31-DECISION.md`'s SCHED-07 recommendation) by deployment target:
interim host keeps cron + `flock -n`, verified; AWS Kubernetes needs a `CronJob` per command
with `concurrencyPolicy: Forbid` (`flock` retained only as a same-pod belt-and-braces guard,
never the cross-pod migration story), carried forward as an open item for whoever owns the
AWS deployment.

## WR-01: Missed-invocation layer 1 unimplementable as written

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `de43c0e`
**Applied fix:** Extended the "Missed-invocation visibility" decision cell to record what
`31-DECISION.md` already documented but the published page dropped: `_notify_staff()`
builds its link via `self.request.build_absolute_uri(...)` (no `request` in a management
command) and calls `send_mail(..., fail_silently=True)`; Phase 34 must extract a request-free
helper with an explicit base URL and use `fail_silently=False` with a caught-and-logged
exception.

## WR-02: False "the real crontab already runs this project's own management commands"

**Files modified:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `ca300ab`
**Applied fix:** Reworded all three occurrences of the "FOMO management-command entries"
mislabel (two in `31-DECISION.md`, one in the published page's Overlap-prevention cell) to
what the transcript actually supports: a sibling checkout's crontab invokes Django
management commands from the third-party `tom_jpl`/`tom_dataservices` plugins, not FOMO's
own commands. The underlying conclusion (an unguarded crontab is a live condition) is
unaffected.

## WR-03/WR-04: Nullable-FK cost understated / chosen shape's downside omitted

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `cf525cd`
**Applied fix:** Both findings land in the same decision cell, so fixed together: amended
"costs a single field-level migration and needs no backfill" to also name the read-site null
guard obligation (cross-referencing `31-DECISION.md`'s inventory), and added the
Block-B-measured accepted cost (SQL NULL-inequality silently disables
`unique_campaign_run_resolved_window` for null-campaign rows) that the table previously
listed only for the two *rejected* options.

## WR-05: `31_dbsnapshot_probe.py` docstring's privacy claim is factually wrong

**Files modified:** `tmp/31_dbsnapshot_probe.py` (gitignored — no commit),
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit (DECISION.md only):** `f9edae7`
**Applied fix:** The script's docstring claimed it "never selects contact_person or
contact_email values"; Block (C) actually does
`values_list('telescope_instrument', 'contact_person')`. Corrected the docstring to the true,
weaker guarantee (no individual value is ever *printed*, only the resulting count) rather
than adding a hashing step, since the counts already recorded in the transcript are
unaffected by a docstring-only correction. The identical overclaim was independently present
in `31-DECISION.md`'s evidence-posture statement and SCHEMA-01 evidence section — both
corrected in the same commit.

## WR-06: Block B omits the `transaction.atomic()` wrapper other blocks use

**File:** `tmp/31_constraint_probe.py` (gitignored — no commit)
**Applied fix:** Wrapped Block B's second `create()` call in `with transaction.atomic():`,
matching Blocks C/D/E, as part of the same edit that fixed CR-03 (same lines). Verified via
Tier 1/Tier 2; not re-run.

## WR-07: Four-collision figure measured over campaign-bearing rows, extrapolated

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `5f40db6`
**Applied fix:** Reworded the "Schema shape" cell's opening sentence to state what was
actually measured (a non-zero recurrence rate among the 49 existing, all-campaign-bearing
rows) rather than implying a direct measurement of the future collision itself.

## WR-08: Transcripts disclose developer-host infrastructure detail, now public

**File:** `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `4ad1dcc`
**Applied fix:** Redacted the developer hostname, the exact kernel build string, one
unrelated commented-out cron entry, two `flock`-guarded cron entries for an unrelated
project, and an internal container-registry hostname from the committed transcript
quotations, keeping only what the evidence needs (flock present; 3 active `manage.py`
entries; 0 guarded). Per task instructions, the git-excluded `tmp/*.txt` transcript files
themselves were **not** edited — they never leave the local machine — only what is quoted
into the committed document was redacted.

## WR-09: Transcripts carry no command/redaction marker

**File:** `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `f472cfd`
**Applied fix:** Per task instructions (do not regenerate the transcript), added a note to
`31-DECISION.md`'s evidence section stating the four commands that produced the host-probe
transcript and an explicit redaction accounting (0 environment-assignment lines redacted —
the substitution ran and found nothing secret-shaped, which is now stated rather than left
ambiguous).

## WR-10: `source_identifier` mixes three cardinality granularities, no rule recorded

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `80cb1c9`
**Applied fix:** Added a sentence to the per-ingest-path cell stating the cardinality each
key implies (one `CampaignRun` per LCO request, per Gemini observation, per classical
telescope/instrument/night) and noting that a many-requests-to-one-run design would require
moving the LCO key up to the request-group URL.

## WR-11: Guard's `OK` line printed before any evidence of the intended copy

**File:** `tmp/31_constraint_probe.py` (gitignored — no commit)
**Applied fix:** Added a `BASELINE_CAMPAIGNRUN_ROWS` print and an abort-if-zero check
immediately after the guard passes and models are importable, before any write. Verified via
Tier 1/Tier 2; not re-run — the historical transcript's `RESOLVED_DB_NAME=...` /
`GUARD_DISPOSABLE_COPY=OK` lines are unaffected (a correctly-populated copy was in fact used,
per the pk-arithmetic cross-check already in `31-DECISION.md`).

## WR-12: Model class and global settings mutated with no restore

**File:** `tmp/31_constraint_probe.py` (gitignored — no commit)
**Applied fix:** Added a prominent `WARNING` paragraph to the module docstring stating the
process is permanently poisoned after it runs (connection NAME re-pointed globally,
`campaign.null` flipped on the live model class, `source_identifier` permanently added via
`add_to_class()`) and that it must always be invoked as a one-shot
`manage.py shell < ...`, never pasted into an interactive session — the documented
alternative to a `try/finally` restore, since wrapping ~200 lines of positive-case writes in
`try/finally` was judged higher-risk to introduce indentation errors than the documented
alternative.

## WR-13: Recommended field definition omits the width the probe used

**Files modified:** `docs/design/run_identity_and_unattended_invocation_spike.rst`,
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
**Commit:** `d6417fc`
**Applied fix:** Recorded `CharField(max_length=500)` — the width the probe actually
exercised — in the published page. Cross-checking `31-DECISION.md`'s own SCHEMA-02
recommendation found it separately specified an **unvalidated** `max_length=255`, which
does not match `tmp/31_constraint_probe.py:157`'s actual `max_length=500`; corrected it to
the validated value with a note explaining the discrepancy.

## IN-05: RST omits the two guarded crontab entries its companion excludes

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `98e0b18`
**Applied fix:** Added the parenthetical noting two further `flock -n`-guarded crontab
entries for an unrelated project on the same host — the strongest supporting evidence for
the mechanism, previously visible only in `31-DECISION.md`. Included in this fix pass per
explicit task instructions (same evidence section as WR-02, already in scope).

## IN-06: Same stray `</content>` tag in the phase summary

**File:** `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-05-SUMMARY.md`
**Commit:** `6f38e93` (same commit as CR-01, per explicit task instruction to fix both
together since it is the identical one-line root cause)
**Applied fix:** Deleted the trailing `</content>` line.

## IN-07: RST omits that a `flock -n` skip is itself a silent failure

**File:** `docs/design/run_identity_and_unattended_invocation_spike.rst`
**Commit:** `98e0b18`
**Applied fix:** Appended a sentence to the "Overlap prevention" cell stating that a
`flock -n` non-zero exit, left unlogged, is indistinguishable from a healthy no-op, and that
Phase 34 must log it. Included in this fix pass per explicit task instructions.

## Skipped Issues

None — all 22 in-scope findings were fixed. (IN-01 through IN-04 were left untouched, as
they are Info-severity findings outside `fix_scope: critical_warning` and were not named in
the task's explicit scope-override list.)

## Verification performed

- **Where verification ran:** all git-tracked edits (the `.rst` page, `31-DECISION.md`,
  `31-05-SUMMARY.md`) were made and committed inside the isolated worktree
  `/home/tlister/git/fomo_devel/.claude/worktrees/rf-31-1362021-1788369623` on temp branch
  `gsd-reviewfix/31-1362021`, which was fast-forward-merged onto
  `issue37-telescope-runs-calendar` and then torn down (`git worktree remove --force`,
  `git branch -D`, sentinel removed). The two gitignored `tmp/*.py` edits were made directly
  in the main checkout (`/home/tlister/git/fomo_devel/tmp/`), since `git worktree add` does
  not copy gitignored/untracked files into a new worktree — these edits were never inside the
  worktree and are reproducible from the current main checkout as shown.
- Each of the 14 commits ran this project's full pre-commit suite (ruff, `ruff-format`,
  Sphinx build, Django unit test suite) via the pre-commit hook, and all passed.
- Post-fix acceptance checks, re-run in the main checkout after fast-forward:
  - `git status --porcelain -- solsys_code src` → empty.
  - `git ls-files tmp/` → empty.
  - Credential/email/hex regex gate over `31-DECISION.md` and the design page → clean.
  - `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D 'exclude_patterns=notebooks/*,_build'`
    → `build succeeded, 12 warnings` (down from the pre-fix baseline of 13 in
    `31-05-SUMMARY.md` — the stray-`</content>` docutils warning is gone; none of the
    remaining 12 warnings reference `run_identity_and_unattended_invocation_spike.rst`).
  - `python3 -c "import ast; ast.parse(...)"` on both edited `tmp/*.py` scripts → clean.

## Logic-verification caveat

Several fixes (CR-02's guard rewrite, CR-03/WR-06's Block B correction, WR-11's baseline
check) change probe-script *logic*, not just prose, and were deliberately **not re-run**
against the database per task instructions — the historical transcripts quoted in
`31-DECISION.md` were produced by the pre-fix script versions and were left as the recorded
evidence. Tier 1 (re-read) and Tier 2 (`ast.parse`) confirm the scripts are syntactically
correct and structurally sound, but semantic correctness of the corrected guard/assertion
logic has not been re-exercised against a live database. A human should sanity-check these
script logic changes (particularly CR-02's `realpath`/`samefile` guard chain) the next time
this kind of probe is run, rather than treating them as re-verified by this fix pass.
