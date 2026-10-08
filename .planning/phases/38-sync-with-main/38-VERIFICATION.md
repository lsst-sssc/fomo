---
phase: 38-sync-with-main
verified: 2026-10-08T02:46:09Z
status: passed
score: 53/53 must-haves verified
covered_files:
  - ".copier-answers.yml"
  - ".github/workflows/smoke-test.yml"
  - ".github/workflows/testing-and-coverage.yml"
  - ".gitignore"
  - ".planning/phases/38-sync-with-main/38-01-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-01-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-02-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-02-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-03-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-03-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-04-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-04-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-05-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-05-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-05-red-evidence.json"
  - ".planning/phases/38-sync-with-main/38-06-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-06-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-07-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-07-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md"
  - ".planning/phases/38-sync-with-main/38-PATTERNS.md"
  - ".planning/phases/38-sync-with-main/38-PR43-BODY.md"
  - ".planning/phases/38-sync-with-main/38-RESEARCH.md"
  - ".pre-commit-config.yaml"
  - "CLAUDE.md"
  - "docs/conf.py"
  - "docs/design/design.rst"
  - "docs/installation.rst"
  - "docs/notebooks.rst"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "pyproject.toml"
  - "solsys_code/admin.py"
  - "solsys_code/apps.py"
  - "solsys_code/management/commands/backfill_lco_observations.py"
  - "solsys_code/tests/test_bootstrap5_rendering.py"
  - "solsys_code/tests/test_urls.py"
  - "solsys_code/tests/test_views.py"
  - "src/fomo/settings.py"
  - "src/fomo/urls.py"
covered_digest: "v3:sha256:4eb80ee27b30ee5464517621e3fb20fd9bc4767e19aa778660dcbb149f3455f2"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 46/46
  gaps_closed:
    - "G-38-1 (UAT Test 1, the previous report's WR-01 human item): docs/installation.rst and the PR #43 body (file and live) now say local_settings.py lives at src/fomo/local_settings.py and that a file at the repository root is silently ignored; the FOMO_BASE_URL note and runbook fresh-host step 2 are path-qualified"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "Future merge of PR #58 (production-deploy -> main) into this branch or into PR #43: PR #58's `from .local_settings import *` is named `src.fomo.local_settings` under manage.py (DJANGO_SETTINGS_MODULE=src.fomo.settings), so taking its import line while keeping this branch's guard (`if exc.name != 'fomo.local_settings': raise`, settings.py:415-423) makes every checkout without the file crash at settings import; taking its whole block (`except ImportError: pass`) silently undoes WR-32"
    category: architectural
    reason: "Raised as 38-REVIEW WR-02. Reproduced by the verifier with a scratch package (src.fomo.settings -> exc.name 'src.fomo.local_settings'; fomo.settings -> 'fomo.local_settings'). Not a Phase 38 gap: origin/main (a910c17) does not contain PR #58, settings.py is unchanged, and no v2.5 phase (39-42) syncs with main again. Resolve by recording the constraint (keep the absolute `from fomo.local_settings import *` spelling when resolving the settings.py conflict) in STATE.md deferred items or a todo for Phase 41's triage"
    evidence_status: "reproduced in scratch (not a repo test)"
human_verification:
  - test: "38-REVIEW WR-01 (with the docs half of WR-02): docs/installation.rst:111-112, the last sentence of the new warning, reads '``main`` makes the same change through pull request #58 (``from .local_settings import *``), so ``src/fomo/local_settings.py`` is the location on every current FOMO branch.' Today PR #58 is OPEN (mergedAt null) and origin/main:src/fomo/settings.py:366 is still the bare `from local_settings import *`, so the present-tense claim is false for main. A main-based host whose operator followed it would move the file to src/fomo/, which is not on main's sys.path, and silently run on the development defaults, the failure the section exists to prevent. 'The same change' is also only true of the file location, not of the module name (see the advisory). The sentence is not yet published: the four 38-07 commits are unpushed and issue37-code-only was not re-snapshotted. The developer saw this text in the Task 2 round-2 diff and answered 'apply'."
    expected: "Developer chooses before the next push or snapshot refresh: (a) accept the sentence as a forward-looking statement (PR #58 'will go to main soon') and record an override; or (b) a one-line /gsd-quick docs fix, either dropping the sentence (the rest of the warning holds without it) or scoping it to versions, for example 'From this release on, FOMO reads the file only from src/fomo/local_settings.py; main reads the top-level local_settings module until PR #58 is merged.' The live PR #43 body says '`main` gets the same location through PR #58', which is accurate about location; (b) need not touch it."
    why_human: "No 38-07 must-have asserts this sentence, and the developer approved its wording with knowledge of PR #58's state. Whether a present-tense claim about an unmerged PR is acceptable is a documentation-policy call, not something the codebase can settle."
---

# Phase 38: Sync with main Verification Report

**Phase Goal:** The branch carries everything `main` already has — its dependency floors, ruff 0.16.9, LINCC python-project-template v2.2.0 and the Django test runner in CI — and the full suite passes on tomtoolkit 3.1.0, so every later v2.5 phase works on the merged tree.
**Verified:** 2026-10-08T02:46:09Z
**Status:** human_needed
**Re-verification:** Yes. This follows gap-closure plan 38-07, which closes UAT gap G-38-1. The previous report was 07a4061: human_needed, 46/46.

## Goal Achievement

The previous report had three human items. UAT resolved two of them: WR-02 (accepted, no CI step added) and the judgment-tier prohibitions (confirmed by the developer). The third, WR-01, became gap G-38-1. Plan 38-07 closes it, and I checked that against the files, git and GitHub rather than against 38-07-SUMMARY.md:

- **Installation section.** `docs/installation.rst` has the `.. _local-settings:` section, defined once, and it comes before "Initializing FOMO and the database" and the first `runserver`. It says production overrides go in `src/fomo/local_settings.py`, that the file is gitignored (`.gitignore:61` matches it, per `git check-ignore`) and that settings.py imports it as `fomo.local_settings`. Its warning says a copy at the repository root (or in `src/`) is no longer read, and it gives the move and check commands.
- **Path-qualified mentions.** The FOMO_BASE_URL note and runbook fresh-host step 2 both name `src/fomo/local_settings.py` and link `:ref:`local-settings``. The runbook change is a 1/1 numstat.
- **Docs match the code.** settings.py still has `from fomo.local_settings import *`. I ran the documented check command and it printed `~/git/fomo_devel/src/fomo/local_settings.py`.
- **Live PR #43.** The body equals 38-PR43-BODY.md after normalisation. The PR is still a draft, head `issue37-code-only` @ 846be34, base `main`, with no labels and no review requests. GitHub's `userContentEdits` shows exactly one body edit since 38-04's edit (2026-10-07T18:41:37Z), at 2026-10-08T02:27:34Z. That is after the revise-round commits (a060f9d 01:21Z, 82a097c 01:23Z) and before the SUMMARY commit (02:28Z).
- **Nothing pushed.** `git ls-remote` gives the same tips as `$HOME/tmp/phase38-07-tips.txt` (846be34, 49be149).

Since 07a4061, the only changes outside `.planning/` are `docs/installation.rst` (+36/-1) and `docs/runbooks/telescope_runs_calendar.rst` (1/1). Truths 1-46 were therefore regression-checked, not re-derived.

Status is `human_needed` because of one new item: review finding WR-01, a factually false present-tense sentence in the new warning. No must-have failed.

### Observable Truths

Truths 1-46 are the previous report's truths (regression). Truths 47-53 are 38-07's `must_haves.truths`.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC#1: one merge commit; origin/main an ancestor; nothing rewritten | ✓ VERIFIED (regression) | origin/main still a910c17; `merge-base --is-ancestor origin/main HEAD` ok; 38-07 added only ordinary commits |
| 2 | SC#2: floors and full suite on tomtoolkit 3.1.0 | ✓ VERIFIED (regression) | pyproject.toml and code unchanged since 07a4061; previous 2182-test OK log stands (no Python file changed) |
| 3 | SC#3: ruff 0.16.9 clean, 0.2.1 gone | ✓ VERIFIED (regression) | no Python or hook config changed |
| 4 | SC#4: Django runner + coverage in CI, no pytest, LINCC v2.2.0 | ✓ VERIFIED (regression) | .github, pyproject, .pre-commit-config unchanged |
| 5 | SC#5: PR #43 still a draft; body covers the four pillars and links the runbook | ✓ VERIFIED | `gh pr view 43`: isDraft true, OPEN; body has all four `###` sections, the runbook path, "stays a draft until v2.5's Phases 39-42 land", "Related to #37." |
| 6 | Goal: branch carries main's ada2000 removal of `alerts/` | ✓ VERIFIED (regression) | src/fomo/urls.py unchanged since 07a4061 |
| 7-23 | 38-01/38-02/38-03 truths (parents, D-02, nav_items, admin, conf.py, URL order, deleted tests, no commit under MERGE_HEAD, pyproject, venv, boot, style ordering, SIM103, CI one-line diffs, hooks, segfault tag, notebooks) | ✓ VERIFIED (regression) | none of their files changed since 07a4061 |
| 24 | CLAUDE.md, installation.rst, settings in step (38-02 D-04) | ✓ VERIFIED (regression, re-run) | installation.rst did change, so I re-ran 38-02's checks: `* timezonefinder>=6.0` present, no "registration" anywhere on the page, settings.py unchanged |
| 25-28 | runbook system-check paragraph, CI-form proof, DB backup/crontab, override comparison | ✓ VERIFIED (regression) | 38-07 changed one runbook line in fresh-host step 2, not the system-check paragraph |
| 29 | D-11 snapshot equals the v2.5 tree minus `.planning/` | ✓ VERIFIED (scoped) | It held at 846be34 when 38-06 ran. HEAD now differs from origin/issue37-code-only only in the two 38-07 docs files, which is planned: 38-07's third prohibition puts the re-snapshot out of scope and names it as the follow-up |
| 30 | Worktree removed; primary checkout on the v2.5 branch | ✓ VERIFIED | `git worktree list`: only the primary checkout, on issue37-telescope-runs-calendar |
| 31-46 | 38-05/38-06 truths (alerts guard, RED/GREEN, CI wiring, 2182 OK, check, planning-doc wording, PR head, no force-push, nothing private published) | ✓ VERIFIED (regression) | their files unchanged; origin tips unchanged (846be34 / 49be149) |
| 47 | G-38-1 installation section: label `local-settings`, the exact heading, placed before "Initializing FOMO and the database"; src/fomo/local_settings.py, gitignored, `fomo.local_settings`; warning with "no longer read", silent dev defaults, move and check commands | ✓ VERIFIED | Plan's Task 1 assertion script re-run: OK. Heading underline is 46 characters. All 12 required phrases are present in the section. Label defined once |
| 48 | G-38-1 path-less mentions: FOMO_BASE_URL note and runbook step 2 name `src/fomo/local_settings.py` + `:ref:`local-settings``; runbook 1/1 | ✓ VERIFIED | Same script: the note reads "in this host's ``src/fomo/local_settings.py`` (see :ref:`local-settings`)", and no path-less "in this host's ``local_settings.py``" is left. Runbook line 2048 matches exactly, the old line is gone, numstat `1 1` |
| 49 | Documentation matches the code: settings.py imports `fomo.local_settings`; the check command prints .../src/fomo/local_settings.py | ✓ VERIFIED | settings.py:415 `from fomo.local_settings import *`, diff vs a003a36 empty. Verifier ran `python manage.py shell -c "import fomo.local_settings as m; print(m.__file__)"`, last line `~/git/fomo_devel/src/fomo/local_settings.py`. See the WR-01 human item for a claim about *main* that does not hold |
| 50 | PR body file Settings line: `fomo.local_settings` instead of top-level (c0f883d), move from the repository root to `src/fomo/local_settings.py`, silently ignored; only line changed (1/1); settings list and D-12 sections intact | ✓ VERIFIED | `git diff --numstat a003a36 HEAD -- 38-PR43-BODY.md` → `1 1`. Exactly one `- [x] **Settings:**` line, containing all 12 required tokens (incl. FOMO_* settings, EMAIL_BACKEND, tom_registration). It does not name the installation page. All D-12 markers present. The wording was revised at Task 2 (round 1: "revise") to be branch-neutral and to name `src/` as well, and it still meets every element of the truth |
| 51 | Live PR #43 replaced only after "apply"; the pre-edit body equalled the pre-plan file; afterwards body == file, still a draft, head/base unchanged | ✓ VERIFIED | Live body == file (normalised) and != the BASE file, so the edit landed. isDraft true, `issue37-code-only` → `main`, headRefOid 846be34. `userContentEdits`: one edit (02:27:34Z) after 38-04's (18:41:37Z), with nothing in between, so no GitHub-side edit was overwritten. The developer's verbatim "apply" is recorded in the SUMMARY |
| 52 | Both rst files parse under docutils (ref/doc stubbed) at warning level; no added docs line >120 columns | ✓ VERIFIED | Verifier ran the plan's docutils script: OK. One `warning` node, with both `literal_block`s nested inside it. Longest added docs line: 119 columns |
| 53 | Docs only: settings.py (import and guard) unchanged; nothing pushed; no re-snapshot | ✓ VERIFIED | `git diff --stat a003a36 HEAD -- . ':(exclude).planning'` → only the two docs files. settings.py clean in the working tree. ls-remote equals the recorded tips |

**Score:** 53/53 truths verified (0 present, behavior-unverified)

### Prohibitions (ADR-550)

| Prohibition | Tier | Disposition | Evidence |
|-------------|------|-------------|----------|
| 38-01..38-06 test-tier items | test | ✓ held (regression) | history and files unchanged |
| 38-01/38-03/38-04/38-06 judgment-tier items (package gate, backup before migrate, no live portal calls, wheels only diffed, checkout never switched) | judgment | ✓ resolved by human | 38-UAT.md Test 3: "Developer confirmed all five prohibitions from the session" (result: pass) |
| 38-07: no change to src/fomo/settings.py | test | ✓ held | `git diff a003a36 HEAD -- src/fomo/settings.py` empty; working tree clean |
| 38-07: nothing on PR #43 before "apply"; only the body edited | test | ✓ held | One `userContentEdits` entry in the window. Title unchanged, no labels, 0 review requests, isDraft true, head/base unchanged |
| 38-07: nothing pushed, no re-snapshot | test | ✓ held | ls-remote == `phase38-07-tips.txt`; origin/issue37-code-only still 846be34 |
| 38-07: PR body Settings line does not point at the installation page | test | ✓ held | the Settings line contains no "installation" |
| 38-07: task commits touch only the three planned files | test | ✓ held | `git show --stat`: 5421abb (installation.rst, runbook), cf24051 (PR43-BODY), a060f9d (PR43-BODY, installation.rst), 82a097c (installation.rst). The SUMMARY, STATE/ROADMAP and review commits are workflow commits, not task commits. No CLAUDE.md, notebook, VERIFICATION, REVIEW, DISPOSITION or UAT edit in a task commit |

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | 38-REVIEW WR-02: in a future PR #58 merge, the relative import is incompatible with this branch's `exc.name` guard | architectural | Reproduced in scratch, but it concerns a merge that Phase 38 does not perform: PR #58 is not on origin/main, and no v2.5 phase syncs again. Record the constraint (keep the absolute `fomo.local_settings` spelling) for whoever merges PR #58 into this line. See frontmatter `advisory` |

### Code Review Findings (38-REVIEW.md, cf01184), Classification

| ID | Classification | Reasoning |
|----|----------------|-----------|
| WR-01 ("is the location on every current FOMO branch"; PR #58 unmerged) | **Human decision (WARNING)**, not a must-have gap | Confirmed false today: `gh pr view 58` → OPEN, mergedAt null. `origin/main:src/fomo/settings.py:366` is `from local_settings import *`. None of 38-07's truths or prohibitions, and no roadmap SC, asserts anything about main's import, and truth 49's stated content holds. The consequence would be real if a main-based operator acted on it, but the text is unpublished (unpushed commits, no re-snapshot), and the developer approved it at round 2 knowing PR #58 was pending. So this is a policy choice between accepting it as forward-looking (override) and a one-line fix before the next push, and it is routed to human verification rather than FAILED. The live PR body's phrase ("`main` gets the same location through PR #58") is accurate about location and is not affected |
| WR-02 (PR #58's relative import is not "the same change") | **Split.** The docs wording ("the same change") lives in the WR-01 sentence and is resolved by the same decision. The merge-compatibility concern is **out of Phase 38's scope** (advisory 1) | Verified: under `src.fomo.settings` the relative import's missing-module name is `src.fomo.local_settings`, which the guard re-raises. settings.py is unchanged by 38-07 (a prohibition held) and Phase 38's merge was with a910c17, which has no PR #58, so the merged tree works today. The risk lands when PR #58 merges and PR #43 (or a later sync) has to resolve the settings.py conflict. No later v2.5 phase covers it, so it is not deferred. It should be recorded |
| IN-01 (`ModuleNotFoundError` also fires for a dependency missing inside the file) | Info | Diagnostic precision of the check command. The plan's truth only requires a check command that prints the path, and it does |
| IN-02 ("nothing reports it" overstated vs `check_unattended` / `check --deploy`) | Info | The phrase is one of the plan's required strings. It is accurate about the misplaced file itself |
| IN-03 (pre-move `migrate` / cron runs went to the default SQLite) | Info | Upgrade-procedure hardening outside the gap's three missing items |
| IN-04 (location rule in the parentheses is approximate; `mv` without `-n`) | Info | Both `mv` commands are correct from the repository root (review confirms). `-n` is a cheap hardening |
| IN-05 ("anything set there replaces the default" vs the FOMO_STATE_DIR trap) | Info | The runbook documents the trap at :1986-1991. A clarifying clause would help |
| IN-06 (wrap width of the two in-place edits) | Info | Cosmetic. Under 120 columns, as the plan requires |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/installation.rst` | the `local-settings` section; path-qualified FOMO_BASE_URL note | ✓ VERIFIED | contains `.. _local-settings:` (line 79); referenced from :150 and runbook :2048; docutils clean |
| `docs/runbooks/telescope_runs_calendar.rst` | fresh-host step 2 path-qualified | ✓ VERIFIED | contains "``src/fomo/local_settings.py`` file (see :ref:`local-settings`)"; 1/1 |
| `.planning/phases/38-sync-with-main/38-PR43-BODY.md` | Settings line with the move | ✓ VERIFIED | contains `` `fomo.local_settings` ``; 1/1; equals live PR #43 body |
| (truths 1-46 artifacts) | as in the previous report | ✓ VERIFIED (regression) | unchanged except installation.rst, re-checked under truth 24 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| docs/installation.rst | src/fomo/settings.py | module name `fomo.local_settings`; documented check imports it | ✓ WIRED | settings.py:415 matches `from fomo\.local_settings import \*`; the check command resolves the file at src/fomo/ |
| docs/runbooks/telescope_runs_calendar.rst | docs/installation.rst | `:ref:`local-settings`` | ✓ WIRED | the label exists once; the ref is in the runbook at :2048 (Sphinx resolution is not run locally; docutils with stubbed roles parses cleanly) |
| 38-PR43-BODY.md | PR #43 live body | `gh pr edit 43 --body-file` after "apply" | ✓ WIRED | live body == file; one edit at 02:27:34Z |
| (previous links) | | | ✓ WIRED (regression) | files unchanged |

### Data-Flow Trace (Level 4)

Not applicable. Plan 38-07 is documentation only, and plans 38-01..38-06 add no dynamic-data rendering.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Plan 38-07 content assertions | plan Task 1 verify script #1 | OK | ✓ PASS |
| rst structure | plan Task 1 docutils script (report_level 2, halt_level 2) | OK; the warning node holds 2 literal blocks | ✓ PASS |
| Documented check command | `python manage.py shell -c "import fomo.local_settings as m; print(m.__file__)"` | `.../src/fomo/local_settings.py` | ✓ PASS |
| PR body file contents | plan Task 1 PR-body script (re-implemented) | OK | ✓ PASS |
| Live PR state | `gh pr view 43 --json isDraft,headRefName,baseRefName,state,body`; GraphQL `userContentEdits` | draft, OPEN, issue37-code-only→main, body == file, 1 edit in window | ✓ PASS |
| Nothing pushed | `git ls-remote` vs recorded tips | equal | ✓ PASS |
| WR-02 claim | scratch package, relative import under `src.fomo.settings` vs `fomo.settings` | exc.name `src.fomo.local_settings` vs `fomo.local_settings` | confirmed (advisory) |
| WR-01 claim | `gh pr view 58`; `git show origin/main:src/fomo/settings.py` | OPEN / bare `from local_settings import *` | confirmed (human item) |

### Probe Execution

Step 7c: SKIPPED. No plan declares a probe script, and none exists under `scripts/*/tests/probe-*.sh`.

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| SYNC-01 | 38-01 | ✓ SATISFIED | Truths 1, 7, 14 (regression) |
| SYNC-02 | 38-01, 38-03 | ✓ SATISFIED | Truths 2, 15, 16 (regression) |
| SYNC-03 | 38-02 | ✓ SATISFIED | Truths 3, 18, 19, 24 |
| SYNC-04 | 38-01, 38-05, 38-06 | ✓ SATISFIED | Truths 4, 6, 12, 31-33, 40 (regression) |
| SYNC-05 | 38-02, 38-03, 38-04, 38-06 | ✓ SATISFIED | Truths 4, 20, 26, 45 (regression). D-05 trade-off accepted at UAT Test 2 |
| SYNC-06 | 38-02 | ✓ SATISFIED | Truths 4, 21, 24 |
| SYNC-07 | 38-03, 38-05 | ✓ SATISFIED | Truths 2, 36 (regression) |
| SYNC-08 | 38-04, 38-06, 38-07 | ✓ SATISFIED | Truths 5, 45, 50, 51: the live description now also says what a deployment upgraded from main must move. Still a draft |

No orphaned requirements: REQUIREMENTS.md maps exactly SYNC-01..08 to Phase 38, and every one is claimed by at least one plan. Bookkeeping note (not a gap, carried): REQUIREMENTS.md still shows SYNC-01, -02, -03 and -06 unchecked, with traceability "Gaps Found", left over from the earlier revert. The orchestrator should tick all eight.

### Paired Docs (CLAUDE.md rule)

No Python module in the notebook map changed, so no notebook is due. The only `docs/runbooks/` page affected (`telescope_runs_calendar.rst`, fresh-host step 2) was updated in the same plan. The plan scoped the runbook's later path-less mentions (lines 1988-2120) as following from step 2. CLAUDE.md:106 still names `local_settings.py` with no path. That is a listed follow-up which PR #58 also rewrites on main, and CLAUDE.md is not a paired doc under the rule.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| docs/installation.rst, docs/runbooks/telescope_runs_calendar.rst, 38-PR43-BODY.md (added lines) | — | TBD/FIXME/XXX/TODO/HACK, trailing whitespace, tabs, home path, `api_key` | — | none found |
| docs/installation.rst | 111-112 | Present-tense claim about an unmerged PR | ⚠️ Warning | See the WR-01 human item |
| docs/_build/ (untracked, git-ignored, predates the phase) | — | Stale local Sphinx build | ℹ️ Info | Carried. Not tracked or published |

### Human Verification Required

1. **WR-01: the PR #58 sentence in the new warning** (`docs/installation.rst:111-112`).
   - **Test:** Decide whether to keep "``main`` makes the same change through pull request #58 ..., so ``src/fomo/local_settings.py`` is the location on every current FOMO branch."
   - **Expected:** Either accept it as forward-looking and record an override, or fix it with a one-line `/gsd-quick` docs change before the next push or snapshot refresh: drop the sentence, or scope it to "from this release on; main until PR #58 merges". Optionally, in the same quick task, record advisory 1 (keep the `fomo.local_settings` spelling when PR #58's settings.py conflict is resolved) in STATE.md's deferred items.
   - **Why human:** It is false today (PR #58 is OPEN, main has the bare import), but no must-have asserts it. It is still unpublished, and the developer approved the wording knowing PR #58 was pending.

### Gaps Summary

There are no gaps. G-38-1 is closed in all three of its missing items:

1. The installation guide states the location, the silent-fallback trap, and the move and check commands. It is placed before `migrate`, and it is consistent with the import in settings.py and with the check command run on this checkout.
2. The PR body file's Settings line carries the move (1/1).
3. The live PR #43 body equals that file and is still a draft. It was edited exactly once, after the developer's "apply", with no GitHub-side edit overwritten.

Settings.py is unchanged and nothing was pushed, as the plan's prohibitions required.

The earlier two human items were resolved at UAT. Status is `human_needed` only because of review finding WR-01: one factually false present-tense sentence about main in the new warning, not yet published, which needs a keep-or-fix decision. WR-02's docs half falls under the same decision. Its merge-compatibility half is an advisory for whoever merges PR #58, outside Phase 38. IN-01..IN-06 are informational.

Carried follow-ups, not gaps:
- the next issue37-code-only snapshot refresh (D-11 recipe), which brings the two docs files onto PR #43's diff;
- CLAUDE.md:106 path;
- REQUIREMENTS.md checkboxes.

---

_Verified: 2026-10-08T02:46:09Z_
_Verifier: Claude (gsd-verifier)_
