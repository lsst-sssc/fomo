---
phase: "33"
slug: "series-identity-reconciler-inversion"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-03"
updated: "2026-09-03"
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Populated from `33-RESEARCH.md` § "Validation Architecture" during plan revision 1.
> Runtimes below are **measured on this host on 2026-09-03**, not estimated.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` / `TransactionTestCase` via `python manage.py test`. No pytest — `pyproject.toml`'s pytest config is LINCC-template legacy and does not collect the Django app tests (CLAUDE.md § Testing). |
| **Config file** | none — Django's own test discovery. Settings module `src.fomo.settings`, set by `manage.py`. |
| **Test database** | in-memory. `src/fomo/settings.py:126-131` sets no `TEST` key, so the SQLite backend resolves `TEST['NAME'] or ':memory:'` → `file:memorydb_default?mode=memory&cache=shared`. No test-database file is written, and concurrent same-wave runs cannot collide. |
| **Quick run command** | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_attribution_views` |
| **Quick run measured runtime** | **81 s** wall (133 tests; 70.9 s in-runner + ~10 s Django/SPICE import) |
| **Narrowest useful command** | `python manage.py test solsys_code.tests.test_campaign_reconciler` — **46 s** wall (45 tests). Use this mid-task when only the reconciler is in play. |
| **Full suite command** | see the fenced block below — it contains pipes and cannot be quoted inside a table cell without escaping. [VERIFIED verbatim: `.planning/config.json` → `workflow.test_command`] |
| **Full suite measured runtime** | **424 s** wall (~7 min), green on the pre-phase tree |
| **Why the floor is tens of seconds** | Importing `solsys_code.views` / `ephem_utils` furnishes SPICE kernels at module load (CLAUDE.md § Architecture). The ~1.6 GB kernel set is already cached in `~/.cache/sorcha`, so this is load time, not download time. It is a fixed per-process cost no test selection avoids. |
| **Lint gate** | `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` (ruff pinned to v0.2.1 by `.pre-commit-config.yaml`; an unpinned `ruff` on PATH is not the enforced gate — 33-CONTEXT D-07) |
| **Notebook gate** | `jupyter nbconvert --to notebook --execute --inplace <notebook>` for the two paired notebooks (CLAUDE.md paired-docs rule) |

Full suite command, verbatim from `.planning/config.json` → `workflow.test_command`:

```bash
LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py | grep -v "tests/test_views\.py$" | sed "s|/|.|g; s|\.py\$||" | tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery
```

`solsys_code.tests.test_views.TestEphemeris` is excluded deliberately — it segfaults in native
ASSIST, which is why the command enumerates labels instead of running bare `manage.py test`.

---

## Sampling Rate

- **After every task commit:** the task's own `<automated>` block — in practice one or two
  targeted `manage.py test` labels, 46–81 s.
- **After every plan wave:** the full suite command, 424 s. Concurrent same-wave runs are safe
  (in-memory test database — see each plan's `<parallel_safety>` block), so this does not need
  serialising to one plan per wave.
- **Before `/gsd-verify-work`:** full suite green, both ruff gates clean, and both paired
  notebooks re-executed with committed output.
- **Max feedback latency:** **81 s** (quick run). Not the template's 33 s: this project's
  process-start SPICE furnish sets a hard floor in the tens of seconds, measured above.
  Continuity is met — every one of the 15 tasks carries a real `<automated>` command, so no
  task, let alone three consecutive, runs unsampled.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 33-01-01 | 01 | 1 | ANNOT-01, ANNOT-02 | T-33-01 / T-33-02 / T-33-03 | A campaign-less run's decoration returns `table_url=None` instead of raising `NoReverseMatch` on the public calendar; the tag exposes no contact field or `source`; no reconciler write lands outside `RUN:` | unit + template | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_calendar_template` | ✅ both modules exist; this task adds `TestAttributedNightSkip` and the campaign-less fixture | ⬜ pending |
| 33-01-02 | 01 | 1 | ANNOT-01 | T-33-03 / T-33-04 | A URL-keyed attributed event survives a sweep byte-identical; a foreign attribution stays `blocked` and is never reset to the reconciling run | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler` | ✅ module exists; this task adds `TestAttributedEventsSurviveReconcile` (Wave 0 gap 2) | ⬜ pending |
| 33-01-03 | 01 | 1 | ANNOT-01 | T-33-03 | Dropping the campaign name from `event_title()` does not drop `RUN_STATUS_CALENDAR_PREFIX`, which `status_border_css` matches on | unit | `python manage.py test solsys_code.tests.test_null_campaign_guards solsys_code.tests.test_write_and_reconcile solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_reconciler` | ✅ all four exist | ⬜ pending |
| 33-02-01 | 02 | 2 | ANNOT-02 | T-33-05 / T-33-06 / T-33-08 | The month cell omits the link for a campaign-less run rather than failing the whole public page; a non-public run renders no marker; the prefetch keeps the marker off the N+1 path | template | `python manage.py test solsys_code.tests.test_calendar_template` | ✅ exists | ⬜ pending |
| 33-02-02 | 02 | 2 | ANNOT-02 | T-33-07 | `run-{pk}` row ids expose only a primary key on rows the non-staff queryset already returns | unit (view) | `python manage.py test solsys_code.tests.test_campaign_views` | ✅ exists | ⬜ pending |
| 33-02-03 | 02 | 2 | ANNOT-02 | T-33-05 / T-33-08 | Decoration survives a from-scratch rewrite of the event's own title/description; month-view query count stays flat as event count grows | template + query-count | `python manage.py test solsys_code.tests.test_calendar_template` then the full suite command | ✅ exists; this task adds the month-cell campaign-less fixture (Wave 0 gap 3) | ⬜ pending |
| 33-03-01 | 03 | 1 | PROJ-04 | T-33-10 | Migration `0017` carries no `RunPython`/`RunSQL`, so no existing companion row is read or rewritten | migration | `python manage.py makemigrations --check --dry-run` (plus `python manage.py migrate` — see the task `<precondition>`) | n/a — generates the migration | ⬜ pending |
| 33-03-02 | 03 | 1 | PROJ-04 | T-33-09 / T-33-12 | A POSTed value for either new link is not bound and cannot be written through a staff form | unit (admin) | `python manage.py test solsys_code.tests.test_admin` | ✅ exists; this task adds the read-only class (Wave 0 gap: D-09 admin test) | ⬜ pending |
| 33-03-03 | 03 | 1 | PROJ-04 | T-33-10 / T-33-11 | Rows seeded against the `0016` schema keep `run`/`is_verified`/`confirmed_by`/`confirmed_at` across the forward migration; a duplicate `observation_record` raises `IntegrityError` while NULLs do not collide | unit + `MigrationExecutor` | `python manage.py test solsys_code.tests.test_calendar_event_meta_links` then the full suite command | ❌ **created by this task** — `solsys_code/tests/test_calendar_event_meta_links.py` (Wave 0 gap 1) | ⬜ pending |
| 33-04-01 | 04 | 2 | ANNOT-01 | T-33-13 / T-33-15 | The helper's clear is one conditional `.update()` filtered on both event and run; its body contains no `delete`, no `is_verified`, and neither new link field | unit | `python manage.py test solsys_code.tests.test_campaign_attribution_views` | ✅ exists | ⬜ pending |
| 33-04-02 | 04 | 2 | ANNOT-01 | T-33-13 / T-33-14 / T-33-16 | The detach step stays scoped to `owned_events(run)`; every call site clears `confirmed_by`/`confirmed_at` with `run`, leaving no stale confirmation stamp | unit | `python manage.py test solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_admin solsys_code.tests.test_attribution_dismissals` | ✅ all four exist | ⬜ pending |
| 33-04-03 | 04 | 2 | ANNOT-01 | T-33-15 / T-33-16 | An unlink deletes nothing and touches no `CalendarEvent` field; an event with `observation_record` set but no `run` is still offered by the attribution queue | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_admin solsys_code.tests.test_campaign_attribution` then the full suite command | ✅ exists; this task adds the detach audit-clearing test (Wave 0 gap 4) | ⬜ pending |
| 33-05-01 | 05 | 3 | ANNOT-01 | T-33-17 / T-33-18 | No contact field reaches committed output; the sweep is previewed with `--dry-run` and asserted to name no non-`RUN:` url **before** it runs for real | notebook (executed) | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | ✅ notebook exists; this task adds the dry-run, diff and skip cells | ⬜ pending |
| 33-05-02 | 05 | 3 | ANNOT-02, PROJ-04 | T-33-17 / T-33-19 | No cell imports `solsys_code.views` / `ephem_utils` (SPICE furnish); no contact field in output | notebook (executed) | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | ✅ notebook exists | ⬜ pending |
| 33-05-03 | 05 | 3 | ANNOT-01, ANNOT-02 | T-33-20 | The runbook's three affected sections say "attributed to" and carry no surviving ownership wording, so the operator doc cannot silently drift from the shipped behaviour | doc grep + full pre-commit | `grep -ci 'attributed to' docs/runbooks/telescope_runs_calendar.rst` and `pre-commit run --all-files` | ✅ runbook exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

RESEARCH.md § "Validation Architecture" named four coverage gaps. All four are closed by tasks
inside this phase's own plan set — each gap's covering task creates the missing test before
running it — so no separate Wave 0 scaffolding plan is needed and every task's `<automated>`
block is a real command rather than a `MISSING` placeholder.

- [x] `solsys_code/tests/test_calendar_event_meta_links.py` — new module: nullability,
      `on_delete=SET_NULL`, one-to-one collision vs. NULL non-collision, and a
      `MigrationExecutor` forward-migration test over pre-`0017` rows.
      **Covered by 33-03 Task 3 (wave 1)** — creates the file. Gap 1, PROJ-04.
- [x] A fixture test proving D-04's "byte-identical after reconcile" for **both** a blank-url and
      a facility-URL-keyed attributed event (no URL-keyed fixture exists anywhere in
      `test_campaign_reconciler.py` today).
      **Covered by 33-01 Task 2 (wave 1)** — adds `TestAttributedEventsSurviveReconcile` to the
      existing module. Gap 2, ANNOT-01.
- [x] A `campaign=None` fixture proving the decoration link is omitted rather than raising
      `NoReverseMatch` (RESEARCH.md Pitfall 1).
      **Covered by 33-01 Task 1 (wave 1)** for the modal (`EventModalCampaignRunLinkTest`) and by
      **33-02 Task 3 (wave 2)** for the month cell. Gap 3, ANNOT-02.
- [x] A test proving `_detach_stale_family_events()` clears `confirmed_by`/`confirmed_at`
      alongside `run` once routed through the shared helper — new behaviour, untested today.
      **Covered by 33-04 Task 3 (wave 2)**, with the call-site rewiring in 33-04 Task 2. Gap 4,
      D-16.
- [x] A D-09 admin read-only test for both new fields (RESEARCH.md's fifth row, no existing
      coverage). **Covered by 33-03 Task 2 (wave 1)** — adds a class to `test_admin.py`.

No framework install is required: Django's test runner is already the project's suite.

---

## Manual-Only Verifications

`workflow.human_verify_mode` is `end-of-phase`, so these are checked once at the phase gate via
`<verify><human-check>`, not as blocking mid-phase checkpoints.

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The month-cell marker's visual form (chip / icon / dot — CONTEXT.md discretion) coexists legibly with the proposal-colour legend, the status ring, and the `is_verified == False` dashed border, and consumes none of the truncated title text | ANNOT-02, D-10 | The template test can assert the marker's presence, its tooltip and that `truncatechars:16`/`:18` still apply to the event's own title only — it cannot judge whether four overlaid visual signals remain readable together | Open the month view on a date range containing an attributed event, a non-attributed event, an unverified event and a run with a status prefix; confirm all four signals are separately readable and no title is visually clipped by the marker |
| The one-time title churn from D-12 (74 reconciler events losing their `"{campaign.name}: "` prefix on the next sweep) reads correctly on the real calendar | ANNOT-02, D-12 | Correctness is unit-tested; whether the shorter titles are still self-explanatory to an operator at a glance is a judgment call | After the phase's first real sweep, scan a month containing `RUN:` events and confirm each is still identifiable without opening the modal, with the campaign name now in the pop-up |
| The `#run-{pk}` anchor scrolls to and highlights the right row in a real browser | ANNOT-02, D-13 | `:target` highlighting is a browser behaviour; the view test can only assert the `id` attribute is emitted | Click "View campaign" from an attributed event's modal and confirm the campaign table opens scrolled to that run's row with the row highlighted |

---

## Validation Sign-Off

- [x] All 15 tasks have a real `<automated>` verify — no `MISSING` placeholders, no Wave 0
      dependency left open
- [x] Sampling continuity: every task is sampled; no run of tasks without automated verify
- [x] Wave 0 covers all MISSING references (5 gaps, each mapped to a covering task above)
- [x] No watch-mode flags in any command
- [x] Feedback latency bounded and **measured**: 46 s narrowest, 81 s quick run, 424 s full
      suite. The template's 33 s target is unreachable on this codebase — process-start SPICE
      furnish is a fixed floor — so the contract is the measured 81 s per task commit.
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-09-03 (planner, plan revision 1). `status` advances from `draft` to
`validated` when `/gsd-validate-phase 33` runs after execution.
