---
phase: 26
slug: canonical-record-spike
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-27
---

# Phase 26 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

**Investigation-only phase.** This phase ships no application code, so validation here means
*how each claim in the decision doc gets evidenced* — an executable check's printed output, a
real test-suite run's pass/fail result, or an explicit manual step — not application test
coverage. Source: `26-RESEARCH.md` § Validation Architecture (all runtime figures measured
directly, not estimated).

---

## Test Infrastructure

This repo has **two independent test suites** (CLAUDE.md). Phase 26's own throwaway evidence
scripts use neither directly — they run via `manage.py shell` — but D-02's rename measurement
runs the Django suite.

| Property | Value |
|----------|-------|
| **Framework** | Two suites: `pytest` (over `tests/`, `src/`, `docs/`) and the Django test runner (over `solsys_code/`) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (`testpaths = ["tests", "src", "docs"]` — does **not** collect `solsys_code/`); Django uses settings module `src.fomo.settings`, no separate test-settings file |
| **Quick run command** | `./manage.py test solsys_code.tests.test_calendar_template` — the single most load-bearing module; covers rename integration points #3 (view `prefetch_related`) and #4 (calendar template), the two "safe by construction" ones the spike must actually confirm |
| **Full suite command** | The **named, narrow six-module list** from `26-RESEARCH.md` § Measuring the Rename. Deliberately excludes `test_ephem_utils.py` and `test_views.py` |
| **Estimated runtime** | Quick: **21.7s** wall (24 tests). Full narrow list: **2m 5.6s** wall (242 tests, all passing). `python -m pytest`: **0.6s** (1 test, unrelated to `solsys_code`) |

**Do not run literal `./manage.py test solsys_code`** for this phase. It timed out past 2
minutes on one attempt and **segfaulted inside REBOUND/ASSIST** on a second (8-minute budget)
before reaching a normal exit. Use the narrow list. (`26-RESEARCH.md` Pitfall 3.)

The ~9s fixed overhead in every Django run is `manage.py check` importing the full URLconf,
which transitively imports `solsys_code.ephem_utils` and furnishes SPICE kernels. Narrowing
the test label does **not** avoid this — `DiscoverRunner.run_tests()` calls `check`
unconditionally.

---

## Sampling Rate

- **After every `models.py`/migration edit, before any test run:** `python manage.py check`
  (~3.5s) as an import/syntax smoke test, before spending 22s or 2min on a real run
- **After every task commit:** `./manage.py test solsys_code.tests.test_calendar_template` (~22s)
- **After every plan wave:** the narrow six-module list (~2m 6s)
- **Before `/gsd-verify-work`:** narrow six-module list must be green
- **Max feedback latency:** 22 seconds (quick run); 126 seconds (full narrow list)

### Scratch-DB safety sampling (highest-value check in the procedure)

1. **Immediately after writing or refreshing `local_settings.py`**, and again before *every*
   subsequent `migrate` or write-script invocation:
   `python manage.py shell -c "from django.conf import settings; print(settings.DATABASES['default']['NAME'])"`
   — confirm it ends in the scratch-copy filename. Sub-second; run it every time.
2. **Not needed before `manage.py test`** — Django's test runner always builds its own isolated
   in-memory test DB regardless of `DATABASES['default']['NAME']` (verified: no `test_*.sqlite3`
   ever appeared on disk). The guard matters for `migrate`, `shell`-based writes, and `runserver`.
3. **Immediately after applying the throwaway migration:** re-run the Fresh DB Snapshot query and
   diff against the pre-migration baseline (31 `CampaignRun`s, 20 `CalendarEvent`s, 11 companion
   rows). A companion-row count that dropped to 0 is the signature of Pitfall 4 — an autodetected
   `DeleteModel`/`CreateModel` instead of `RenameModel` destroying the scratch data. Stop and
   re-author the migration by hand.
4. **After each D-11 prototype run:** diff the resulting `CalendarEvent` count in pk=1's window
   against the expected figure for that scenario (15 adopt / 15 gap-fill / 26 rejected baseline).
   Check per-run, not batched.
5. **At discard (Pitfall 5):** `git status --porcelain` on the real phase-26 branch, **and**
   separately confirm `local_settings.py` is gone (`ls local_settings.py` should error) and `tmp/`
   is removed. A leaked gitignored override is invisible to `git status`.

---

## Per-Task Verification Map

*The planner fills this table once plans exist — one row per task, mapping each to the evidence
artifact it produces. The Evidence Map below is the authoritative source for what each task must
prove; the table below is its per-task projection.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| *(planner to fill)* | | | | | | | | | ⬜ pending |

### Evidence Map — ROADMAP criteria + SPIKE-01..04

| Criterion | Requirement | Evidence Artifact | Command | Confidence Level |
|-----------|-------------|-------------------|---------|------------------|
| 1 | SPIKE-01 (`source` vocabulary + `IntegrityError` coexistence) | Printed PASS/FAIL lines from the integrity check script, including the negative control proving the original resolved-window constraint still fires on a genuine duplicate | `python manage.py shell < tmp/26_integrity_check.py` against the scratch copy | **Confirmed against real rows** — `CampaignRun` pk=1 and its real 11 LCO-sourced companion rows |
| 2 | SPIKE-02 (per-adapter identity-key-to-run mapping) | Snapshot query output for classical (blank `url`) and LCO (real portal `url`) rows; source read of `sync_gemini_observation_calendar.py`'s `GEM:{prog}/{obsid}` key construction | Read-only `manage.py shell -c` against the real `src/fomo_db.sqlite3` for classical/LCO; direct source inspection for Gemini | Classical + LCO: **confirmed against real rows**. **Gemini: confirmed via constructed input / code reading only** — D-18 establishes zero real `GEM:` events exist. The decision doc must state this distinction explicitly (Phase 18 D-09 precedent), not present it at equal confidence |
| 3 | SPIKE-03 (canonical event-key scheme + stage-2 fan-out) | Three-copy adopt / gap-fill / rejected-baseline event-count comparison | Three `tmp/26_reconciler_prototype.py` runs, one per scratch copy | **Confirmed against real rows** for the adopt-vs-gap-fill comparison (real pk=1 window, real 11 LCO events). The D-05 80×5=400 fan-out figure is a **computed value from real field values** (pk=29's real window length × `SITE_TELESCOPE_MAP`'s real site count), not an executed DB check — state that distinction |
| 4 | SPIKE-04 (migration + rename checklist) | (a) throwaway migration applies cleanly to the scratch copy; (b) narrow six-module suite pass/fail output; (c) `/calendar/` manual load | (a) `python manage.py migrate solsys_code`; (b) narrow-list command above; (c) manual, below | (a) **Confirmed against real rows** — the scratch copy is a real copy of the dev DB. (b) **Confirmed via constructed input** — Django `TestCase` builds an isolated test DB from factories, so a green run proves the rename doesn't break *tested behaviors*, not that it held against the real 11 companion rows; that proof is (a) plus the SPIKE-01 script. (c) manual |
| 5 | Durable `docs/design/` page | Clean `sphinx-build` + toctree entry present | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` (the exact pre-commit invocation) | Build-tooling quality gate — confirms the page is reachable and RST-valid; not a factual claim needing a confidence tier |

---

## Wave 0 Requirements

**None — existing infrastructure covers all phase requirements.**

- The Django test runner, settings module, and `solsys_code/tests/` suite already exist and
  already exercise all four rename-relevant integration points.
- `local_settings.py` is an existing, already-gitignored project mechanism — nothing to scaffold
  or add to `.gitignore`.
- `insert_or_create_calendar_event()` — the one piece of application code every throwaway script
  reuses — already exists and is unchanged by this phase.
- The throwaway evidence scripts (`tmp/26_integrity_check.py`, `tmp/26_reconciler_prototype.py`)
  are investigation tooling created *during* execution, written/run/discarded in the same session.
  They are not Wave 0 deliverables.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `/calendar/` renders after the companion-record rename | SPIKE-04 / D-02 | No automated assertion substitutes for loading the rendered page; CONTEXT.md D-02 asks for it explicitly | 1. Run the DB-path guard. 2. `python manage.py runserver` with `local_settings.py` pointed at the migrated scratch copy; load `/calendar/?year=2026&month=7`. 3. **Pass:** HTTP 200, no traceback, rendered month shows the same event count as the snapshot baseline. 4. Record verbatim: HTTP status line and visible event count |

**Known gap in this check, to record explicitly:** D-20 confirms all 11 real companion rows have
`is_verified=1` — **zero** real rows exercise the `is_verified=False` dashed-border fallback branch
(`calendar.html:228,244`). A clean load proves the `prefetch_related()` string and the
`event.telescope_label_meta` accessor still resolve, but **cannot** visually confirm the dashed
border renders. Two acceptable ways to close it: (a) on the **scratch copy only**, temporarily flip
one companion row's `is_verified` to `False`, reload, confirm the border, and note in the decision
doc that this was a deliberately-constructed check; or (b) rely on `test_calendar_template.py`,
which already has fixture-based coverage of that branch and is already in the quick-run command.
State in the decision doc which of the two the evidence rests on.

A `django.test.Client()`-based text substitute may be run in the same session as a corroborating,
non-interactive second data point — but not as a replacement for the actual browser load.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or a recorded manual-only justification
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references — *satisfied: no Wave 0 needed*
- [x] No watch-mode flags
- [x] Feedback latency < 126s
- [x] Every Evidence Map row is tagged "confirmed against real rows" vs "confirmed via
      constructed input", and the decision doc preserves that distinction (Phase 18 D-09)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated (Phase 30 plan 30-04, D-08 reconciliation)

## Validation Audit 2026-08-31

Reconciled by Phase 30 plan 30-04 (D-08). Investigation-only phase, no application code, so
audit means re-checking that each Evidence Map claim has real recorded evidence, not running
new tests.

| Evidence Map row | Verified against | Result |
|---|---|---|
| Criterion 1 (SPIKE-01) | `26-DECISION.md:370-392` — 4 blocks, 5 PASS lines, 0 FAIL lines, including both negative controls on the existing unique constraints | Confirmed present |
| Criterion 2 (SPIKE-02) | `26-DECISION.md` classical/LCO real-row snapshot output; Gemini confirmed via constructed input only, and the doc states that distinction explicitly | Confirmed present |
| Criterion 3 (SPIKE-03) | `26-DECISION.md:816` — 5 PASS/0 FAIL across the three-copy adopt/gap-fill/rejected-baseline comparison; D-05 fan-out figure flagged as computed, not executed | Confirmed present |
| Criterion 4a (migration) | `26-DECISION.md:370-392` PASS block re-run against the scratch copy | Confirmed present |
| Criterion 4b (test suite) | `26-DECISION.md:254-255` — narrow six-module suite green post-fix, cited per integration point | Confirmed present |
| Criterion 4c (manual `/calendar/` load) | `26-DECISION.md:321-362` — a human loaded `/calendar/?year=2026&month=7` against the scratch copy with a companion row temporarily flipped to `is_verified=False`; HTTP 200, `[CANCELLED]` dashed border with hover text confirmed rendering (closing the "Known gap" option (a) this file itself flagged) | Confirmed present, closes the documented known gap |
| Criterion 5 (durable docs page) | `docs/design/canonical_record_spike.rst` exists (19782 bytes), wired into `docs/design/design.rst:47`'s toctree → `docs/index.rst:23`; `python -m sphinx -b html docs docs/_build/html -q` produces no warning naming this file (pre-existing warnings are all unrelated: `autoapi/fomo/urls`, `sync_lco_observation_calendar_demo.ipynb`, `ESO_How_to_download_data.ipynb`) | Confirmed present |

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
