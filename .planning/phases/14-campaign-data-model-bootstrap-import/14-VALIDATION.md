---
phase: 14
slug: campaign-data-model-bootstrap-import
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-02
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` via `./manage.py test` (this app's Django-test-suite half; NOT the `pytest` suite, which excludes `solsys_code/` per `pyproject.toml`'s `testpaths`) |
| **Config file** | none dedicated — relies on `src.fomo.settings` (same as every existing `solsys_code` test) |
| **Quick run command** | `./manage.py test solsys_code.tests.test_import_campaign_csv` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~10-30 seconds (no SPICE/ephemeris imports in this phase's code paths) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_import_campaign_csv` (and `test_campaign_models` once split out)
- **After every plan wave:** Run `./manage.py test solsys_code` (full Django suite)
- **Before `/gsd-verify-work`:** Full `./manage.py test solsys_code` green, plus `ruff check .` / `ruff format --check .` clean, plus the demo notebook re-executed and committed with output
- **Max feedback latency:** ~30 seconds (no network calls in the test suite — MPC API interactions must be mocked per CONTEXT.md D-11)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 14-01-T3 | 14-01 | 1 | CAMP-01 | T-14-05 | `CampaignRun` stores full field inventory + required `campaign` FK | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ W0 | ⬜ pending |
| 14-02-T3 | 14-02 | 2 | CAMP-02 | — | Optional `target`; single-target campaign auto-resolves (D-07) without manual setting | unit | `./manage.py test solsys_code.tests.test_import_campaign_csv.TestImportCampaignCsv.test_auto_resolves_single_target_campaign` | ❌ W0 | ⬜ pending |
| 14-01-T3 | 14-01 | 1 | CAMP-03 | — | Two independent status fields, 8 `run_status` values, correct defaults | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ W0 | ⬜ pending |
| 14-02-T3 | 14-02 | 2 | CAMP-04 | T-14-01 | Command reports created/updated/skipped; skip-and-log on natural-key failure; non-key failures null just that field; idempotent re-run | integration | `./manage.py test solsys_code.tests.test_import_campaign_csv` | ❌ W0 | ⬜ pending |
| 14-03-T2 | 14-03 | 3 | CAMP-05 | T-14-02 | Demo notebook executes end-to-end with no live network call, against the synthetic fixture only | manual/notebook-execution | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are `{phase}-{plan}-T{n}` (the task that owns the requirement's automated verify). Wave 0
test scaffolds are created within the owning plan's task, not a separate wave — this phase's tests
are part of each plan's deliverable (model tests in 14-01, import tests in 14-02, notebook in 14-03).*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_models.py` — covers CAMP-01/CAMP-02/CAMP-03 model-level behavior (field presence, status defaults, optional `target`)
- [ ] `solsys_code/tests/test_import_campaign_csv.py` — covers CAMP-04 (mirrors `test_load_telescope_runs.py`'s shape: temp-file CSV fixtures, `call_command`, stdout/stderr assertions)
- [ ] `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` — the D-10/D-11 synthetic fixture (also doubles as a manually-inspectable input for hand-verifying the notebook's demonstrated behavior)
- [ ] `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — covers CAMP-05
- [ ] Migration `solsys_code/migrations/0002_campaignrun.py` (generated via `./manage.py makemigrations solsys_code`, not hand-written — see `0001_calendareventtelescopelabel.py` for the expected auto-generated shape)
- Framework install: none — `./manage.py test` is already fully configured, no new packages needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Demo notebook output is human-legible and demonstrates the full status lifecycle (including `pending_review`, per CONTEXT.md D-03) | CAMP-05 | `jupyter nbconvert --execute` proves the notebook runs without error, but confirming the *displayed* output actually demonstrates the intended demo narrative (not just "didn't crash") needs a human read-through | Run `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`, then open the executed notebook and confirm: (1) no real PII appears anywhere in output, (2) at least one row demonstrates each `run_status` bucket exercised by the fixture, (3) the created/updated/skipped summary is printed and matches the fixture's row count |
| Operator's live bootstrap import against the real 3I/ATLAS sheet (external to this phase's automated tests, since the real CSV is not committed to the repo per CONTEXT.md's "Specific Ideas" note) | CAMP-04 | The real sheet requires operator-provided input (Google Sheets export) not available in CI/test fixtures | Operator runs the import command against the exported real CSV with `--campaign "3I/ATLAS"` and reviews the created/updated/skipped summary and any `site_needs_review`-flagged rows |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
