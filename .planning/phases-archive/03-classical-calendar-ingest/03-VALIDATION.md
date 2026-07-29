---
phase: 03
slug: classical-calendar-ingest
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-14
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` (via `./manage.py test`) |
| **Config file** | none — Django test runner configured via `src.fomo.settings` (`DJANGO_SETTINGS_MODULE`) |
| **Quick run command** | `./manage.py test solsys_code.tests.test_load_telescope_runs` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_load_telescope_runs`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green (`./manage.py test solsys_code`); also `ruff check .` and `ruff format --check .` per CLAUDE.md
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | INGEST-01/02/03 | — | N/A | integration (RED) | `./manage.py test solsys_code.tests.test_load_telescope_runs` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | INGEST-01/02/03 | T-03-01 | Path traversal / large-file DoS — accepted (operator-trust CLI, consistent with `fetch_jplsbdb_objects`) | integration (GREEN) | `./manage.py test solsys_code.tests.test_load_telescope_runs -v 2` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 2 | INGEST-01/02/03 | — | N/A | manual/notebook | `jupyter nbconvert --to notebook --execute docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_load_telescope_runs.py` — new file, covers INGEST-01/02/03 and D-02/D-04:
  - `test_creates_one_event_per_night` (INGEST-01)
  - `test_event_durations_within_range` (INGEST-01)
  - `test_event_fields_set_from_parsed_run` (INGEST-02)
  - `test_idempotent_rerun_no_duplicates` (INGEST-03)
  - `test_unparseable_line_logged_and_skipped` (D-02)
  - `test_unchanged_rerun_does_not_update_existing_rows` (D-04)
  - Needs `setUpTestData` seeding the 3 in-scope `Observatory` records (Magellan-Clay/Magellan-Baade, NTT, FTS), mirroring `test_telescope_runs.py`'s fixture
- [ ] `solsys_code/management/commands/load_telescope_runs.py` — new command, no existing scaffold
- [ ] A small fixture schedule file (inline string or `solsys_code/tests/fixtures/*.txt`) containing the documented sample lines for `call_command()`-based integration tests

---

## Manual-Only Verifications

*All phase behaviors have automated verification (Django TestCase suite). The demo notebook (03-02) is a Definition-of-Done artifact, not a substitute for automated tests.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-14
