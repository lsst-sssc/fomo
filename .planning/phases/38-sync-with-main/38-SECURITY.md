---
phase: "38"
slug: "sync-with-main"
status: secured
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-07"
---

# Phase 38 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

Register authored at plan time (every PLAN carries a `<threat_model>` block). ASVS level 1, block threshold `high`.
Verified 2026-10-07 by the execute-phase `verify:post` hook at L1 evidence depth (the short-circuit rule applies:
`threats_open: 0`, plan-time register, ASVS 1), against the tree at HEAD, the remote branches and PR #43.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| origin/main → branch | 62 main commits enter through one merge commit; nine conflicted files hand-resolved | source, config, templates |
| PyPI → dev venv / fresh venv | tomtoolkit 3.1.0, tom_jpl 0.3.0, django-allauth, coverage installed into the venv the developer's cron uses, and into a throwaway venv | third-party code |
| working tree → generated docs | `docs/conf.py` decides which modules autoapi renders, including any `local_settings.py` with credentials | credentials (if present) |
| GSD tooling → git index | an in-progress merge is exposed to any commit run while the checkpoint waits | staged tree |
| repo → GitHub Actions | workflow files decide what CI runs, with which actions and secrets | CI secrets |
| pre-commit hooks → working tree | hooks rewrite files (ruff, notebook metadata) and gate commits | source, notebooks |
| executor → developer database | `migrate` writes the database the developer's cron and dev server use | operational data |
| downloaded wheels → executor | third-party archives unpacked for comparison | third-party code |
| local repo → GitHub (public) | two branch pushes and a PR body edit publish content | source, docs |
| executor → PR state | `gh` could change draft state or merge | PR #43 |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-38-01 | Information disclosure | docs/conf.py `autoapi_ignore` | high | mitigate | `autoapi_ignore = ['*/__main__.py', '*/local_settings.py']` present (fixed-string grep: 1 match); only `*/_version.py` dropped | closed |
| T-38-02 | Elevation of privilege | src/fomo/urls.py user-delete route | medium | mitigate | `ProtectedUserDeleteView` route `name='user-delete'` present and precedes `include('tom_common.urls')`; `TestUserDeleteView` ran before the merge commit (38-01 Task 3) | closed |
| T-38-03 | Tampering | the merge commit's content | medium | mitigate | `e12158c` equals `git merge-tree` outside the nine paths (re-checked: ok); touches no `.planning/` path; developer reviewed the full staged diff at the blocking-human checkpoint (`approve`); ruff hooks skipped so no rewrite landed | closed |
| T-38-04 | Spoofing | user sign-up and login (tomtoolkit 3.1.0 allauth, `TOM_REGISTRATION_STRATEGY = 'open'`) | medium | accept | Same open sign-up policy the branch had through tom_registration, now tomtoolkit-native; flagged at the 38-01 checkpoint; no `'tom_registration'` reference remains in settings.py — see Accepted Risks Log | closed (accepted) |
| T-38-05 | Denial of service | solsys_code/admin.py Target registration | low | mitigate | exactly one `admin.site.unregister(Target)` and one `admin.site.register(Target, SolsysTargetAdmin)`; no module-level `TargetAdmin` class; test_admin/test_search ran before the commit | closed |
| T-38-06 | Tampering | PyPI packages installed in 38-01 Task 3 | medium | mitigate | blocking-human legitimacy confirmation (`approve`) before any install; only origin/main's declared floors via `.[dev]`; `pip check`: "No broken requirements found."; versions recorded in 38-01-SUMMARY.md | closed |
| T-38-07 | Denial of service | the developer's run_unattended cron during the merge window | medium | mitigate | one crontab line paused (`#PHASE38-PAUSED `) after saving the original, flock waited; restored in 38-03 after the migrate — `crontab -l` is byte-identical to `$HOME/tmp/phase38-crontab.bak`, 0 paused lines remain | closed |
| T-38-08 | Tampering | .github/workflows/*.yml | low | mitigate | `git diff --numstat origin/main -- .github/workflows` = `1 1 smoke-test.yml; 1 1 testing-and-coverage.yml` (the two exclusion tokens only); `check-github-workflows` passed | closed |
| T-38-09 | Tampering | ruff fixes applied to production modules | medium | mitigate | reformat commit `ff8dd3c` proven AST-identical per file; SIM103 rewrite present with the same truth table; `[tool.ruff.lint]` unchanged since the merge commit | closed |
| T-38-10 | Repudiation | test exclusions hiding failures from CI and the hook | medium | mitigate | `@tag('ephemeris_segfault')` on exactly one class under solsys_code/; `functional` tests still run in CI's functional-tests job (`python manage.py test --tag functional` present) | closed |
| T-38-11 | Tampering | committed output of the eight pre-executed notebooks | medium | mitigate | `exclude: ^docs/notebooks/pre_executed` kept on jupyter-nb-clear-output; metadata commit `1ac9a50` verified cell-for-cell identical; reformat kept outputs identical | closed |
| T-38-12 | Tampering | fresh-venv install from PyPI | medium | mitigate | only `.[dev]` of this tree installed; fresh venv `pip check`: "No broken requirements found."; resolved versions recorded in 38-03-SUMMARY.md | closed |
| T-38-13 | Tampering | src/fomo_db.sqlite3 migration | medium | mitigate | cron paused at the time; consistent copy via SQLite backup API `src/fomo_db_20261007_pre_phase38.sqlite3`, `PRAGMA integrity_check` ok, git-ignored; `migrate --check` clean afterwards | closed |
| T-38-14 | Repudiation | the suite's pass signal | medium | mitigate | both logs hold an exact `OK` line (2178 / 2170 tests), no `FAILED`, no `skipped=`; diff check rejects any added skip/tag except main's `@tag('functional')` | closed |
| T-38-15 | Tampering | downloaded tomtoolkit wheels | low | mitigate | wheels downloaded `--no-deps --only-binary` into new directories and unpacked with `python -I -m zipfile` from the repo root; diffed only, never imported or installed | closed |
| T-38-16 | Information disclosure | live portals and notifications | low | mitigate | no portal-calling command run against the developer database; cron restored only after both suite runs were green | closed |
| T-38-17 | Information disclosure | issue37-code-only snapshot | high | mitigate | `.planning/` removed from index and disk before the commit; leak grep over `git ls-tree -r origin/issue37-code-only` for `.planning/`, `*.sqlite3`, `local_settings.py`, `reqgroup_*.json`: 0 matches; developer saw the result at the checkpoint (`publish`) | closed |
| T-38-18 | Tampering | published branch history | high | mitigate | plain pushes only; `5a1f27e` is an ancestor of `origin/issue37-code-only` and `75ad2be` of `origin/issue37-telescope-runs-calendar` (re-checked: yes) | closed |
| T-38-19 | Elevation of privilege | PR #43 draft state | high | mitigate | only `gh pr edit 43 --body-file` was run; `gh pr view 43 --json isDraft` → true (re-checked) | closed |
| T-38-20 | Tampering | branch-implicit commands after a push | medium | mitigate | every block started with `git branch --show-current` / `git -C ../fomo_code_only branch --show-current`; primary checkout never switched (on issue37-telescope-runs-calendar now); `../fomo_code_only` removed | closed |
| T-38-21 | Information disclosure | PR body text | low | mitigate | body check rejects `/home/` paths and api_key text (live PR body: 0 / 0); developer read the full body at the checkpoint | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-38-01 | T-38-04 | Open self-registration is the policy the branch already had through `tom_registration`; tomtoolkit 3.1.0 now owns it (`TOM_REGISTRATION_STRATEGY = 'open'` arrived from main by automatic merge). FOMO does not re-add `ModelBackend` or `tom_registration`. Shown to the developer at the 38-01 Task 2 checkpoint, answered `approve`. | developer (38-01 checkpoint) | 2026-10-07 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-07 | 21 | 21 | 0 | execute-phase verify:post hook (L1 evidence check, no auditor — short-circuit rule) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
