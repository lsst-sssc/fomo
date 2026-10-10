---
phase: "40"
slug: "notebook-isolation-attribution-page"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 1
asvs_level: 1
created: "2026-10-09"
---

# Phase 40 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time (40-01..40-09 `<threat_model>` blocks); verified at ASVS L1 (grep-depth evidence), short-circuit per secure-phase §3 (`threats_open: 0`, plan-time register, `asvs_level: 1`).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| notebook kernel -> developer database | A notebook run on the developer machine can reach `src/fomo_db.sqlite3`, which the live cron runner also writes | database writes (high) |
| settings resolution -> notebook writes | `local_settings.py` (imported last by settings.py) can override `DATABASES`, so the variable alone does not decide where Django writes | database routing (high) |
| /tmp scratch directory | A snapshot holds a full copy of the developer database (password hashes, tokens, sessions) for the run's duration | credentials at rest (medium) |
| committed notebook output | Pre-executed notebooks are committed with output | snapshot text (medium) |
| staff browser -> AttributionDecisionView | POSTed hidden fields (band, page, sort) are client-controlled and can be edited | redirect query values (low) |
| GET query -> attribution page forms | Query values are echoed into hidden inputs on the next render | reflected values (low) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-40-01 | Tampering | `setup_scratch_django` routing | high | mitigate | `check_resolved_database` raises `RuntimeError` before `migrate` (`docs/notebooks/fomo_nb_utils.py:105,121`); override subprocess test in `test_notebook_helper.py` | closed |
| T-40-02 | Tampering | snapshot copy of a database another process writes | high | mitigate | `?mode=ro` URI + `Connection.backup()` (`fomo_nb_utils.py:83-84`); test pins `mode=ro` (`test_notebook_helper.py:158`) and the import allow-list | closed |
| T-40-03 | Information disclosure | leftover `/tmp/fomo-notebook-db-*` snapshot | medium | mitigate | `tempfile.mkdtemp(prefix=SCRATCH_DIR_PREFIX)` (0700) with cleanup on setup failure (`fomo_nb_utils.py:150-156`); teardown `shutil.rmtree(path)` without `ignore_errors` (`:238`) | closed |
| T-40-04 | Tampering | `teardown_scratch_django` removing the wrong path | medium | mitigate | `SCRATCH_DIR_PREFIX = 'fomo-notebook-db-'` check before removal; `test_teardown_refuses_a_directory_without_the_scratch_prefix` (`test_notebook_helper.py:213`) | closed |
| T-40-05 | Elevation of privilege | `python -O` stripping the database guard | high | mitigate | `raise RuntimeError`, no `ast.Assert` in the helper (static test `test_notebook_helper.py:105`); subprocess runs with `-O` / `PYTHONOPTIMIZE=1` | closed |
| T-40-06 | Tampering | guard bypass by aliasing, nesting or stale output | high | mitigate | AST rules over top-level statements, import-based Django detection and alias bans (`test_pre_executed_notebooks.py:183,202`), execution counts 1..N; per-bypass synthetic tests | closed |
| T-40-07 | Information disclosure | credential-shaped text in committed snapshot output | medium | mitigate | `test_no_credential_shaped_text_in_any_notebook` (`test_pre_executed_notebooks.py:762`) | closed |
| T-40-08 | Tampering | a write hidden inside an `assert` | medium | mitigate | Rule R11 (`test_pre_executed_notebooks.py:58,435`); audit 236 asserts, none wrapping a write (40-02-SUMMARY) | closed |
| T-40-09 | Tampering | template edits (XSS surface) | low | mitigate | Class names only; `test_dismissal_reason_xss_payload_is_escaped` (`test_campaign_attribution_views.py:832`) passes | closed |
| T-40-10 | Spoofing | Bootstrap CSS supplied by a CDN | low | accept | Unchanged from before the phase; see AR-40-01 | closed |
| T-40-11 | Tampering | `_redirect` open redirect / header injection | high | mitigate | Path from `reverse('campaigns:attribution')`; query only from `attribution_return_state()` (`campaign_views.py:1209`; band whitelist `:1227`, page `[1-9][0-9]{0,5}` `:1198`, sort `-?[a-z_]{1,40}` `:1199`, both `fullmatch`) via `urlencode` (`:1475`); fragment from constant map `_ATTRIBUTION_RETURN_FRAGMENTS` (`:1203,1476`); `test_tampered_return_state_is_dropped` (`test_campaign_attribution_views.py:1652`) | closed |
| T-40-12 | Tampering | reflected values in hidden inputs (XSS) | medium | mitigate | Only validated values emitted via `format_html_join` (`campaign_tables.py:596`) and template autoescape; `attribution_return_state.html` has no `safe` filter (0 occurrences) | closed |
| T-40-13 | Tampering | CSRF on the decide endpoint | medium | mitigate | `test_every_attribution_form_keeps_its_csrf_token` (`test_attribution_template.py:424`) | closed |
| T-40-14 | Elevation of privilege | staff-only attribution actions | high | mitigate | `StaffRequiredMixin` on `AttributionQueueView` / `AttributionDecisionView` (`campaign_views.py:1292,1418`); `TestAttributionStaffGating` (`test_campaign_attribution_views.py:148`) passes | closed |
| T-40-15 | Denial of service | oversized page numbers | low | mitigate | Page values limited to 1-6 ASCII digits with no leading zero before use (`_RETURN_PAGE_RE`, `campaign_views.py:1198`); django-tables2 clamps out-of-range pages | closed |
| T-40-16 | Tampering | `setup_scratch_django` refusal path (CR-01) | high | mitigate | Settings read lazily and checked before `django.setup()` (`fomo_nb_utils.py:295,300`); any failure runs `_disable_django_databases` (`:184`); child tests `test_a_refused_setup_leaves_django_unable_to_reach_any_database` / `test_a_failure_after_django_setup_leaves_no_usable_database` (`test_notebook_helper.py:470,492`) | closed |
| T-40-17 | Tampering | resolved-path check covering only `default` (WR-02) | medium | mitigate | `check_database_settings` refuses extra aliases and `DATABASE_ROUTERS` (`fomo_nb_utils.py:137,164`); `test_refuses_a_second_alias_even_when_it_names_the_scratch_file`, `test_refuses_database_routers`, child test `:534` (`test_notebook_helper.py:350,360,534`) | closed |
| T-40-18 | Tampering | `teardown_scratch_django` removing a wrong `fomo-notebook-db-*` path (IN-06) | medium | mitigate | Directory must sit directly in resolved `tempfile.gettempdir()` and be the one setup recorded (`fomo_nb_utils.py:322,342`); `test_teardown_refuses_a_scratch_named_directory_outside_the_temp_directory`, `test_teardown_refuses_a_scratch_directory_setup_did_not_create` (`test_notebook_helper.py:314,323`) | closed |
| T-40-19 | Information disclosure | checkout path in the `Django ready:` line (IN-02) | low | mitigate | Line prints only the settings module (`fomo_nb_utils.py:317`); `test_fresh_setup_routes_migrates_and_tears_down` asserts no path (`test_notebook_helper.py:398`); all eight notebooks re-executed (40-07) | closed |
| T-40-20 | Tampering | routing environment left after teardown or failure (IN-01) | low | mitigate | `_ROUTING_ENVIRON` restored on failure and teardown (`fomo_nb_utils.py:61,360`); `test_teardown_restores_the_routing_environment` (`test_notebook_helper.py:612`) | closed |
| T-40-21 | Denial of service | disable step run against a Django the helper did not start | medium | mitigate | `apps.ready or settings.configured` refusal before the `try` (`fomo_nb_utils.py:275`); `test_setup_refuses_when_django_is_already_set_up` (`test_notebook_helper.py:264`) | closed |
| T-40-22 | Tampering | guard miss: re-route after the helper call (WR-01) | high | mitigate | Rule R12 `_database_rerouting_problems` wired into `isolation_problems` (`test_pre_executed_notebooks.py:361,425`); real-notebook replays (`:841,847`), unparseable-cell case (`:942`), one synthetic test per form | closed |
| T-40-23 | Repudiation | guard docstring overstating coverage | low | mitigate | Module docstring rewritten to list what is and is not seen (40-06 Task 2, commit 26940ce) | closed |
| T-40-24 | Tampering | re-executing eight notebooks, two snapshots of the real database | high | mitigate | All runs through the fail-closed helper (`mode=ro` backup for snapshots); `test_every_real_pre_executed_notebook_is_isolated` (`test_pre_executed_notebooks.py:950`); `src/fomo_db.sqlite3` untouched (40-07-SUMMARY) | closed |
| T-40-25 | Information disclosure | committed snapshot output (paths, credential-shaped text) | medium | mitigate | `test_no_credential_shaped_text_in_any_notebook` (`test_pre_executed_notebooks.py:998`) passes; no helper cell prints a home path; one third-party spiceypy log line listed in 40-07-SUMMARY | closed |
| T-40-26 | Information disclosure | schedule temp files left in `/tmp` (IN-05) | low | mitigate | All eight `NamedTemporaryFile(..., dir=scratch_db_dir)` in `load_telescope_runs_demo.ipynb` (8 occurrences); removed by teardown (commit 29490c3) | closed |
| T-40-27 | Repudiation | ledger marking findings fixed without the work | low | mitigate | 40-REVIEW-DISPOSITION.md rows cite post-0ad6991 commits per finding; IN-07 `deferred` with its remaining part named (commit 49a0530) | closed |
| T-40-28 | Tampering | guard miss: ConnectionHandler store, child manage.py with a bare or cleared environment, shell escape (re-review WR-01) | high | mitigate | R12 bans whole capabilities (`PROCESS_MODULES` `test_pre_executed_notebooks.py:97`, `R12_ALL_NAMES` `:147`, `ATTACH_SQL` `:157`); R0 `UNPARSEABLE_SUSPECT_TEXT` (`:221`); `TestReReviewFormsOnARealNotebook` (`:1054`) and `TestCapabilityRulesOnSyntheticNotebooks` (`:1214`); commits 301bd93, f1b4bb6; **reopened 2026-10-10 by 40-REVIEW.md CR-01/CR-02**: readable plain-source process, `connection`, `ATTACH (%s)` and IPython alias/automagic routes still pass the guard and were reproduced writing another database ; **closed 2026-10-10 after 40-10**: every listed route (re-review forms, round-3 CR-01/CR-02 process, connection, ATTACH ( and alias forms) is reported on a real-notebook copy -- `TestReReviewFormsOnARealNotebook` (`:1432`), `TestThirdReviewFormsOnARealNotebook` (`:1487`); the new automagic-exemption route is tracked as T-40-38 | closed |
| T-40-29 | Tampering | `_disable_django_databases` close raising before the dummy switch (re-review WR-02) | medium | mitigate | Every alias switched first, `connections.close_all()` best-effort with `logger.warning` (`fomo_nb_utils.py:187,210,212`), original error re-raised (`:254,337`); `TestHelperFailClosedIsExceptionSafe` (`test_notebook_helper.py:615`, three child tests incl. no default alias); commit c45edbc | closed |
| T-40-30 | Repudiation | teardown line and guard docstring overstating protection | low | mitigate | Teardown line says what setup checked (`fomo_nb_utils.py:389`); guard docstring names the residual (`test_pre_executed_notebooks.py:45,48`) | closed |
| T-40-31 | Repudiation | README refusal wording (IN-01) | low | mitigate | README S1-S3 (`docs/notebooks/README.md:68`); same sentences in all eight setup cells (40-09) | closed |
| T-40-32 | Elevation of privilege | residual: deliberate runtime obfuscation or code outside the notebook | medium | accept | Named in the guard docstring (`test_pre_executed_notebooks.py:45,48`); see AR-40-03 and the 40-VERIFICATION.md `overrides:` entry (commit b6bfbca) | closed |
| T-40-33 | Tampering | re-executing eight notebooks, two snapshots of the real database (40-09) | high | mitigate | All runs through the 40-08 helper; new teardown line in all eight committed notebooks; `test_every_real_pre_executed_notebook_is_isolated` (`test_pre_executed_notebooks.py:1415`) passes; `src/fomo_db.sqlite3` untouched (commits 25eae59, c68bf7b, 0020360) | closed |
| T-40-34 | Information disclosure | committed snapshot output and log lines | medium | mitigate | `test_no_credential_shaped_text_in_any_notebook` (`test_pre_executed_notebooks.py:1463`) passes; home-path inventory in 40-09-SUMMARY.md lists only the third-party spiceypy line | closed |
| T-40-35 | Repudiation | notebook prose contradicting its output (IN-02) | low | mitigate | NF-14 labels `plus any unexplained snapshot event` (4 occurrences in `reconcile_campaign_runs_demo.ipynb`); projector table cites `The whole real corpus, not a sample` (commit 0020360) | closed |
| T-40-36 | Repudiation | ledger / override recording a fix or acceptance without the work | low | mitigate | 40-REVIEW-DISPOSITION.md IN-01/IN-02 rows cite 40-08/40-09 commits, `open: 3`, `total: 11` (commit 93bb514); override names the developer and timestamp (40-VERIFICATION.md `accepted_by`, commit b6bfbca) | closed |
| T-40-37 | Tampering | _database_rerouting_problems (CR-01 a, e): a process module imported from a re-exporting module, or a process looked up by a literal string, starts a child Django on the developer database | high | mitigate | R12 matches every listed name at every identifier and in string parts (482606b, 3232341); `test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook` (`test_pre_executed_notebooks.py:1508`), `test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook` (`:1517`) | closed |
| T-40-38 | Tampering | isolation_problems R0 (CR-02 and the planning-time automagic finding): an alias, automagic word, escape after a line magic or prompt prefix makes IPython run a shell command that copies over or deletes the developer database | high | mitigate | R0 allow-list over TransformerManager + automagic rule (c6c6322); `TestR0AllowList` (`test_pre_executed_notebooks.py:2083`), `test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook` (`:1606`), `test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook` (`:1620`); **open 2026-10-10 by round-4 40-REVIEW.md CR-01**: the automagic exemption accepts a source-level store that binds nothing at runtime (`name: int`, a store under `if False:`, `for name in []:`, `except ... as name`), so a later one-line `cp`/`rm`/`mv` alias cell is not reported -- a literal route outside the stated residual | open |
| T-40-39 | Tampering | R12 (CR-01 b-d): the connection proxy, Database.connect or ATTACH ( re-points Django's connection at another file | medium | mitigate | `connection`, `_connections`, `Database` listed and `ATTACH_SQL` widened (3232341); `test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem` (`test_pre_executed_notebooks.py:2013`), `test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook` (`:1551`) | closed |
| T-40-40 | Tampering | _byte_copy_problems (WR-01): copytree, copyfileobj, move or write_bytes byte-copies a scratch file over the developer database by its shared base name | medium | mitigate | R6 adds copytree, copyfileobj, move, write_bytes (3232341); `test_every_copy_and_write_function_in_any_spelling_is_a_problem` (`test_pre_executed_notebooks.py:2063`), `test_copytree_and_write_bytes_are_reported_on_a_real_notebook` (`:1576`); round-4 40-REVIEW.md WR-01: R6 follows `shutil` only through `import shutil as X` (a plain `sh = shutil` rebinding or `from <module> import shutil` hides the call; the destination still needs a runtime-joined path) | open — below high threshold |
| T-40-41 | Elevation of privilege | residual: values assembled or looked up at runtime, code decoded at runtime, code outside the notebook (imported modules, IPython profile and start-up files, unlisted library functions that start a process or write a file), JavaScript in outputs | medium | accept | Named in the guard docstring's "What the guard does not see" paragraph (7af64d0) and in the developer-approved 40-VERIFICATION.md override reason (44c33c6, accepted 2026-10-10T19:14:57Z); see AR-40-03 | closed |
| T-40-42 | Repudiation | guard module docstring and README guard paragraph (gap 2, CR-01): they claim the guard reports more than it does, so the override is accepted on a false basis | medium | mitigate | Docstring "What the guard checks" rewritten from the 40-10 code and RES verbatim; README_GUARD/README_IN01 (7af64d0); round-4 40-REVIEW.md CR-01: the "unless an earlier cell binds that name" claim (docstring and README) overstates the automagic exemption | open — below high threshold |
| T-40-43 | Repudiation | notebook teardown and setup prose (WR-02): "never opened" claims sit above output that says something different | low | mitigate | Six markdown sentences replaced by cell id with nbformat, code cells and outputs identical (17168ff); round-4 review confirms WR-02 closed and no "never opened" text remains | closed |
| T-40-44 | Tampering | _disable_django_databases docstring (IN-01): it credits the in-place switch with protection only the wrapper drop gives, inviting a later change that drops the drop step | low | mitigate | `_disable_django_databases` docstring says what the in-place switch does on its own (7af64d0); the logged warning text is tracked as round-4 40-REVIEW.md IN-01 (info) | closed |
| T-40-45 | Repudiation | 40-VERIFICATION.md override: a reason recorded without the developer's approval, or one that claims more than the code keeps, masks a guard gap | medium | mitigate | Draft check (every claim backed by a named test or docstring phrase), blocking-human checkpoint, approved reason recorded verbatim with `accepted_at` 2026-10-10T19:14:57Z (44c33c6); round-4 40-REVIEW.md CR-01: the reason's "unless an earlier cell binds that name" clause overstates the code -- the verifier decides whether the override applies | open — below high threshold |
| T-40-SC | Tampering | npm/pip/cargo installs | low | accept | No package installed by any phase-40 plan; see AR-40-02 | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-40-01 | T-40-10 | Bootstrap CSS comes from the CDN via `tom_common/base.html` / django_bootstrap5 exactly as before; 40-03 adds no external resource | plan-time threat model (40-03) | 2026-10-09 |
| AR-40-02 | T-40-SC | 40-01 to 40-12 install no package (RESEARCH Package Legitimacy Audit: none; gap plans 40-05..40-09 confirm) | plan-time threat model (40-01..40-09) | 2026-10-09 |
| AR-40-03 | T-40-32, T-40-41 | A static check of notebook source cannot enumerate values assembled or looked up at runtime, or code outside the notebook; the residual is named in the guard docstring and accepted as the WARN-02 override in 40-VERIFICATION.md | Tim Lister | 2026-10-10 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-09 | 16 | 16 | 0 | secure-phase State B, orchestrator (L1 grep evidence; short-circuit) |
| 2026-10-10 | 28 | 28 | 0 | secure-phase State A, orchestrator (gap plans 40-05..40-07: T-40-16..27 added, L1 grep evidence; short-circuit) |
| 2026-10-10 | 37 | 36 | 1 | secure-phase State A, orchestrator (gap plans 40-08..40-09: T-40-28..36 added, L1 grep evidence, T-40-32 accepted as AR-40-03; T-40-28 left open on the 40-REVIEW.md CR-01/CR-02 reproductions) |
| 2026-10-10 | 46 | 42 | 4 | secure-phase State A, orchestrator (gap plans 40-10..40-12: T-40-37..45 added, L1 grep evidence; T-40-28 closed; T-40-41 accepted under AR-40-03; T-40-38 left open (high) on round-4 40-REVIEW.md CR-01; T-40-40/42/45 open below the high threshold) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [ ] `threats_open: 0` confirmed (1 open after the 40-10..40-12 re-audit: T-40-38; T-40-40, T-40-42 and T-40-45 open below the high threshold)
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-09; re-verified 2026-10-10 after gap plans 40-05..40-07; re-audited 2026-10-10 after gap plans 40-08..40-09 -- 1 threat open (T-40-28, high); re-audited 2026-10-10 after gap plans 40-10..40-12 -- 1 threat open (T-40-38, high; round-4 review CR-01), T-40-28 closed
