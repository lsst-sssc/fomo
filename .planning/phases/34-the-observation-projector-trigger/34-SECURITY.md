---
phase: "34"
slug: "the-observation-projector-trigger"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
block_on: high
register_authored_at_plan_time: true
created: "2026-09-12"
---

# Phase 34 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| LCO/SOAR portal → `ObservationRecord` fields | `updatestatus` / `get_observation_status()` write portal-controlled status, block times and (34-05) ISO-string schedule values onto records the projector renders publicly | externally-sourced strings; public once projected |
| any record save → shared `CalendarEvent` rows | the `post_save` receiver fires for every save in the process and writes calendar state on the caller's behalf, inside the caller's transaction | shared calendar rows; facility-URL key namespace only |
| anonymous visitor → calendar month cell / event modal | every projector-written title, description and telescope string, plus the display-time series decoration, is public | public text; series decoration gated on authenticated viewer + run visibility |
| operator shell → the sweep command | `--proposal` / `--facility` reach an ORM filter | operator input |
| portal block fields → `ObservationRecord.parameters` | one-time observed-site lookup writes `site`/`telescope`/`enclosure` under fixed keys | externally-sourced strings, mapped through a fixed table before display |
| developer database / LCO portal → committed notebook output and `tmp/` evidence | pre-executed notebooks are committed with output; summary lines are quoted in SUMMARY files | must never carry `LCO_API_KEY` or a token-bearing URL |
| scratch copy → `src/fomo_db.sqlite3` and the SCHED-06 baseline JSON | 34-06/34-07 re-executions must write to a copy only; the developer DB holds UAT evidence | UAT evidence integrity |

---

## Threat Register

| Threat ID | Plan | Category | Component | Severity | Disposition | Mitigation / Evidence | Status |
|-----------|------|----------|-----------|----------|-------------|-----------------------|--------|
| T-34-01 | 01 | Denial of Service | `receiver_on_record_save()` | high | mitigate | `observation_projector.py` imports neither `resolve_placement_block` nor `make_request` (grep: 0); `test_observation_projector_signals.py` patches `make_request` and asserts no call during a save (3 refs); 342 tests green 2026-09-12 | closed |
| T-34-02 | 01 | Denial of Service | projector exception inside a caller's transaction | high | mitigate | 6 `except Exception` guards in `observation_projector.py`; 4 raising-projector tests prove `save()`/`.add()`/`.delete()` still succeed | closed |
| T-34-03 | 01 | Tampering | `project_record()` / `receiver_on_record_delete()` writes | high | mitigate | all writes via `insert_or_create_calendar_event` keyed on the facility URL (3 call sites); delete path compares `meta.event.url`; `TestNamespaceIsolation` in both projector and sweep test modules | closed |
| T-34-04 | 01 | Tampering | `write_event_meta()` on an attributed companion row | medium | mitigate | `defaults` names exactly `is_verified`, `observation_record`, `observation_group`; `confirmed_by`/`confirmed_at`/`campaign_run` referenced 0 times in the module | closed |
| T-34-05 | 01 | Information Disclosure | `logger.warning` on an unprojectable record | medium | mitigate | warnings interpolate `type(exc).__name__` only (9 sites); no `.text`/`.content` of any response reaches a log line (0) | closed |
| T-34-06 | 01 | Tampering | externally-sourced `parameters` rendered into a title/description | low | accept | Django auto-escaping, no `|safe`; title capped at 200 chars — accepted per ASVS L1 at plan time | closed |
| T-34-SC | 01 | Tampering | npm/pip/cargo installs | low | accept | no install step; `pyproject.toml` unchanged across the phase's commits | closed |
| T-34-07 | 02 | Information Disclosure | sweep site-lookup failure path | high | mitigate | `resolve_placement_block()` converts every failure to `None` (9 `return None` sites in `calendar_utils.py`); sweep has 4 `except` clauses, none binds/interpolates the exception (`except ... as` count 0) | closed |
| T-34-08 | 02 | Denial of Service | repeated portal lookups on every sweep | high | mitigate | lookup gated on a missing `observed_site` (guards in sweep/projector/utils); 4 second-sweep-zero-lookup tests; UAT dry-run on the real corpus: `site_lookups: 0` | closed |
| T-34-09 | 02 | Tampering | portal-supplied `site`/`telescope`/`enclosure` written into `parameters` | medium | mitigate | sweep saves with `update_fields=['parameters']` (2 sites); title token only via `derive_telescope()`'s fixed in-repo map | closed |
| T-34-10 | 02 | Input Validation | `--proposal` / `--facility` arguments | medium | mitigate | `--facility` uses argparse `choices=`; `--proposal` feeds an exact `parameters__proposal__in` filter; no `raw()`/`cursor()` in the command | closed |
| T-34-11 | 02 | Tampering | the sweep writing outside its namespace | high | mitigate | all writes route through `project_record()`; `TestNamespaceIsolation` in `test_project_observation_calendar.py`; notebook cell `05528b38` asserts RUN:/GEM:/blank-url byte-identical (72/0/10) | closed |
| T-34-12 | 02 | Repudiation | retiring a command without migrating its guarantees | medium | mitigate | 34-02 SUMMARY carries the per-test classification table (already-covered / migrated / retired rows) for all 38 retired tests; `sync_lco_observation_calendar` referenced 0 times in `solsys_code/`, `src/`, docs, CLAUDE.md | closed |
| T-34-SC | 02 | Tampering | npm/pip/cargo installs | low | accept | no install step | closed |
| T-34-13 | 03 | Information Disclosure | `observation_series_decoration()` return dict | medium | mitigate | fixed six-key return dict; the only `submitter`/`contact`/`provenance` mentions in the module are docstring text (lines 515, 624, 658), no field reads; additionally gated on viewer authentication and `run.is_publicly_visible` | closed |
| T-34-14 | 03 | Denial of Service | `reverse()` inside a public template tag | high | mitigate | `reverse()` called only on `tom_observations:group-list` (no args) and `tom_observations:detail` with a non-null int pk; isinstance / `ObjectDoesNotExist` / missing-link / <2 members / index-None guards all return `None` | closed |
| T-34-15 | 03 | Denial of Service | per-event query fan-out in the month view | medium | mitigate | `fomo_render_calendar` Prefetch widened; `test_calendar_template.py` carries 10 query-count guards incl. `test_month_view_query_count_does_not_grow_with_a_second_grouped_event` | closed |
| T-34-16 | 03 | Tampering | a display-time tag writing to the event | high | mitigate | 0 write-method names in the tag body; `test_render_then_reproject_leaves_title_and_description_byte_identical` green | closed |
| T-34-17 | 03 | Tampering | externally-influenced title text reaching a style attribute | low | accept | `status_border_css()` returns fixed CSS constants by prefix match; accepted per ASVS L1 at plan time | closed |
| T-34-18 | 03 | Repudiation | losing an existing status ring while adding new markers | medium | mitigate | pre-existing `test_calendar_display_extras.py` assertions unmodified; `TestProjectorMarkerRings` covers the full ring vector; suite green | closed |
| T-34-SC | 03 | Tampering | npm/pip/cargo installs | low | accept | no install step | closed |
| T-34-19 | 04 | Information Disclosure | committed notebook output cells | high | mitigate | credential-shaped text (`api_key`/`authorization:`/`bearer`/`token=`) in `project_observation_calendar_demo.ipynb`: 0; manual read-through recorded at the 34-04, 34-06 and 34-07 commits | closed |
| T-34-20 | 04 | Repudiation | a runbook still naming a deleted command | medium | mitigate | `sync_lco_observation_calendar` in the runbook: 0; `project_observation_calendar`: 8 mentions | closed |
| T-34-21 | 04 | Tampering | the takeover sweep run against the developer database | medium | accept | deliberate one-time takeover (D-19); every field re-derivable by another sweep; accepted at plan time | closed |
| T-34-22 | 04 | Repudiation | a SCHED-06 claim with no recorded baseline | medium | mitigate | `project_observation_calendar_demo.sched06-baseline.json` present and git-clean; `34-UAT.md` SCHED-06 section with dated re-check row (closed 2026-09-12 on receiver-only evidence) | closed |
| T-34-23 | 04 | Information Disclosure | the Gemini notebook's existing committed output | low | mitigate | `sync_gemini_observation_calendar_demo.ipynb` still carries 12 executed output cells; not re-executed | closed |
| T-34-SC | 04 | Tampering | npm/pip/cargo installs | low | accept | no install step | closed |
| T-34-05-01 | 05 | Tampering | `coerce_schedule_datetime()` | medium | mitigate | strict `parse_datetime`; raises `ValueError` on unparseable, bare-date (WR-03) and wrong-type values; no `dateutil`/`dateparser`; `TestCoerceScheduleDatetime` (8 tests) green | closed |
| T-34-05-02 | 05 | Information Disclosure | `project_record()` warning log | medium | mitigate | log line unchanged (`type(exc).__name__`); the three `ValueError` messages interpolate only `{value!r}` | closed |
| T-34-05-03 | 05 | Denial of Service | `post_save` receiver | low | accept | unparseable schedule → `unprojectable`, save still succeeds (TRIG-02); accepted at plan time | closed |
| T-34-05-SC | 05 | Tampering | npm/pip/cargo installs | low | accept | no install step; `django.utils.dateparse` already a dependency | closed |
| T-34-06-01 | 06 | Information Disclosure | `updatestatus` / sweep output, notebook output, SUMMARY | high | mitigate | only summary lines captured to gitignored `tmp/34-06-*.txt` (2 files present) and quoted in the SUMMARY; no key/token text in the notebook (scan: 0) | closed |
| T-34-06-02 | 06 | Tampering | `src/fomo_db.sqlite3` | high | mitigate | every command routed with absolute `FOMO_DATABASE_PATH`; notebook guard cells reference the override (11 mentions); developer DB mtime/size verified unchanged in 34-06 and 34-07; UAT 2026-09-12 confirmed the sweep has still never run against it | closed |
| T-34-06-03 | 06 | Tampering | `project_observation_calendar_demo.sched06-baseline.json` | high | mitigate | baseline write guarded on `scratch_routed` (4 refs in the notebook); `git status --porcelain` on the path is empty | closed |
| T-34-06-04 | 06 | Repudiation | SCHED-06 evidence chain | medium | mitigate | notebook prose names the database used; SUMMARY names the copy path and evidence files; `34-UAT.md` re-check row records which writer narrowed which events | closed |
| T-34-06-SC | 06 | Tampering | npm/pip/cargo installs | low | accept | no install step | closed |
| T-34-07-01 | 07 | Tampering | `src/fomo_db.sqlite3` (SCHED-06 evidence) | high | mitigate | `tmp/34-07-devdb-stamp.txt` present and re-compared as an automated verify; cell `7022f987` guard; UAT 2026-09-12 found the 33 legacy events still in place (14 terminal ones remain, F-34-1) | closed |
| T-34-07-02 | 07 | Tampering | `sched06-baseline.json` | high | mitigate | cell `250b5d0b` scratch-routed guard reads only; path git-clean; `test_projector_demo_notebook.py` asserts the notebook quotes the file's `captured_at` | closed |
| T-34-07-03 | 07 | Repudiation | the committed takeover evidence | high | mitigate | `tmp/34-07-preflight-dryrun.txt` present; cells `05528b38`/`556d2a9f` assert non-vacuous takeover; `--allow-errors` absent from the plan/summary; `test_projector_demo_notebook` (6 tests) green and mutation-tested against emptied evidence | closed |
| T-34-07-04 | 07 | Information Disclosure | notebook output, `tmp/` evidence files, SUMMARY | high | mitigate | credential-shaped text scan of the committed notebook: 0 | closed |
| T-34-07-05 | 07 | Denial of Service | LCO observation portal | low | accept | bounded one-time observed-site lookups per clone; accepted at plan time | closed |
| T-34-07-SC | 07 | Tampering | npm/pip/cargo installs | low | accept | no install step | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

Register: 42 entries (30 mitigate, 12 accept), all authored in the seven plans' `<threat_model>` blocks; no SUMMARY declared additional Threat Flags. Classification depth: ASVS L1 (grep-level evidence plus the phase's green test suite — 342 tests across the nine affected modules, 2026-09-12). `T-34-SC` is the shared per-plan supply-chain entry and is listed once per plan.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-34-01 | T-34-06 (plan 01) | Django auto-escaping, no `|safe`; title capped at 200 chars — accepted per ASVS L1 at plan time | plan-time threat model (34-01-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-02 | T-34-SC (plan 01) | no install step; `pyproject.toml` unchanged across the phase's commits | plan-time threat model (34-01-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-03 | T-34-SC (plan 02) | no install step | plan-time threat model (34-02-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-04 | T-34-17 (plan 03) | `status_border_css()` returns fixed CSS constants by prefix match; accepted per ASVS L1 at plan time | plan-time threat model (34-03-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-05 | T-34-SC (plan 03) | no install step | plan-time threat model (34-03-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-06 | T-34-21 (plan 04) | deliberate one-time takeover (D-19); every field re-derivable by another sweep; accepted at plan time | plan-time threat model (34-04-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-07 | T-34-SC (plan 04) | no install step | plan-time threat model (34-04-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-08 | T-34-05-03 (plan 05) | unparseable schedule → `unprojectable`, save still succeeds (TRIG-02); accepted at plan time | plan-time threat model (34-05-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-09 | T-34-05-SC (plan 05) | no install step; `django.utils.dateparse` already a dependency | plan-time threat model (34-05-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-10 | T-34-06-SC (plan 06) | no install step | plan-time threat model (34-06-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-11 | T-34-07-05 (plan 07) | bounded one-time observed-site lookups per clone; accepted at plan time | plan-time threat model (34-07-PLAN.md), confirmed at audit | 2026-09-12 |
| AR-34-12 | T-34-07-SC (plan 07) | no install step | plan-time threat model (34-07-PLAN.md), confirmed at audit | 2026-09-12 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-12 | 42 | 42 | 0 | /gsd-secure-phase 34 (orchestrator, L1 short-circuit — register authored at plan time, threats_open 0) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-12
