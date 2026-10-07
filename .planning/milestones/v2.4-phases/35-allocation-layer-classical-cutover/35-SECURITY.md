---
phase: "35"
slug: "allocation-layer-classical-cutover"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-09-15"
---

# Phase 35 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| staff action -> `CampaignRunObservation` row | a staff confirmation in the attribution queue crosses into automated calendar mutation | attribution decision |
| `CampaignRun` state -> `CalendarEvent` rows | run field values (window, status, site) become publicly visible calendar entries | schedule/site metadata |
| test suite -> merge gate | the suite is the only automated statement of what the dispatch and key families mean | test coverage |
| staff admin edit -> `CampaignRun` sub-night fields | a hand-typed time crosses into an automatically re-minted set of public calendar entries | time-of-day fields |
| stored `TimeField` -> `CalendarEvent.start_time`/`end_time` | a time-of-day with no date becomes an absolute instant on a shared calendar | derived datetime |
| staff HTTP request -> `CampaignRunObservation` write | an authenticated staff action crosses into automatic calendar mutation inside the same transaction | attribution link |
| facility portal -> `ObservationRecord.scheduled_start`/`scheduled_end` | externally-supplied schedule values reach the receiver on the in-memory instance | schedule datetimes |
| operator-supplied schedule file -> `parse_run_line()` | free text from outside the application becomes a persisted identity key and a public calendar entry | schedule text |
| `source_identifier` -> `CampaignRun` find-or-create | a computed key decides whether a write updates an existing row or creates a new one | identity key |
| pre-existing calendar rows -> the cutover command | rows written by earlier code versions, of unknown provenance, become the input to an automated mutation | legacy calendar rows |
| recovered `Source line:` text -> `parse_run_line()` | a string stored in a description field is re-parsed as if it were a fresh schedule line | recovered text |
| operator invocation -> irreversible re-key | one command run changes the key of real, publicly-visible calendar rows | calendar keys |
| notebook execution -> the developer database | a re-executed notebook runs real commands that write real rows unless routed at a copy | database writes |
| published runbook -> operator action | a wrong or stale instruction causes an operator to run an irreversible command in the wrong order | operational procedure |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-35-01 | Tampering | `telescope_runs._resolve_proposal()` / `parse_run_line()` | medium | mitigate | `[proposal]` bracket grammar (`_PROPOSAL_TOKEN`, `telescope_runs.py:113`) consumed first, disjoint from every other token grammar; multiple/empty/unbalanced forms raise `ValueError` (`:423`, `:428`, `:434-435`). | closed |
| T-35-02 | Tampering/Repudiation | `_source_identifier()` find-or-create (ingest + cutover) | high | mitigate | App layer: within-file collision reported on stderr with both line numbers, counted under `skipped_collision`, never merged (`load_telescope_runs.py:225,249-254,349`). DB layer: partial `UniqueConstraint('source_identifier', ...)` (`models.py:414-418`, migration 0015). Cutover computes the identical key via the same `_source_identifier()` (`cutover_classical_allocations.py:116,372`), pinned by `test_cutover_classical_allocations.py:139-150`. | closed |
| T-35-03 | Tampering | `allocation_projector.project_allocation()` attribution bridge + `_stale_attributions()` | high | mitigate | Every attribution write routes through `campaign_utils.adopt_event_into_run()` (`allocation_projector.py:468`), which refuses when the companion row already points at a different run (`campaign_utils.py:1001-1002`), logged and counted (`:469-476`). Confirmed-attribution coverage moved (not deleted) to a container-run fixture (`test_campaign_reconciler.py:621-666`). | closed |
| T-35-04 | Denial of Service | `receiver_on_run_observation_save`/`_delete` and the record-side linked-run re-project step | medium | mitigate | Every new receiver body wrapped in `try`/`except Exception`; the record-side step has its own guard. The `campaign_run_links` lookup itself is now resolved into a plain list inside its own `try` (`observation_projector.py:647-658`, fixed 2026-09-15 commit `24875bf` closing the F-34-1 gap). Never-raise pinned for `ValueError` and bare `Exception`, save and delete (`test_allocation_projector_signals.py:277-287`). Advisory: the two `CampaignRun.objects.filter(pk=...).first()` lookups in the `CampaignRunObservation` receivers (`allocation_projector.py:829-831`, `:910-912`) sit outside their `try` — same structural class as the closed F-34-1 gap, narrow (implies the caller's transaction is already failing), not blocking; candidate for a future symmetry pass. | closed |
| T-35-05 | Tampering | `cutover_classical_allocations` unexplained-event handling + runbook instructions | high | mitigate | Zero delete call sites in the command (grep-confirmed); `raise CommandError` on any unexplained event (`:619`), per-reason listing (`:613-614`), `--dry-run` available (`:289`). Runbook (`docs/runbooks/telescope_runs_calendar.rst:858-990`) states the four steps in order, dry-run first, and "never deletes a `CalendarEvent` on any path." | closed |
| T-35-06 | Repudiation | `ALLOC:` retire/convergence delete | low | accept | Derived/re-creatable state with no human audit of its own; every delete counted under `retired` and surfaced in the sweep summary (`reconcile_campaign_runs.py:82,95,145`). | closed |
| T-35-07 | Elevation of Privilege | new `CampaignRunObservation` receivers | low | accept | Both fire downstream of `AttributionDecisionView(StaffRequiredMixin, View)` (`campaign_views.py:1168`); no new URL, view, or form in the phase's commit range. | closed |
| T-35-08 | Information Disclosure | new signal-receiver log lines | low | accept | Every new signal-receiver log line is type-only (`type(exc).__name__`, never the exception message) — `observation_projector.py:642,656,671`, `allocation_projector.py:838,919`, pinned by `test_allocation_projector_signals.py:277-287`. (Narrower than a blanket "no exception text" claim: operator-facing command stderr/stdout in `cutover_classical_allocations.py`/`load_telescope_runs.py` does interpolate `ValueError` text composed of operator-supplied schedule content — permitted by each plan's own accepted wording, no credential/contact data.) | closed |
| T-35-09 | Repudiation | test suite migration (35-02) | medium | mitigate | Every retired test class's disposition row names a destination module or reason (`35-02-SUMMARY.md:127-140+`). | closed |
| T-35-10 | Tampering | night-boundary date-offset rule | medium | mitigate | Superseded during review (CR-06/NF-03): the declared `t.hour < 12` rule was replaced by a nearest-candidate-to-observing-night-span rule (`allocation_projector.py:205-239`, `_time_of_day_to_datetime()`) plus a `_raise_if_inverted()` backstop (`:286-318`) after the original rule was found actively wrong for east-of-UTC sites. Full UTC datetime (date included) pinned both hemispheres (`test_allocation_projector.py:1033-1051`, `:1053+`). Mitigation is stronger than originally planned. | closed |
| T-35-11 | Denial of Service | D-13 re-mint comparison | medium | mitigate | `_span_needs_remint()` (`allocation_projector.py:321-354`) makes no `sun_event()` call on any path; null-null short-circuit (`:344-345`). Zero-call property pinned with fields set (`test_allocation_projector.py:990-1002`, `:1234-1252`). | closed |
| T-35-12 | Denial of Service | unbounded work inside a caller's transaction (linked-run re-project) | medium | mitigate | `reproject_allocation_if_dispatched()` returns early before `project_allocation()` (`allocation_projector.py:774-794`); no-network asserted via a `facility_for` patched to raise (`test_allocation_projector_signals.py:303`); exact invocation count pinned (`:312`). | closed |
| T-35-13 | Tampering | batch ingest relabelling a public web submission | medium | mitigate | `write_and_reconcile_campaign_run()` strips `source`/`approval_status` when the matched row is `Source.WEB` (`campaign_utils.py:1063-1066`); ingest routes through the helper (`load_telescope_runs.py:9,323`). | closed |
| T-35-14 | Tampering | recovered `Source line:` text re-parsed by the cutover command | medium | mitigate | Re-parsed through the identical `parse_run_line()` a fresh import uses (`cutover_classical_allocations.py:121,349`); every raise routes to `_mark_unexplained` rather than guessing (`:351-352,358,366,374-375,413`). | closed |
| T-35-15 | Tampering | one-time deletion of leftover date-bearing container-run events | medium | mitigate | `writable_events()` excludes a row attributed to another run (`campaign_reconciler.py:161-164`); a `confirmed_by` companion row is left alone and reported declined (`test_campaign_reconciler.py:813-841`). | closed |
| T-35-16 | Tampering | `load_telescope_runs_demo.ipynb` writing through to the developer database | high | mitigate | Notebook routes at a throwaway copy via `FOMO_DATABASE_PATH` set before `django.setup()` (cell 0), with a teardown cell; a hard `assert` (not just prose) confirms the resolved DB name; all 14 code cells carry non-null execution counts and output. | closed |
| T-35-17 | Repudiation | stale or hand-edited notebook output (`reconcile_campaign_runs_demo.ipynb`) | medium | mitigate | Committed (HEAD) copy satisfies the invariant (17/17 code cells with non-null execution counts and output). **Working tree at audit time fails**: an uncommitted hand-edit (present before this session started, `+35/-7`) reordered the scratch-DB setup block and nulled the execution counts on cells 0 and 15 — exactly the hand-editing the mitigation forbids. Remedy: re-execute via `jupyter nbconvert --to notebook --execute --inplace`, never hand-patch. Below `block_on: high`, non-blocking. | open — below high threshold (non-blocking) |
| T-35-SC | Tampering | npm/pip/cargo installs | high | accept | No package installed in the phase-35 commit range; `pyproject.toml` unchanged. | closed |

*Status: open · closed · open — below {block_on} threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-35-01 | T-35-06 | `ALLOC:` events are derived/re-creatable state with no human audit of their own; every delete is counted and surfaced in the sweep summary. | plan 35-01 (author) | 2026-09-13 |
| AR-35-02 | T-35-07 | New `CampaignRunObservation` receivers fire downstream of an already-staff-gated view (`AttributionDecisionView`); no new entry point. | plan 35-04 (author) | 2026-09-13 |
| AR-35-03 | T-35-08 | New signal-receiver log lines carry only primary keys and `type(exc).__name__` — never credentials, contact details, or exception message text. | plans 35-01/04/05/06/07 (author) | 2026-09-13 |
| AR-35-04 | T-35-SC | No package is installed by any plan in this phase; `pyproject.toml` unchanged. | plans 35-01..07 (author) | 2026-09-13 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-15 | 18 | 17 | 1 (non-blocking) | gsd-security-auditor (retroactive, State B — first run) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed (T-35-17 is open but below `block_on: high`, non-blocking)
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-15
