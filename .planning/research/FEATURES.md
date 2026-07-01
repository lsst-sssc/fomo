# Feature Research

**Domain:** ESO/VLT ObservationRecord calendar sync (FOMO v1.7, Stage 4 continuation of issue #37)
**Researched:** 2026-07-01
**Confidence:** HIGH for "what tom_eso/p2api actually expose" (verified directly against installed package source, `tom-eso==0.2.4`, `p2api==1.0.10`, and this dev DB); MEDIUM for "how ESO OB execution/status conventions work" (ESO Operations Helpdesk / eso.org docs, cross-corroborated across multiple pages but not hands-on verified against a live P2 account)

## Headline Finding (read this first)

**There is currently no path for `ObservationRecord(facility='ESO')` rows to exist in this codebase, and if they did, `tom_eso.eso.ESOFacility` exposes no execution status at all.** Both facts were verified directly, not inferred:

1. `ESOFacility.submit_observation()` calls `submit_new_observation_block()`, which does create a real OB in ESO's P2 tool, but then returns a hardcoded empty list: `created_observation_ids = []` (`tom_eso/eso.py:669-672`). `ObservationCreateView.form_valid()` in `tom_observations/views.py` only creates an `ObservationRecord` for each id in that returned list — so the standard "Submit Observation" UI flow **never creates an ObservationRecord for ESO**, on any FOMO deployment running `tom-eso==0.2.4`. Confirmed empirically: `ObservationRecord.objects.filter(facility='ESO').count()` is `0` in this dev DB (in fact the whole `ObservationRecord` table is currently empty for every facility).
2. Even setting that aside, `ESOFacility.get_observation_status()`, `get_observation_url()`, and `data_products()` all `raise NotImplementedError`, and `get_terminal_observing_states()` returns `[]` — so `ObservationRecord.terminal` / `.failed` / `.url` (which delegate to these) are unusable for ESO today.

This reframes the whole milestone: **the interesting research question isn't "what does the sync command's window-derivation logic look like" (that part is a straightforward Gemini-style fallback) — it's "what do we do about the fact that there's no live data to sync yet, and no status even if there were."** Recommend surfacing this as an explicit spike/decision task at the start of the v1.7 phase rather than assuming it away.

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `sync_eso_observation_calendar` command, one `CalendarEvent` per `ObservationRecord(facility='ESO')` | Matches the shipped LCO/Gemini pattern exactly (low-surprise for anyone who's used the other two commands) | LOW (mechanical, once the data gap below is resolved) | Reuse `insert_or_create_calendar_event()` from `calendar_utils.py` verbatim — no new create-or-update logic needed |
| Idempotent no-churn create-or-update keyed on a stable string | Consistency with LCO (`url` = portal URL) and Gemini (`GEM:{prog}/{observation_id}`) — re-running must not duplicate or touch `modified` | LOW | `ESOFacility.get_observation_url()` raises `NotImplementedError`, so the key must be synthetic like Gemini's, e.g. `ESO:{progId}/{obId}`, not derived from the facility |
| Submission-time window banner (no placed-block state) | ESO Service Mode has no advance `scheduled_start`/`scheduled_end` publication — Paranal Science Operations chooses which OBs to execute in real time based on current conditions, not a queue schedule visible ahead of time (unlike LCO). A Gemini-style "best window we know today" banner is the honest MVP | MEDIUM | Window source is the OB's **observing run/period validity dates** (from `getRun()` — same call `observing_run_choices()` already uses), or a PI-set `getAbsoluteTimeConstraints(obId)` window if present (rare) — never a per-night schedule |
| Telescope/instrument derived from the OB's run metadata | Already surfaced by `ESOAPI.observing_run_choices()` as `f"{progId} - {telescope} - {instrument}"` — no new API surface needed | LOW | Cheaper than LCO's per-record site-resolution API call; the run-level `telescope`/`instrument` strings are already fetched for the submission form |
| Data-existence precondition resolved (spike, not code) | Without this, `sync_eso_observation_calendar` is untestable against real data and will always report 0 records | MEDIUM (research/decision, not implementation) | Either (a) patch `submit_new_observation_block`/`submit_observation` upstream in `tom_eso` to actually append the new OB id, (b) seed test `ObservationRecord` rows manually for dev/testing, or (c) explicitly scope this milestone to "command exists and is correct against fixture data; real production data pending an upstream fix" |

### Differentiators (Competitive Advantage — Defer These)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Real per-night execution status prefix (e.g. `[EXECUTED]`/`[MUST REPEAT]`/`[ABORTED]`, mirroring LCO's terminal-state prefixing) | Would give astronomers the same at-a-glance execution signal LCO/Gemini already have | HIGH | The *installed* `p2api` package (already a `tom_eso` dependency) exposes `getOBExecutions(obId, night)` and `getNightExecutions(instrument, night)` — real per-night execution records with grade — but `tom_eso` doesn't wrap either. Bypassing `ESOFacility` to call `p2api` directly is possible but needs (1) a decision on which night(s) to poll per OB (no single "current status" endpoint exists — it's queried per night), and (2) credentials: `ESOProfile` today is per-Django-user only; there is no `settings.FACILITIES['ESO']` service-account entry the way `GEM`/`LCO` have, so a standalone management command has no obvious identity to authenticate as |
| Deep link to ESO's Run Progress / Night Report page | Lets the astronomer click through to ESO's own execution report instead of FOMO re-implementing it | LOW | These pages are dynamically generated, per-run, web-only (no public API), and gated behind ESO User Portal login — confirmed via ESO Operations Helpdesk docs. A plain hyperlink (no data parsing) is cheap and genuinely useful precisely *because* FOMO can't reliably show the real status itself |
| Distinguishing VLT UT1-4 (specific unit telescope) vs bare "VLT" | Parity with the existing Magellan Baade/Clay ambiguity decision, but for VLT | MEDIUM (unconfirmed) | Unverified whether `getRun()`'s `telescope` field ever returns a specific UT (e.g. `UT2`) vs just `VLT` generically — needs inspection of a real run response before committing to a scheme; don't guess |
| Grouping multiple same-night OBs into one visual cluster | Nice-to-have polish, same spirit as the proposal-color legend already shipped in v1.4 | MEDIUM | Depends on real usage patterns (how many concurrent ESO OBs per night is typical) — defer until there's real synced data to observe |
| Ingesting PI-set absolute time constraints (`getAbsoluteTimeConstraints`) for time-critical OBs | Slightly tighter window than the full run-period fallback for the (rare) OB that has one | LOW-MEDIUM | Small, well-scoped enhancement layered on top of the run-period banner; most OBs won't have one, so it's an optional refinement not a baseline requirement |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| "Poll ESO for a single live status field per OB" | Mirrors the LCO/Gemini `status` field mental model | No such endpoint exists in the ESO P2 API — status must be reconstructed by querying `getOBExecutions`/`getNightExecutions` night-by-night across a date range, which is expensive and still doesn't capture pre-execution p2 states (Defined/Accepted/Rejected) in the same call | Submission-time window banner (table stakes) + optional deep link to ESO's own Run Progress page (differentiator) for humans who want the real answer |
| "Fix the ObservationRecord-creation gap as part of this milestone's sync command" | It's the actual root blocker, so it's tempting to just patch it while you're in the area | That's an upstream `tom_eso.eso.ESOFacility.submit_observation()` bug fix — a different code path (OB *submission*) and a different package boundary than "sync existing ObservationRecord rows to CalendarEvents" (this milestone's actual scope per PROJECT.md) | Treat as an explicit blocking dependency/spike to flag to the user, not something the sync command silently works around (e.g., by reading OBs directly from p2 instead of from `ObservationRecord`, which would abandon the LCO/Gemini contract this whole feature line is built on) |
| "Reuse the LCO queued-banner -> placed-block state machine (SYNC-02/03) verbatim for ESO" | Copy-paste consistency with the existing, well-tested LCO pattern | Factually wrong for ESO: there is no advance per-OB `scheduled_start`/`scheduled_end` publication for Service Mode — every event would permanently render as "queued" and never transition, which is a silent UX bug, not a faithful status representation | A single-state submission-time banner (Gemini's pattern, not LCO's two-state pattern) is the honest model for ESO |

## Feature Dependencies

```
[ObservationRecord(facility='ESO') data-gap spike]
    |--blocks--> [sync_eso_observation_calendar command -- table stakes]
                       |--requires--> [insert_or_create_calendar_event() -- already shared, calendar_utils.py]
                       |--requires--> [synthetic idempotency key, e.g. ESO:{progId}/{obId}]

[sync_eso_observation_calendar command] --enhances-with--> [Deep link to ESO Run Progress page]
[sync_eso_observation_calendar command] --enhances-with--> [Absolute time constraint ingestion]

[Real per-night execution status] --requires--> [sync_eso_observation_calendar command]
[Real per-night execution status] --requires--> [Service-account ESO P2 credential story for management commands]
                                                     (does not exist today -- ESOProfile is per-Django-user only)

[VLT UT1-4 disambiguation] --requires--> [confirming getRun()'s telescope field granularity against a real response]
```

### Dependency Notes

- **The data-gap spike blocks the sync command in practice, not in principle:** the command can be written and tested against fixture data regardless, but it will process zero real records in this dev DB (and likely any current FOMO deployment) until either `tom_eso` is patched upstream or records are seeded another way. Plan this explicitly rather than discovering it mid-phase the way the LCO `SITE_TELESCOPE_MAP` gap was discovered in v1.3.
- **Real per-night execution status requires a credential story that doesn't exist yet:** `GEM`/`LCO` both authenticate via a hardcoded `settings.FACILITIES[...]` dict usable by a standalone management command; ESO's only configured credential path (`ESOProfile`) is tied to a Django `User` and isn't wired into `settings.FACILITIES` at all in this codebase. Building this differentiator means solving that credential-context problem first, which is a meaningfully bigger lift than "one more per-record API call" (the LCO Phase 7 pattern).
- **Deep link and absolute-time-constraint ingestion are both cheap enhancements that don't need the credential-context problem solved** — they layer on top of the table-stakes banner without requiring new authenticated polling infrastructure.

## MVP Definition

### Launch With (v1 = v1.7)

- [ ] Resolve/confirm the `ObservationRecord(facility='ESO')` creation gap — spike/decision, not code; must happen before or alongside command-writing so the phase isn't built and verified against a dataset that structurally can't exist yet
- [ ] `sync_eso_observation_calendar` management command: one `CalendarEvent` per record, submission-time window banner (run-period dates, or PI-set absolute time constraint if present), telescope/instrument from run metadata, idempotent no-churn create-or-update keyed on a synthetic `ESO:{progId}/{obId}`-style string
- [ ] Title/status semantics honestly reflect "no fixed schedule known yet" (a single-state banner, not LCO's queued->placed transition) — essential to avoid shipping a status that silently lies

### Add After Validation (v1.x)

- [ ] Deep link to ESO's Run Progress / Night Report page per synced record — trigger: once the banner ships and real (or seeded) records exist to link from
- [ ] Absolute time constraint ingestion for the rare time-critical OB — trigger: once a real OB with `getAbsoluteTimeConstraints` data is observed in practice

### Future Consideration (v2+)

- [ ] Real per-night execution status via direct `p2api` polling — defer: requires solving the management-command credential-context gap first, plus a per-night polling design; disproportionate to the value until the data-gap spike is resolved and there's a real backlog of records to check
- [ ] VLT UT1-4 disambiguation — defer: unverified whether the data even distinguishes UTs; needs a real API response inspected first
- [ ] Multi-OB same-night grouping/visual clustering — defer: needs real usage data on typical concurrent-OB counts per night

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|----------------------|----------|
| Data-gap spike/decision | HIGH (blocks everything else) | MEDIUM | P1 |
| `sync_eso_observation_calendar` command (banner, idempotent) | HIGH | LOW-MEDIUM | P1 |
| Telescope/instrument from run metadata | MEDIUM | LOW | P1 |
| Deep link to ESO Run Progress page | MEDIUM | LOW | P2 |
| Absolute time constraint ingestion | LOW-MEDIUM | LOW-MEDIUM | P2 |
| Real per-night execution status via direct `p2api` polling | HIGH | HIGH | P3 |
| VLT UT1-4 disambiguation | LOW | MEDIUM (unconfirmed) | P3 |
| Multi-OB same-night grouping | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for v1.7 launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Comparison: LCO vs Gemini vs ESO Sync Patterns

| Aspect | LCO (`sync_lco_observation_calendar`) | Gemini (`sync_gemini_observation_calendar`) | ESO (this milestone) |
|--------|----------------------------------------|-----------------------------------------------|------------------------|
| Idempotency key | `LCOFacility().get_observation_url(observation_id)` (real portal URL) | Synthetic `GEM:{prog}/{observation_id}` | Must be synthetic — `get_observation_url()` raises `NotImplementedError`; use `ESO:{progId}/{obId}` style |
| Advance schedule available? | Yes — `scheduled_start`/`scheduled_end` published by the LCO queue scheduler | No — window derived from explicit params or ToO-type default | No — Service Mode has no advance per-OB schedule; run-period fallback is the ceiling of what's knowable ahead of time |
| Two-state (banner -> placed block)? | Yes (SYNC-02/03) | No — single-state window banner | No — single-state window banner (same shape as Gemini, not LCO) |
| Terminal/failure status available? | Yes, via `get_failed_observing_states()`/`get_terminal_observing_states()` | Partial — `[ON_HOLD]` prefix from `ready` flag | No — `get_terminal_observing_states()` returns `[]`; `get_observation_status()` raises `NotImplementedError` |
| Does `ObservationRecord` data exist to sync? | Yes (real rows exist/existed in this dev DB, drove the v1.3 correctness fixes) | Presumably (pattern shipped and tested) | **No** — confirmed empty; standard submission flow doesn't even create rows for ESO today |
| Credential story for a management command | `settings.FACILITIES['LCO']` (already exists) | `settings.FACILITIES['GEM']['programs']` (already exists) | None configured — `ESOProfile` is per-Django-user only, not wired into `settings.FACILITIES` |

## Sources

- `tom_eso/eso.py`, `tom_eso/eso_api.py`, `tom_eso/models.py` (installed `tom-eso==0.2.4` at `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_eso/`) — read directly; source of the `NotImplementedError`/empty-`created_observation_ids` findings
- `p2api/p2api.py` (installed `p2api==1.0.10`) — read directly; source of `getOBExecutions`, `getNightExecutions`, `getAbsoluteTimeConstraints` method inventory
- `tom_observations/views.py` (`ObservationCreateView.form_valid`), `tom_observations/models.py` (`ObservationRecord.terminal`/`.failed`/`.url` properties) — read directly; confirms the `ObservationRecord`-creation dependency on `submit_observation()`'s return value
- Live query against this repo's dev DB: `ObservationRecord.objects.filter(facility='ESO').count()` == 0 (whole table currently empty)
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` — existing fallback-window pattern this research compares ESO against
- `.planning/PROJECT.md`, `docs/design/telescope_runs_calendar.rst` — milestone scope and Stage 4 design notes
- [ESO — Phase 2 status](https://www.eso.org/sci/observing/phase2/p2intro/phase-2-status.html) — OB status code lifecycle (P/D/R/+/-/C/X/M/A/F/K/T)
- [Program execution status — ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/program-execution-status) — how astronomers monitor OB execution (Run Progress pages, night-log email subscription, p2 tool status)
- [Help for Night Report / Progress Pages](https://www.eso.org/sci/php/phase2/run_progress_legend.html) — confirms Run Progress pages are dynamic, web-only, no public API
- [After the execution my OBs have status C, A, M — ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/after-the-execution-my-obs-have-status-c-or-a-and-m-what-does-it-mean)
- [How do I choose between Service Mode, Visitor Mode and Designated Visitor Mode — ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/how-to-choose-between-service-sm-visitor-vm-and-designated-visitor-dvm-mode) — Service Mode = real-time queue scheduling by Paranal staff; Visitor/Designated-Visitor Mode = pre-arranged dates (already out of scope, handled by Stage 2 classical ingest)

---
*Feature research for: ESO/VLT ObservationRecord calendar sync (FOMO v1.7)*
*Researched: 2026-07-01*
