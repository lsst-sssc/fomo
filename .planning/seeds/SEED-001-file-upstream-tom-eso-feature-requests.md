---
id: SEED-001
status: dormant
planted: 2026-07-02T14:02:03.000Z
planted_during: v1.7 ESO/VLT Calendar Sync — Feasibility Spike (Phase 13)
trigger_when: >
  When the TOM Toolkit team has bandwidth for tom_eso again (per Tim,
  not their focus for at least 1 month as of 2026-07-02 — check
  github.com/TOMToolkit/tom_eso activity before filing), OR when FOMO
  starts a v2 ESO/VLT sync implementation milestone (ESO-10/ESO-11) and
  needs to re-evaluate whether any of these gaps still block the chosen
  approach.
scope: small-per-issue
---

# SEED-001: File upstream tom_eso feature requests

## Why This Matters

Phase 13's ESO Feasibility Spike (v1.7) found several genuine gaps in the
installed `tom_eso==0.2.4` library — not FOMO-specific problems, but missing
or broken functionality that affects any `tom_eso` user. One such gap (the
`ESOAPI`/`p1api` wrapper unconditionally requiring a Phase 1 connection that
doesn't support `production_lasilla`) was already filed as
[TOMToolkit/tom_eso#55](https://github.com/TOMToolkit/tom_eso/issues/55).

Given FOMO's limited ESO development time and the small TOM Toolkit team's
current bandwidth, fixing the remaining gaps upstream (once maintainers have
capacity) is a better use of that limited time than reinventing the missing
functionality inside FOMO — a fix in `tom_eso` benefits every user of the
library, not just FOMO, and keeps FOMO's own ESO sync code (if/when built)
smaller and closer to the library's intended abstractions.

## When to Surface

**Trigger:** See `trigger_when` above — either TOM Toolkit team bandwidth
returns, or FOMO opens a v2 milestone for `ESO-10`/`ESO-11`
(`sync_eso_observation_calendar`).

This seed will surface automatically when you run `/gsd-new-milestone` if the
milestone scope matches (ESO/`tom_eso`-related).

## Scope Estimate

**Small per issue.** Each of the 4 items below is a focused, well-diagnosed
GitHub issue draft (root cause + evidence + suggested fix), similar in
weight to #55. Filing all 4 is maybe an hour of review/posting time once the
maintainers have bandwidth; none require FOMO-side code changes to write up.

## Draft Feature Requests

All four gaps below were found by reading the installed `tom_eso==0.2.4`
source (`tom_eso/eso.py`, `tom_eso/eso_api.py`) directly and cross-checking
against live `p2api` behavior during Phase 13's investigation — not
guessed from docs. Full research trail: `.planning/research/STACK.md`,
`.planning/phases/13-eso-feasibility-spike/13-DECISION.md`.

### 1. `submit_observation()` never returns created observation IDs — blocks `ObservationRecord` creation entirely

**Highest-value item — this is the "Headline Finding" from Phase 13's research.**

`ESOFacility.submit_observation()` (`tom_eso/eso.py`) hardcodes
`created_observation_ids = []` and returns it unconditionally, even after
`self.submit_new_observation_block(observation_payload)` successfully creates
a real P2 Observation Block. Downstream, `tom_observations`'
`ObservationCreateView.form_valid()` only creates an `ObservationRecord` row
for each ID in that returned list — so the standard TOM Toolkit submission
UI can **never** create an `ObservationRecord(facility='ESO')` row, for any
user, under any circumstances, with the installed version.

**Suggested fix:** capture and return the actual created OB ID(s) from
`submit_new_observation_block()`'s return value instead of the hardcoded
empty list.

### 2. `get_observation_status()`, `get_observation_url()`, `data_products()` all unconditionally `raise NotImplementedError`

All three TOM Toolkit standard facility methods are bare stubs in
`ESOFacility`:

- **`get_observation_status()`** — could be implemented by calling
  `ESOAPI.getOB(observation_id)` and mapping the real `obStatus` field into
  the `state`/`scheduled_start`/`scheduled_end` dict shape the base class
  expects (Phase 13 captured real `obStatus` values — `'P'` and `'M'` — via
  exactly this call).
- **`get_observation_url()`** — a working, differently-shaped method already
  exists in the same file: `get_p2_tool_url(observation_block_id=...)`,
  which builds a real `https://www.eso.org/p2[demo]/home/ob/<obId>` URL. It
  just isn't wired up to the base class's expected
  `get_observation_url(observation_id)` signature/contract.
- **`data_products()`** — lower confidence whether a P2-side equivalent
  exists at all; worth raising as a question rather than assuming a fix
  exists, or at minimum documenting it as intentionally unsupported rather
  than a silent stub.

### 3. `ESOAPI` wrapper doesn't expose per-OB execution/night history (`getOBExecutions`, `getNightExecutions`)

`tom_eso.eso_api.ESOAPI` only wraps a thin slice of `p2api.ApiConnection`:
`observing_run_choices`, `folder_name_choices`, `folder_item_choices`,
`folder_ob_choices`, `getOB`, `create_observation_block`. It does not expose
`p2api.ApiConnection.getOBExecutions(obId, night)` or
`getNightExecutions(instrument, night)` — both confirmed live during Phase
13 to be real, working Phase 2 API methods carrying exactly the
execution/completion data (`obStatus`, execution time windows) a
facility status-check flow needs. Callers currently have to reach into
`ESOAPI.api2.*` directly (bypassing the wrapper), which defeats the point of
having a wrapper.

**Suggested fix:** add thin `get_ob_executions(ob_id, night)` /
`get_night_executions(instrument, night)` passthrough methods to `ESOAPI`,
mirroring the existing `getOB` passthrough style.

### 4. No `obStatus`/terminal-state vocabulary anywhere in the library

`ESOFacility.get_terminal_observing_states()` unconditionally `return []` —
an honest "not implemented" rather than a bug, but it means the base class's
generic status-checking flow (`update_observation_status()` /
`update_all_observation_statuses()`) can never recognize a terminal ESO
state. There is also no `get_failed_observing_states()` at all — that method
only exists on `tom_observations.facilities.ocs.OCSFacility` (the LCO/SOAR
base class), which `ESOFacility` does not inherit from.

The real ESO Phase 2 `obStatus` vocabulary is a documented 12-code set
(`P`/`D`/`-`/`R`/`+`/`C`/`X`/`M`/`A`/`F`/`K`/`T`, from ESO's public Phase 2
status docs) with `C`/`X`/`F`/`K`/`T` being terminal — none of this is
captured anywhere in `tom_eso`.

**Suggested fix:** populate `get_terminal_observing_states()` with the
terminal subset (`C`, `X`, `F`, `K`, `T`), and consider adding a
`get_failed_observing_states()` analogous to `OCSFacility`'s (at minimum
`F`, `K`, `T`).

### (Optional / lower priority) `get_observing_sites()` hardcoded two-site dict

Not drafted as a full issue — the library's own source already has a
`# TODO: get data for all the ESO sites for production` comment
acknowledging this is incomplete, so it may already be a known/tracked gap
on the maintainers' side. Worth a quick check before filing a duplicate.

## Breadcrumbs

- `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` — full spike
  findings, including the confirmed workaround for item 3 above (direct
  `p2api.ApiConnection` calls) already used by FOMO's throwaway probe script.
- `.planning/research/STACK.md` — "Answers to the Research Questions"
  section has the original detailed evidence for items 1, 2, and 4.
- `docs/design/eso_feasibility_spike.rst` — durable summary; links to
  [tom_eso#55](https://github.com/TOMToolkit/tom_eso/issues/55).
- Installed source read directly: `tom_eso/eso.py`, `tom_eso/eso_api.py`
  (`tom-eso==0.2.4`), `p2api/p2api.py` (`p2api==1.0.10`).

## Notes

Filed via `/gsd-capture --seed` during Phase 13 close-out, at the operator's
explicit request to park this work rather than file it immediately (small
TOM Toolkit team, `tom_eso` not their current focus). Issue #55 (the
`p1api`/`production_lasilla` wrapper bug) was filed separately, already, as
a distinct issue — see `13-DECISION.md` ESO-01 finding.
