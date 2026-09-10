# Phase 34: The Observation Projector & Trigger - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-10
**Phase:** 34-The Observation Projector & Trigger
**Areas discussed:** Compact title & series stem, Verified telescope labels & is_verified, Edge lifecycles, Sweep/retirement/live proof

---

## Todo cross-reference

| Todo | Folded |
|------|--------|
| Retarget ADAPT-03 to SOAR / Gemini caveat (`2026-09-02-…`) | ✓ |
| Extract site/telescope mapping into own module (`2026-06-23-…`) | |
| Skip sun_event for existing reconciler nights (`2026-09-01-…`) | |
| The two attribution-UI todos (`2026-09-01-…` ×2) | |

---

## Compact title & series stem

| Option | Description | Selected |
|--------|-------------|----------|
| Short marker + target | `[Q] 3I/ATLAS`; telescope/instrument in modal; prefix maps updated | ✓ (with aperture class added) |
| Aperture + target, long prefix | `[QUEUED] 2m0 3I/ATLAS`; target usually cut off | |
| Keep long title, derive compact at render | Stored title unchanged; template tag builds cell text | |

**User's choice:** Option 1 amended to `[Q] 2m0 3I/ATLAS` — keep the telescope class. Asked what token non-LCO events (classical runs, VLT/NTT/GN/GS) would use; answered from the code: site short name from `telescope_runs.SITES` for allocation/classical events (Phase 35), aperture class / observed telescope for LCO/SOAR records — recorded as a cross-layer convention (D-05).

| Option | Description | Selected |
|--------|-------------|----------|
| Stem only in title; position at render time | "Night n of N" and group link rendered in the modal from meta.observation_group | ✓ |
| Stem + position appended to title | `[Q] 1m0 11P 22/28`; group-wide re-title when N changes | |
| Stem + group name in description only | Description churns group-wide when N changes | |

**User's choice:** Stem only; position at render time.

| Option | Description | Selected |
|--------|-------------|----------|
| Short letters everywhere | [Q]/[S]/[X]/[C]/[F], no marker when observed; prefix maps updated in-phase | ✓ (with [O] added) |
| Short for live stages, long words for terminal | [Q]/[S] + today's [EXPIRED]/[CANCELLED]/[FAILED] | |
| No marker for queued | Clean title while queued; markers only for placed/terminal | |

**User's choice:** Option 1, but questioned "no marker" for observed.

| Option | Description | Selected |
|--------|-------------|----------|
| [O] explicit | Every projector title carries exactly one marker; "no marker" reserved for non-observation events | ✓ |
| Clean title = observed | Observed stays unmarked as the old sync did | |

**User's choice:** `[O]` explicit.
**Notes:** Rationale accepted: a bare title cannot be told apart from an allocation/classical/`RUN:` event that simply has no marker.

---

## Verified telescope labels & is_verified

| Option | Description | Selected |
|--------|-------------|----------|
| Drop live verification from the observation layer | Coarse aperture class always; [UNVERIFIED]/is_verified retired | partly |
| Verify at discovery, store on the record | backfill stores the block's site; projector reads it | |
| Sweep-only opt-in verification | `--verify-labels` on the sweep | |

**User's choice (free text):** coarse `2m0` while pending/scheduled; once observed, the actual telescope/site — FTN/FTS for 2m0, OGG/ELP/TFN/LSC/CPT/COJ for 1m0/0m4.

| Option | Description | Selected |
|--------|-------------|----------|
| The sweep, once per newly-observed record | One resolve_placement_block() call per record, ever; receiver network-free | ✓ |
| FOMO observation_change_state hook | Fetch on →COMPLETED inside updatestatus's save | |
| FOMO facility subclass keeps the site | Override update_observation_status(); zero extra calls but invasive | |

**User's choice:** The sweep, once per newly-observed record.

| Option | Description | Selected |
|--------|-------------|----------|
| On CalendarEventMeta (FOMO-owned field) | New nullable field, migration 0018 | |
| In ObservationRecord.parameters under a FOMO key | e.g. `fomo_observed_site` | ✓ (amended: generic, un-prefixed keys) |

**User's choice (free text):** `ObservationRecord.parameters`, but not a FOMO-specific key, to be useful to non-FOMO TOMs; asked whether base TOM / SNEx2 / BHTOM populate parameters with the observed site or whether it comes from `DataProduct`.
**Notes:** Checked: base TOM has no convention (`parameters` = submitted request; LCO form's `site` is a submission constraint; only `ReducedDatum.telescope` records an observed telescope, per datum). Decision: generic keys mirroring the OCS block field names (`observed_site`, `observed_telescope`), exact names planner's.

| Option | Description | Selected |
|--------|-------------|----------|
| Site code only | `[O] LSC 3I/ATLAS` (16 chars) | |
| Site + aperture | `[O] LSC-1m0 3I/ATLAS` (20 chars; target truncated in cell) | ✓ |
| Site code in title, site-aperture in event.telescope | Two vocabularies | |

**User's choice:** Site + aperture. Retirement of `[UNVERIFIED]`/`is_verified=False`/dashed border for observation events was accepted via the area summary.

---

## Edge lifecycles

| Option | Description | Selected |
|--------|-------------|----------|
| Full request window, marked | What the old sync and record_time_window() do | ✓ |
| Collapse to the window's last night | Single-night event on window close | |
| Collapse to the window's first night | Single-night event on window open | |

**User's choice:** Full request window, marked.

| Option | Description | Selected |
|--------|-------------|----------|
| Project what is knowable; skip the rest | COMPLETED-no-block → [O] on window; half-set/no window → unprojectable, logged | |
| Mark unprojectable records visibly | As above, but half-set schedule with a window gets a `[?]` marker on the calendar | ✓ |
| Skip anything that isn't a clean stage | COMPLETED-no-block also skipped | |

**User's choice:** Mark unprojectable records visibly.

| Option | Description | Selected |
|--------|-------------|----------|
| Delete the event with it | pre_delete receiver via meta.observation_record before SET_NULL | ✓ |
| Keep the event, link cleared | Becomes an unlinked attributable event | |
| Keep the event but mark it | `[D]`-style title never maintained again | |

**User's choice:** Delete the event with it.

| Option | Description | Selected |
|--------|-------------|----------|
| m2m_changed receiver too | Re-project pk_set records on post_add/post_remove/post_clear | ✓ |
| Sweep only | observation_group filled by the next sweep | |

**User's choice:** m2m_changed receiver too.

---

## Sweep, retirement & live proof

| Option | Description | Selected |
|--------|-------------|----------|
| Zero-arg default, optional narrowing | Sweeps every LCO/SOAR record; optional --proposal/--facility/--dry-run | ✓ |
| Mirror the retired command's interface | Required `--proposal <codes|ALL>` | |
| Mirror reconcile_campaign_runs exactly | Only `--dry-run` | |

**User's choice:** Zero-arg default, optional narrowing.

| Option | Description | Selected |
|--------|-------------|----------|
| Delete outright | Command, 38 tests, notebook go; tests/notebook/runbook migrated; helpers stay | ✓ |
| Stub that raises CommandError | Friendlier to external scripts; criterion 5 would need re-wording | |

**User's choice:** Delete outright.

| Option | Description | Selected |
|--------|-------------|----------|
| First sweep does it; notebook shows the diff | Re-title, link, ~60 site lookups; no migration | ✓ |
| One-time takeover command, then sweep | Separate command to write and retire | |
| Data migration links, sweep re-titles | RunPython parses observation_id from URL | |

**User's choice:** First sweep does it; notebook shows the diff.

| Option | Description | Selected |
|--------|-------------|----------|
| Notebook live-narrowing section, re-run after nights | From spike 004's recheck.py; operator runs only updatestatus; re-executed later; dates in 34-UAT.md | ✓ |
| Hold phase verification open until the re-check | Blocks Phase 35 planning on the scheduler/weather | |
| UAT log only | Eyeball check; no re-runnable evidence | |

**User's choice:** Notebook live-narrowing section, re-run after nights.

---

## Claude's Discretion

Module/command naming and home; exact `parameters` key names; whether `--dry-run` performs site lookups (recommended no); multi-group pick; `target_list` rule; description body; log levels; summary wording; `[?]` in the legend; receiver silencing for bulk fixtures; test layout; notebook diff presentation.

## Deferred Ideas

- `GN`/`GS`-style telescope tokens for Gemini/other facilities in the `SITES` vocabulary — Phase 35/37 concern once a facility with real read-back exists.
- Reviewed, not folded: extract-site/telescope-mapping todo (overtaken by the command's deletion), skip-sun_event todo (Phase 35), the two attribution-UI todos.
