---
created: 2026-06-23T04:22:14.003Z
title: Extract site/telescope mapping and instrument extraction into own module
area: general
files:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
---

## Problem

`solsys_code/management/commands/sync_lco_observation_calendar.py` has grown two
fairly general-purpose pieces of logic that live inside the management command
module rather than a dedicated module:

- `SITE_TELESCOPE_MAP` (site code -> telescope-class label) and its resolution
  helpers (`_derive_telescope`, and as of Phase 7, `_resolve_placement_block` /
  `_aperture_class_from_telescope_code`).
- The `c_1..c_5` configuration-type / instrument-extraction logic
  (`_extract_instrument`, `_find_science_config`, `_find_exposure_signal_config`,
  `_has_muscat_exposure_signal`).

This surfaced during Phase 7 review (live telescope-label resolution) — it feels
like this mapping/extraction logic doesn't belong long-term inside a management
command file, and should probably be importable from elsewhere.

Checked for a reusable shared implementation before deciding to extract anything:
`LCOGT/mop` uses the identical `c_N_instrument_type` flat-key convention
(`mop/views.py`) and the same instrument-code vocabulary (e.g.
`1M0-SCICAM-SINISTRO`), confirming the data shape is industry-wide rather than a
FOMO quirk — but it has no shared site/instrument-mapping module FOMO could
import; each LCOGT TOM app (MOP, presumably SNEX2) re-derives this independently.
SNEX2 search hit GitHub's API rate limit before it could be checked — worth a
second look if pursuing this.

Note: `SITE_TELESCOPE_MAP` must stay separate from `telescope_runs.py:SITES`
(different purpose/shape — MPC code lookup vs. telescope-class label — per Phase
7 decision D-05). Don't conflate them when extracting.

Deliberately deferred until Phase 7 ships, since its plans (07-01-PLAN.md,
07-02-PLAN.md) were already written against the current file layout and
mid-phase reshuffling would add risk for no immediate benefit (single consumer
today).

## Solution

TBD. Options to weigh once Phase 7 is done and stable:
- Extract into a new `solsys_code/lco_observation_mapping.py` (or similar) module
  importable by the management command, keeping `telescope_runs.py` untouched.
- Or leave as-is if no second consumer ever materializes — premature extraction
  for a single-consumer module has its own cost.
- Re-check SNEX2 for a possibly-shared implementation pattern (rate-limited out
  during the Phase 7 investigation).
