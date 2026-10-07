---
created: 2026-10-07T02:59:35.000Z
title: "Keep a request's site restriction and show it in the event title from the start (LCO- for any site)"
area: observation-projector
severity: minor
files:
  - solsys_code/management/commands/backfill_lco_observations.py:274
  - solsys_code/observation_projector.py:225
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

Discovery's `_build_parameters()` drops `location`, so a site-restricted request (e.g. `site: lsc` or `site: coj`) is titled with the coarse class (`1m0 11P`, `2m0 Didymos …`) until a block is placed; an unrestricted unplaced/expired record reads `[X] 1m0 11P` beside placed `[O] CPT-1m0 11P`.

Source: `.planning/v2.4-INTENT-REVIEW.md`, F10 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Store `location.site`; the title token is the restricted site when there is one, `LCO-1m0` when there is not, and the observed site once a block is placed. Same width as today's site-prefixed token.
