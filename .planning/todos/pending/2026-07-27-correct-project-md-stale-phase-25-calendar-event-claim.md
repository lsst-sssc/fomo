---
created: 2026-07-27T00:00:00.000Z
title: Correct PROJECT.md's stale Phase 25 calendar-event claim
area: docs
files:
  - .planning/PROJECT.md
---

## Problem

`.planning/PROJECT.md`'s Phase 25 paragraph (in **Current State → Working code**) states:

> the real (non-dry-run) backfill was run during UAT and confirmed live: pk=34 now has its
> 4 per-night `CalendarEvent`s

This does not reproduce against the current dev DB (`src/fomo_db.sqlite3`, checked
2026-07-27 during Phase 26's discuss-phase):

- The maximum `CampaignRun` pk is **31**, so there is no pk=34.
- There is no `GS-2026A-FT-115` row at all.
- There are **zero** `CAMPAIGN:`-namespaced `CalendarEvent`s. The 20 events present are
  9 classical (blank `url`) and 11 LCO (`https://observe.lco.global/...`).

The same paragraph's secondary claim — that the dry run surfaced pk=27 and pk=29 failing
with "Observatory 'FTN' has no timezone set" — also doesn't line up with current rows
(pk=27 is a JWST row with no window and no site; pk=29 is `LCO 1m`).

The dev DB was evidently re-imported after Phase 25's UAT. The Phase 25 work itself is not
in question — the *record* of its verification is what has gone stale.

This matters because Phases 27-29 read PROJECT.md as milestone context and would otherwise
trust a claim about live calendar state that is no longer true.

## Solution

TBD. Options to weigh:

- Rewrite the claim to be explicitly historical and date-pinned — e.g. "confirmed live
  against the dev DB as of 2026-07-18; the dev DB has since been re-imported, so these pks
  no longer resolve" — preserving the verification record without asserting current state.
- Or re-run the Phase 25 backfill against the current DB and restate with fresh pks, if a
  range-window run that needs it still exists.
- While touching this, consider whether other "confirmed live against the dev DB" claims in
  PROJECT.md's Current State section carry the same rot risk and should be date-pinned as a
  convention.

Recorded as a finding in `.planning/phases/26-canonical-record-spike/26-CONTEXT.md` (D-16),
per that phase's D-04 decision — the spike is investigation-only and deliberately does not
edit PROJECT.md itself.
