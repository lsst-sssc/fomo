# Phase 26: Canonical-Record Spike - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-27
**Phase:** 26-canonical-record-spike
**Areas discussed:** Evidence standard, Stage-2 class-wide fan-out, Canonical event key, `source` for legacy rows

---

## Gray-area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Evidence standard | How the spike proves things against the real dev DB | ✓ |
| Stage-2 class-wide fan-out | One event per candidate site, or a single class-wide event | ✓ |
| Canonical event key | One key scheme stable across all four pipeline stages | ✓ |
| `source` for legacy rows | What provenance value the pre-milestone runs get | ✓ |

**User's choice:** all four areas.

---

## Todo cross-reference

| Option | Description | Selected |
|--------|-------------|----------|
| Neither (Recommended) | Both are code-cleanup todos; Phase 26 ships no code | |
| Fold the calendar_utils naming todo | `2026-07-02-rename-calendar-utils-py-private-helpers` — adjacent to the companion-record rename | ✓ |
| Fold the mapping-extraction todo | `2026-06-23-extract-site-telescope-mapping...` — already `resolves_phase: 11` | |

**User's choice:** fold the calendar_utils naming todo.
**Notes:** Folded as a *recommendation to record* rather than code to write, since the phase
is investigation-only. Does not close the todo. The 2026-06-23 mapping-extraction todo was
reviewed and left unfolded (already resolved by Phase 11; previously rejected in both
Phase 14's and Phase 18's discussions).

---

## Evidence standard

### Q1 — Evidence vehicle for SPIKE-01

| Option | Description | Selected |
|--------|-------------|----------|
| Throwaway migration on a DB copy | Real migration on a scratch branch applied to a copy of `src/fomo_db.sqlite3`, verbatim results recorded, then discarded. Phase 13 `eso_p2_probe.py` precedent | ✓ |
| Committed TestCase + PII-free fixture | Survives as a regression asset for Phases 27-29, but is a reconstruction rather than the real DB | |
| Both — throwaway proves it, fixture preserves it | Breaks the fidelity/durability tension, at more cost | |
| Shell/SQL simulation, transcript in the doc | Fastest; proves constraint mechanics rather than the Django migration path | |

**User's choice:** Throwaway migration on a DB copy.
**Notes:** SPIKE-01 requires demonstrating that pk=1 and its 11 LCO events coexist without
an `IntegrityError`, but `source` does not exist yet — so the field must actually be created
to prove anything.

### Q2 — How far the rename verification goes

| Option | Description | Selected |
|--------|-------------|----------|
| Execute and measure it | Rename + migrate + full `./manage.py test solsys_code` + load `/calendar/`; record what actually broke | ✓ |
| Grep-verified checklist only | Matches the roadmap's literal wording; Phase 27 discovers breakage itself | |
| Execute, but skip the browser check | Test suite only, relying on `test_calendar_template.py` for the template path | |

**User's choice:** Execute and measure it.
**Notes:** Surfaced during the question — because `related_name='telescope_label_meta'` is
locked unchanged, the template and `prefetch_related()` string are safe by construction, so
only two class-name imports (`admin.py`, `sync_lco_observation_calendar.py`) are actually at
risk, and both fail loudly. Recorded in CONTEXT.md as a prediction to test, not to assume.

### Q3 — New model class name

| Option | Description | Selected |
|--------|-------------|----------|
| `CalendarEventMeta` | Generic; absorbs `run`, `is_verified`, and future fields without a second rename | ✓ |
| `FomoCalendarEventMeta` | Same generality with an explicit FOMO namespace prefix; no other FOMO model carries one | |
| `CalendarEventRunLink` | Sharpest for v2.2's purpose, but misdescribes the 11 existing rows | |
| You decide | Let the spike weigh it | |

**User's choice:** `CalendarEventMeta`.

### Q4 — Handling the stale PROJECT.md claim

| Option | Description | Selected |
|--------|-------------|----------|
| Pin a snapshot + record as finding + todo | Dated git-excluded DB snapshot; discrepancy recorded; PROJECT.md fixed via separate todo | ✓ |
| Also correct PROJECT.md inline | Same, but the spike edits PROJECT.md directly | |
| Pin the snapshot only | Don't spend spike time on documentation archaeology | |

**User's choice:** Pin a snapshot + record as finding + todo.
**Notes:** Keeps the phase investigation-only.

---

## Stage-2 class-wide fan-out

### Q1 — Fan-out or single event

| Option | Description | Selected |
|--------|-------------|----------|
| Single class-wide event | One 00:00–23:59 event per day, no site. pk=29 costs 80 events | ✓ |
| One event per candidate site | 80 × 5 = 400 events for pk=29; 4 of every 5 hypothetical | |
| Single event, candidate sites in the description | Readable calendar, information preserved in prose | |

**User's choice:** Single class-wide event.
**Notes:** Decided against measured cost — `SITE_TELESCOPE_MAP` carries `1m0` at five sites,
and pk=29 (`LCO 1m`) is an 80-night window. LCO's scheduler picks exactly one site, so
stage 3 narrows to it when a record appears.

### Q2 — The space-mission case

| Option | Description | Selected |
|--------|-------------|----------|
| Widen the field to a why-no-site vocabulary | Three values: class allocation, space mission, unresolved. Closes Phase 18's deferred D-07 | ✓ |
| Keep `telescope_class` class-only; handle space separately | Narrower change to Phase 27's scope | |
| Record as a finding, defer the design | Leaves Phase 27 making a call the spike existed to remove | |

**User's choice:** Widen the field to a why-no-site vocabulary.
**Notes:** Grounded in real counts — 5 space-mission rows (pk=8, 12, 13, 21, 26) against 2
class-wide ones (pk=29, 30). CANON-02 as written names only two of the three meanings.

### Q3 — Space-mission event window

| Option | Description | Selected |
|--------|-------------|----------|
| One spanning event for the whole window | JUICE pk=26's 24-day window → one 24-day event | ✓ |
| Same as class-wide: 00:00–23:59 per day | Uniform reconciler path, but contradicts `claimed_dates()` | |
| No event until the window narrows | Reintroduces the invisibility this milestone exists to fix | |

**User's choice:** One spanning event for the whole window.
**Notes:** Keeps the calendar consistent with v2.1's asset-aware `campaign_gap.claimed_dates()`.

### Q4 — Runs with no window at all

| Option | Description | Selected |
|--------|-------------|----------|
| Define it as stage 0 explicitly | No event, but counted and reported in the reconciler summary | ✓ |
| Silent skip, documented as a non-case | Minimal, but pk=4 gets no signal it's waiting on a date | |
| You decide | Let the spike weigh it | |

**User's choice:** Define it as stage 0 explicitly.
**Notes:** Three real rows need it — pk=4 (ESO VLT FORS2, site-resolved, approved) and
pk=27/28 (JWST). Gives RECON-06's "reported and skipped" a defined case.

---

## Canonical event key

### Q1 — The canonical key

| Option | Description | Selected |
|--------|-------------|----------|
| String key for identity + `run` FK for ownership | `RUN:{run_pk}:{date}` for idempotency; companion FK as the RECON-05 ownership rule | ✓ |
| `run` FK alone, no string key | Avoids stuffing a non-URL into `CalendarEvent.url`, but needs a per-night discriminator | |
| Keep the `CAMPAIGN:{pk}:{date}` namespace | Zero migration cost, but keeps the lineage RECON-09 retires | |

**User's choice:** String key for identity + `run` FK for ownership.
**Notes:** The ownership rule — no companion row, or `run=NULL`, means never touch — is
already provable against the live DB: the 9 classical events have no companion row, and all
11 LCO ones will be `run=NULL` until attribution sets them.

### Q2 — Which date in the key

| Option | Description | Selected |
|--------|-------------|----------|
| Always the observing night, local to the site | Stages 3/4 change times but never the key | ✓ |
| UTC date of the event's `start_time` | Simplest, but the key shifts when stage 3 lands and churns every re-run | |
| Per-night sequence index | Timezone-immune but opaque, and silently re-points if the window is edited | |

**User's choice:** Always the observing night, local to the site.
**Notes:** Direct mitigation for research Pitfall #5. At Siding Spring (UTC+10) the night of
7 July begins ~09:00 UTC on 7 July but can run into 8 July UTC.

### Q3 — Adopt vs. gap-fill on the real pk=1 case

| Option | Description | Selected |
|--------|-------------|----------|
| Adopt attributed events | Update the attributed event in place; mint only the 4 uncovered nights. 15 events | |
| Fill only the gaps; adapter keeps its nights | Also 15 events; stricter RECON-05 reading, but adapter still drives stages 3/4 | |
| Reconciler always mints its own | 26 events for one run — the double-booking ATTRIB-06 prevents | |
| Spike measures both, recommends after | Prototype on the throwaway DB copy and pick from what the calendar looks like | ✓ |

**User's choice:** Spike measures both, recommends after.
**Notes:** Deliberately left to measurement rather than decided in the abstract. The rejected
"always mints its own" baseline is still to be recorded in the decision doc, with the reason.

---

## `source` for legacy rows

### Q1 — Provenance value for the 31 existing runs

| Option | Description | Selected |
|--------|-------------|----------|
| A distinct `LEGACY` value | Honest that these rows predate provenance tracking; never produced by new code | ✓ |
| All `CSV_IMPORT` | Probably true, but an unverifiable assertion written into 31 rows | |
| `LEGACY` by default, operator corrects known rows | Most accurate end state; needs a manual pass | |
| You decide | Let the spike weigh it | |

**User's choice:** A distinct `LEGACY` value.
**Notes:** Established during the question that nothing in the data discriminates provenance
— `original_obs_date_raw` is set on only 2 rows (pk=27/28), making it a parse-failure marker
rather than an import signature.

### Q2 — Vocabulary scope for Phase 27

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 + `LEGACY`, unreachable ones documented | Downstream code written once against the final vocabulary | ✓ |
| Only the reachable 3 (`WEB`, `CSV_IMPORT`, `LEGACY`) | Every value has a real code path and test | |
| All 5, no `LEGACY` — old rows get `CSV_IMPORT` | Would reverse Q1 | |

**User's choice:** All 5 + `LEGACY`, unreachable ones documented.
**Notes:** Cheap either way — Django `TextChoices` values are validation-only, so adding
them later is a no-op `AlterField`.

### Q3 — Recording that a non-web run never needed approval

| Option | Description | Selected |
|--------|-------------|----------|
| Keep `APPROVED`; `source` is the disambiguator | No new enum value, no blast radius; derivation rule recorded explicitly | ✓ |
| Add a fourth `NOT_REQUIRED` value | Directly readable, but every `approval_status` reader must handle it | |
| Change the gate, not the data | Smallest change, but a no-op today | |

**User's choice:** Keep `APPROVED`; `source` is the disambiguator.
**Notes:** Established during the question that `import_campaign_csv.py:194` already writes
`ApprovalStatus.APPROVED`, so the roadmap's claimed `approval_status` behaviour change for
the importer is already satisfied. Its real change is writing `source`.

---

## Claude's Discretion

- Structure, wording and section ordering of the decision doc and the `docs/design/` page,
  and whether they are one document or two.
- PII redaction mechanics for quoted real-sheet/dev-DB evidence (Phase 18's D-01 posture
  carries forward; not re-asked).
- Throwaway branch and DB-copy mechanics (naming, location, git-exclusion).
- How deep to take attribution-scoring prototyping beyond what the adopt-vs-gap-fill
  comparison requires.

## Deferred Ideas

- Correcting PROJECT.md's stale Phase 25 paragraph — separate todo, outside the
  investigation-only boundary.
- Renaming `related_name='telescope_label_meta'` — explicitly out of scope per REQUIREMENTS.md.
- `CalendarEvent.url` non-uniqueness (`23-REVIEW.md`) — sidestepped by putting ownership on
  the companion FK; its own change if a later phase wants `get_or_create` to be race-safe.
- v2.3 items confirmed untouched: STATUS-01/02, ADAPT-01..03, GAPB-01, UNUSED-01.

## Reviewed Todos (not folded)

- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`2026-06-23-...`) — `resolves_phase: 11`; already done by Phase 11's `calendar_utils.py`
  extraction; previously rejected in Phase 14 and Phase 18 discussions.
