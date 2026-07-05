# Phase 14: Campaign Data Model & Bootstrap Import - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-02
**Phase:** 14-Campaign Data Model & Bootstrap Import
**Areas discussed:** Status vocabulary shape, CSV row identity & re-import, Campaign/target/site
resolution, PII fixture strategy (CAMP-05)

---

## Pre-discussion: sequencing check

Discussion originally started on Phase 15 (Per-Campaign Table View). Flagged that Phase 15 depends on
Phase 14 (`CampaignRun` model), which had no phase directory, plans, or migration yet.

**User's choice:** Stop and discuss/plan/build Phase 14 first.
**Notes:** Phase 15's discussion was abandoned with no files created; restarted here on Phase 14.

---

## Todo cross-reference

One pending todo scored a partial match (0.6): `2026-06-23-extract-site-telescope-mapping-and-
instrument-extraction-int.md`, proposing extraction of `SITE_TELESCOPE_MAP`/instrument-extraction
logic out of `sync_lco_observation_calendar.py`.

| Option | Description | Selected |
|--------|-------------|----------|
| Leave deferred | Different problem (LCO-only telescope-class labels vs. free-text site/telescope parsing across many facilities) | |
| Fold into Phase 14 | Extract the shared module now and have the CSV importer reuse it | |

**User's choice:** Neither — informed that the extraction already happened.
**Notes:** Confirmed live: `solsys_code/calendar_utils.py` already contains `SITE_TELESCOPE_MAP`,
`_derive_telescope`, `_extract_instrument`. Todo is stale; reviewed but not folded.

---

## Status vocabulary shape

| Option | Description | Selected |
|--------|-------------|----------|
| One flat CharField, ~6 choices | pending_review/approved/rejected/observed/reduced/published as one linear list | |
| Two fields, exposed as one | Separate approval_status + lifecycle_status, computed display property | ✓ (refined further) |
| Other | — | |

**User's choice:** Two fields — but first questioned whether CAMP-03's status was per-campaign or
per-run (clarified: per-`CampaignRun`, `TargetList` has no status). Then flagged that neither flat
option nor the two-field option as first framed captured "a DDT/proposal request whose outcome is
still pending" as distinct from admin approval state.

**Follow-up — run_status value set:**

| Option | Description | Selected |
|--------|-------------|----------|
| requested/planned/observed/reduced/published (5 linear states) | Recommended baseline | |
| Same + single combined terminal state | One 'cancelled' covering both not-granted and called-off | |
| Same + two distinct terminal states | not_granted, cancelled | |
| Three distinct terminal values | cancelled, not_awarded, weather_tech_failure | ✓ |

**User's choice:** Three distinct terminal values: `cancelled`, `not_awarded`, `weather_tech_failure`.
**Notes:** Final `run_status`: requested → planned → observed → reduced → published, cancelled,
not_awarded, weather_tech_failure (8 values). `approval_status`: pending_review/approved/rejected
(3 values, unchanged from CAMP-03's literal text).

**Follow-up — bootstrap import default approval_status:**

| Option | Description | Selected |
|--------|-------------|----------|
| approved | Historical vetted data, immediately visible | ✓ |
| pending_review | Every imported row treated like a fresh submission | |

**User's choice:** approved for the bootstrap import; the pending_review path is demonstrated via the
demo notebook (synthetic data) instead.

---

## CSV row identity & re-import

| Option | Description | Selected |
|--------|-------------|----------|
| (campaign, telescope, obs date/UT start) | Mirrors existing CalendarEvent find-or-create key | ✓ |
| (campaign, contact_email, obs date/UT start) | Keys off submitter instead of telescope | |
| Row position in the CSV | No semantic key | |
| Other | — | |

**User's choice:** (campaign, telescope, obs date/UT start).

**Follow-up — partial row failure granularity:**

| Option | Description | Selected |
|--------|-------------|----------|
| Skip whole row only if key fields fail; null non-key fields | | ✓ |
| Skip whole row on ANY field failure | | |
| Other | — | |

**User's choice:** Skip whole row only if key fields fail; null non-key fields.

---

## Campaign/target/site resolution

| Option | Description | Selected |
|--------|-------------|----------|
| Required --campaign CLI arg, find-or-create TargetList | | ✓ |
| A column in the CSV names the campaign per-row | No such column in real data | |
| Other | — | |

**User's choice:** Required `--campaign` CLI arg, find-or-create by name.

**Follow-up — optional Target FK:**

| Option | Description | Selected |
|--------|-------------|----------|
| Leave it unset for this import | Matches CAMP-02's own framing | |
| Resolve it automatically | Auto-assign if TargetList has exactly one Target | ✓ |

**User's choice:** Resolve it automatically.

**Follow-up — site field FK vs. free text:**

| Option | Description | Selected |
|--------|-------------|----------|
| Free-text field, no FK | Most sites outside FOMO's Observatory registry | |
| Observatory FK, stub-create unknown sites | | |
| Other | 3-tier resolution: local lookup → MPC API → placeholder+flag | ✓ |

**User's choice:** Try existing Observatory records first; if not found, query the MPC Obscodes API
(same one `Observatory`/`MPCObscodeFetcher` already uses); if still not found, create a placeholder
Observatory record and flag for further attention.

**Follow-up — storage shape for raw text + flag:**

| Option | Description | Selected |
|--------|-------------|----------|
| site (FK, nullable) + site_raw (text) + site_needs_review (bool) | Mirrors CalendarEventTelescopeLabel.is_verified precedent | ✓ |
| site (FK, nullable) only | Placeholder Observatory rows double as review signal | |
| Other | — | |

**User's choice:** site (FK, nullable) + site_raw (text) + site_needs_review (bool).

---

## PII fixture strategy (CAMP-05)

| Option | Description | Selected |
|--------|-------------|----------|
| Small hand-built synthetic CSV, same columns, fake data | | ✓ |
| Real sheet with contact columns redacted at notebook run-time | | |
| Other | — | |

**User's choice:** Small hand-built synthetic CSV, same columns as the real sheet, fake names/emails.

**Follow-up — fixture location and MPC API live-call coverage:**

| Option | Description | Selected |
|--------|-------------|----------|
| docs/notebooks/pre_executed/fixtures/, seeded sites only | Avoids live network dependency in committed notebook | ✓ |
| Same location, one row exercising a live MPC API lookup | Realistic but network-dependent | |
| Other | — | |

**User's choice:** docs/notebooks/pre_executed/fixtures/, using only already-seeded Observatory sites.
Tier-2/tier-3 site resolution covered by Django tests (mocked) instead of the notebook.

---

## Claude's Discretion

- Exact management command name (follow `load_telescope_runs`/`fetch_jplsbdb_objects` convention).
- Exact CSV column-to-model-field mapping and date/time parsing strategy for free-text columns.
- Whether `site_needs_review` rows get a distinct counter in the command's summary output.

## Deferred Ideas

None — discussion stayed within Phase 14's scope. Phase 15 discussion was started first this session
and explicitly deferred until after Phase 14 (see "Pre-discussion: sequencing check" above).
