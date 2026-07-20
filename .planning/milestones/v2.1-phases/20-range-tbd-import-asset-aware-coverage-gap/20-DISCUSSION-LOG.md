# Phase 20: Range/TBD Import & Asset-Aware Coverage Gap - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-10
**Phase:** 20-range-tbd-import-asset-aware-coverage-gap
**Areas discussed:** TBD-row needs-review persistence, Import summary counters, Space-mission gap-page bucketing, Range-parsing edge-case scope

---

## TBD-row "needs review" persistence

| Option | Description | Selected |
|--------|-------------|----------|
| Add original_obs_date_raw text field | New field storing the raw `Obs. Date` cell verbatim whenever the row lands as TBD | ✓ |
| Boolean flag only, no raw text | Mirror `site_needs_review`; raw text only in the import log, not stored | |
| No new field — rely on existing TBD badge | No new persisted signal beyond the existing TBD badge | |

**User's choice:** Add `original_obs_date_raw` text field.
**Notes:** None.

| Option | Description | Selected |
|--------|-------------|----------|
| Treat both as plain TBD (recommended) | Both blank and `2025-12-?` collapse to `window_start=window_end=None`; the raw-text field naturally distinguishes them | ✓ |
| Add a distinct sub-status | New field/choice for "no date info at all" vs. "month known, day unknown" | |

**User's choice:** Treat both as plain TBD.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — add window_needs_review boolean | Mirrors `site_needs_review`; supports future filtering | ✓ |
| No — non-empty raw text is the signal | Skip the extra boolean, rely on `original_obs_date_raw` non-blank | |

**User's choice:** Add `window_needs_review` boolean.

| Option | Description | Selected |
|--------|-------------|----------|
| TBD rows only (recommended) | Matches IMPORT-02's ask directly; parsed rows already have structured data | ✓ |
| Populate for every row, always | General audit trail regardless of parse outcome | |

**User's choice:** TBD rows only.

---

## Import summary counters

| Option | Description | Selected |
|--------|-------------|----------|
| Add both range_imported and window_needs_review counts | Two new counters, full granularity | |
| Add window_needs_review only | Range rows count as ordinary created/updated | ✓ |
| No new counters — fold into existing ones | Only a per-row log line changes | |

**User's choice:** Add `window_needs_review` only.

| Option | Description | Selected |
|--------|-------------|----------|
| Import as TBD + needs-review (recommended) | "Never silently dropped" — any unparseable text becomes a TBD row | ✓ |
| Still skip and log (current behavior for garbage) | Only the 4 enumerated shapes get special handling | |

**User's choice:** Import as TBD + needs-review.
**Notes:** This supersedes the Phase 18 spike's narrower suggestion ("raise only on truly malformed values outside the enumerated shapes") — the roadmap criterion is the locked requirement.

| Option | Description | Selected |
|--------|-------------|----------|
| Surface as a tooltip on the TBD badge (recommended) | `render_window_start` gains a `title` attribute with `original_obs_date_raw` | ✓ |
| Store only, defer UI display | No UI surface-area growth this phase | |

**User's choice:** Surface as a tooltip on the TBD badge.

---

## Space-mission gap-page bucketing

| Option | Description | Selected |
|--------|-------------|----------|
| New distinct bucket (recommended) | A 3rd list, `pending_narrowing_runs`, with its own alert message | ✓ |
| Fold into undated_runs | Reuse the existing bucket/messaging for both TBD and not-yet-narrowed space-mission runs | |

**User's choice:** New distinct bucket (`pending_narrowing_runs`).

| Option | Description | Selected |
|--------|-------------|----------|
| Manual/re-import edit only (matches current model) | No automated narrowing mechanism; a staff edit or re-import sets window_start==window_end | ✓ |
| Something else / flag a gap | A different expectation (e.g. background job) | |

**User's choice:** Manual/re-import edit only.

---

## Range-parsing edge-case scope

| Option | Description | Selected |
|--------|-------------|----------|
| Same-month only (recommended, matches spike evidence) | Only the exact worked shape from the spike | |
| Detect and handle month rollover | Roll to next month (and year) if second day < first day-of-month | ✓ |

**User's choice:** Detect and handle month rollover.

| Option | Description | Selected |
|--------|-------------|----------|
| Strictly " to " only (recommended, matches spike evidence) | Only the exact separator confirmed in the real snapshot | |
| Also accept en-dash/hyphen-separated ranges | Broaden the regex proactively | ✓ |

**User's choice:** Also accept en-dash/hyphen-separated ranges.

| Option | Description | Selected |
|--------|-------------|----------|
| parse_obs_window() itself never raises for Obs. Date (recommended) | Documented contract changes; always returns a tuple | ✓ |
| parse_obs_window() still raises; import_campaign_csv catches it | Keeps the utility function's contract narrow | |

**User's choice:** `parse_obs_window()` itself never raises for `Obs. Date`.

| Option | Description | Selected |
|--------|-------------|----------|
| Reserved for non-date failures only (recommended) | skipped_count only for blank Telescope/Instrument and natural-key collisions | ✓ |
| Still counts some date cases | A specific date-related case should still skip | |

**User's choice:** Reserved for non-date failures only.

---

## Claude's Discretion

- Exact regex/parsing implementation for month/year rollover and en-dash/hyphen separator variants.
- Whether to extract a shared `is_space_mission(site)` helper vs. keeping the `Observatory.observations_type == SATELLITE_OBSTYPE` check inline at each call site.
- `original_obs_date_raw` field type (`CharField` vs `TextField`) and max length; migration mechanics.
- Exact wording of the new `pending_narrowing_runs` alert block in `campaignrun_gap_analysis.html` beyond the substance captured in the decision.

## Deferred Ideas

None — discussion stayed within Phase 20's IMPORT-01/02/ASSET-01/02 scope. Site disambiguation and VIEW-05 remain Phase 21 (already scoped there by ROADMAP.md).

**Reviewed but not folded:** `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` (weak match, score 0.2) — same assessment as Phase 19: unrelated to `CampaignRun`/window schema/coverage-gap work.
