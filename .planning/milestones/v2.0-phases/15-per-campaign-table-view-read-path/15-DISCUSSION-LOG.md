# Phase 15: Per-Campaign Table View (Read Path) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-03
**Phase:** 15-Per-Campaign Table View (Read Path)
**Areas discussed:** Campaign discovery & navigation, Approval-status visibility, Table columns
sort & paging, Staff-only contact gating

---

## Todo cross-reference

Two pending todos scored low partial matches against this phase; both reviewed and not folded
(topic mismatch — both are `calendar_utils.py` cleanup, unrelated to the campaign table view).

| Todo | Score | Reason |
|------|-------|--------|
| Rename calendar_utils.py private helpers to reflect shared-module API | 0.4 | keyword overlap only |
| Extract site/telescope mapping and instrument extraction into own module | 0.2 | already resolved per Phase 14 context; unrelated anyway |

---

## Campaign discovery & navigation

### How should a Target's detail page find "its" campaign?

| Option | Description | Selected |
|--------|-------------|----------|
| Via TargetList membership | TargetList.objects.filter(targets=target, campaign_runs__isnull=False) | ✓ |
| Via CampaignRun.target directly | Simpler, but misses rows where the optional target FK wasn't set | |
| You decide | | |

**User's choice:** Via TargetList membership.

### Multiple qualifying campaigns for one Target

| Option | Description | Selected |
|--------|-------------|----------|
| List all as separate buttons/links | One button per matching campaign | ✓ |
| Single button to a chooser | Intermediate page listing matches | |
| You decide | | |

**User's choice:** List all as separate buttons/links.

### Navbar "Campaigns" entry destination

| Option | Description | Selected |
|--------|-------------|----------|
| New dedicated campaigns list page | Lists TargetLists with >=1 CampaignRun | ✓ |
| Reuse tom_targets:targetgrouping | Existing TOM view; login-required, lists all TargetLists | |
| You decide | | |

**User's choice:** New dedicated campaigns list page.

### Access level for campaign pages

| Option | Description | Selected |
|--------|-------------|----------|
| Open to anonymous | Matches VIEW-03 framing and FOMO's OPEN/READ_ONLY convention | ✓ |
| Login required | | |

**User's choice:** Open to anonymous (Recommended).

---

## Approval-status visibility

### Filter non-approved rows for non-staff now?

| Option | Description | Selected |
|--------|-------------|----------|
| Filter to approved-only for non-staff | Forward-compatible with Phase 16 | |
| Show all rows regardless of approval_status | No visible effect yet since all rows are bootstrap-approved | ✓ |
| You decide | | |

**User's choice:** Show all rows regardless of approval_status. Filter deferred to Phase 16.

### Should staff see rejected rows in the main table?

| Option | Description | Selected |
|--------|-------------|----------|
| Staff sees rejected rows too, in the main table | | ✓ |
| Rejected rows hidden from the table entirely | | |
| You decide | | |

**User's choice:** Staff sees rejected rows too, in the main table.

### Default run_status filter state

| Option | Description | Selected |
|--------|-------------|----------|
| Show everything by default | | ✓ |
| Default-hide dead-end statuses | cancelled/not_awarded/weather_tech_failure hidden until opted-in | |
| You decide | | |

**User's choice:** Show everything by default.

### approval_status visual treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Plain column, no special styling | | |
| Visually distinct badge/highlight | | ✓ |
| You decide | | |

**User's choice:** Visually distinct badge/highlight.

---

## Table columns, sort & paging

### Column set

| Option | Description | Selected |
|--------|-------------|----------|
| Spreadsheet-parity: most fields as columns | Mirror the real 3I sheet closely | ✓ |
| Curated subset + detail expansion | At-a-glance columns + detail link for verbose fields | |
| You decide | | |

**User's choice:** Spreadsheet-parity: most fields as columns.

### Default sort order

| Option | Description | Selected |
|--------|-------------|----------|
| obs_date, most recent first | | ✓ |
| obs_date, chronological (oldest first) | | |
| You decide | | |

**User's choice:** obs_date, most recent first.

### Pagination page size

| Option | Description | Selected |
|--------|-------------|----------|
| 25 rows/page | | ✓ |
| 50 rows/page | | |
| You decide | | |

**User's choice:** 25 rows/page.

### run_status filter cardinality

| Option | Description | Selected |
|--------|-------------|----------|
| Multi-select run_status | | ✓ |
| Single-value run_status dropdown | | |
| You decide | | |

**User's choice:** Multi-select run_status.

---

## Staff-only contact gating

### Staff definition

| Option | Description | Selected |
|--------|-------------|----------|
| request.user.is_staff | Django built-in, no new permission plumbing | |
| A specific permission or group | Finer-grained, more plumbing | |
| You decide | | ✓ |

**User's choice:** You decide. Claude's discretion: use `request.user.is_staff`.

### Contact column shape for non-staff

| Option | Description | Selected |
|--------|-------------|----------|
| Omit the columns entirely | Gate at view layer, defense in depth | ✓ |
| Show columns with masked/blank placeholder | | |
| You decide | | |

**User's choice:** Omit the columns entirely.

### Contact/reach-out path for open_to_collaboration runs

| Option | Description | Selected |
|--------|-------------|----------|
| No contact path in Phase 15 | Deferred to VIEW-05 | ✓ |
| Generic "contact FOMO admins" link | | |
| You decide | | |

**User's choice:** No contact path in Phase 15.

---

## Claude's Discretion

- Staff check mechanism: `request.user.is_staff`.
- Exact `django-tables2`/`django-filter` implementation details (Table subclass, FilterSet wiring,
  template choice).
- Exact URL names/paths for the new campaigns list view and per-campaign table view.
- Exact badge/styling mechanism for the approval_status treatment (D-08).

## Deferred Ideas

None — discussion stayed within Phase 15's scope. VIEW-05 (submitter contact opt-in) and the
Phase 16 approval-status filter are already tracked in REQUIREMENTS.md/ROADMAP.md, not newly
deferred here.
