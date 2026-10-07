---
created: 2026-10-07T02:59:35.000Z
title: "Say in WatchedProposal.attributed_to help text and the runbook that it applies only to newly discovered records"
area: unattended-discovery
severity: cosmetic
files:
  - solsys_code/models.py:793
  - docs/runbooks/telescope_runs_calendar.rst
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

`WatchedProposal.attributed_to` becomes `ObservationRecord.user` only on records the sweep creates (a `get_or_create()` default); it never re-owns existing ones. Intended (developer, 2026-10-06): a new owner applies going forward; past records are fixed by hand in the Django shell. Nothing at the field or in the runbook says so, and the label "Attribute records to" collides with campaign attribution.

Source: `.planning/v2.4-INTENT-REVIEW.md`, Q8c (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Add `help_text`, the same sentence in the runbook's "Adding a proposal" step, and relabel (e.g. "Owner of discovered records").
