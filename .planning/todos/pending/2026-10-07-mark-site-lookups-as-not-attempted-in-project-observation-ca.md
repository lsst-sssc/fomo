---
created: 2026-10-07T02:59:35.000Z
title: "Mark site lookups as 'not attempted' in project_observation_calendar --dry-run output"
area: observation-projector
severity: cosmetic
files:
  - solsys_code/management/commands/project_observation_calendar.py
---

## Problem

A dry-run sweep reports `site_lookups: 0`, which reads as "none needed" rather than "not attempted in a dry run".

Source: `.planning/v2.4-INTENT-REVIEW.md`, Q6 (34-UAT test 2 note) (v2.4 intent review walkthrough, 2026-10-06).

## Solution

Print a clear "not attempted (dry run)" marker for the site-lookup counts in `--dry-run` mode.
