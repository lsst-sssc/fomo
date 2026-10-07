---
created: 2026-10-07T02:59:35.000Z
title: "Revisit 'a human-confirmed allocation night is not retired' if a real doubled night appears"
area: allocation-projector
severity: minor
files:
  - solsys_code/allocation_projector.py
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

A confirmed allocation night stays when an observation links to it, so the calendar shows that night twice (Phase 35 CR-05). Kept as built on 2026-10-06; no live case known.

Source: `.planning/v2.4-INTENT-REVIEW.md`, Q2 (v2.4 intent review walkthrough, 2026-10-06).

## Solution

If a real doubled night appears, decide whether the observation should retire it after all.
