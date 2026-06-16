---
title: Target-linked run submission form with approval queue
trigger_condition: Calendar feature is mature (v1.x complete) and there is a coordination need for ad-hoc multi-PI follow-up campaigns (e.g. a new interstellar object, IAWN campaign)
planted_date: 2026-06-16
context: Explored during Phase 03 UAT session
---

# Seed: Target-linked run submission form with approval queue

## Idea

A web form that lets PIs and community members submit telescope runs directly into FOMO,
tied to a specific Target, with an admin approval queue before runs appear as CalendarEvents.

## Motivation

The `load_telescope_runs` file-based command works well for pre-planned, PI-coordinated programs
(e.g. a Didymos observing season loaded from a schedule file). But when a rare, urgent object
appears (e.g. 4I/Borisov-class interstellar object, IAWN campaign), coordination happens
fast and involves people outside the core team. A web form lowers the barrier to entry
without bypassing oversight.

## Shape of the feature

- **Mandatory:** Target (must exist in FOMO) — anchors the run to a specific object
- **Optional:** Proposal code — required only if the submitter wants to trigger observations through FOMO; otherwise the run is calendar-only
- **Submitter types:**
  - Approved-program PI (committed time) — may bypass approval
  - DDT PI (Director's Discretionary Time, urgent/new program) — requires admin approval
  - Community member / FOMO admin — requires admin approval
- **Approval flow:** Submission creates a pending CalendarEvent (or a pre-event record); admin reviews and approves before it becomes visible on the shared calendar
- **Spam/error control:** Approval step catches accidental duplicates, bad telescope/date combinations, and uncoordinated community submissions

## Open questions when this seed is revisited

- What model stores the pending submission (a separate `RunSubmission` model, or a `CalendarEvent` with a `status=pending` field)?
- How does the admin get notified (email, in-app notification, both)?
- Should approved-program PIs have a self-service approval path (no admin needed) or always go through admin?
- What proposal-code validation is needed — free text, or cross-checked against an LCO/ESO proposal database?
