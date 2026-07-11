# Phase 21: Site Disambiguation & Submitter Contact Opt-In - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-11
**Phase:** 21-site-disambiguation-submitter-contact-opt-in
**Areas discussed:** Fuzzy-match scope & trigger, Staff resolution UI pattern, SITE-03 clobber-fix mechanism, VIEW-05 opt-in placement & scope

---

## Fuzzy-match candidate pool

| Option | Description | Selected |
|--------|-------------|----------|
| Local Observatory table only (recommended) | Match against existing local rows (~8 today); matches what Phase 18's spike tested; no new API dependency | |
| Widen to the live MPC Obscodes list | Fetch/cache the full MPC list and fuzzy-match against that too; more candidates, adds API dependency/latency | ✓ |

**User's choice:** Widen to the live MPC Obscodes list.
**Notes:** Acts directly on Phase 18's spike finding that the local table (8 rows) is too narrow a candidate pool to meaningfully fuzzy-match arbitrary external codes.

---

## MPC list fetch/cache strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse MPCObscodeFetcher, cache locally (recommended) | Fetch once via existing fetcher, cache (DB or Django cache), refresh periodically | ✓ |
| Fetch live on every approval-queue page load | Always fresh, but adds latency/failure risk to a frequently-loaded staff page | |
| Something else | User-described alternative | |

**User's choice:** Reuse MPCObscodeFetcher, cache locally.
**Notes:** Exact cache mechanism (dedicated table vs. Django cache vs. periodic sync command) left to Claude's discretion.

---

## Staff resolution UI pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Inline dropdown + free-text, submitted with approve/reject POST (recommended) | Site column becomes a <select> + free-text fallback in the existing row; rides along with the existing decision POST, no new endpoint | ✓ |
| Separate small AJAX endpoint | Decoupled "resolve site" action independent of approve/reject | |
| Modal/detail view per row | Clicking the cell opens a modal or detail page | |

**User's choice:** Inline dropdown + free-text, submitted with the existing approve/reject POST.

---

## SITE-02 create-new-Observatory flow

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing CreateObservatory form (recommended) | Link/redirect to the existing MPC-obscode-driven creation view, then return to the queue | ✓ |
| Lightweight inline create in the queue row | Minimal inline form duplicating some CreateObservatory validation | |

**User's choice:** Reuse existing CreateObservatory form.

---

## SITE-03 clobber-fix mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Skip resolve_site() whenever site is already set (recommended) | Trust an already-resolved site (site is not None) on approve; only unresolved runs get auto-resolved; no new field | ✓ |
| New explicit 'site_manually_resolved' flag | Add a boolean field to distinguish manual resolution from auto-resolution; new migration | |

**User's choice:** Skip resolve_site() whenever site is already set.

---

## VIEW-05 opt-in placement & scope

| Option | Description | Selected |
|--------|-------------|----------|
| Next to contact_person/contact_email, fixed at submit time (recommended) | Single checkbox after contact fields, default unchecked; no submitter self-service edit exists today, so no edit-after-submission mechanism built | ✓ |
| Same placement, but also staff-editable in the approval queue | Adds another editable field to the queue UI | |

**User's choice:** Next to contact_person/contact_email, fixed at submit time.

---

## Claude's Discretion

- Exact MPC-list cache mechanism (dedicated table, Django cache, or periodic sync command)
- Fuzzy-match candidate count/threshold shown in the dropdown
- VIEW-05 checkbox field name, verbose label, help text (mirror `open_to_collaboration`'s style)
- Whether the per-row "opted in" state needs its own visible indicator in the approval queue

## Deferred Ideas

None — discussion stayed within phase scope. Two weak-keyword-match todos (site/telescope
extraction into own module; calendar_utils.py helper renaming) were reviewed and confirmed
not relevant to this phase's scope (both are LCO/Gemini calendar-sync concerns, not
campaign-approval-queue site resolution) — consistent with their rejection during Phase 13
and Phase 18's discussions.
