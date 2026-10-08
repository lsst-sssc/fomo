# Phase 39: Calendar Write Access - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-07
**Phase:** 39-Calendar Write Access
**Areas discussed:** Who may write, Anonymous pop-up, Click targets & refusal, WARN-01 header & BS4 tidy

---

## Who may write

| Option | Description | Selected |
|--------|-------------|----------|
| Any logged-in user | Matches `AUTH_STRATEGY='READ_ONLY'` and `tom_targets` create/update precedent; `login_required` around the upstream views | ✓ |
| Staff only | Reuse `StaffRequiredMixin` posture from the approval/attribution queues; non-staff lose write access | |
| Django model permissions | `permission_required` on `tom_calendar.add/change/delete_calendarevent`; finest control but perms must be granted | |

**User's choice:** Any logged-in user

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, same guard for todo URLs | All five write routes guarded; "anonymous = read-only" | ✓ |
| Leave todos unguarded | Only the three event routes change | |

**User's choice:** Yes, same guard

| Option | Description | Selected |
|--------|-------------|----------|
| Delete same as create/update | Any logged-in user may delete, as today | ✓ |
| Delete is staff-only | Destructive action restricted to `is_staff` | |

**User's choice:** Same as create/update
**Notes:** Research shown before the area: Django guidance prefers model permissions over `is_staff` for write gating; TOM's own split is `LoginRequiredMixin` for create/update, a model permission for delete. User chose the simplest rule consistent with `READ_ONLY`.

---

## Anonymous pop-up

| Option | Description | Selected |
|--------|-------------|----------|
| Method-aware guard, same URL | `update-event` GET open to everyone; POST on all five routes guarded; template renders read-only for non-editors | ✓ |
| Separate read-only detail view | New FOMO view + template; `update-event` fully guarded; second template to keep in step | |
| No pop-up for anonymous | Contradicts success criterion 3 and Phase 33 D-14/D-17 | |

**User's choice:** Method-aware guard, same URL

| Option | Description | Selected |
|--------|-------------|----------|
| Plain text, no form controls | Labelled text, no `<form>`, no Save/Delete, todos as read-only list | ✓ |
| Same form, fields disabled | Keep form layout with disabled inputs, controls hidden | |

**User's choice:** Plain text, no form controls

| Option | Description | Selected |
|--------|-------------|----------|
| Login redirect for `create-event` GET | GET and POST both require login; nothing to read on a blank form | ✓ |
| GET open, form read-only | Symmetry with `update-event`, little value | |

**User's choice:** Login redirect

---

## Click targets & refusal

| Option | Description | Selected |
|--------|-------------|----------|
| Remove "+ New Event" and day-cell hx-get entirely | Cell inert; event rows keep their pop-up link | ✓ |
| Show a "Log in to add events" link | Replace the button with a login link | |

**User's choice:** Remove them entirely

| Option | Description | Selected |
|--------|-------------|----------|
| Redirect to login | `login_required` 302; `HTMXRedirectMiddleware` → `HX-Redirect` for htmx | ✓ |
| 403 Forbidden | `Raise403Middleware` adds a message then redirects to login anyway | |

**User's choice:** Redirect to login

| Option | Description | Selected |
|--------|-------------|----------|
| Django TestCase template assertions | Extend `test_calendar_template.py`; runs in pre-commit and CI unit matrix | |
| Template tests + one Playwright test | Same plus a `@tag('functional')` browser test of the anonymous read-only modal | ✓ |

**User's choice:** Template tests + one Playwright test
**Notes:** Research shown: the htmx login-redirect trap (login page swapped into the modal) is already solved in FOMO by `tom_common`'s `HTMXRedirectMiddleware`.

---

## WARN-01 header & BS4 tidy

| Option | Description | Selected |
|--------|-------------|----------|
| Numbered list per block, pinned to 3.1.0 | Every FOMO-only block named, upstream path and version stated; reason paragraphs kept trimmed | ✓ |
| Short pointer to a diff recipe | Header points at `diff -u`; each block gets a marker comment | |

**User's choice:** Numbered list per block, pinned to 3.1.0

| Option | Description | Selected |
|--------|-------------|----------|
| Swap BS4 class names to BS5 | `border-start/end`, `me-2`, `fw-bold`, `var(--bs-white)`; keep `data-url` | ✓ |
| Leave for later | Note as a todo for Phase 41 | |

**User's choice:** Yes, swap to the BS5 names

| Option | Description | Selected |
|--------|-------------|----------|
| Both ledgers | 37.1 WR-05 → fixed; note under 33-REVIEW.md WR-05 | ✓ |
| 37.1 only | Leave 33-REVIEW.md for Phase 42 | |

**User's choice:** Both ledgers

---

## Claude's Discretion

- Where the guard lives (inline in `calendar_urls.py` vs. a small `calendar_views.py` wrapper for the method-aware `update_event`).
- Where the read-only branch lives (`event_form.html` `is_authenticated` branch vs. an included `event_detail.html` partial); context flag vs. `request.user`.
- Runbook wording and placement in `docs/runbooks/telescope_runs_calendar.rst`.
- Home of the Playwright test and whether it also round-trips an editor's save.
- Threat-model shape for the security gate.

## Deferred Ideas

None. Todos reviewed but not folded: the 19 keyword matches (two at 0.9 already belong to Phase 40 WARN-05 and Phase 41 triage; the rest go to TRIAGE-01).
