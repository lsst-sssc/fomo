# Phase 39: Calendar Write Access - Context

**Gathered:** 2026-10-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the public `/calendar/` read-only for anyone not logged in. An anonymous visitor can neither
create, change nor delete a `CalendarEvent` or an `EventTodo` through any of the five write routes in
`solsys_code/calendar_urls.py` (`create-event`, `update-event`, `delete-event`, `create-todo`,
`update-todo`), and the month view offers them no control that would try — yet they can still open an
event's pop-up and read it, including its attributed-run block and observation-series line. A signed-in
user keeps today's month-view workflow exactly (create from the "+ New Event" button or a day cell,
edit and delete from the pop-up, add and tick todos). `src/templates/tom_calendar/partials/event_form.html`'s
header comment states truthfully which blocks differ from tomtoolkit 3.1.0's upstream partial.

Requirements: ACCESS-01, ACCESS-02, WARN-01. This is the milestone's security gate
(`security_enforcement` on): the plan's threat model covers anonymous writes through every calendar URL.

Verified facts the phase starts from (Phase 38's `38-OVERRIDE-COMPARISON.md`, re-checked in the
installed venv 2026-10-07): tomtoolkit 3.1.0's `tom_calendar.views` (`create_event`, `update_event`,
`delete_event`, `create_todo`, `update_todo`) carry **no login guard** — an anonymous `POST` saves — and
the whole `tom_calendar` package is byte-identical between 3.0.1 and 3.1.0. The guard is therefore
FOMO's to add, wrapped around the upstream view functions at FOMO's URL layer; the vendored package is
never edited.

Not in this phase: notebook isolation and the attribution page (Phase 40), todo triage (Phase 41),
re-verification (Phase 42), any change to who may read the calendar, model permissions or per-event
ownership, a run-detail view.

</domain>

<decisions>
## Implementation Decisions

### Who may write
- **D-01:** **Any logged-in user** may create, update and delete calendar events — the same set of
  accounts that can today, minus anonymous. Matches `AUTH_STRATEGY = 'READ_ONLY'` (anonymous reads,
  authenticated writes) and `tom_targets`' `LoginRequiredMixin` on create/update. Not staff-only, not
  Django model permissions. — **Reversibility:** reversible — tightening later is a one-decorator change
  (`user_passes_test(is_staff)` or `permission_required`) at the same wrapping point.
- **D-02:** **All five write routes get the same guard**, including `create-todo` and `update-todo`
  (ACCESS-01 names only the three event routes; the roadmap left the todo routes to this discussion).
  Rule for the whole namespace: anonymous = read-only.
- **D-03:** **Delete is not gated more tightly** than create/update — any logged-in user may delete,
  as today. No model permission, no staff check.

### The anonymous pop-up (ACCESS-02 read path)
- **D-04:** **Method-aware guard on the same URL.** `GET update-event/<id>/` stays open to everyone —
  it is the pop-up, and success criterion 3 plus Phase 33 D-14/D-17 require anonymous visitors to read
  the attributed-run block there. `POST` on all five write routes requires login. No separate detail
  view, no second URL: `calendar.html`'s event `hx-get` targets and the Bootstrap 5 modal JS stay as they
  are. — **Reversibility:** reversible — a dedicated read-only view could be added later behind the same
  template branch.
- **D-05:** For a non-editor (anonymous), the pop-up renders as a **plain-text detail card**: title,
  start/end, description, URL (with the existing `is_web_url`-gated "View" link), target list, user,
  proposal, telescope, instrument as labelled text — **no `<form>`, no Save / Save-and-edit / Delete
  buttons, no todo inputs**; todos shown as a read-only list (description, done/not done). The
  observation-series line and the attributed-campaign-run block render unchanged. Nothing on the card
  looks editable. Not the "same form with disabled inputs" option.
- **D-06:** **`GET create-event/` requires login too** (there is nothing for an anonymous visitor to
  read on a blank form); an anonymous GET or POST there is redirected to login. Only `update-event` GET
  is the open read path.

### Click targets and refusal (ACCESS-02 write surface)
- **D-07:** For an anonymous visitor the month view **removes the "+ New Event" button and the day-cell
  `hx-get` to `calendar:create-event` entirely** — the cell is inert; no "log in to add events" link.
  Event entries keep their `hx-get` to `calendar:update-event` (the read path, D-04). A logged-in user
  sees both click targets exactly as today.
- **D-08:** An anonymous `POST` that does reach a write URL (script, stale tab) is **redirected to
  login** (`login_required` semantics: 302 to `LOGIN_URL` with `?next=`), never 403. For an htmx request
  `tom_common`'s `HTMXRedirectMiddleware` turns that 302 into an `HX-Redirect` full-page navigation, so
  a stale-tab save lands on the login page rather than swapping it into the modal. Tests assert the
  redirect and that the `CalendarEvent`/`EventTodo` count and the targeted row's fields are unchanged.
- **D-09:** ACCESS-02 is proven by **Django `TestCase` template assertions plus one
  `@tag('functional')` Playwright test**: anonymous `GET /calendar/` contains no `create-event` URL and
  the anonymous pop-up contains no Save/Delete/todo form while a `force_login`ed user's does
  (`test_calendar_template.py` already has the modal + `force_login` pattern); the browser test clicks an
  event as an anonymous visitor and sees the read-only modal open through the Bootstrap 5 modal API.
  The functional test runs only in CI's `functional-tests` job, like the existing Playwright tests.

### WARN-01 header, Bootstrap 4 tidy, ledgers
- **D-10:** `event_form.html`'s header comment is rewritten as a **numbered list, one item per FOMO-only
  block, pinned to tomtoolkit 3.1.0** with the upstream path
  (`tom_calendar/templates/tom_calendar/partials/event_form.html`). Items: the four already catalogued in
  `38-OVERRIDE-COMPARISON.md` — (1) header + `{% load … attribution_display_extras calendar_display_extras %}`,
  (2) `is_web_url`-gated URL "View" link, (3) plain `<button>` Save/Save-and-edit/Delete markup,
  (4) the observation-series, attributed-campaign-run and staff-only candidate blocks after `</form>` —
  plus whatever this phase adds (the non-editor read-only branch, D-05). The sentence "starts as an exact
  copy of the upstream partial with one new block" goes. The existing reason paragraphs (D-08 Phase 27,
  T-27-20, G-37.1-1-allocurl) stay, trimmed. Success criterion 4: a `diff -u` against the 3.1.0 file must
  match the list.
- **D-11:** **Tidy the leftover Bootstrap 4 utility class names in `calendar.html`** to the Bootstrap 5
  names upstream 3.1.0 uses: `border-left`/`border-right` → `border-start`/`border-end`, `mr-2`/`mr-3` →
  `me-2`/`me-3`, `font-weight-bold` → `fw-bold`, `var(--white)` → `var(--bs-white)` (and any sibling the
  diff shows). **Keep `data-url`** — upstream's `calendar_page.html` reads `cal.dataset.url` and
  `test_calendar_template.py` asserts it (38-OVERRIDE-COMPARISON observation 2); do not switch to
  upstream's `data-bs-url`.
- **D-12:** **Both review ledgers record the fix**: the WR-05 row in
  `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md`
  → `fixed` (WARN-01), and a "fixed in Phase 39 (ACCESS-01)" note under WR-05 in
  `.planning/milestones/v2.4-phases/33-series-identity-reconciler-inversion/33-REVIEW.md`.

### Claude's Discretion
- **Where the guard lives:** a thin FOMO wrapping layer that delegates to the upstream view functions —
  either inline in `solsys_code/calendar_urls.py` (`login_required(create_event)` etc.) or a small
  `solsys_code/calendar_views.py` holding a method-aware wrapper for `update_event` (GET passes through,
  POST requires login). Planner picks; the wrapper must not duplicate upstream view logic.
- **Where the read-only branch lives:** inside `event_form.html` behind `request.user.is_authenticated`
  (the template already gates on `request.user.is_staff` for the candidate hint, 27-07 convention), or a
  separate included partial such as `event_detail.html`. Either way the series/attribution blocks are
  rendered once, not duplicated, and the WARN-01 header lists the result.
- **Context flag name** the view passes (e.g. `can_edit`) versus reading `request.user` directly in the
  template.
- **Runbook wording:** `docs/runbooks/telescope_runs_calendar.rst`'s pop-up section ("Clicking a calendar
  entry opens a pop-up …", ~line 2567) and any line that says anyone can add events gain a sentence that
  the calendar is read-only when not logged in and the pop-up is a read-only card for a visitor; exact
  placement is the executor's.
- **Where the Playwright test goes** (`test_bootstrap5_rendering.py`'s `@tag('functional')` pattern or
  a new functional test module) and whether it also round-trips a logged-in editor's save.
- **Threat-model shape** for the security gate: anonymous POST per route, GET on create-event, CSRF
  (upstream forms already carry `{% csrf_token %}`), htmx redirect path, and the shadowed
  `tom_common.urls` `calendar` namespace (FOMO's include comes first in `src/fomo/urls.py`, so the
  unguarded upstream routes are unreachable — a test hitting the real `/calendar/create/` path proves it).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` §"Phase 39: Calendar Write Access" — goal, scope note (three WR-05 findings
  share an ID: Phase 33's is ACCESS-01, Phase 37.1's is WARN-01, and `solsys_code/views.py`'s
  `fomo_render_calendar` comment cites Phase 34's, unrelated), paired-docs note, four success criteria.
- `.planning/REQUIREMENTS.md` §"Calendar write access (ACCESS)" and §"Review warnings (WARN)" WARN-01.
- `.planning/STATE.md` §"Operator Next Steps" — the two questions this discussion answered (who may
  write; todo URLs).

### What upstream has and how FOMO differs (read these, not memory)
- `.planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md` — the per-file diff of
  `calendar_urls.py`, `calendar.html`, `event_form.html` against tomtoolkit 3.1.0; the exact list of
  FOMO-only blocks D-10 itemises; the Bootstrap 4 class names D-11 swaps; the `data-url` note.
- Installed upstream (dev venv `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/`):
  `tom_calendar/views.py` (the five unguarded view functions and their htmx response shapes:
  `HX-Retarget`/`HX-Reswap`, `calClose`/`calRefresh` triggers), `tom_calendar/urls.py`,
  `tom_calendar/templates/tom_calendar/partials/event_form.html`, `partials/calendar.html`,
  `partials/todos.html`, `calendar_page.html` (the modal and `cal.dataset.url`), and
  `tom_common/middleware.py` (`HTMXRedirectMiddleware`, `Raise403Middleware`, `AuthStrategyMiddleware`).

### The findings being closed
- `.planning/milestones/v2.4-phases/33-series-identity-reconciler-inversion/33-REVIEW.md` §WR-05 — the
  original finding and its suggested `login_required` wrapping at FOMO's URL layer (ACCESS-01).
- `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md`
  — WR-05 row (header comment) to mark `fixed` (D-12).

### Repository conventions
- `CLAUDE.md` §"Testing" (`@tag('functional')` split; Django runner only), §"Conventions"
  (paired docs: `docs/runbooks/telescope_runs_calendar.rst` is the paired doc for the calendar
  templates — no notebook pairs with them; `NonSiderealTargetFactory` for any Target fixture;
  `pre-commit run ruff` / `ruff-format`).
- `docs/runbooks/telescope_runs_calendar.rst` — the pop-up section to update.
- `src/fomo/settings.py` — `AUTH_STRATEGY = 'READ_ONLY'`, `LOGIN_URL = '/accounts/login/'`,
  `TOMTOOLKIT_MIDDLEWARE` (includes the htmx redirect and 403 middleware).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/calendar_urls.py` — the six-route FOMO URL conf (`app_name = 'calendar'`); root →
  `solsys_code.views.fomo_render_calendar`, the other five → upstream functions imported directly. The
  wrap goes here (or in a sibling module it imports).
- `solsys_code/mixins.py:StaffRequiredMixin` — `user_passes_test` via `method_decorator`; the
  function-view analogue for D-01 is `django.contrib.auth.decorators.login_required`.
- `src/templates/tom_calendar/partials/event_form.html:224` — `{% elif … and request.user.is_staff %}`:
  the template already reads `request.user`, so an `is_authenticated` branch needs no new context.
- `solsys_code/tests/test_calendar_template.py` — modal rendering tests with `staff_user` +
  `self.client.force_login(...)` (lines ~431-560, ~789-860): the pattern for D-09's template assertions
  and for an editor-unchanged test.
- `solsys_code/tests/test_bootstrap5_rendering.py` — `@tag('functional')` `StaticLiveServerTestCase` +
  Playwright; the home for or model of D-09's browser test.
- `tom_common.middleware.HTMXRedirectMiddleware` — already in `TOMTOOLKIT_MIDDLEWARE`; makes D-08's
  302 safe for htmx with no FOMO code.

### Established Patterns
- Overrides of `tom_calendar` live at FOMO's layer (URL conf + two template partials); upstream code is
  never edited. The guard follows the same rule (33-REVIEW WR-05's suggested fix).
- `AUTH_STRATEGY = 'READ_ONLY'`: anonymous read, authenticated write, across TOM — D-01 aligns the
  calendar with the rest of the site.
- Template-side gating on `request.user.*` (27-07 staff hint; `campaign_list.html` nav banner) rather
  than a separate context flag — D-05's branch may follow it.
- Click targets in `calendar.html`: `+ New Event` button (`hx-get calendar:create-event`), day cell
  (`hx-get calendar:create-event?date=`), event rows (`hx-get calendar:update-event`); the inner event
  `<div>` stops click propagation to the cell. D-07 removes the first two for anonymous users and must
  keep the event rows clickable when the cell has no `hx-get`.
- `event_form.html` is rendered by upstream `create_event`/`update_event` with `form`, `event`,
  `action` in context — both the GET and the invalid-POST branches — so a template-only read-only
  branch is reachable without a view override of the rendering itself.

### Integration Points
- `src/fomo/urls.py:30` — `path('calendar/', include('solsys_code.calendar_urls', namespace='calendar'))`
  precedes `tom_common.urls`, which mounts upstream `tom_calendar.urls` under the same namespace
  (`urls.W005`); FOMO's routes shadow upstream's for every path.
- `src/templates/tom_calendar/partials/calendar.html` — D-07 click targets, D-11 class names.
- `src/templates/tom_calendar/partials/event_form.html` — D-05 read-only branch, D-10 header; upstream
  `partials/todos.html` is *not* overridden today — D-05's read-only todo list either branches around
  the include or adds a third override (and the header/comparison must then list it).
- `docs/runbooks/telescope_runs_calendar.rst` — paired doc (pop-up section).
- The two review ledgers in `.planning/milestones/v2.4-phases/` (D-12).

</code_context>

<specifics>
## Specific Ideas

- The anonymous pop-up should read as a detail card, not a disabled form — "nothing looks editable".
- A visitor should not be told how to get write access from the month view; the write controls simply
  are not there (D-07).
- The header comment's block list is what success criterion 4 is checked against; keep it literal
  (block names and their position in the file), not prose.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
- "Run pre-executed demo notebooks against a scratch DB copy" (2026-10-02, score 0.9) — already
  WARN-05 in Phase 40.
- "Isolate the campaign table query-count test from the shared file cache" (2026-10-07, score 0.9) —
  a test fix for Phase 41's triage.
- The remaining 17 keyword matches (score 0.6: attribution banner cache, dismiss-action guard,
  reconciler sun-event skip, bulk LCO fetch, `load_telescope_runs` comment lines, projector window
  retention, `run_status` value, stale `RUN:` container, `telescope_class` explainer, gap-analysis date
  control, site restriction in titles, campaign table row links, dry-run site lookups, discovery tick
  summary, confirmed-night retirement, `attributed_to` help text, proposal-level unused figure) — none
  concerns calendar write access; all go to Phase 41 triage (TRIAGE-01).

</deferred>

---

*Phase: 39-Calendar Write Access*
*Context gathered: 2026-10-07*
