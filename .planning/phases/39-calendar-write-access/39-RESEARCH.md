# Phase 39: Calendar Write Access - Research

**Researched:** 2026-10-08
**Domain:** Django function-view access control around vendored `tom_calendar` views; Django template gating; htmx/Bootstrap 5 modal behaviour
**Confidence:** HIGH (every claim about upstream behaviour was checked by reading the installed 3.1.0 source and, for the guard, by running a prototype against a real test database)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Who may write
- **D-01:** **Any logged-in user** may create, update and delete calendar events — the same set of accounts that can today, minus anonymous. Matches `AUTH_STRATEGY = 'READ_ONLY'` (anonymous reads, authenticated writes) and `tom_targets`' `LoginRequiredMixin` on create/update. Not staff-only, not Django model permissions. — **Reversibility:** reversible — tightening later is a one-decorator change (`user_passes_test(is_staff)` or `permission_required`) at the same wrapping point.
- **D-02:** **All five write routes get the same guard**, including `create-todo` and `update-todo` (ACCESS-01 names only the three event routes; the roadmap left the todo routes to this discussion). Rule for the whole namespace: anonymous = read-only.
- **D-03:** **Delete is not gated more tightly** than create/update — any logged-in user may delete, as today. No model permission, no staff check.

#### The anonymous pop-up (ACCESS-02 read path)
- **D-04:** **Method-aware guard on the same URL.** `GET update-event/<id>/` stays open to everyone — it is the pop-up, and success criterion 3 plus Phase 33 D-14/D-17 require anonymous visitors to read the attributed-run block there. `POST` on all five write routes requires login. No separate detail view, no second URL: `calendar.html`'s event `hx-get` targets and the Bootstrap 5 modal JS stay as they are. — **Reversibility:** reversible — a dedicated read-only view could be added later behind the same template branch.
- **D-05:** For a non-editor (anonymous), the pop-up renders as a **plain-text detail card**: title, start/end, description, URL (with the existing `is_web_url`-gated "View" link), target list, user, proposal, telescope, instrument as labelled text — **no `<form>`, no Save / Save-and-edit / Delete buttons, no todo inputs**; todos shown as a read-only list (description, done/not done). The observation-series line and the attributed-campaign-run block render unchanged. Nothing on the card looks editable. Not the "same form with disabled inputs" option.
- **D-06:** **`GET create-event/` requires login too** (there is nothing for an anonymous visitor to read on a blank form); an anonymous GET or POST there is redirected to login. Only `update-event` GET is the open read path.

#### Click targets and refusal (ACCESS-02 write surface)
- **D-07:** For an anonymous visitor the month view **removes the "+ New Event" button and the day-cell `hx-get` to `calendar:create-event` entirely** — the cell is inert; no "log in to add events" link. Event entries keep their `hx-get` to `calendar:update-event` (the read path, D-04). A logged-in user sees both click targets exactly as today.
- **D-08:** An anonymous `POST` that does reach a write URL (script, stale tab) is **redirected to login** (`login_required` semantics: 302 to `LOGIN_URL` with `?next=`), never 403. For an htmx request `tom_common`'s `HTMXRedirectMiddleware` turns that 302 into an `HX-Redirect` full-page navigation, so a stale-tab save lands on the login page rather than swapping it into the modal. Tests assert the redirect and that the `CalendarEvent`/`EventTodo` count and the targeted row's fields are unchanged.
- **D-09:** ACCESS-02 is proven by **Django `TestCase` template assertions plus one `@tag('functional')` Playwright test**: anonymous `GET /calendar/` contains no `create-event` URL and the anonymous pop-up contains no Save/Delete/todo form while a `force_login`ed user's does (`test_calendar_template.py` already has the modal + `force_login` pattern); the browser test clicks an event as an anonymous visitor and sees the read-only modal open through the Bootstrap 5 modal API. The functional test runs only in CI's `functional-tests` job, like the existing Playwright tests.

#### WARN-01 header, Bootstrap 4 tidy, ledgers
- **D-10:** `event_form.html`'s header comment is rewritten as a **numbered list, one item per FOMO-only block, pinned to tomtoolkit 3.1.0** with the upstream path (`tom_calendar/templates/tom_calendar/partials/event_form.html`). Items: the four already catalogued in `38-OVERRIDE-COMPARISON.md` — (1) header + `{% load … attribution_display_extras calendar_display_extras %}`, (2) `is_web_url`-gated URL "View" link, (3) plain `<button>` Save/Save-and-edit/Delete markup, (4) the observation-series, attributed-campaign-run and staff-only candidate blocks after `</form>` — plus whatever this phase adds (the non-editor read-only branch, D-05). The sentence "starts as an exact copy of the upstream partial with one new block" goes. The existing reason paragraphs (D-08 Phase 27, T-27-20, G-37.1-1-allocurl) stay, trimmed. Success criterion 4: a `diff -u` against the 3.1.0 file must match the list.
- **D-11:** **Tidy the leftover Bootstrap 4 utility class names in `calendar.html`** to the Bootstrap 5 names upstream 3.1.0 uses: `border-left`/`border-right` → `border-start`/`border-end`, `mr-2`/`mr-3` → `me-2`/`me-3`, `font-weight-bold` → `fw-bold`, `var(--white)` → `var(--bs-white)` (and any sibling the diff shows). **Keep `data-url`** — upstream's `calendar_page.html` reads `cal.dataset.url` and `test_calendar_template.py` asserts it (38-OVERRIDE-COMPARISON observation 2); do not switch to upstream's `data-bs-url`.
- **D-12:** **Both review ledgers record the fix**: the WR-05 row in `.planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md` → `fixed` (WARN-01), and a "fixed in Phase 39 (ACCESS-01)" note under WR-05 in `.planning/milestones/v2.4-phases/33-series-identity-reconciler-inversion/33-REVIEW.md`.

### Claude's Discretion
- **Where the guard lives:** a thin FOMO wrapping layer that delegates to the upstream view functions — either inline in `solsys_code/calendar_urls.py` (`login_required(create_event)` etc.) or a small `solsys_code/calendar_views.py` holding a method-aware wrapper for `update_event` (GET passes through, POST requires login). Planner picks; the wrapper must not duplicate upstream view logic.
- **Where the read-only branch lives:** inside `event_form.html` behind `request.user.is_authenticated` (the template already gates on `request.user.is_staff` for the candidate hint, 27-07 convention), or a separate included partial such as `event_detail.html`. Either way the series/attribution blocks are rendered once, not duplicated, and the WARN-01 header lists the result.
- **Context flag name** the view passes (e.g. `can_edit`) versus reading `request.user` directly in the template.
- **Runbook wording:** `docs/runbooks/telescope_runs_calendar.rst`'s pop-up section ("Clicking a calendar entry opens a pop-up …", ~line 2567) and any line that says anyone can add events gain a sentence that the calendar is read-only when not logged in and the pop-up is a read-only card for a visitor; exact placement is the executor's.
- **Where the Playwright test goes** (`test_bootstrap5_rendering.py`'s `@tag('functional')` pattern or a new functional test module) and whether it also round-trips a logged-in editor's save.
- **Threat-model shape** for the security gate: anonymous POST per route, GET on create-event, CSRF (upstream forms already carry `{% csrf_token %}`), htmx redirect path, and the shadowed `tom_common.urls` `calendar` namespace (FOMO's include comes first in `src/fomo/urls.py`, so the unguarded upstream routes are unreachable — a test hitting the real `/calendar/create/` path proves it).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.

Reviewed todos not folded: "Run pre-executed demo notebooks against a scratch DB copy" (already WARN-05, Phase 40); "Isolate the campaign table query-count test from the shared file cache" (Phase 41 triage); 17 score-0.6 keyword matches (none concern calendar write access; all go to Phase 41 TRIAGE-01). Also out of scope per the domain: notebook isolation and the attribution page (Phase 40), todo triage (Phase 41), re-verification (Phase 42), any change to who may read the calendar, model permissions or per-event ownership, a run-detail view.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ACCESS-01 | An anonymous `POST` to any of the five write endpoints in `solsys_code/calendar_urls.py` does not create, change or delete a row; it is redirected to login; a test per endpoint asserts row count and targeted row unchanged | Upstream views carry no guard and accept any method (verified, see Pitfall 1); guard wrapper around the upstream callables at FOMO's URL conf, prototype verified 302 for all 5 routes and unchanged rows; shadowing of `tom_common.urls` verified by `resolve()` |
| ACCESS-02 | The month view's create/update click targets are hidden from anonymous users | Three click targets located in `calendar.html` (lines 218-224 `+ New Event`, 234-237 day cell, event rows keep `hx-get`); pop-up save/delete/todo controls are in `event_form.html` and `todos.html`; template branch design below |
| WARN-01 | `event_form.html` header states accurately which blocks differ from upstream | Full `diff -u` against installed 3.1.0 run this session; exact block list below; ledger rows to update located (two places in the 37.1 file) |
</phase_requirements>

## Summary

The five write routes in `solsys_code/calendar_urls.py` bind tomtoolkit 3.1.0's `tom_calendar.views` functions directly, and those functions have no login check and no method check. This session's prototype against a throwaway test database confirmed the baseline: an anonymous `POST` to `calendar:create-event` created a `CalendarEvent`; an anonymous `GET` to `calendar:update-todo` blanked a todo's description and cleared its completed flag; and an anonymous `GET` to `calendar:delete-event` deleted the event. The whole `tom_calendar` package is byte-identical to the installed 3.1.0 copy and to 3.0.1 (Phase 38), so the guard is FOMO's to add around the upstream callables at the URL layer. The same prototype showed that wrapping all five routes (anonymous requests redirected to login, `update-event` passing `GET`/`HEAD` through) leaves every row unchanged for an anonymous caller, returns `200` + `HX-Redirect` for an htmx request, and leaves a logged-in user's create/update/delete/todo flows working.

Because the delete and todo views act on `GET`, a plain `login_required` has a replay hazard: the `?next=` it appends is the write URL itself, so after logging in the browser issues a `GET` to that URL and the action executes (a login-replay delete). Redirect with `next` set to the calendar page instead, and (recommended) refuse non-`POST` on the destructive routes.

The pop-up change is template-only: `update_event` renders `event_form.html` with `form`, `event`, `action` for anonymous visitors, so a `request.user.is_authenticated` branch inside `event_form.html` can turn the form into a read-only card while the series/campaign blocks (already between `</form>` and the Todo heading) are rendered once for both audiences. The month view needs the `+ New Event` button and the day-cell `hx-get`/`hx-target`/`hx-on` removed for anonymous users; the inner event container keeps its own `hx-target` and `hx-on::after-request`, so event rows still open the modal. Existing tests that exercise the anonymous pop-up for form-only content (`EventFormUrlLinkTest`) and two anonymous Playwright tests (New Event button, empty day cell) must be re-pointed at a logged-in user.

**Primary recommendation:** Add a small module (e.g. `solsys_code/calendar_access.py`) with two decorators — `write_requires_login` (all methods; for create-event, delete-event, create-todo, update-todo) and `read_open_write_requires_login` (GET/HEAD pass through; for update-event) — each redirecting anonymous callers with `redirect_to_login(reverse('calendar:calendar'))`; wrap the five upstream callables in `calendar_urls.py`; add one `request.user.is_authenticated` branch to `event_form.html` and `calendar.html`; rewrite the `event_form.html` header from the actual `diff -u` below.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Reject anonymous writes | API / Backend (Django URL conf wrapper, server-side) | — | The only tier an attacker cannot bypass; the browser tier can only hide controls |
| Hide create/update click targets | Frontend Server (SSR Django template) | Browser (htmx attrs absent) | Rendered server-side from `request.user`; no client JS decides it |
| Read-only event detail card | Frontend Server (SSR template branch) | — | `update_event` GET already supplies `event`; the branch is a template concern |
| Redirect of a stale-tab htmx write | API / Backend (302) | Frontend Server (`HTMXRedirectMiddleware` converts to `HX-Redirect`) | Middleware already in `MIDDLEWARE`; no FOMO code |
| Modal open/close | Browser (Bootstrap 5 JS, htmx events) | — | Unchanged |
| Provenance header comment | Template source (documentation) | Test (source-level assertion) | Pure text; guarded by a source-level test |

## Standard Stack

No new packages. Everything is already installed and used.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django | 5.2.17 [VERIFIED: pip list this session] | `redirect_to_login`, `functools.wraps`, `require_POST`, test `Client` | In-repo stack; `django.contrib.auth.decorators` is the idiomatic guard |
| tomtoolkit / `tom_calendar` | 3.1.0 [VERIFIED: `pip show tomtoolkit`] | Upstream views being wrapped | Vendored, never edited |
| django-htmx | 1.29.0 [VERIFIED: pip list] | `request.htmx`, `trigger_client_event` used by upstream views | Already in `MIDDLEWARE` (`django_htmx.middleware.HtmxMiddleware`) |
| playwright (Python) | 1.62.0 [VERIFIED: pip list]; chromium present in `~/.cache/ms-playwright` | `@tag('functional')` browser test | Existing pattern in `test_bootstrap5_rendering.py` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom `redirect_to_login(next=calendar)` wrapper | Plain `login_required` | `login_required` sets `next` to the write URL itself; for `delete-event` / `update-todo` (act on GET) the post-login redirect replays the write (Pitfall 2). `login_required(v, redirect_field_name=None)` drops `next` entirely but D-08 says "with `?next=`" |
| Inline wrappers in `calendar_urls.py` | Separate `calendar_access.py` | A separate module is unit-testable and keeps `calendar_urls.py` a readable table of six routes; both satisfy the "do not duplicate upstream logic" rule |
| Template branch in `event_form.html` | Separate `event_detail.html` partial | A separate partial would need the series/campaign blocks included from both, or duplicated; one file with one branch keeps them rendered once and keeps the override count at the existing two |

**Installation:** none. **Package Legitimacy Audit:** no external packages are installed or added by this phase; the audit is not applicable. Packages removed (SLOP): none. Packages flagged (SUS): none.

## Architecture Patterns

### System Architecture Diagram

```
Browser (anonymous or signed-in)
   |
   |  GET /calendar/            POST /calendar/{create,update/<id>,delete/<id>,todo/...}
   v                            GET  /calendar/update/<id>/   (pop-up)
Django middleware stack (CSRF, Session, Auth, HtmxMiddleware, HTMXRedirectMiddleware,
   |                      Raise403Middleware, AuthStrategyMiddleware[READ_ONLY = no-op])
   v
src/fomo/urls.py:  path('calendar/', include('solsys_code.calendar_urls'))   <-- first match wins
   |                (tom_common.urls -> tom_calendar.urls is the same 6 paths, shadowed)
   v
solsys_code/calendar_urls.py
   |-- ''                  -> fomo_render_calendar            (read; template hides write targets)
   |-- create/             -> write_requires_login(create_event)
   |-- update/<id>/        -> read_open_write_requires_login(update_event)
   |-- delete/<id>/        -> write_requires_login(delete_event)
   |-- todo/create/<id>/   -> write_requires_login(create_todo)
   '-- todo/update/<id>/   -> write_requires_login(update_todo)
              |
     authenticated? --no--> redirect_to_login(next=/calendar/) -> 302
              |                  (HX-Request: HTMXRedirectMiddleware -> 200 + HX-Redirect)
             yes (or GET/HEAD on update/)
              v
   tom_calendar.views.<view>  (unchanged upstream)  -> renders event_form.html / calendar.html
              v
   event_form.html:  {% if request.user.is_authenticated %} <form> ... {% else %} read-only card {% endif %}
                     shared: observation-series block, attributed-run block, staff candidate hint
                     todo area: editable todos.html include (signed in) | read-only list (anonymous)
```

### Recommended Project Structure
```
solsys_code/
├── calendar_urls.py          # wraps 5 upstream callables with the guards (root view unchanged)
├── calendar_access.py        # NEW: the two guard decorators (no upstream logic)
└── tests/
    ├── test_calendar_write_access.py   # NEW: ACCESS-01 per-route tests, editor-unchanged, resolve() shadowing, ACCESS-02 render assertions, WARN-01 source assertions
    ├── test_calendar_template.py       # EDIT: EventFormUrlLinkTest logs in
    └── test_bootstrap5_rendering.py    # EDIT: 2 existing anonymous tests log in; ADD anonymous read-only modal test
src/templates/tom_calendar/partials/
├── calendar.html             # EDIT: gate 2 click targets; D-11 class-name tidy
└── event_form.html           # EDIT: read-only branch; rewritten header
docs/runbooks/telescope_runs_calendar.rst   # EDIT: pop-up section sentence
```

### Pattern 1: Guard wrapper around a vendored view
**What:** `functools.wraps` keeps the upstream callable's name/module; the wrapper only decides allow/redirect.
**When to use:** all five write routes.
**Example (prototype run against the real stack this session; outputs quoted below):**
```python
# Source: prototype in the session scratchpad; django.contrib.auth.views.redirect_to_login is Django public API
from functools import wraps

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse


def _login_redirect():
    # next = the calendar page, never the write URL: delete-event / update-todo act on GET,
    # so a next= pointing at them would replay the write after login (Pitfall 2).
    return redirect_to_login(reverse('calendar:calendar'))


def write_requires_login(view):
    """Anonymous callers are redirected to login for every method; the upstream view is untouched."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return view(request, *args, **kwargs)
        return _login_redirect()
    return wrapper


def read_open_write_requires_login(view):
    """GET/HEAD (the pop-up) pass through for anyone; every other method needs a login."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.method in ('GET', 'HEAD') or request.user.is_authenticated:
            return view(request, *args, **kwargs)
        return _login_redirect()
    return wrapper
```
Prototype output (anonymous, then signed-in), quoted verbatim:
```
create-event post 302 /accounts/login/?next=/calendar/
update-event post 302 /accounts/login/?next=/calendar/
delete-event post 302 /accounts/login/?next=/calendar/
delete-event get 302 /accounts/login/?next=/calendar/
create-todo post 302 /accounts/login/?next=/calendar/
update-todo post 302 /accounts/login/?next=/calendar/
update-todo get 302 /accounts/login/?next=/calendar/
create-event get 302 /accounts/login/?next=/calendar/
after anon: events 1 todo keep 1
htmx 200 /accounts/login/?next=/calendar/
anon GET update: 200
editor create 200 2
editor update 200 NEW
editor todo create 200 2
editor todo update 200 chg
editor delete 200 False
```
`reverse('calendar:calendar')` is evaluated inside the call (not at import), so the namespace-ambiguity warning `urls.W005` is irrelevant.

### Pattern 2: Template branch inside `event_form.html` with the shared blocks rendered once
**What:** one `{% if request.user.is_authenticated %}` wrapping `<form>...</form>`; the `{% else %}` renders the plain-text card from `event.*` (not from `form.*` widgets); the series/campaign/candidate blocks stay between the two exactly where they are; the todo area gets a second branch (editable include vs a read-only `{% for todo in event.todos.all %}` list).
**Why `event.*` and not `form.*`:** `CalendarEvent.user`, `proposal`, `telescope`, `instrument` are plain `CharField`s, `url` a `URLField`, `target_list` a nullable FK [VERIFIED: tom_calendar/models.py, read this session]. Rendering the card from `event` means no `<select>`/`<input>` is ever emitted for an anonymous visitor.
**The context:** `request` is available (`django.template.context_processors.request` is in `TEMPLATES[0]['OPTIONS']['context_processors']`, settings.py:69) and the template already reads `request.user.is_staff` at the candidate hint, so no view change is needed (matches CONTEXT "Established Patterns").

```django
{% if request.user.is_authenticated %}
<form ...unchanged...>...</form>
{% else %}
<dl class="row">
  <dt class="col-sm-3">Title</dt><dd class="col-sm-9">{{ event.title }}</dd>
  ... start/end, description (linebreaks), URL (is_web_url link), target list, user, proposal, telescope, instrument ...
</dl>
{% endif %}
{# series block, campaign block, staff candidate hint: unchanged, rendered once #}
<h6 class="mb-2">Todo list</h6><hr>
<div id="cal-todos">
{% if request.user.is_authenticated %}
  {% if action == "update" %}{% include 'tom_calendar/partials/todos.html' with event=event %}{% else %}<p>Save the event to add TODOs</p>{% endif %}
{% else %}
  {% for todo in event.todos.all %}...description, done/not done...{% empty %}<p class="text-muted small">No todos.</p>{% endfor %}
{% endif %}
</div>
```
Use `{% comment %}…{% endcomment %}` for any multi-line commentary; a multi-line `{# #}` fails `TemplateCommentSyntaxSweepTest`. Do not write the literal `pending_review` anywhere in the file (`test_template_source_never_contains_pending_review_literal`).

### Pattern 3: Gating the month-view click targets
- `+ New Event` `<button>` (calendar.html:218-225): wrap in `{% if request.user.is_authenticated %}`. The header row is `d-flex justify-content-between` with three children (nav buttons div, `<h4>`, button); removing the third makes `justify-content-between` push the `<h4>` to the right edge, so emit an empty `<div></div>` (or `<span></span>`) in the `{% else %}` to keep the title centred.
- Day cell (calendar.html:234-238): the three attributes `hx-get`, `hx-target`, `hx-on::after-request` go inside the authenticated branch. The inner event container (line 242-244) carries its own `hx-target="#cal-modal-body"` and `hx-on::after-request=...show()` and stops click propagation; the htmx `after-request` event bubbles to that container, so event rows still open the modal without the cell's attributes [VERIFIED: read calendar.html:234-245; behaviour to be re-proven by the functional test]. Do NOT remove the inner container's attributes.
- Event rows keep `hx-get="{% url 'calendar:update-event' event.id %}"` in all three variants (all-day, timed dashed, timed normal; lines 263, 302, 309).
- Optional polish: `.cal-day:hover` (lines 15-20) darkens every cell; an anonymous cell is inert, so scope the hover rule to an editor-only class (e.g. add `cal-day-editable` on the cell when authenticated) so the cell does not look clickable. Not required by a success criterion.
- Note `calendar.html` is rendered with `render(request, ...)` by both `fomo_render_calendar` and the upstream `render_calendar` that `create_event`/`update_event`/`delete_event` call, so `request.user` is always in context [VERIFIED: views.py read].

### Anti-Patterns to Avoid
- **Plain `login_required` on `delete-event` / `todo` routes** (replays the write after login; Pitfall 2).
- **Disabled-input form for the anonymous pop-up:** D-05 forbids it, and it would still emit the `<select>` option lists and CSRF token.
- **Guarding in the template only / hiding controls without the server guard:** ACCESS-01 is the server rule; ACCESS-02 is presentation only.
- **Editing the vendored package or re-implementing the upstream views:** forbidden by the established pattern and the CONTEXT discretion note.
- **Adding a third override (`todos.html`):** unnecessary; branch around the include in `event_form.html` so the header lists only blocks of one file.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Redirect to login | A custom `HttpResponseRedirect` with string-built URL | `django.contrib.auth.views.redirect_to_login` | Handles `LOGIN_URL`, query-encoding of `next`, resolved names |
| htmx-safe login redirect | Custom `HX-Redirect` handling | Existing `tom_common.middleware.HTMXRedirectMiddleware` | Already in `MIDDLEWARE`; prototype showed `200` + `HX-Redirect` header |
| Method restriction on destructive routes | `if request.method != 'POST'` blocks | `django.views.decorators.http.require_POST` | Standard 405 response with `Allow` header |
| Preserving wrapped-view identity | Manual `__name__` copying | `functools.wraps` | Keeps `__module__`/`__name__`/`__wrapped__` for tracebacks and `resolve()` assertions |
| Login simulation in tests | Posting to the login form | `self.client.force_login(user)` (repo pattern) and, for Playwright, the session-cookie hand-off at `test_bootstrap5_rendering.py:277-290` | Existing, proven patterns |
| Reading the upstream file for the WARN-01 check | Hard-coded copy of upstream | `tom_calendar.__file__` -> `templates/tom_calendar/partials/event_form.html` | The installed package is the source of truth |

**Key insight:** every piece (guard, redirect, htmx conversion, login in tests) already exists in Django, `tom_common` or this repo; the phase is wiring plus two template branches.

## Runtime State Inventory

Not a rename/refactor/migration phase — section omitted per the template rule.

## Common Pitfalls

### Pitfall 1: Upstream write views act on any HTTP method
**What goes wrong:** `delete_event`, `create_todo` and `update_todo` have no `request.method` check; `create_event` and `update_event` treat every non-POST as the form-render path. Baseline run this session (no guard): anonymous `GET` to `calendar:delete-event` returned `200` and the row was gone (`CalendarEvent.objects.filter(pk=ev.pk).exists()` printed `False`); anonymous `GET` to `calendar:update-todo` returned `200` and the todo's description no longer equalled `'keep'` (it is overwritten with `""` and `is_completed=False` by `request.POST.get(...)` defaults).
**Why it happens:** the upstream code assumes htmx `hx-post` only.
**How to avoid:** the guard for the four non-read routes must cover every method (not just POST); ACCESS-01 tests must include an anonymous `GET` to `delete-event` and `update-todo` as well as the POSTs the requirement names. Only `update-event` is method-aware.
**Warning signs:** a test that only POSTs passes while `GET /calendar/delete/<id>/` still deletes.

### Pitfall 2: `login_required`'s `next` replays the write after login
**What goes wrong:** `login_required` redirects to `/accounts/login/?next=/calendar/delete/5/`. After the visitor signs in, the browser issues a `GET` to that URL; upstream `delete_event` deletes on `GET`. The same applies to `update-todo` (blanks the todo) and `create-todo` crashes. A crafted link `/calendar/delete/5/` sent to a not-logged-in user becomes a delete once they log in.
**Why it happens:** Django's post-login redirect is always a `GET` to `next`.
**How to avoid:** redirect with `next` = the calendar page (the prototype's `redirect_to_login(reverse('calendar:calendar'))`), still "a 302 to `LOGIN_URL` with `?next=`" as D-08 describes. Additionally (recommended hardening, see Open Question 1) apply `require_POST` inside the guard for `delete-event`, `create-todo`, `update-todo` so a `GET` by a logged-in user is a `405`, not a delete. The UI only ever uses `hx-post` for these [VERIFIED: event_form.html:97 and todos.html `hx-post` attributes, read this session].
**Warning signs:** a test asserting the redirect target equals `LOGIN_URL?next=<the write URL>`.

### Pitfall 3: Existing tests that read the anonymous pop-up as a form
**What goes wrong:** after the read-only branch lands, `EventFormUrlLinkTest` (test_calendar_template.py:1596-1640) fails: `test_allocation_key_is_not_a_link` asserts `self.assertIn(f'value="{key}"', content)` — an `<input value=...>` that no longer renders for an anonymous client. The two Playwright tests `test_calendar_modal_opens_for_new_event_button_with_no_page_errors` and `test_calendar_modal_opens_for_empty_day_cell_with_no_page_errors` (test_bootstrap5_rendering.py:143-171, 173-189) click targets that anonymous visitors no longer get.
**How to avoid:** `force_login` a user in `EventFormUrlLinkTest._form_html` (it then tests the editor form as before) and add anonymous variants asserting the card; give the two Playwright tests the session-cookie hand-off. The other anonymous modal tests (`EventModalCampaignRunLinkTest`, `EventModalRunTallyTest`, `EventModalSeriesDecorationTest`, `EventModalAttributionHintTest`) assert on decoration text and links that the card still renders; `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` cells GET `calendar:update-event` through `public_client` and assert only on the decoration text (read this session, lines ~1301-1313), so it stays valid and needs no regeneration.
**Warning signs:** run `EventFormUrlLinkTest` and `TestBootstrap5Rendering` early.

### Pitfall 4: `observation_series_decoration` is already hidden from anonymous viewers
**What goes wrong:** D-05 says the observation-series line "renders unchanged". For an anonymous visitor it renders nothing — by Phase 34's design (`observation_series_decoration` returns `None` when `_viewer_is_authenticated(context)` is false, calendar_display_extras.py:841-842; the runbook states "This block requires logging in, full stop", lines 284-292). Success criterion 3 therefore means the *attributed-run* block, not the series line, is what an anonymous visitor reads. Tests/assertions that expect a series line on the anonymous card would be wrong.

### Pitfall 5: Header-comment and rendered-output tests
`test_modal_renders_no_django_comment_delimiters` asserts the rendered modal never contains `{#`, `#}` or `FOMO override of the upstream tom_calendar partial`. Keep the header inside `{% comment %}…{% endcomment %}` and do not echo the header text into any rendered element. The WARN-01 list must use text that does not contain `{#`.

### Pitfall 6: Centring of the month title when the button disappears
See Pattern 3; verify visually in the functional test screenshot or by DOM assertion that the header still has three children.

### Pitfall 7 (pre-existing, not this phase's fix): upstream bugs reachable by a logged-in user
- `create_todo` returns `None` when the description is blank after `.strip()`; Django raises `ValueError: The view tom_calendar.views.create_todo didn't return an HttpResponse object` (reproduced this session as a 500 for a signed-in POST with `description=' '`). The UI input is `required`, so this needs a hand-crafted POST. Out of scope; note for Phase 41 triage if desired.
- `create_event`/`update_event`/`delete_event` re-render the month with the **upstream** `render_calendar`, not `fomo_render_calendar`, so the refreshed partial lacks FOMO's `active_todo_count` annotation and prefetch. Unchanged by this phase ("exactly as before").

## Code Examples

### URL conf after the change
```python
# solsys_code/calendar_urls.py (sketch)
from django.urls import path
from django.views.decorators.http import require_POST
from tom_calendar.views import create_event, create_todo, delete_event, update_event, update_todo

from solsys_code.calendar_access import read_open_write_requires_login, write_requires_login
from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
    path('create/', write_requires_login(create_event), name='create-event'),
    path('update/<int:event_id>/', read_open_write_requires_login(update_event), name='update-event'),
    path('delete/<int:event_id>/', write_requires_login(require_POST(delete_event)), name='delete-event'),
    path('todo/create/<int:event_id>/', write_requires_login(require_POST(create_todo)), name='create-todo'),
    path('todo/update/<int:todo_id>/', write_requires_login(require_POST(update_todo)), name='update-todo'),
]
```
(`require_POST` shown for the recommended hardening; drop it if Open Question 1 is answered "no". Guard outermost so an anonymous caller always gets the login redirect, never a `405`.)

### ACCESS-01 test skeleton (Django `TestCase`, anonymous client)
```python
# Source: patterns from solsys_code/tests/test_calendar_template.py + this session's prototype
class AnonymousCalendarWriteTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.event = CalendarEvent.objects.create(
            title='Keep me', start_time=datetime(2026, 7, 4, 20, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 4, 21, tzinfo=dt_timezone.utc))
        cls.todo = EventTodo.objects.create(event=cls.event, description='keep', is_completed=True)

    def assert_login_redirect(self, response):
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(settings.LOGIN_URL))

    def test_anonymous_post_update_event_changes_nothing(self):
        response = self.client.post(reverse('calendar:update-event', args=[self.event.pk]),
                                    {'title': 'Hacked', 'start_time': '2026-07-04T20:00', 'end_time': '2026-07-04T21:00'})
        self.assert_login_redirect(response)
        self.event.refresh_from_db()
        self.assertEqual(self.event.title, 'Keep me')
        self.assertEqual(CalendarEvent.objects.count(), 1)
    # ... create-event (count), delete-event (row exists), create-todo (EventTodo count), update-todo (fields) ...
    # plus: GET create-event and GET delete-event / update-todo are redirected and change nothing
    # plus: htmx request -> status 200 and 'HX-Redirect' header
    # plus: resolve('/calendar/create/').func.__wrapped__ is tom_calendar.views.create_event  (shadowing proof, guard is in the real URL conf)
```
Editor-unchanged test: `force_login` a plain (non-staff) `User`, POST each route with valid form data (`start_time`/`end_time` in `'%Y-%m-%dT%H:%M'`), assert count/field changes (prototype printed `editor create 200 2`, `editor update 200 NEW`, `editor delete 200 False`).

### Playwright login hand-off (already in the repo)
```python
# Source: solsys_code/tests/test_bootstrap5_rendering.py:277-290
self.client.force_login(user)
self.page.context.add_cookies([{
    'name': settings.SESSION_COOKIE_NAME,
    'value': self.client.cookies[settings.SESSION_COOKIE_NAME].value,
    'url': self.live_server_url,
}])
```

## The exact diff: FOMO overrides vs tomtoolkit 3.1.0 (run this session)

Installed upstream: `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_calendar/templates/tom_calendar/partials/{event_form,calendar,todos}.html` and `calendar_page.html`. `diff -u upstream FOMO` for `event_form.html` yields exactly four regions [VERIFIED: diff run and both files read in full]:

| # | Where in FOMO's file | What differs from 3.1.0 |
|---|---------------------|-------------------------|
| 1 | Lines 1-23 (before the `<form>`) and the load line | A `{% comment %}` header block; `{% load django_bootstrap5 attribution_display_extras calendar_display_extras %}` where upstream has `{% load django_bootstrap5 %}` |
| 2 | URL label inside the form | `{% if form.url.value|is_web_url %}` link with `rel="noopener noreferrer"` `{% elif form.url.value %}<small class="text-muted">(not a web link)</small>{% endif %}`; upstream: `{% if form.url.value %}` link with no `rel` |
| 3 | Button row at the end of the form | Plain `<button type="submit" class="btn btn-primary">Save</button>`, `<button type="submit" name="save_and_edit" value="1" ...>Save and edit</button>` and a one-line-attribute Delete `<button type="button" hx-post=... class="btn btn-danger">`; upstream uses `{% bootstrap_button "Save" ... %}` and `"Save and Edit"` (capital E) and a differently laid-out Delete button |
| 4 | Between `</form>` and `<h6 class="mb-2">Todo list</h6>` | Absent upstream: two `{% comment %}` blocks (D-14/D-17 Phase 33; PROJ-04/05 Phase 34), `{% observation_series_decoration event as series %}` block, `{% campaign_decoration event as deco %}` block (with `run_tally` and the `{% elif not event.telescope_label_meta.run and request.user.is_staff %}` candidate-hint branch) |

Phase 39 adds a fifth region: the `{% if request.user.is_authenticated %}`/`{% else %}` read-only card around the `<form>` (D-05) and the second branch inside `<div id="cal-todos">` for the read-only todo list. Everything else in the body is byte-for-byte upstream. The WARN-01 header should list these five items, each as "block name + position (before/after which upstream line)", pin "tomtoolkit 3.1.0", give the upstream path `tom_calendar/templates/tom_calendar/partials/event_form.html`, and delete "starts as an exact copy of the upstream partial with one new block" and the stale "(tomtoolkit 3.0.1 ... 3.0.0a9 -> 3.0.1 Bootstrap4->5 migration)" history sentence (that history is now irrelevant to a 3.1.0 comparison; keep the D-08 / T-27-20 / G-37.1-1-allocurl reasons, trimmed). Re-run `diff -u` after editing and make the list match hunk for hunk; a `{% if %}` split that opens in one hunk and closes in another counts as one item.

`calendar.html` differs from upstream by much more than the Bootstrap 4 names (see 38-OVERRIDE-COMPARISON); D-11's tidy is limited to the utility classes. Exact occurrences to change [VERIFIED: diff run this session; each appears in FOMO's file where upstream has the BS5 name]:
- `var(--white)` -> `var(--bs-white)` (the `.cal-day.is-current-month.today .day-num` rule)
- `border-top border-left` -> `border-top border-start` (cal-grid div)
- `font-weight-bold` -> `fw-bold`, `border-right` -> `border-end` (day-header div)
- `border-right` -> `border-end` (day-cell div)
- `mr-2` -> `me-2` (target-list footer link); `mr-3` -> `me-3` (three legend `<span>` classes: proposal swatch, telescope legend entry, and both status legend branches — four occurrences in total; `grep -n 'mr-\|border-left\|border-right\|font-weight-bold\|var(--white)'` finds all)
Also `.sr-only` / `form-group` / `form-inline` appear in upstream's own `todos.html`, which FOMO does not override — leave them. `test_calendar_template.py` asserts `data-url`; keep it.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| jQuery `$('#cal-modal').modal('show')` | `bootstrap.Modal.getOrCreateInstance(...).show()` | tomtoolkit 3.x (Bootstrap 5), Phase 33 G-33-2 | Already done; do not regress when editing the cell attributes |
| Bootstrap 4 utility names (`mr-3`, `border-left`) | Bootstrap 5 (`me-3`, `border-start`) | tomtoolkit 3.0.1 | D-11 tidy |

**Deprecated/outdated:** the header's "tomtoolkit 3.0.1 ... 3.0.0a9" history; upstream `data-bs-url` attribute mismatch (FOMO keeps `data-url`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | django-allauth's login view honours a `next` query parameter that points at a same-host path, so `?next=/calendar/` returns the visitor to the calendar after login | Pitfall 2 / Pattern 1 | Low: a `next` that is ignored just sends the user to `LOGIN_REDIRECT_URL = '/'`; the guard still holds. (`next` was not exercised through a real login in this session; only the redirect URL was.) |
| A2 | The htmx `after-request` event from an event row bubbles to the inner container's `hx-on::after-request` handler, so event rows still open the modal once the day cell's own attributes are removed | Pattern 3 | Medium: if wrong, anonymous (and editor) event clicks would not show the modal. The functional test in D-09 is the proof; the planner must keep it. |
| A3 | A `GET`-only restriction (`require_POST`) on delete/todo routes does not break any current caller | Code Examples / Open Question 1 | Low: repo-wide grep for the five URL names (this session) found only `hx-post`/`hx-get` uses in `calendar.html`, `event_form.html`, `todos.html` (upstream) and tests that GET `update-event` |
| A4 | ASVS L1 mapping: access control (V4 — enforce on a trusted server layer) and anti-CSRF (V4.2/V13-style) are the applicable categories | Security Domain | Low: category numbering varies by ASVS version; planner should cite by name |

## Open Questions

1. **Add `require_POST` to `delete-event`, `create-todo`, `update-todo`?**
   - What we know: upstream acts on any method; a signed-in user's `GET /calendar/delete/<id>/` deletes (a top-level navigation `GET` sends the session cookie even under `SameSite=Lax`, so a crafted link is a CSRF-style delete). The UI uses `hx-post` only.
   - What's unclear: whether the user counts this as in scope ("exactly as before" for editors; the requirement is about anonymous writes).
   - Recommendation: include it (one decorator per route, `405` for non-POST by a signed-in user, tests for each). If declined, the `next=/calendar/` redirect (Pitfall 2) alone is still required.
2. **D-08 wording versus `next=/calendar/`.** D-08 says "`login_required` semantics: 302 to `LOGIN_URL` with `?next=`". The recommended wrapper keeps that shape but sets `next` to the calendar page. A test written literally against `?next=<write URL>` would not match. Recommendation: tests assert `startswith(settings.LOGIN_URL)` and that `next` is the calendar page; record the reason in the plan.
3. **URL row on the anonymous card for non-web values.** The projectors store `ALLOC:<run>:<night>` / `RUN:<pk>` namespace keys in `CalendarEvent.url`. D-05 says "URL (with the existing `is_web_url`-gated View link)". Recommendation: show the link for http(s) URLs and, for a non-web value, show only the "(not a web link)" note without echoing the raw key (internal identifier; the editor form still shows it). Planner/executor choice; `EventFormUrlLinkTest` has the cases to mirror.
4. **Todo text visibility.** D-05 locks showing todos read-only to anonymous visitors (they could already read them in the old form's todo include). No change, but the runbook sentence should say so.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | all | yes | 3.11.13 (dev venv `devel_fomo311_venv`) | — |
| Django | guard, tests | yes | 5.2.17 | — |
| tomtoolkit / tom_calendar | upstream views, WARN-01 diff | yes | 3.1.0 | — |
| Playwright + chromium | D-09 functional test | yes | playwright 1.62.0; `~/.cache/ms-playwright/chromium-1234`, `-1243` | CI `functional-tests` job installs it |
| SPICE kernels (~1.6 GB, `~/.cache/sorcha/`) | importing `solsys_code.views` (needed by `calendar_urls`) | yes (prototype imported it without a download) | — | — |
| Node 22 + gsd-tools | planning workflow only | not probed (not needed by this phase's code) | — | — |

**Missing dependencies with no fallback:** none. **With fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`, `SimpleTestCase`, `StaticLiveServerTestCase` + Playwright for `@tag('functional')`) — the only runner in this repo |
| Config file | none (settings `src.fomo.settings`, set by `manage.py`) |
| Quick run command | `python manage.py test solsys_code.tests.test_calendar_write_access solsys_code.tests.test_calendar_template` |
| Full suite command | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (CI/pre-commit: `python manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault`) |
| Functional only | `python manage.py test --tag functional` (needs chromium; runs in CI `functional-tests`) |

Note: importing `solsys_code.views` (via `calendar_urls`) loads `ephem_utils` and its SPICE kernels, so even a single-module run pays the import cost.

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ACCESS-01 | Anonymous POST to each of `create-event`, `update-event`, `delete-event` redirected to login; count and targeted row unchanged | unit (Django TestCase) | `python manage.py test solsys_code.tests.test_calendar_write_access.AnonymousCalendarWriteTest` | Wave 0 |
| ACCESS-01 (D-02) | Same for `create-todo`, `update-todo`; plus anonymous GET on `delete-event`, `update-todo`, `create-event` | unit | same class | Wave 0 |
| ACCESS-01 (D-08) | htmx request gets `200` + `HX-Redirect` to login; no `403` | unit | same class | Wave 0 |
| ACCESS-01 (D-04) | Anonymous `GET update-event` is `200` and contains the attributed-run block | unit | `...test_calendar_template.EventModalCampaignRunLinkTest` (existing) + new case | existing |
| ACCESS-01 | Real URL conf is guarded: `resolve('/calendar/create/')` (and the other four) reach the wrapped upstream view, proving the `tom_common.urls` copy is shadowed | unit | `...test_calendar_write_access` | Wave 0 |
| Success 2 | Signed-in non-staff user can still create, update, delete, add and tick a todo (POST each route; counts/fields change) | unit | `...test_calendar_write_access.SignedInCalendarWriteTest` | Wave 0 |
| ACCESS-02 | Anonymous `GET /calendar/` contains no `calendar:create-event` URL and no `+ New Event`; signed-in contains both | unit | `...test_calendar_write_access` | Wave 0 |
| ACCESS-02 | Anonymous pop-up has no `<form`, no Save / Delete / `hx-post`, no `csrfmiddlewaretoken`, no `<select`; shows title/description/todos as text; signed-in pop-up has the form | unit | `...test_calendar_write_access` | Wave 0 |
| ACCESS-02 | Anonymous browser opens the read-only modal by clicking an event; no `+ New Event` button; no page errors | functional (Playwright) | `python manage.py test --tag functional solsys_code.tests.test_bootstrap5_rendering` | edit (add test) |
| ACCESS-02 | Signed-in browser still opens the modal from `+ New Event` and from an empty day cell | functional | same | edit (log in two existing tests) |
| WARN-01 | Header names `tomtoolkit 3.1.0` and the upstream path, no longer contains "exact copy", lists every region; `diff -u` against the installed upstream file matches the list | unit (source-level) + manual `diff -u` | `...test_calendar_write_access.EventFormHeaderTest` | Wave 0 |
| D-11 | `calendar.html` source contains none of `mr-`, `border-left`, `border-right`, `font-weight-bold`, `var(--white)` and still contains `data-url=` | unit (source-level) | `...test_calendar_write_access` | Wave 0 |
| D-12 | Ledger rows updated (37.1 disposition YAML front matter line 198 `disposition: open` and table row line 335; note under WR-05 in 33-REVIEW.md) | manual / grep | `grep -n "WR-05" .../37.1-REVIEW-DISPOSITION.md` | manual |

### Sampling Rate
- **Per task commit:** the quick run command above.
- **Per wave merge:** `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`.
- **Phase gate:** full suite green plus `python manage.py test --tag functional solsys_code.tests.test_bootstrap5_rendering`, `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` clean, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_calendar_write_access.py` — new module covering ACCESS-01 (per-route anonymous, htmx, shadowing), signed-in unchanged, ACCESS-02 render assertions, WARN-01 header and D-11 source assertions.
- [ ] `EventFormUrlLinkTest._form_html` needs `self.client.force_login(...)` (it asserts on `<input value=...>`, form-only), plus anonymous-card variants.
- [ ] `test_bootstrap5_rendering.py` two existing anonymous tests need the session-cookie hand-off; add one anonymous read-only modal test (click an event, assert `#cal-modal-body` has no `form`, has `View campaign`, `+ New Event` count is 0).
- Framework install: none needed.

## Security Domain

`security_enforcement` is on (`.planning/config.json`: `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (no new login surface; uses existing allauth login) | — |
| V3 Session Management | yes (indirectly: the guard trusts `request.user` from the session middleware) | Django session auth; unchanged |
| V4 Access Control | **yes — the core of the phase** | Server-side guard at FOMO's URL conf wrapping the upstream callables; deny by default for anonymous on every write route |
| V5 Input Validation | partially | Upstream `EventForm` unchanged; read-only card must autoescape `event.*` (Django default; do not use `|safe`) |
| V6 Cryptography | no | — |
| V13 / anti-CSRF | yes | Django `CsrfViewMiddleware` stays on; the signed-in forms keep `{% csrf_token %}`; the anonymous card has no forms. `require_POST` on destructive routes closes GET-based CSRF |

### Known Threat Patterns (URL-by-URL for the stack)

| URL name | Path | Upstream callable (module) | Methods that act upstream | Guard (FOMO layer) | Anonymous result |
|----------|------|---------------------------|---------------------------|--------------------|------------------|
| `calendar:calendar` | `/calendar/` | `solsys_code.views.fomo_render_calendar` (FOMO's own) | GET (read) | none needed (read) | 200; template omits write targets |
| `calendar:create-event` | `/calendar/create/` | `tom_calendar.views.create_event` | any (POST saves; non-POST renders form) | `write_requires_login` | 302 to login (HX: 200 + `HX-Redirect`) |
| `calendar:update-event` | `/calendar/update/<event_id>/` | `tom_calendar.views.update_event` | POST saves; any other method renders the pop-up | `read_open_write_requires_login` (GET/HEAD open) | GET 200 read-only card; POST/PUT/PATCH/DELETE/OPTIONS 302 |
| `calendar:delete-event` | `/calendar/delete/<event_id>/` | `tom_calendar.views.delete_event` | **any, including GET** | `write_requires_login` (+ `require_POST` recommended) | 302 |
| `calendar:create-todo` | `/calendar/todo/create/<event_id>/` | `tom_calendar.views.create_todo` | any | `write_requires_login` (+ `require_POST`) | 302 |
| `calendar:update-todo` | `/calendar/todo/update/<todo_id>/` | `tom_calendar.views.update_todo` | **any, including GET (blanks the todo)** | `write_requires_login` (+ `require_POST`) | 302 |

All five paths are claimed by FOMO's include (`src/fomo/urls.py:30`) before `tom_common.urls` mounts `tom_calendar.urls` under the same namespace; upstream defines exactly the same six paths, so no upstream write route is reachable unguarded [VERIFIED: `resolve()` for all six paths this session returned FOMO-order matches; `tom_calendar/urls.py` read].

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Anonymous POST creates/changes/deletes a row | Tampering | Server-side login guard on the URL conf; tests per route assert unchanged rows |
| Anonymous GET triggers a destructive upstream view (delete, todo wipe) | Tampering | Guard covers all methods on these routes; `require_POST` for signed-in users |
| Login replay: `?next=<write URL>` executes the write after login | Tampering / CSRF | `next` = calendar page; `require_POST` |
| CSRF on a signed-in session | Tampering | Django CSRF middleware (unchanged); forms keep `{% csrf_token %}`; no GET-acting routes |
| Stale-tab htmx save swaps login page into the modal | Spoofing / UX | `HTMXRedirectMiddleware` -> `HX-Redirect` (already installed); test asserts the header |
| Information exposure in the pop-up (user dropdown, form widgets) for anonymous | Information disclosure | Read-only card renders `event.*` text only; no `<select>` |
| Information exposure of internal keys (`ALLOC:`/`RUN:`, series group name) | Information disclosure | Series block already authenticated-only; do not echo non-web URL values on the anonymous card (Open Question 3) |
| Open redirect via `next` | Spoofing | `next` is a fixed server-built path (`reverse('calendar:calendar')`), never user input |
| Reflected/stored XSS in event fields shown on the card | Tampering | Django autoescape; no `|safe`; `rel="noopener noreferrer"` on the external link as today |
| Residual: any logged-in user may delete any event | Elevation of privilege | Accepted by D-01/D-03; reversible with `user_passes_test`/`permission_required` at the same wrapper point |

## Sources

### Primary (HIGH confidence)
- Installed `tom_calendar` 3.1.0 source, read this session: `views.py`, `urls.py`, `models.py`, `templates/tom_calendar/{calendar_page.html,partials/event_form.html,partials/calendar.html,partials/todos.html}` (`/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_calendar/`)
- Installed `tom_common/middleware.py` (`HTMXRedirectMiddleware`, `AuthStrategyMiddleware`, `Raise403Middleware`), read this session
- FOMO files read this session: `solsys_code/calendar_urls.py`, `src/templates/tom_calendar/partials/{event_form,calendar}.html`, `src/fomo/settings.py` (MIDDLEWARE via `python manage.py shell`), `src/fomo/urls.py`, `solsys_code/views.py:fomo_render_calendar`, `solsys_code/templatetags/calendar_display_extras.py`, tests in `solsys_code/tests/test_calendar_template.py` and `test_bootstrap5_rendering.py`
- `diff -u` of the three FOMO/upstream template pairs and a prototype run of the guard against a real test database (baseline unguarded behaviour and guarded behaviour, outputs quoted above)
- `.planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md`, `.planning/REQUIREMENTS.md`, 33-REVIEW.md WR-05, 37.1-REVIEW-DISPOSITION.md WR-05 (both places)

### Secondary (MEDIUM confidence)
- none

### Tertiary (LOW confidence)
- Django post-login `next` handling by allauth (A1), ASVS numbering (A4) — training knowledge, tagged `[ASSUMED]` in the log

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — nothing new; all versions read from the environment
- Architecture: HIGH — guard prototype run end to end; template structure read in full
- Pitfalls: HIGH for 1-5 (reproduced or read in source); MEDIUM for A2 (htmx event bubbling, needs the functional test)

**Research date:** 2026-10-08
**Valid until:** until tomtoolkit is upgraded past 3.1.0 (the WARN-01 list and the unguarded-upstream finding are pinned to it); otherwise 30 days
