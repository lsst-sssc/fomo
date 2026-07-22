---
phase: 21-site-disambiguation-submitter-contact-opt-in
reviewed: 2026-07-11T00:00:00Z
depth: deep
files_reviewed: 12
files_reviewed_list:
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py
  - solsys_code/models.py
  - solsys_code/solsys_code_observatory/utils.py
  - solsys_code/solsys_code_observatory/views.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_forms.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
findings:
  critical: 2
  warning: 2
  info: 1
  total: 5
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-07-11
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed the SITE-01/SITE-02 site-disambiguation UI (fuzzy-match datalist + "Create new
Observatory" round-trip) and the VIEW-05 submitter contact-visibility opt-in, at deep depth
(full call-chain tracing across `campaign_tables.py` → `campaign_views.py` →
`campaign_utils.py` → `solsys_code_observatory/utils.py`/`views.py`, plus the paired
templates that the required-reading list didn't include but which the diff's own behavior
depends on).

The VIEW-05 contact-opt-in work is solid: the `Case`/`When` SQL-level gate is correctly
ordered relative to `.values()`, is exercised directly against the raw queryset (not just
rendered HTML) by `test_campaign_views.py::TestContactPublicOptIn`, and the migration/model/
form additions are all consistent.

The SITE-01/SITE-02 site-disambiguation work has two significant end-to-end gaps that the
test suite does not catch because the tests bypass the real request path at exactly the
point where each gap lives: (1) the fuzzy-matched candidates offered in the datalist are, in
the overwhelming majority of cases, not actually resolvable by the server-side code that
consumes the submitted value, defeating the stated purpose of the feature; and (2) the
"Create new Observatory" `?next=`/`?obscode=` round-trip is silently dropped by the real HTML
form (hardcoded `action=`, no hidden `next`/`obscode` field), so `get_success_url()`'s new
open-redirect-safe `next` handling is unreachable via the actual UI even though its unit
tests pass (they inject `next` directly into the POST payload, which a real browser session
submitting the actual template never would).

## Critical Issues

### CR-01: Fuzzy-matched site candidates (name/short_name/old_names) cannot actually resolve on approve

**File:** `solsys_code/campaign_tables.py:208-246`, `solsys_code/campaign_views.py:371-374`, `solsys_code/campaign_utils.py:111-145`

**Issue:** `ApprovalQueueTable.render_site()` builds a `<datalist>` whose `<option>` values are
the *display strings* returned by `fuzzy_match_candidates()` — i.e. an obscode, `name_utf8`,
`short_name`, or `old_names` string (campaign_tables.py:226-231). The row's own docstring
states the intent explicitly:

> "the resolved obscode itself is read server-side from site_selection's submitted text"
> (campaign_tables.py:227-229)

But `CampaignRunDecisionView.post()` never performs that server-side lookup. It takes the
raw submitted text verbatim and passes it straight into `resolve_site()`:

```python
selection = request.POST.get('site_selection', '').strip() or run.site_raw
site, needs_review = resolve_site(selection, create_placeholder=False)
```

`resolve_site()` treats its argument as a literal MPC obscode: it immediately rejects
anything longer than `Observatory.obscode`'s `max_length` (4 chars, `campaign_utils.py:142-145`),
and even for strings that happen to be ≤4 chars, tier 1/2 look the string up *as an obscode*
(`Observatory.objects.get(obscode=code)`, then `MPCObscodeFetcher.query(code)` which POSTs
`{'obscode': code}` to the MPC API). A `name_utf8` candidate like `'Siding Spring Observatory'`
or even a `short_name` candidate like `'SSO'`/`'FTS'` is not the actual obscode, so selecting
it from the datalist and submitting the form does **not** resolve the site — `resolve_site()`
silently falls through to `(None, True)`, exactly the same "unresolved, needs review" outcome
as if the staff member had typed nothing at all.

In practice this means the fuzzy-match datalist only "works" for the one candidate string
that happens to equal the real obscode (which was already an option before this feature:
just retype the correct obscode). Every other candidate the feature was built to surface —
names, short names, old names, i.e. precisely the strings a staff member is likely to
recognize when they don't already know the obscode — is a dead end that produces no visible
error and looks like a successful selection.

This gap is not caught by the test suite: `TestSiteFuzzyMatch` only tests
`fuzzy_match_candidates()`'s matching logic in isolation, and
`TestSiteSelectionResolution::test_staff_typed_existing_obscode_resolves_via_site_selection_tier_1_hit`
only exercises `site_selection='G37'` (the literal obscode), never a name/short_name
candidate actually offered by the datalist.

**Fix:** Resolve the submitted `site_selection` text back to its obscode via the same
candidate pool used to build the datalist, before calling `resolve_site()`:

```python
# campaign_views.py, inside CampaignRunDecisionView.post()
if run.site is None:
    selection = request.POST.get('site_selection', '').strip() or run.site_raw
    candidate_pool = build_site_candidates()
    obscode_selection = candidate_pool.get(selection, selection)  # exact-match display string -> obscode
    site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
```

(A stricter alternative: emit the obscode as the `<option>` *value* instead of the display
string, with the display text shown via a paired `<label>`/visible text node, so the browser
submits the obscode directly — but the code comment at campaign_tables.py:227-229 explicitly
rejected that approach in favor of server-side resolution, which was then never implemented.)

### CR-02: "Create new Observatory" `?next=`/`?obscode=` round-trip is dropped by the real form — feature is non-functional outside tests

**File:** `solsys_code/solsys_code_observatory/views.py:27-48`, `src/templates/solsys_code_observatory/observatory_create.html:14`

**Issue:** `CreateObservatory.get_success_url()` was updated to honor a validated `?next=`
query param, and `get_initial()` pre-fills `obscode` from `?obscode=` — both intended to
support the round-trip from `ApprovalQueueTable.render_site()`'s "Create new Observatory"
link (`campaign_tables.py:232-235`), which builds a URL like
`.../observatory/create/?obscode=<site_raw>&next=/campaigns/approval-queue/`.

However, the actual template's `<form>` element hardcodes its `action` with no query string
and provides no hidden field to carry `next` (or `obscode`) through to the POST:

```html
<form action="{% url 'solsys_code_observatory:create' %}" enctype="multipart/form-data" method="POST">
  {% csrf_token %}
  {% bootstrap_form form %}
  ...
</form>
```

A browser submitting this form posts to the bare `/observatory/create/` URL — no query
string is carried over from the page's own URL (that's not how HTML form submission works
when `action` is explicitly set), and `CreateObservatoryForm` (in
`solsys_code_observatory/forms.py`) has no `next` field either. So on the real POST,
`self.request.GET.get('next')` and `self.request.POST.get('next')` are both empty, and
`get_success_url()` always falls back to the Observatory detail page — the entire point of
the round-trip (returning staff to the approval queue after creating the missing
Observatory) silently never happens.

This is masked by the tests: `TestCreateObservatoryRoundTrip::test_valid_create_with_safe_next_redirects_to_approval_queue`
and `test_unsafe_next_falls_back_to_detail_redirect` both call
`self.client.post(create_url, {'obscode': 'G37', 'next': self.next_url})` — i.e. they inject
`next` directly into the POST body by hand, which proves `get_success_url()`'s validation
logic is correct in isolation, but does not exercise (and therefore doesn't catch the failure
of) the actual template that a real staff user's browser would submit.

**Fix:** Carry `next` (and, if desired, `obscode`) through the real form, e.g. add a hidden
field to the template:

```html
<form action="{% url 'solsys_code_observatory:create' %}{% if request.GET.next %}?next={{ request.GET.next|urlencode }}{% endif %}"
      enctype="multipart/form-data" method="POST">
  {% csrf_token %}
  {% if request.GET.next %}<input type="hidden" name="next" value="{{ request.GET.next }}">{% endif %}
  {% bootstrap_form form %}
  ...
</form>
```

and/or add a regression test that renders the actual template (`self.client.get(create_url, {...})`
then parses the response for a `next` hidden input / the form's `action` query string) rather
than only posting a hand-built payload.

## Warnings

### WR-01: `build_site_candidates()` can still raise despite its "never raises" contract

**File:** `solsys_code/campaign_utils.py:254-296` (except clause at 283), `solsys_code/campaign_utils.py:212-233`

**Issue:** `build_site_candidates()`'s docstring promises: "a bulk-fetch network/parse
failure is caught narrowly and falls back to the local-only `Observatory` pool, never
raising into `ApprovalQueueView`'s page render." The except clause is:

```python
except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
```

`_flatten_mpc_candidates(obscode_dict)` (called inside the `try`) does
`for code, rec in obscode_dict.items():` — if the MPC bulk endpoint ever returns a JSON value
that isn't a dict (e.g. a list, or `None`, both of which are plausible API-contract-drift
failure modes for a third-party endpoint), `.items()` raises `AttributeError`, which is not
in the caught tuple. That exception would propagate out of `build_site_candidates()` into
`ApprovalQueueView.get_context_data()` uncaught, taking down the staff approval-queue page
(the primary admin workflow) on what should be a gracefully-degraded MPC outage/format
change.

**Fix:** Broaden the except clause (or validate the shape defensively):

```python
except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError):
```

or guard `_flatten_mpc_candidates` itself: `if not isinstance(obscode_dict, dict): return {}`.

### WR-02: `get_initial()` pre-fills a `max_length=3` field with arbitrary-length `site_raw` text

**File:** `solsys_code/solsys_code_observatory/views.py:41-48`, `solsys_code/solsys_code_observatory/forms.py:11`

**Issue:** `CreateObservatory.get_initial()` pre-fills `obscode` from `?obscode=` with no
length/shape validation:

```python
initial['obscode'] = self.request.GET.get('obscode', '')
```

The "Create new Observatory" link passes `site_raw` verbatim as `obscode`
(`campaign_tables.py:232-235`), which for an unresolved row is frequently a full site name
(e.g. `'Siding Spring Observatory'`), not a 3-character code. `CreateObservatoryForm.obscode`
is `forms.CharField(max_length=3, min_length=3)`, so the pre-filled value is guaranteed
invalid on first render — the staff user sees a field that already looks "filled in" but
must fully overwrite it, with no indication that the pre-fill is a hint rather than a
starting point. Low-severity UX papercut, not a correctness bug, but worth flagging since it
compounds CR-01/CR-02's confusion (a staff member trying to fix an unresolved site now hits
three separate places where the workflow silently fails to do what it visually implies).

**Fix:** Only pre-fill `obscode` when it plausibly is one (e.g.
`len(raw) <= 4 and raw.isupper()`-ish heuristic), or truncate/uppercase before setting
`initial['obscode']`, or leave it blank and instead show `site_raw` as read-only help text
next to the field.

## Info

### IN-01: `resolve_site()`'s length-guard silently discards the very “name → obscode” lookup this feature exists to enable

**File:** `solsys_code/campaign_utils.py:142-145`

**Issue:** This isn't new code (the `_MAX_OBSCODE_LEN` guard predates Phase 21), but Phase 21
is the first caller that hands `resolve_site()` a value sourced from a name-matching UI
(SITE-01's datalist) rather than a value a human was always expected to type as a literal
code. Documenting this here because it's the root cause underlying CR-01: any future fix to
CR-01 that tries to pass the raw display string through `resolve_site()` directly (rather
than mapping it back to an obscode first) will hit this same length guard and silently fail
for every candidate longer than 4 characters.

**Fix:** No action needed beyond CR-01's fix (map display string → obscode before calling
`resolve_site()`); noted for the fixer's awareness so the CR-01 fix doesn't reintroduce the
same failure via a different code path.

---

_Reviewed: 2026-07-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
