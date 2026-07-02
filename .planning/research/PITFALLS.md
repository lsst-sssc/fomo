# Pitfalls Research

**Domain:** Adding community campaign-coordination features (data model + PII contact fields, one-off CSV bootstrap import, per-target campaign table, public submission form + approval queue, ephemeris-aware coverage-gap analysis) to the existing FOMO Django + TOM Toolkit app for milestone v2.0
**Researched:** 2026-07-02
**Confidence:** HIGH for facts read directly from this repo's installed code (`tom_calendar.models.CalendarEvent`, `solsys_code/calendar_utils.py`, `solsys_code/views.py` imports, `src/fomo/settings.py`, installed package list, `.planning/PROJECT.md`/`CLAUDE.md`) and for the CLAUDE.md-convention collision (Pitfall 1); MEDIUM for general Django/CSV/concurrency best-practice claims sourced from web search (blog posts / community threads, not official docs, but consistent across multiple independent sources).

**Method:** Read `.planning/PROJECT.md`, `.planning/codebase/CONCERNS.md`, `.planning/seeds/target-linked-run-submission-form.md`, `CLAUDE.md`, `tom_calendar/models.py` (installed `tom_calendar` package), `solsys_code/models.py`, `solsys_code/calendar_utils.py`, `solsys_code/forms.py`, and the import block of `solsys_code/views.py`. Confirmed `src/fomo/settings.py`'s `AUTH_STRATEGY`/`TARGET_PERMISSIONS_ONLY` values and that no CAPTCHA/honeypot/rate-limit package is installed in this repo's venv. Cross-checked general Django spam-prevention, `select_for_update()` concurrency, and messy-CSV-import practices via web search.

## Critical Pitfalls

### Pitfall 1: Committing real people's PII into git history via the paired-demo-notebook convention

**What goes wrong:**
CLAUDE.md has a hard, repeatedly-enforced convention: modules like `load_telescope_runs.py` and the sync commands each ship a paired demo notebook under `docs/notebooks/pre_executed/`, and — unlike every other notebook in the repo — pre-commit does **not** clear output from files in that directory; they are committed *with* real executed output. The one-off CSV bootstrap importer for the real 3I/ATLAS sheet (`.planning/seeds/target-linked-run-submission-form.md`) is exactly the shape of module this convention applies to (a new management command whose behavior needs demonstrating), and its "real executed output" is, by construction, the actual contact names and emails from the real community spreadsheet. Following the existing convention literally means permanently committing real people's names/emails into `docs/notebooks/pre_executed/*.ipynb` output cells — a much sharper problem than the PII-on-a-public-page question the milestone already flags, because it lands in *git history*, which is far harder to scrub than a template.

**Why it happens:**
The paired-notebook rule was written (and enforced twice already, per CLAUDE.md's own account of quick tasks 260619-f7u and 260620-v9x) for modules whose demo data is either synthetic or already-public (LCO/Gemini API responses, classical schedule lines). It was never evaluated against a module whose "real data" is a batch of private contact PII. A plan that mechanically applies "new module → paired notebook with real executed output" without pausing on this specific case will reproduce the exact CLAUDE.md gap pattern it warns about, just with a privacy consequence instead of a staleness one.

**How to avoid:**
Decide explicitly, before writing the bootstrap-import phase's plan, how its demo notebook (if any) satisfies the convention without leaking PII: (a) demo against a synthetic/redacted CSV fixture with fake names/emails and keep the real 3I sheet import as a one-off, uncommitted/interactively-run operation, or (b) redact the PII columns in the notebook's displayed output (e.g. `.head()` on a PII-scrubbed subset, or print row counts/status distributions instead of raw rows) while still exercising the real file for correctness, or (c) get explicit sign-off that this module is an exception to the paired-notebook convention and document why in CLAUDE.md itself (mirroring how other project-specific exceptions are documented). Do not let "the convention says every new module gets a notebook with real output" default to committing real PII.

**Warning signs:**
Any notebook cell that calls `.head()`, `print()`, or renders a QuerySet/DataFrame containing a `contact_email`/`contact_name`-shaped column against the real 3I sheet, committed under `docs/notebooks/pre_executed/`. A plan whose `files_modified` includes a new pre-executed notebook for the CSV importer without a corresponding task deciding the redaction/fixture strategy.

**Phase to address:**
The campaign-run data model + bootstrap import phase — this must be resolved as a design decision in that phase's planning/discussion step, not discovered during code review after the notebook is already committed (git history retention makes this a HIGH recovery-cost pitfall, see Recovery Strategies).

---

### Pitfall 2: PII exposure on the campaign table view via FOMO's OPEN/READ_ONLY permission model

**What goes wrong:**
`AUTH_STRATEGY = 'READ_ONLY'` and `TARGET_PERMISSIONS_ONLY = True` (confirmed in `src/fomo/settings.py`) mean FOMO's default posture is "any page reachable without a specific auth-gate is visible to anonymous visitors, by design." A per-target campaign table view built by pattern-matching the existing target-detail page (itself openly viewable) will inherit that openness unless the contact-email/contact-name fields are explicitly excluded from the default template context or explicitly gated behind an authentication/permission check. The seed document already names this as an open question ("who can see submitter emails needs an explicit decision") — the pitfall is a plan that ships the table view first and treats email-visibility as a follow-up polish task, at which point real emails have already been served to anonymous requests in production.

**Why it happens:**
FOMO's other list/detail views (calendar, ephemeris, observatory) intentionally have no PII to guard, so there's no existing precedent in this codebase for a template that must render some fields to everyone and withhold others. The natural implementation path (copy an existing table/list view template) carries none of the necessary gating logic, and gating is easy to add as an afterthought that "just needs a `{% if %}`" — except by then the field has likely already been included in the view's queryset/serialized context, which is the more consequential place to guard it.

**How to avoid:**
Decide the PII-display policy (per the seed's options: auth-gated column, opt-in display, or store-but-never-render) as part of the campaign-run data model phase, before the table view is built — not during it. Whatever policy is chosen, enforce it at the view/context-building layer (never pass the raw contact field into the template context for an unauthenticated request) rather than only at the template layer, so a future template change can't accidentally re-expose it. Add a test that renders the campaign table view as an anonymous client and asserts the contact email string never appears in the response body.

**Warning signs:**
A `CampaignRun` queryset passed directly to a template with no `.values()`/serializer restriction based on `request.user`; a template `{% if user.is_staff %}` guard around *display* of a field that was already fetched into context (the field is still in the rendered HTML source via other means, e.g. a hidden form field or JSON blob, if the guard is template-only).

**Phase to address:**
Data-model phase (policy decision) and campaign-table-view phase (enforcement); verified by an anonymous-client test in whichever phase ships the first user-facing render of the field.

---

### Pitfall 3: A new campaign-approval writer fighting the existing idempotent, no-churn calendar sync commands

**What goes wrong:**
`solsys_code/calendar_utils.py::insert_or_create_calendar_event()` is the single shared create-or-update path used by `load_telescope_runs`, `sync_lco_observation_calendar`, and `sync_gemini_observation_calendar` — each keyed on a specific, carefully-chosen lookup (`(telescope, instrument, start_time)` for classical runs, `url` for LCO/SOAR, a hand-built `GEM:{prog}/{observation_id}` string for Gemini) with a no-churn field-diff before any `save()`, specifically to avoid modifying `CalendarEvent.modified` on unchanged records and to avoid duplicate events on re-run. If an approved `CampaignRun` also needs to appear as a `CalendarEvent` (the seed explicitly floats this — "does any of it flow back into the calendar display"), a naive implementation that calls `CalendarEvent.objects.create(...)` directly, or that reuses one of the existing lookup keys (e.g. `url`) for an unrelated purpose, either bypasses the no-churn discipline entirely or risks a key collision with a legitimate sync-command-created event for the same telescope/time (e.g. a PI who both submits a community campaign entry *and* later gets the run through the LCO queue for the same target).

**Why it happens:**
`CalendarEvent` (from `tom_calendar`, a third-party model) has no `Target` foreign key at all — confirmed by reading its model source — so a campaign feature that wants a target-linked run naturally reaches for a *new* model (`CampaignRun`, as the seed recommends) rather than widening `CalendarEvent`. That's the right call, but the moment the campaign feature also wants calendar visibility, it's tempting to write directly to `CalendarEvent` by copy-pasting from one of the three existing commands without routing through `insert_or_create_calendar_event()` or without picking a namespaced key (e.g. `CAMPAIGN:{campaign_run.pk}`) that can never collide with an LCO/Gemini/classical-run key.

**How to avoid:**
If the campaign feature writes to `CalendarEvent` at all, route it through `insert_or_create_calendar_event()` for the no-churn/create-or-update contract, and pick a `url`/lookup-key namespace that is provably disjoint from the three existing schemes (LCO's `get_observation_url()` output, Gemini's `GEM:` prefix, classical runs' `(telescope, instrument, start_time)` tuple) — e.g. `CAMPAIGN:{campaign_run.pk}`. Treat "does an approved CampaignRun create a CalendarEvent" as an explicit design decision for the data-model/submission-form phases, not an implicit yes.

**Warning signs:**
Any new code path calling `CalendarEvent.objects.create()` or `.get_or_create()` directly instead of `insert_or_create_calendar_event()`; a lookup key for campaign-originated events that isn't prefixed/namespaced distinctly from the three existing schemes; no test asserting that running `sync_lco_observation_calendar`/`sync_gemini_observation_calendar` after a campaign-originated `CalendarEvent` exists doesn't touch or duplicate it.

**Phase to address:**
Whichever phase decides calendar integration for approved campaign runs (likely the submission-form/approval-queue phase, or deferred explicitly if the campaign table view is judged sufficient without calendar integration) — this should be a named decision, not an implicit consequence of "the table view links to the calendar."

---

### Pitfall 4: New campaign code transitively importing `ephem_utils.py` and paying the 1.6 GB SPICE cost

**What goes wrong:**
`solsys_code/views.py` imports `spiceypy`, `sorcha.ephemeris.*`, and other heavy dependencies at module scope (confirmed directly in its import block), and `solsys_code/ephem_utils.py` downloads ~1.6 GB of SPICE kernels at *import* time, not first-use — a fact CLAUDE.md, the milestone brief, and `.planning/codebase/CONCERNS.md` all flag independently. The coverage-gap-analysis feature is the obvious candidate to need this (it must know where the object is and whether it's observable), but the risk is broader: any new campaign view/model file that does something as innocuous as `from solsys_code.views import <helper>` for an unrelated utility, or that gets test-collected by `./manage.py test solsys_code` (which imports every app module), pays the same cost. `telescope_runs.py` was deliberately written to *avoid* importing `ephem_utils` for exactly this reason (documented in `.planning/PROJECT.md`'s Context section) — new campaign code needs the same discipline applied consciously, not by accident.

**Why it happens:**
Django app-wide imports make it easy to reach for "the existing ephemeris function" without checking what it drags in transitively; the cost is invisible in a quick local test run once the kernel cache is warm (`~/.cache/sorcha/`), so a developer's own dev environment won't surface the problem — it only bites on a clean CI runner, a fresh contributor's machine, or a network-constrained environment.

**How to avoid:**
For any campaign-code ephemeris need, first check whether `solsys_code/telescope_runs.py`'s lightweight `astropy`-only helpers (sun events, dark windows) are sufficient — they likely are for "is this target's site dark and open at this time" style gap detection. If genuine target-position/visibility ephemeris (RA/Dec, airmax) is required, isolate the import behind a lazy, function-local import (not module-level) so merely importing the campaign module (e.g. for URL routing or admin registration) doesn't trigger the kernel download, and confirm no new campaign test file imports `ephem_utils`/`views` at module scope without deliberately accepting the cost.

**Warning signs:**
A `grep -rn "import.*ephem_utils\|from solsys_code.views import\|import spiceypy\|import sorcha" solsys_code/campaign*.py solsys_code/tests/test_campaign*.py` hit at module scope in new files; a CI run for the campaign phase suddenly taking multiple extra minutes / requiring network access it didn't before.

**Phase to address:**
Coverage-gap-analysis phase primarily (it's the feature most likely to need real ephemeris); worth a quick check in every other campaign phase too, since the risk is accidental transitive import, not just deliberate use.

---

### Pitfall 5: Computing coverage-gap ephemeris synchronously inside the request-response cycle

**What goes wrong:**
Even the existing `MakeEphemerisView` — FOMO's only current ephemeris-computation entry point — is synchronous and, per `.planning/codebase/CONCERNS.md`, already flagged as blocking the request thread for large date ranges (ASSIST/Sorcha n-body integration is CPU-heavy, and ASSIST + REBOUND run in-process). Coverage-gap analysis is described in the seed as "FOMO's differentiator" precisely because it's ephemeris-aware — but if it's implemented as "compute observability for every date in range, on every page load of the campaign table view," it multiplies the exact cost pattern CONCERNS.md already warns about, on a page that (unlike the ephemeris form) is meant to be a default, frequently-visited view rather than an explicit one-shot user action.

**Why it happens:**
The feature is naturally described as "show observable-but-unclaimed dates" as if it were a simple lookup, when the underlying computation (site geometry + dark windows + target ephemeris over a date range) is the same class of expensive operation the ephemeris form already treats as a deliberate, submit-triggered action — not something to compute implicitly on every GET request.

**How to avoid:**
Scope coverage-gap computation to an explicit user action (a button/endpoint separate from the default table-view render) rather than inline in the table view's `get()`; cache computed results (per target, per date-range, with a TTL or explicit invalidation on new `CampaignRun` creation) rather than recomputing per visit; and if full n-body ephemeris precision isn't actually required for a "dark and above horizon" gap check, prefer `telescope_runs.py`'s cheaper sun/dark-window helpers over the full `ephem_utils` pipeline (this also sidesteps Pitfall 4). The milestone brief itself already flags this feature as "scoped last so it can defer to v2.1 if needed" — treat that as license to keep the first version's date range and computation narrow rather than building an always-on background recomputation system.

**Warning signs:**
A campaign-table-view request that measurably takes seconds under load testing; no caching layer between the coverage-gap computation and the view; a date-range parameter with no upper bound (users could request a full year of gap analysis on one page load).

**Phase to address:**
Coverage-gap-analysis phase, and explicitly informed by the milestone's own note that this phase can be deferred to v2.1 — a plan that ships it as an always-on, inline-computed feature undermines that intentional scoping decision.

---

### Pitfall 6: Messy real-world CSV import repeats the project's own v1.2 "guessed shape" bug pattern

**What goes wrong:**
This project has already been burned twice by shipping code against an assumed data shape instead of a confirmed one — v1.2's flat `instrument_type` key that doesn't exist in real LCO submissions, and v1.3's unconfirmed `SITE_TELESCOPE_MAP` entries — both documented at length in `.planning/PROJECT.md`'s Key Decisions table as bugs found only once real data was checked. The 3I/ATLAS bootstrap CSV import is exactly this risk again, and arguably worse, because spreadsheet data entered by many different human contributors over time is *reliably* messier than an API response: inconsistent date formats in the same column ("July 5", "2026-07-05", "7/5/26"), merged concepts in a single free-text cell (e.g. one "Contact" cell containing both a name and an email, or multiple observers' emails comma/slash-separated in one field), a free-text "status" column with no controlled vocabulary (real values likely won't cleanly map to `planned`/`observed`/`reduced`/`published`), blank/merged header rows or trailing summary rows common in Google Sheets CSV exports, and non-ASCII characters in names. A parser written against an *imagined* shape of "what the sheet probably looks like" — rather than the actual downloaded CSV — will silently mis-map or `KeyError` on exactly the rows that matter, the same way v1.2 did.

**Why it happens:**
The Google Sheet's field inventory is already documented in the enriched seed, which makes it tempting to write the importer against that field *list* without first opening the actual CSV export and checking every column's real value distribution, especially for the status and date columns where free text is expected.

**How to avoid:**
Download and directly inspect the real CSV (not a mocked/imagined version of it) before writing the parser, mirroring the discipline the project applied (eventually) to `SITE_TELESCOPE_MAP`. Build the import the same way `load_telescope_runs` handles malformed classical-schedule lines: per-row try/except that logs and skips (or routes to a "needs manual review" bucket) rather than aborting the whole import or crashing on the first unparseable row. Use `pandas.to_datetime(..., errors='coerce')`-style tolerant date parsing and explicitly report (not silently drop) any row that fails to parse. For the free-text status column, build an explicit mapping table from observed real values to the four lifecycle states, with an explicit "unknown/other" fallback rather than a `KeyError` or a silent default to `planned`. Treat this as a one-off script with an operator-in-the-loop review step (a dry-run/report mode showing skip counts and reasons) before any final commit to the database, not a fully-automated unattended import.

**Warning signs:**
A CSV-parsing function with no `try/except` per row; a status-mapping dict built from the seed's documented lifecycle names without cross-checking the sheet's actual free-text values; any test fixture for the importer that's hand-typed to look like "clean" data rather than a excerpt of the real file's messiest rows (merged cells, blank rows, non-ISO dates).

**Phase to address:**
The bootstrap-import phase — should include an explicit research/inspection step (open the real CSV, catalog its actual date formats and status vocabulary) before the parsing-logic task is planned, not discovered mid-implementation.

---

### Pitfall 7: Approval-queue race conditions masked by SQLite's dev-environment concurrency behavior

**What goes wrong:**
Two admins (or one admin double-clicking, or a slow page + a retry) approving the same pending submission concurrently is a classic read-modify-write race: both requests read `status='pending'`, both decide to transition it, both write `status='approved'` and (per Pitfall 3) potentially both create a duplicate `CalendarEvent`. This project's dev database is SQLite (per CLAUDE.md and `.planning/codebase/CONCERNS.md`, which already flags SQLite's "concurrent write limitations" as a known production risk), and SQLite's locking model means this class of bug is easy to *not* notice in local dev/manual QA — file-level locking effectively serializes writes in a way that can mask a race that would manifest under real concurrent load on a production Postgres deployment (or even under SQLite's own `database is locked` errors, which surface as a different symptom — request failures — rather than duplicate data).

**Why it happens:**
The approval action is naturally implemented as "look up the object, check its status, change it, save" in a view — a pattern that works correctly under sequential manual testing (the only kind SQLite-backed local dev testing usually exercises) and gives no visible signal that it's unsafe under concurrency.

**How to avoid:**
Implement the approval transition as an atomic, conditional update — either `CampaignRun.objects.filter(pk=pk, status='pending').update(status='approved', ...)` and checking the returned row count (works identically on SQLite and Postgres, no explicit locking needed), or `select_for_update()` inside `transaction.atomic()` if a more complex multi-step approval action is needed (note: `select_for_update()` is effectively a no-op under SQLite, so if this path is chosen, the safety-net check should be the filtered-update pattern regardless, to avoid a false sense of protection in dev testing). Write a test that simulates the double-approval case explicitly (two sequential calls to the approval endpoint/service function on the same pending record) and asserts the second call is a no-op — don't rely on manual QA, which is exactly the testing mode this pitfall is invisible under.

**Warning signs:**
An approval view/service function with a plain `obj.status = 'approved'; obj.save()` and no filtered-update or row-count check; no test exercising "approve the same submission twice."

**Phase to address:**
Submission-form + approval-queue phase — the conditional-update pattern should be part of the initial implementation, not a hardening pass added after a production duplicate is reported.

---

### Pitfall 8: Silent-rejection UX and admin-notification gaps undermine the feature's entire motivation, compounded by zero anti-spam infrastructure on an open form

**What goes wrong:**
The seed's stated motivation is explicitly that community coordination "happens fast" and a web form should "lower the barrier to entry without bypassing oversight" — but if the approval queue has no notification path (an open question the seed itself flags: "How does the admin get notified?"), submissions can sit unseen indefinitely, defeating the "fast coordination" goal the feature exists for. Symmetrically, if a submitter never learns their submission was rejected (or is still pending), the UX looks identical to the form silently doing nothing — worse than the ad-hoc Google Sheet it's meant to replace, where at least edits are visible immediately to everyone. Compounding this: this repo's venv currently has **no CAPTCHA, honeypot, or rate-limiting package installed** (confirmed by a package-list check) — an open, public, low-friction submission form (explicitly *not* meant to require heavyweight gatekeeping, per the seed's "lowers the barrier to entry" goal) is a plausible spam-bot target, and without any admin-notification filtering, a wave of bot submissions could bury genuine community submissions in the same unfiltered queue that Pitfall 8's first half already risks leaving unmonitored.

**Why it happens:**
Notification and anti-spam are both "infrastructure" concerns that are easy to defer past the MVP submission-form-plus-queue implementation, especially since the moderation step (approval before public visibility) already prevents spam from becoming *visible* — but it doesn't prevent spam from silently drowning out real submissions in an unmonitored queue, which is a distinct failure mode from public visibility.

**How to avoid:**
At minimum, add a low-friction honeypot field (a hidden input real users never fill, bots typically do — no CAPTCHA/JS challenge required, consistent with the "low barrier to entry" goal) to catch the cheapest bot traffic before it reaches the queue at all. Decide and implement *some* admin-notification path proportionate to expected volume — even a simple email to a configured campaign-admin address (following the existing `os.getenv()`-based settings pattern already used for other credentials in this repo) is enough to prevent Pitfall 8's silent-accumulation failure mode; a full in-app notification system is not required for v2.0. For submitter-facing status, consider a private, unguessable status-lookup link included in the submission confirmation (works for both authenticated and anonymous submitters) rather than requiring an account, so "pending"/"rejected" isn't indistinguishable from "vanished."

**Warning signs:**
A submission form + approval queue with no email/notification hook anywhere in the code; no honeypot or equivalent field on the public form; a rejected submission that leaves no trace visible to the submitter; queue-size growing unmonitored in manual testing with no operator-facing signal.

**Phase to address:**
Submission-form + approval-queue phase — both the honeypot field and the notification hook are small, cheap additions that are far more expensive to retrofit once the form is live and already accumulating unmonitored submissions.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Writing the 3I CSV parser against the seed's documented field list instead of the real downloaded file | Faster to start coding | Repeats the exact v1.2 "guessed shape" bug — silent mis-mapping or crashes on real messy rows | Never — inspect the real file first (Pitfall 6) |
| Approval transition as a plain `obj.status = X; obj.save()` instead of a filtered/conditional update | Simpler code, passes manual QA on SQLite | Race condition invisible in dev, real duplicate-approval bug in production concurrency (Pitfall 7) | Never for the approval action itself; fine for genuinely single-writer admin-only paths with no realistic concurrent access |
| Deferring admin-notification and honeypot to "a follow-up polish task" | Ships the form faster | Queue silently fills with unmonitored submissions and/or spam, undermining the feature's core "fast coordination" motivation (Pitfall 8) | Only for a true internal-only prototype never exposed publicly; not acceptable for the actual v2.0 public form |
| Committing the bootstrap-import demo notebook with real 3I sheet PII in output cells, "because that's the convention" | Satisfies CLAUDE.md's paired-notebook rule with minimal extra thought | Permanent PII leak into git history, much harder to remediate than a stale notebook (Pitfall 1) | Never — resolve the redaction/fixture strategy explicitly first |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `tom_calendar.models.CalendarEvent` | Assuming it has (or adding) a `Target` FK to link campaign runs directly to it | It has no `Target` field (confirmed by source read); keep the target link on a new `CampaignRun` model instead, per the seed's own recommendation |
| `solsys_code/calendar_utils.py::insert_or_create_calendar_event()` | Writing to `CalendarEvent` directly from new campaign code, bypassing the shared no-churn helper and its lookup-key discipline | Route any campaign-originated `CalendarEvent` writes through this helper with a distinctly-namespaced key (e.g. `CAMPAIGN:{pk}`), never reusing the LCO/Gemini/classical-run key schemes (Pitfall 3) |
| `solsys_code/ephem_utils.py` (SPICE/Sorcha/ASSIST stack) | Importing it (directly or transitively via `solsys_code/views.py`) at module scope from new campaign view/model files | Prefer `solsys_code/telescope_runs.py`'s lightweight astropy-only helpers; if full ephemeris is unavoidable, import it lazily inside the function that needs it, not at module load (Pitfall 4) |
| `AUTH_STRATEGY='READ_ONLY'` / `TARGET_PERMISSIONS_ONLY=True` | Building the campaign table view by copying an existing open target-detail template pattern without adding field-level PII gating | Explicitly exclude/gate the contact-email field in the view's context-building step, verified by an anonymous-client test (Pitfall 2) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Inline coverage-gap ephemeris computation on every campaign-table-view page load | Slow page loads, worker thread blocked, mirrors the existing `MakeEphemerisView` cost already flagged in CONCERNS.md | Compute on explicit user action or cache results with TTL/invalidation; keep initial date-range scope narrow (Pitfall 5) | Any date range beyond a few weeks, or moderate concurrent traffic on the campaign table view |
| Unbounded date range on the coverage-gap request | A user (or bot) requesting a full year of gap analysis in one call | Cap the max date range server-side, independent of any client-side default | First real request outside the happy-path date range used in dev testing |
| Unfiltered/unmonitored approval queue growth from spam submissions | Admin queue fills with bot noise, real submissions buried, DB grows with junk rows | Honeypot field + admin notification (Pitfall 8) | Any exposure of the form to public internet traffic (not just this project's known contributors) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Rendering contact email/name in a template context reachable by an unauthenticated request | Real PII exposed to anyone on an `AUTH_STRATEGY='READ_ONLY'` open site, and to any web crawler/scraper | Field-level gating at the view/context layer, not just the template layer (Pitfall 2) |
| Committing real submitter PII into a pre-executed demo notebook's output cells | Permanent PII leak into git history, unlike a template which can be fixed forward | Redact/synthesize notebook demo data or get an explicit convention exception (Pitfall 1) |
| Shipping the public submission form with zero anti-abuse measures (confirmed: no CAPTCHA/honeypot/rate-limit package installed today) | Spam-bot submissions pollute the DB and admin queue; potential injection of malicious free-text into fields later rendered without escaping | Add a low-friction honeypot field; rely on Django's default HTML auto-escaping for any free-text fields rendered back to users; keep moderation-before-visibility as the second line of defense (Pitfall 8) |
| Storing plaintext contact emails with no minimization/retention policy in a public-facing coordination tool that may collect EU citizens' data | GDPR-adjacent exposure if a submitter requests deletion/correction and there's no path to find/redact their record | Provide a private way for a submitter to view/correct/withdraw their own submission (ties into Pitfall 8's status-lookup recommendation); don't collect more PII fields than the 3I sheet's own field inventory already establishes as necessary |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| Approved-vs-rejected-vs-still-pending is indistinguishable to the submitter (no notification, no status page) | Community members can't tell if their fast-turnaround submission was seen at all, undermining the "lowers the barrier to entry" goal | Give submitters a status-check path (private link/token) even without requiring an account (Pitfall 8) |
| Campaign table view silently omits contact info for anonymous users with no indication *why* | Logged-in-vs-not behavior differences look like a bug rather than an intentional privacy gate | Show a clear "sign in to see contact details" affordance rather than just quietly dropping the column |
| Coverage-gap analysis presented as instant/always-current when it's actually cached or computed on a delayed/manual trigger | Users trust a stale gap analysis as current, potentially double-booking a date that was actually just claimed | Show a "last computed at" timestamp alongside any cached coverage-gap result |

## "Looks Done But Isn't" Checklist

- [ ] **PII gating on the campaign table view:** Often "done" by hiding the column in the template only — verify by checking the view's context/queryset doesn't include the raw field for anonymous requests, and by testing with Django's anonymous test client.
- [ ] **CSV bootstrap import:** Often "done" against a hand-typed clean fixture — verify it was run against (or its logic reviewed against) the actual downloaded 3I sheet CSV, with a reported count of skipped/malformed rows, not just a happy-path row.
- [ ] **Approval queue concurrency:** Often "done" via manual single-admin QA — verify with an automated test that calls the approval action twice on the same record and asserts no duplicate `CampaignRun`/`CalendarEvent` results.
- [ ] **Demo notebook for the CSV importer:** Often "done" by literally running the real import and saving output — verify the committed notebook's output cells contain no real names/emails (per Pitfall 1), and that it's scoped into the phase's `files_modified` from the start per CLAUDE.md's existing convention.
- [ ] **Coverage-gap analysis performance:** Often "done" by confirming it returns correct results on a small manual test — verify it doesn't transitively import `ephem_utils` at module scope, and that its cost is bounded (cached or explicitly triggered, not computed inline on every table-view GET).
- [ ] **Notification/anti-spam on the submission form:** Often "done" by shipping just the form + queue — verify there is at least one honeypot field and one admin-notification mechanism, not deferred as "polish."

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|----------------|-----------------|
| Real PII already committed in a pre-executed notebook's output cells | HIGH | Requires git history rewriting (`git filter-repo`/BFG) across all clones/forks, plus notifying affected individuals if this is a public repo — far more invasive than a normal code fix; strongly prefer prevention (Pitfall 1) |
| PII already rendered to anonymous users in production | MEDIUM-HIGH | Patch the view/template gate immediately; audit access logs/CDN caches for how long the exposure window was; consider whether affected contacts need notification |
| Duplicate `CampaignRun`/`CalendarEvent` rows from an approval race | LOW-MEDIUM | Add the filtered-update fix; write a one-off management command or Django admin action to de-duplicate existing rows, keyed on the same fields the no-churn helper already compares |
| CSV importer shipped against a wrong assumed shape, discovered against real data | MEDIUM | Same recovery shape as the v1.2→v1.3 precedent already in this project's history: re-inspect the real file, fix the mapping, re-run against a truncated/rollback-able transaction so the bad import can be undone |
| Spam submissions flooding the approval queue post-launch | LOW | Add honeypot/rate-limiting retroactively; bulk-reject/delete flagged rows via a management command or admin bulk action |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| PII in committed demo-notebook output (Pitfall 1) | Campaign-run data model + bootstrap import phase | Explicit design decision on notebook redaction/fixture strategy before the notebook is committed; code review checks output cells for PII |
| PII exposed on the campaign table view (Pitfall 2) | Data-model phase (policy) + campaign-table-view phase (enforcement) | Anonymous-client test asserting the contact email never appears in the rendered response |
| New writer colliding with existing calendar sync commands (Pitfall 3) | Submission-form/approval-queue phase (if calendar integration is in scope) | Test that running the existing sync commands after a campaign-originated `CalendarEvent` exists doesn't duplicate/overwrite it |
| Transitively importing `ephem_utils` (Pitfall 4) | Coverage-gap-analysis phase (and a quick check in every other campaign phase) | `grep` for module-scope heavy imports in new campaign files; CI runtime/network-access check |
| Synchronous per-request ephemeris cost (Pitfall 5) | Coverage-gap-analysis phase | Load/timing test on the campaign table view; explicit cache/trigger design reviewed before implementation |
| Messy CSV import mis-mapping real data (Pitfall 6) | Bootstrap-import phase, gated on an explicit real-file inspection step | Dry-run/report mode showing skip counts and reasons, reviewed against the actual 3I sheet before final DB commit |
| Approval-queue race conditions (Pitfall 7) | Submission-form/approval-queue phase | Automated double-approval test, not manual QA alone |
| Silent-rejection UX + spam-bot exposure (Pitfall 8) | Submission-form/approval-queue phase | Honeypot field present; admin-notification hook present; submitter status-check path present |

## Sources

- `.planning/PROJECT.md` — v1.2/v1.3 "guessed shape" bug history, existing `insert_or_create_calendar_event`/no-churn pattern, current milestone scope and open PII question (HIGH confidence, primary project history)
- `.planning/codebase/CONCERNS.md` — SQLite production concurrency risk, `MakeEphemerisView`/`ephem_utils` synchronous-blocking cost, SPICE kernel import-time download (HIGH confidence, direct codebase audit)
- `.planning/seeds/target-linked-run-submission-form.md` — 3I sheet field inventory, PII/open-question framing, submission/approval shape (HIGH confidence, primary source)
- `CLAUDE.md` — paired-demo-notebook convention and its two prior enforcement gaps, `AUTH_STRATEGY='READ_ONLY'`/PII exposure risk framing (HIGH confidence, primary source)
- `tom_calendar/models.py` (installed package, `/home/tlister/venv/fomo_venv/lib/python3.12/site-packages/tom_calendar/models.py`) — confirmed no `Target` FK on `CalendarEvent` (HIGH confidence, direct source read)
- `solsys_code/calendar_utils.py`, `solsys_code/models.py`, `solsys_code/forms.py`, `solsys_code/views.py` import block — confirmed the shared no-churn helper, the `CalendarEventTelescopeLabel` sidecar precedent, and the heavy SPICE/Sorcha import surface (HIGH confidence, direct source read)
- `src/fomo/settings.py` — confirmed `AUTH_STRATEGY='READ_ONLY'`, `TARGET_PERMISSIONS_ONLY=True` (HIGH confidence, direct source read)
- Installed package list (`pip list` in this repo's venv) — confirmed no CAPTCHA/honeypot/rate-limit package currently installed (HIGH confidence, direct environment check)
- [Prevent Spam with Django Honeypot Fields: A Complete Guide](https://webpedia.net/how-to-use-honeypot-fields-in-django-forms) — honeypot field naming/placement practice (MEDIUM confidence, blog source)
- [django-honeypot (GitHub)](https://github.com/jamesturk/django-honeypot) — reference implementation of the honeypot pattern (MEDIUM confidence)
- [Bot protection for publicly accessible form — Django Forum](https://forum.djangoproject.com/t/bot-protection-for-publicly-accessible-form/12080) — community consensus on low-friction anti-spam for public Django forms (MEDIUM confidence)
- [Solving Django race conditions with select_for_update and optimistic updates](https://www.youssefm.com/posts/solving-django-race-conditions) and [Django @atomic Doesn't Prevent Race Conditions: Use select_for_update()](https://medium.com/@anas-issath/djangos-atomic-decorator-didn-t-prevent-my-race-condition-and-the-docs-never-warned-me-58a98177cb9e) — conditional-update/`select_for_update()` pattern for approval-style transitions (MEDIUM confidence, cross-consistent across sources)
- [How to Clean CSV Data: 10 Fixes for Failed Imports](https://www.filefeed.io/blog/how-to-clean-csv-data) and [Quickly Format Dates with Pandas](https://medium.com/@connect.hashblock/quickly-format-dates-with-pandas-1cbd1e66910b) — messy-CSV/inconsistent-date-format handling practice, `pandas.to_datetime(errors='coerce')` (MEDIUM confidence, blog sources, consistent with each other)

---
*Pitfalls research for: FOMO v2.0 Campaign Coordination for Rare/Urgent Objects*
*Researched: 2026-07-02*
