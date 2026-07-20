# Stack Research

**Domain:** Approval-queue site disambiguation (fuzzy match + resolve-or-create) and range/TBD scheduling additions to `CampaignRun`, inside an existing Django + TOM Toolkit app (FOMO) — v2.1 "Uncertain Scheduling & Site Disambiguation" milestone
**Researched:** 2026-07-05
**Confidence:** MEDIUM (fuzzy-matching library choice cross-checked against multiple independent web sources = MEDIUM; the Django widget/schema recommendations are derived directly from this codebase's existing, already-verified conventions = HIGH on their own, but the file's overall confidence rolls up to the lowest input)

This is a **milestone addendum**, not a full project stack. It covers only the two new
capabilities v2.1 needs: (a) fuzzy-matching a free-text site name against
`Observatory.name`/`short_name`/`old_names` for the approval-queue site-disambiguation
dropdown, and (b) representing a date window + "TBD" state on `CampaignRun`. Everything
already validated in v1.0–v2.0 (Django 5.2.15, `django-tables2` 3.0.0, `django-filter` 24.3,
`django_htmx`, Bootstrap4/crispy-forms, SQLite, the `resolve_site()` 3-tier resolver) is
unchanged and not re-justified here.

## Bottom Line

**Exactly one new third-party package is needed: `rapidfuzz`.** Everything else — the
searchable/creatable dropdown and the date-window/TBD representation — is coverable with
plain Django (`forms.Form`, `forms.ChoiceField`, nullable `DateField`s, `UniqueConstraint`,
`CheckConstraint`) following patterns already shipped in this codebase (`CampaignRunSubmissionForm`
as a plain `forms.Form`; `ApprovalQueueTable.render_actions`'s per-row CSRF-protected mini-form).
This matches the project's existing bias toward plain Django models/forms over generic
frameworks (v2.0's STACK.md reached the same "no new packages" conclusion for four of its five
features).

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `rapidfuzz` | `>=3.9` (current: 3.14.5, released 2026-04-07) | Fuzzy string matching of a submitter's free-text site name against `Observatory.name`/`short_name`/`old_names`, feeding the approval-queue candidate dropdown | MIT-licensed, zero runtime dependencies, C++-backed with prebuilt wheels for cp310–cp312 on Linux/macOS/Windows (no compiler needed in CI or `.setup_dev.sh`). At "a few hundred `Observatory` rows," raw speed is irrelevant — **match quality** is what matters: `rapidfuzz.fuzz.WRatio`/`token_sort_ratio` handle word-order differences and partial-name matches ("Cerro Tololo" vs. "Cerro Tololo Inter-American Observatory") far better than stdlib `difflib.SequenceMatcher`, whose longest-common-subsequence approach scores transposed/partial matches poorly. |

No other new core technology is needed for this milestone.

### Supporting Libraries / Built-ins

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `rapidfuzz.process.extract` (part of `rapidfuzz`) | 3.x | Rank multiple `Observatory` candidate strings against one query and return the top-N above a score cutoff | Build one `(observatory, candidate_string)` pair per `name`, per `short_name`, and per comma/semicolon-split token in `old_names` (a free-text `TextField`, per its `models.py` definition), then call `process.extract(query, candidates, scorer=fuzz.WRatio, limit=5, score_cutoff=60)` once per unresolved approval-queue row. Dedupe results back to `Observatory` by pk before rendering, since one Observatory can appear via more than one of its 3 name fields. |
| Django ORM (`forms.Form`, `forms.ChoiceField`, `forms.CharField`) | Django 5.2.15 (already installed) | Per-row site-resolution mini-form: a dynamic `ChoiceField` populated with the top-N fuzzy candidates (value = `Observatory.pk`) plus one sentinel choice (e.g. `'__new__'`), paired with an optional free-text `CharField` used only when the sentinel is chosen | This is the "searchable/creatable dropdown" — a plain `forms.Form` (not a `ModelForm`), matching the existing `CampaignRunSubmissionForm` convention (`campaign_forms.py`) of hand-built plain Forms for flows that don't map 1:1 onto a single model. A `ModelForm` on `CampaignRun` can't cleanly express "resolve to an existing FK, or fall through to creating a brand-new row of a *different* model (`Observatory`)" in one field. |
| Django ORM (`models.DateField(null=True, blank=True)` x2, `UniqueConstraint`, `CheckConstraint`) | Django 5.2.15 (already installed) | Represent `window_start`/`window_end` on `CampaignRun`, with "TBD, no dates yet" expressed as both fields `NULL` (no separate boolean flag needed) | No new package required. `DateField` (not `DateTimeField`) matches the window's actual granularity — a window is "this night" or "Aug 1–15," never a time-of-night, which only exists once a run is concretely scheduled (`ut_start`/`ut_end` stay as-is for that). A `CheckConstraint` (`window_end__gte=F('window_start')`, only enforced when both are non-null) is a free, DB-level correctness guarantee, no library needed. |
| `django_htmx` (already installed, used by `fomo_render_calendar` since Phase 12) | current pinned version | Optional: could power a live "re-search as you edit the free-text fallback" interaction | Not required for MVP scope — the fuzzy match runs once, server-side, against the already-submitted `site_raw` string at page-render time. Only reach for this if a future iteration needs the operator to *retype* a query and re-fetch candidates without a full page reload; the v2.1 approval-queue UI does not need it. |

## Installation

```bash
# Core
pip install "rapidfuzz>=3.9"
```

No dev-dependency or `INSTALLED_APPS` changes needed — `forms.Form`/`DateField` are already
part of the installed Django. The mini-form-in-a-table-column pattern extends
`ApprovalQueueTable` (already in `solsys_code/campaign_tables.py`), which already renders a
CSRF-protected mini-form per row (`render_actions`, using `get_token(self.request)`); the new
site-resolution column follows that exact pattern rather than introducing a new mechanism.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| `rapidfuzz` | `thefuzz` (formerly `fuzzywuzzy`) | Never, for this project: `thefuzz` is GPL-licensed (FOMO is MIT), and its current implementation just wraps `rapidfuzz` internally for its scoring functions anyway — using it here would add a GPL dependency and an extra indirection layer for no benefit. |
| `rapidfuzz` | stdlib `difflib.get_close_matches` | Only if the project had a hard "zero new dependencies, ever" rule. It would technically work at this scale, but `difflib`'s `SequenceMatcher`-based ratio is materially worse at partial/reordered-name matching than `rapidfuzz`'s `WRatio` (which combines full-string, partial-string, and token-sort comparisons). Given `rapidfuzz` is a zero-dependency, wheel-distributed, MIT-licensed single `pip install`, there's no real cost to preferring it. |
| Plain `<select>` + free-text `<input>` (server-rendered, per-row) | `django-select2` (v8.4.8, MIT, actively maintained by `codingjoe/django-select2`) | Only if the dropdown needed to search/AJAX-query across a *large* (thousands+) or growing-without-bound candidate set live in the browser. At "a few hundred `Observatory` rows" with the list already pre-filtered server-side to the top ~5 fuzzy matches per row, a plain HTML `<select>` has nothing meaningful left for a JS widget to add — and it would introduce a new static-assets pipeline (JS/CSS bundle, its own template tags) inconsistent with this codebase's existing convention of hand-written vanilla-JS IIFEs for interactive behavior (Phase 9's click-to-filter legend). Revisit only if a future feature needs a *global*, unfiltered, type-to-search picker over all `Observatory` rows (not just a handful of pre-ranked candidates). |
| Two nullable `DateField`s + `NULL`-means-TBD | `django.contrib.postgres.fields.DateRangeField` / a dedicated date-range package | Never for this project as currently deployed: `DateRangeField` requires the PostgreSQL backend (`psycopg2` range types), and FOMO's dev/production DB is SQLite (`src/fomo_db.sqlite3`, per `CLAUDE.md`). Would only become relevant if FOMO migrates off SQLite to Postgres — not in scope here. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| `thefuzz`/`fuzzywuzzy` | GPL license (FOMO is MIT); historically needed a C-extension (`python-Levenshtein`) to reach acceptable speed, and even without it now just re-wraps `rapidfuzz` | `rapidfuzz` directly |
| A full-text-search engine (Postgres `tsvector`, Elasticsearch, `django-haystack`/`whoosh`, `pg_trgm`) | Massive overkill for ranking a handful of candidates out of a few hundred `Observatory` rows on every render; SQLite (current backend) also has no native trigram/FTS support wired into this project | `rapidfuzz.process.extract` computed in-Python per request — trivially fast at this scale, no index to build or maintain |
| `django.contrib.postgres.fields.DateRangeField` / any Postgres-only range type | Requires the PostgreSQL backend; FOMO runs SQLite | Two plain nullable `DateField`s (`window_start`, `window_end`) |
| A JS autocomplete/select widget library (`django-select2`, `django-autocomplete-light`, Chosen, Selectize) for the approval-queue dropdown | Adds a new static-asset pipeline and JS dependency for a widget rendering ≤5 pre-filtered options per row — no UX gain over a plain `<select>` at this scale, and inconsistent with the codebase's existing vanilla-JS convention | Server-rendered `<select>` (candidates already ranked/filtered by `rapidfuzz` before the template ever sees them) + a plain text `<input>` fallback |
| A new field on `CampaignRun` for "is this a space mission" | The milestone explicitly reuses `Observatory.observations_type` (`SATELLITE_OBSTYPE`, already defined) — a run's asset type is derived from its resolved `site`, not stored redundantly on `CampaignRun` | Compute `run.site.observations_type == Observatory.ObservationsType.SATELLITE_OBSTYPE` (or equivalent) at read time |

## Stack Patterns by Variant

**If the approval-queue site column needs to distinguish "no good candidate found" from "candidates found, pick one":**
- Use `rapidfuzz`'s `score_cutoff` parameter on `process.extract` — below cutoff, render the existing "unresolved, pending review" state (already implemented in `campaign_tables.py`'s `render_site`); at/above cutoff, render the new dropdown.
- Because this reuses the exact `site_needs_review` semantics already shipped in quick task `260705-l1v` rather than inventing a third state.

**If a fuzzy match must be resolved against `Observatory.old_names` (a free-text field, not a list):**
- Split on comma/semicolon before matching (`re.split(r'[,;]', old_names)`), stripping whitespace per token, and match each token individually against the query.
- Because `old_names` is documented as "Any previous names used by the observatory" (plain `TextField`) with no enforced delimiter convention beyond what existing data uses — matching the whole blob as one string would dilute the match score against a single name buried in a longer comma-separated list.

**If a space-mission `Observatory` needs an obscode longer than 4 characters (e.g. JWST's `'500@-170'`):**
- This is a schema-widening question (`Observatory.obscode` is `CharField(max_length=4)`) explicitly flagged for the milestone's phase-time spike, not a stack/library choice — no new package is implicated either way (widening `max_length` is a plain migration).

## Version Compatibility

| Package | Compatible With | Notes |
|---------|------------------|-------|
| `rapidfuzz>=3.9` | Python 3.10, 3.11, 3.12 | Matches this project's tested CI matrix (`.github/workflows/`); prebuilt wheels exist for all three, no compiler toolchain required in CI or dev setup (`.setup_dev.sh`). |
| `rapidfuzz` | Django 5.2.15, `django-tables2` 3.0.0 | No interaction — `rapidfuzz` is a pure computation library called from view/table-column code before templates render; it has no Django-specific integration surface. |
| Nullable `DateField` pair + `UniqueConstraint` | SQLite (current backend) | SQLite (like Postgres) treats `NULL` as never equal to another `NULL` in a `UniqueConstraint`, so multiple `CampaignRun` rows with `window_start=NULL` (all-TBD rows) do **not** collide against each other on that column — relevant input for the milestone's phase-time spike deciding the replacement natural key for TBD rows, but the exact key composition is a spike decision, not settled here. |

## Sources

- [RapidFuzz · PyPI](https://pypi.org/project/RapidFuzz/) — version/license/dependency confirmation (MEDIUM confidence, web search, cross-checked against GitHub and deps.dev)
- [RapidFuzz versus FuzzyWuzzy](https://plainenglish.io/python/rapidfuzz-versus-fuzzywuzzy) and [fuzzywuzzy vs rapidfuzz vs thefuzz](https://piptrends.com/compare/fuzzywuzzy-vs-rapidfuzz-vs-thefuzz) — licensing/maintenance comparison (MEDIUM confidence, cross-checked across independent sources)
- [django-select2 · PyPI](https://pypi.org/project/django-select2/) — version/maintenance status for the rejected-alternative writeup (MEDIUM confidence)
- This repository: `solsys_code/campaign_tables.py` (`ApprovalQueueTable.render_site`/`render_actions` mini-form + CSRF pattern), `solsys_code/campaign_utils.py` (`resolve_site` 3-tier resolution, `_MAX_OBSCODE_LEN` guard), `solsys_code/solsys_code_observatory/models.py` (`Observatory.name`/`short_name`/`old_names`/`observations_type` field definitions), `solsys_code/models.py` (`CampaignRun`'s current `obs_date`/`ut_start`/`ut_end` fields and its `(campaign, telescope_instrument, ut_start)` `UniqueConstraint`), `src/fomo/settings.py` (`INSTALLED_APPS`, confirming no existing fuzzy-match or JS-select-widget dependency), `pyproject.toml` (current dependency set) — HIGH confidence, direct codebase read
- `.planning/PROJECT.md` (v2.1 "Current Milestone" section) — HIGH confidence, project-internal source of truth for this milestone's target features and open spike questions

---
*Stack research for: FOMO v2.1 "Uncertain Scheduling & Site Disambiguation" — site-disambiguation dropdown + date-window scheduling*
*Researched: 2026-07-05*
