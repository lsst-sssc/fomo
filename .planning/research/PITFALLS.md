# Pitfalls Research

**Domain:** Adding new features to an existing Django/TOM Toolkit app (FOMO) — v2.1 "Uncertain Scheduling & Site Disambiguation" milestone
**Researched:** 2026-07-05
**Confidence:** HIGH (grounded directly in this repo's models/migrations/utils — `solsys_code/models.py`, `solsys_code/campaign_utils.py`, `solsys_code/migrations/0003_campaignrun_natural_key_unique_constraint.py`, `solsys_code/solsys_code_observatory/models.py`/`forms.py` — plus empirically verified SQL NULL-uniqueness behavior and cross-checked Django/PostgreSQL documentation for the constraint-widening fix)

This file supersedes the v2.0-scoped `PITFALLS.md`. It is scoped to the three integration risks named in the v2.1 milestone: (a) migrating `CampaignRun` from a single `ut_start` field to a date-range/TBD representation while `(campaign, telescope_instrument, ut_start)` is a DB-level `UniqueConstraint`; (b) widening `Observatory.obscode` (`CharField(max_length=4, unique=True)`) to fit spacecraft-style codes; (c) adding fuzzy-match site disambiguation UI on top of `resolve_site()`'s existing exact→API→flag 3-tier path.

## Critical Pitfalls

### Pitfall 1: NULLable `ut_start` silently defeats the natural-key `UniqueConstraint`, re-opening the exact race WR-05 was added to close

**What goes wrong:**
`unique_campaign_run_natural_key` is `UniqueConstraint(fields=['campaign', 'telescope_instrument', 'ut_start'])`. Both SQLite and PostgreSQL treat NULL as "not equal to itself" for uniqueness purposes (SQL-standard behavior — confirmed empirically against this repo's own SQLite backend: two rows with identical `(campaign, telescope_instrument)` and `ut_start=NULL` insert without an `IntegrityError`). Once TBD rows genuinely have `ut_start=NULL` (rather than today's midnight-UTC fallback, which is a real, if wrong, timestamp), the DB stops deduplicating them entirely. A double CSV import, a resubmitted approval, or two overlapping imports racing (the exact scenario WR-05 was added in Phase 14 to protect against) can silently create unlimited duplicate TBD `CampaignRun` rows for the same campaign+telescope, and nothing at the DB layer will stop it.

**Why it happens:**
Engineers extend an existing natural key to a "still works, now nullable" field and assume the existing `UniqueConstraint` continues to enforce one-row-per-key, because that's how it worked for every other value the field ever held in this table. NULL is the one value class that was never exercised in Phase 14's real-data testing (all 3I/ATLAS rows had a parseable-or-fallback UT time), so this gap wasn't hit before.

**How to avoid:**
- Give TBD rows a genuinely different natural key, not a NULL `ut_start` sharing the old one. Two workable patterns, decide in the spike:
  - Split the constraint: `UniqueConstraint(fields=[...,'ut_start'], condition=Q(ut_start__isnull=False), name=...)` for scheduled rows, plus a second constraint/model-level dedup path for TBD rows keyed on something that *is* populated for every TBD row (e.g. `(campaign, telescope_instrument, window_start)` if a coarse window start always exists, or an explicit external reference like a CSV row identity/content hash).
  - Django's `condition=` on `UniqueConstraint` works on both SQLite (partial indexes since 3.8.0) and PostgreSQL — this is not a Postgres-only feature, so it's safe for this project's dev-SQLite/prod-Postgres split.
- Do not rely on `get_or_create`'s `.get()` step alone as the safety net — it correctly matches existing NULL rows via `IS NULL` translation, but only protects sequential single-process runs; concurrent imports/approvals still race past it without a DB constraint.
- Write the migration to explicitly test both "two normal rows with different `ut_start`" and "two rows with the new TBD key" cases, not just the always-tested "same key rejected" case.

**Warning signs:**
- A migration that only changes `ut_start` to `null=True` without touching the `UniqueConstraint`/`0003_campaignrun_natural_key_unique_constraint.py` migration at all.
- Tests that assert the constraint blocks a duplicate scheduled row but never assert what happens with two duplicate TBD rows.
- Any code comment claiming "the natural key still protects against duplicate re-imports" without a partial-index condition backing it for the nullable case.

**Phase to address:**
The investigation spike (settles the replacement natural key and window schema) — this must be decided before any implementation phase writes the migration, not discovered afterward via a manual-UAT gap like `260705-l1v`.

---

### Pitfall 2: The CR-02 disambiguating-offset hack breaks once `ut_start` can be a real, structural NULL

**What goes wrong:**
`import_campaign_csv.py` already has a workaround (CR-02, `seen_fallback_keys`) for the *current* meaning of "unparseable time": `parse_obs_window()` always returns a concrete midnight-UTC `ut_start`, so two distinct rows that both fail to parse their UT range would silently collide on the natural key — the fix adds a per-batch second-offset to `ut_start` so they don't merge. That workaround depends on `ut_start` always being a real datetime to offset. Once "no known date/time yet" becomes a legitimate first-class state (TBD rows, not just unparseable text), there is nothing to add seconds to, and the existing collision-avoidance mechanism has no equivalent for the new case — re-importing two different TBD rows for the same campaign+telescope will either collide (Pitfall 1) or need an entirely different disambiguator (e.g. a stored raw-row hash or explicit sheet row reference).

**Why it happens:**
The v2.0 fix was scoped narrowly to the data actually seen in the real sheet at the time (garbled/unparseable UT Time Range text, not literally blank/TBD dates). It's an easy trap to assume the new TBD path can reuse the same "add an offset" trick since both look like "we don't have a real time."

**How to avoid:**
Treat range/TBD parsing as a new case in `parse_obs_window()`'s contract, not an extension of the existing fallback branch — give it its own explicit flag (distinct from `ut_needs_review`) and its own disambiguation strategy decided in the spike (a content hash of the raw CSV cells is simplest and matches the "never silently merge distinct rows" invariant already established in D-05/CR-02).

**Warning signs:**
Re-running `import_campaign_csv` against a CSV with two different TBD/range rows for the same campaign+telescope and getting only one `CampaignRun` back (silent merge) instead of two.

**Phase to address:**
CSV-import-handles-range/TBD-dates implementation phase — write this exact two-distinct-TBD-rows scenario as a required test before considering the feature done.

---

### Pitfall 3: Widening `Observatory.obscode` fixes the model field but leaves at least one hardcoded 3/4-char assumption that will reject or corrupt spacecraft codes

**What goes wrong:**
`campaign_utils.py`'s `_MAX_OBSCODE_LEN` is already computed dynamically from `Observatory._meta.get_field('obscode').max_length`, so `resolve_site()`'s length guard will pick up a widened `max_length` automatically — that part is safe. But `solsys_code_observatory/forms.py`'s `CreateObservatoryForm.obscode` is a **separately hand-declared** `forms.CharField(max_length=3, min_length=3)` with a `clean_obscode()` that force-uppercases the value. This form is the human-facing "create a new Observatory" path — exactly the free-text-fallback path the v2.1 disambiguation UI needs for "explicitly create a new Observatory." Left untouched, a widened model field is still fronted by a form that hard-rejects anything other than exactly 3 characters, so a staff member trying to create an `Observatory` for `'500@-170'` (8 chars) via the existing form gets a validation error, not a working create path — the model migration alone does not deliver the feature.
Additionally, `MPCObscodeFetcher.query()` sends the raw code straight to the real MPC Obscodes API, which only knows real MPC codes — for an 8-character spacecraft-style code, tier 2 will reliably 404/miss (not crash, since `resolve_site()` already treats API misses as fall-through), but it's a wasted round-trip on every space-mission code, every time, unless short-circuited.

**Why it happens:**
The model's `max_length=4` and the form's independent `max_length=3, min_length=3` were never meant to be the same source of truth (the form is stricter than the model already, and nobody previously needed to reconcile them). A migration that only touches `models.py` looks complete because `makemigrations`/`migrate` succeeds and existing tests pass — none of the current 39 `test_campaign_models.py`/`test_import_campaign_csv.py` tests exercise the human-facing create form with a long code.

**How to avoid:**
- Grep for every hardcoded `3`/`4`/`max_length` near `obscode` before considering the widening done: `solsys_code_observatory/forms.py` (`CreateObservatoryForm`), any serializer/admin list truncation, and any test fixture asserting a fixed-width obscode column.
- Decide explicitly (in the spike) whether `CreateObservatoryForm` needs a widened `max_length` too, or whether spacecraft-style `Observatory` rows are created through a different path entirely (e.g. programmatically via the disambiguation UI's "create new" action, bypassing `CreateObservatoryForm`) — don't assume the existing MPC-obscode form is reusable as-is.
- Skip tier 2 (`MPCObscodeFetcher`) outright for codes that don't look like a plausible MPC code (e.g. contain `@` or exceed the *original* 3-4 char MPC convention), even after widening the DB column — MPC will never resolve a spacecraft code, so querying it is pure latency with no chance of success.
- Since SQLite implements `ALTER TABLE ... ALTER COLUMN` by rebuilding the table (Django's SQLite backend does this transparently via migrations), verify the widening migration runs cleanly against the dev DB with existing `Observatory` rows present — this is a full table rewrite, not a metadata-only change like it would be on PostgreSQL for a `VARCHAR` length increase.

**Warning signs:**
- A migration diff that touches only `solsys_code_observatory/migrations/` and `models.py`, with zero changes to `forms.py`.
- Manually exercising "create an Observatory via the web form with an 8-character code" still fails after the migration is applied.
- `MPCObscodeFetcher.query()` being called (and logged) for every space-mission-style code during import, even though it can never succeed.

**Phase to address:**
The investigation spike must explicitly resolve the obscode-length constraint (per the milestone's own framing) — but "resolve" must include the create-form path, not just the model field, or the implementation phase will discover the gap only when a staff member tries to actually create a JWST `Observatory` row through the UI.

---

### Pitfall 4: Fuzzy-match disambiguation UI silently auto-selects a wrong site, undermining the "never fabricate, always flag" invariant `resolve_site()`/`260705-l1v` just established

**What goes wrong:**
`resolve_site()` was deliberately hardened (via quick task `260705-l1v`) so that an unresolvable public-submitted site *never* silently becomes a fabricated `Observatory` row — it now returns `(None, True)` and waits for a human. A fuzzy-match layer that auto-selects the top-scoring `Observatory` candidate above some similarity threshold (rather than always presenting a dropdown for human confirmation) reintroduces exactly the failure mode `260705-l1v` fixed, just one layer up: instead of fabricating a placeholder, it silently picks the *wrong existing* site. This is arguably worse than the placeholder bug, because a wrong-but-real `Observatory` looks correct everywhere downstream (ephemeris/timezone/coverage-gap computations all silently use the wrong site's coordinates) with no `site_needs_review=True` flag to signal it. Concretely: free-text `site_raw` values on the public submission form are up to 255 characters of unvalidated prose (`campaign_forms.py`), so short/ambiguous strings like `'VLT'`, `'Hubble'`, or a truncated/misspelled name are exactly the inputs a naive top-match-wins fuzzy matcher will get wrong most often — and Las Campanas already has two real, confusable sites in this data (Magellan Baade/Clay), so a coincidental fuzzy hit is not a hypothetical.

**Why it happens:**
"Auto-select the best fuzzy match if it's above threshold X" is the natural first implementation of fuzzy matching, because it removes a UI step and demoing it against a handful of near-exact typos ("Maggellan" → "Magellan") looks great. It fails precisely on the inputs that matter most — genuinely ambiguous or unfamiliar site names from external submitters who aren't LCO staff and don't know canonical site names — which aren't well represented in whatever small sample was used to pick the threshold.

**How to avoid:**
- Fuzzy match must always be presented as *candidates for a human to pick*, never as an automatic resolution — the UI's job is exactly what the milestone spec says: "inline dropdown of fuzzy-matched candidates ... never auto-fabricates a placeholder." Extend that same never-auto-decide discipline to never-auto-*select* either — a human still clicks the candidate or explicitly creates a new `Observatory`.
- Keep `resolve_site()`'s exact-match tier 1 and MPC-API tier 2 as the *only* code paths that can set `site`/`site_needs_review` without human interaction; the fuzzy layer only ever populates a suggestions list attached to `site_needs_review=True` rows — it must not call `.save()` on `site` itself.
- Match against `name`, `short_name`, **and** `old_names` (free-text field, comma/semicolon-separated historical aliases per `solsys_code_observatory/utils.py`) since a submitter may type a superseded site name — but normalize/split `old_names` before scoring it as a single blob, or a long aliases string will score poorly against a short typed name purely due to length, hiding a legitimate match.
- Log (without auto-acting on) every fuzzy suggestion shown and whether staff accepted it, so a future threshold-tuning pass has real acceptance-rate data instead of guessing.

**Warning signs:**
- Any code path where a fuzzy-match score above a threshold sets `run.site = candidate` without an intervening staff click.
- A UI that hides the "explicitly create a new Observatory" fallback whenever a fuzzy match scores "high enough," on the theory that showing it is redundant — this removes the human's ability to notice the match is wrong.
- No `site_needs_review=True` on rows where a fuzzy candidate was auto-applied (if auto-apply exists at all, this is retroactively the earlier bug).

**Phase to address:**
The site-disambiguation UI implementation phase, with the never-auto-select rule written into that phase's plan and UAT criteria explicitly (not left as an implicit assumption) — the spike should also pick and record a fuzzy-matching library/approach (none is currently a project dependency; stdlib `difflib.SequenceMatcher` avoids a new dependency but scores differently than a trigram/edit-distance library — pick deliberately, don't default silently).

---

### Pitfall 5: Range/window `CampaignRun` fields silently break `insert_or_create_campaign_run`'s no-churn diff and the natural-key `lookup`/`fields` split

**What goes wrong:**
`insert_or_create_campaign_run(lookup, fields)` treats `lookup` (the natural key) and `fields` (everything else) as disjoint by contract — the caller is explicitly responsible for keeping them that way ("Not merged with `lookup`"). If the window/range representation adds new fields (e.g. `window_start`/`window_end`/`is_tbd`) and the new natural key for TBD rows ends up reusing one of those same fields as a *lookup* key (per Pitfall 1's fix), a caller that also includes it in `fields` will silently pass through the no-churn diff and update-loop cleanly (no crash), but any future refactor that assumes `lookup` and `fields` are fully disjoint sets will double-write or double-diff that field without an obvious symptom.

**Why it happens:**
The existing contract was correct for the current 1-field-changed-at-a-time reality; range/TBD adds enough new fields that it's easy to lose track of which ones moved from "always in `fields`" to "sometimes part of `lookup`" as the natural key is redesigned.

**How to avoid:**
Re-derive the full field list for `import_campaign_csv`'s call site explicitly against the new schema before writing the implementation task — don't just add new keys to the existing `fields` dict literal without re-checking against the (also-changing) `lookup` dict.

**Warning signs:**
A field appearing in both the `lookup` dict and the `fields` dict passed to the same `insert_or_create_campaign_run` call.

**Phase to address:**
Range-first `CampaignRun` scheduling implementation phase — cover with a test asserting `set(lookup) & set(fields) == set()` (or equivalent) for the actual call site.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|-----------------|
| Leaving `ut_start` nullable with the *same* `UniqueConstraint` unchanged, deferring the partial-index fix | Migration ships fast, existing tests still pass | Silent duplicate TBD rows under concurrent import/approval (Pitfall 1) — a real, if rare, data-correctness bug that looks identical to the WR-05 gap already fixed once | Never — this is the exact class of bug the milestone context flags as needing spike resolution before implementation |
| Auto-select top fuzzy match above a threshold instead of always requiring a human click | Fewer clicks for staff on the "obvious typo" case | Wrong-site false positives are silent and look like correct data everywhere downstream (Pitfall 4) | Never for this feature's UX — the milestone spec itself requires the dropdown-and-confirm pattern |
| Widening only `Observatory.obscode`'s model field, not `CreateObservatoryForm`'s independent `max_length=3` | Migration is a one-line model change | Feature looks "done" (migration ran, tests pass) but the human create-path still rejects long codes (Pitfall 3) | Acceptable only as an interim step *if* explicitly tracked as a follow-up task in the same phase, never left implicit |
| Reusing CR-02's second-offset disambiguation trick for TBD rows instead of designing a proper TBD-row key | Looks like a natural extension of existing, tested code | Breaks the moment `ut_start` is a real NULL rather than a "wrong but present" timestamp (Pitfall 2) | Never — the TBD case needs its own explicit disambiguator |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `CampaignRun.Meta.constraints` / migration `0003_...` | Changing `ut_start` to `null=True` without touching the `UniqueConstraint` at all, assuming it "just still works" | Replace with a `condition=Q(ut_start__isnull=False)` partial constraint for scheduled rows plus an explicit, separately-keyed dedup mechanism for TBD rows (works on both SQLite and Postgres) |
| `Observatory.obscode` widening | Bumping `models.CharField(max_length=4)` and stopping there | Also audit `solsys_code_observatory/forms.py:CreateObservatoryForm` (hardcoded `max_length=3, min_length=3`, independently declared, not derived from the model) and any other hardcoded length assumption before calling the widening done |
| MPC Obscodes API (`MPCObscodeFetcher`) | Sending every widened/long/spacecraft-style code through tier 2 same as before | Short-circuit tier 2 for codes that structurally can't be real MPC codes (contain `@`, exceed the traditional 3-4 char convention) — avoids a guaranteed-miss network round-trip per row |
| `resolve_site()`'s `create_placeholder` keyword | Adding a new call site (the disambiguation UI's "explicitly create new") that defaults `create_placeholder=True` without thinking it through, silently reintroducing auto-fabrication for a UI path that should always be an explicit, human-initiated create | Keep `create_placeholder=False` as the default assumption for any public/UI-triggered call; only the vetted CSV-import call site should pass `True` |
| Fuzzy-match candidate source (`name`/`short_name`/`old_names`) | Scoring `old_names` as a single opaque blob | Split `old_names` on its separator before scoring each alias individually against the typed text |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Fuzzy-matching against the full `Observatory` table per approval-queue row render | Approval queue page slows down as more `Observatory` rows accumulate (one fuzzy-score pass per candidate per row) | Compute fuzzy candidates on-demand (e.g. htmx-triggered per row) rather than eagerly for every pending row on page load; cap the `Observatory` queryset scanned (e.g. exclude clearly-irrelevant satellite/ground mismatches when the ground/space distinction is already known) | Noticeable once the approval queue has more than a handful of pending rows and `Observatory` grows past a few hundred entries |
| Tier-2 MPC API call attempted for every spacecraft-style code | Every space-mission `CampaignRun` import/approval pays a network round-trip that can never succeed | Short-circuit before tier 2 for codes that don't look like real MPC codes (see Integration Gotchas) | Any batch import containing multiple space-mission rows, especially under WR-01's existing timeout handling (10s each) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting a fuzzy-match "create new Observatory" free-text path with the same laxity as the already-vetted CSV import | A public submitter (unauthenticated `site_raw` on the submission form) could spam-create many bogus `Observatory` rows via a naive "always allow create" fallback | Gate the actual `Observatory.objects.create(...)` action behind `StaffRequiredMixin` (already used elsewhere in `campaign_views.py`) — public submitters can suggest a site name, only staff resolve/create through the disambiguation UI |
| Logging fuzzy-match scoring input/output verbatim including submitter free text | Low risk here (no PII in a site name normally) but `site_raw` is unvalidated 255-char free text from an anonymous form — could contain anything a submitter typed | Keep logging consistent with the project's existing PII-safe logging conventions (per Phase 14's WR-hardening) — don't assume "it's just a site field" means it's always safe to log raw |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|--------------|-------------------|
| Dropdown shows fuzzy candidates with no indication of *why* each matched (score/matched-field) | Staff can't judge confidence, defaults to picking the first item without real evaluation — functionally the same risk as auto-select | Show which field matched (name/short_name/old_names) and a rough confidence indicator per candidate |
| No way to say "none of these, and I don't want to create a new Observatory right now" | Staff feel forced into a wrong choice or an unwanted create, or leave the row in whatever state it was in without an explicit "still needs review" outcome | Preserve an explicit "leave unresolved" action distinct from both "pick a candidate" and "create new" — matches `resolve_site()`'s existing `(None, True)` contract |
| Silently reusing `site_needs_review` semantics for "has fuzzy suggestions" vs. "no automatic resolution possible at all" | Staff can't tell a row with good fuzzy candidates apart from one with none, from the flag alone | Surface fuzzy-candidate count/quality in the approval-queue UI, not just the boolean flag |

## "Looks Done But Isn't" Checklist

- [ ] **Widened `Observatory.obscode`:** Often missing the `CreateObservatoryForm` create-path update — verify a staff user can actually create an `Observatory` with an 8-character spacecraft code through the UI, not just via `Observatory.objects.create(...)` in a test/shell.
- [ ] **Range/TBD `CampaignRun` migration:** Often missing a test with two distinct TBD rows sharing the same campaign+telescope — verify no silent merge, not just that a single TBD row round-trips correctly.
- [ ] **Fuzzy-match disambiguation UI:** Often missing an explicit "never auto-select" UAT check — verify by feeding a genuinely ambiguous free-text site name (e.g. one that scores moderately against two different real sites) and confirming a human decision is still required, not just that exact typos resolve correctly.
- [ ] **Space-mission asset distinction (`observations_type`):** Often missing the check that a resolved space-mission `Observatory` (created via the widened path) actually gets `observations_type=SATELLITE_OBSTYPE` set — a spacecraft `Observatory` created through the generic disambiguation "create new" flow could default to `OPTICAL_OBSTYPE` (the model's default of `0`) unless the UI explicitly sets it, silently breaking the ground/space distinction downstream in coverage-gap analysis.
- [ ] **Coverage-gap asset-awareness:** Often missing a check that ephemeris-based observability computation (`telescope_runs.sun_event()`, which needs a real `Observatory` location) isn't accidentally invoked for a space-mission run at all — verify the ground/space branch happens before any ephemeris call, not after one silently succeeds/fails.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|-----------------|------------------|
| Duplicate TBD `CampaignRun` rows created past the missing constraint | MEDIUM | Add the missing `condition=`-based constraint retroactively; write a one-off management command or shell script to find and merge/delete duplicate TBD rows by the intended (new) natural key before applying the migration, since the migration itself will fail to apply a unique index over already-duplicate data |
| Fuzzy-match auto-selected a wrong site on already-approved runs | MEDIUM-HIGH | Because a wrong-but-real site looks correct, this requires an audit query (e.g. runs whose resolved site's coordinates are implausible for the stated telescope/instrument name) rather than an obvious error signal; re-flag affected rows `site_needs_review=True` and re-run through the (now-fixed) human-confirm UI |
| Widened `obscode` migration applied but `CreateObservatoryForm` left stale | LOW | Follow-up form fix is additive and low-risk — same pattern as the `260705-l1v` quick task; no data correction needed, just ship the form fix |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|-----------------|
| NULL `ut_start` defeats natural-key `UniqueConstraint` (Pitfall 1) | Investigation spike (schema decision) + range-scheduling implementation phase | Test: two TBD rows for the same campaign+telescope via concurrent/re-run import both fail to merge into one row and don't silently duplicate under the DB constraint |
| CR-02 offset hack breaks for real NULLs (Pitfall 2) | CSV-import range/TBD parsing implementation phase | Test: two distinct unparseable/TBD rows in one CSV import produce two distinct `CampaignRun` rows |
| Obscode widening incomplete (form/API/tier-2 short-circuit) (Pitfall 3) | Investigation spike (must explicitly scope the create-form path) + implementation phase | Manual/UAT: create an `Observatory` for an 8-character spacecraft code through the actual UI path used by the disambiguation feature, not just the ORM |
| Fuzzy-match auto-selects wrong site (Pitfall 4) | Site-disambiguation UI implementation phase | UAT: an ambiguous (not exact-typo) free-text site name always requires an explicit human click before `site` is set; `site_needs_review` never silently flips to `False` without a staff action |
| `lookup`/`fields` disjointness broken by new window fields (Pitfall 5) | Range-scheduling implementation phase | Test asserting no field name appears in both the natural-key lookup and the update-fields dict at the actual call site |

## Sources

- Direct repository inspection (highest-confidence source for this codebase-integration question): `solsys_code/models.py` (`CampaignRun`, `Observatory.SATELLITE_OBSTYPE`), `solsys_code/campaign_utils.py` (`resolve_site`, `parse_obs_window`, `_MAX_OBSCODE_LEN`), `solsys_code/migrations/0003_campaignrun_natural_key_unique_constraint.py`, `solsys_code/solsys_code_observatory/models.py` and `forms.py` (`CreateObservatoryForm`), `solsys_code/campaign_forms.py`, `solsys_code/management/commands/import_campaign_csv.py`, `.planning/PROJECT.md` (Key Decisions log for D-04/D-05/D-08/D-09/WR-01..08/CR-01/CR-02/`260705-l1v`).
- Empirical verification: in-memory `sqlite3` reproduction confirming a 3-column `UNIQUE` constraint accepts two rows that are identical except both have `NULL` in the third column (matches this project's dev DB engine).
- [PostgreSQL unique constraint null: Allowing only one Null | EDB](https://www.enterprisedb.com/postgres-tutorials/postgresql-unique-constraint-null-allowing-only-one-null)
- [PostgreSQL unique constraint, but NULL conflicts with everything — Cybertec](https://www.cybertec-postgresql.com/en/unique-constraint-null-conflicts-with-everything/)
- [Why Django unique_together Fails with Nullable ForeignKey (and How to Fix It)](https://www.codestudy.net/blog/django-unique-together-with-nullable-foreignkey/)
- [Django ticket #9781 — Admin refuses multiple NULL values for fields marked unique](https://code.djangoproject.com/ticket/9781)

---
*Pitfalls research for: FOMO v2.1 — CampaignRun window/TBD scheduling, Observatory.obscode widening, fuzzy site disambiguation*
*Researched: 2026-07-05*
