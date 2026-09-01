# Phase 27: The Canonical Run Record - Research

**Researched:** 2026-07-29
**Domain:** Django/TOM-Toolkit schema evolution (model rename + new fields + two new link
models), admin surfaces, and a single upstream-template override. No new libraries.
**Confidence:** HIGH (every claim below is either a direct code read, a `grep`/symbol
inspection, or an executed proof against a disposable copy of the real dev DB — this phase
has almost no external-library surface to research, so there is very little `[ASSUMED]`
material)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase boundary:** Phase 27 makes `CampaignRun` canonical **in the schema**: it records
how it was created (`source`), says why it has no site when it has none
(`telescope_class`), owns the calendar events that show it (a `run` link on the renamed
companion record), and owns the observation records that realise it (a new link model) —
with every existing row and all six rename integration points surviving the change.
CANON-01 through CANON-05.

**In scope:** the model changes, their migrations and backfills, the admin surfaces that
make the new links visible and editable, one calendar-template change so an event links
back to its run, and a data-repair task for site-less runs discovered to be stale rather
than unresolvable.

**Out of scope:** the reconciler (Phase 29), the attribution queue and its confidence
scoring (Phase 28), rewiring the four ingest adapters to create `CampaignRun`s (v2.3 /
ADAPT-01..03), and the adopt-vs-gap-fill write strategy (deliberately deferred by Phase 26
to Phase 29).

- **D-01: Confirmed rows only.** A `CampaignRunObservation`-style link row exists only once
  a staff member confirms it. Phase 28 computes candidates on the fly; nothing is written
  until confirmation.
- **D-02: One run per observation record, expressed so it can be broadened cheaply.**
  `ForeignKey(ObservationRecord)` + a **named `UniqueConstraint`**, not `OneToOneField`.
- **D-03: The link records who and when.** `confirmed_by` (FK to `User`,
  `on_delete=SET_NULL`) and `confirmed_at` (timestamp). No boolean.
- **D-04: Deleting a run deletes the observation-link row** (`CASCADE` on the run FK); the
  `ObservationRecord` itself is untouched. Deliberately *not* `SET_NULL` (contrast with D-05).
- **D-05: `CalendarEventMeta.run` stays a bare FK, `SET_NULL` on delete.** No
  `confirmed_by`/`confirmed_at` on the event side — exactly as Phase 26 locked it. Audit
  asymmetry vs. the observation link is accepted deliberately.
- **D-06: Editable admin inlines now, a real staff page in Phase 28.** A `CalendarEventMeta`
  inline and a `CampaignRunObservation` inline on the existing `CampaignRunAdmin`. No
  run-detail view exists today.
- **D-07: The inlines are editable, and that obliges `save_formset` wiring.**
  `CampaignRunAdmin.save_formset()` must populate `confirmed_by=request.user`/
  `confirmed_at=now` on admin-created rows. **Required task, not a discovery item.**
- **D-08: An event links back to its run from the calendar event modal**, via a
  **template override alone** on `tom_calendar/partials/event_form.html` — `event` is
  already in `update_event`'s render context. Cost: FOMO now owns a second upstream
  `tom_calendar` template.
- **D-09: The modal link is shown to everyone, but hidden for `pending_review` runs** —
  mirrors `CampaignRunTableView.get_queryset()`'s existing `exclude()`. This *adds*
  non-staff-visible surface; it does not change the run table's existing gating.
- **D-10: The visibility rule gets one definition.** Add `CampaignRun.is_publicly_visible`
  (`approval_status != PENDING_REVIEW`); the modal template reads it. The existing
  queryset-level `exclude()` stays (a Python property can't be used in a `.filter()`).
- **D-11: The spike's "space missions are permanently site-less" premise is false.** Space
  observatories resolve to a real `Observatory` via MPC obscode or the Horizons alias table
  (`campaign_utils.py:42-47`) just like ground sites. The genuine exception is a space
  observatory with a Horizons code but **no MPC code assigned at all** — JUICE (`500@-28`).
  **Final vocabulary:** `2M0`/`1M0`/`0M4` (class allocation) + `SPACE` (no-MPC-code space
  observatory specifically). Blank otherwise. "Unresolved" is deliberately **not** a
  `telescope_class` value — `site_needs_review` already carries that meaning.
- **D-12: Three classes, not four — cross-check is a subset assertion.** `telescope_class`
  is normal Django `TextChoices`, limited to `2M0`/`1M0`/`0M4`. `calendar_utils`'s existing
  vocabulary has **four** (includes `4m0`, for SOAR). The cross-check test asserts a
  **subset**, explicitly naming `4m0` as the known-excluded value.
- **D-13: Backfill by derived rule, in a data migration**, not a hand-enumerated pk list —
  must generalise beyond the dev DB. Leaves `telescope_class` blank when a site-less row
  shows neither a class nor a no-MPC-code space signal.
- **D-14: Success criterion 2 vs. the Phase 26 lock — measure before choosing.** *(This
  research's Priority 1 — see Architecture Patterns, Pattern 1, below for the full
  executed proof.)*
- **D-15: pk=31 is not an anomaly.** `site_raw='X05'` is a valid `Observatory` (Rubin/X05);
  the row is `rejected` so resolution was never attempted. Kept as-is.
- **D-16: Stale site-less rows are repaired in Phase 27, as their own separately-committed
  task**, calling full `resolve_site()` including tier 2 (live MPC API):

  | Rows | `site_raw` | What happens |
  |---|---|---|
  | pk 21, 27, 28 (JWST) | `500@-170` | Alias -> `274`; tier 1 hit. Offline. |
  | pk 8, 12 (HST) | `250` | Tier 1 miss -> **tier 2 MPC API** -> creates HST Observatory. |
  | pk 13 (Swift) | *empty* | Returns `(None, True)` immediately — needs a code first. |
  | pk 26 (JUICE) | *empty* | Same; no MPC code exists -> `telescope_class=SPACE`. |
  | pk 29, 30 | *empty* | Class-wide; get `1M0`/`2M0`. |

  D-16a: tier-2-dependent, not reproducible offline/CI — plan tests accordingly (see
  Priority 3 findings below). D-16b: pk=13 gets `site_raw='C52'`, supplied by the project
  owner as domain authority, not inferred.
- **D-17: `site_needs_review` is not touched by the backfill** — D-16 clears it through
  the normal resolution path.
- **D-18: `telescope_class` is visible to non-staff; `source` is not.** `telescope_class`
  joins `ALLOWED_FIELDS_FOR_NON_STAFF`; `source` stays staff-only, omission commented.
- **D-19: Both fields get `list_display`/`list_filter` in the admin.** `source` is **not**
  added to `readonly_fields`.
- **D-20: One shared derivation helper, taking primitives** (`site_raw`,
  `telescope_instrument`), called by **both** the data migration and `import_campaign_csv`.
  **Home: `calendar_utils.py`**, next to `_aperture_class_from_telescope_code` and
  `SITE_TELESCOPE_MAP`.

### Claude's Discretion

- The exact name of the observation-link model (`CampaignRunObservation` used as
  placeholder; 26-DECISION.md rejected `CalendarEventRunLink` for being too link-specific
  for the *event* side — the same generality argument applies here).
- Whether `is_publicly_visible` is also used to simplify any existing call site beyond the
  modal template.
- Test organisation, beyond D-12's specific subset assertion.

### Folded Todos (all four in scope)

1. Drop the leading underscore on five cross-module-consumed `calendar_utils.py` helpers
   (`_aperture_class_from_telescope_code`, `_derive_telescope`, `_resolve_placement_block`,
   `_extract_instrument`, `_coarse_telescope_label`), **and** move `calendar_utils.py`-owned
   tests out of `test_sync_lco_observation_calendar.py` into their own module — scope
   explicitly (see Priority 5 findings below for exact consumer/test counts).
2. Correct the "queue windows are sets of owned nights" framing in upstream planning docs
   (docs-only).
3. Correct PROJECT.md's stale Phase 25 claim, **and** its "five space-mission rows are
   permanently site-less" claim, now false per D-11 (docs-only).
4. Size the `SITE_TELESCOPE_MAP`-extraction todo before it lands in the plan — **no CANON
   requirement behind it**; D-20 removes any dependency on it (see Priority 5 findings —
   recommendation: drop it).

### Deferred Ideas (OUT OF SCOPE)

- A real staff run-detail page (Phase 28).
- Where dismissed attribution candidates persist (Phase 28's problem).
- Audit fields on `CalendarEventMeta.run` (declined by D-05).
- Adding `4M0` to `telescope_class` (declined by D-12).
- Adding `telescope_class` to a unique constraint (only if D-14's measurement showed a real
  colliding pair — it does not; see Pattern 1).
- Extracting `SITE_TELESCOPE_MAP` into its own module (foldable back out if it inflates the
  phase — recommendation below is to drop it).
- v2.3 items untouched here: ADAPT-01..03, GAPB-01, STATUS-01/02, UNUSED-01.
- The adopt-vs-gap-fill write strategy (Phase 29, per explicit human decision).

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CANON-01 | `CampaignRun` records ingest path (`source`); approval required only for web submissions | Pattern 2 (source/`ApprovalStatus` derivation, already correct in `import_campaign_csv.py:194`); Priority 4 findings on `ALLOWED_FIELDS_FOR_NON_STAFF`/admin |
| CANON-02 | `telescope_class` distinguishes class-wide allocation from unresolved site | Pattern 1 (D-14 executable proof); Pattern 3 (D-20 helper design, naming precedent found in `campaign_utils.py`'s own docstring) |
| CANON-03 | Generalised companion record (rename) carries the event->run link; all 4(6) integration points keep working | Pattern 4 (rename checklist re-verified against live code, all 6 points confirmed unchanged since Phase 26) |
| CANON-04 | `ObservationRecord` link records human confirmation; deletes never cascade destructively | Priority 4 findings (`ObservationRecord` import path/pk type); D-02..D-05 already locked |
| CANON-05 | Staff see a run's linked events/records; can reach the run from an event | Pattern 5 (admin inlines + `save_formset`); Pattern 6 (template-override mechanics, verified against installed `tom_calendar`) |

</phase_requirements>

## Summary

Phase 27 is almost entirely a "read the code correctly and don't re-derive settled
decisions" phase — CONTEXT.md's D-01..D-20 already lock nearly every design choice, and this
research's job was (1) resolve the one named open question (D-14) with an executable proof,
(2) re-verify Phase 26's rename checklist and migration-shape claims against the *current*
live code (nothing has drifted — every citation still matches), and (3) surface the concrete
integration-point facts (`admin.py`'s exact current contents, `ALLOWED_FIELDS_FOR_NON_STAFF`'s
exact current list, the exact template context `tom_calendar.views.update_event` provides)
so the planner can write diff-level tasks rather than re-discovering them during execution.

**D-14, resolved:** constructing the actual colliding pair and testing it against a
disposable copy of the real dev DB (`CampaignRun` pk=29, the real `LCO 1m` class-wide row)
proves that two rows sharing `(campaign, telescope_instrument, window_start, window_end)`
**do** collide with an `IntegrityError` — but that this can never happen from any ingest path
that exists today, or that Phase 27 adds. The web-submission path already treats an
identical-natural-key `.create()` as an existing-duplicate error (friendly form message);
the CSV-import path's `insert_or_create_campaign_run()` treats an identical-key row as
**the same run** and silently merges fields onto it via `get_or_create()`. Both behaviours
are by design (`WR-05`'s constraint docstring: idempotent re-import support), not bugs.
**No new constraint change is needed; ROADMAP criterion 2 is satisfiable exactly as
CONTEXT.md designed it**, because a class-wide row's generic `telescope_instrument` text
(`'LCO 1m'`) and a genuinely-unresolved row's specific attempted-instrument text will
essentially never be byte-identical in real data — and on the one occasion they already are
byte-identical in the live DB (the TBD-branch pk=27/pk=28 pair, both `JWST`, both site-less),
the system already coexists them correctly, discriminated by `contact_person`, with zero
`telescope_class` involvement.

**Rename checklist (CANON-03), re-verified 2026-07-29:** all six integration points named
in `26-DECISION.md` Criterion 4 still match the live code exactly, line-for-line where cited
(`admin.py:4,28,41`; `sync_lco_observation_calendar.py:18,369`; `views.py:114`
`.prefetch_related('telescope_label_meta')`; `calendar.html:228,244`;
`test_admin.py`'s `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')`;
class-name references in three test modules). Nothing has drifted since the spike.

**Primary recommendation:** implement in this order — (1) rename migration (hand-authored,
no autodetection) + fix the six integration points; (2) `AddField` for `source`/
`telescope_class` (both nullable/defaulted, no dependency on the rename); (3) new
`CalendarEventMeta.run` and the new `CampaignRunObservation` model, both nullable/optional
so they never block on existing data; (4) the D-13/D-16 data-migration backfill, which
depends on (2) existing and D-20's helper existing in `calendar_utils.py`; (5) the D-16
site-repair task as its own separately-committed step (per CONTEXT.md, it is explicitly
**not** a schema change and has no CANON requirement — it just makes (4)'s backfill act on
current data); (6) admin/template/view surface work, which depends on (2)+(3) existing.

## Architectural Responsibility Map

FOMO is a monolithic Django MVT app with HTMX partial-swap (no separate frontend/backend
split, no SPA) — the tiers below map onto Django's own layers rather than a
browser/SSR/API/CDN split, per CLAUDE.md's documented architecture.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `source`/`telescope_class` fields + constraints | Database/Storage | API/Backend (Django ORM model definition) | Schema-level; no view/template touches these beyond display |
| Rename migration + `source`/`telescope_class` backfill | Database/Storage | API/Backend (data migration is Python code, runs at the DB layer) | `RenameModel`/`AddField`/`RunPython` are pure schema+data operations |
| `CalendarEventMeta.run`, `CampaignRunObservation` models | Database/Storage | API/Backend | New FK/link tables; ownership semantics enforced at ORM/constraint level |
| Admin inlines + `save_formset` attribution stamping | API/Backend (Django admin) | Browser/Client (renders the inline HTML forms) | `ModelAdmin`/`InlineModelAdmin` are server-side Python; `request.user` is only available server-side |
| Calendar event modal run-link (D-08) | Frontend Server (SSR — Django template rendering inside `tom_calendar.views.update_event`) | Browser/Client (HTMX swaps the modal HTML) | Template-override-only; no new view, no new URL |
| `is_publicly_visible` gating (D-09/D-10) | API/Backend (model property) | Frontend Server (template reads the property) | One Python definition, read from both the queryset-exclude call site and the new template |
| `telescope_class` derivation helper (D-20) | API/Backend | Database/Storage (also called from a migration) | Pure function of primitives; imported by both a view-adjacent module and a migration file |

## Project Constraints (from CLAUDE.md)

- Two independent test suites: `python -m pytest` (collects `tests/`, `src/`, `docs/` only —
  **never** `solsys_code/`) and `./manage.py test solsys_code` (Django DB tests). All new
  tests for this phase are Django DB tests -> `solsys_code/tests/`.
- **Heavy import side effect:** `solsys_code.views`/`solsys_code.ephem_utils` download ~1.6GB
  of SPICE kernels at import time. Verified this phase: `campaign_views.py`, `admin.py`,
  `calendar_utils.py`, and `campaign_utils.py` **do not** import either module — every test
  module this phase touches (`test_admin.py`, `test_campaign_models.py`,
  `test_campaign_approval.py`, `test_calendar_template.py`, `test_calendar_utils.py`,
  `test_sync_lco_observation_calendar.py`, `test_load_telescope_runs.py`,
  `test_import_campaign_csv.py`) is in the **fast** category. Only `test_ephem_utils.py` and
  `test_views.py` pay the SPICE cost — exclude them from the phase's routine/fast test runs
  (matches Phase 26's own 7-module selection precedent).
- **Paired docs are part of the deliverable:** `docs/notebooks/pre_executed/
  import_campaign_csv_demo.ipynb` and `docs/runbooks/telescope_runs_calendar.rst` must be in
  `files_modified` from the start (CANON-01's `source` write + D-20's `telescope_class`
  derivation genuinely changes `import_campaign_csv`'s behaviour and output).
- Planning-doc terminology: "create or update" / "find-or-create", never "upsert" —
  `insert_or_create_campaign_run()`'s own docstring already uses this phrasing; match it.
- Target factories: always `NonSiderealTargetFactory` (already the convention in
  `test_campaign_models.py`).
- ruff: single quotes, 120-col, Rubin DM naming exceptions already configured.
- No new dependency — REQUIREMENTS.md's Out of Scope table and CONTEXT.md both forbid one;
  this phase needs none (rename/`AddField`/`RunPython`/`ModelAdmin` inlines are all
  stdlib-Django).

## Standard Stack

**N/A — no new dependency for this phase.** REQUIREMENTS.md's Out of Scope table explicitly
rejects `GenericForeignKey`, a new dependency for reconciliation/field-diffing, renaming
`related_name`, and making the `run` link required. Every mechanism this phase needs
(`RenameModel`, `AddField`, `RunPython`, named `UniqueConstraint`, `ForeignKey(on_delete=...)`,
`admin.TabularInline`, `ModelAdmin.save_formset()`, a template override) is already-installed
Django 5.2.13 `[VERIFIED: installed package, python -c "import django; print(django.VERSION)"
-> (5, 2, 13, 'final', 0)]`.

## Package Legitimacy Audit

**N/A — no packages installed by this phase.**

## Architecture Patterns

### Pattern 1 — D-14: the constructed colliding pair, executed and proven

**Setup verified against the real dev DB (read-only probe, zero writes; fingerprint
`fdf94f7924f52424b8e0b14a4e26d8c1` unchanged before/after)
`[VERIFIED: solsys_code/models.py, live query against src/fomo_db.sqlite3]`:**

`CampaignRun.Meta.constraints`, read directly from the live model (not re-typed from the
decision doc):

```
UniqueConstraint name='unique_campaign_run_resolved_window'
  fields=('campaign', 'telescope_instrument', 'window_start', 'window_end')
  condition=Q(window_start__isnull=False)
UniqueConstraint name='unique_campaign_run_tbd_natural_key'
  fields=('campaign', 'telescope_instrument', 'contact_person')
  condition=Q(window_start__isnull=True)
CheckConstraint name='campaign_run_window_start_end_null_together'
```

Neither `site` nor `telescope_class` (once added) is in either field tuple — confirmed by
direct inspection, matching 26-DECISION.md exactly, unchanged since the spike.

Of the 31 real rows, exactly **one** existing 4-tuple duplicate exists today: **pk=27/pk=28**
(`campaign=3`, `telescope_instrument='JWST'`, `window_start=None`, `window_end=None`) — a
real, live instance of two site-less rows sharing a natural key, discriminated only by the
TBD branch's `contact_person` field (confirmed unequal between the two rows without printing
either value, per this project's PII discipline). **This is the direct, already-existing
precedent for "two site-less rows can coexist safely" — and `telescope_class` plays no part
in it.**

**Executed proof, against a disposable scratch copy of the real dev DB (matching Phase 26's
evidence posture — `local_settings.py` DB-path override, guard-asserted before every write,
both files deleted at the end, real DB md5sum `fdf94f7924f52424b8e0b14a4e26d8c1` unchanged
throughout) `[VERIFIED: executed against a scratch copy of src/fomo_db.sqlite3, 2026-07-29]`:**

- **Test A** — literal duplicate of pk=29 (`'LCO 1m'`, the real class-wide row: `campaign=3`,
  window `2025-07-05`..`2025-09-22`, `site=None`), representing a hypothetical "unresolved
  sibling" row (`site=None, site_needs_review=True`):
  `CampaignRun.objects.create(campaign_id=3, telescope_instrument='LCO 1m', window_start=...,
  window_end=..., site=None, site_needs_review=True)` **raises `IntegrityError`** —
  `UNIQUE constraint failed: ...campaign_id, ...telescope_instrument, ...window_start,
  ...window_end`. This is the literal collision D-14 asked to be constructed: two rows with
  the same 4-tuple **do** collide, regardless of any other field difference (`site`,
  `site_needs_review`, and by direct extension `telescope_class` once it exists), exactly
  the same pattern SPIKE-01 already proved for `source`.
- **Test B** — same scenario, `telescope_instrument` text differs
  (`'1m0 (site TBD)'` instead of `'LCO 1m'`): **coexists with no `IntegrityError`** (pk=32
  created, then deleted for cleanup). Distinct free text is sufficient to avoid the
  constraint entirely — this is the realistic case, since a class-wide row's generic label
  and a specific submitter's own free-text instrument description are not usually
  byte-identical.
- **Test D** — `insert_or_create_campaign_run()` (the CSV-import write path) called with
  pk=29's exact lookup key and a differing field (`filters_bandpass` set to a probe value):
  returns `action='updated'`, **silently overwrites pk=29's real field value** (confirmed via
  a second run showing the change took effect in-transaction, then rolled back — pk=29's
  `filters_bandpass` unchanged after rollback). **No `IntegrityError` at all** — `get_or_create()`
  finds the existing row and treats the "would-be-unresolved" row as an update to the SAME
  run, not a second row.

**Conclusion (answers D-14 directly, per its own required standard):** **(a) no realistic
pair collides.** The DB-level collision is real and reproducible (Test A), but no ingest
path that exists today, or that Phase 27 adds, can produce it as two competing rows:

1. `CampaignRunSubmissionView.form_valid()` wraps `CampaignRun.objects.create()` in
   `transaction.atomic()` and catches `IntegrityError` with a friendly "already exists" form
   error (`campaign_views.py:251-265`) — an identical-natural-key second submission is
   **rejected before it ever becomes two rows**.
2. `import_campaign_csv.py` (via `insert_or_create_campaign_run()`) uses `get_or_create()` on
   the identical lookup key — an identical-natural-key CSV row is **merged into the existing
   row**, not inserted as a genuine second row (Test D, above).
3. The one real live analogue (pk=27/pk=28, TBD branch) already coexists safely, using the
   TBD constraint's own discriminator (`contact_person`), with zero involvement from
   `telescope_class`.

The constraint's design intent (`WR-05`'s own comment: backs `insert_or_create_campaign_run`'s
idempotent-re-import contract) is that an identical `(campaign, telescope_instrument, window)`
triple **is** the same physical run, by definition — so the roadmap's "coexist without
colliding" requirement is satisfied by construction whenever the two rows' `telescope_instrument`
text differs even slightly (the overwhelming real-world case), and is *correctly* treated as
one run (needing its `telescope_class`/site status resolved, not two rows) on the rare
occasion the text is identical. **No constraint change is needed. D-14's lock is not
reopened; the Phase 26 lock stands, with executed evidence.** The one residual, pre-existing
(not new) risk this proof surfaces: `insert_or_create_campaign_run()`'s silent-merge behaviour
(Test D) already exists for every field, not introduced by `telescope_class` — worth a
one-line planner note but not a design change.

### Pattern 2 — CANON-01: `source` and the approval derivation, current code state

`import_campaign_csv.py:194` (verified, still the exact line number CONTEXT.md cites)
`[VERIFIED: solsys_code/management/commands/import_campaign_csv.py]`:

```python
'approval_status': CampaignRun.ApprovalStatus.APPROVED,  # D-03: bootstrap rows are vetted backfill
```

The importer already writes `APPROVED` unconditionally — CANON-01's real change here is
adding `'source': CampaignRun.Source.CSV_IMPORT` (or whichever enum member name the planner
picks) to this same `fields` dict, **not** touching `approval_status` at all. The
`CampaignRunSubmissionView.form_valid()` path (`campaign_views.py:229-250`) never sets
`approval_status` either (model default `PENDING_REVIEW` applies) — it should set
`source=WEB` explicitly, since that is the one value where approval genuinely IS required.
No other existing write path creates a `CampaignRun` today (`ADAPT-01..03` — the
classical/LCO/Gemini adapters — are v2.3, out of scope).

### Pattern 3 — CANON-02: the `telescope_class` derivation helper (D-20)

**A naming precedent already exists in the codebase, undiscovered until this research:**
`campaign_utils.py`'s own module docstring (line 8) already references
`` `_derive_telescope_class` `` as a design precedent — `"per the `_derive_telescope_class`
precedent in `calendar_utils.py`"` — **but no function by that name exists yet**
`[VERIFIED: grep -rn "_derive_telescope_class" solsys_code/ -> only the docstring reference,
zero definitions]`. This is either forward-looking language written in anticipation of
exactly this phase's D-20 helper, or a stale reference to `_derive_telescope`/
`_aperture_class_from_telescope_code`'s "never raise, return value+flag" pattern — either
way, `derive_telescope_class` (or `_derive_telescope_class`, matching the still-private
naming of its neighbours until the folded-todo rename lands) is a strong, already-anticipated
name for the planner to use for D-20's new function, rather than inventing a fresh one.

**Case-convention flag (see Assumptions Log A1):** CONTEXT.md's own prose (D-11/D-12) writes
the vocabulary uppercase (`2M0`/`1M0`/`0M4`/`SPACE`), matching `CampaignRun`'s existing
`TextChoices` convention of readable value strings. But `calendar_utils.py`'s existing
aperture-class vocabulary (`_aperture_class_from_telescope_code`, `SITE_TELESCOPE_MAP`,
`_coarse_telescope_label`) is **lowercase** (`'0m4'`, `'1m0'`, `'2m0'`, `'4m0'`) throughout —
confirmed by direct read of `calendar_utils.py:37-52,101,291`. D-12's subset-assertion test
("every `telescope_class` value appears in `calendar_utils`' aperture-class set") will only
pass without an explicit case-fold if `CampaignRun.telescope_class`'s stored values are
lowercase (`'2m0'`/`'1m0'`/`'0m4'`), matching `calendar_utils`'s existing convention, *despite*
CONTEXT.md's prose using uppercase. Recommend: store lowercase to match `calendar_utils`'s
one true source of the vocabulary, treat CONTEXT.md's uppercase spelling as prose/display
convenience only. **Flagged as an assumption for planner/discuss confirmation** — this is a
concrete implementation choice CONTEXT.md's decisions don't pin down explicitly.

The helper's home (`calendar_utils.py`, per D-20) is import-safe from a migration: verified
`calendar_utils.py`'s own top-level imports (`tom_calendar.models.CalendarEvent`,
`tom_common.exceptions`, `tom_observations.facilities.lco/ocs`) contain **no** reference to
`solsys_code.views`/`solsys_code.ephem_utils` `[VERIFIED: grep across calendar_utils.py and
campaign_utils.py]` — importing it from a `RunPython` step therefore never triggers the SPICE
kernel download. The only caveat: `calendar_utils.py` does import the live `CalendarEvent`
model at module scope — harmless at migration-run time (Django's app registry is fully
populated before any migration executes), but it means the migration's `RunPython` function
should still fetch `CampaignRun` via `apps.get_model('solsys_code', 'CampaignRun')` (the
historical/frozen model, per this project's own existing precedent — see Pattern 4's
migration-precedent discussion) rather than importing `solsys_code.models.CampaignRun`
directly, even though the *helper function itself* is safe to import directly (it takes only
primitives, never a model instance — exactly D-20's stated rationale).

### Pattern 4 — CANON-03: the rename checklist, re-verified against live code (2026-07-29)

All six integration points from `26-DECISION.md` Criterion 4 were re-checked directly against
the current tree; **nothing has drifted** since the spike:

| # | Integration point | Verified location (2026-07-29) |
|---|---|---|
| 1 | `admin.py` import/class/register | `solsys_code/admin.py:4` (`from solsys_code.models import CalendarEventTelescopeLabel, CampaignRun`), `:28` (`class CalendarEventTelescopeLabelAdmin`), `:41` (`admin.site.register(CalendarEventTelescopeLabel, ...)`) |
| 2 | `sync_lco_observation_calendar.py` import/call | `:18` (import), `:369` (`CalendarEventTelescopeLabel.objects.update_or_create(...)`) |
| 3 | `views.py` prefetch (safe by construction) | `:114` `.prefetch_related('telescope_label_meta')` |
| 4 | `calendar.html` accessor (safe by construction) | `src/templates/tom_calendar/partials/calendar.html:228,244` `event.telescope_label_meta.is_verified` |
| 5 | `test_admin.py` reverse-URL name | Lines 49-55: `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` (2 call sites) |
| 6 | Class-name references in tests | `test_load_telescope_runs.py:16,289,291`; `test_sync_lco_observation_calendar.py:26,262,287,475,486,489`; `test_calendar_template.py:22,51,58,72,79,128` |

`CalendarEventTelescopeLabel.event` is a `OneToOneField(primary_key=True,
related_name='telescope_label_meta')` — confirmed directly in `models.py:16-22`. This
`related_name` is what the Requirements' Out-of-Scope table forbids renaming (it would
silently break points 3/4 with no static check); the class name is what's actually renamed,
and Django's admin autodiscovery/reverse-URL machinery is what makes points 1/2/5 fail
*loudly* (`ImportError`/`NoReverseMatch`) rather than silently.

**Migration-shape precedent, confirmed by direct read of `solsys_code/migrations/`**
`[VERIFIED: solsys_code/migrations/0004_campaignrun_window_schema.py,
0005_campaignrun_campaign_run_window_start_end_null_together.py]`: this project already has
**two** precedent data migrations using `migrations.RunPython(fn,
reverse_code=migrations.RunPython.noop)`, both fetching models via
`apps.get_model('solsys_code', 'CampaignRun')` (the historical/frozen model state), never
importing the live model class directly inside the `RunPython` function body. Phase 27's
rename + `AddField` + backfill migration should follow this exact convention: `RenameModel`
first, then `AddField` x3 (`CalendarEventMeta.run`, `CampaignRun.source`,
`CampaignRun.telescope_class`), then a `RunPython` backfill step calling
`calendar_utils.derive_telescope_class(site_raw=run.site_raw,
telescope_instrument=run.telescope_instrument)` per D-13/D-20 (importing the plain function
directly is safe, per Pattern 3 above — only the *model* needs `apps.get_model`).

No migration ordering hazard was found: the rename must precede any `AddField` referencing
the renamed model (obviously), and the backfill `RunPython` step must come after both the
`telescope_class` field exists (2) and the derivation helper exists in `calendar_utils.py`
(a code dependency, not a migration-graph dependency — the helper just needs to be committed
in the same changeset before the migration file is written).

### Pattern 5 — CANON-05: admin inlines and `save_formset`, current `CampaignRunAdmin` state

Full current contents of `CampaignRunAdmin`, verified by direct read
`[VERIFIED: solsys_code/admin.py]`:

```python
class CampaignRunAdmin(admin.ModelAdmin):
    list_display = ['pk', 'campaign', 'telescope_instrument', 'approval_status',
                     'run_status', 'site', 'window_start', 'window_end']
    list_filter = ['approval_status', 'run_status', 'campaign']
    search_fields = ['telescope_instrument', 'site_raw', 'contact_person']
    readonly_fields = ['approval_status']
```

**No `save_formset`/`save_model` override exists today** — confirmed, this is genuinely new
work per D-07, not a discovery item. Django's base `ModelAdmin.save_formset()`, confirmed by
direct inspection of the installed package
`[VERIFIED: python -c "import inspect, django.contrib.admin as admin;
print(inspect.getsource(admin.ModelAdmin.save_formset))"]`:

```python
def save_formset(self, request, form, formset, change):
    """Given an inline formset save it to the database."""
    formset.save()
```

The standard override shape for stamping `confirmed_by`/`confirmed_at` on newly-created
inline rows only (not touching pre-existing rows) is `[ASSUMED — standard, widely-documented
Django idiom, not fetched from docs.djangoproject.com this session]`:

```python
def save_formset(self, request, form, formset, change):
    instances = formset.save(commit=False)
    for instance in instances:
        if isinstance(instance, CampaignRunObservation) and instance.pk is None:
            instance.confirmed_by = request.user
            instance.confirmed_at = timezone.now()
        instance.save()
    formset.save_m2m()
```

D-19 adds `source`/`telescope_class` to `list_display`/`list_filter` on the same class —
`source` explicitly not in `readonly_fields` (only `approval_status` stays there, per its own
documented reason: its transition triggers calendar-projection side effects in
`CampaignRunDecisionView`, unrelated to `source`/`telescope_class`).

`ALLOWED_FIELDS_FOR_NON_STAFF`, full current contents, verified
`[VERIFIED: solsys_code/campaign_views.py:70-87]`:

```python
ALLOWED_FIELDS_FOR_NON_STAFF = [
    'pk', 'telescope_instrument', 'site__short_name', 'site_raw', 'site_needs_review',
    'window_start', 'window_end', 'filters_bandpass', 'run_status', 'approval_status',
    'open_to_collaboration', 'observation_details', 'weather', 'observation_outcome',
    'publication_plans', 'comments',
]
```

D-18 adds `telescope_class` to this list (hand-enumerated, deliberately not introspected);
`source` is deliberately **not** added, and the omission must be commented in the diff (per
D-18's own instruction), since this project's convention is that "a new field is invisible
to non-staff unless explicitly added" — the omission must read as a decision, not a miss.

No run-detail view exists — `campaign_urls.py` confirmed to have only `list` / `submit` /
`approval_queue` / `site_search` / `decide` / `gap_analysis` / `table` (by-campaign, not
by-run) `[VERIFIED: solsys_code/campaign_urls.py]` — matching D-06's claim exactly.

### Pattern 6 — CANON-05: the calendar-event-modal template override (D-08), verified end-to-end

`tom_calendar.views.update_event` (installed package, both `GET` and post-invalid-`POST`
branches) renders `tom_calendar/partials/event_form.html` with `{"form": form, "event":
event, "action": "update"}` `[VERIFIED: direct read of the installed tom_calendar package's
views.py]` — `event` is genuinely already in context, confirming D-08's claim without
needing a view override.

FOMO's `TEMPLATES['DIRS']` includes `os.path.join(BASE_DIR, 'templates')` **ahead of**
`APP_DIRS=True` `[VERIFIED: src/fomo/settings.py:93-107]` — so
`src/templates/tom_calendar/partials/event_form.html` (mirroring the existing
`src/templates/tom_calendar/partials/calendar.html` override precedent, confirmed to exist
at that exact path) will correctly shadow the upstream installed template. This is the
**second** upstream `tom_calendar` template FOMO takes ownership of — the cost D-08 already
names (drift risk on `tomtoolkit` upgrades) is real and confirmed, not hypothetical.

`ObservationRecord`, verified via direct model inspection
`[VERIFIED: python -c "from tom_observations.models import ObservationRecord;
print(ObservationRecord._meta.pk)"]`: import path `tom_observations.models.ObservationRecord`,
pk field `id` (`AutoField`, standard integer pk — no special-casing needed for D-02's FK +
named `UniqueConstraint`). No existing FK from any `solsys_code` model to `ObservationRecord`
today — this is the first, so there is no existing `related_name` convention to match beyond
`CampaignRun`'s own pattern (`related_name='campaign_runs'` on each of its own FKs).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Detecting whether a migration would collide with existing data | A hand-written pre-flight duplicate scan | The existing `RunPython` dedup precedent (`0004`/`0005`) — write the backfill assuming the constraint already prevents true duplicates, since D-14's proof shows no genuine duplicate-natural-key pair can exist for a resolved-window row today | Two precedent migrations already do exactly this; re-inventing a scanner duplicates `0004`'s `dedupe_*` logic for no benefit |
| Recording who approved/edited an inline admin row | A custom audit-log app or signal | `ModelAdmin.save_formset()` override (D-07), the stdlib Django mechanism this project already exercises for `CampaignRunDecisionView`'s own approval-audit pattern | No new dependency permitted (REQUIREMENTS.md Out of Scope); Django's own hook is sufficient and is the only mechanism that has `request.user` available |
| Testing a data migration's backfill logic | A one-off manual dry-run script against a DB copy (Phase 26's own spike style) | `django.db.migrations.executor.MigrationExecutor` + `TransactionTestCase`, exactly as `test_window_schema_migration.py` already does | This project has an existing, working precedent for exactly this need — reproduce its shape, don't reinvent |
| Verifying the MPC-API-dependent site-repair task (D-16a) | A live network call inside the automated test suite | `unittest.mock.patch('requests.get')` returning a `MagicMock` with `.ok`/`.json()`, exactly as `test_import_campaign_csv.py`'s `_MPC_OBS_DATA_E10` fixture already does | Existing, working convention in this codebase for the exact same API; the real live repair run is a separate, manual, one-time execution (see Validation Architecture) |

**Key insight:** almost nothing in this phase needs new machinery — this codebase already
has a working precedent for every mechanism CANON-01..05 needs (data migrations with
`RunPython`, migration-testing via `MigrationExecutor`, MPC-API mocking, named partial unique
constraints, admin registration). The research risk here was never "what library do we need"
— it was "did any of the five 26-DECISION.md citations drift since the spike," and the answer,
re-verified line-by-line, is no.

## Runtime State Inventory

*(Included because this phase performs a model rename + data-migration backfill.)*

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | The 11 real `CalendarEventTelescopeLabel` rows (`is_verified` history) in `src/fomo_db.sqlite3`, preserved through `RenameModel` because the model's OneToOne `event` field is its actual primary key (a `DeleteModel`/`CreateModel` pair would drop the table — already proven safe in Phase 26's scratch-copy test: 31/20/11 row counts identical before/after). Also: 31 `CampaignRun` rows needing `source`/`telescope_class` backfill values. | Code edit (hand-authored `RenameModel` + `AddField` + `RunPython` backfill migration) — no manual data migration outside Django's own migration framework |
| Live service config | **None found.** No external service (LCO Observation Portal, Gemini ToO API, ESO, MPC Obscodes API) stores or references the string `CalendarEventTelescopeLabel`/`CalendarEventMeta` — this is a purely internal Django model name, never sent over the wire | None |
| OS-registered state | **None found.** FOMO is a Django web app with no cron/systemd/pm2/Task-Scheduler registrations that reference this model by name; `[VERIFIED: grep -rn "CalendarEventTelescopeLabel" across the repo returns only Python source files listed in Pattern 4's table]` | None |
| Secrets/env vars | **None found.** No env var, SOPS key, or `.env` entry references this model's class name | None |
| Build artifacts | **None found.** No compiled binary or installed-package `egg-info` references this class name; the only "build artifact" risk is the six Python-source integration points already enumerated in Pattern 4, which are code edits, not build-artifact staleness | None beyond the six-point rename checklist itself |

## Common Pitfalls

### Pitfall 1: Reading CONTEXT.md's uppercase `telescope_class` vocabulary as the literal stored value
**What goes wrong:** implementing `TextChoices` with uppercase values (`'2M0'`, `'1M0'`,
`'0M4'`) breaks D-12's subset-assertion test against `calendar_utils`'s existing lowercase
aperture-class set (`{'0m4', '1m0', '2m0', '4m0'}`) unless the test explicitly case-folds.
**Why it happens:** CONTEXT.md's prose is written uppercase throughout D-11/D-12/D-13/D-16/D-19/D-20.
**How to avoid:** store lowercase (`'2m0'`/`'1m0'`/`'0m4'`), matching `calendar_utils`'s one
true source of the vocabulary; treat CONTEXT.md's capitalisation as prose styling. Confirm
with the user/planner before locking — flagged in Assumptions Log A1.
**Warning signs:** D-12's subset-assertion test failing on a case mismatch, not a genuine
vocabulary mismatch.

### Pitfall 2: Importing `solsys_code.models.CampaignRun` directly inside a `RunPython` function
**What goes wrong:** couples the migration to the *current* model definition, which breaks if
a later migration changes a field this one depends on — the exact reason Django's own docs
warn against it, and the exact reason this project's own `0004`/`0005` precedent migrations
use `apps.get_model()` instead.
**Why it happens:** it "just works" today and is one import shorter.
**How to avoid:** always `apps.get_model('solsys_code', 'CampaignRun')` inside the
`RunPython` function body, exactly as `0004_campaignrun_window_schema.py` and
`0005_campaignrun_campaign_run_window_start_end_null_together.py` already do. The
`calendar_utils.derive_telescope_class()` helper is the one exception (D-20's whole point) —
it takes primitives, not a model instance, so it's safe to import directly.
**Warning signs:** `RunPython` step raising `AttributeError` on a field only present in a
later migration state.

### Pitfall 3: Treating D-14's proven collision as a reason to add `telescope_class` to a constraint
**What goes wrong:** reopening a Phase 26 lock based on Test A's `IntegrityError` alone,
without reading Tests B/C/D — the collision is real but unreachable by any real ingest path
(see Pattern 1).
**Why it happens:** Test A in isolation looks like a bug.
**How to avoid:** read the full four-test sequence together; the conclusion is "no realistic
pair collides," not "add a constraint."
**Warning signs:** a plan task proposing to add `telescope_class` to
`unique_campaign_run_resolved_window`'s field tuple — this would be a Phase 26 lock reopen
requiring fresh human sign-off, and this research's evidence does not support it.

### Pitfall 4: Assuming `resolve_site()`'s tier-2 (MPC API) behaviour needs a live network call in CI
**What goes wrong:** a test that makes a real HTTP request to the MPC Obscodes API is slow,
flaky, and will fail in any offline/CI environment — exactly the risk D-16a flags.
**Why it happens:** the *real* D-16 repair task (a one-time data-fixing run against the live
dev DB, resolving pk=8/12 via a genuine tier-2 hit) is not automatable offline by definition.
**How to avoid:** the *automated test suite* should mock `requests.get` (or
`MPCObscodeFetcher.query`) exactly as `test_import_campaign_csv.py`'s
`_MPC_OBS_DATA_E10`/`@patch('requests.get')` pattern already does, proving the repair task's
*code path* is correct offline. The actual live repair run against real HST data is a
separate, manual, one-time execution — a "Manual-Only Verification," not a CI-gated test.
**Warning signs:** a new test importing `requests` without a `@patch`, or a CI run that
depends on network reachability to `minorplanetcenter.net`.

## Code Examples

### `RunPython` data migration, this project's own precedent shape
```python
# Source: solsys_code/migrations/0004_campaignrun_window_schema.py (verified in-repo)
def backfill_telescope_class(apps, schema_editor):
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    from solsys_code.calendar_utils import derive_telescope_class  # primitives-only helper: safe
    for run in CampaignRun.objects.filter(site__isnull=True):
        run.telescope_class = derive_telescope_class(
            site_raw=run.site_raw, telescope_instrument=run.telescope_instrument
        )
        run.save(update_fields=['telescope_class'])

class Migration(migrations.Migration):
    dependencies = [('solsys_code', '0008_...')]  # after RenameModel + the 3 AddFields
    operations = [
        migrations.RunPython(backfill_telescope_class, reverse_code=migrations.RunPython.noop),
    ]
```

### Migration-testing pattern, this project's own precedent
```python
# Source: solsys_code/tests/test_window_schema_migration.py (verified in-repo, adapt for Phase 27)
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

class TestTelescopeClassBackfill(TransactionTestCase):
    migrate_from = [('solsys_code', '0007_campaignrun_contact_public_opt_in')]
    migrate_to = [('solsys_code', '00XX_telescope_class_backfill')]

    def setUp(self):
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps
        CampaignRun = old_apps.get_model('solsys_code', 'CampaignRun')
        # ... seed rows mirroring pk 8/12/13/21/26/29/30/31's real shapes ...
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
```

### Mocking the MPC API for D-16a's repair-task tests
```python
# Source: solsys_code/tests/test_import_campaign_csv.py (verified in-repo pattern)
@patch('requests.get')
def test_repair_task_hst_resolves_via_mocked_tier2(self, mock_get):
    mock_response = MagicMock(ok=True)
    mock_response.json.return_value = _MPC_OBS_DATA_250  # HST fixture, mirroring _MPC_OBS_DATA_E10
    mock_get.return_value = mock_response
    site, needs_review = resolve_site('250')
    self.assertFalse(needs_review)
```

### Admin inline + `save_formset` attribution stamping (D-07)
```python
# Standard Django idiom [ASSUMED — not fetched from docs.djangoproject.com this session]
class CalendarEventMetaInline(admin.TabularInline):
    model = CalendarEventMeta
    fk_name = 'run'
    extra = 0

class CampaignRunObservationInline(admin.TabularInline):
    model = CampaignRunObservation
    fk_name = 'run'
    extra = 0
    readonly_fields = ['confirmed_by', 'confirmed_at']  # set only via save_formset, never hand-typed

class CampaignRunAdmin(admin.ModelAdmin):
    inlines = [CalendarEventMetaInline, CampaignRunObservationInline]

    def save_formset(self, request, form, formset, change):
        instances = formset.save(commit=False)
        for instance in instances:
            if isinstance(instance, CampaignRunObservation) and instance.pk is None:
                instance.confirmed_by = request.user
                instance.confirmed_at = timezone.now()
            instance.save()
        formset.save_m2m()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| An event's parent run found by matching telescope/instrument/date by hand | A stored `CalendarEventMeta.run` FK | Phase 27 (this phase) | RECON-05's ownership scoping becomes a query, not a heuristic |
| `telescope_class` did not exist; class-wide and unresolved runs were indistinguishable (`site=None` either way) | Explicit `telescope_class` field with a 4-value vocabulary (`2M0`/`1M0`/`0M4`/`SPACE`/blank) | Phase 27 (this phase) | CANON-02 satisfied; `site_needs_review` keeps its own distinct meaning |
| "Space missions are permanently site-less" (26-DECISION.md's original Criterion 3 framing) | Space observatories resolve to a real `Observatory` like any ground site; `SPACE` means specifically "no MPC code assigned" | 2026-07-29, this discussion (D-11) | Narrows the vocabulary from a recommended 3-meaning scheme to the locked 4-value one above; PROJECT.md's stale claim needs correcting (folded todo #3) |

**Deprecated/outdated:** 26-DECISION.md's Criterion 3 "three-meaning vocabulary" recommendation
(telescope-class / space-mission / unresolved) — superseded by D-11's narrower, corrected
vocabulary. Do not implement the spike's original recommendation; implement D-11/D-12 instead.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `telescope_class` stored values should be lowercase (`'2m0'`/`'1m0'`/`'0m4'`) to match `calendar_utils`'s existing vocabulary, despite CONTEXT.md's uppercase prose | Pattern 3, Pitfall 1 | D-12's subset-assertion test needs an explicit case-fold, or the admin/CSV-derived values silently never satisfy the subset check; low risk (easy one-line fix either way), but should be confirmed before the migration is written, since it's harder to change after real rows exist |
| A2 | The observation-link model's `related_name` on its `CampaignRun` FK should follow `CampaignRun`'s own existing style (e.g. `related_name='campaign_run_observations'` or similar) — no explicit CONTEXT.md guidance beyond "Claude's Discretion" for the model *name* itself | Pattern 6 | Low risk — purely a naming/readability choice, not a behavior risk; any consistent choice works |
| A3 | The `save_formset` code example shown (`instance.pk is None` gate, `formset.save(commit=False)` + per-instance stamping) is the correct idiom for stamping `confirmed_by`/`confirmed_at` only on newly-created rows | Pattern 5, Code Examples | Low risk — this is a very standard, stable Django idiom; if wrong, the failure mode is loud (an `AttributeError`/`IntegrityError` at save time, not silent data corruption) |

**If this table is empty:** N/A — three low-risk assumptions logged above, all confirmable
cheaply during planning/discuss rather than requiring new research.

## Open Questions

1. **Exact name for the observation-link model and its `source`/`telescope_class`
   `TextChoices` enum class name**
   - What we know: `CampaignRunObservation` is CONTEXT.md's own placeholder; Claude's
     Discretion explicitly leaves the final name open, following the same generality
     argument 26-DECISION.md used to reject `CalendarEventRunLink`.
   - What's unclear: nothing blocking — this is purely a naming choice.
   - Recommendation: planner picks a name consistent with `CalendarEventMeta`'s generality
     posture (e.g. `CampaignRunObservation`, `ObservationAttribution`, or similar) and states
     it once in the plan; no further research needed.

2. **Whether Observatory `E10`'s (Siding Spring) blank `timezone` field should be backfilled
   in this phase**
   - What we know: 26-DECISION.md's "Timezone gap found during this spike" section explicitly
     recommends "Phase 27 should backfill the timezone for this observatory record before the
     reconciler ships" — but this is a Phase-29-reconciler-blocking concern (site-local-night
     key derivation), not one of CANON-01..05's five success criteria, and CONTEXT.md's own
     Phase 27 scope boundary does not mention it.
   - What's unclear: whether the planner should fold this in as a small opportunistic fix
     (it's a one-row `Observatory.timezone` update, low effort) or explicitly defer it to
     Phase 29's own prep, since it has no CANON requirement behind it (mirroring the
     `SITE_TELESCOPE_MAP`-extraction todo's own "no CANON requirement, size before absorbing"
     framing).
   - Recommendation: surface to the planner as a candidate very-small addendum, not a required
     task — consistent with how CONTEXT.md treats the other no-CANON-requirement folded todo.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| MPC Obscodes API (`minorplanetcenter.net`) | D-16's live tier-2 resolution of HST (pk=8/12) | Not verified this session (network access not exercised) | — | The repair task's own code path (`resolve_site()`) already degrades gracefully to tier-3/flag-for-review on any network failure (`requests.exceptions.RequestException` caught) — see `campaign_utils.py:217-225`. Automated tests mock this call entirely (Pitfall 4); only the one-time live repair run needs real reachability |
| SQLite (dev DB) | All migration/backfill work | Yes | 3.x (Django default backend for this project) | — |
| Django admin/staff auth | Admin inlines (D-06/D-07), modal gating (D-09) | Yes — already in use throughout the codebase | Django 5.2.13 | — |

**Missing dependencies with no fallback:** none — this phase has no hard external dependency;
the one live-network step (D-16's HST tier-2 resolution) is explicitly accepted by CONTEXT.md
as a one-time, not-CI-gated cost (D-16a).

**Missing dependencies with fallback:** MPC API reachability during the live repair run —
`resolve_site()`'s existing network-failure handling already provides the fallback (flag for
review, tier-3 placeholder skipped since `create_placeholder` defaults `True` for this
call site — planner should confirm whether the repair task wants `create_placeholder=True`
or `False` for pk=8/12, since a network failure with `True` would fabricate a placeholder
`Observatory` for HST rather than leaving it flagged, which may not be the desired D-16
outcome on a bad day).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase`/`TransactionTestCase` (`./manage.py test`), plus `pytest` for the separate `tests/`/`src/`/`docs/` suite (unaffected by this phase — no app-level test lives there) |
| Config file | none dedicated — `pyproject.toml`'s `[tool.pytest.ini_options]` scopes `testpaths`; Django test discovery is `./manage.py test solsys_code` (app-default) |
| Quick run command | `./manage.py test solsys_code.tests.test_admin solsys_code.tests.test_campaign_models solsys_code.tests.test_campaign_approval solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_utils solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_import_campaign_csv` (mirrors Phase 26's own narrow selection, excluding the two SPICE-heavy modules) |
| Full suite command | `./manage.py test solsys_code` (pays the SPICE cost once, via `test_ephem_utils.py`/`test_views.py`) plus `python -m pytest` (unrelated suite, run for completeness) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CANON-01 | `source` written correctly per ingest path; approval derivation rule holds | unit/DB | `./manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_submission` | Both ✅ exist |
| CANON-02 | `telescope_class` backfill correctness (pk 8/12/13/21/26/29/30/31 shapes); D-12 subset assertion | unit/DB + migration test | `./manage.py test solsys_code.tests.test_campaign_models` + new `TransactionTestCase` (see Code Examples) | test_campaign_models.py ✅ exists; migration test ❌ Wave 0 |
| CANON-03 | Rename survives with `is_verified` intact; all six integration points pass | unit/DB + migration test | `./manage.py test solsys_code.tests.test_admin solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_calendar_template solsys_code.tests.test_load_telescope_runs` + new rename migration test | All four ✅ exist (need rename applied, per Pattern 4); migration test ❌ Wave 0 |
| CANON-04 | Observation link: `confirmed_by`/`confirmed_at` set only via `save_formset`; CASCADE on run delete | unit/DB | new tests in `test_campaign_models.py` or a new `test_campaign_run_observation.py` | ❌ Wave 0 |
| CANON-05 | Admin inlines visible/editable; modal shows run link, hidden for `pending_review` | unit/DB (admin client) + manual (visual modal check) | `./manage.py test solsys_code.tests.test_admin` (extend) | Extend existing ✅; manual step is out-of-suite |

### Sampling Rate
- **Per task commit:** the quick-run command above (fast, no SPICE cost).
- **Per wave merge:** `./manage.py test solsys_code` (full suite, once per wave, per this
  project's SPICE-cost-avoidance convention).
- **Phase gate:** full suite green + `ruff check .`/`ruff format --check .` clean before
  `/gsd-verify-work`, per CLAUDE.md's quality gates.

### Wave 0 Gaps
- [ ] A new migration-testing module (e.g. `test_canonical_record_migration.py`), following
  `test_window_schema_migration.py`'s exact `MigrationExecutor` shape, covering: rename
  preserves the 11 companion rows' `is_verified`; `source` defaults to the chosen legacy
  value for pre-existing rows; `telescope_class` backfill produces the correct value for each
  of the D-16 row shapes (JWST-alias, HST-tier2-mocked, Swift-empty, JUICE-empty,
  class-wide-empty, TBD-pair pk27/28-analogue).
- [ ] New tests for the observation-link model (`CampaignRunObservation` or chosen name):
  named `UniqueConstraint` fires on a genuine duplicate; `CASCADE` on run delete leaves the
  `ObservationRecord` untouched; `confirmed_by`/`confirmed_at` are set correctly by
  `save_formset` and left blank by any other write path.
- [ ] Extend `test_admin.py` with inline-formset submission tests proving D-07's stamping
  behavior (new row via admin gets `confirmed_by=request.user`; existing row edited via
  admin does *not* get re-stamped).
- [ ] A fixture/mock for the D-16a repair task's HST tier-2 resolution
  (`_MPC_OBS_DATA_250`-style, mirroring `test_import_campaign_csv.py`'s existing
  `_MPC_OBS_DATA_E10`).

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No new auth surface — admin already requires Django staff login; no new public endpoint |
| V3 Session Management | No | Unaffected — no new session state |
| V4 Access Control | Yes | The modal's new event->run link (D-08/D-09) is a **new non-staff-visible surface** — must read `CampaignRun.is_publicly_visible` (D-10) before rendering, exactly mirroring `CampaignRunTableView.get_queryset()`'s existing `exclude(approval_status=PENDING_REVIEW)` gate. Admin inlines are already gated by Django's own `is_staff`/model-permission checks (existing `ModelAdmin` convention, unchanged by this phase) |
| V5 Input Validation | Yes | `source`/`telescope_class` are `TextChoices` (enum-validated at the model layer, matching `ApprovalStatus`/`RunStatus`'s existing pattern) — never free-text for these two fields |
| V6 Cryptography | No | Not applicable — no new secret/credential handling |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A non-staff user seeing a `pending_review` run's telescope/window info via the new calendar modal link, bypassing the run table's existing gate | Information Disclosure | D-09/D-10's `is_publicly_visible` template check — must be applied in the *new* `event_form.html` override exactly as it would be in the run table, since this is a genuinely new rendering path that the existing queryset-level `exclude()` does not cover |
| An admin-created observation-link row silently lacking `confirmed_by`/`confirmed_at`, breaking ATTRIB-03's "no association without explicit staff confirmation" downstream in Phase 28 | Tampering / Repudiation | D-07's `save_formset` override is the structural fix — without it, an inline-created row would look "confirmed" by D-01's own rule (row exists = confirmed) but carry no attribution, which is exactly the gap D-07 exists to close |
| A malformed/malicious CSV `Telescope / Instrument` or `Site Code` cell being trusted as a literal `Observatory.obscode`/`telescope_instrument` value | Tampering | Already mitigated by existing code (`resolve_site()`'s length guard, `_MAX_OBSCODE_LEN` computed from the field itself) — unchanged by this phase, no new input surface introduced by `source`/`telescope_class` since both are derived/enum values, not user-typed free text |

## Sources

### Primary (HIGH confidence — direct code read/execution this session)
- `solsys_code/models.py` — full read, `CampaignRun`/`CalendarEventTelescopeLabel` constraints and fields
- `solsys_code/campaign_utils.py` — full read, `resolve_site()`, `insert_or_create_campaign_run()`, `HORIZONS_OBSERVER_TO_OBSCODE`
- `solsys_code/calendar_utils.py` — partial read (lines 1-330), `SITE_TELESCOPE_MAP`, the five folded-todo helpers, `insert_or_create_calendar_event()`
- `solsys_code/campaign_views.py` — partial read, `ALLOWED_FIELDS_FOR_NON_STAFF`, `CampaignRunSubmissionView`, resolve-site decision flow
- `solsys_code/admin.py` — full read
- `solsys_code/management/commands/import_campaign_csv.py` — full read
- `solsys_code/migrations/0004_campaignrun_window_schema.py`, `0005_...`, `0007_...` — read for migration-shape precedent
- `solsys_code/tests/test_window_schema_migration.py` — full read, migration-testing precedent
- `solsys_code/tests/test_import_campaign_csv.py`, `test_campaign_approval.py` — grepped for MPC-API mocking precedent
- Executed proof against a disposable scratch copy of `src/fomo_db.sqlite3` (Tests A/B/C/D, this session, 2026-07-29) — real DB fingerprint (md5sum `fdf94f7924f52424b8e0b14a4e26d8c1`) confirmed unchanged before and after
- Installed package inspection: `django.contrib.admin.ModelAdmin.save_formset` source, `tom_observations.models.ObservationRecord._meta.pk`, `tom_calendar`'s `views.py`/`event_form.html`
- `.planning/phases/26-canonical-record-spike/26-DECISION.md`, `docs/design/canonical_record_spike.rst` — read in full, cross-checked against live code (all citations confirmed still accurate)

### Secondary (MEDIUM confidence)
- None — this phase required no external documentation lookups (no new libraries).

### Tertiary (LOW confidence)
- The `save_formset` code example's exact idiom (Code Examples, Assumptions Log A3) — standard
  Django pattern, not verified against docs.djangoproject.com this session.

## Metadata

**Confidence breakdown:**
- Standard stack: N/A — no new dependency.
- Architecture (rename checklist, migration shape, admin/template integration): HIGH — every
  claim re-verified against live code this session, matching Phase 26's own citations exactly.
- D-14 resolution: HIGH — executed proof against a disposable copy of the real dev DB, four
  distinct test scenarios, real DB fingerprint confirmed unchanged.
- Pitfalls: HIGH for migration/testing pitfalls (grounded in this project's own existing
  precedent code); MEDIUM for the case-convention pitfall (a genuine open implementation
  choice, not a code-verified fact).

**Research date:** 2026-07-29
**Valid until:** Until Phase 27's plan is written and executed — this research is
code-state-specific (line numbers, exact field lists) and should be treated as stale the
moment any of the cited files change. No fixed 30/7-day estimate applies; re-verify citations
if execution is delayed by more than a few days.
