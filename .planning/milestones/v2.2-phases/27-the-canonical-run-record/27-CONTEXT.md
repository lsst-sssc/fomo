# Phase 27: The Canonical Run Record - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 27 makes `CampaignRun` canonical **in the schema**: it records how it was created
(`source`), says why it has no site when it has none (`telescope_class`), owns the calendar
events that show it (a `run` link on the renamed companion record), and owns the observation
records that realise it (a new link model) — with every existing row and all six
rename integration points surviving the change. CANON-01 through CANON-05.

**In scope:** the model changes, their migrations and backfills, the admin surfaces that make
the new links visible and editable, one calendar-template change so an event links back to its
run, and a data-repair task for site-less runs that this discussion discovered are stale rather
than unresolvable.

**Out of scope:** the reconciler (Phase 29), the attribution queue and its confidence scoring
(Phase 28), rewiring the four ingest adapters to create `CampaignRun`s (v2.3 / ADAPT-01..03),
and the adopt-vs-gap-fill write strategy (deliberately deferred by Phase 26 to Phase 29).

</domain>

<decisions>
## Implementation Decisions

### The observation-record link (CANON-04)

- **D-01: Confirmed rows only.** A link row exists only once a staff member confirms it.
  Phase 28 computes attribution candidates on the fly and writes nothing until confirmation.
  This keeps ATTRIB-03 ("no association without explicit staff confirmation") structural rather
  than a rule code must remember, and it keeps 26-DECISION.md's calendar-side ownership rule
  ("a companion record whose `run` link is unset means not mine, never touch") true by
  construction — an unconfirmed guess can never be mistaken for ownership.

  The consequence Phase 28 inherits: dismissed candidates have nowhere to persist, so Phase 28
  must decide where dismissals live. That is Phase 28's decision, not this phase's.

- **D-02: One run per observation record, expressed so it can be broadened cheaply.** Modelled
  as `ForeignKey(ObservationRecord)` plus a **named `UniqueConstraint`**, not a
  `OneToOneField`. Behaviour today is identical, but broadening later is a single
  `RemoveConstraint` with no field change and no reader rewrites, because the reverse accessor
  is already a manager. A `OneToOneField` would instead require an `AlterField` that changes the
  accessor from an object to a manager, breaking every reader at once.

  This mirrors `CampaignRun.Meta.constraints`' existing use of named partial unique constraints,
  so it is in-house convention rather than a new pattern. The residual cost of broadening is
  semantic (gap analysis and the reconciler would each need a rule for a doubly-claimed record),
  not technical.

- **D-03: The link records who and when.** `confirmed_by` (FK to `User`, `on_delete=SET_NULL`)
  and `confirmed_at` (timestamp). No boolean — under D-01 the row's existence already means
  "confirmed", so a flag would be redundant state that could contradict the row.

- **D-04: Deleting a run deletes the link row** (`CASCADE` on the run FK). The
  `ObservationRecord` is on the other side of the relation and is untouched, satisfying success
  criterion 4. Deliberately *not* `SET_NULL`: 26-DECISION.md chose `SET_NULL` for
  `CalendarEventMeta.run` because that row also carries `is_verified` and must survive, whereas
  a run-less observation link would carry nothing and mean nothing.

- **D-05: `CalendarEventMeta.run` stays a bare FK.** No `confirmed_by`/`confirmed_at` on the
  event side, exactly as Phase 26 locked it. The resulting audit asymmetry between the two link
  types — an observation attribution records who and when, an event attribution does not — is
  accepted deliberately, not overlooked. Phase 28's undo on the event side will be untraceable.

### Where staff see the links (CANON-05)

- **D-06: Editable admin inlines now, a real staff page in Phase 28.** Phase 27 adds a
  `CalendarEventMeta` inline and a `CampaignRunObservation` inline to the existing
  `CampaignRunAdmin`, satisfying CANON-05 without building a view, URL and template. The
  staff-facing run-detail page is deferred to Phase 28, where the attribution UI it would host
  actually gets built. There is no run-detail view today — `campaign_urls.py` has only
  list / table / submit / approval-queue / decide / gaps / site-search.

- **D-07: The inlines are editable, and that obliges `save_formset` wiring.** Editable inlines
  are what let Phase 27 create a link at all (nothing else can until Phase 28 ships), but they
  are also a path that bypasses the confirmation flow. `CampaignRunAdmin.save_formset()` must
  therefore populate `confirmed_by=request.user` and `confirmed_at=now` on admin-created rows.
  Without it, the audit trail D-03 designs has holes on precisely the rows this phase creates.
  **This is a required task, not an implementation detail to discover during execution.**

- **D-08: An event links back to its run from the calendar event modal.** Verified feasible
  during discussion, not assumed: `tom_calendar.views.update_event` renders
  `tom_calendar/partials/event_form.html` with `event` already in the template context, so
  `event.telescope_label_meta.run` is reachable through a **template override alone** — no view
  override, no URL change. This is the same mechanism `src/templates/tom_calendar/partials/
  calendar.html` already uses for `event.telescope_label_meta.is_verified`.

  The cost is real and should be stated: FOMO takes ownership of a second upstream
  `tom_calendar` template, which will drift on tomtoolkit upgrades.

- **D-09: The modal link is shown to everyone, but hidden for `pending_review` runs.** This
  mirrors `CampaignRunTableView.get_queryset()`'s existing `exclude()`. Note that this *adds*
  non-staff-visible surface rather than changing existing behaviour — success criterion 1's
  "existing non-staff visibility behaviour is unchanged" is about the run table's approval
  gating, which is untouched.

- **D-10: The visibility rule gets one definition.** Add a `CampaignRun.is_publicly_visible`
  property (`approval_status != PENDING_REVIEW`) and have the modal template read it. The
  existing view keeps its queryset-level `exclude()` because a Python property cannot be used in
  a filter — so this is one definition in meaning, two in code, which is the best available
  given the ORM constraint. The point is that the `pending_review` literal never appears in a
  template where nothing would catch it drifting from the model's `TextChoices`.

### `telescope_class` (CANON-02)

**This area overturned a premise in 26-DECISION.md. Read D-11 before planning anything that
depends on the spike's "three-meaning vocabulary" recommendation.**

- **D-11: The spike's premise that space missions are permanently site-less is false, and the
  vocabulary is narrower than it recommended.** 26-DECISION.md's Criterion 3 recommends
  `telescope_class` carry a three-meaning vocabulary (telescope-class allocation, space mission,
  unresolved) on the grounds that "the live data has five space-mission rows (pk=8, 12, 13, 21,
  26) against two class-wide ones (pk=29, 30)". Corrected by the project owner during this
  discussion and confirmed against the code and DB: **space observatories resolve to an
  `Observatory` like any ground site**, via an MPC obscode or via the Horizons→MPC alias table
  that already exists at `solsys_code/campaign_utils.py:43-46` (added by quick task `260726-fqb`).
  `Observatory` already holds `274` (JWST), `289` (Roman), `C51` (WISE).

  The genuine exception is a space observatory with a Horizons code but **no MPC code assigned
  at all** — JUICE (`500@-28`), which cannot be aliased to an obscode because none exists.
  Swift has `C52`; HST has `250`; JWST has `274`.

  **Final vocabulary:** `2M0`, `1M0`, `0M4` (telescope-class allocation), plus `SPACE` meaning
  specifically *a space observatory with no MPC code assigned*. Blank otherwise. **"Unresolved"
  is deliberately NOT a `telescope_class` value** — `site_needs_review` already carries exactly
  that meaning and is already wired into the approval queue's site-resolution work list.

- **D-12: Three classes, not four — and the cross-check test is a subset assertion.**
  `telescope_class` gets normal Django `TextChoices` on `CampaignRun` (needed anyway for admin
  filters and form rendering), limited to `2M0`/`1M0`/`0M4` per CANON-02's wording. The existing
  code vocabulary in `calendar_utils._aperture_class_from_telescope_code` (line 102) has **four**
  — it includes `4m0`, and `SITE_TELESCOPE_MAP` has `('sor', '4m0'): 'SOR-4m0'` for SOAR.

  A test asserts every `telescope_class` value appears in `calendar_utils`' aperture-class set —
  a **subset** assertion, not equality, since equality would fail on day one over `4m0`. The
  test must name `4m0` explicitly as the known-excluded value, so a future reader does not
  "fix" the discrepancy by adding `4M0` to the model.

- **D-13: Backfill by derived rule, in a data migration.** Not a hand-enumerated pk list, which
  would be correct only for the dev DB. The rule must leave `telescope_class` blank when a
  site-less row shows neither a class nor a no-MPC-code space signal, so it stays correct against
  any other database — that blank-plus-flagged state is what a genuine resolution failure
  correctly looks like.

- **D-14: Success criterion 2 conflicts with a Phase 26 lock; measure before choosing.**
  ROADMAP criterion 2 requires a class-wide run and an unresolved run to "coexist for the same
  campaign, telescope and window without colliding", but 26-DECISION.md locked `telescope_class`
  out of **both** partial unique constraints, and the SPIKE-01 finding proved rows differing only
  by a new field *do* collide (that was the intended result for `source`).

  **Research must construct the actual colliding pair against the real constraints** and
  determine whether any real ingest path can produce it — the same executable-proof standard
  Phase 26 held itself to. If no realistic pair collides, record that reading with the evidence.
  If one does, reopen the lock with evidence rather than by argument. Do not resolve this by
  reading the requirement more loosely.

- **D-15: pk=31 is not an anomaly.** Initially framed as one during discussion; that framing was
  wrong. `site_raw='X05'` **is** a valid `Observatory` (Simonyi Survey Telescope, Rubin). The row
  is `rejected`, and site resolution only runs at approval, so resolution was never attempted
  rather than having failed. The row stays. It is a throwaway screenshot row per the project
  owner, but deleting it would make Phase 26's recorded 31-row counts and DB fingerprint describe
  a database that no longer exists, for no benefit.

- **D-16: Stale site-less rows are repaired in Phase 27, as their own separately-committed
  task.** This discussion discovered that several site-less rows are *stale*, not unresolvable —
  they were imported before the Horizons alias table landed on 2026-07-26. Measured reach of
  re-running `resolve_site()`:

  | Rows | `site_raw` | What happens |
  |---|---|---|
  | pk 21, 27, 28 (JWST) | `500@-170` | Alias → `274`; tier 1 hit on the existing Observatory. Offline. |
  | pk 8, 12 (HST) | `250` | Tier 1 miss → **tier 2 MPC Obscodes API** → creates the HST Observatory row. |
  | pk 13 (Swift) | *empty* | `resolve_site` returns `(None, True)` immediately — no tier runs. Needs a code first. |
  | pk 26 (JUICE) | *empty* | Same; and no MPC code exists, so it gets `telescope_class=SPACE`. |
  | pk 29, 30 | *empty* | Class-wide; get `1M0` / `2M0`. |

  **D-16a:** the task calls full `resolve_site()` including tier 2, so HST resolves via the MPC
  API. Accepted cost: a data-changing task now depends on a third-party API being reachable, and
  its result is not reproducible offline or in CI. Plan the tests accordingly.

  **D-16b:** the task also sets `site_raw='C52'` for pk=13 (Swift) so it can resolve. This value
  is supplied by the project owner as domain authority — it is not inferred, and it does not
  generalise to another database.

  Note this task has no CANON requirement behind it. It is included because the
  `telescope_class` backfill would otherwise operate on data known to be stale.

- **D-17: `site_needs_review` is not touched by the backfill.** The earlier framing that 9 rows
  were "permanently stuck" in the site-resolution queue was wrong — the space rows are stale, not
  stuck, and D-16 repairs them through the normal resolution path, which clears the flag the way
  it always does. No special-case flag clearing.

### New fields' visible surface (CANON-01/02)

- **D-18: `telescope_class` is visible to non-staff; `source` is not.** `telescope_class` joins
  `ALLOWED_FIELDS_FOR_NON_STAFF` (`campaign_views.py:70`) because it is observing information of
  the same kind as `site_raw` and `filters_bandpass`, which are already public. `source` stays
  staff-only: it is internal provenance about which FOMO ingest path created the row, and
  `LEGACY` in particular would surface as a meaningless value to an outside reader.

  That list is deliberately hand-enumerated rather than introspected, so **a new field is
  invisible to non-staff unless explicitly added, silently** — the omission of `source` must be
  deliberate and commented, not accidental.

- **D-19: Both fields get `list_display` and `list_filter` in the admin.** Filtering by `source`
  is how staff would audit "which runs came from the CSV import"; by `telescope_class`, how they
  would find class-wide runs. Consistent with `approval_status`/`run_status`/`campaign` already
  being filters. `source` is **not** added to `readonly_fields` — only `approval_status` stays
  read-only there, for its existing documented reason.

- **D-20: One shared derivation helper, taking primitives.** The `telescope_class` derivation is
  extracted into a helper that **both** the data migration and `import_campaign_csv` call, so a
  newly imported class-wide run gets the same value an existing one got, instead of the field
  rotting the moment the backfill lands.

  **The helper must take primitives** (`site_raw`, `telescope_instrument`) rather than a model
  instance, so the migration can import it without coupling to a model that keeps changing
  through Phases 28–29. This is the mitigation for the usual "migrations should be
  self-contained" objection, and it is a requirement of the decision, not a style preference.

  **Home:** `calendar_utils.py`, next to `_aperture_class_from_telescope_code` and
  `SITE_TELESCOPE_MAP`, which is where the class vocabulary already lives.

### Resolved during planning (2026-07-29, after research)

Research surfaced four open points that materially changed the plan. All four were decided by the
project owner before the planner ran.

- **D-21: `telescope_class` stores lowercase `2m0` / `1m0` / `0m4`.** Research found a
  case-convention conflict: this document writes the vocabulary uppercase throughout D-11..D-20,
  but `calendar_utils._aperture_class_from_telescope_code` already holds it lowercase
  (`{'0m4', '1m0', '2m0', '4m0'}`). Since D-20 itself names `calendar_utils.py` as "where the
  class vocabulary already lives", the lowercase form wins and this document's capitalisation is
  prose styling, not a literal value. **Consequence:** D-12's subset assertion compares directly
  with no case-folding, which keeps it a genuine drift detector — a casing divergence would be
  caught rather than silently normalised. `TextChoices` *labels* may still render uppercase to
  users; only the stored values are lowercase. `SPACE` is unaffected.

- **D-22: D-16's repair task calls `resolve_site(..., create_placeholder=False)` for HST
  (pk 8, 12).** `resolve_site`'s default is `True`, which on a tier-2 network failure would
  fabricate a placeholder `Observatory` for HST rather than leaving the row flagged. With `False`,
  a network failure leaves the row site-less and `site_needs_review` set — exactly the state D-17
  says a genuine resolution failure should look like. This makes D-16a's accepted "not reproducible
  offline" cost fail safe instead of fail silent: a bad day produces no row, not a fake one.

- **D-23: Observatory `E10` (Siding Spring)'s blank `timezone` is backfilled in this phase, as its
  own separately-committed task.** 26-DECISION.md's "Timezone gap found during this spike" section
  asked Phase 27 to do this before the Phase 29 reconciler ships, which needs it for site-local-night
  key derivation. It is a one-row update and has no CANON requirement behind it — the same standing
  as D-16's repair task, and it is committed separately for the same reason.

- **D-24: The `SITE_TELESCOPE_MAP` / instrument-extraction module split is dropped back out of this
  phase.** It was the one folded todo flagged at selection time as having no CANON requirement, and
  D-20 removed any dependency on it by homing the new derivation helper in `calendar_utils.py`.
  Phase 27 already carries two model changes, a rename across six integration points, a new link
  model, several migrations with backfills, admin inlines, a template override, and two data-repair
  tasks. The todo stays open for a later cleanup pass. **The other three folded todos remain in
  scope**, including the `calendar_utils.py` private-helper rename and its test-module split.

### Claude's Discretion

- The exact name of the observation-link model (`CampaignRunObservation` is used throughout this
  document as a placeholder). Pick a name consistent with `CalendarEventMeta`'s generality
  posture — 26-DECISION.md rejected `CalendarEventRunLink` for being too link-specific.
- Whether the `is_publicly_visible` property is also used to simplify any existing call site
  beyond the modal template.
- Test organisation, beyond the specific subset assertion required by D-12.

### Folded Todos

All four matched todos were folded into this phase's scope.

- **`2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`** — drop the
  leading underscore on five cross-module-consumed helpers in `calendar_utils.py`
  (`_aperture_class_from_telescope_code`, `_derive_telescope`, `_resolve_placement_block`,
  `_extract_instrument`, `_coarse_telescope_label`). 26-DECISION.md explicitly recommends Phase 27
  do this rather than a separate cleanup pass, since the phase is editing these modules anyway.
  D-12 and D-20 both reference `_aperture_class_from_telescope_code` directly, so this phase
  touches it regardless. The todo's second half — moving `calendar_utils.py`-owned tests out of
  `test_sync_lco_observation_calendar.py` into their own module — is part of the same todo and
  should be scoped explicitly by the planner rather than silently dropped.

- **`2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md`** — `26-CONTEXT.md`
  still describes queue-run windows as sets of owned nights needing backfill, a framing
  26-DECISION.md's domain correction retracted. The todo says to fix it before Phase 29 is
  planned. Docs-only.

- **`2026-07-27-correct-project-md-stale-phase-25-calendar-event-claim.md`** — PROJECT.md's
  Phase 25 paragraph does not reproduce against the live dev DB (26-DECISION.md D-16). Docs-only.
  **This phase adds a second correction to make:** PROJECT.md and any doc repeating the spike's
  "five space-mission rows are permanently site-less" premise is now known false per D-11.

- **`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`** — extract
  `SITE_TELESCOPE_MAP` and instrument extraction into their own module. **Flagged at selection
  time as the one folded todo with no CANON requirement behind it, and it should be sized before
  it lands in the plan.** D-20 places the new derivation helper in `calendar_utils.py` rather
  than in a new module, so this extraction is *not* a prerequisite for anything in Phase 27.
  **Resolved: dropped from this phase by D-24.** The remaining three folded todos stay in scope.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes

- `.planning/phases/26-canonical-record-spike/26-DECISION.md` — the locked verdicts Phase 27
  implements. Read Criterion 1 (source vocabulary, constraint interaction, the
  `APPROVED`-and-not-`WEB` derivation rule) and Criterion 4 (migration shape, the six-point
  rename checklist, the `related_name`-unchanged decision) in full. **Read its Criterion 3
  "three-meaning vocabulary" recommendation against D-11 above — this discussion falsified its
  premise about space missions.**
- `docs/design/canonical_record_spike.rst` — the durable, redaction-free summary of the same
  decisions, written for Phases 27-29 to reference. Also needs the D-11 correction.
- `.planning/REQUIREMENTS.md` — CANON-01..05 and the Out of Scope table, which explicitly rejects
  `GenericForeignKey` for both links, renaming `related_name='telescope_label_meta'`, making the
  `run` link required, and any new dependency.
- `.planning/ROADMAP.md` §Phase 27 — the five success criteria. Criterion 2 conflicts with a
  Phase 26 lock; see D-14.

### Code this phase changes or depends on

- `solsys_code/models.py` — `CampaignRun` (lines 30-163, including both partial unique
  constraints and the window-null-together check) and `CalendarEventTelescopeLabel` (lines 7-27).
- `solsys_code/campaign_utils.py:33-46` — `HORIZONS_OBSERVER_TO_OBSCODE`, and `resolve_site()` at
  line 146 with its three-tier behaviour and `create_placeholder` flag. Central to D-11 and D-16.
- `solsys_code/calendar_utils.py:37-52` — `SITE_TELESCOPE_MAP`; `:84-104` —
  `_aperture_class_from_telescope_code`, which holds the four-value class vocabulary D-12
  cross-checks against.
- `solsys_code/campaign_views.py:70-87` — `ALLOWED_FIELDS_FOR_NON_STAFF`; `:117-151` — the
  staff/non-staff queryset split; `:340-353` — the site-resolution work queue that reads
  `site_needs_review`.
- `solsys_code/admin.py` — rename integration point 1, and where D-06/D-07/D-19 land.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — rename integration point 2.
- `solsys_code/views.py` `.prefetch_related('telescope_label_meta')` — rename point 3, safe by
  construction.
- `src/templates/tom_calendar/partials/calendar.html:228,244` — rename point 4, safe by
  construction; also the precedent for D-08's template-override approach.
- `solsys_code/management/commands/import_campaign_csv.py:194` — writes
  `ApprovalStatus.APPROVED` already; its real CANON-01 change is writing `source`, not changing
  approval. Also the second caller of D-20's shared helper.

### Paired docs (CLAUDE.md rule — required in `files_modified` up front)

- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — CANON-01 changes what
  `import_campaign_csv` writes, and D-20 adds `telescope_class` derivation to it. Behaviour
  change, so the notebook is in scope from the start, regenerated with real executed output.
- `docs/runbooks/telescope_runs_calendar.rst` — documents `import_campaign_csv`'s approval
  behaviour, which changes for non-web sources.

### Upstream code read during discussion (not modified)

- `tom_calendar/views.py:185-210` (`update_event`) and
  `tom_calendar/templates/tom_calendar/partials/event_form.html` — the render path D-08 overrides.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable assets

- **`resolve_site()`** (`campaign_utils.py:146`) — three-tier resolution with a documented
  `create_placeholder` flag, already handling Horizons observer notation, MPC API failure, and
  placeholder detection. D-16's repair task calls it directly rather than reimplementing.
- **`CampaignRun.Meta.constraints`** — named partial `UniqueConstraint`s with explanatory
  comments. D-02's observation-link uniqueness follows this same in-house pattern.
- **`StaffRequiredMixin`** (`mixins.py`) — the existing staff gate, available if Phase 28's
  run-detail page needs it.
- **`is_placeholder_observatory()`** — already distinguishes a real resolution from a fabricated
  placeholder; relevant to interpreting D-16's results.

### Established patterns

- **Hand-enumerated allow-lists over introspection** — `ALLOWED_FIELDS_FOR_NON_STAFF` is
  explicitly not derived from `_meta`, so field visibility is a deliberate act. D-18 must respect
  this and comment the omission of `source`.
- **Queryset-level gating, not template conditionals** — the non-staff run table excludes pending
  rows in SQL so they never enter the SELECT. D-09/D-10 add a template-level check only because
  the modal is rendered by an upstream view; this is the exception, and it is why D-10 insists on
  a single property rather than an inline literal.
- **Read-only admin fields where a model transition has side effects** — `approval_status` is
  read-only in `CampaignRunAdmin` because its transition triggers calendar projection in the
  view. D-19 deliberately does not extend this to `source`.
- **Template overrides of `tom_calendar` partials** — already done once for `calendar.html`;
  D-08 makes it twice.

### Integration points

- The six rename points from 26-DECISION.md Criterion 4 — two that fail loudly at Django startup
  or command import (`admin.py`, `sync_lco_observation_calendar.py`), two safe by construction
  (`views.py` prefetch, `calendar.html`) because `related_name` is unchanged, and two test-side
  points the original four-point checklist missed (`test_admin.py`'s
  `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')`, and class-name
  references in three test modules).
- `CampaignRunAdmin` — gains two inlines plus `save_formset` wiring (D-06/D-07).
- `import_campaign_csv` — gains `source` and the shared `telescope_class` derivation (D-20).

</code_context>

<specifics>
## Specific Ideas

- **`SPACE` means one specific thing**, supplied by the project owner: a space observatory that
  has a Horizons code but no MPC code assigned. JUICE (`500@-28`) is the case. Swift is *not*
  (it has `C52`), HST is *not* (`250`), JWST is *not* (`274`). Do not widen `SPACE` back into
  "any space mission" — that is exactly the premise D-11 falsified.
- **pk=13 (Swift) gets `site_raw='C52'`**, supplied directly by the project owner.
- **pk=31 is a throwaway screenshot row**, not indicative of real-world workflow — but it is kept
  (D-15), so no test or rule should treat it as representative data.
- The user explicitly asked how hard it would be to broaden the observation link to many runs
  later. D-02 is the answer to that question, chosen for that reason.

</specifics>

<deferred>
## Deferred Ideas

- **A real staff run-detail page** — deferred to Phase 28 by D-06, where the attribution UI it
  would host gets built.
- **Where dismissed attribution candidates persist** — a consequence of D-01 that Phase 28 must
  answer.
- **Audit fields on `CalendarEventMeta.run`** — declined by D-05; if Phase 28's undo flow proves
  the event-side gap painful, that is the phase to revisit it.
- **Adding `4M0` to `telescope_class`** — declined by D-12 to match CANON-02's wording. Revisit if
  a SOAR class-wide allocation ever arrives; the test's explicit `4m0` note is the breadcrumb.
- **Adding `telescope_class` to a unique constraint** — only if D-14's measurement shows a real
  colliding pair.
- **Extracting `SITE_TELESCOPE_MAP` and instrument extraction into their own module** — folded in
  as a todo, but D-20 removed any dependency on it. **Dropped back out of Phase 27 by D-24**; the
  todo stays open for a later cleanup pass.
- **v2.3 items untouched here:** adapter rewiring (ADAPT-01..03), provenance-blind gap analysis
  (GAPB-01), status vocabulary unification (STATUS-01/02), unused-allocation display (UNUSED-01).
- **The adopt-vs-gap-fill write strategy** — Phase 26 deferred it to Phase 29 by explicit human
  decision. Not reopened here.

</deferred>

---

*Phase: 27-the-canonical-run-record*
*Context gathered: 2026-07-29*
