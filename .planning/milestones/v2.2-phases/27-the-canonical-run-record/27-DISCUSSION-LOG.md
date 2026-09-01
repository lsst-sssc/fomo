# Phase 27: The Canonical Run Record - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-29
**Phase:** 27-the-canonical-run-record
**Areas discussed:** The observation-record link, Where staff see the links, telescope_class values, New fields' visible surface

---

## Todo folding

| Option | Description | Selected |
|--------|-------------|----------|
| calendar_utils helper renames | Drop the leading underscore on 5 cross-module-consumed helpers; 26-DECISION.md recommends Phase 27 do it | ✓ |
| Correct owned-nights framing | 26-CONTEXT.md still describes queue-run windows as sets of owned nights; docs-only | ✓ |
| Correct PROJECT.md Phase 25 claim | PROJECT.md's Phase 25 paragraph doesn't reproduce against the live dev DB; docs-only | ✓ |
| Extract site/telescope mapping module | Move SITE_TELESCOPE_MAP and instrument extraction into their own module | ✓ |

**User's choice:** all four folded.
**Notes:** The extraction todo was flagged at selection time as the only one with no CANON
requirement behind it, and should be sized before it lands in the plan. D-20 later placed the new
derivation helper in `calendar_utils.py` instead of a new module, so the extraction became a
non-prerequisite.

---

## The observation-record link

Opened by surfacing an asymmetry between CANON-03 and CANON-04: 26-DECISION.md's ownership rule
treats a set `run` link as the confirmation signal, while CANON-04 implies a link row can exist
unconfirmed. Both cannot be the model.

| Option | Description | Selected |
|--------|-------------|----------|
| Confirmed rows only | Link row exists only once staff confirm; candidates computed on the fly. Keeps ATTRIB-03 structural and the ownership rule true by construction. Dismissals have nowhere to persist | ✓ |
| Suggestions stored, flagged unconfirmed | Queue is the unconfirmed rows; free persistence for scores and dismissals. Forces the ownership rule to change to "set AND confirmed", a correctness risk Phase 29 inherits | |
| Split: suggestions in their own model | Phase 27 ships the confirmed link only; Phase 28 adds a separate candidate model | |

**User's choice:** Confirmed rows only.

| Option | Description | Selected |
|--------|-------------|----------|
| At most one run per record | Unique constraint on the record side; matches the measured pk=1 case | ✓ |
| Many runs per record allowed | Plain FK; allows genuinely ambiguous shared observations | |
| You decide | Pick from live data and Phase 28/29 needs | |

**User's choice:** "At most one sounds like the correct answer; how hard would it be to broaden
this later and make an ObservationRecord belong to more than one Campaign?"
**Notes:** This question changed the answer's shape. Answered that the schema change is cheap
either way but reader assumptions cost — and that modelling it as `ForeignKey` + a named
`UniqueConstraint` rather than `OneToOneField` makes broadening a single `RemoveConstraint` with
zero reader rewrites, because the reverse accessor is already a manager. Recorded as D-02.

| Option | Description | Selected |
|--------|-------------|----------|
| Who and when | confirmed_by (User FK, SET_NULL) + confirmed_at; no bool needed since row existence means confirmed | ✓ |
| Just a timestamp | Records when without who | |
| Nothing — existence is the record | Bare link row, no audit trail | |

**User's choice:** Who and when.

| Option | Description | Selected |
|--------|-------------|----------|
| Delete the link row | CASCADE on the run FK; ObservationRecord untouched | ✓ |
| Keep the row, clear the run | SET_NULL, mirroring CalendarEventMeta.run | |
| Block the deletion | PROTECT | |

**User's choice:** Delete the link row.
**Notes:** Deliberately diverges from the calendar side, because `CalendarEventMeta` also carries
`is_verified` and must survive, whereas a run-less observation link would carry nothing.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — same audit on both | Add confirmed_by/confirmed_at to CalendarEventMeta too | |
| No — keep the event link bare | Exactly as Phase 26 locked it; smaller migration | ✓ |
| Defer to Phase 28 | Let the attribution UI decide | |

**User's choice:** No — keep the event link bare.
**Notes:** The audit asymmetry between the two link types is therefore accepted deliberately;
Phase 28's undo on the event side will leave no trace.

---

## Where staff see the links

| Option | Description | Selected |
|--------|-------------|----------|
| Django admin inlines | Cheapest; staff-gated by construction; no new templates | |
| New staff run-detail page | Fills a real gap (none exists today) but is the largest non-schema piece of the phase | |
| Both — admin now, page later | Admin inlines in 27; the real page in 28 alongside the attribution UI | ✓ |

**User's choice:** Both — admin now, page later.

| Option | Description | Selected |
|--------|-------------|----------|
| Admin only | run on CalendarEventMetaAdmin; zero template risk | |
| In the calendar event modal | Where staff actually look; costs a second upstream template override | ✓ |
| On the calendar tile itself | No new override, but the tile is already very dense | |

**User's choice:** In the calendar event modal.
**Notes:** Verified feasible before recording rather than assumed — `tom_calendar.views.update_event`
renders `partials/event_form.html` with `event` in context, so `event.telescope_label_meta.run` is
reachable via a template override alone, no view override. Same mechanism the existing
`calendar.html` override already uses for `is_verified`. This made the option materially cheaper
than originally framed.

| Option | Description | Selected |
|--------|-------------|----------|
| Staff only | Matches CANON-05's wording; leak impossible by construction | |
| Everyone, but hide pending runs | Mirrors CampaignRunTableView's exclude(); puts the rule in two places | ✓ |
| Everyone, no gate | Real disclosure regression | |

**User's choice:** Everyone, but hide pending runs.

| Option | Description | Selected |
|--------|-------------|----------|
| Read-only inlines | One attribution path; audit trail can't have holes | |
| Editable inlines | The only way Phase 27 can create a link at all; needs save_formset wiring or audit fields are null | ✓ |
| Read-only, but allow delete | Interim undo without a create path | |

**User's choice:** Editable inlines.
**Notes:** Accepted with the explicit condition, raised at the time, that
`CampaignRunAdmin.save_formset()` populates `confirmed_by`/`confirmed_at` — otherwise the audit
trail has holes on precisely the rows this phase creates. Recorded as a required task (D-07), not
an implementation detail.

| Option | Description | Selected |
|--------|-------------|----------|
| One model property | CampaignRun.is_publicly_visible, read by the template; view keeps its exclude() since a property can't be used in a filter | ✓ |
| Template conditional only | Duplicates the pending_review literal where nothing catches drift | |
| Pass it from the view | Would require overriding the upstream view the template approach avoided | |

**User's choice:** One model property.

---

## telescope_class values

The area where the most premises turned out wrong. Two separate corrections landed mid-area.

**Grounding measurement taken first:** 10 site-less rows in the live dev DB — 7 apparent
space-mission, 2 class-wide, 1 other (pk=31) — with `site_needs_review=1` on all but pk=31.

| Option | Description | Selected |
|--------|-------------|----------|
| One field, five values + blank | 2M0/1M0/0M4/SPACE/UNRESOLVED | initially ✓, superseded |
| Classes only, blank means unresolved | Keeps the field name honest; unresolved becomes an absence | |
| Classes only, keep site_needs_review for unresolved | Reuses an existing field; splits the answer across two | |
| Classes plus SPACE | Classes, plus SPACE for space observatories with no MPC code | ✓ final |
| Classes plus UNRESOLVED | Asserts resolution failure rather than inferring it | |

**User's choice:** first "one field, five values + blank", then superseded by the correction below.

**Notes — the correction that reshaped this area.** The user rejected the "row has no site because
it's a space mission" framing: *"All space missions we care about have either a MPC Code (which can
resolve to an existing Observatory or used to create one) or resolve through a Horizons->MPC code
mapping dictionary."* Verified: that dictionary already exists at `campaign_utils.py:43-46` (quick
task `260726-fqb`), and `Observatory` already holds 274 (JWST), 289 (Roman), C51 (WISE). So
26-DECISION.md's three-meaning vocabulary rested on a false premise, and the "9 permanently stuck
rows" claim made during this discussion was also wrong — those rows are *stale*, not stuck.

The user then narrowed `SPACE` precisely: *"pk=13 (Swift) has MPC Code C52 but JUICE has no MPC Code
(yet?) but it does have a Horizons code '500@-28'. So possibly Classes plus SPACE is needed for the
rare space observatory cases which don't have a MPC code."* That is the definition recorded in D-11.

| Option | Description | Selected |
|--------|-------------|----------|
| Measure first, then decide | Construct the actual colliding pair against the real constraints, to Phase 26's executable-proof standard | ✓ |
| Accept the lock, narrow the criterion | Record a reading that realistic pairs differ in telescope_instrument | |
| Add telescope_class to the constraint | Satisfies criterion 2 literally; reopens a Phase 26 lock | |

**User's choice:** Measure first, then decide.
**Notes:** Raised because ROADMAP criterion 2 requires class-wide and unresolved runs to coexist for
the same campaign/telescope/window, while 26-DECISION.md locked `telescope_class` out of both unique
constraints and the spike proved such rows collide.

| Option | Description | Selected |
|--------|-------------|----------|
| Derived rule in a data migration | Reproducible against other databases; pattern-matching risk | ✓ |
| Hand-enumerated pk list | Exact and auditable; wrong for any other database | |
| Leave blank, let staff set it | No guesswork; field unavailable for every existing row | |

**User's choice:** Derived rule in a data migration.

| Option | Description | Selected |
|--------|-------------|----------|
| Keep it as the unresolved test case | Row stays; Phase 26's recorded counts stay accurate | ✓ |
| Delete it, note the count change | Backfill has no ambiguous rows; Phase 26's counts describe a prior state | |
| Delete it later, outside this phase | Todo for dev-database tidying | |

**User's choice:** Keep it.
**Notes:** The user asked whether deleting pk=31 would change the options. Answered that it would
remove the only ambiguous backfill case but that the derived rule must handle "site NULL with no
class or space signal" correctly regardless, since that state arises in any other database — so
deleting is optional tidying, not a prerequisite. Separately corrected: pk=31 was *not* an anomaly.
`site_raw='X05'` is a valid Observatory (Rubin); the row is rejected, and resolution only runs at
approval, so it was never attempted rather than having failed.

| Option | Description | Selected |
|--------|-------------|----------|
| Out — capture as a todo | Phase 27 stays schema plus surfaces | |
| In — as its own task | Cheap; makes the backfill operate on correct data | ✓ |
| Investigate in research first | Decide with numbers | |

**User's choice:** In — as its own task.

| Option | Description | Selected |
|--------|-------------|----------|
| Offline only — tier 1 matches | Deterministic, testable; leaves HST unresolved | |
| Allow tier 2 — full resolve_site() | Fixes 5 rows; depends on the MPC API being up, not reproducible in CI | ✓ |
| Tier 2, but as a management command | Network call becomes a deliberate operator act | |

**User's choice:** Allow tier 2 — full resolve_site().

| Option | Description | Selected |
|--------|-------------|----------|
| No — operator data entry | Keeps Phase 27 from hand-writing domain values | |
| Yes — set it in the same task | Value supplied by domain authority, so not a guess | ✓ |
| Capture as a todo | Value preserved, applied later | |

**User's choice:** Yes — set `site_raw='C52'` for pk=13 (Swift).

---

## New fields' visible surface

| Option | Description | Selected |
|--------|-------------|----------|
| telescope_class yes, source no | telescope_class is observing info like site_raw; source is internal provenance | ✓ |
| Neither — staff only | Most literal reading of "non-staff visibility unchanged" | |
| Both visible | Leaks ingest-path detail; LEGACY is meaningless to outside readers | |

**User's choice:** telescope_class yes, source no.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — one shared helper | Migration and importer call the same function; prevents drift | ✓ |
| Yes, but duplicate the logic | Migrations stay self-contained; two copies of one rule | |
| No — migration only | Field starts rotting immediately after the backfill | |

**User's choice:** Yes — one shared helper.
**Notes:** Accepted with the guard, raised at the time, that the helper must take primitives
(`site_raw`, `telescope_instrument`) rather than a model instance, so the migration can import it
without coupling to a model that keeps changing through Phases 28–29.

| Option | Description | Selected |
|--------|-------------|----------|
| New module, do the extraction now | Closes the June todo; small stable import target for the migration | |
| campaign_utils.py for now | Smallest diff; leaves the todo open | |
| New module, but only move what's needed | Partial extraction | |

**User's choice:** *"This could maybe go in `campaign_utils.SITE_TELESCOPE_MAP` for the generic
telescope classes such as '2m0', '1m0', '0m4'?"*
**Notes:** Two corrections given. `SITE_TELESCOPE_MAP` is in `calendar_utils.py:37`, not
`campaign_utils.py`, and it is keyed `(site, aperture_class) → label` rather than being a class
vocabulary. The actual class vocabulary is one function below, in
`_aperture_class_from_telescope_code` (`calendar_utils.py:102`) — which happens to be one of the
five helpers the folded rename todo covers. The helper's home was recorded as `calendar_utils.py`
accordingly.

| Option | Description | Selected |
|--------|-------------|----------|
| Both in list_display and list_filter | Consistent with existing approval_status/run_status/campaign filters | ✓ |
| Display only, no filters | Keeps the filter sidebar small | |
| source read-only as well | Protects provenance from being rewritten | |

**User's choice:** Both in list_display and list_filter.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — include 4M0 | Matches the vocabulary calendar_utils already enforces | |
| No — three classes only | Matches CANON-02's wording exactly; no live row needs 4M0 | ✓ |
| Include it, and amend the requirement | Requirement and code agree explicitly | |

**User's choice:** No — three classes only.
**Notes:** Raised because `_aperture_class_from_telescope_code` names four classes including `4m0`,
and `SITE_TELESCOPE_MAP` has `('sor','4m0'): 'SOR-4m0'` for SOAR, while CANON-02 names only three.

| Option | Description | Selected |
|--------|-------------|----------|
| Model TextChoices, cross-checked by a test | Drift fails loudly; two definitions reconciled by a test | ✓ |
| TextChoices derived from calendar_utils | One definition, but couples models.py to calendar_utils and loses readable labels | |
| Plain TextChoices, no cross-check | Nothing catches drift | |

**User's choice:** Model TextChoices, cross-checked by a test.
**Notes:** This composed badly with the three-classes-only choice — a test asserting the two
vocabularies *agree* would fail on day one over `4m0`. Flagged and resolved: the test asserts a
**subset** relationship instead.

| Option | Description | Selected |
|--------|-------------|----------|
| Subset — as described | Catches renames and typos, permits the deliberate three-class scope | |
| Subset, plus an explicit 4m0 note | Same assertion; the test names 4m0 as known-excluded so a future reader doesn't "fix" it | ✓ |
| No test after all | Drops the conflict entirely | |

**User's choice:** Subset, plus an explicit 4m0 note.

---

## Claude's Discretion

- The observation-link model's name (`CampaignRunObservation` used as a placeholder throughout).
- Whether `is_publicly_visible` is reused to simplify any existing call site beyond the modal.
- Test organisation, beyond D-12's required subset assertion.

## Deferred Ideas

- A real staff run-detail page → Phase 28.
- Where dismissed attribution candidates persist → Phase 28 (a consequence of D-01).
- Audit fields on `CalendarEventMeta.run` → revisit if Phase 28's undo flow proves the gap painful.
- Adding `4M0` to `telescope_class` → revisit if a SOAR class-wide allocation arrives.
- Adding `telescope_class` to a unique constraint → only if D-14's measurement finds a real
  colliding pair.
- Extracting `SITE_TELESCOPE_MAP` / instrument extraction into their own module → folded in, but
  D-20 removed any dependency on it, so it can be dropped back out if it inflates the phase.
- v2.3 items untouched: ADAPT-01..03, GAPB-01, STATUS-01/02, UNUSED-01.
- The adopt-vs-gap-fill write strategy → Phase 29, per Phase 26's explicit deferral.
