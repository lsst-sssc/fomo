# Phase 30: v2.2 Tech-Debt Cleanup - Pattern Map

**Mapped:** 2026-08-31
**Files analyzed:** 12 (code) + 5 planning/bookkeeping files (no code-pattern analog needed)
**Analogs found:** 9 / 9 code files with a meaningful analog

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|-----------------|---------------|
| `solsys_code/campaign_attribution.py` (`_eligible_runs_for_event`, `_eligible_runs_for_record`) | service (query gate) | CRUD (queryset filter) | same file, sibling function `orphan_calendar_events()` / `orphan_observation_records()` and the two functions themselves (self-analog, extend in place) | exact |
| `solsys_code/management/commands/import_campaign_csv.py` (`telescope_class` guard) | service (management command, batch) | batch / CRUD | same file, `preserve_site` guard (`:336-369`) | exact |
| `solsys_code/campaign_reconciler.py` (docstring cleanup) | service | batch | same file (in-place doc edit, no external analog needed) | n/a (doc-only) |
| `pyproject.toml:42` | config | n/a | `.pre-commit-config.yaml:48-62` (the pin being matched) | exact |
| `CLAUDE.md` §Commands | config/doc | n/a | `.pre-commit-config.yaml` ruff hook id (`pre-commit run ruff --all-files`) | exact |
| `docs/runbooks/telescope_runs_calendar.rst` (attribution section) | doc | n/a | same file, existing "Sites Needing Review" / "source-lock" sections' prose style | exact |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (new cell) | test/demo | request-response | same notebook, cells 19-22 (orphan event creation + attribution confirm) | exact |
| `solsys_code/tests/test_campaign_attribution.py` (new D-01/D-02 tests) | test | CRUD | same file, `test_cross_campaign_run_never_offered_for_event_even_at_perfect_score` / `_for_record` (`:257-265`) plus the dismissal test class (`:420`) | exact |
| `solsys_code/management/commands/tests/*` (CSV guard tests, if present) | test | batch | `test_admin.py` `SourceProvenanceLockTests` / `SourceProvenanceTwoStepBypassTests` (`:837`, `:998`) as the "pin an invariant with a non-vacuous control test" shape | role-match |

## Pattern Assignments

### `solsys_code/campaign_attribution.py` — `_eligible_runs_for_event` / `_eligible_runs_for_record` (service, CRUD gate)

**Analog:** itself (extend the existing functions in place) — read `campaign_attribution.py:435-509` for full context.

**Current code to extend** (`:474-509`):
```python
def _eligible_runs_for_event(event: CalendarEvent):
    """D-11/ROADMAP criterion 3's hard gate for a CalendarEvent orphan: only runs in the SAME
    campaign (TargetList) as the event. An event with no ``target_list`` at all (e.g. a
    conference or proposal-deadline entry -- D-03's noise filter) is eligible for nothing.

    Args:
        event: the orphan CalendarEvent.

    Returns:
        QuerySet[CampaignRun]: runs eligible to be scored at all for this event.
    """
    if event.target_list_id is None:
        return CampaignRun.objects.none()
    return CampaignRun.objects.filter(campaign_id=event.target_list_id)


def _eligible_runs_for_record(record: ObservationRecord):
    """D-11/ROADMAP criterion 3's hard gate for an ObservationRecord orphan: only runs whose
    campaign (TargetList) the record's target belongs to.

    Deliberately compares the CAMPAIGN only -- this must NOT additionally require the run's
    target FK to equal the record's target FK. [...]

    Args:
        record: the orphan ObservationRecord.

    Returns:
        QuerySet[CampaignRun]: runs eligible to be scored at all for this record.
    """
    if record.target_id is None:
        return CampaignRun.objects.none()
    return CampaignRun.objects.filter(campaign__in=record.target.targetlist_set.all())
```

**Docstring convention to follow** (findings cited by ID, e.g. `campaign_attribution.py`'s existing citations of `28-REVIEW.md IN-01`, `D-11`, `D-03`): the new filter must cite **`27-REVIEW IN-02`** and **`D-01`/`D-02`/`D-03`** by name, and must explicitly say this is a *hard* gate — a deliberate departure from the "eligibility gates are deliberately permissive" convention documented in both existing docstrings (see `candidates_for_event`'s docstring at `:512-529` for how that permissiveness is normally phrased, so the new exception reads as an intentional contrast rather than an oversight).

**Shape to add** (illustrative — plan should keep the existing early-return-on-None-FK shape, then add one `.exclude()`):
```python
    return CampaignRun.objects.filter(campaign_id=event.target_list_id).exclude(
        approval_status=CampaignRun.ApprovalStatus.REJECTED
    )
```
Apply the identical `.exclude(...)` to `_eligible_runs_for_record`'s return statement. Per CONTEXT.md Claude's Discretion, either `.exclude()` inline or a shared module-level constant naming the vocabulary is acceptable — no existing convention in this file favors one over the other, so follow whichever the plan already leans toward for readability.

**Vocabulary source — `ApprovalStatus`** (`solsys_code/models.py:89-93`):
```python
    class ApprovalStatus(models.TextChoices):
        """Admin review state for a CampaignRun (independent of real-world run outcome)."""

        PENDING_REVIEW = 'pending_review', 'Pending Review'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'
```
Reference as `CampaignRun.ApprovalStatus.REJECTED`, matching the enum-reference convention used throughout `campaign_reconciler.py` (e.g. `CampaignRun.RunStatus.CANCELLED`) and `campaign_attribution.py`.

---

### `solsys_code/management/commands/import_campaign_csv.py` — `telescope_class` re-import guard (service, batch)

**Analog:** `preserve_site` guard in the same file (`:336-369`), which CONTEXT.md names as the exact shape to mirror.

**Full excerpt to copy the shape of** (`:336-369`):
```python
            # WR-01/CANON-01: never relabel a run that came in through the public web form.
            # insert_or_create_campaign_run() setattr's every key in `fields` onto a matched
            # row, so without this guard a CSV row colliding on the natural key with a
            # WEB-sourced submission (entirely plausible -- the sheet and the form describe
            # the same runs) would rewrite it to source=CSV_IMPORT, approval_status=APPROVED.
            # [...]
            if existing is not None and existing.source == CampaignRun.Source.WEB:
                fields.pop('source', None)
                fields.pop('approval_status', None)

            # WR-01 (criterion 5): a re-import must not silently revert a site that
            # repair_stale_campaign_run_sites already fixed. When the existing row already
            # carries a resolved site (existing.site_id is not None) and the CSV's own Site
            # Code cell did NOT genuinely resolve this time [...] drop site, site_raw and
            # site_needs_review from fields as a unit. [...] The condition itself is computed
            # further up as `preserve_site`, because the telescope_class derivation has to
            # gate on the same decision (CR-01).
            if preserve_site:
                fields.pop('site', None)
                fields.pop('site_raw', None)
                fields.pop('site_needs_review', None)
                # WR-01: say so. [...]
                site_preserved_count += 1
                self.stderr.write(
                    f'Row {row_num}: kept existing resolved site '
                    f'{existing.site.obscode!r} (Site Code={site_raw!r} did not resolve); '
                    f'CSV site/site_raw discarded'
                )
```

**Existing D-04-relevant code already in place** (the blanking half, `:371-378`):
```python
            # telescope_class is NEVER cleared by any writer once set
            # (solsys_code/models.py:207-219) -- Phase 27 code-review finding CR-01 proposed
            # clearing it here on site resolution and the user REJECTED CR-01
            # (27-REVIEW-FIX.md). Without this pop a re-import whose Site Code cell resolves
            # would write telescope_class='' over a non-blank value (see the
            # `telescope_class = ... if site is None else ''` computation above).
            if existing is not None and existing.telescope_class and not telescope_class:
                fields.pop('telescope_class', None)
```

**What D-04 adds:** this existing block only stops *blanking*. D-04 needs a second condition, mirroring `preserve_site`'s shape exactly: compute a `preserve_telescope_class` boolean up front (parallel to how `preserve_site` is computed — check the code above `:336` for that computation, likely near where `site`/`telescope_class` are first derived from the row), pop the field(s) as a unit if the CSV's own derived value did not genuinely resolve to something *different and non-empty*, increment a counter (`telescope_class_preserved_count`, parallel to `site_preserved_count`), and emit a `self.stderr.write(...)` line naming what was kept and what was discarded — copy the exact phrasing pattern `f'Row {row_num}: kept existing ... ; CSV ... discarded'`.

**WR-04 summary-count interaction** (`:378-` onward) — read this too, since the guard's pop interacts with the "how many rows END UP flagged" summary logic already present for `site_needs_review`; if `telescope_class` has any similar downstream summary counter, the same care applies.

---

### `solsys_code/campaign_reconciler.py` — docstring cleanup (doc-only, no code pattern)

**Analog:** none needed — this is D-10 cosmetic text editing, not a code pattern.

**Exact lines to edit** (verified via grep):
- `:78` — `` `_project_calendar_event()` ``'s bool return`` → replace stale name reference with `_resolve_site()` or the current successor concept, consistent with the phrasing already used at `:81` (`_resolve_site() (plan 29-04) uses it...`).
- `:100` — "a deliberate divergence from the ported `_project_calendar_event()` code"
- `:173` — `` `campaign_views._calendar_event_title()` ``'s cancelled/weathered output`` — replace with the current name of that function if renamed, or drop the stale reference entirely if the ported function no longer exists under any name.
- `:195` — "Preserves today's exact 'no event yet' cases from `_project_calendar_event()`, plus..."
- `:339` — "Ports `_project_calendar_event()`'s ground loop: iterates every night in..."

These are prose-only edits inside existing docstrings; no code behavior changes. Confirm via `grep -n "_project_calendar_event\|_calendar_event_title" solsys_code/*.py` after editing that no stray references remain outside intentional historical mentions (if any function legitimately still needs to reference the ported origin for provenance, keep exactly one clear mention rather than five scattered ones — CONTEXT.md doesn't specify which, so use judgment based on what each docstring is trying to convey).

---

### `pyproject.toml:42` (config)

**Analog:** `.pre-commit-config.yaml:48-62` — the source of truth for the pin.

**Current line to change:**
```toml
    "ruff", # Used for static linting of files
```
**Target line** (verify exact rev string in `.pre-commit-config.yaml` before writing):
```bash
grep -n "rev:" .pre-commit-config.yaml
```
Change to `"ruff==0.2.1",` (or `>=0.2.1,<0.2.2` style if the project's other pins in `pyproject.toml` use ranges — check neighboring dev-dependency lines at `pyproject.toml:38-44` for the prevailing pin style before choosing exact syntax).

---

### `CLAUDE.md` §Commands (doc)

**Current lines to change** (`CLAUDE.md:33-34`):
```bash
ruff check . --fix
ruff format .
```
**Target:** replace with pre-commit-mediated invocation, e.g.:
```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```
Verify the exact hook `id`s in `.pre-commit-config.yaml` (`grep -n "id:" .pre-commit-config.yaml`) before writing the replacement, since `pre-commit run <id>` requires an exact match.

Also update the other CLAUDE.md line at `:184` ("`ruff check .` and `ruff format --check .` must stay clean") if it exists in the Stage-1 project-specific block — check whether it should stay as a description of the *outcome* (still accurate) versus the *invocation* (D-07's target).

---

### `docs/runbooks/telescope_runs_calendar.rst` (doc, D-12)

**Analog:** same file — extend the attribution section's existing prose style (`:187-208`).

**Section to extend** (excerpt, `:187-209`):
```rst
How do I attribute existing calendar events and observation records to a run?
--------------------------------------------------------------------------------

The attribution page (``campaigns:attribution``, at ``/campaigns/attribution/``)
is where staff connect a calendar event or an observation record that
already exists to the ``CampaignRun`` that actually produced it. [...]

**The two worklists, and why an orphan may be absent.** The page lists
"Calendar events awaiting attribution" and "Observation records awaiting
attribution" as two sibling tables. Only an event or record with *at
least one* candidate run appears in either one -- the same campaign/target
boundary check that keeps a suggestion from ever crossing into the wrong
campaign also filters out the noise. [...] **The queue shows attributable orphans, not every
un-attributed row** -- an empty worklist does not mean nothing is
un-attributed, only that nothing un-attributed has a run to offer it to.
```
Match this bold-lead-in, plain-English paragraph style for the new sentence about rejected runs never being offered (D-12): add a new bold-lead-in paragraph (or extend "The two worklists..." paragraph) stating a run with `approval_status=REJECTED` is never offered as a candidate, alongside the existing campaign/target boundary explanation. Do not touch `:292-357` (source-lock section) — CONTEXT.md explicitly marks it "already correct — do not re-edit."

---

### `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — fifth run, rejected (D-12)

**Analog:** cells 19-22 (orphan event creation + attribution confirm), same notebook.

**Cell 19-20 shape to mirror for creating the orphan/demo entity** (markdown then code):
```
## An orphan calendar event, and the attribution queue

A `load_telescope_runs`-style entry that predates the canonical run record is represented
here by creating one `CalendarEvent` directly -- a hand-entered/legacy-sync-created row
with NO `CalendarEventMeta` companion row at all [...]
```
```python
from datetime import datetime
from datetime import timezone as dt_timezone
from tom_calendar.models import CalendarEvent
from solsys_code import campaign_attribution

classical_telescope, _sep, classical_instrument = classical_run.telescope_instrument.partition('/')

orphan_event = CalendarEvent.objects.create(
    title=f'{campaign.name}: hand-entered pre-canon night (demo)',
    description='A pre-existing, hand-entered calendar entry that predates the canonical CampaignRun record.',
    start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
    end_time=datetime(2026, 9, 2, 10, 0, ...),
    ...
)
```

**Cell 21-22 shape to mirror for the assertion/demonstration cell:**
```python
response = staff_client.post(
    reverse('campaigns:attribution_decide'),
    {'action': 'confirm', 'kind': 'event', 'orphan_pk': orphan_event.pk, 'run_pk': classical_run.pk},
)
assert response.status_code == 302
...
assert meta.run_id == classical_run.pk
```

**New cell to add:** a fifth `CampaignRun` (public submission via the same `staff_client`/submission flow the other four use — check earlier cells in the notebook for the "submit" POST shape, e.g. via `campaigns:submit` or `CampaignRunDecisionView`), then `staff_client.post(reverse('campaigns:decide', args=[fifth_run.pk]), {'action': 'reject'})` (mirror the `resolve_site` POST shape at cell 18: `{'action': 'resolve_site', 'site_selection': 'Y23'}` → `{'action': 'reject'}`), then call `campaign_attribution.candidates_for_event(orphan_event)` (or the record equivalent) and assert the fifth run's pk is NOT in the candidate set — mirroring the exact assertion shape from `test_cross_campaign_run_never_offered_for_event_even_at_perfect_score` (see below). Label the run's title/description visibly as "(demo — rejected, for attribution exclusion)" per CONTEXT.md's Specific Ideas note, so it's not mistaken as part of the four-run lifecycle narrative.

Regenerate via `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` and commit with output present (this directory is the one exception to the repo's "pre-commit clears notebook output" convention).

---

### `solsys_code/tests/test_campaign_attribution.py` — new D-01/D-02 tests (test, CRUD)

**Analog:** existing candidate-pk assertions in the same file.

**Pattern at `:255-266`** (`test_cross_campaign_run_never_offered_for_event_even_at_perfect_score`):
```python
    def test_cross_campaign_run_never_offered_for_event_even_at_perfect_score(self):
        candidate_run_pks = {c.run.pk for c in candidates_for_event(self.event_a)}
        self.assertNotIn(self.run_b.pk, candidate_run_pks)

    def test_cross_campaign_run_never_offered_for_record_even_at_perfect_score(self):
        candidate_run_pks = {c.run.pk for c in candidates_for_record(self.record_a)}
        self.assertNotIn(self.run_b.pk, candidate_run_pks)
```
The D-01/D-02 tests should follow this exact shape: create a `REJECTED` `CampaignRun` fixture in the same campaign/window as an eligible event or record (so it would otherwise score highly), call `candidates_for_event`/`candidates_for_record` (or `_eligible_runs_for_event`/`_eligible_runs_for_record` directly, if testing the gate in isolation is preferred), and `assertNotIn(rejected_run.pk, candidate_run_pks)`. Per `SourceProvenanceLockTests`' "non-vacuous control test" convention (see below), pair each exclusion test with a control asserting an `APPROVED` or `PENDING_REVIEW` run in the identical position IS still offered (D-01 explicitly keeps both eligible) — this is the "pin an invariant with a non-vacuous control" pattern.

**Dismissal-adjacent test class shape at `:420`** (class-level `setUpTestData` building event + runs, then per-test dismissal/assertion) is the natural home if the new tests are added as their own `TestCase` subclass rather than appended to an existing one — check the class this line belongs to (`CandidateDismissalTests` or similar) for the `setUpTestData` fixture-building convention (uses `NonSiderealTargetFactory` per CLAUDE.md — confirm any new `Target` fixtures in these tests do the same).

**Non-vacuous control test convention** (`solsys_code/tests/test_admin.py:837-843`):
```python
class SourceProvenanceLockTests(TestCase):
    """27.1-05 (closing criterion 6, WR-03): `source` is non-overwritable on every
    `source == WEB` run, at any approval status, and stays editable on every non-WEB row
    of every approval status (D-19 preserved)."""
```
The class docstring names both the locked case and the control case explicitly ("non-overwritable on X... editable on Y") — new D-01/D-02 test classes/docstrings should do the same: name both what's excluded (REJECTED) and what stays included (APPROVED, PENDING_REVIEW).

---

## Shared Patterns

### Docstring finding-ID citation convention
**Source:** `solsys_code/campaign_attribution.py` (cites `28-REVIEW.md IN-01`, `D-11`, `D-03`), `solsys_code/admin.py` (cites `WR-10`), `import_campaign_csv.py` (cites `WR-01`, `CR-01`, `WR-04`)
**Apply to:** every code change in this phase — new/edited docstrings and inline comments must cite `27-REVIEW IN-02` and the relevant `D-0N` decision ID from `30-CONTEXT.md`, matching the existing convention of citing findings by ID rather than describing them anew.

### "Say so" per-row stderr reporting
**Source:** `import_campaign_csv.py:365-369` (the `preserve_site` guard's `self.stderr.write(...)` call)
**Apply to:** the new `telescope_class` guard (D-04) — every guard that silently preserves a value must also report it, using the same `f'Row {row_num}: kept existing ... ; CSV ... discarded'` phrasing shape.

### Enum-reference convention
**Source:** `solsys_code/models.py:89-93` (`CampaignRun.ApprovalStatus`), used throughout `campaign_reconciler.py` and `campaign_attribution.py` as `CampaignRun.ApprovalStatus.X` / `CampaignRun.RunStatus.X`
**Apply to:** the new `.exclude(approval_status=CampaignRun.ApprovalStatus.REJECTED)` filters — always reference via the enum class, never a bare string `'rejected'`.

### Test fixture convention (project-wide, CLAUDE.md)
**Source:** `solsys_code/tests/test_campaign_attribution.py:242` (`NonSiderealTargetFactory.create()`)
**Apply to:** any new `Target` fixture created for D-01/D-02 tests or the notebook's fifth run — must use `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory`.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `.planning/v2.2-MILESTONE-AUDIT.md`, five phase `VALIDATION.md` files, `26-DECISION.md` header | planning bookkeeping | n/a | Pure prose edits to `.planning/` artifacts; no code pattern applies. Follow each file's own existing structure/section headers when editing (D-08, D-09, D-10 portion of `26-DECISION.md`). |

## Metadata

**Analog search scope:** `solsys_code/` (campaign_attribution.py, campaign_reconciler.py, models.py, management/commands/import_campaign_csv.py, tests/test_campaign_attribution.py, tests/test_admin.py), `docs/runbooks/`, `docs/notebooks/pre_executed/`, `pyproject.toml`, `CLAUDE.md`, `.pre-commit-config.yaml`
**Files scanned:** 9 code/doc files read directly, plus grep sweeps for stale-name references and ruff config lines
**Pattern extraction date:** 2026-08-31
</content>
