# Phase 40: Notebook Isolation & Attribution Page - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-09
**Phase:** 40-notebook-isolation-attribution-page
**Areas discussed:** Scratch DB source, Guard strictness, Bootstrap 5 sweep scope, Undo return state

---

## Scratch DB source

**Q:** How should the three byte-copy notebooks build their scratch DB?

| Option | Selected |
|--------|----------|
| Backup snapshot |  |
| Fresh migrate, all three |  |
| Mixed per notebook | ✓ |

**User's choice:** Mixed per notebook

**Q:** What rule decides snapshot vs fresh?

| Option | Selected |
|--------|----------|
| Fresh unless output needs real rows | ✓ |
| Fixed list: only reconcile snapshots |  |

**User's choice:** Fresh unless committed output depends on rows the notebook did not create; researcher checks each; README states the rule

**Q:** Does project_observation_calendar_demo keep its mode=ro backup snapshot?

| Option | Selected |
|--------|----------|
| Keep its snapshot | ✓ |
| Convert it to fresh too |  |

**User's choice:** Keep its snapshot; SC1 reworded to 'no notebook byte-copies or writes the developer DB' in ROADMAP and REQUIREMENTS

**Q:** Snapshot notebook with no src/fomo_db.sqlite3?

| Option | Selected |
|--------|----------|
| Raise a clear error | ✓ |
| Fall back to fresh migrate |  |

**User's choice:** Raise a clear RuntimeError

**Q:** Shared setup code or per-notebook copy?

| Option | Selected |
|--------|----------|
| Copy per notebook |  |
| Shared helper module | ✓ |

**User's choice:** Shared helper module

**Q:** Where does the helper live?

| Option | Selected |
|--------|----------|
| New docs/notebooks/fomo_nb_utils.py |  |
| Bring skymap's version over now | ✓ |
| Separate file in pre_executed/ |  |

**User's choice:** docs/notebooks/fomo_nb_utils.py, starting from feature/skymap's version (special_locations included) plus the scratch-DB function; Django imports lazy; pre_executed setup cells add docs/notebooks to sys.path

**Notes:** The developer asked to check other branches for an earlier helper; `docs/notebooks/fomo_nb_utils.py` was found on unmerged `feature/skymap`. The developer then pointed out that a helper in `docs/notebooks/` can serve both docs-build and pre-executed notebooks (`pre_executed/` is a subdirectory of `notebooks/`), and chose to bring skymap's version over and extend it.

---

## Guard strictness

**Q:** What does the guard check now setup goes through the helper?

| Option | Selected |
|--------|----------|
| Helper call + real helper tests | ✓ |
| Helper call + keep the cell AST checks |  |

**User's choice:** Helper call + real helper tests: top-level helper call before any django/app import; notebook never calls django.setup() nor sets FOMO_DATABASE_PATH; no shutil copy of a DB; helper tests execute the real function (scratch routing, mode=ro backup, RuntimeError on a local_settings-style override). Replace source-pattern checks.

**Q:** Require execution counts exactly 1..N?

| Option | Selected |
|--------|----------|
| Yes, require 1..N | ✓ |
| No |  |

**User's choice:** Yes

**Q:** How is a notebook detected as Django-backed?

| Option | Selected |
|--------|----------|
| Any django/solsys_code/tom_* import | ✓ |
| Only notebooks calling the helper or django |  |

**User's choice:** Any import of django, solsys_code or tom_* packages

**Q:** Do the ~250 demonstration asserts change?

| Option | Selected |
|--------|----------|
| Leave them | ✓ |
| Convert them all to raises |  |

**User's choice:** Leave them; audit records each was reviewed and protects no write

**Q:** Keep committed-output checks (Resolved database line, teardown last cell)?

| Option | Selected |
|--------|----------|
| Keep both, via helper | ✓ |
| Drop the output checks |  |

**User's choice:** Keep both, provided by the helper (print + teardown function)

**Q:** Horizon dip unit (WARN-06)?

| Option | Selected |
|--------|----------|
| Degrees | ✓ |
| Arcminutes |  |

**User's choice:** Degrees

---

## Bootstrap 5 sweep scope

**Q:** How far does the BS4->5.3 tidy go?

| Option | Selected |
|--------|----------|
| All 8 templates + Python HTML | ✓ |
| Attribution page only |  |

**User's choice:** All 8 FOMO templates + the ml-1 in campaign_tables.py; new class guard scans every template

**Q:** Close IN-12 (scan all DIRS + app templates)?

| Option | Selected |
|--------|----------|
| Yes, scan all DIRS + app templates | ✓ |
| No, leave IN-12 open |  |

**User's choice:** Yes, for both data-attribute and class checks; mark IN-12 fixed

**Q:** Guard also scans Python-built HTML?

| Option | Selected |
|--------|----------|
| No, templates only | ✓ |
| Yes, scan .py sources too |  |

**User's choice:** No, templates only; fix ml-1 by hand

**Q:** How is the High-band marker proven?

| Option | Selected |
|--------|----------|
| Template test on classes |  |
| Template test + Playwright style check | ✓ |

**User's choice:** Template TestCase on classes + functional-tagged Playwright computed-style check (4px green left border)

---

## Undo return state

**Q:** Mechanism?

| Option | Selected |
|--------|----------|
| Hidden inputs + redirect whitelist | ✓ |
| HTMX in-place row update |  |

**User's choice:** Hidden inputs + redirect whitelist (validated keys)

**Q:** Which actions keep state?

| Option | Selected |
|--------|----------|
| All actions | ✓ |
| Undo only |  |

**User's choice:** All actions (confirm, dismiss, bulk confirm, both undos) through one redirect

**Q:** Scroll to section?

| Option | Selected |
|--------|----------|
| Yes, add a #section fragment | ✓ |
| No, top of page |  |

**User's choice:** Yes, #section fragment

**Q:** Undo from Confirmed lands where?

| Option | Selected |
|--------|----------|
| Confirmed section, same page | ✓ |
| Open Dismissed too |  |

**User's choice:** Confirmed section, same page; Dismissed keeps its prior state

**Q:** Undone row was last on its page?

| Option | Selected |
|--------|----------|
| Land on the last valid page | ✓ |
| Whatever django-tables2 does |  |

**User's choice:** Clamp to last valid page

---

## Claude's Discretion

- Helper function names/signature; exact Bootstrap 4 class regex; whether the WARN-06 cell adds a value check; plan split within the fixed order.

## Deferred Ideas

- Merging `feature/skymap` itself (not this phase).
- The other 19 pending todos go to Phase 41 triage.
