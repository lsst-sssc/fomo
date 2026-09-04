# Phase 33: Series Identity & Reconciler Inversion - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-03
**Phase:** 33-Series Identity & Reconciler Inversion
**Areas discussed:** Reconciler write paths, Series-identity fields, Decoration at display time, Unlink & orphan surfaces

---

## Todo cross-reference

Five pending todos matched at score 0.6 (keyword hits). Presented with the recommendation to fold none; none were selected. All five recorded as reviewed-not-folded in CONTEXT.md.

---

## Reconciler write paths

| Option | Description | Selected |
|--------|-------------|----------|
| Skip the night | Attributed non-`RUN:` event is the night; reconciler writes nothing for it (Phase 35 handoff rule) | ✓ |
| Mint `RUN:` night alongside | Reconciler always writes `RUN:{pk}:{date}`; duplicate entry until Phase 35 | |
| Drop attribution lookup entirely | Reconciler never reads `meta.run` during projection; same duplicate consequence | |

| Option | Description | Selected |
|--------|-------------|----------|
| Keep blocking | Foreign attribution outranks the automated writer (T-29-19) | ✓ |
| Write anyway, keep the link | Ownership purely by key; decoration shows the other run | |
| Write anyway, re-link to self | Overrides a staff attribution | |

| Option | Description | Selected |
|--------|-------------|----------|
| Leave in `RUN:` namespace | Past adopts are reconciler-owned by key; Phase 35 converts | ✓ |
| Restore blank url | Data migration un-keys them; no field distinguishes adopted from minted | |

| Option | Description | Selected |
|--------|-------------|----------|
| Unit test + notebook diff | Fixture test + real-DB before/after snapshot of every non-`RUN:` event | ✓ |
| Unit test only | | |
| Notebook diff only | | |

**User's choice:** all four recommended options.
**Notes:** Detach step confirmed already `RUN:`-scoped via `owned_events()`; no change needed there.

---

## Series-identity fields

| Option | Description | Selected |
|--------|-------------|----------|
| `OneToOneField`, nullable | DB-enforced one event per record (D1) | ✓ |
| Plain `ForeignKey`, nullable | Allows several events per record | |

| Option | Description | Selected |
|--------|-------------|----------|
| `SET_NULL` on both | Mirrors `run`; history survives | ✓ |
| `CASCADE` on both | Deletes the companion row | |
| `SET_NULL` record, `CASCADE` group | Inconsistent | |

| Option | Description | Selected |
|--------|-------------|----------|
| No backfill — Phase 34's sweep fills it | Schema + semantics only; one writer per source | ✓ |
| Data migration | Imports facility classes; throwaway second writer | |
| One-off helper from notebook | Still a second writer | |

| Option | Description | Selected |
|--------|-------------|----------|
| Read-only in admin | Written only by code | ✓ |
| Editable in admin | Hand-linked events could drift outside the projector's namespace | |
| Hidden from admin | Staff lose visibility | |

**User's choice:** all four recommended options.
**Notes:** Web research (Django docs / ticket #26044) confirmed `OneToOneField` ≡ `ForeignKey(unique=True)` and that nullable uniques migrate existing rows without collisions.

---

## Decoration at display time

| Option | Description | Selected |
|--------|-------------|----------|
| Cell marker + modal block | Compact chip/icon in the cell, existing modal block extended | ✓ |
| Modal only | Cell unchanged | |
| Text prefix in cell + modal | Eats the truncation budget | |

| Option | Description | Selected |
|--------|-------------|----------|
| Campaign name + run status | Cell: name; modal: name, telescope/instrument + window, status | ✓ |
| Campaign name only | | |
| Full run identity | Duplicates the campaign table row | |

| Option | Description | Selected |
|--------|-------------|----------|
| Stop writing the campaign name | `event_title()` drops the prefix; one decoration path; 74-event churn accepted | ✓ |
| Keep the prefix; decoration skips `RUN:` | Two mechanisms for two phases | |
| Keep the prefix; decorate everything | Campaign shown twice | |

| Option | Description | Selected |
|--------|-------------|----------|
| Campaign table, run row anchored | `campaigns:table` + `#run-{pk}` anchor, row highlighted | ✓ |
| Campaign table only | As today | |
| New run-detail view | Phase 37's surface — scope creep | |

**User's choice:** all four recommended options.

---

## Unlink & orphan surfaces

| Option | Description | Selected |
|--------|-------------|----------|
| `meta.run` is the single source | Phase 35 syncs it from `CampaignRunObservation`; queue/admin may set directly | ✓ |
| Derive from `CampaignRunObservation` at render | Two lookups, two possible answers | |
| `CampaignRunObservation` only for observation events | Event-level queue/admin become dead ends | |

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, keep offering them | Phase 28 surfaces unchanged | ✓ |
| No — attribute via the record instead | Exclude observation-backed events from the orphan queue | |

| Option | Description | Selected |
|--------|-------------|----------|
| One unlink helper, clears all three | `run` + `confirmed_by` + `confirmed_at`; used by undo, detach, admin | ✓ |
| Leave each surface as it is | Docs only | |

| Option | Description | Selected |
|--------|-------------|----------|
| Rename to "Attributed campaign run" everywhere | verbose_name, admin, modal, docstrings, runbook | ✓ |
| Docs and docstrings only | Admin still says "Owning" | |

**User's choice:** all four recommended options.

---

## Claude's Discretion

- Names/homes of the unlink helper and decoration template tag(s).
- Visual form of the cell marker.
- Fate of `update_calendar_event_key_and_fields()`.
- Whether `ReconcileResult` gains a `skipped_nights` counter and how `--dry-run` reports it.
- `related_name`s, `__str__`, and one-vs-two migrations.

## Deferred Ideas

- Run-detail view (offered as a link target; declined as Phase 37's surface).
