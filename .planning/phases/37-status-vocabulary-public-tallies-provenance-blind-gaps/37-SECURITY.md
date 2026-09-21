---
phase: "37"
slug: "status-vocabulary-public-tallies-provenance-blind-gaps"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-09-21"
---

# Phase 37 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

Register origin: authored at plan time. All ten plans (37-01 .. 37-10) carried a
`<threat_model>` block, so this audit verifies the mitigations named there rather than
building a register retroactively. ASVS level 1 (grep-depth verification).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| LCO Observation Portal -> `calendar_utils.resolve_placement_block()` | Outbound credentialed request; untrusted JSON response body. | API token (outbound), portal response body (inbound) |
| LCO Observation Portal -> `proposal_allocation.fetch_proposal_allocations()` | Outbound credentialed request for proposal time allocations. | API token (outbound), proposal figures (inbound) |
| Anonymous visitor -> `campaigns:list` / `campaigns:table` / gap page | All three require no login; every rendered value is world-readable. | Campaign tallies, run rows, gap dates |
| Client query string -> gap page and runs table | `target`, `site`, `sort`, `page`, `per_page` and filter fields arrive untrusted. | Attacker-controlled parameters |
| `CampaignRunTableView.get_queryset()` -> rendered row | The PII boundary: only `ALLOWED_FIELDS_FOR_NON_STAFF` plus the opt-in-gated contact annotation may cross. | `contact_person` / `contact_email` |
| `CalendarEventMeta.run` link -> rendered calendar decoration | Decides whether a campaign fact is shown to an anonymous reader at all. | Pending-review run existence |
| Django admin -> `ProposalTimeAllocation` | A staff user could otherwise hand-edit a runner-owned, portal-sourced figure. | Public allocation figures |
| Developer database -> committed notebook output | Pre-executed notebook output is published in the repository. | Contact details, credentials, `RequestGroup` ids |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-37-01 | Information Disclosure | `calendar_utils.resolve_placement_block()` | high | mitigate | `except (RequestException, ImproperCredentialsException, ValidationError, ValueError):` binds no name and returns `None` — the exception is never referenced, stringified or logged (SYNC-09/D-11). Verified at `calendar_utils.py:299,313-314`. | closed |
| T-37-02 | Information Disclosure | `status_vocabulary.LEGEND` | low | accept | Legend is a fixed hand-listed vocabulary with no database read (D-04); cannot leak a row's existence. | closed |
| T-37-03 | Tampering | `status_vocabulary.state_for_title()` | low | mitigate | Trailing-space marker rule enforced and documented at `status_vocabulary.py:167-181`; a user-authored title cannot claim an unearned status ring. | closed |
| T-37-04 | Information Disclosure | `proposal_allocation.fetch_proposal_allocations()` | high | mitigate | `raise PortalUnavailable(type(exc).__name__) from None` (`proposal_allocation.py:143`) — only the class name crosses; chaining suppressed so no response body or token reaches a traceback. | closed |
| T-37-05 | Information Disclosure | `unattended.step_proposal_allocation()` | high | mitigate | Summary built from three integer counters plus the first exception's class name only (`unattended.py:443-446`). | closed |
| T-37-06 | Information Disclosure | `proposal_allocation.proposal_codes_to_fetch()` | medium | mitigate | `.values_list(..., flat=True).distinct()` on both sides; no `CampaignRun` row and no contact field enters the runner process (`proposal_allocation.py:83-89`). | closed |
| T-37-07 | Tampering | `ProposalTimeAllocationAdmin` | low | mitigate | Every field read-only, plus `has_add_permission()`/`has_delete_permission()` closed after WR-10 (`admin.py:485-495`). | closed |
| T-37-08 | Repudiation | stored allocation rows | low | accept | `fetched_at` records the refresh time; no per-user audit trail, accepted because only the runner writes. | closed |
| T-37-09 | Information Disclosure | `campaign_gap.observation_claimed_dates()` | high | mitigate | Site obscode read as a single annotated scalar (`F('calendar_event_meta__run__site__obscode')`, `campaign_gap.py:214`) with `.only(...)` at :215; `select_related` appears nowhere in `campaign_gap.py`. | closed |
| T-37-10 | Information Disclosure | claimed-kind label on the gap page | medium | mitigate | Page renders dates and counts only — no record id, proposal code or group name. | closed |
| T-37-11 | Tampering / Information Disclosure | gap-page query parameters | high | mitigate | No new parameter added; `_as_pk_or_none()` plus the server-side allowed-site re-derivation preserved (`campaign_views.py:1062-1080`). | closed |
| T-37-12 | Denial of Service | second claim query per gap computation | low | accept | One extra bounded query behind the existing 1-hour result cache. | closed |
| T-37-13 | Information Disclosure | `campaign_tally.link_counts_for_runs()` / `night_counts_for_run()` | high | mitigate | Every query pk-keyed and field-restricted (`campaign_tally.py:141,156,195-196,400`); the one `select_related('site')` is paired with an explicit `.only()` naming non-PII columns (:544-545). No `CampaignRun` row fetched wholesale. | closed |
| T-37-14 | Information Disclosure | `campaign_tally.campaign_rollup()` | high | mitigate | Pending-review excluded at queryset level (`campaign_tally.py:507,543`) before any count is taken. | closed |
| T-37-15 | Information Disclosure | cached tally dicts | medium | mitigate | Cached values hold integers and booleans only — no name, proposal code, record id or group name. | closed |
| T-37-16 | Tampering | `CampaignRun.run_status` | high | mitigate | TALLY-03 two-part guard: behavioural before/after snapshots plus `TestTallyNeverWritesRunStatus`'s AST walk (`test_campaign_tally.py:1241-1293`). No `run_status =` assignment exists in the computation path. | closed |
| T-37-17 | Information Disclosure | `CampaignRunTableView.get_queryset()` | high | mitigate | Unmodified; tally joins by pk in Python. `.values()`-before-`.annotate()` ordering and `ALLOWED_FIELDS_FOR_NON_STAFF` intact (`campaign_views.py:93,242,309`). | closed |
| T-37-18 | Information Disclosure | `campaign_rollup()` on public pages | high | mitigate | Roll-up inherits T-37-14's queryset-level exclusion; a visitor cannot deduce a hidden run by comparing strip to rows. | closed |
| T-37-19 | Denial of Service | tally computation per table load | medium | mitigate | SQL-expressible counts are two queries per page; night counts behind the shared TTL cache; `assertNumQueries` pins that rows do not add queries. | closed |
| T-37-20 | Spoofing | rendered marker letters | low | accept | Markers rendered via `format_html` from the shared table; a run's free text cannot inject one. | closed |
| T-37-21 | Information Disclosure | `run_tally()` template tag | high | mitigate | `run is None or not run.is_publicly_visible` gate (`calendar_display_extras.py:568-577`); a pending-review run's tally never reaches a non-staff visitor. | closed |
| T-37-22 | Information Disclosure | pop-up tally line | medium | mitigate | Counts only — no group name, record id or proposal code; the internal `RequestGroup` identifier stays withheld. | closed |
| T-37-23 | Information Disclosure | `unused_night_decoration()` | medium | mitigate | Same guarded companion-row lookup and visibility gate (`calendar_display_extras.py:628-641`); emits a fixed token and fixed label only. | closed |
| T-37-24 | Tampering | rendered `[U]` token | low | mitigate | Token prepended by the template from a constant (`calendar.html:182,259,295`), never written into `CalendarEvent.title`; a test asserts the stored title is unchanged after render. | closed |
| T-37-25 | Information Disclosure | regenerated notebook output | high | mitigate | Committed notebook output contains only demo-scoped fixtures (`@example.org` / `@example.com`); no real contact name, API key or `RequestGroup` identifier present. | closed |
| T-37-26 | Information Disclosure | runbook prose | medium | mitigate | `docs/runbooks/telescope_runs_calendar.rst` names the setting key holding the portal credential, never a value (:1491,1551,1555,1656,1662). | closed |
| T-37-27 | Denial of Service | deleting the retirement list before a deployment is swept | medium | mitigate | Task precondition halts on any surviving legacy title; runbook notes the re-title lands within one unattended tick. | closed |
| T-37-40 | Denial of Service | live unused split on every anonymous `campaigns:list` GET | medium | mitigate | Constant-marginal-cost assertion plus relaxed bound test; `CampaignListView.paginate_by = 100` bounds the fan-out; added queries are indexed single-table lookups. | closed |
| T-37-41 | Information Disclosure | roll-up dict on the cache-hit path | low | mitigate | Applier writes only the three existing `unused_*` keys; no new key, model field or change to `ALLOWED_FIELDS_FOR_NON_STAFF`. | closed |
| T-37-42 | Tampering | double allocation lookup collapsed into one source | low | mitigate | `estimated_unused_nights` has exactly two call sites in `campaign_tally.py` (:423, :621 — the other two occurrences are docstring prose), pinned by the AST gate and the in-code trade comment. | closed |
| T-37-43 | Repudiation | staff cancellation appearing not to take effect | medium | mitigate | Edit is visible on the next load on every surface after the fix; the runbook states the rule so an operator can tell a stale cache from an unsaved decision. | closed |
| T-37-44 | Tampering | `CampaignRun.run_status` (37-08 path) | high | mitigate | New applier reads `run_status`/`proposal_code`, writes neither; covered by the same AST walk as T-37-16. | closed |
| T-37-09-01 | Information Disclosure | `get_table()`'s `CampaignRun.objects.filter(pk__in=...)` | medium | mitigate | Model queryset feeds `tallies_for_runs()` only; result is a counts dict assigned to `table.tallies`, never merged into `table.data`. `get_queryset()` untouched. | closed |
| T-37-09-02 | Denial of Service | unbounded `per_page` query parameter | high | mitigate | `MAX_TABLE_PER_PAGE = 100` (`campaign_views.py:132`) normalises `per_page` on a mutable copy of the query parameters before `RequestConfig.configure()` reads it; too-large clamps down to the cap, degenerate values fall back to `DEFAULT_TABLE_PER_PAGE` (CR-01). | closed |
| T-37-09-03 | Tampering | `CampaignRun.run_status` (37-09 path) | high | mitigate | Read-only aggregation; covered by the same AST walk as T-37-16. | closed |
| T-37-09-04 | Spoofing | `sort` / `page` query parameters | low | accept | django-tables2 resolves `sort` against declared bound columns only and clamps out-of-range pages; clamped-page cases are tested rather than assumed. | closed |
| T-37-10-01 | Information Disclosure | public `unused_unknown_runs` count | medium | mitigate | Derived inside `_apply_rollup_unused_fields()` from `_rollup_runs()`'s already-excluded run set (`campaign_tally.py:591,645,659`); a pending-review run cannot raise it. | closed |
| T-37-10-02 | Tampering | `CampaignRun.run_status` (37-10 path) | high | mitigate | Widened applier reads and writes nothing; covered by the same AST walk as T-37-16. | closed |
| T-37-10-03 | Repudiation | a displayed total that cannot say what it is missing | medium | mitigate | `unused_unknown_runs` plus the `at least ...` rendering and the accounting invariant make the total self-accounting. | closed |
| T-37-10-04 | Denial of Service | widened applier's per-run and per-code loops | low | accept | Derived inside loops that already run — no new queryset, pass or query; existing exact-count gates hold unchanged. | closed |
| T-37-SC | Tampering | npm/pip/cargo installs (plans 37-01 .. 37-08) | high | mitigate | No new dependency: `pyproject.toml` was not touched during Phase 37, and RESEARCH.md's Package Legitimacy Audit records zero new packages. | closed |
| T-37-09-SC | Tampering | npm/pip/cargo installs (plan 37-09) | high | accept | No install task; `django_tables2` already installed (3.0.0) and pinned. Package Legitimacy Audit not engaged. | closed |
| T-37-10-SC | Tampering | npm/pip/cargo installs (plan 37-10) | high | accept | No install task and no new dependency. Package Legitimacy Audit not engaged. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-37-01 | T-37-02 | The status legend is a fixed, hand-listed vocabulary with no database read (D-04), so it cannot leak a row's existence to an anonymous visitor. Accepted with no control beyond keeping it non-data-driven. | plan 37-01 | 2026-09-21 |
| AR-37-02 | T-37-08 | `fetched_at` records when an allocation figure was last refreshed; no per-user audit trail is kept, accepted because only the unattended runner writes these rows. | plan 37-02 | 2026-09-21 |
| AR-37-03 | T-37-12 | The gap computation sits behind the existing 1-hour result cache and adds one bounded query; accepted without a new control, consistent with the existing gap-page posture. | plan 37-03 | 2026-09-21 |
| AR-37-04 | T-37-20 | Markers are rendered from the shared table through `format_html`, so a run's own free-text fields cannot inject a marker letter. | plan 37-05 | 2026-09-21 |
| AR-37-05 | T-37-09-04 | django-tables2 resolves `sort` against the table's declared bound columns and ignores anything else, and clamps out-of-range or non-integer pages; no attacker-chosen string reaches the ORM as an ordering expression. | plan 37-09 | 2026-09-21 |
| AR-37-06 | T-37-10-04 | The unknown-run count is derived inside loops that already run — no new queryset, second pass or query — and the existing exact-count query gates must still hold unchanged. | plan 37-10 | 2026-09-21 |
| AR-37-07 | T-37-09-SC, T-37-10-SC | Neither plan has a package-manager install task and neither adds a dependency; `django_tables2` 3.0.0 is already installed and pinned by `pyproject.toml`. | plans 37-09, 37-10 | 2026-09-21 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-21 | 43 | 43 | 0 | /gsd-secure-phase (L1 grep-depth, orchestrator) |

### Security Audit 2026-09-21

| Metric | Count |
|--------|-------|
| Threats found | 43 |
| Closed | 43 |
| Open | 0 |

Method: register parsed from the `<threat_model>` blocks of all ten plans
(`register_authored_at_plan_time: true`), each mitigation verified against the
implementation at ASVS L1 grep depth. No auditor escalation was required and no
threat at or above the `high` block threshold remains open.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-21
