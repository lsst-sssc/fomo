---
phase: quick-260726-fqb
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_utils.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_import_campaign_csv.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "resolve_site('500@-170') resolves the same real Observatory that resolve_site('274') would (JWST), with needs_review=False and no 'NEEDS REVIEW:' placeholder"
    - "resolve_site('500@-48') -> 250 (Hubble), resolve_site('500@-163') -> C51 (WISE), resolve_site('500@-95') -> C57 (TESS), each a real SATELLITE_OBSTYPE Observatory with needs_review=False"
    - "resolve_site('500@-999') -- an unrecognized NAIF ID -- still returns (None, True), creates no Observatory row, and makes no network call: an unknown spacecraft is flagged for manual review, never guessed"
    - "A bare '-170' (no '500@' prefix) is NOT translated -- only the full 500@<naif> form observed in real data maps"
    - "Matching is exact on the already-stripped cell value: no case-folding, no internal-whitespace tolerance, no prefix/regex parsing of the '500@' shape"
    - "A plain obscode ('274', 'F65', 'E10') and every other Site Code shape behaves byte-for-byte as before -- the alias lookup is a pass-through miss for anything not in the 4-entry table"
    - "The importer still stores the verbatim raw cell in CampaignRun.site_raw ('500@-170'), so the translation is a resolution detail and never rewrites what the submitter typed"
    - "Observatory.obscode stays CharField(max_length=4): no migration, and campaign_utils.py's _MAX_OBSCODE_LEN guard is still the sole gate for every untranslated value"
  artifacts:
    - path: "solsys_code/campaign_utils.py"
      provides: "HORIZONS_OBSERVER_TO_OBSCODE alias table + its application at the top of resolve_site(), before the length guard"
      contains: "HORIZONS_OBSERVER_TO_OBSCODE"
    - path: "solsys_code/tests/test_campaign_approval.py"
      provides: "TestResolveSiteHorizonsObserverNotation -- per-alias end-to-end Tier-2 resolution, unknown-NAIF flagging, and the not-translated negative cases"
    - path: "solsys_code/tests/test_import_campaign_csv.py"
      provides: "Repointed over-length guard tests (no longer using 500@-170 as the unresolvable example) plus a command-level test that a real 500@-170 CSV row now resolves to 274"
  key_links:
    - "resolve_site() -> strip/empty check -> HORIZONS_OBSERVER_TO_OBSCODE lookup -> _MAX_OBSCODE_LEN guard (translation runs BEFORE the guard, never instead of it)"
    - "Translated code -> Tier 1 Observatory.objects.get(obscode=...) / Tier 2 MPCObscodeFetcher.query() -> real satellite Observatory (unblocked by quick task 260725-kn4's null-coordinate fix in to_observatory())"
    - "import_campaign_csv row -> resolve_site(site_raw) -> CampaignRun.site FK, with CampaignRun.site_raw still carrying the untranslated cell text"
---

<objective>
Teach `resolve_site()` that a `Site Code` cell in JPL Horizons / SPICE observer notation
(`500@<NAIF SPK ID>`, i.e. "geocentric observer at body N") names a spacecraft that already has
a real, short MPC obscode -- and translate it to that obscode before the existing over-length
guard runs.

The real 3I/ATLAS campaign sheet carries `500@-170` in three `CampaignRun`s (pks 21, 27, 28).
Today every one of them is flagged unresolved: 8 characters exceeds `Observatory.obscode`'s
`max_length=4`, so `resolve_site()` returns `(None, True)` before any tier is attempted. That
guard is **correct and stays exactly as it is** -- `.planning/PROJECT.md:120` records the
operator-caught correction that `500@-170` is Horizons notation, not an MPC obscode, and that
`obscode`'s `CharField(max_length=4)` deliberately does **not** need widening. Widening it would
achieve nothing anyway: `MPCObscodeFetcher.query('500@-170')` misses at MPC, and Tier 3 would
then fabricate a placeholder carrying a bogus obscode.

The fix is a small, explicit, hand-verified translation table applied at the top of
`resolve_site()`. Four mappings, each confirmed on **both** sides on 2026-07-26 -- NAIF ID ->
spacecraft via the JPL Horizons API (`ssd.jpl.nasa.gov/api/horizons.api`), obscode -> same
spacecraft via the MPC obscodes API:

| Horizons form | NAIF ID | Spacecraft | MPC obscode | MPC name |
|---|---|---|---|---|
| `500@-170` | -170 | James Webb Space Telescope | `274` | James Webb Space Telescope |
| `500@-48`  | -48  | Hubble Space Telescope     | `250` | Hubble Space Telescope |
| `500@-163` | -163 | WISE Spacecraft            | `C51` | WISE |
| `500@-95`  | -95  | TESS                       | `C57` | TESS |

All four MPC records are `observations_type: satellite` with null
`longitude`/`rhocosphi`/`rhosinphi` -- the exact shape the immediately-preceding quick task
`260725-kn4` taught `MPCObscodeFetcher.to_observatory()` to store as a coordinate-less row
instead of raising `TypeError`. So once the alias resolves, Tier 2 now works end-to-end for
these codes with no further change to the observatory app.

**Governing principle: never guess.** A `500@<something-not-in-the-table>` input must fall
through to the unchanged length guard and return `(None, True)`. Silently mapping an unknown
spacecraft to a wrong site is far worse than flagging it for manual review.

**Locked design decisions** (made here so the executor does not have to re-derive them):

- **D-01 -- Exact whole-string match only.** The lookup runs on the already-stripped cell value
  and is a plain `dict.get`. No case-folding (moot today: no alias key contains a letter, so
  folding would be a no-op that only creates future ambiguity), no internal-whitespace
  normalization, no prefix/regex parsing of the `500@` shape. `'500 @ -170'` is therefore *not*
  translated -- it falls to the length guard and is flagged, which is the correct outcome: an
  unexpected variant deserves human eyes, and the table can be extended once a real variant is
  actually observed. Surrounding whitespace *is* tolerated, because `resolve_site()` already
  strips before this point; that is existing behavior, not a new allowance.
- **D-02 -- Only the full `500@<naif>` form maps.** A bare `-170` or `@-170` is deliberately NOT
  translated. Rationale: `-170` is 4 characters, fits `_MAX_OBSCODE_LEN`, and today flows
  through Tier 1/2/3 like any other short code; mapping it would silently change behavior for a
  token that has never appeared in real data and could plausibly mean something else. Only the
  form actually seen in the sheet maps.
- **D-03 -- Translate before the guard, never instead of it.** The guard remains the sole gate
  for everything the table does not recognize. A translated value is a real <=4-char obscode and
  passes the guard on its own merits.

**Explicit non-goals** (do not do these):

- No migration, and no change to `Observatory.obscode`'s `max_length`.
- No weakening, relaxing or removal of the `_MAX_OBSCODE_LEN` guard at `campaign_utils.py:161`.
- No data migration touching Observatory rows. The operator has already applied a direct DB
  data-fix out of band (pk=3 `C51` and pk=4 `274` had bogus `lon=0.0, lat=90.0,
  altitude=-6356752.3` reset to `None`; pk=5 `289` retyped to `SATELLITE_OBSTYPE`; pk=10 `500`
  Geocentric deliberately left alone). Do not redo or codify any of it.
- No addition of the alias strings to `build_site_candidates()`'s fuzzy-match pool. The
  site-search widget is out of scope for this task.

**No demo-notebook obligation.** CLAUDE.md's paired-notebook rule covers exactly four modules --
`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`,
`sync_gemini_observation_calendar.py`. This plan's `files_modified` touches none of them
(confirmed against the list above), so no `docs/notebooks/pre_executed/` update is required.

Purpose: the three real JWST `CampaignRun`s stop being permanently stuck in "Sites Needing
Review" and gain a correct, typed `SATELLITE_OBSTYPE` site FK -- which in turn unblocks
asset-aware coverage-gap analysis and satellite calendar projection for them.

Output: one new module-level constant, one lookup line in `resolve_site()`, corrected
docstring/comment examples, one new test class, two repointed existing tests, one new
command-level test.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

@solsys_code/campaign_utils.py
@solsys_code/tests/test_import_campaign_csv.py
@solsys_code/tests/test_campaign_approval.py

**Key call sites and precedents (read these ranges, do not re-read whole files):**

- `solsys_code/campaign_utils.py:29-31` -- `_MAX_OBSCODE_LEN`, computed from the model field.
  The new alias table goes immediately after this block.
- `solsys_code/campaign_utils.py:130-245` -- `resolve_site()`. Line 157 strips, 158-159 rejects
  blank, **161-164 is the length guard**. The translation goes between 159 and 161. Line 140
  (docstring) and line 162 (inline comment) both cite `'500@-170'` as the canonical
  never-resolves example and are now wrong -- both must be corrected.
- `solsys_code/tests/test_campaign_approval.py:2246-2288` -- `TestResolveSiteSatelliteObscode`.
  **This is the pattern to copy**: it patches `requests.get` with a satellite MPC payload (null
  `longitude`/`rhocosphi`/`rhosinphi`) so the real, kn4-fixed `to_observatory()` is exercised
  end-to-end through the actual Tier-2 path, rather than stubbing `to_observatory` directly.
- `solsys_code/tests/test_import_campaign_csv.py:47-59` -- `_MPC_OBS_DATA_E10`, the full key set
  `to_observatory()` reads: `created_at`, `longitude`, `name_utf8`, `obscode`,
  `observations_type`, `old_names`, `rhocosphi`, `rhosinphi`, `short_name`, `updated_at`,
  `uses_two_line_observations`.
- `solsys_code/tests/test_import_campaign_csv.py:97-102` and `:552-575` -- the **only two**
  places in `solsys_code/` that use `'500@-170'` as test data. Both currently assert it is
  unresolvable. Both break the moment Task 2 lands (`:552` would additionally make a *live* MPC
  call, since it patches nothing). Task 1 repoints them.

**Environment rules (from CLAUDE.md and the task brief):**

- Django DB tests only: `./manage.py test solsys_code`. **Not** pytest -- `pyproject.toml`'s
  `testpaths` excludes `solsys_code/`.
- Never import `solsys_code.ephem_utils` or `solsys_code.views` from a new test -- module import
  downloads ~1.6 GB of SPICE kernels. Keep test imports narrow (`campaign_utils`, models,
  `unittest.mock` only).
- Tests must never reach the live MPC or JPL Horizons APIs. Every new test patches
  `requests.get`; the no-network cases patch it with `side_effect=AssertionError(...)` so an
  accidental call fails loudly instead of silently succeeding.
- `ruff check` / `ruff format --check` **scoped to the three touched files only**. The tree has
  known unrelated violations (a `D103` in `sync_gemini_observation_calendar_demo.ipynb`, format
  diffs in four `pre_executed` notebooks and two `.planning/quick/260619-f7u/verify_*.py`
  scripts). Do not run repo-wide and do not chase them.
- `src/fomo/settings.py` has uncommitted user-local modifications. It must never be staged,
  committed, reverted or reformatted.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the failing alias tests and repoint the two 500@-170 fixtures (RED)</name>
  <files>solsys_code/tests/test_campaign_approval.py, solsys_code/tests/test_import_campaign_csv.py</files>
  <action>
Write the tests first, before any change to `campaign_utils.py`.

**A. New test class in `solsys_code/tests/test_campaign_approval.py`**, placed immediately after
`TestResolveSiteSatelliteObscode` (which ends around line 2288) -- co-located because these
mappings are all satellite obscodes whose Tier-2 path only works thanks to that class's
subject, quick task 260725-kn4.

Add `HORIZONS_OBSERVER_TO_OBSCODE` to the existing `from solsys_code.campaign_utils import ...`
line (currently line 34). This import will fail until Task 2 lands -- that is the intended RED.

Define a module-level fixture dict `HORIZONS_MPC_PAYLOADS` keyed by MPC obscode, one full MPC
record per mapped obscode, each shaped exactly like `TestResolveSiteSatelliteObscode`'s
`satellite_payload`: `observations_type='satellite'`, `longitude=None`, `rhocosphi=None`,
`rhosinphi=None`, `old_names=None`, plausible `created_at`/`updated_at` strings,
`uses_two_line_observations=True`, and `name_utf8`/`short_name` set to the real MPC name:

  - `'274'` -> James Webb Space Telescope
  - `'250'` -> Hubble Space Telescope
  - `'C51'` -> WISE
  - `'C57'` -> TESS

Class `TestResolveSiteHorizonsObserverNotation(TestCase)`, with a docstring stating what
Horizons observer notation is, that all four mappings were verified on both the Horizons and
MPC sides on 2026-07-26, and that translation is exact-match-only (D-01) and never guesses.
Tests:

  1. `test_alias_map_is_the_four_verified_mappings` -- assert `HORIZONS_OBSERVER_TO_OBSCODE`
     equals exactly `{'500@-170': '274', '500@-48': '250', '500@-163': 'C51', '500@-95': 'C57'}`.
     Pins the operator-verified table so a future entry cannot be added without a deliberate
     test update.
  2. `test_every_alias_has_a_payload_fixture` -- assert
     `set(HORIZONS_OBSERVER_TO_OBSCODE.values()) == set(HORIZONS_MPC_PAYLOADS)`, so extending
     the map without extending the fixtures fails loudly instead of silently going untested.
  3. `test_every_alias_resolves_to_its_mpc_obscode` -- loop `HORIZONS_OBSERVER_TO_OBSCODE.items()`
     under `self.subTest(horizons=...)`; for each, patch `requests.get` to return a
     `MagicMock(ok=True)` whose `.json()` yields `HORIZONS_MPC_PAYLOADS[expected_obscode]`, call
     `resolve_site(horizons_form)`, and assert: observatory is not None, `obscode ==
     expected_obscode`, `needs_review is False`, `is_placeholder_observatory(observatory)` is
     False, `observations_type == Observatory.SATELLITE_OBSTYPE`, `lon is None`.
  4. `test_unknown_naif_id_is_flagged_not_guessed` -- patch `requests.get` with
     `side_effect=AssertionError('resolve_site must not reach the network for an unknown NAIF id')`,
     call `resolve_site('500@-999')`, assert `(None, True)` and `Observatory.objects.count() == 0`.
     **This is the single most important test in the plan** -- it proves the length guard still
     catches everything the table does not recognize, and that no network call is even attempted.
  5. `test_bare_naif_id_is_not_translated` -- D-02. `resolve_site('-170')` with `requests.get`
     patched to a `MagicMock(ok=False, status_code=501)` MPC-miss response; assert the result is
     NOT the JWST observatory: `observatory.obscode == '-170'`, `needs_review is True`, and
     `is_placeholder_observatory(observatory)` is True (a Tier-3 placeholder, exactly as today).
  6. `test_internal_whitespace_variant_is_not_translated` -- D-01. `resolve_site('500 @ -170')`
     with `requests.get` patched to `side_effect=AssertionError(...)`; assert `(None, True)`.
  7. `test_surrounding_whitespace_is_tolerated` -- `resolve_site('  500@-170  ')` (payload for
     `'274'` patched in) resolves to obscode `'274'` with `needs_review is False`. Documents that
     the pre-existing `.strip()` runs first and the alias table needs no separate allowance.
  8. `test_plain_obscode_is_unaffected` -- `resolve_site('274')` with the `'274'` payload patched
     in still resolves to `'274'`, `needs_review is False`. No-regression anchor for the already
     correct form.

**B. Repoint the two existing `'500@-170'` fixtures in
`solsys_code/tests/test_import_campaign_csv.py`.** Both currently assert `500@-170` is
unresolvable; that assertion is about to become false. Replace the *data*, keep the *intent*.
Both edits are safe in both states (they pass before and after Task 2):

  - Line 97 `test_resolve_site_oversized_returns_none_needs_review` -> rename to
    `test_resolve_site_oversized_or_unknown_horizons_returns_none_needs_review`. Assert the
    `(None, True)` + zero-Observatory outcome for **two** values: `'500@-999'` (an unrecognized
    Horizons NAIF form) and `'Lowell Discovery Telescope (G37)'` (long free text -- preserves the
    original test's plain over-length coverage). Wrap both calls in
    `patch('requests.get', side_effect=AssertionError(...))` to prove the guard short-circuits
    before Tier 2. Rewrite the comment: `500@-170` is no longer an example of an unresolvable
    over-length code.
  - Line 552 `test_unresolvable_site_flags_needs_review_without_skipping_row` -> change the CSV
    row's `'Site Code'` from `'500@-170'` to `'500@-999'` and update the final `site_raw`
    assertion to match. Update the docstring: the D-09 point is now "an *unknown* Horizons
    observer form doesn't skip the row -- just flags it". Assertions on
    `run.site is None` / `run.site_needs_review` are unchanged.

**C. New command-level test in the same file**, alongside the one above:
`test_horizons_site_code_resolves_via_alias_map`. Same single-row CSV shape as the
`test_unresolvable_site_flags_...` fixture but with `'Site Code': '500@-170'` and
`'Telescope / Instrument': 'JWST'`; `@patch('requests.get')` returning a `MagicMock(ok=True)`
whose `.json()` is the JWST (`274`) satellite payload -- define it inline in this file rather
than importing across test modules. Assert: `CampaignRun.objects.count() == 1`,
`run.site.obscode == '274'`, `run.site_needs_review is False`, and -- critically --
`run.site_raw == '500@-170'`, proving the translation is a resolution detail that never rewrites
the verbatim submitted text.

Do not touch `solsys_code/campaign_utils.py` in this task.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_campaign_approval.TestResolveSiteHorizonsObserverNotation 2>&1 | tail -20; ./manage.py test solsys_code.tests.test_import_campaign_csv 2>&1 | tail -20</automated>
  </verify>
  <done>
RED is confirmed and correctly shaped:
- `TestResolveSiteHorizonsObserverNotation` errors at import (`cannot import name
  'HORIZONS_OBSERVER_TO_OBSCODE'`) -- the expected pre-implementation failure.
- In `test_import_campaign_csv`, the two repointed tests **pass** (they are pure fixture
  corrections, valid before and after), and only the new
  `test_horizons_site_code_resolves_via_alias_map` fails -- and it fails on the assertion
  (`run.site` is None / `site_needs_review` True), never by making a live network call.
- `grep -rn '500@-170' solsys_code/` now shows it only in `campaign_utils.py` (docstring/comment,
  fixed in Task 2) and in the new/renamed passing-after-GREEN tests.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add HORIZONS_OBSERVER_TO_OBSCODE and apply it at the top of resolve_site (GREEN)</name>
  <files>solsys_code/campaign_utils.py</files>
  <behavior>
    - `resolve_site('500@-170')` behaves exactly as `resolve_site('274')` -> real JWST Observatory, `needs_review=False`
    - `'500@-48'` -> `250`, `'500@-163'` -> `C51`, `'500@-95'` -> `C57`, each via the normal Tier 1/Tier 2 path
    - `'500@-999'` -> unchanged `(None, True)` via the untouched `_MAX_OBSCODE_LEN` guard, no network call
    - `'-170'`, `'500 @ -170'`, `'274'`, `'F65'`, `''` -> byte-for-byte unchanged behavior
  </behavior>
  <action>
**1. Add the alias table** immediately after the `_MAX_OBSCODE_LEN` block (line 31), before
`NEEDS_REVIEW_NAME_PREFIX`. Name it `HORIZONS_OBSERVER_TO_OBSCODE` -- public (no leading
underscore), because it is a documented operator-facing extension point that tests import and
iterate, matching `NEEDS_REVIEW_NAME_PREFIX`'s precedent in this same constant block.

Type it `dict[str, str]`. Precede it with a comment block that states, in this order: what
Horizons/SPICE observer notation is (`500@<NAIF SPK ID>` = "geocentric observer at body N"); that
it is **not** an MPC obscode (cite `.planning/PROJECT.md:120`'s operator-caught correction, and
that `Observatory.obscode`'s `max_length=4` deliberately does not need widening); that the real
3I/ATLAS sheet carries `500@-170` in three `CampaignRun`s; that each entry was verified on
**both** sides on 2026-07-26 (NAIF ID -> spacecraft via the JPL Horizons API, obscode -> the same
spacecraft via the MPC obscodes API); and the extension rule -- **verify both sides before adding
a row, and never infer a mapping from the NAIF ID alone**. Give each entry an inline
spacecraft-name comment:

  `'500@-170'` -> `'274'` (JWST), `'500@-48'` -> `'250'` (Hubble), `'500@-163'` -> `'C51'`
  (WISE), `'500@-95'` -> `'C57'` (TESS).

**2. Apply the translation in `resolve_site()`**, between the blank-code return (line 159) and
the `_MAX_OBSCODE_LEN` guard (line 161) -- per D-03, before the guard, never instead of it. A
plain exact-match `dict.get(code, code)` (per D-01: no case-folding, no whitespace
normalization, no `500@` prefix/regex parsing). When and only when a translation actually fires,
emit a `logger.debug` naming both the original and the translated form, so an operator debugging
a surprising site resolution can see it happened; the module's `logger` already exists at line 27.
Add a short comment above it explaining that anything unrecognized deliberately falls through to
the guard below and is flagged (D-09: flag, don't guess).

**3. Correct the two now-wrong `'500@-170'` examples:**

  - Docstring line ~140: it currently offers `'500@-170'` as the canonical "can't possibly be a
    real MPC obscode" example. Replace with the two-part story: a **recognized** Horizons
    observer form (see `HORIZONS_OBSERVER_TO_OBSCODE`) is translated to its real MPC obscode
    first, so `resolve_site('500@-170')` behaves exactly like `resolve_site('274')`; anything
    else over-length -- including an **unrecognized** `500@<naif>` such as `'500@-999'` -- is
    still flagged immediately with no Observatory row created and no network call, so a
    non-obscode is never truncated or fabricated (D-09/Pitfall 2).
  - Also extend the `site_code_raw` entry in the `Args:` section to note the Horizons-notation
    translation. Keep Google-style docstring formatting.
  - Inline comment at line ~162: change its example from `'500@-170'` to `'500@-999'` (an
    unrecognized Horizons form) so the comment matches what the guard now actually catches.

**Do NOT:** widen `Observatory.obscode`; add any migration; weaken, relax, reorder or remove the
`_MAX_OBSCODE_LEN` guard; add the alias strings to `build_site_candidates()`'s candidate pool;
add prefix- or regex-based `500@` parsing; or change any test file in this task.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_campaign_approval.TestResolveSiteHorizonsObserverNotation solsys_code.tests.test_campaign_approval.TestResolveSiteSatelliteObscode solsys_code.tests.test_import_campaign_csv 2>&1 | tail -20</automated>
  </verify>
  <done>
All eight tests in `TestResolveSiteHorizonsObserverNotation` pass, the pre-existing
`TestResolveSiteSatelliteObscode` still passes, and the whole `test_import_campaign_csv` module
passes including the new `test_horizons_site_code_resolves_via_alias_map`.
`git diff solsys_code/campaign_utils.py` shows exactly: the new constant block, one lookup line
plus its comment and `logger.debug` inside `resolve_site()`, and the docstring/comment example
corrections -- no migration file, no `max_length` change, no edit to the guard's condition.
  </done>
</task>

<task type="auto">
  <name>Task 3: Full solsys_code regression sweep and scoped quality gates</name>
  <files>solsys_code/campaign_utils.py, solsys_code/tests/test_campaign_approval.py, solsys_code/tests/test_import_campaign_csv.py</files>
  <action>
No new feature work. Prove nothing else regressed and the tree is clean, editing only if a
genuine fallout from Tasks 1-2 surfaces.

1. Run the **whole** app suite: `./manage.py test solsys_code`. `resolve_site()` is consumed by
   `campaign_views.py`, `campaign_tables.py`, `import_campaign_csv.py` and
   `backfill_range_calendar_events.py`, so a wider blast radius is possible. If a test outside
   the three touched files fails, it is almost certainly encoding the old "`500@-170` is
   permanently unresolvable" behavior -- surface it explicitly and decide deliberately whether
   the assertion or the implementation is wrong; do not paper over it with a broadened assertion.
2. Scoped lint/format, **on the three touched files only** -- do not run repo-wide, do not chase
   the known pre-existing notebook/`verify_*.py` violations:
   `ruff check solsys_code/campaign_utils.py solsys_code/tests/test_campaign_approval.py solsys_code/tests/test_import_campaign_csv.py`
   and the same file list under `ruff format --check`.
3. Confirm the diff is exactly the three declared files and nothing else:
   `git status --short` must show no other modified/added path -- in particular
   `src/fomo/settings.py` must remain unstaged and untouched (it carries uncommitted user-local
   modifications), and no new file may appear under any `migrations/` directory.
4. Confirm no heavy-import leak: `git diff` contains no new import of `solsys_code.ephem_utils`
   or `solsys_code.views`.
  </action>
  <verify>
    <automated>./manage.py test solsys_code 2>&1 | tail -15 && ruff check solsys_code/campaign_utils.py solsys_code/tests/test_campaign_approval.py solsys_code/tests/test_import_campaign_csv.py && ruff format --check solsys_code/campaign_utils.py solsys_code/tests/test_campaign_approval.py solsys_code/tests/test_import_campaign_csv.py && git status --short</automated>
  </verify>
  <done>
`./manage.py test solsys_code` reports OK with zero failures and zero errors (test count is the
prior baseline plus the newly added tests). Both scoped ruff invocations exit clean.
`git status --short` lists only the three declared files as modified; `src/fomo/settings.py` is
not staged and shows no new changes; no `migrations/` file was added.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CSV `Site Code` cell / public submission free text -> `resolve_site()` | Semi-trusted operator-curated data and fully untrusted public form input cross into a lookup that decides which persisted `Observatory` a run is attributed to |
| `resolve_site()` -> `Observatory` FK on `CampaignRun` | A wrong resolution here silently mis-attributes a scientific observation to the wrong facility, with no downstream check |
| MPC Obscodes API -> `MPCObscodeFetcher.to_observatory()` -> persisted `Observatory` | Untrusted third-party JSON is read key-by-key and written into a row (unchanged by this plan) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-fqb-01 | Spoofing | `resolve_site()` alias lookup | mitigate | A crafted site string resolving to the wrong facility is the primary risk of this change. Bounded by construction: exact whole-string `dict.get` over a 4-entry, both-sides-verified table (D-01) -- no prefix, regex, fuzzy or case-insensitive matching, so no input outside those four literal strings can be translated at all. `test_alias_map_is_the_four_verified_mappings` pins the table; `test_bare_naif_id_is_not_translated` and `test_internal_whitespace_variant_is_not_translated` pin the negative space. |
| T-fqb-02 | Elevation of Privilege | `_MAX_OBSCODE_LEN` guard at `campaign_utils.py:161` | mitigate | The change must not become a bypass of the guard that stops a non-obscode being truncated or fabricated into an `Observatory` row. D-03 places the translation *before* the guard, never in place of it: a translated value is a genuine <=4-char MPC obscode that passes on its own merits, and every untranslated value still meets the unchanged guard. `test_unknown_naif_id_is_flagged_not_guessed` is the pinning test. |
| T-fqb-03 | Tampering | Data integrity of `CampaignRun.site_raw` | mitigate | The translation must not rewrite what a submitter actually typed, or the audit trail from sheet cell to resolved site is lost. `site_raw` is set by the importer from the CSV cell and is untouched by `resolve_site()`; `test_horizons_site_code_resolves_via_alias_map` asserts `run.site_raw == '500@-170'` explicitly. |
| T-fqb-04 | Information Disclosure | Test suite network egress | mitigate | New tests must never reach the live MPC or JPL Horizons APIs. Every test patches `requests.get`; the two no-network cases patch it with `side_effect=AssertionError(...)` so an accidental call fails loudly rather than silently succeeding against a live endpoint. No credentials or PII are involved on any path. |
| T-fqb-05 | Tampering | MPC API response -> persisted `Observatory` | accept | A hostile or compromised MPC response could set an arbitrary `name`/`short_name` on a Tier-2 row. Unchanged by this plan and already partially bounded (`Observatory.clean()` rejects the reserved `NEEDS REVIEW: ` prefix on form-validated saves; `name` and `obscode` are `unique=True`). MPC is a trusted scientific authority reached over HTTPS. Same disposition as quick task `260725-kn4`. |
| T-fqb-06 | Denial of Service | `resolve_site()` per-row cost | accept | The added work is a single `dict.get` on an in-memory 4-entry table -- no I/O, no new network call, and for a translated code it *removes* a per-import row from the flagged-for-review backlog. Negligible. |
| T-fqb-SC | Tampering | npm/pip/cargo installs | n/a | This plan installs no packages and adds no dependency. Every import used (`logging`, `requests`, `unittest.mock`, Django, existing project modules) is already in use in the touched files. |
</threat_model>

<verification>
1. `./manage.py test solsys_code` -- OK, zero failures/errors, count = prior baseline + new tests.
2. `./manage.py test solsys_code.tests.test_campaign_approval.TestResolveSiteHorizonsObserverNotation`
   -- all eight tests pass, including the unknown-NAIF and both not-translated negative cases.
3. `./manage.py test solsys_code.tests.test_import_campaign_csv` -- passes, including the
   repointed guard tests and the new command-level `500@-170` -> `274` resolution test.
4. `grep -n 'HORIZONS_OBSERVER_TO_OBSCODE' solsys_code/campaign_utils.py` -- appears exactly
   twice: the constant definition and the lookup inside `resolve_site()`.
5. `grep -n 'max_length' solsys_code/campaign_utils.py` and `git status --short` -- the
   `_MAX_OBSCODE_LEN` derivation is unchanged and no `migrations/` file was added.
6. `ruff check` and `ruff format --check` clean **over the three touched files only**.
7. `git status --short` shows only the three declared files; `src/fomo/settings.py` is neither
   staged nor modified by this work.
8. No new import of `solsys_code.ephem_utils` / `solsys_code.views` anywhere in the diff.
</verification>

<success_criteria>
- `resolve_site('500@-170')` returns the real JWST `Observatory` (obscode `274`,
  `SATELLITE_OBSTYPE`, coordinate-less) with `needs_review=False` -- identical to
  `resolve_site('274')`.
- The other three verified mappings (`500@-48` -> `250`, `500@-163` -> `C51`, `500@-95` -> `C57`)
  resolve the same way.
- `resolve_site('500@-999')` still returns `(None, True)`, creates no `Observatory`, and makes no
  network call -- an unknown spacecraft is flagged for review, never guessed.
- A bare `-170`, an internal-whitespace variant `500 @ -170`, a plain obscode `274`, a ground
  code `F65` and a blank cell all behave exactly as before.
- `CampaignRun.site_raw` still carries the verbatim `'500@-170'` cell text after import.
- `Observatory.obscode` is still `CharField(max_length=4)`; no migration exists; the
  `_MAX_OBSCODE_LEN` guard is unmodified.
- `./manage.py test solsys_code` is fully green and the scoped ruff gates are clean.
</success_criteria>

<output>
Create `.planning/quick/260726-fqb-map-jpl-horizons-naif-observer-notation-/260726-fqb-SUMMARY.md` when done
</output>
</content>
</invoke>
