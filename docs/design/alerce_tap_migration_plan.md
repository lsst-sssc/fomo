# ALeRCE DataService: tests + TAP general queries + TAP classifiers

## Context

`AlerceDataService` (`~/git/tom_base/tom_dataservices/data_services/alerce.py`, branch
`migrate-alerce-to-tap`) is mid-migration from the ALeRCE REST python client (`alerce` 2.3.0) to the
ALeRCE TAP service (`https://tap.alerce.online/tap` via `pyvo`). Only the single-`oid` LSST lookup has
been ported so far. Live testing against both APIs surfaced concrete breakage:

- **LSST general/classifier queries crash**: `alerce.query_objects(survey='lsst')` returns a bare
  `list`, but `query_service()` calls `.get("items", [])` on it (ZTF returns a `dict`) → `AttributeError`.
- **LSST `ndet` filter rejected**: the multisurvey REST client whitelists `n_det` and raises
  `ValueError: Invalid parameter: ndet`, but `build_query_parameters()` always emits `ndet`.
- **LSST classifier list unavailable**: `alerce.query_classifiers(survey='lsst')` raises
  `NotImplementedError` (multisurvey client stub missing); `get_classifiers()` also never passes
  `survey=` so it silently returns ZTF classifiers regardless of form selection.
- **Field-name drift**: TAP + LSST-REST use `n_det`/`deltamjd`; ZTF-REST uses `ndet`/`deltajd`. Only
  `n_det`→`ndet` is normalized today (and only on the TAP oid path).
- There are **zero tests** for this module (only the deprecated `tom_alerts` broker has coverage).

Direction confirmed by ALeRCE's own LSST example notebook: TAP for everything tabular; REST client
retained only for stamps. Decisions taken with the user:
- **LSST general queries move to TAP; ZTF general queries stay on REST** (TAP has no ZTF string names —
  `alerce_tap.ztf_object.oid` is numeric — so ZTF-via-TAP would regress target naming from `ZTF18…` to
  numeric ids).
- **`get_classifiers()` → TAP is in scope** (verified live: `alerce_tap.classifier ⋈ alerce_tap.taxonomy`
  filtered by `tid` reproduces the REST shape for any survey). Querying *objects by* LSST
  classifier/probability stays deferred.
- Out of scope (follow-ups): LSST light-curve ingestion (flux-vs-mag decision pending), LSST bands in
  `ALERCE_FILTERS`, stamps, forced photometry.

Live-verified reference data for fixtures/ADQL:
- `alerce_tap.object` cols: `oid, tid, sid, meanra, meandec, sigmara, sigmadec, firstmjd, lastmjd,
  deltamjd, n_det, n_forced, n_non_det, created_date, updated_date`
- `sid` map: 0=ZTF, 1=LSST diaObject, 2=LSST ssObject; `tid`: 0=ZTF, 1=LSST
- Cone search works: `1 = CONTAINS(POINT('ICRS', meanra, meandec), CIRCLE('ICRS', ra, dec, radius_deg))`
- `alerce_tap.classifier` cols: `classifier_id, classifier_name, classifier_version, tid, created_date`;
  `alerce_tap.taxonomy` cols: `class_id, class_name, taxonomy_order, classifier_id, created_date`
- REST ZTF `query_classifiers` shape: `[{classifier_name, classifier_version, classes: [...]}, ...]`

## Files

- **Edit**: `tom_base/tom_dataservices/data_services/alerce.py`
- **New**: `tom_base/tom_dataservices/tests/data_services/test_alerce.py`
  (pattern: `tom_base/tom_dataservices/tests/data_services/test_mpc.py` — `django.test.TestCase`,
  `unittest.mock.patch`/`MagicMock`; fixtures inline as dicts, no JSON files needed given small size)

## Testability constraints (drive the mocking strategy)

- Module-level singletons `alerce = Alerce()` and `tap_service = pyvo.dal.TAPService(TAP_URL)` at
  `alerce.py:21-23` → patch **module attributes** `tom_dataservices.data_services.alerce.alerce` and
  `...alerce.tap_service` (both constructors are lazy/no-network, so import is safe).
- `AlerceForm.__init__` → `add_classifiers_fields()` → `get_classifiers()` hits network + Django cache
  (`ds_alerce_classifiers`, 24h TTL) → every form test patches the classifier query and calls
  `cache.clear()` in `setUp`.
- Mock TAP results as a `list` of plain `dict`s: code does `dict(result[0])` and iteration — both work;
  add one case with `np.float64`/`np.int64` values to cover `_to_native_types`.

## Step 1 — Baseline tests for current behavior (commit 1)

`test_alerce.py`, roughly four TestCase classes:

**`TestAlerceForm`** (patch classifier query, `cache.clear()`):
- classifier + `prob_cfield_*` fields dynamically added from mocked
  `[{classifier_name, classifier_version, classes}]`
- caching: second instantiation does not re-query (assert single call)
- `clean()` bundles selected classifier fields into `cleaned_data["classifiers"]`

**`TestBuildQueryParameters`** (no mocks needed):
- ZTF → `sid=0`, `survey='ztf'`; LSST diaObject → `sid=1`; LSST ssObject → `sid=2`
- `firstmjd`/`lastmjd` become `[gt, lt]` lists only when both bounds given
- `ndet`: `[min, max]`, `[0, max]` when only max, `[min]` when only min, absent when neither
- cone params set only when all of ra/dec/radius present
- `object_id` → `oid`; `classifiers` defaults to `[]`

**`TestQueryService`** (patch `alerce` and `tap_service` module attrs):
- ZTF oid path: REST `query_objects` called, `sid` stripped
- LSST oid path (`sid=1`): `tap_service.search` called with ADQL containing `oid =` and `sid = 1`;
  `n_det` renamed to `ndet`; numpy scalars converted to native types
- ZTF general path: `.get("items")` unwrap, `survey` annotated onto each result
- ZTF classifier loop: one `query_objects` call per classifier with `classifier`/`class_name`/`probability`
- `ObjectNotFoundError`/`APIError`/`ValueError` → `QueryServiceError`
- **`@expectedFailure`**: LSST general query (REST returns `list` → `AttributeError`) — pins the bug
  Step 2 fixes; flips to a real test then

**`TestTargetAndDatumCreation`**:
- `create_target_from_query`: name=oid, SIDEREAL, ra/dec from meanra/meandec
- `create_target_extras_from_query`: excludes `oid`/`meanra`/`meandec`
- `create_reduced_datums_from_query`: ZTF detections (`magpsf`/`sigmapsf`/`fid`) →
  `PhotometryReducedDatum` rows; non-detections → `limit` rows (needs a saved `Target` in DB)
- **`@expectedFailure`**: LSST detection fixture (`psfFlux`, `band: 4`, no `magpsf`) → `KeyError` —
  documents the flux-vs-mag gap for the follow-up

Run: `cd ~/git/tom_base && python manage.py test tom_dataservices.tests.data_services.test_alerce`

## Step 2 — TAP general query path for LSST (commit 2)

In `alerce.py`:

1. **New helper `_build_tap_object_query(query_parameters) -> str`** building ADQL against
   `alerce_tap.object`:
   - `SELECT TOP {page_size, default 20} * FROM alerce_tap.object`
   - `WHERE sid = {sid}`
   - cone (when ra/dec/radius present):
     `AND 1 = CONTAINS(POINT('ICRS', meanra, meandec), CIRCLE('ICRS', {ra}, {dec}, {radius/3600.0}))`
     (**form radius is arcsec; ADQL CIRCLE wants degrees**)
   - `firstmjd`/`lastmjd`: consume the existing `[gt, lt]` list format from `build_query_parameters`
     → `AND firstmjd >= {gt} AND firstmjd <= {lt}` (same for lastmjd)
   - `ndet` list → `AND n_det >= {min}` and, if 2 elements, `AND n_det <= {max}`
   - Numeric interpolation only (all these params are floats/ints via form cleaning — no string
     injection surface; keep oid out of this helper)
2. **Rewire `query_service`**: when `sid != 0` and no `oid`, run
   `tap_service.search(self._build_tap_object_query(...))`, convert rows via `_to_native_types`,
   apply `n_det`→`ndet` normalization (factor the existing rename into a small
   `_normalize_tap_record` helper reused by the oid path), annotate `survey`. ZTF paths untouched.
   LSST **classifier** queries: raise `QueryServiceError('LSST classifier queries not yet supported')`
   instead of crashing.
3. **`deltajd`→`deltamjd`** normalization on ZTF REST results (so downstream/extras see one name).
4. Remove the leftover debug `pprint` import/call (lines 9, 171) while touching `query_service`.

Tests added in the same commit: ADQL builder unit tests (cone/radius conversion, one-sided ndet,
mjd ranges, TOP/sid), LSST general query end-to-end with mocked `tap_service.search` (flips the
Step-1 `@expectedFailure`), LSST classifier query raises `QueryServiceError`, `deltajd` normalization.

## Step 3 — `get_classifiers()` via TAP (commit 3)

1. Replace REST call in `AlerceForm.get_classifiers()` with TAP:
   ```sql
   SELECT c.classifier_name, c.classifier_version, t.class_name
   FROM alerce_tap.classifier c
   JOIN alerce_tap.taxonomy t ON t.classifier_id = c.classifier_id
   WHERE c.tid = {tid} ORDER BY c.classifier_name, t.taxonomy_order
   ```
   then group rows into the existing `[{classifier_name, classifier_version, classes: [...]}]` shape
   (keeps `add_classifiers_fields` unchanged).
2. Survey-aware: `tid` from the form's bound/initial `survey` value (ZTF→0, LSST→1; module const
   `SURVEY_TID = {'ZTF': 0, 'LSST': 1}`). Cache key becomes `ds_alerce_classifiers_{tid}`.
3. Tests: grouping logic from mocked TAP rows, per-survey cache keys, LSST form now gets
   `stamp_classifier_rubin_beta`-style entries (mocked), no REST `query_classifiers` call remains.

## Out of scope (recorded for follow-ups)

LSST light curves/`PhotometryReducedDatum` (flux→mag decision needed), LSST bands z/y/u in
`ALERCE_FILTERS` (LSST rows carry `band_name` directly — likely fix), LSST classifier *object* queries
(TAP `probability` join), forced photometry naming drift (`mag`/`e_mag` vs `magpsf`), stamps ingestion.

Added 2026-08-20, per [Rubin Community confirmation](https://community.lsst.org/t/issues-querying-lsst-ssos-through-alerce-client/12428/2)
that Solar System objects are TAP-only (no REST support at all, matching the direction already taken
here) and ALeRCE's `notebooks/LSST/ALeRCE_LSST_SSO.ipynb` (`alercebroker/usecases`):

- **`alerce_tap.lsst_mpc_orbits` (known-SSO orbital elements) is unexposed.** Keyed by `ssObjectId`
  (not `alerce_tap.object.oid`, though for `sid=2` rows the two are the same value); columns include
  `designation`, `a`, `mean_anomaly`, `period`, `mean_motion`, `earth_moid`. This is arguably the most
  FOMO-relevant ALeRCE table (orbital elements for minor planets) and isn't touched by `query_service`,
  `create_target_from_query`, or `create_target_extras_from_query` today.
- **Designation-based lookup for `sid=2` (ssObject) isn't handled.** `AlerceForm.object_id` /
  `build_query_parameters` feed straight into `oid =` against `alerce_tap.object`, which for ssObjects
  is the numeric `ssObjectId`, not a human designation (e.g. "2000 SK234"). The notebook resolves
  designation → `ssObjectId` via `alerce_tap.lsst_mpc_orbits WHERE designation = '...'` (noted there as
  unindexed/slower than an `ssObjectId` lookup).
- **Classifier scope note (behavior already correct, now confirmed):** the stamp classifier only
  applies to `sid=1` (diaObject) — known SSOs (`sid=2`) are pre-assigned probability 1 asteroid rather
  than classified. `query_service`'s blanket `QueryServiceError` for any LSST classifier query is safe
  as-is; if classifier *object* queries are ever brought in-scope, they should be restricted to `sid=1`
  rather than offered for `sid=2`.

Given FOMO's minor-planet focus, `lsst_mpc_orbits`/designation lookup may be worth pulling into scope
sooner than the rest of this list — revisit before treating the above purely as deferred follow-ups.

## Verification

1. `cd ~/git/tom_base && python manage.py test tom_dataservices.tests.data_services.test_alerce` — all pass
   (two `@expectedFailure` in Step 1; the LSST-general one flips to passing in Step 2).
2. `python manage.py test tom_dataservices` — no regressions in the app suite.
3. Lint per tom_base config (`flake8` via its setup.cfg/tox config if present).
4. Optional live smoke test (network): LSST cone search + classifier fetch through a Django shell —
   `AlerceDataService().query_service(ds.build_query_parameters({'survey': 'LSST', 'ra': 305.58, 'dec': -18.79, 'radius': 60.0}))`.
5. Commits land on `migrate-alerce-to-tap` in `~/git/tom_base` (one per step; **no push** unless asked).
