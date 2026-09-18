# Plan: `ADESReducedDatum` — link astrometry and photometry rows from one ADES observation

Written 2026-09-17 for a fresh Claude Code session (no prior context), from the 2026-09-16 schema
audit ("ADES OpticalType Coverage", https://claude.ai/artifact/W4vLKBCokkKatLGoiPB5px) and a design
discussion on 2026-09-17. Everything below was verified on this machine on 2026-09-17 unless marked
**(check)**.

## Priority context (read first)

This depends on, and comes after, the ADES processor port:

- `feature/add_mpc_obs` → FOMO [PR #51](https://github.com/lsst-sssc/fomo/pull/51), open against
  `issue29-jpl-scout-ingest`, itself queued behind [PR #50](https://github.com/lsst-sssc/fomo/pull/50).
  Base the FOMO branch for this work on `feature/add_mpc_obs` until #51 merges, then rebase onto
  wherever it landed.
- `docs/plans/visibility_windows.md` is the parallel, lower-priority track; don't mix the two.

## Why

`ADESProcessor` (`solsys_code/processors/ades_processor.py`) turns each ADES row into **one**
`AstrometryReducedDatum`, burying `mag`/`rmsMag`/`band` in the free-form `value` JSON. The audit
showed that of the 75 fields in ADES `OpticalType` only 14 reach a typed column, and that the fields
sit at **five different grains**, not two:

| Grain | Constant over | Fields | Belongs in |
|---|---|---|---|
| Tracklet | frames of one tracklet | `trkSub trkID trkMPC` (3) | tracklet table, or denormalised onto the frame row |
| **Frame / exposure** | every detection on one exposure | `mode stn sys ctr pos1-3 vel1-3 posCov11-33 prog obsTime fltr seeing exp` (21) | **`ADESReducedDatum`** — this plan |
| Reduction | all detections reduced the same way | `astCat band photCat rmsFit nStars subFrm` (6) | reduction table keyed by `reduction_version` (later) |
| Detection | one measured source | `ra dec rmsRA rmsDec rmsCorr mag rmsMag photAp logSNR obsID …` (27) | `AstrometryReducedDatum` / `PhotometryReducedDatum` rows |
| Orbit solution × detection | one observation under one orbit fit | `orbID resRA resDec sigRA … photMod` (18, all NoSubmit) | separate residual table (later) |

Frame-grain data is what a link row should carry once, instead of being duplicated onto both the
astrometry and the photometry row. `PhotometryReducedDatum` has `exposure_time` and
`AstrometryReducedDatum` does not — the asymmetry that makes the case.

Audit findings that constrain the design (details in the artifact):

- **`obsID`** is MPC-assigned (`NoSubmit`) but is *their* globally unique key for the observation:
  store it, nullable + unique-when-present, as the reconciliation handle for idempotent re-ingest and
  for following `deprecated` flags. Never a primary key (locally produced data has none).
- **`fltr` ≠ `band`.** The processor maps `band` → `bandpass`, so a Z24 file with g/i/r exposures
  under one catalogue passband `G` loses the filter. `fltr` is frame-grain; keep it on the link row.
- **`logSNR` is not a stored form of `rmsMag`** (ratio ranges 0.9–22 across the corpus) and ADES
  defines it over the *astrometric* aperture. Keep it as-is on the astrometry side; never derive one
  from the other on write.
- **Residuals go in their own table** (ADES `standaloneResidual.xml` proves the block stands alone);
  folding them into the link row caps you at one orbit solution per observation.
- **`ObservationRecord` is the wrong home** (models a *request*: `facility`, `parameters`, `status`,
  `user`; `save()` fires `observation_change_state`).

## The seam between tom_base and FOMO (verified in tomtoolkit 3.0.1)

`tom_dataproducts/data_processor.py::run_data_processor` (call sites: `views.py:216` upload view,
`api_views.py:92`, `tasks.py:98`, PanSTARRS service):

1. `data = processor.process_data(dp)` → list of `(timestamp, dict, source_name)`.
2. Each row becomes `try_parse_reduced_datum({'target': …, 'data_product': dp, 'timestamp': …,
   'source_name': …, 'data_type': data_type, **datum[1]})`. **Because `**datum[1]` comes last, a
   per-row `data_type` key in the processor's dict overrides the processor-level one.** So one
   processor can already emit an `astrometry` dict *and* a `photometry` dict per ADES row with no
   tom_base change — but this is accidental and untested upstream.
3. Instances are grouped by concrete class and created with
   `model_class.objects.bulk_create(instances, ignore_conflicts=True)`. With `ignore_conflicts=True`
   Django does **not** set primary keys on the returned instances, and `bulk_create` fires no
   signals. There is no hook between build and create, and nothing is called afterwards except
   `continuous_share_data`.

Consequence: **pair emission can live entirely in FOMO; link-row creation cannot.** The only way to
get the astrometry/photometry PKs to point a link row at is a change to `run_data_processor` in
tom_base. That is the split, the same way tom_jpl took the generic Scout client and FOMO kept the
`ScoutCandidate` model and views.

Unique constraints you will need for re-fetching after `bulk_create`:

- `AstrometryReducedDatum`: `(target, timestamp, telescope, instrument, ra, dec, reduction_version)`
- `PhotometryReducedDatum`: `(target, bandpass, timestamp, limit, brightness, instrument, reduction_version)`

Both inherit `ReducedDatumCommon`: `target`, `data_product`, `timestamp`, `value` (JSON), `telescope`,
`instrument`, `source_name`, `source_location`, `reduction_version`.

## Where everything is

| What | Where | Notes |
|---|---|---|
| tom_base clone | `~/git/tom_base`, on `dev`, clean | remotes: `origin` = `talister/tom_base` (fork), `upstream` = `TOMToolkit/tom_base` |
| tom_base base commit | `upstream/dev` `b6edb22b` 2026-09-09 | latest `tom_dataproducts` migration is `0019_alter_astrometryreduceddatum_options_and_more` |
| tom_base tests | `cd ~/git/tom_base && python manage.py test tom_dataproducts` | **No poetry env exists on this machine** (`poetry env info` → none); either `poetry install` first or run against `~/venv/fomo311_venv` with tom_base's source first on `PYTHONPATH`. Linter is **flake8** (`.flake8` at repo root, `flake8>=7.3,<7.4` in `pyproject.toml`), not ruff |
| Installed toolkit | `~/venv/fomo311_venv`: tomtoolkit 3.0.1, Django 5.2.17, Python 3.11 | its editable `fomo` points at `~/git/fomo_fresh`; always run FOMO tests with `PYTHONPATH=<checkout>/src:<checkout>` |
| FOMO processor | `solsys_code/processors/ades_processor.py` on `feature/add_mpc_obs` (`af926eb`) | `_process_astrometry_from_plaintext` / `_from_df`, two column mappings (ADES camelCase, MPC-Explorer lowercase) |
| FOMO fixtures | `solsys_code/tests/test_data/test_ades.psv` (9 rows), `test_ades_mpcexplorer.psv` (5), `test_ades_df.csv` (6), `sample.psv` (Z24: 6 frames × 6 apertures = 36 rows, g/i/r under band `G`) | Location-group columns are present but **empty** in all FOMO fixtures — they won't catch a regression there. The audit's cited `ADES-Master/tests/input/pass_headers.psv` does **not** exist here; the only PSV with a populated Location group is `~/git/ADES-Master/tests/input/319.psv` (4 rows, `sys=ICRF_KM ctr=399 pos1..3` set) and it is `OccultationType` (`raStar`/`decStar`, mode `OCC`), which the processor doesn't handle. Build a synthetic `OpticalType` fixture instead (step B3) |
| FOMO settings | `src/fomo/settings.py` `DATA_PRODUCT_TYPES['astrometry']`, `DATA_PROCESSORS['astrometry']` | data type string stays `'astrometry'` |
| Audit artifact | https://claude.ai/artifact/W4vLKBCokkKatLGoiPB5px | field-by-field tables; "Five grains"; "What this means for the link table" |
| Prior fork branch | `talister/tom_base` `add-ades-reduced-data` — **deleted** after the port | don't look for it |

## Decisions already made (don't re-litigate)

- Two branches, one purpose: **tom_base** gets the generic processor-pipeline change; **FOMO** gets
  the ADES-specific model(s), the processor changes and the UI. Nothing MPC/ADES-specific goes into
  tom_base in this round (upstreaming `ADESProcessor` itself remains a separate, later task).
- The link model is called `ADESReducedDatum`, lives in `solsys_code/models.py`, with its migrations
  in `solsys_code/migrations/`. It is **not** a `ReducedDatumCommon` subclass (it isn't a datum; it
  is a frame record that two datums hang off).
- `ADESReducedDatum` carries the **frame grain** only. Reduction, residual and tracklet tables are
  out of scope for this plan.
- `fltr` and `band` are stored separately (`fltr` on the link row, `band` → `PhotometryReducedDatum.bandpass`).
- `logSNR` stays on the astrometry side, in `AstrometryReducedDatum.value`. No derivation from `rmsMag`.
- Photometry from ADES rows **does** now become `PhotometryReducedDatum` rows (this reverses the
  "keep mag in `value`" default from `docs/plans/ades_processor_port.md`; that default was only ever
  a stop-gap until this link model existed).
- Land the FOMO side first in a form that works against stock tomtoolkit 3.0.1 (pair emission, link
  rows left unpopulated), then switch linking on when the tom_base hook is released. Pinning FOMO to
  the fork, as was done with `tom_jpl`, is the fallback if the upstream review stalls.

## Open decisions — pick the default unless the user says otherwise

- **Shape of the tom_base hook.** Default: add `DataProcessor.post_process(self, data_product,
  reduced_datums)` (no-op in the base class) and have `run_data_processor` call it after
  `bulk_create` with **re-fetched** instances (query each model class back by its unique-constraint
  fields so pre-existing rows skipped by `ignore_conflicts` are included). Alternative: have
  `process_data` return model instances directly — bigger contract change, harder to upstream.
- **Two models, not one — a frame is one-to-many to detections.** Verified in `sample.psv`
  (Z24, 36 rows): there are only **6 frames**, each with **6 rows at different `photAp`**
  (1.0–2.9 arcsec), and each row has its own `ra`/`dec` centroid, `rmsRA`/`rmsDec`, `mag`,
  `rmsMag` and `logSNR`. `(obsTime, stn, mode, fltr)` is therefore *not* unique per row; only
  `(obsTime, fltr, photAp)` is. Default shape:
  - `ADESFrame` — one per exposure, the 21 frame-grain fields, `obs_id` **not** here (it is
    per observation, i.e. per detection). `UniqueConstraint(stn, obs_time, mode, fltr)`. No target
    FK (a frame can hold several targets); `data_product = ForeignKey(DataProduct, null=True,
    on_delete=SET_NULL)`.
  - `ADESReducedDatum` — one per ADES row (detection): `frame = ForeignKey(ADESFrame,
    on_delete=CASCADE)`, `astrometry = OneToOneField(AstrometryReducedDatum, null=True,
    on_delete=SET_NULL)`, `photometry = OneToOneField(PhotometryReducedDatum, null=True,
    on_delete=SET_NULL)`, `target = ForeignKey(Target, on_delete=CASCADE)`, `obs_id` nullable +
    unique-when-present (`UniqueConstraint(fields=['obs_id'], condition=Q(obs_id__isnull=False))`),
    `phot_ap` (nullable float), `value = JSONField(default=dict)`.
  The name `ADESReducedDatum` stays on the per-row link (it is the thing the two datums hang off);
  if the two-model split feels heavy, the fallback is a single `ADESReducedDatum` with
  `UniqueConstraint(target, obs_time, stn, mode, fltr, phot_ap)` and the frame fields duplicated
  per row — which is exactly the duplication the audit argued against, so prefer the two models.
- **Silent-drop risk on the photometry side.** `PhotometryReducedDatum`'s unique key is
  `(target, bandpass, timestamp, limit, brightness, instrument, reduction_version)`: two apertures
  on one frame that happen to report the same `mag` collide and `ignore_conflicts=True` drops the
  second without error. Put `photAp` into `instrument` (e.g. `instrument='photAp=1.8'`)? No —
  ugly and lies about the instrument. Default: accept the risk, detect it in `post_process`
  (fewer photometry datums than dicts emitted → `logger.warning`), and raise it in the tom_base PR
  as motivation for a processor-supplied dedup key. **(check upstream appetite)**
- **Location group storage.** Default: `sys`, `ctr` as `CharField`s, `pos1..3`, `vel1..3`,
  `pos_cov11..33` as nullable `FloatField`s (14 columns). Alternative: one `location = JSONField`.
  Columns are the point of the exercise; go with columns.
- **Field naming.** Default: snake_case of the ADES name (`obs_time`, `stn`, `mode`, `exp`, `fltr`,
  `seeing`, `prog`, `obs_id`, `pos_cov11` …) so the ADES ↔ column mapping is mechanical and reversible
  for a future PSV/XML writer.

## Steps

### Part A — tom_base branch (generic; upstreamable)

Branch: `feature/data-processor-post-process` off `upstream/dev`, pushed to `origin` (the fork), PR
against `TOMToolkit/tom_base:dev`.

```bash
cd ~/git/tom_base && git fetch upstream --prune && git branch --show-current   # expect dev, clean
git checkout -b feature/data-processor-post-process upstream/dev
```

A1. `tom_dataproducts/data_processor.py`:
   - `DataProcessor.post_process(self, data_product, reduced_datums)` — base implementation `return None`,
     docstring explaining it receives the persisted (PK-bearing) instances for this call, including
     rows that already existed.
   - In `run_data_processor`, after the `bulk_create` loop, re-fetch: for each `model_class`, build a
     `Q` per instance from that model's `UniqueConstraint.fields` (read it from `Meta.constraints`
     rather than hard-coding), OR them together, and query. Replace `reduced_datums` with the fetched
     list, then call `data_processor.post_process(dp, reduced_datums)` **before** `continuous_share_data`.
   - Keep the return value semantics (list of persisted datums) — callers in `views.py`/`api_views.py`
     use `len()` on it for the "N datums created" message; with re-fetching that count now includes
     pre-existing rows. **(check: is that acceptable upstream, or should the return stay
     "newly created only" and `post_process` get the full list? Ask in the PR description.)**
   - Document the per-row `data_type` override in the `process_data` docstring: "each dict may include
     a `data_type` key to select the concrete ReducedDatum subclass for that row".
A2. Tests in `tom_dataproducts/tests/`:
   - a processor emitting mixed `astrometry` + `photometry` rows creates one of each model;
   - `post_process` receives instances with PKs, including a pre-existing duplicate;
   - the `_build_*` heuristics still work (regression).
A3. `python manage.py test tom_dataproducts` green; `flake8` clean (config in `.flake8`).
A4. Push to the fork and open the upstream PR. Note the FOMO use case in the description; link the
   audit artifact if useful. Ask before pushing (fork is Tim's, externally visible).

### Part B — FOMO branch (ADES-specific)

Branch: `feature/ades-reduced-datum` off `feature/add_mpc_obs` (rebase onto its merge target once
#51 lands). Work in a scratchpad worktree if `~/git/fomo` is on another branch; copy
`src/fomo/_version.py` into it.

B1. Models. `solsys_code/models.py` — `ADESFrame` and `ADESReducedDatum` per the open-decisions
   default:
   - `ADESFrame`: `obs_time` (DateTimeField), `stn` (CharField 4), `mode`, `prog`, `fltr`, `sys`,
     `ctr`, `pos1..3`, `vel1..3`, `pos_cov11..33`, `exp`, `seeing`, `data_product` FK,
     `value = JSONField(default=dict)`; `UniqueConstraint(stn, obs_time, mode, fltr)`;
     `__str__` → `f'{stn} {obs_time:%Y-%m-%dT%H:%M:%S.%f} {fltr}'`.
   - `ADESReducedDatum`: `frame` FK, `astrometry`/`photometry` OneToOnes (nullable, SET_NULL),
     `target` FK, `obs_id` (nullable, unique when present), `phot_ap`, `value`.
   `python manage.py makemigrations solsys_code`; check the migration in.
B2. Processor, phase 1 (works on stock 3.0.1). In `ADESProcessor._process_astrometry_from_*`:
   - emit **two** dicts per row when `mag` is present: the astrometry dict as today (minus
     `magnitude`/`mag_error`/`filter`, plus `'data_type': 'astrometry'`), and a photometry dict
     `{'data_type': 'photometry', 'magnitude': mag, 'magnitude_error': rmsMag, 'filter': band,
     'telescope': stn, 'exposure_time': exp}`. Verified accepted keys in 3.0.1
     `_build_photometry_reduced_datum`: brightness ∈ `{brightness, magnitude, mag}`, error ∈
     `{error, brightness_error, magnitude_error, mag_err}`, bandpass ∈ `{bandpass, filter, band, f}`;
     `exposure_time`, `limit`, `unit`, `telescope`, `instrument`, `reduction_version` pass straight
     through as model fields (`**data`), so those names must be exact. Anything else lands in `value`;
   - carry the frame-grain fields through on the astrometry dict under a single key, e.g.
     `'ades_frame': {...}`, so they land in `AstrometryReducedDatum.value['ades_frame']` and are
     available to `post_process` without re-parsing the file. Keep `logSNR`, `rmsCorr`, `photAp`,
     `astCat`, `photCat` etc. in `value` as now.
   - update `test_ades_processor.py`: counts double where `mag` is present; the end-to-end
     `run_data_processor` test now asserts both `AstrometryReducedDatum` and `PhotometryReducedDatum`
     rows, with `bandpass == band` and `fltr` **not** collapsed into it.
B3. Processor, phase 2 (needs Part A). Implement `ADESProcessor.post_process(dp, reduced_datums)`:
   - pairing key: `(timestamp, telescope)` is **not** enough (6 apertures per frame in
     `sample.psv`); pair by `(timestamp, telescope, phot_ap)` carried in both dicts' `value`
     (astrometry: `value['photAp']` already; photometry: add it), falling back to
     `(timestamp, telescope)` when `photAp` is absent. `get_or_create` the `ADESFrame` from
     `value['ades_frame']`, then `bulk_create` the `ADESReducedDatum` rows with
     `ignore_conflicts=True` (idempotent on `obs_id`), then link;
   - pop `ades_frame` out of `value` after use (or leave it; decide and document — default: pop, to
     avoid two sources of truth).
   - Guard with `hasattr(super(), 'post_process')` is unnecessary — just define the method; on 3.0.1
     it is simply never called.
   - Tests: frames and link rows created (`sample.psv` → 6 frames, 36 link rows, `phot_ap` set),
     `obs_id` populated from MPC-sourced PSVs (`test_ades_mpcexplorer.psv`), re-ingesting the same
     file creates no duplicates, and a populated Location group round-trips. For the last one add
     `solsys_code/tests/test_data/test_ades_location.psv`: 2–3 `OpticalType` rows copied from
     `test_ades.psv` with `sys|ctr|pos1|pos2|pos3` filled using the values from
     `~/git/ADES-Master/tests/input/319.psv` (`ICRF_KM|399|1833.310|-1882.459|-3409.399`) — a
     geocentric-frame satellite-style record; the numbers only need to be plausible, not real.
B4. Wiring. `pyproject.toml`: bump `tomtoolkit>=` to the release carrying Part A once it exists (or
   a git pin to the fork if pursuing the tom_jpl route). Until then, nothing to change.
B5. UI (minimal). A "Frames" table on the target detail page (`target_detail_buttons`/
   integration-hook pattern from `solsys_code/apps.py`) listing `ADESReducedDatum` rows with their
   linked RA/Dec and mag — enough to see the linking works. Anything richer is a follow-up.
B6. Lint (`ruff check . --fix && ruff format .`), commit by file name (never `-A`), pre-commit must
   pass. Suggested commits: (1) model + migration; (2) processor pair emission + tests; (3)
   `post_process` linking + tests; (4) UI. Add `solsys_code.tests.test_ades_processor` is already
   in the CI Django step (PR #51); add any new test module there too.

## Done when

- tom_base PR open against `dev` adding `DataProcessor.post_process` and documented per-row
  `data_type`, with tests; `tom_dataproducts` suite green.
- FOMO: `ADESFrame` + `ADESReducedDatum` models + migration; `ADESProcessor` emits astrometry +
  photometry pairs and (when `post_process` is available) one `ADESFrame` per exposure and one
  `ADESReducedDatum` per ADES row, with `obs_id`, `phot_ap` and the frame-grain fields populated;
  `sample.psv` yields 6 frames / 36 link rows; `fltr` preserved; re-ingest idempotent; Location
  group round-trips.
- `python manage.py test solsys_code.tests.test_ades_processor solsys_code.tests.test_utils
  solsys_code.tests.test_compare_utils` green; `ruff` clean; pre-commit passes; pushed (ask before
  any force-push).

## Explicitly out of scope

- Reduction table (`astCat band photCat rmsFit nStars subFrm`, keyed by `reduction_version`).
- Residual table (the 18 `NoSubmit` orbit-solution fields).
- Tracklet table / `trkSub` handling beyond storing it in `value`.
- `OccultationType` and `OffsetType` records (`319.psv`-style); they hit the same wall later.
- Upstreaming `ADESProcessor` to tom_base.
- Writing ADES PSV/XML back out from the link model (the snake_case naming keeps the door open).
