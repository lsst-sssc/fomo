---
phase: 260717-iae-wire-the-existing-pre-executed-demo-note
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/notebooks.rst
autonomous: true
requirements:
  - DOCS-NOTEBOOKS-TOCTREE
must_haves:
  truths:
    - The five committed, pre-executed demo notebooks appear in the published Notebooks section of the docs (no longer orphaned from any toctree).
    - A full Sphinx HTML build (notebooks NOT excluded) succeeds with exit code 0 and renders each new notebook to HTML.
    - No "toctree contains reference to nonexisting document" warning is emitted for the new entries.
  artifacts:
    - docs/notebooks.rst
  key_links:
    - Each toctree entry path (notebooks/pre_executed/<name>) resolves to the matching docs/notebooks/pre_executed/<name>.ipynb document that nbsphinx renders.
---

<objective>
Wire the five existing, committed, pre-executed demo notebooks under
`docs/notebooks/pre_executed/` into the `docs/notebooks.rst` Sphinx toctree so
they actually appear in the published documentation site. Today `notebooks.rst`
only lists `notebooks/intro_notebook`; the five demo notebooks are read by
nbsphinx but belong to no toctree, so they are invisible in the built docs and
emit "document isn't included in any toctree" warnings on full builds.

Purpose: Make the already-authored demo notebooks discoverable in the rendered
documentation without altering their content.
Output: An updated `docs/notebooks.rst` toctree plus a passing full Sphinx build.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@docs/notebooks.rst
@docs/conf.py
@docs/design/design.rst

Reference facts already confirmed during planning (do NOT re-derive):
- Exact notebook filenames in `docs/notebooks/pre_executed/`:
  `telescope_runs_demo.ipynb`, `load_telescope_runs_demo.ipynb`,
  `sync_lco_observation_calendar_demo.ipynb`,
  `sync_gemini_observation_calendar_demo.ipynb`, `import_campaign_csv_demo.ipynb`.
- toctree paths in `notebooks.rst` are relative to `docs/` and carry NO `.ipynb`
  extension (existing entry: `notebooks/intro_notebook`). So the new entries are
  `notebooks/pre_executed/telescope_runs_demo`, etc.
- `docs/design/design.rst` already demonstrates the exact target pattern: a
  titled section ("Design Notes") followed by its own `.. toctree::` block.
- `conf.py` already registers `nbsphinx` and sets `nbsphinx_allow_errors = True`.
  nbsphinx runs in default `auto` execute mode; because every one of the five
  notebooks has committed cell output, nbsphinx will NOT re-execute them, so the
  build does not import `solsys_code` / trigger the ~1.6 GB SPICE kernel
  download. `autoapi_type = 'python'` parses via AST (no import), so it does not
  trigger SPICE either. A full build is therefore light and safe.
- `conf.py` sets `suppress_warnings = ['toc.excluded']`. The pre-commit
  "Build documentation with Sphinx" hook passes
  `-D exclude_patterns=notebooks/*,_build`, which EXCLUDES all notebooks; that
  suppression already covers the excluded-document toctree warning for the new
  entries too, so the pre-commit hook stays clean after this change. Because the
  pre-commit build excludes notebooks, it does NOT actually exercise the new
  toctree entries — the meaningful verification is a FULL build (no notebook
  exclusion), mirroring CI / ReadTheDocs.
- `docs/_build/` is gitignored; `sphinx-build` is on PATH.
- This is a docs-wiring change only. It does NOT change the behavior of any
  demo-notebook companion module, so the CLAUDE.md "demo notebook companions"
  rule requires no notebook content edits here. Do NOT modify notebook content.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add the five pre-executed demo notebooks to the notebooks.rst toctree</name>
  <files>docs/notebooks.rst</files>
  <action>
Edit `docs/notebooks.rst` to surface the five pre-executed demo notebooks in the
published Notebooks page. Keep the existing top-level "Notebooks" title and the
existing `.. toctree::` entry `Introducing Jupyter Notebooks <notebooks/intro_notebook>`
untouched.

Below the existing intro toctree, add a new subsection titled
"Demonstration Notebooks" (an RST section heading, underlined to a level nested
under the page title — follow the section-then-toctree pattern used by the
"Design Notes" section in `docs/design/design.rst`), followed by its own
`.. toctree::` block containing these five entries, each as `Title <path>` with
paths relative to `docs/` and NO `.ipynb` extension, in this order:
  - `Telescope Runs (site / ephemeris helper) <notebooks/pre_executed/telescope_runs_demo>`
  - `Loading Telescope Runs <notebooks/pre_executed/load_telescope_runs_demo>`
  - `Syncing the LCO Observation Calendar <notebooks/pre_executed/sync_lco_observation_calendar_demo>`
  - `Syncing the Gemini Observation Calendar <notebooks/pre_executed/sync_gemini_observation_calendar_demo>`
  - `Importing a Campaign CSV <notebooks/pre_executed/import_campaign_csv_demo>`

Preserve the file's existing indentation convention (the existing toctree entry
is indented four spaces under `.. toctree::`). Do NOT add or change any Sphinx
extension config, and do NOT edit `docs/conf.py` — the existing nbsphinx setup
already renders these notebooks; only the toctree wiring is missing. Do NOT
touch any `.ipynb` file.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && rm -rf docs/_build/notebook_wiring_verify && sphinx-build -M html ./docs ./docs/_build/notebook_wiring_verify -T -E -n 2>&1 | tee /tmp/claude-1000/-home-tlister-git-fomo-devel/6eed0a8a-08f5-47c3-921b-ac7c86c78467/scratchpad/sphinx_verify.log; rc=${PIPESTATUS[0]}; test "$rc" -eq 0 || { echo "BUILD FAILED rc=$rc"; exit 1; }; ! grep -Ei "toctree contains reference to (nonexisting|excluded) document.*pre_executed" /tmp/claude-1000/-home-tlister-git-fomo-devel/6eed0a8a-08f5-47c3-921b-ac7c86c78467/scratchpad/sphinx_verify.log || { echo "TOCTREE WARNING for new entries"; exit 1; }; for n in telescope_runs_demo load_telescope_runs_demo sync_lco_observation_calendar_demo sync_gemini_observation_calendar_demo import_campaign_csv_demo; do test -f "./docs/_build/notebook_wiring_verify/html/notebooks/pre_executed/$n.html" || { echo "MISSING RENDERED HTML: $n"; exit 1; }; done; echo "OK: full build clean, all five notebooks rendered"</automated>
  </verify>
  <done>
`docs/notebooks.rst` contains the existing `notebooks/intro_notebook` entry plus
a "Demonstration Notebooks" section whose toctree lists all five
`notebooks/pre_executed/*` demo notebooks. A full `sphinx-build -M html` (no
notebook exclusion) exits 0, emits no toctree warning for the new entries, and
produces `notebooks/pre_executed/<name>.html` for each of the five notebooks.
No `.ipynb` file and no `docs/conf.py` line is modified.
  </done>
</task>

</tasks>

<verification>
Full documentation build (mirrors CI / ReadTheDocs, which do NOT exclude
notebooks) must succeed and render the new entries:

- `sphinx-build -M html ./docs ./docs/_build/notebook_wiring_verify -T -E -n`
  exits 0.
- No "toctree contains reference to nonexisting/excluded document" warning
  mentions a `pre_executed` path.
- `docs/_build/notebook_wiring_verify/html/notebooks/pre_executed/<name>.html`
  exists for all five notebooks.

The pre-commit "Build documentation with Sphinx" hook (which passes
`-D exclude_patterns=notebooks/*,_build`) continues to pass unchanged, because
`conf.py`'s `suppress_warnings = ['toc.excluded']` already covers the new
entries while notebooks are excluded on that build.
</verification>

<success_criteria>
- `docs/notebooks.rst` wires all five `notebooks/pre_executed/*` demo notebooks
  into a toctree, in addition to the untouched `notebooks/intro_notebook` entry.
- Full Sphinx build passes and renders each notebook to HTML (verified above).
- Only `docs/notebooks.rst` is modified; no notebook content and no `conf.py`
  changes.
</success_criteria>

<output>
Create `.planning/quick/260717-iae-wire-the-existing-pre-executed-demo-note/260717-iae-SUMMARY.md` when done
</output>