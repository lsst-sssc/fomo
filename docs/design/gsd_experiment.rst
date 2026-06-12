GSD Experiment: Spec-Driven Feature Development
===============================================

This document records an assessment (2026-06-11) of using GSD as an experiment
to design and build a new FOMO feature from scratch, and the FOMO-specific
considerations to plan around before trying it.

What GSD is
-----------

There are two closely related GSD projects:

* `get-shit-done <https://github.com/gsd-build/get-shit-done>`_ by TÂCHES — the
  original (~54K stars as of June 2026).
* `gsd-core <https://github.com/open-gsd/gsd-core>`_ from
  `Open GSD <https://opengsd.net/products/gsd-core>`_ — a multi-runtime open
  fork whose motto is "Git. Ship. Done."

Both implement the same idea: a disciplined **discuss → plan → execute →
verify → ship** loop per phase, where heavy research, planning, and execution
run in fresh-context subagents (avoiding "context rot" as an AI session's
context window fills), and structured artifacts (``STATE.md``, ``CONTEXT.md``,
per-phase task plans) carry memory across sessions.  Each task gets an atomic
git commit.

Why FOMO is a good testbed
--------------------------

* **A ready-made spec exists.**  The telescope-runs calendar design doc is
  committed (issue #37, nothing built yet, Stage 1 = ephemeris helper).  GSD's
  discuss/plan phase normally has to extract a spec from conversation; feeding
  it a finished design doc tests how faithfully the system decomposes and
  executes a spec, rather than testing one's ability to articulate
  requirements on the fly.

* **The feature is multi-stage by design.**  Staged work (ephemeris helper →
  calendar model → views) maps directly onto GSD's phase loop, which is where
  the system earns its keep.  A one-file feature would not be a meaningful
  test.

* **FOMO's work pattern matches GSD's pitch.**  Development happens across
  many sessions over weeks; GSD's persistent artifacts are built exactly for
  surviving session boundaries, the thing plain AI-assistant sessions are
  worst at.

* **Solo-project benefit:** the mandatory verify step is a stand-in for the
  code reviewer the project does not have.

FOMO-specific friction to plan around
-------------------------------------

* **The ~1.6 GB SPICE kernel import side effect.**  Any subagent that runs
  ``./manage.py test`` (or imports ``solsys_code.ephem_utils``) triggers
  ``fomo_furnish_spiceypy()``.  The download is cached in ``~/.cache/sorcha/``
  so it is a one-time cost per machine, but the first verify phase will be
  slow, and the ASSIST ephemeris build cost recurs.  Warm the cache before
  starting.

* **The two-test-suite split.**  Fresh-context subagents will reflexively run
  ``python -m pytest``, which does not collect the ``solsys_code/`` Django app
  tests.  ``CLAUDE.md`` documents this and GSD subagents do read project
  instructions, but each task plan should explicitly name which suite to run —
  this is the most likely silent failure mode of the experiment.

* **Pre-commit cost × atomic commits.**  The pre-commit hooks run ruff, a
  Sphinx build, and the pytest suite on every commit.  GSD commits per-task,
  so a 12-task phase pays that hook chain 12 times.  Tolerable, but expect it.

* **Start clean.**  Run the experiment on a fresh branch off ``main`` so GSD's
  commit stream does not tangle with in-flight work (e.g. the tomtoolkit 3.0
  migration).  Decide up front whether GSD's planning artifacts (its
  ``.gsd/``/planning directory) get committed or gitignored.

Model choice on a Claude Pro plan
---------------------------------

(Assessed 2026-06-12.)  GSD assigns models per phase via **model profiles** —
six slots (planning / discuss / research / execution / verification /
completion) accepting tier aliases (``opus``, ``sonnet``, ``haiku``,
``inherit``).  The built-in profiles are ``quality`` (all Opus), ``balanced``
(Opus for planning only, Sonnet elsewhere — GSD's default), ``budget`` (Sonnet
for code, Haiku for research/verification), and ``adaptive``.

What a Pro plan provides (per the official
`Claude Code model docs <https://code.claude.com/docs/en/model-config>`_):
the tier default is **Sonnet 4.6** (Max/API accounts default to Opus 4.8).
Opus access on Pro is limited at best and drains the 5-hour usage window
several times faster than Sonnet; Opus with 1M context requires extra usage
credits.  **Fable 5 is effectively out of reach on Pro** — it is not the
default on any plan, is priced above Opus tier, and the ``best`` alias falls
back to the latest Opus where Fable access is absent.

Recommendation for the GSD experiment on Pro:

* **Sonnet 4.6 as the workhorse.**  GSD's fresh-context subagents with small,
  atomic task plans are exactly the regime where Sonnet performs closest to
  Opus; the binding constraint on Pro is token *volume* (every phase spawns
  multiple subagents), not per-task intelligence.

* **Profile = ``balanced`` if Opus works on the account** — spend Opus only on
  the planning agent, where the issue #37 design doc gets decomposed into task
  plans (the one step where extra reasoning compounds).  Claude Code's
  ``opusplan`` alias is the same idea at the harness level.

* **Fall back to ``budget`` if Opus is unavailable or limits bite.**  The
  verify phase is mostly "run the right test suite and read the output", which
  does not need Opus.

* **Avoid ``quality`` (all-Opus) on Pro** — it would exhaust a usage window
  mid-phase, and a GSD run interrupted by rate limits is worse than one run on
  Sonnet throughout.

* Leave Sonnet 4.6's effort parameter at its default ``high``; ``max`` is
  session-only and token-hungry.

Recommendation
--------------

Do it, but scope the first run to **Stage 1 (the ephemeris helper) only**.
That is a self-contained unit with a clear spec, it touches the gnarliest part
of the codebase (``ephem_utils``), and it will surface all the friction points
above cheaply.  If GSD handles the test-suite split and the heavy-import quirk
gracefully on Stage 1, scale it to the full calendar feature; if it stumbles,
the lesson — where its fresh-context model breaks on a repo with non-obvious
conventions — costs only one small phase.

References
----------

* `gsd-build/get-shit-done <https://github.com/gsd-build/get-shit-done>`_
* `GSD User Guide <https://github.com/gsd-build/get-shit-done/blob/main/docs/USER-GUIDE.md>`_
* `open-gsd/gsd-core <https://github.com/open-gsd/gsd-core>`_
* `Open GSD — Git. Ship. Done. <https://opengsd.net/products/gsd-core>`_
* `Augment Code on GSD <https://www.augmentcode.com/learn/gsd-stars-spec-driven>`_
* `Claude Code model configuration <https://code.claude.com/docs/en/model-config>`_
* `GSD model profiles and cost optimization <https://deepwiki.com/gsd-build/get-shit-done/8.9-model-profiles-and-cost-optimization>`_
* `Configuring GSD model profiles <https://docs.bswen.com/blog/2026-04-21-gsd-model-profiles/>`_
