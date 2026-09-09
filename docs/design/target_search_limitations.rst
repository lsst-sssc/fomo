Target Search Limitations in TOM Toolkit 3.0
============================================

This document records two gaps in the TOM Toolkit 3.0 target search, found while
investigating a target that appeared to be missing from FOMO, and the local
workarounds FOMO carries for them.

Background
----------

On 2026 September 6 an imminent impactor was ingested from JPL Scout under its
NEOCP tracklet designation ``CERNQ52``.  The MPC designated it later the same
day, and ``updatescout``'s MPC pass renamed the Target to ``2026 RW1``, keeping
``CERNQ52`` as a ``TargetName`` alias.

Searching the target list for ``CERNQ52`` then returned nothing, which looked
like an ingest failure.  The target had in fact been ingested correctly, with a
full Scout history; it had simply become unfindable under the designation it was
ingested with, and under which it was discussed for the first several hours of
its visibility.

This is the normal life cycle of a solar system target rather than an edge case.
Anything ingested pre-designation and renamed later has the same problem, and the
old designation is usually the string an operator remembers.

The General Search box ignores aliases
--------------------------------------

``TargetFilterSet.general_search`` (``tom_targets/filters.py:277``) matches the
primary name only::

    return queryset.filter(Q(name__icontains=value))

The same ``TargetFilterSet`` already defines an alias-aware filter,
``filter_name`` at ``tom_targets/filters.py:143-150``::

    q_set |= Q(name__icontains=term) | Q(aliases__name__icontains=term)

but it carries the comment ``# NOTE: this field is not displayed; the 'query'
field is used instead`` and is omitted from the crispy ``Layout``.  The
alias-aware filter still exists and still works through the query string
(``/targets/?name=CERNQ52`` finds the target), but the box that replaced it in
the move to the HTMX filter set dropped alias matching.

Inheriting ``HTMXTableFilterSet.general_search`` would not help: it sweeps model
fields but explicitly skips ``many_to_many``/``many_to_one``, and ``aliases`` is
a reverse foreign key.

The GENERAL_SEARCH_FUNCTIONS key trap
-------------------------------------

``tom_common.htmx_table`` provides a settings hook for exactly this override, so
no subclassing or patching is needed.  Its lookup key is built from the concrete
model::

    model_label = f'{self.Meta.model._meta.app_label}.{self.Meta.model.__name__}'

With no ``TARGET_MODEL_CLASS`` configured, ``Target`` *is*
``tom_targets.base_models.BaseTarget``, so the key is ``tom_targets.BaseTarget``.
The docstring immediately above that line tells developers to write
``'tom_targets.Target'``, which matches nothing.

A key that does not match is **silently ignored**: ``search_map.get(...)``
returns ``None``, the method falls back to the default search, and no error or
warning is raised.  The application boots normally and the search box keeps its
old behaviour.

This cost real time during development.  The lesson worth keeping is that the
test must exercise the filter set end to end -- ``TargetFilterSet(...).qs`` --
rather than calling the search function directly.  A direct-call test passes
against a registration that is never consulted.  See
``test_filterset_dispatches_to_the_configured_function`` in
``solsys_code/tests/test_search.py``.

The Django admin has no search at all
-------------------------------------

``tom_targets/admin.py:12-14`` registers ``Target`` with an effectively empty
``ModelAdmin``.  With no ``search_fields``, Django renders no search box on the
changelist; with no ``list_display`` or ``list_filter``, the page is a single
unlabelled column with no sidebar.  The admin is where one looks when the normal
UI is behaving oddly, which is how this was found in the first place.

FOMO's workarounds
------------------

``solsys_code/search.py``
    ``target_general_search()`` matches ``name`` or ``aliases__name``, with
    ``.distinct()`` -- without it, a target matching on both its name and an
    alias is returned twice by the join.

``src/fomo/settings.py``
    Registers that callable under ``GENERAL_SEARCH_FUNCTIONS``, keyed
    ``tom_targets.BaseTarget``.  If ``TARGET_MODEL_CLASS`` is ever enabled the
    key must change to match the new concrete model.

``solsys_code/admin.py``
    Unregisters the toolkit's ``TargetAdmin`` and re-registers a subclass adding
    ``search_fields``, ``list_display``, ``list_filter`` and a ``TargetName``
    inline.  This relies on ``tom_targets`` preceding ``solsys_code`` in
    ``INSTALLED_APPS`` so the original registration exists to unregister.

An earlier, independent attempt
-------------------------------

Commit ``b572dfb`` ("register custom TargetAdmin with type filter",
2026-07-22) had already added a ``TargetAdmin`` to ``solsys_code/admin.py`` on
``issue37-telescope-runs-calendar``, and it was carried onto
``issue37-code-only`` (via ``5a1f27e``), the branch behind the open draft PR.
It is not on ``main``.  That version declares ``search_fields = ['name']``, so
it restores a search box but not alias matching -- it closes the "no search at
all" half of the problem and leaves the half that hid ``CERNQ52``.  It also
subclasses ``admin.ModelAdmin`` directly rather than the toolkit's
``TargetAdmin``, which drops the ``TargetExtra`` inline that ships with it.

Both branches therefore rewrite ``solsys_code/admin.py`` from main's stub and
both register a ``Target`` admin, so whichever lands second must resolve the
collision.  The version here is the reconciled one and should be preferred.

Its ``list_display`` deliberately does **not** follow ``b572dfb``'s fixed
``['name', 'type', 'ra', 'dec']``.  The two target types have disjoint useful
fields: ``ra``/``dec`` are null on every ``NON_SIDEREAL`` row, and
``scheme``/``abs_mag`` on every ``SIDEREAL`` one, so any single fixed column set
is half blank.

Columns cannot vary per row -- ``list_display`` defines one header row for the
whole table -- but ``get_list_display()`` may vary them per *request*, so they
follow the ``type`` filter that ``list_filter`` already provides.  The default
suits the non-sidereal catalogue, which dominates by count (at the time of
writing 913 rows against 11); filtering to Sidereal swaps in the coordinates.

The sidereal rows are not incidental.  They cover Rubin Science Verification
field centres, sidereal positions for interstellar objects, and
spectrophotometric standards such as CD-32 5216 -- and calibration stars for
spectroscopic follow-up are expected to become a standing part of the
catalogue, not a handful of one-offs.

Upstream status
---------------

Both gaps are TOM Toolkit issues rather than anything FOMO does wrong, and
affect any TOM whose targets carry aliases.  Neither was reported on
``TOMToolkit/tom_base`` at the time of writing.  These workarounds should be
removed once fixes land upstream.

Drafted reports are kept alongside this note, in ``docs/design/upstream/``:
``tom_base_alias_search.md`` covers the General Search box and the
``GENERAL_SEARCH_FUNCTIONS`` key, ``tom_base_target_admin.md`` the admin.  They
are Markdown, for pasting into the upstream tracker, and so are not part of this
rendered documentation.  Record the issue numbers here once filed.
