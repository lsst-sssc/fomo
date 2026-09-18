"""Tests for the ``LCO_API_KEY`` settings fold in ``src/fomo/settings.py`` (Phase 36 Plan
08, gap G-36-4).

These tests execute the REAL fold tail of the live settings module -- not a
re-implementation of it -- against a synthetic ``fomo.local_settings`` module injected
into ``sys.modules``, so a future edit that drops the SOAR line fails this test instead
of passing a source-token grep. They pin four behaviors: the flat ``LCO_API_KEY``
setting reaches both the LCO and the SOAR facility ``api_key`` entries; an absent
setting is a clean no-op; the bracketed dict-subscript form the old runbook wrongly
documented raises ``NameError``; and the SOAR accessor the unattended status-refresh
step reaches through ``SOARFacility`` reads the entry the fold fills.

Uses ``django.test.SimpleTestCase`` -- no database row is touched anywhere in this
module, so ``TestCase``'s transaction machinery would only cost time. No ``Target``
fixture is created anywhere in this module, so the ``NonSiderealTargetFactory`` rule in
CLAUDE.md does not arise.
"""

import importlib
import sys
import types

from django.conf import settings as django_settings
from django.test import SimpleTestCase, override_settings
from tom_observations.facilities.soar import SOARSettings

# Non-secret placeholder literal -- never anything key-shaped or UUID-shaped.
_FAKE_LCO_API_KEY = 'fake-portal-key-test-settings-api-key-fold'

# The exact anchor the real settings module's fold tail begins with. Slicing from here
# to end of file captures the try/except import guard, its explanatory comment, and the
# fold itself -- everything this test needs to execute, and nothing that touches a real
# fomo local settings module on disk.
_FOLD_TAIL_ANCHOR = 'try:\n    from fomo.local_settings import *'


class _FoldExecutionTestCase(SimpleTestCase):
    """Shared helper: execute the real settings fold tail against a synthetic injected
    module, with no read of, write to, or dependency on the operator's real local
    settings module."""

    def _run_fold(self, injected_attrs):
        """Execute the live settings module's fold tail in a synthetic namespace.

        Installs a ``types.ModuleType('fomo.local_settings')`` carrying ``injected_attrs``
        into ``sys.modules`` for the duration of the test (restored via ``addCleanup``,
        including the case where no such module was previously registered), slices the
        live settings module's source from ``_FOLD_TAIL_ANCHOR`` to end of file, and
        executes that slice with ``compile(..., 'exec')`` in a namespace pre-seeded with a
        synthetic ``FACILITIES`` dict. The ``from fomo.local_settings import *`` statement
        inside the executed slice resolves against the injected module via the
        ``sys.modules`` cache -- it never touches the real file on disk.

        Returns the namespace's ``FACILITIES`` dict for the caller to assert against.
        """
        settings_module = importlib.import_module(django_settings.SETTINGS_MODULE)
        settings_path = settings_module.__file__

        with open(settings_path) as fh:
            source = fh.read()
        anchor_index = source.find(_FOLD_TAIL_ANCHOR)
        if anchor_index == -1:
            self.fail(f'fold-tail anchor not found in {settings_path}: {_FOLD_TAIL_ANCHOR!r}')
        tail_source = source[anchor_index:]

        fake_module = types.ModuleType('fomo.local_settings')
        for name, value in injected_attrs.items():
            setattr(fake_module, name, value)

        previous_module = sys.modules.get('fomo.local_settings')
        sys.modules['fomo.local_settings'] = fake_module

        def _restore_module():
            if previous_module is None:
                sys.modules.pop('fomo.local_settings', None)
            else:
                sys.modules['fomo.local_settings'] = previous_module

        self.addCleanup(_restore_module)

        namespace = {'FACILITIES': {'LCO': {'api_key': ''}, 'SOAR': {'api_key': ''}}}
        exec(compile(tail_source, settings_path, 'exec'), namespace)  # executing our own settings source
        return namespace['FACILITIES']


class TestFlatKeyReachesBothFacilities(_FoldExecutionTestCase):
    """The documented flat assignment reaches BOTH facility entries -- this is the case
    that would have caught the SOAR half of G-36-4."""

    def test_flat_key_fills_lco_and_soar(self):
        facilities = self._run_fold({'LCO_API_KEY': _FAKE_LCO_API_KEY})
        self.assertEqual(facilities['LCO']['api_key'], _FAKE_LCO_API_KEY)
        self.assertEqual(facilities['SOAR']['api_key'], _FAKE_LCO_API_KEY)


class TestAbsentKeyIsCleanNoop(_FoldExecutionTestCase):
    """An operator who omits the setting entirely gets the documented no-op: the
    presence guard skips, both facility entries stay empty strings, nothing raises."""

    def test_absent_key_leaves_both_entries_empty(self):
        facilities = self._run_fold({})
        self.assertEqual(facilities['LCO']['api_key'], '')
        self.assertEqual(facilities['SOAR']['api_key'], '')


class TestLiveFacilitiesCarriesBothFoldTargets(SimpleTestCase):
    """The namespace ``_run_fold()`` executes into is *seeded* with both facility entries
    (see ``_FoldExecutionTestCase._run_fold``), so it cannot detect the one prerequisite
    the fold actually requires: that ``FACILITIES['SOAR']`` exists in the real settings
    module at all (``src/fomo/settings.py``, the entry above the fold tail). Deleting that
    entry left every test above green while a configured host raised ``KeyError: 'SOAR'``
    at import (WR-29, 36-REVIEW.md iteration 5). This case asserts against the live,
    already-imported settings object instead of the synthetic namespace, so it fails if
    that entry -- or the LCO entry the fold has always required -- is ever removed."""

    def test_live_facilities_has_both_entries_the_fold_writes_into(self):
        from django.conf import settings as live

        for facility in ('LCO', 'SOAR'):
            self.assertIn(facility, live.FACILITIES)
            self.assertIn('api_key', live.FACILITIES[facility])


class TestSoarPortalUrlMatchesLcoBeforeKeyIsCopied(SimpleTestCase):
    """IN-31 (36-REVIEW.md iteration 5): the fold copies the LCO key into the SOAR entry
    unconditionally, and the only thing that makes that safe is that both facilities point
    at the same LCO Observation Portal today. Nothing else ties the two together -- if
    ``FACILITIES['SOAR']['portal_url']`` is ever repointed at a different host (e.g. a
    NOIRLab-hosted SOAR portal) without also removing the fold's SOAR line, this asserts
    the drift instead of letting the LCO key be silently sent to the new host."""

    def test_soar_and_lco_portal_urls_still_match(self):
        self.assertEqual(
            django_settings.FACILITIES['SOAR']['portal_url'],
            django_settings.FACILITIES['LCO']['portal_url'],
        )


class TestFoldTailUsesStarImportIntoOwnNamespace(SimpleTestCase):
    """WR-30 (36-REVIEW.md iteration 5): the case this replaces asserted a property of
    Python itself (``exec("d['k'] = 1", {})`` raises ``NameError`` for ANY empty
    namespace -- true of every Python program ever written) rather than a property of
    this codebase, so it could not fail no matter how the fold tail changed. This case
    instead pins the actual mechanism the runbook's ``NameError`` explanation, and G-36-4's
    failure mode, both rest on: the fold tail imports ``fomo.local_settings`` with a plain
    ``from ... import *`` (so a local settings module can only ASSIGN new names into the
    namespace it executes in -- it cannot mutate ``FACILITIES`` above it), and it does NOT
    use a bare ``except:`` (which would also swallow the ``NameError`` such a mutation
    attempt raises, rather than letting Django refuse to start)."""

    def test_fold_tail_star_imports_and_does_not_use_a_bare_except(self):
        settings_module = importlib.import_module(django_settings.SETTINGS_MODULE)
        settings_path = settings_module.__file__
        with open(settings_path) as fh:
            source = fh.read()
        anchor_index = source.find(_FOLD_TAIL_ANCHOR)
        if anchor_index == -1:
            self.fail(f'fold-tail anchor not found in {settings_path}: {_FOLD_TAIL_ANCHOR!r}')
        tail_source = source[anchor_index:]
        self.assertIn('from fomo.local_settings import *', tail_source)
        self.assertNotRegex(tail_source, r'except\s*:', msg='a bare except would also swallow NameError')


class TestSoarAccessorReadsFoldTarget(SimpleTestCase):
    """The fold target is the key the consumer reads: pins the fold target to the
    accessor the unattended status-refresh step reaches through ``SOARFacility``,
    which is the whole reason the SOAR half of the fold is worth having."""

    def test_soar_settings_get_setting_reads_facilities_soar_api_key(self):
        with override_settings(FACILITIES={'SOAR': {'api_key': _FAKE_LCO_API_KEY}}):
            self.assertEqual(SOARSettings('SOAR').get_setting('api_key'), _FAKE_LCO_API_KEY)
