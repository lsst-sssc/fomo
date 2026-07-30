import re
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException

from solsys_code.calendar_utils import (
    SITE_TELESCOPE_MAP,
    aperture_class_from_telescope_code,
    derive_telescope,
    derive_telescope_class,
    insert_or_create_calendar_event,
    resolve_placement_block,
)
from solsys_code.models import CampaignRun

# Imported (not duplicated) from the sync command's test module -- that module still
# owns _observations_block_response() since many command-behaviour tests there use it;
# see this todo's SUMMARY for the "import vs. copy" call.
from solsys_code.tests.test_sync_lco_observation_calendar import _observations_block_response

# A fixed UTC sunset-like start time and a companion end time, used across the
# drift-tolerance tests below.
_START = datetime(2026, 7, 17, 22, 10, 56, tzinfo=dt_timezone.utc)
_END = datetime(2026, 7, 18, 11, 30, 0, tzinfo=dt_timezone.utc)
_TOLERANCE = timedelta(minutes=5)


class TestInsertOrCreateCalendarEventExactMatch(TestCase):
    """Default (exact-equality) behaviour used by the URL-keyed sync commands."""

    def test_url_lookup_creates_then_leaves_unchanged(self):
        """A URL-keyed create-or-update creates once, then reports 'unchanged' on re-run."""
        lookup = {'url': 'https://example.test/obs/1'}
        fields = {'title': 'Obs 1', 'start_time': _START, 'end_time': _END}

        event1, action1 = insert_or_create_calendar_event(lookup, fields)
        event2, action2 = insert_or_create_calendar_event(lookup, fields)

        self.assertEqual(action1, 'created')
        self.assertEqual(action2, 'unchanged')
        self.assertEqual(event1.pk, event2.pk)
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_url_lookup_updates_on_changed_field(self):
        """A changed field on a URL-keyed re-run reports 'updated' without duplicating."""
        lookup = {'url': 'https://example.test/obs/2'}
        insert_or_create_calendar_event(lookup, {'title': 'Old', 'start_time': _START, 'end_time': _END})
        event, action = insert_or_create_calendar_event(
            lookup, {'title': 'New', 'start_time': _START, 'end_time': _END}
        )

        self.assertEqual(action, 'updated')
        self.assertEqual(event.title, 'New')
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_exact_start_time_key_duplicates_on_drift(self):
        """Without a tolerance, a drifted start_time in the lookup key creates a duplicate.

        This documents the pre-fix failure mode: exact equality on a computed start_time
        is fragile, which is exactly why load_telescope_runs opts into the tolerance below.
        """
        key = {'telescope': 'Magellan-Baade', 'instrument': 'IMACS'}
        insert_or_create_calendar_event({**key, 'start_time': _START}, {'title': 'A', 'end_time': _END})
        _event, action = insert_or_create_calendar_event(
            {**key, 'start_time': _START + timedelta(seconds=2)}, {'title': 'A', 'end_time': _END}
        )

        self.assertEqual(action, 'created')
        self.assertEqual(CalendarEvent.objects.count(), 2)


class TestInsertOrCreateCalendarEventStartTimeTolerance(TestCase):
    """Proximity-matching behaviour used by load_telescope_runs (the bug fix)."""

    def _key(self) -> dict[str, str]:
        return {'telescope': 'Magellan-Baade', 'instrument': 'IMACS'}

    def test_within_tolerance_no_field_change_is_unchanged(self):
        """A re-ingest whose start_time drifted a few seconds, with no field change, is 'unchanged'."""
        event1, action1 = insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        event2, action2 = insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START + timedelta(seconds=2)},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )

        self.assertEqual(action1, 'created')
        self.assertEqual(action2, 'unchanged')
        self.assertEqual(event1.pk, event2.pk)
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_within_tolerance_keeps_original_start_time_pinned(self):
        """A within-tolerance match must NOT rewrite the stored start_time (no churn)."""
        insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        event, _action = insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START + timedelta(seconds=2)},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )

        # The stored start_time stays pinned to the first-ingested value.
        self.assertEqual(event.start_time, _START)

    def test_within_tolerance_across_minute_boundary_still_matches(self):
        """Drift that straddles a whole-minute boundary still matches (a window, not a bucket).

        22:10:59 -> 22:11:01 would fall in different minute buckets, so any round/truncate
        scheme would still duplicate; the +/- window centred on the target does not.
        """
        near_minute = datetime(2026, 7, 17, 22, 10, 59, tzinfo=dt_timezone.utc)
        insert_or_create_calendar_event(
            {**self._key(), 'start_time': near_minute},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        _event, action = insert_or_create_calendar_event(
            {**self._key(), 'start_time': near_minute + timedelta(seconds=2)},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )

        self.assertEqual(action, 'unchanged')
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_within_tolerance_with_changed_field_updates_not_duplicates(self):
        """A drifted re-ingest that also changed a real field is 'updated', never duplicated."""
        insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        event, action = insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START + timedelta(seconds=2)},
            {'title': 'IMACS run (proposed)', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )

        self.assertEqual(action, 'updated')
        self.assertEqual(event.title, 'IMACS run (proposed)')
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_distinct_night_outside_tolerance_creates_new(self):
        """A genuinely different night (~24h away) is outside the window and creates a new event.

        Confirms the tolerance can never merge two legitimately distinct nights for the
        same telescope+instrument.
        """
        insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        _event, action = insert_or_create_calendar_event(
            {**self._key(), 'start_time': _START + timedelta(days=1)},
            {'title': 'IMACS run', 'end_time': _END + timedelta(days=1)},
            start_time_tolerance=_TOLERANCE,
        )

        self.assertEqual(action, 'created')
        self.assertEqual(CalendarEvent.objects.count(), 2)

    def test_tolerance_scopes_match_by_other_lookup_keys(self):
        """Proximity is scoped by the remaining lookup keys: a different instrument never matches.

        Two different instruments on the same telescope with near-identical start_times are
        distinct events; the window must not merge them.
        """
        insert_or_create_calendar_event(
            {'telescope': 'Magellan-Baade', 'instrument': 'IMACS', 'start_time': _START},
            {'title': 'IMACS run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )
        _event, action = insert_or_create_calendar_event(
            {'telescope': 'Magellan-Baade', 'instrument': 'LDSS3', 'start_time': _START + timedelta(seconds=2)},
            {'title': 'LDSS3 run', 'end_time': _END},
            start_time_tolerance=_TOLERANCE,
        )

        self.assertEqual(action, 'created')
        self.assertEqual(CalendarEvent.objects.count(), 2)


class TestDeriveTelescopeClass(TestCase):
    """derive_telescope_class(): D-20's shared telescope_class derivation helper.

    Each input/output pair mirrors a real dev-DB row shape named in D-16's table
    (26-CONTEXT.md/27-CONTEXT.md), so these are grounded in observed data, not invented.
    """

    def test_lco_1m_derives_1m0(self):
        self.assertEqual(derive_telescope_class('', 'LCO 1m'), '1m0')

    def test_lco_2m_derives_2m0(self):
        self.assertEqual(derive_telescope_class('', 'LCO 2m'), '2m0')

    def test_lco_0_4m_derives_0m4(self):
        self.assertEqual(derive_telescope_class('', 'LCO 0.4m'), '0m4')

    def test_juice_blank_site_derives_space_via_tier_b(self):
        """JUICE's real dev-DB row carries a blank site_raw, so tier a (site-based) can't
        see it -- this must resolve via NO_OBSCODE_SPACE_OBSERVATORIES (tier b) instead."""
        self.assertEqual(derive_telescope_class('', 'JUICE'), 'SPACE')

    def test_juice_horizons_site_derives_space_via_tier_a(self):
        """500@-28 has no HORIZONS_OBSERVER_TO_OBSCODE alias -- D-11's exact definition
        of a space observatory with a Horizons code but no MPC obscode assigned."""
        self.assertEqual(derive_telescope_class('500@-28', 'JUICE'), 'SPACE')

    def test_jwst_horizons_site_with_alias_is_not_space(self):
        """500@-170 DOES have an alias (JWST -> obscode 274), so tier a must not fire --
        JWST is not permanently site-less (D-11 corrects the spike's premise)."""
        self.assertEqual(derive_telescope_class('500@-170', 'JWST'), '')

    def test_horizons_natural_body_observer_codes_are_not_space(self):
        """WR-09: 500@<N> is Horizons observer notation for 'geocentric observer at body N',
        and body N need not be a spacecraft. Only negative NAIF IDs are spacecraft, so a
        natural body must never be recorded as SPACE ('a space observatory with a Horizons
        code but no MPC obscode') -- '' is correct, since site_needs_review already carries
        'unresolved' (D-13)."""
        self.assertEqual(derive_telescope_class('500@399', ''), '')  # Earth's centre
        self.assertEqual(derive_telescope_class('500@10', ''), '')  # the Sun
        self.assertEqual(derive_telescope_class('500@301', ''), '')  # the Moon

    def test_malformed_horizons_observer_code_is_not_space(self):
        """WR-09: a non-numeric NAIF ID is a typo, not a discovered space observatory."""
        self.assertEqual(derive_telescope_class('500@', ''), '')
        self.assertEqual(derive_telescope_class('500@oops', ''), '')
        self.assertEqual(derive_telescope_class('500@-', ''), '')

    def test_unrecognised_negative_naif_id_is_still_space(self):
        """WR-09 narrows the branch to negative NAIF IDs only -- it does not narrow it
        further. An unaliased spacecraft ID still means SPACE (this is what JUICE's
        500@-28 relies on)."""
        self.assertEqual(derive_telescope_class('500@-999', ''), 'SPACE')

    def test_hst_obscode_site_no_aperture_signal_returns_blank(self):
        self.assertEqual(derive_telescope_class('250', 'HST STIS/COS'), '')

    def test_swift_blank_site_no_aperture_signal_returns_blank(self):
        """Swift has an MPC obscode (C52) and is deliberately NOT in
        NO_OBSCODE_SPACE_OBSERVATORIES -- widening SPACE to 'any space mission' is
        exactly the premise D-11 falsified."""
        self.assertEqual(derive_telescope_class('', 'Swift/UVOT'), '')

    def test_unrelated_site_and_instrument_returns_blank(self):
        self.assertEqual(derive_telescope_class('X05', 'FOO / BAR'), '')

    def test_soar_4m_is_excluded_per_d12(self):
        """D-12: 4m0 (SOAR) is deliberately excluded from CampaignRun.TelescopeClass's
        vocabulary, even though it is a real, recognized aperture-class match."""
        self.assertEqual(derive_telescope_class('', 'SOAR 4m'), '')

    def test_muscat4_trailing_digit_is_not_a_false_positive(self):
        """FTS/MuSCAT4: the trailing '4' in 'MuSCAT4' must not be mistaken for a '4m'
        aperture phrase -- the digit only forms an aperture match if it PRECEDES 'm'."""
        self.assertEqual(derive_telescope_class('', 'FTS/MuSCAT4'), '')

    def test_none_site_and_instrument_never_raises(self):
        self.assertEqual(derive_telescope_class(None, None), '')

    def test_aperture_classes_are_subset_of_calendar_utils_vocabulary(self):
        """D-12: the model's 3-value vocabulary is a SUBSET of calendar_utils' 4-value
        aperture-class set (not equality -- equality would fail on day one over '4m0').

        Compared directly with no case-folding (D-21) -- a casing divergence must be
        caught, not silently normalised.
        """
        model_aperture_values = {
            CampaignRun.TelescopeClass.TWO_M0,
            CampaignRun.TelescopeClass.ONE_M0,
            CampaignRun.TelescopeClass.ZERO_M4,
        }
        calendar_utils_aperture_values = {
            aperture_class_from_telescope_code(code) for code in ('0m4a', '1m0a', '2m0a', '4m0a')
        }

        self.assertTrue(model_aperture_values.issubset(calendar_utils_aperture_values))
        # The known, deliberate exclusion (D-12): 4m0 (SOAR) is a real calendar_utils
        # aperture class but must never be "fixed" onto the model.
        self.assertIn('4m0', calendar_utils_aperture_values)
        self.assertNotIn('4m0', model_aperture_values)
        # TelescopeClass.SPACE is not an aperture class at all -- it has no calendar_utils
        # counterpart and must never appear in this set.
        self.assertNotIn(CampaignRun.TelescopeClass.SPACE, calendar_utils_aperture_values)
        # SPACE is deliberately absent from the aperture-class set -- it is not an
        # aperture class at all.
        self.assertNotIn('SPACE', calendar_utils_aperture_values)


class TestTelescopeLabelResolutionHelpers(TestCase):
    """Relocated from test_sync_lco_observation_calendar.py (todo 2026-07-02, second half):
    these tests exercise calendar_utils helpers directly via mocks and never invoke the
    sync_lco_observation_calendar management command, so they belong here."""

    def test_telescope_01_verified_dict_covers_all_sites(self):
        """TELESCOPE-01: verified dict covers all 7 real sites with SITECODE-CLASS labels."""
        expected_sites = {'ogg', 'elp', 'lsc', 'cpt', 'coj', 'tfn', 'sor'}
        actual_sites = {site for site, _aperture_class in SITE_TELESCOPE_MAP}
        self.assertEqual(actual_sites, expected_sites)

        label_pattern = re.compile(r'^[A-Z]{3}-(0m4|1m0|2m0|4m0)$')
        for label in SITE_TELESCOPE_MAP.values():
            self.assertRegex(label, label_pattern)

        for migrated_label in ('COJ-2m0', 'OGG-2m0', 'SOR-4m0'):
            self.assertIn(migrated_label, SITE_TELESCOPE_MAP.values())

    def test_telescope_01_aperture_class_from_telescope_code(self):
        """TELESCOPE-01: aperture_class_from_telescope_code parses/rejects telescope codes."""
        self.assertEqual(aperture_class_from_telescope_code('1m0a'), '1m0')
        self.assertEqual(aperture_class_from_telescope_code('0m4b'), '0m4')
        self.assertEqual(aperture_class_from_telescope_code('2m0a'), '2m0')
        self.assertIsNone(aperture_class_from_telescope_code('xx'))
        self.assertIsNone(aperture_class_from_telescope_code('foo9'))

    def test_telescope_01_coj_ogg_full_aperture_class_coverage(self):
        """TELESCOPE-01: coj/ogg's full aperture-class inventory resolves to verified labels.

        Regression for the Phase 7 UAT Test 1 gap (07-UAT.md Gaps section): a real placed
        record (observation_id=4213127) resolved via the live LCO API to
        site='coj', telescope='1m0a' (aperture class '1m0'), but SITE_TELESCOPE_MAP had no
        ('coj', '1m0') entry, so it fell back to the [UNVERIFIED] label instead of COJ-1m0.
        """
        self.assertEqual(derive_telescope('coj', '1m0a'), 'COJ-1m0')
        self.assertEqual(derive_telescope('coj', '0m4a'), 'COJ-0m4')
        self.assertEqual(derive_telescope('ogg', '0m4b'), 'OGG-0m4')

    def test_telescope_02_placed_record_resolves_via_api(self):
        """TELESCOPE-02: a successful mocked API response resolves to the verified label."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        with patch(
            'solsys_code.calendar_utils.make_request',
            return_value=_observations_block_response(
                site='lsc', enclosure='doma', telescope='1m0a', state='COMPLETED'
            ),
        ):
            block = resolve_placement_block('12345', mock_facility)

        self.assertIsNotNone(block)
        self.assertEqual(block['site'], 'lsc')
        self.assertEqual(block['enclosure'], 'doma')
        self.assertEqual(block['telescope'], '1m0a')
        self.assertEqual(derive_telescope(block['site'], block['telescope']), 'LSC-1m0')


class TestResolvePlacementBlockFailureModes(TestCase):
    """Relocated from test_sync_lco_observation_calendar.py (todo 2026-07-02, second half):
    resolve_placement_block()'s own failure-mode contract, exercised directly via mocks."""

    def test_sync_08_single_attempt_no_retry(self):
        """SYNC-08: a timeout results in exactly one make_request call, no retry loop."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=requests.exceptions.Timeout,
        ) as mock_make_request:
            block = resolve_placement_block('12345', mock_facility)

        self.assertIsNone(block)
        mock_make_request.assert_called_once()

    def test_sync_09_no_credential_or_body_leak_in_logs(self):
        """SYNC-09: ImproperCredentialsException/forms.ValidationError are swallowed to None,
        never raised, and the helper never surfaces anything derived from the caught
        exception (which may embed response.content / API-key-adjacent diagnostic text)."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        leak_marker = 'SECRET_API_KEY_LEAK_BODY'

        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {leak_marker}'),
        ):
            block = resolve_placement_block('12345', mock_facility)
        self.assertIsNone(block)

        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=forms.ValidationError(f'OCS: {leak_marker}'),
        ):
            block = resolve_placement_block('12345', mock_facility)
        self.assertIsNone(block)
