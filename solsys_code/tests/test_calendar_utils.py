import re
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.calendar_utils import (
    OBSERVED_TELESCOPE_SITE_CODES,
    SITE_TELESCOPE_MAP,
    aperture_class_from_telescope_code,
    coerce_schedule_datetime,
    derive_telescope,
    derive_telescope_class,
    extract_instrument,
    insert_or_create_calendar_event,
    preview_calendar_event_action,
    record_time_window,
    resolve_placement_block,
    update_calendar_event_key_and_fields,
)
from solsys_code.models import CampaignRun

# IN-02: imported (not duplicated) from the shared fixture module, not from the retired
# LCO/SOAR sync command's own test module -- importing one test module from another meant
# any import-time failure over there also failed this module, for a helper unrelated to the
# sync command.
from solsys_code.tests.helpers import observations_block_response

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
    """Relocated from the retired LCO/SOAR sync command's own test module (todo 2026-07-02,
    second half): these tests exercise calendar_utils helpers directly via mocks and never
    invoke a management command, so they belong here."""

    def test_telescope_01_verified_dict_covers_all_sites(self):
        """TELESCOPE-01: verified dict covers all 7 real sites; D-07 (34-02 Task 3) renamed
        the three 2m0/4m0 entries to the telescope's own operating name instead of the
        SITECODE-CLASS form."""
        expected_sites = {'ogg', 'elp', 'lsc', 'cpt', 'coj', 'tfn', 'sor'}
        actual_sites = {site for site, _aperture_class in SITE_TELESCOPE_MAP}
        self.assertEqual(actual_sites, expected_sites)

        observed_telescope_labels = frozenset(OBSERVED_TELESCOPE_SITE_CODES)
        label_pattern = re.compile(r'^[A-Z]{3}-(0m4|1m0|2m0|4m0)$')
        for label in SITE_TELESCOPE_MAP.values():
            self.assertTrue(
                label_pattern.match(label) or label in observed_telescope_labels,
                f'{label!r} matches neither the SITECODE-CLASS pattern nor an observed-telescope label',
            )

        for renamed_label in ('FTN', 'FTS', 'SOAR'):
            self.assertIn(renamed_label, SITE_TELESCOPE_MAP.values())

        self.assertEqual(set(OBSERVED_TELESCOPE_SITE_CODES), {'FTN', 'FTS', 'SOAR'})
        for label, site in OBSERVED_TELESCOPE_SITE_CODES.items():
            self.assertIn(site, actual_sites, f'{label!r} maps to {site!r}, not a key present in the map')

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

    def test_telescope_01_d07_renamed_observed_telescope_labels(self):
        """D-07 (34-02 Task 3): the three 2m0/4m0 entries resolve to the telescope's own
        operating name, not the SITECODE-CLASS form; the rest of the network is unaffected."""
        self.assertEqual(derive_telescope('ogg', '2m0a'), 'FTN')
        self.assertEqual(derive_telescope('coj', '2m0a'), 'FTS')
        self.assertEqual(derive_telescope('sor', '4m0a'), 'SOAR')
        self.assertEqual(derive_telescope('lsc', '1m0a'), 'LSC-1m0')

    def test_telescope_02_placed_record_resolves_via_api(self):
        """TELESCOPE-02: a successful mocked API response resolves to the verified label."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        with patch(
            'solsys_code.calendar_utils.make_request',
            return_value=observations_block_response(site='lsc', enclosure='doma', telescope='1m0a', state='COMPLETED'),
        ):
            block = resolve_placement_block('12345', mock_facility)

        self.assertIsNotNone(block)
        self.assertEqual(block['site'], 'lsc')
        self.assertEqual(block['enclosure'], 'doma')
        self.assertEqual(block['telescope'], '1m0a')
        self.assertEqual(derive_telescope(block['site'], block['telescope']), 'LSC-1m0')


class TestResolvePlacementBlockFailureModes(TestCase):
    """Relocated from the retired LCO/SOAR sync command's own test module (todo 2026-07-02,
    second half): resolve_placement_block()'s own failure-mode contract, exercised directly
    via mocks."""

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


class TestCoerceScheduleDatetime(SimpleTestCase):
    """coerce_schedule_datetime() (G-34-2) -- no database rows needed; every case is a pure
    function of its input value."""

    def test_trailing_z_form_returns_the_aware_utc_instant(self):
        """The portal's own form: a trailing 'Z' with no explicit offset."""
        result = coerce_schedule_datetime('2026-09-18T07:14:00Z')
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))

    def test_plus_zero_offset_form_returns_the_same_instant(self):
        """The equivalent '+00:00' offset form parses to the same instant as trailing-Z."""
        result = coerce_schedule_datetime('2026-09-18T07:14:00+00:00')
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))

    def test_non_utc_offset_form_equals_the_same_instant(self):
        """A non-UTC ('-04:00') offset form still resolves to the same absolute instant."""
        result = coerce_schedule_datetime('2026-09-18T03:14:00-04:00')
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))

    def test_naive_iso_string_is_read_as_utc(self):
        """A naive ISO string with no offset is read as UTC -- the convention this module
        already documents for parameters['start']/['end']."""
        result = coerce_schedule_datetime('2026-09-18T07:14:00')
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))

    def test_aware_non_utc_datetime_is_converted_to_utc_with_the_same_instant(self):
        """CR-01: an already-aware, non-UTC-offset datetime is converted to UTC -- the
        instant is preserved even though the wall-clock fields and tzinfo change. This used
        to pass through untouched, which rendered the wrong wall clock under the
        'Window (UTC):' label downstream (see test_observation_projector.py's regression
        test for that consumer)."""
        aware = datetime(2026, 9, 18, 3, 14, 0, tzinfo=dt_timezone(timedelta(hours=-4)))
        result = coerce_schedule_datetime(aware)
        self.assertEqual(result, aware)
        self.assertIs(result.tzinfo, dt_timezone.utc)
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))

    def test_naive_datetime_gets_utc_attached_with_the_same_wall_clock_fields(self):
        """A naive datetime comes back with UTC attached and identical wall-clock fields."""
        naive = datetime(2026, 9, 18, 7, 14, 0)
        result = coerce_schedule_datetime(naive)
        self.assertEqual(result, datetime(2026, 9, 18, 7, 14, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result.tzinfo, dt_timezone.utc)

    def test_none_returns_none(self):
        """None in, None out."""
        self.assertIsNone(coerce_schedule_datetime(None))

    def test_unparseable_string_raises_value_error(self):
        """A string that is not a timestamp at all raises ValueError naming the rejected
        value, so the message stays diagnostic."""
        with self.assertRaisesRegex(ValueError, re.escape(repr('not-a-timestamp'))):
            coerce_schedule_datetime('not-a-timestamp')

    def test_non_string_non_datetime_value_raises_value_error(self):
        """WR-04: the elif not isinstance(value, datetime) raise branch -- previously
        uncovered -- for both a bare int (epoch-seconds-shaped) and a `date` (no time
        component)."""
        with self.assertRaisesRegex(ValueError, 'Unusable schedule datetime value'):
            coerce_schedule_datetime(1758000000)
        with self.assertRaisesRegex(ValueError, 'Unusable schedule datetime value'):
            coerce_schedule_datetime(date(2026, 9, 18))

    def test_bare_iso_date_string_raises_value_error(self):
        """WR-03: a bare ISO date string (no time component) is rejected rather than
        silently accepted as midnight -- a schedule field is a block boundary, not a day."""
        with self.assertRaisesRegex(ValueError, 'Schedule value is a date, not a datetime'):
            coerce_schedule_datetime('2026-09-18')


class TestRecordTimeWindow(TestCase):
    """record_time_window() -- promoted from the retired LCO/SOAR sync command's own
    _time_window() (Plan 28-02 Task 2) so the matcher (campaign_attribution.py) and every
    calendar-writing consumer share one definition. Covers both branches; the
    parameters-fallback branch is the common case for real LCO orphan records (NULL
    scheduled_start), not an edge case (RESEARCH.md)."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()
        cls.user = User.objects.create(username='record-time-window-owner')

    def test_scheduled_pair_populated_returns_scheduled_times(self):
        """When both scheduled_start/scheduled_end are set, they are returned verbatim."""
        start = datetime(2026, 7, 10, 22, 0, tzinfo=dt_timezone.utc)
        end = datetime(2026, 7, 11, 6, 0, tzinfo=dt_timezone.utc)
        record = ObservationRecord.objects.create(
            target=self.target,
            user=self.user,
            facility='LCO',
            observation_id='333333',
            status='COMPLETED',
            parameters={'proposal': 'TEST'},
            scheduled_start=start,
            scheduled_end=end,
        )

        result_start, result_end = record_time_window(record)

        self.assertEqual(result_start, start)
        self.assertEqual(result_end, end)

    def test_both_scheduled_none_falls_back_to_parameters_start_end(self):
        """The common real-data case: NULL scheduled_start/end falls back to the naive-UTC
        ISO strings in parameters['start']/['end']."""
        record = ObservationRecord.objects.create(
            target=self.target,
            user=self.user,
            facility='LCO',
            observation_id='444444',
            status='PENDING',
            parameters={'start': '2026-07-10T22:00:00', 'end': '2026-07-11T06:00:00'},
        )

        result_start, result_end = record_time_window(record)

        self.assertEqual(result_start, datetime(2026, 7, 10, 22, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result_end, datetime(2026, 7, 11, 6, 0, tzinfo=dt_timezone.utc))

    def test_both_scheduled_none_falls_back_to_z_suffixed_parameters_start_end(self):
        """CR-02: real records ingested via backfill_lco_observations store the portal's
        'Z'-suffixed window string verbatim in parameters['start']/['end'] -- not only the
        naive '.isoformat()' form the sibling test above covers.
        `datetime.fromisoformat()` rejects a trailing 'Z' before Python 3.11, so this branch
        must route through the same `coerce_schedule_datetime()` parser the
        scheduled_start/scheduled_end branch uses."""
        record = ObservationRecord.objects.create(
            target=self.target,
            user=self.user,
            facility='LCO',
            observation_id='444445',
            status='PENDING',
            parameters={'start': '2026-07-20T00:00:00Z', 'end': '2026-07-20T23:59:59Z'},
        )

        result_start, result_end = record_time_window(record)

        self.assertEqual(result_start, datetime(2026, 7, 20, 0, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result_end, datetime(2026, 7, 20, 23, 59, 59, tzinfo=dt_timezone.utc))

    def test_in_memory_instance_with_portal_iso_strings_returns_aware_utc_pair(self):
        """G-34-2: the post-save-instance case, not a database row --
        update_observation_status() assigns the portal's raw ISO strings onto
        scheduled_start/scheduled_end and calls save(), so record_time_window() must
        coerce the in-memory string the same way a DB-fetched datetime would.

        WR-10: constructed without ``.objects.create()`` so no INSERT happens and no
        post_save signal fires -- this test's name and docstring claim "not a database
        row", and until now it used a real ``.create()`` call that also fired the
        projector's post_save receiver (silently swallowed since the fixture's parameters
        carry no instrument signal). Building the instance directly is what actually pins
        the in-memory contract."""
        start = datetime(2026, 7, 10, 22, 0, tzinfo=dt_timezone.utc)
        end = datetime(2026, 7, 11, 6, 0, tzinfo=dt_timezone.utc)
        record = ObservationRecord(
            target=self.target,
            user=self.user,
            facility='LCO',
            observation_id='555555',
            status='COMPLETED',
            parameters={'proposal': 'TEST'},
            scheduled_start=start.isoformat().replace('+00:00', 'Z'),
            scheduled_end=end.isoformat().replace('+00:00', 'Z'),
        )

        result_start, result_end = record_time_window(record)

        self.assertEqual(result_start, start)
        self.assertEqual(result_end, end)


class TestUpdateCalendarEventKeyAndFields(TestCase):
    """Phase 29 Task 1: the D-02 re-key helper -- writes url as a field, no-churn."""

    def test_rekey_blank_url_to_run_form_reports_updated_and_persists(self):
        """Re-keying an event's url from blank to a RUN:-form string reports 'updated' and the
        reloaded row carries the new url."""
        event = CalendarEvent.objects.create(title='Classical night', url='', start_time=_START, end_time=_END)

        result_event, action = update_calendar_event_key_and_fields(
            event, 'RUN:1:2026-08-01', {'title': 'Classical night'}
        )

        self.assertEqual(action, 'updated')
        self.assertEqual(result_event.url, 'RUN:1:2026-08-01')
        reloaded = CalendarEvent.objects.get(pk=event.pk)
        self.assertEqual(reloaded.url, 'RUN:1:2026-08-01')

    def test_rekey_second_call_with_identical_url_and_fields_is_unchanged(self):
        """Calling it again with the identical url+fields reports 'unchanged' and leaves
        `modified` untouched (no-churn contract)."""
        event = CalendarEvent.objects.create(title='Classical night', url='', start_time=_START, end_time=_END)
        update_calendar_event_key_and_fields(event, 'RUN:1:2026-08-01', {'title': 'Classical night'})
        event.refresh_from_db()
        modified_before = event.modified

        result_event, action = update_calendar_event_key_and_fields(
            event, 'RUN:1:2026-08-01', {'title': 'Classical night'}
        )

        self.assertEqual(action, 'unchanged')
        self.assertEqual(result_event.modified, modified_before)


class TestPreviewCalendarEventAction(TestCase):
    """Phase 29 Task 1: the --dry-run counterpart of the two no-churn writers (D-05/RECON-06)."""

    def test_none_event_returns_created_and_writes_nothing(self):
        """A None event (no existing row) previews as 'created' and issues no write."""
        count_before = CalendarEvent.objects.count()

        action = preview_calendar_event_action(None, {'title': 'New event'})

        self.assertEqual(action, 'created')
        self.assertEqual(CalendarEvent.objects.count(), count_before)

    def test_differing_field_returns_updated_and_writes_nothing(self):
        """A field that differs from the stored value previews as 'updated' without saving."""
        event = CalendarEvent.objects.create(title='Old title', url='', start_time=_START, end_time=_END)
        modified_before = event.modified

        action = preview_calendar_event_action(event, {'title': 'New title'})

        self.assertEqual(action, 'updated')
        reloaded = CalendarEvent.objects.get(pk=event.pk)
        self.assertEqual(reloaded.title, 'Old title')
        self.assertEqual(reloaded.modified, modified_before)
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_identical_fields_returns_unchanged_and_writes_nothing(self):
        """Fields identical to the stored values preview as 'unchanged' without saving."""
        event = CalendarEvent.objects.create(title='Same title', url='', start_time=_START, end_time=_END)
        modified_before = event.modified
        count_before = CalendarEvent.objects.count()

        action = preview_calendar_event_action(event, {'title': 'Same title'})

        self.assertEqual(action, 'unchanged')
        reloaded = CalendarEvent.objects.get(pk=event.pk)
        self.assertEqual(reloaded.modified, modified_before)
        self.assertEqual(CalendarEvent.objects.count(), count_before)


class TestExtractInstrument(TestCase):
    """Migrated from the retired LCO/SOAR sync command's own test module (34-02 Task 2,
    EXTRACT-02/D-01..D-06): extract_instrument() is exercised directly against the real
    c_1..c_5 multi-configuration parameter shape, with no command/facility fixtures needed."""

    def test_soar_multi_config_picks_spectrum_not_calibration(self):
        """EXTRACT-02: a SOAR SPECTRUM+ARC+LAMP_FLAT record extracts the SPECTRUM config's
        instrument_type, never the ARC/LAMP_FLAT calibration configs."""
        parameters = {
            'instrument_type': 'NOT-THE-SOURCE',
            'c_1_configuration_type': 'SPECTRUM',
            'c_1_instrument_type': 'SOAR_GHTS_REDCAM',
            'c_2_configuration_type': 'ARC',
            'c_2_instrument_type': 'SOAR_GHTS_REDCAM_ARC',
            'c_3_configuration_type': 'LAMP_FLAT',
            'c_3_instrument_type': 'SOAR_GHTS_REDCAM_LAMPFLAT',
        }
        result = extract_instrument(parameters)
        self.assertEqual(result, 'SOAR_GHTS_REDCAM')
        self.assertNotEqual(result, 'SOAR_GHTS_REDCAM_ARC')
        self.assertNotEqual(result, 'SOAR_GHTS_REDCAM_LAMPFLAT')

    def test_muscat_per_channel_exposure_extracts_instrument(self):
        """EXTRACT-02/D-04: an LCO MUSCAT record with only per-channel exposure keys (no flat
        c_N_exposure_time) extracts its instrument_type without raising/empty; fewer than 4
        populated channels still extracts correctly (D-04 leniency)."""
        full_channels = {
            'instrument_type': 'NOT-THE-SOURCE',
            'c_1_configuration_type': 'EXPOSE',
            'c_1_instrument_type': '2M0-SCICAM-MUSCAT',
            'c_1_ic_1_exposure_time_g': 30.0,
            'c_1_ic_1_exposure_time_r': 30.0,
            'c_1_ic_1_exposure_time_i': 30.0,
            'c_1_ic_1_exposure_time_z': 30.0,
        }
        self.assertEqual(extract_instrument(full_channels), '2M0-SCICAM-MUSCAT')

        one_channel = {
            'instrument_type': 'NOT-THE-SOURCE',
            'c_1_configuration_type': 'EXPOSE',
            'c_1_instrument_type': '2M0-SCICAM-MUSCAT',
            'c_1_ic_1_exposure_time_g': 30.0,
        }
        self.assertEqual(extract_instrument(one_channel), '2M0-SCICAM-MUSCAT')

    def test_no_recognized_config_and_no_flat_key_returns_none(self):
        """D-06: a fully-malformed record (no recognized configuration_type, no exposure
        signal anywhere, no flat instrument_type) returns None -- the caller
        (observation_projector.event_fields_for) is what raises InstrumentExtractionError
        and routes it to the 'unprojectable' bucket, already proven by the sweep's generic
        failure-isolation tests (test_project_observation_calendar.py)."""
        parameters = {
            'c_1_configuration_type': 'ARC',
            'c_1_instrument_type': 'SOMETHING',
            'instrument_type': None,
        }
        self.assertIsNone(extract_instrument(parameters))
