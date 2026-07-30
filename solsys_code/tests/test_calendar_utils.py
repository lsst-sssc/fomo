from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.test import TestCase
from tom_calendar.models import CalendarEvent

from solsys_code.calendar_utils import (
    aperture_class_from_telescope_code,
    derive_telescope_class,
    insert_or_create_calendar_event,
)

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

        CampaignRun.TelescopeClass does not exist until Plan 27-04, so this asserts
        against the literal expected value set for now.
        # Plan 27-04 wires this to CampaignRun.TelescopeClass
        """
        model_aperture_values = {'2m0', '1m0', '0m4'}
        calendar_utils_aperture_values = {
            aperture_class_from_telescope_code(code) for code in ('0m4a', '1m0a', '2m0a', '4m0a')
        }

        self.assertTrue(model_aperture_values.issubset(calendar_utils_aperture_values))
        # The known, deliberate exclusion (D-12): 4m0 (SOAR) is a real calendar_utils
        # aperture class but must never be "fixed" onto the model.
        self.assertIn('4m0', calendar_utils_aperture_values)
        self.assertNotIn('4m0', model_aperture_values)
        # SPACE is deliberately absent from the aperture-class set -- it is not an
        # aperture class at all.
        self.assertNotIn('SPACE', calendar_utils_aperture_values)
