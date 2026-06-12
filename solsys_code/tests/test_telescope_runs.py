from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from astropy.time import Time
from django.test import TestCase

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import SITES, _find_crossing, _local_noon_utc, get_site, horizon_dip, sun_event

# LCO skycalc reference sunset/sunrise (UTC) for Las Campanas, June 2026.
# June 10 is the design-doc-anchored reference (sunset 21:59 UTC / sunrise
# 11:25 UTC, matching the -18 degree twilight cross-check below to the
# second). June 1/20/30 are validated via internal seasonal-consistency
# (smooth day-to-day drift toward the June solstice) per RESEARCH.md Open
# Question 1's internal-consistency fallback (user-approved 2026-06-12).
LAS_CAMPANAS_SUN_REFERENCE_UTC = {
    date(2026, 6, 1): ('2026-06-01T21:59:00', '2026-06-02T11:22:00'),
    date(2026, 6, 10): ('2026-06-10T21:59:00', '2026-06-11T11:25:00'),
    date(2026, 6, 20): ('2026-06-20T22:00:00', '2026-06-21T11:29:00'),
    date(2026, 6, 30): ('2026-06-30T22:03:00', '2026-07-01T11:30:00'),
}

# -18 degree astronomical twilight crossings for Las Campanas, June 10 2026,
# from the design doc / RESEARCH.md (exact match to computed values).
TWILIGHT_18DEG_JUN10_UTC = ('2026-06-10T23:16:00', '2026-06-11T10:08:00')


class TestTelescopeRuns(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        for obscode, fields in {
            '268': dict(
                name='Magellan Clay Telescope',
                short_name='Magellan-Clay',
                lat=-29.0146,
                lon=-70.6926,
                altitude=2402,
                timezone='America/Santiago',
            ),
            '269': dict(
                name='Magellan Baade Telescope',
                short_name='Magellan-Baade',
                lat=-29.0146,
                lon=-70.6926,
                altitude=2402,
                timezone='America/Santiago',
            ),
            '809': dict(
                name='ESO, La Silla',
                short_name='NTT',
                lat=-29.2567,
                lon=-70.7300,
                altitude=2347,
                timezone='America/Santiago',
            ),
            'E10': dict(
                name='Siding Spring Observatory',
                short_name='FTS',
                lat=-31.2734,
                lon=149.0612,
                altitude=1149,
                timezone='Australia/Sydney',
            ),
        }.items():
            Observatory.objects.update_or_create(obscode=obscode, defaults=fields)

    def setUp(self) -> None:
        self.precision = 6
        return super().setUp()

    def _assert_time_close(self, computed: Time, expected_iso: str, max_seconds: float = 120.0) -> None:
        expected = datetime.fromisoformat(expected_iso).replace(tzinfo=timezone.utc)
        computed_dt = computed.to_datetime(timezone=timezone.utc)
        delta = abs((computed_dt - expected).total_seconds())
        self.assertLessEqual(delta, max_seconds, f'{computed_dt} not within {max_seconds}s of {expected}')

    def test_get_site_returns_observatory(self):
        site = get_site('Magellan-Clay')
        self.assertEqual(site.obscode, '268')
        location = site.to_earth_location()
        self.assertIsNotNone(location)

    def test_get_site_timezone(self):
        self.assertEqual(get_site('NTT').timezone, 'America/Santiago')
        self.assertEqual(get_site('FTS').timezone, 'Australia/Sydney')

    def test_get_site_unknown(self):
        with self.assertRaises(Observatory.DoesNotExist):
            get_site('NoSuchTelescope')

    def test_seeded_records(self):
        self.assertEqual(Observatory.objects.filter(obscode__in=['268', '269', '809', 'E10']).count(), 4)
        clay = Observatory.objects.get(obscode='268')
        self.assertAlmostEqual(clay.lat, -29.0146, self.precision)
        self.assertAlmostEqual(clay.lon, -70.6926, self.precision)
        self.assertAlmostEqual(clay.altitude, 2402, self.precision)
        self.assertEqual(clay.timezone, 'America/Santiago')

        fts = Observatory.objects.get(obscode='E10')
        self.assertAlmostEqual(fts.lat, -31.2734, self.precision)
        self.assertAlmostEqual(fts.lon, 149.0612, self.precision)
        self.assertAlmostEqual(fts.altitude, 1149, self.precision)
        self.assertEqual(fts.timezone, 'Australia/Sydney')

    def test_horizon_dip(self):
        self.assertAlmostEqual(horizon_dip(2402), 1.44, delta=0.02)

    def test_horizon_dip_raises_on_negative_altitude(self):
        with self.assertRaises(ValueError):
            horizon_dip(-10)

    def test_horizon_dip_raises_on_none_altitude(self):
        with self.assertRaises(ValueError):
            horizon_dip(None)

    def test_sun_event_sun(self):
        site = get_site('Magellan-Clay')
        sunset, sunrise = sun_event(site, date(2026, 6, 10), 'sun')
        self.assertIsInstance(sunset, Time)
        self.assertIsInstance(sunrise, Time)
        self._assert_time_close(sunset, '2026-06-10T21:59:00')
        self._assert_time_close(sunrise, '2026-06-11T11:25:00')

    def test_sun_event_dark(self):
        site = get_site('Magellan-Clay')
        sunset, _ = sun_event(site, date(2026, 6, 10), 'sun')
        dark_start, dark_end = sun_event(site, date(2026, 6, 10), 'dark')
        self.assertIsInstance(dark_start, Time)
        self.assertIsInstance(dark_end, Time)
        # Dark window begins after sunset (sun further below horizon)
        self.assertGreater(dark_start.jd, sunset.jd)

    def test_sun_event_bad_kind(self):
        site = get_site('Magellan-Clay')
        with self.assertRaises(ValueError):
            sun_event(site, date(2026, 6, 10), 'badkind')

    def test_sites_dict_contents(self):
        self.assertEqual(SITES['Magellan-Clay'], '268')
        self.assertEqual(SITES['Magellan-Baade'], '269')
        self.assertEqual(SITES['NTT'], '809')
        self.assertEqual(SITES['FTS'], 'E10')

    def test_sunset_sunrise_validation(self):
        """EPHEM-04: Las Campanas sun-event times for Jun 1/10/20/30 2026 match the skycalc reference within 2 min."""
        site = get_site('Magellan-Clay')
        for d, (sunset_ref, sunrise_ref) in LAS_CAMPANAS_SUN_REFERENCE_UTC.items():
            sunset, sunrise = sun_event(site, d, 'sun')
            self._assert_time_close(sunset, sunset_ref)
            self._assert_time_close(sunrise, sunrise_ref)
            # Sunset precedes the following morning's sunrise.
            self.assertLess(sunset.jd, sunrise.jd)
            # The -15 degree dark window sits strictly inside the sun-to-sun window.
            dark_start, dark_end = sun_event(site, d, 'dark')
            self.assertGreater(dark_start.jd, sunset.jd)
            self.assertLess(dark_end.jd, sunrise.jd)

    def test_twilight_18deg_crosscheck(self):
        """EPHEM-05: -18 degree twilight crossings for Jun 10 2026 match 23:16:00/10:08:00 UTC (19:16/06:08 local)."""
        site = get_site('Magellan-Clay')
        anchor = _local_noon_utc(date(2026, 6, 10), site.timezone)
        location = site.to_earth_location()
        crossings = _find_crossing(anchor, location, threshold_deg=-18.0, search_hours=24)
        twilight_end, twilight_start = crossings[0], crossings[1]

        twi_end_ref, twi_start_ref = TWILIGHT_18DEG_JUN10_UTC
        self._assert_time_close(twilight_end, twi_end_ref)
        self._assert_time_close(twilight_start, twi_start_ref)

        santiago = ZoneInfo('America/Santiago')
        twilight_end_local = twilight_end.to_datetime(timezone=timezone.utc).astimezone(santiago)
        twilight_start_local = twilight_start.to_datetime(timezone=timezone.utc).astimezone(santiago)
        self.assertEqual((twilight_end_local.hour, twilight_end_local.minute), (19, 16))
        self.assertEqual((twilight_start_local.hour, twilight_start_local.minute), (6, 8))

    def test_sun_event_raises_on_midnight_sun(self):
        """sun_event raises ValueError (not IndexError) when the sun never sets, e.g. a
        high-latitude site near the summer solstice (no sunset/sunrise crossing pair)."""
        svalbard = Observatory.objects.create(
            obscode='Z99',
            name='Polar Test Site',
            short_name='Polar',
            lat=78.0,
            lon=15.0,
            altitude=0.0,
            timezone='UTC',
        )
        with self.assertRaises(ValueError):
            sun_event(svalbard, date(2026, 6, 21), 'sun')

    def test_sun_event_raises_on_missing_timezone(self):
        """sun_event raises a clear ValueError (not a bare ZoneInfoNotFoundError) when
        Observatory.timezone is unset (its default is '')."""
        no_tz_site = Observatory.objects.create(
            obscode='Z98',
            name='No Timezone Test Site',
            short_name='NoTZ',
            lat=-29.0146,
            lon=-70.6926,
            altitude=2402,
        )
        self.assertEqual(no_tz_site.timezone, '')
        with self.assertRaises(ValueError):
            sun_event(no_tz_site, date(2026, 6, 10), 'sun')

    def test_timezone_dst_resolution(self):
        """EPHEM-06: America/Santiago and Australia/Sydney resolve to the correct UTC offsets across DST boundaries."""
        santiago = ZoneInfo('America/Santiago')
        self.assertEqual(datetime(2026, 6, 15, 12, tzinfo=santiago).utcoffset(), timedelta(hours=-4))
        self.assertEqual(datetime(2026, 1, 15, 12, tzinfo=santiago).utcoffset(), timedelta(hours=-3))

        sydney = ZoneInfo('Australia/Sydney')
        self.assertEqual(datetime(2026, 7, 15, 12, tzinfo=sydney).utcoffset(), timedelta(hours=10))
        self.assertEqual(datetime(2026, 1, 15, 12, tzinfo=sydney).utcoffset(), timedelta(hours=11))
