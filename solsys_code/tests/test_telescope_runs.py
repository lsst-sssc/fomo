from datetime import date, datetime, timezone

from astropy.time import Time
from django.test import TestCase

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import SITES, get_site, horizon_dip, sun_event


class TestTelescopeRuns(TestCase):
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
