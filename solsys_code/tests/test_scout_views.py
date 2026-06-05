"""Tests for the live "currently passing" Rubin ToO Scout view.

These are DB-backed and exercise :class:`solsys_code.scout_views.RubinTooScoutListView`
against real ``tom_jpl.models.ScoutDetail`` rows, confirming the view lists only
candidates that pass all Section 2.1 filters.

Note: requires a ``tom_jpl`` that includes the vmag/rate/ra/dec/t_ephem fields
(the editable source under ../tom_jpl). Run e.g.::

    PYTHONPATH=/home/tlister/git/tom_jpl ./manage.py test solsys_code.tests.test_scout_views
"""

from unittest import mock

from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, TestCase
from django.urls import reverse
from tom_jpl.jpl import ScoutDataService
from tom_jpl.models import ScoutDetail
from tom_targets.models import Target

# A ScoutDetail field set that passes every Section 2.1 filter (Northern, dec > 0).
PASSING_DETAIL = dict(
    neo_score=99,
    geocentric_score=1,
    impact_rating=3,
    rms=0.5,
    num_obs=6,
    arc=2.0 / 24.0,
    vmag=21.7,
    uncertainty_p1=70.0,
    rate=10.0,
    dec=10.0,
    ra=133.5,
)


class RubinTooScoutListViewTest(TestCase):
    def setUp(self):
        self.url = reverse('scout_rubin_too')

        passing_target = Target.objects.create(name='PASS1', type=Target.NON_SIDEREAL, abs_mag=25.0)
        ScoutDetail.objects.create(target=passing_target, **PASSING_DETAIL)

        # Identical except the sky-motion rate now exceeds the 25 arcsec/min cut.
        failing_target = Target.objects.create(name='FAIL1', type=Target.NON_SIDEREAL, abs_mag=25.0)
        failing_detail = dict(PASSING_DETAIL, rate=30.0)
        ScoutDetail.objects.create(target=failing_target, **failing_detail)

    def test_lists_only_passing_candidates(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'PASS1')
        self.assertNotContains(response, 'FAIL1')
        self.assertEqual(response.context['num_passing'], 1)
        self.assertEqual(response.context['num_total'], 2)

    def test_arc_displayed_in_hours(self):
        response = self.client.get(self.url)
        # arc stored as 2/24 days should be annotated as 2.0 hours for display.
        passing = response.context['scout_details']
        self.assertEqual(len(passing), 1)
        self.assertAlmostEqual(passing[0].arc_hours, 2.0)


# --- End-to-end ingest fixtures -------------------------------------------------------
# A raw Scout API object whose parsed detail satisfies the Rubin ToO Section 2.1 filters.
# Kept inline (rather than importing tom_jpl's test helpers) so this module does not pull
# in factory_boy, which need not be installed in the FOMO environment.
_SCOUT_SIGNATURE = {'source': 'NASA/JPL Scout API', 'version': '1.3'}

_PASSING_SCOUT_OBJECT = {
    'objectName': 'ZTF10BL',
    'neoScore': 100,
    'neo1kmScore': 0,
    'phaScore': 0,
    'ieoScore': 0,
    'geocentricScore': 1,
    'rating': 4,
    'unc': '1400',
    'uncP1': '1500',
    'caDist': '0.98',
    'arc': '2.0',
    'nObs': 8,
    'rmsN': '0.12',
    'Vmag': '21.9',
    'rate': '1.9',
    'ra': '08:54',
    'dec': '+28',
    'tEphem': '2026-02-11 16:30',
    'lastRun': '2026-02-11 22:45',
}

_SCOUT_ORBITS = {
    'fields': [
        'idx',
        'epoch',
        'ec',
        'qr',
        'tp',
        'om',
        'w',
        'inc',
        'H',
        'dca',
        'tca',
        'moid',
        'vinf',
        'geoEcc',
        'impFlag',
    ],
    'data': [
        [
            0,
            '2461079.712931752',
            '8.855752093403347E-01',
            '4.998861947435592E-01',
            '2461038.529885748',
            '1.3898404962804094E+02',
            '2.6665272638172337E+02',
            '1.4692644033445657E+01',
            '25.402927',
            '4.34144690247134E-03',
            '2.4610794091831E+06',
            '1.971147172E-03',
            '2.790457665E+01',
            '1.278753791E+03',
            0,
        ]
    ],
    'count': '1',
}

_PERMISSIVE_INPUT_PARAMETERS = {
    'ca_dist_min': None,
    'data_service': 'Scout',
    'geo_score_max': 5,
    'impact_rating_min': None,
    'neo_score_min': None,
    'pha_score_min': None,
    'pos_unc_max': None,
    'pos_unc_min': None,
    'query_name': '',
    'query_save': False,
    'tdes': '',
}


def _scout_api_get(summary_object, detail_object):
    """Return a ``requests.get`` replacement answering the Scout summary and per-object calls."""
    summary_payload = {'signature': _SCOUT_SIGNATURE, 'count': 1, 'data': [summary_object]}
    detail_payload = dict(detail_object, signature=_SCOUT_SIGNATURE)

    def _get(url, data=None, **kwargs):
        payload = detail_payload if (data and data.get('tdes')) else summary_payload
        response = mock.Mock()
        response.json.return_value = payload
        return response

    return _get


class RubinTooScoutIngestEndToEndTest(TestCase):
    """Mocked Scout API -> real ingest -> DB -> live view.

    Drives the whole pipeline: ``requests.get`` is patched so the real
    ``ScoutDataService`` query/parse/save chain runs, then the view is requested to
    confirm the ingested candidate is listed as passing.
    """

    def _ingest(self, summary_object, detail_object):
        ds = ScoutDataService()
        fake_get = _scout_api_get(summary_object, detail_object)
        with mock.patch('tom_jpl.jpl.requests.get', side_effect=fake_get):
            targets_data = ds.query_targets(dict(_PERMISSIVE_INPUT_PARAMETERS))

        request = RequestFactory().get('/')
        request.user = AnonymousUser()
        with mock.patch('tom_dataservices.dataservices.messages'):
            for target_data in targets_data:
                ds.to_target(target_data, request=request)

    def test_ingested_candidate_appears_in_view(self):
        detail_object = dict(_PASSING_SCOUT_OBJECT, orbits=_SCOUT_ORBITS)
        self._ingest(_PASSING_SCOUT_OBJECT, detail_object)

        self.assertEqual(ScoutDetail.objects.count(), 1)

        response = self.client.get(reverse('scout_rubin_too'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'ZTF10BL')
        self.assertEqual(response.context['num_passing'], 1)
