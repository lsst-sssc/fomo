"""Tests for the live "currently passing" Rubin ToO Scout view.

These are DB-backed and exercise :class:`solsys_code.scout_views.RubinTooScoutListView`
against real ``tom_jpl.models.ScoutDetail`` rows, confirming the view lists only
candidates that pass all Section 2.1 filters.

Note: requires a ``tom_jpl`` that includes the vmag/rate/ra/dec/t_ephem fields
(the editable source under ../tom_jpl). Run e.g.::

    PYTHONPATH=/home/tlister/git/tom_jpl ./manage.py test solsys_code.tests.test_scout_views
"""

from django.test import TestCase
from django.urls import reverse
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
