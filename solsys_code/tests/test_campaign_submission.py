"""Tests for the public submission write path (SUBMIT-01/04/05).

Uses `TargetList.objects.create(...)` (this project's fixtures are always non-sidereal per
CLAUDE.md; no sidereal-target factory is used anywhere in this module) and plain
`User.objects.create_user(...)` fixtures for staff/non-staff, mirroring
`test_campaign_views.py`'s conventions.
"""

from datetime import date

from django.contrib.auth.models import User
from django.core import mail
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList

from solsys_code.models import CampaignRun

CONTACT_PERSON = 'Jane Coordinator'
CONTACT_EMAIL = 'jane@example.org'
TELESCOPE_INSTRUMENT = 'FTN/MuSCAT3'
OBS_DATE = date(2026, 8, 1)


class CampaignSubmissionTestBase(TestCase):
    """Shared fixture: one campaign, a staff user with an email, a staff user with a blank
    email, and a non-staff user -- so SUBMIT-05's recipient-filtering logic is fully exercised.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.staff_with_email = User.objects.create_user(
            username='staffwithemail', password='pw', is_staff=True, email='staff@example.org'
        )
        cls.staff_blank_email = User.objects.create_user(
            username='staffblankemail', password='pw', is_staff=True, email=''
        )
        cls.non_staff_user = User.objects.create_user(
            username='regularuser', password='pw', is_staff=False, email='regular@example.org'
        )

    def submit_url(self):
        return reverse('campaigns:submit')

    def thanks_url(self):
        return reverse('campaigns:submission_thanks')

    def minimal_valid_data(self, **overrides):
        data = {
            'campaign': self.campaign.pk,
            'contact_person': CONTACT_PERSON,
            'contact_email': CONTACT_EMAIL,
        }
        data.update(overrides)
        return data


class TestCampaignSubmission(CampaignSubmissionTestBase):
    """SUBMIT-01: minimal valid submission creates a PENDING_REVIEW CampaignRun."""

    def test_minimal_valid_submission_creates_pending_run(self):
        response = self.client.post(self.submit_url(), data=self.minimal_valid_data(obs_date=OBS_DATE.isoformat()))
        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.get()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self.assertEqual(run.campaign, self.campaign)
        self.assertEqual(run.contact_person, CONTACT_PERSON)
        self.assertEqual(run.contact_email, CONTACT_EMAIL)
        # SCHED-02: the form's single observing-date field collapses to window_start ==
        # window_end (a single-night run).
        self.assertEqual(run.window_start, OBS_DATE)
        self.assertEqual(run.window_end, OBS_DATE)
        self.assertRedirects(response, self.thanks_url())

    def test_contact_public_opt_in_checked_persists_true(self):
        """VIEW-05: submitting the box checked persists contact_public_opt_in=True."""
        self.client.post(self.submit_url(), data=self.minimal_valid_data(contact_public_opt_in='on'))
        run = CampaignRun.objects.get()
        self.assertTrue(run.contact_public_opt_in)

    def test_contact_public_opt_in_unchecked_persists_false(self):
        """VIEW-05: an unchecked box (default opt-out) persists contact_public_opt_in=False."""
        self.client.post(self.submit_url(), data=self.minimal_valid_data())
        run = CampaignRun.objects.get()
        self.assertFalse(run.contact_public_opt_in)

    def test_get_returns_200_and_renders_form(self):
        response = self.client.get(self.submit_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'form')

    def test_missing_campaign_invalid(self):
        data = self.minimal_valid_data()
        del data['campaign']
        response = self.client.post(self.submit_url(), data=data)
        self.assertEqual(response.status_code, 200)  # form re-rendered, not redirected
        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertFormError(response.context['form'], 'campaign', 'This field is required.')

    def test_missing_contact_person_invalid(self):
        data = self.minimal_valid_data()
        del data['contact_person']
        response = self.client.post(self.submit_url(), data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertFormError(response.context['form'], 'contact_person', 'This field is required.')

    def test_missing_contact_email_invalid(self):
        data = self.minimal_valid_data()
        del data['contact_email']
        response = self.client.post(self.submit_url(), data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertFormError(response.context['form'], 'contact_email', 'This field is required.')

    def test_duplicate_natural_key_submission_shows_friendly_form_error(self):
        """Pitfall 4: same campaign+telescope_instrument+window_start(==window_end) collides
        on the resolved-window UniqueConstraint -- a friendly non_field_errors banner, never
        a 500.
        """
        data = self.minimal_valid_data(
            telescope_instrument=TELESCOPE_INSTRUMENT,
            obs_date=OBS_DATE.isoformat(),
        )
        first = self.client.post(self.submit_url(), data=data)
        self.assertRedirects(first, self.thanks_url())
        self.assertEqual(CampaignRun.objects.count(), 1)

        second = self.client.post(self.submit_url(), data=data)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(CampaignRun.objects.count(), 1)  # unchanged, no second row
        self.assertTrue(second.context['form'].non_field_errors())


class TestCampaignSubmissionObsDateWindow(CampaignSubmissionTestBase):
    """260714-ilz: end-to-end POST coverage for the flexible obs_date/window intake
    (requirements 1/2/5/7) -- proves no 500, correct DB effect, and correct HTTP status.
    """

    def test_multi_night_range_creates_one_run_with_resolved_window(self):
        response = self.client.post(
            self.submit_url(),
            data=self.minimal_valid_data(obs_date='2027-04-20 -- 2027-05-11'),
        )
        self.assertRedirects(response, self.thanks_url())
        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.get()
        self.assertEqual(run.window_start, date(2027, 4, 20))
        self.assertEqual(run.window_end, date(2027, 5, 11))

    def test_blank_obs_date_creates_one_tbd_run(self):
        response = self.client.post(self.submit_url(), data=self.minimal_valid_data())
        self.assertRedirects(response, self.thanks_url())
        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.get()
        self.assertIsNone(run.window_start)
        self.assertIsNone(run.window_end)

    def test_unparseable_obs_date_re_renders_form_creates_no_run(self):
        response = self.client.post(
            self.submit_url(),
            data=self.minimal_valid_data(obs_date='sometime next spring'),
        )
        self.assertEqual(response.status_code, 200)  # re-rendered, not a redirect or a 500
        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertIn('obs_date', response.context['form'].errors)

    def test_duplicate_range_submission_shows_friendly_form_error(self):
        """Requirement 7: the existing except-IntegrityError handler covers the range case."""
        data = self.minimal_valid_data(
            telescope_instrument=TELESCOPE_INSTRUMENT,
            obs_date='2027-04-20 -- 2027-05-11',
        )
        first = self.client.post(self.submit_url(), data=data)
        self.assertRedirects(first, self.thanks_url())
        self.assertEqual(CampaignRun.objects.count(), 1)

        second = self.client.post(self.submit_url(), data=data)
        self.assertEqual(second.status_code, 200)  # re-rendered, not a 500
        self.assertEqual(CampaignRun.objects.count(), 1)  # unchanged, no second row
        self.assertTrue(second.context['form'].non_field_errors())


class TestHoneypot(CampaignSubmissionTestBase):
    """SUBMIT-04: a tripped honeypot creates nothing, emails nothing, and redirects identically
    to a genuine submission.
    """

    def test_honeypot_filled_creates_no_run_and_sends_no_email(self):
        data = self.minimal_valid_data(alt_contact_info='I am a bot')
        response = self.client.post(self.submit_url(), data=data)
        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertEqual(len(mail.outbox), 0)
        self.assertRedirects(response, self.thanks_url())

    def test_honeypot_response_matches_genuine_submission_redirect(self):
        genuine_response = self.client.post(self.submit_url(), data=self.minimal_valid_data())
        honeypot_response = self.client.post(self.submit_url(), data=self.minimal_valid_data(alt_contact_info='trap'))
        self.assertEqual(genuine_response.url, honeypot_response.url)
        self.assertEqual(genuine_response.status_code, honeypot_response.status_code)


class TestStaffNotification(CampaignSubmissionTestBase):
    """SUBMIT-05: genuine submissions email every is_staff+email user; no PII in the message."""

    def test_genuine_submission_emails_every_staff_user_with_email(self):
        self.client.post(self.submit_url(), data=self.minimal_valid_data())
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, [self.staff_with_email.email])

    def test_staff_with_blank_email_not_a_recipient(self):
        self.client.post(self.submit_url(), data=self.minimal_valid_data())
        self.assertEqual(len(mail.outbox), 1)
        self.assertNotIn(self.staff_blank_email.email, mail.outbox[0].to)

    def test_non_staff_user_not_a_recipient(self):
        self.client.post(self.submit_url(), data=self.minimal_valid_data())
        self.assertEqual(len(mail.outbox), 1)
        self.assertNotIn(self.non_staff_user.email, mail.outbox[0].to)

    def test_email_contains_no_pii(self):
        """D-04: subject/body must never contain contact_person, contact_email,
        telescope_instrument, or campaign name.
        """
        self.client.post(
            self.submit_url(),
            data=self.minimal_valid_data(telescope_instrument=TELESCOPE_INSTRUMENT),
        )
        self.assertEqual(len(mail.outbox), 1)
        sent = mail.outbox[0]
        for pii in (CONTACT_PERSON, CONTACT_EMAIL, TELESCOPE_INSTRUMENT, self.campaign.name):
            self.assertNotIn(pii, sent.subject)
            self.assertNotIn(pii, sent.body)

    def test_no_staff_with_email_sends_no_email(self):
        self.staff_with_email.email = ''
        self.staff_with_email.save()
        self.client.post(self.submit_url(), data=self.minimal_valid_data())
        self.assertEqual(len(mail.outbox), 0)


class TestSubmissionFormSiteSearchWidget(CampaignSubmissionTestBase):
    """D-09/D-10/22-REVIEWS.md findings 1 and 7: the public form's site_raw field is a
    live-search widget wired to campaigns:site_search, with NO create-new-site escape
    hatch (that stays staff-only on the approval queue -- Task 2).
    """

    def test_form_renders_hx_get_and_corrected_trigger_grammar(self):
        response = self.client.get(self.submit_url())
        self.assertContains(response, 'hx-get')
        self.assertContains(response, reverse('campaigns:site_search'))
        # Django HTML-escapes widget attribute values (`>` -> `&gt;`, `"` -> `&quot;`), so
        # assert on escaping-immune substrings either side of the event-filter bracket.
        self.assertContains(response, 'hx-trigger="input[this.value.length')
        self.assertContains(response, '] changed delay:300ms"')
        # 22-REVIEWS.md finding 1: the malformed filter-after-delay ordering must never
        # regress back in.
        self.assertNotContains(response, 'delay:300ms[')

    def test_form_renders_suggestions_container(self):
        response = self.client.get(self.submit_url())
        self.assertContains(response, '<div id="site-suggestions-id_site_raw"')

    def test_form_has_no_create_new_observatory_link(self):
        """D-09: public submitters never get a site-creation path."""
        response = self.client.get(self.submit_url())
        self.assertNotContains(response, 'Create new Observatory')

    def test_click_to_fill_wiring_uses_one_consistent_id(self):
        """22-REVIEWS.md finding 7: the input id, hx-target/container suffix, and hx-vals
        input_id value must all derive from the same 'id_site_raw' string, or the
        endpoint's onclick fill silently breaks.
        """
        response = self.client.get(self.submit_url())
        content = response.content.decode()
        self.assertIn('id="id_site_raw"', content)
        self.assertIn('hx-target="#site-suggestions-id_site_raw"', content)
        self.assertIn('<div id="site-suggestions-id_site_raw"', content)
        self.assertIn('&quot;input_id&quot;: &quot;id_site_raw&quot;', content)
