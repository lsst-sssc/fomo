"""Tests for calendar write access (ACCESS-01, Phase 33 review WR-05).

tomtoolkit 3.1.0's ``tom_calendar`` write views carry no login check and act on any HTTP method, so FOMO
guards them in its own URL conf (``solsys_code/calendar_access.py``). These tests run against FOMO's real
URL conf, so they hold whichever layer does the guarding: an anonymous caller is sent to the login page and
no row changes, while a plain signed-in user still writes exactly as before.
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent, EventTodo

EVENT_POST_DATA = {
    'title': 'Hacked',
    'start_time': '2026-07-04T20:00',
    'end_time': '2026-07-04T21:00',
}


def make_event() -> CalendarEvent:
    """Create the event every test in this module tries to protect (or, signed in, to change)."""
    return CalendarEvent.objects.create(
        title='Keep me',
        description='keep this description',
        start_time=datetime(2026, 7, 4, 20, 0, tzinfo=dt_timezone.utc),
        end_time=datetime(2026, 7, 4, 21, 0, tzinfo=dt_timezone.utc),
        url='https://example.org/keep',
        user='keeper',
        proposal='KEEP-001',
        telescope='FTN',
        instrument='MuSCAT3',
    )


class AnonymousCalendarWriteTest(TestCase):
    """An anonymous request must not create, change or delete a calendar event or todo."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.event = make_event()
        cls.todo = EventTodo.objects.create(event=cls.event, description='keep', is_completed=True)

    def login_url(self) -> str:
        """Return the exact login URL a refused write is redirected to (next is the calendar page)."""
        return f'{settings.LOGIN_URL}?next={reverse("calendar:calendar")}'

    def assert_login_redirect(self, response) -> None:
        """Assert the response is the 302 to the login page."""
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], self.login_url())

    def snapshot(self) -> tuple:
        """Return every protected field of the event and todo plus both table counts."""
        event = CalendarEvent.objects.filter(pk=self.event.pk).first()
        todo = EventTodo.objects.filter(pk=self.todo.pk).first()
        event_fields = (
            None
            if event is None
            else (
                event.title,
                event.description,
                event.start_time,
                event.end_time,
                event.url,
                event.target_list_id,
                event.user,
                event.proposal,
                event.telescope,
                event.instrument,
                event.modified,
            )
        )
        todo_fields = None if todo is None else (todo.description, todo.is_completed)
        return event_fields, todo_fields, CalendarEvent.objects.count(), EventTodo.objects.count()

    def assert_unchanged(self, before: tuple) -> None:
        """Assert nothing in the snapshot changed."""
        self.assertEqual(self.snapshot(), before)

    def test_anonymous_post_update_event_changes_nothing(self) -> None:
        before = self.snapshot()

        response = self.client.post(reverse('calendar:update-event', args=[self.event.pk]), EVENT_POST_DATA)

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_other_methods_on_update_event_are_refused(self) -> None:
        url = reverse('calendar:update-event', args=[self.event.pk])
        for method in ('put', 'patch', 'delete', 'options'):
            with self.subTest(method=method):
                before = self.snapshot()

                response = getattr(self.client, method)(url, EVENT_POST_DATA)

                self.assert_login_redirect(response)
                self.assert_unchanged(before)

    def test_anonymous_get_and_head_update_event_stay_open(self) -> None:
        url = reverse('calendar:update-event', args=[self.event.pk])

        get_response = self.client.get(url)
        head_response = self.client.head(url)

        self.assertEqual(get_response.status_code, 200)
        self.assertContains(get_response, 'Keep me')
        self.assertEqual(head_response.status_code, 200)

    def test_anonymous_repeat_post_update_event_is_refused_both_times(self) -> None:
        url = reverse('calendar:update-event', args=[self.event.pk])
        before = self.snapshot()

        first = self.client.post(url, EVENT_POST_DATA)
        second = self.client.post(url, EVENT_POST_DATA)

        self.assert_login_redirect(first)
        self.assert_login_redirect(second)
        self.assertEqual(first['Location'], second['Location'])
        self.assert_unchanged(before)


class SignedInCalendarWriteTest(TestCase):
    """A plain (non-staff) signed-in user keeps the write access the calendar always gave."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.event = make_event()
        cls.editor = User.objects.create_user(username='calendar-editor', password='pw')

    def setUp(self) -> None:
        self.client.force_login(self.editor)

    def test_update_event_post_saves_for_plain_user(self) -> None:
        response = self.client.post(reverse('calendar:update-event', args=[self.event.pk]), EVENT_POST_DATA)

        self.assertEqual(response.status_code, 200)
        self.event.refresh_from_db()
        self.assertEqual(self.event.title, 'Hacked')
