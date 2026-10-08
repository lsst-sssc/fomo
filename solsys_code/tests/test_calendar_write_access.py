"""Tests for calendar write access (ACCESS-01, Phase 33 review WR-05).

tomtoolkit 3.1.0's ``tom_calendar`` write views carry no login check and act on any HTTP method, so FOMO
guards them in its own URL conf (``solsys_code/calendar_access.py``). These tests run against FOMO's real
URL conf, so they hold whichever layer does the guarding: an anonymous caller is sent to the login page and
no row changes, while a plain signed-in user still writes exactly as before.
"""

import inspect
from datetime import datetime
from datetime import timezone as dt_timezone

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import resolve, reverse
from tom_calendar import views as upstream_views
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

    def test_anonymous_post_create_event_creates_nothing(self) -> None:
        before = self.snapshot()

        response = self.client.post(reverse('calendar:create-event'), EVENT_POST_DATA)

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_post_delete_event_keeps_row(self) -> None:
        before = self.snapshot()

        response = self.client.post(reverse('calendar:delete-event', args=[self.event.pk]))

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_post_create_todo_adds_nothing(self) -> None:
        before = self.snapshot()

        response = self.client.post(reverse('calendar:create-todo', args=[self.event.pk]), {'description': 'second'})

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_post_update_todo_changes_nothing(self) -> None:
        before = self.snapshot()

        response = self.client.post(
            reverse('calendar:update-todo', args=[self.todo.pk]), {'description': 'changed', 'is_completed': 'false'}
        )

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_get_delete_event_redirects_and_keeps_row(self) -> None:
        before = self.snapshot()

        response = self.client.get(reverse('calendar:delete-event', args=[self.event.pk]))

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_get_create_event_redirects(self) -> None:
        before = self.snapshot()

        response = self.client.get(reverse('calendar:create-event'))

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_get_create_todo_redirects(self) -> None:
        before = self.snapshot()

        response = self.client.get(reverse('calendar:create-todo', args=[self.event.pk]))

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_get_update_todo_changes_nothing(self) -> None:
        # Upstream blanks the todo's description and completion flag on a plain GET.
        before = self.snapshot()

        response = self.client.get(reverse('calendar:update-todo', args=[self.todo.pk]))

        self.assert_login_redirect(response)
        self.assert_unchanged(before)

    def test_anonymous_post_to_missing_event_is_redirected_not_404(self) -> None:
        before = self.snapshot()

        delete_response = self.client.post(reverse('calendar:delete-event', args=[999999]))
        update_response = self.client.post(reverse('calendar:update-event', args=[999999]), EVENT_POST_DATA)

        self.assert_login_redirect(delete_response)
        self.assert_login_redirect(update_response)
        self.assert_unchanged(before)

    def write_requests(self) -> list[tuple[str, str, dict]]:
        """Return (label, url, data) for every write route, to POST to."""
        return [
            ('create-event', reverse('calendar:create-event'), EVENT_POST_DATA),
            ('update-event', reverse('calendar:update-event', args=[self.event.pk]), EVENT_POST_DATA),
            ('delete-event', reverse('calendar:delete-event', args=[self.event.pk]), {}),
            ('create-todo', reverse('calendar:create-todo', args=[self.event.pk]), {'description': 'second'}),
            ('update-todo', reverse('calendar:update-todo', args=[self.todo.pk]), {'description': 'x'}),
        ]

    def test_anonymous_repeat_writes_are_refused_both_times(self) -> None:
        for label, url, data in self.write_requests():
            with self.subTest(route=label):
                before = self.snapshot()

                first = self.client.post(url, data)
                second = self.client.post(url, data)

                self.assert_login_redirect(first)
                self.assert_login_redirect(second)
                self.assertEqual(first['Location'], second['Location'])
                self.assert_unchanged(before)

    def test_htmx_anonymous_writes_get_hx_redirect_never_403(self) -> None:
        for label, url, data in self.write_requests():
            with self.subTest(route=label):
                before = self.snapshot()

                response = self.client.post(url, data, headers={'HX-Request': 'true'})

                self.assertEqual(response.status_code, 200)
                self.assertNotEqual(response.status_code, 403)
                self.assertEqual(response['HX-Redirect'], self.login_url())
                self.assert_unchanged(before)

    def test_literal_paths_are_guarded(self) -> None:
        literal_paths = [
            ('/calendar/create/', EVENT_POST_DATA),
            (f'/calendar/update/{self.event.pk}/', EVENT_POST_DATA),
            (f'/calendar/delete/{self.event.pk}/', {}),
            (f'/calendar/todo/create/{self.event.pk}/', {'description': 'second'}),
            (f'/calendar/todo/update/{self.todo.pk}/', {'description': 'x'}),
        ]
        for path, data in literal_paths:
            with self.subTest(path=path):
                before = self.snapshot()

                response = self.client.post(path, data)

                self.assert_login_redirect(response)
                self.assert_unchanged(before)


class SignedInCalendarWriteTest(TestCase):
    """A plain (non-staff) signed-in user keeps the write access the calendar always gave."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.event = make_event()
        cls.todo = EventTodo.objects.create(event=cls.event, description='edit me', is_completed=False)
        cls.editor = User.objects.create_user(username='calendar-editor', password='pw')

    def setUp(self) -> None:
        self.client.force_login(self.editor)

    def test_update_event_post_saves_for_plain_user(self) -> None:
        response = self.client.post(reverse('calendar:update-event', args=[self.event.pk]), EVENT_POST_DATA)

        self.assertEqual(response.status_code, 200)
        self.event.refresh_from_db()
        self.assertEqual(self.event.title, 'Hacked')

    def test_create_event_post_creates_one(self) -> None:
        before = CalendarEvent.objects.count()

        response = self.client.post(reverse('calendar:create-event'), EVENT_POST_DATA)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(CalendarEvent.objects.count(), before + 1)

    def test_delete_event_post_removes_row(self) -> None:
        # D-03: delete is not gated more tightly than "logged in", so a non-staff user can delete.
        response = self.client.post(reverse('calendar:delete-event', args=[self.event.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(CalendarEvent.objects.filter(pk=self.event.pk).exists())

    def test_create_todo_post_adds_one(self) -> None:
        before = EventTodo.objects.count()

        response = self.client.post(reverse('calendar:create-todo', args=[self.event.pk]), {'description': 'second'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(EventTodo.objects.count(), before + 1)

    def test_update_todo_post_changes_fields(self) -> None:
        response = self.client.post(
            reverse('calendar:update-todo', args=[self.todo.pk]), {'description': 'changed', 'is_completed': 'true'}
        )

        self.assertEqual(response.status_code, 200)
        self.todo.refresh_from_db()
        self.assertEqual(self.todo.description, 'changed')
        self.assertTrue(self.todo.is_completed)

    def test_get_create_and_update_event_render(self) -> None:
        self.assertEqual(self.client.get(reverse('calendar:create-event')).status_code, 200)
        self.assertEqual(self.client.get(reverse('calendar:update-event', args=[self.event.pk])).status_code, 200)

    def test_get_on_destructive_routes_is_405_and_changes_nothing(self) -> None:
        # Upstream deletes an event, adds a todo or blanks a todo on a plain GET; require_POST refuses that.
        urls = [
            reverse('calendar:delete-event', args=[self.event.pk]),
            reverse('calendar:create-todo', args=[self.event.pk]),
            reverse('calendar:update-todo', args=[self.todo.pk]),
        ]
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)

                self.assertEqual(response.status_code, 405)
        self.event.refresh_from_db()
        self.todo.refresh_from_db()
        self.assertEqual(self.event.title, 'Keep me')
        self.assertEqual((self.todo.description, self.todo.is_completed), ('edit me', False))
        self.assertEqual(EventTodo.objects.count(), 1)

    def test_post_without_csrf_token_is_refused(self) -> None:
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.editor)
        before = CalendarEvent.objects.count()

        response = client.post(reverse('calendar:create-event'), EVENT_POST_DATA)

        # CsrfViewMiddleware answers 403; tom_common's Raise403Middleware turns every browser 403 into a
        # redirect to the login page whose next is the refused path (not the guard's next=/calendar/).
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], f'{settings.LOGIN_URL}?next={reverse("calendar:create-event")}')
        self.assertEqual(CalendarEvent.objects.count(), before)


class CalendarUrlConfShadowingTest(SimpleTestCase):
    """FOMO's guarded calendar URL conf wins over tom_common's unguarded copy of tom_calendar.urls."""

    def test_write_paths_resolve_to_fomo_guarded_upstream_views(self) -> None:
        rows = [
            ('/calendar/create/', 'create-event', 'write_requires_login', upstream_views.create_event),
            ('/calendar/update/1/', 'update-event', 'read_open_write_requires_login', upstream_views.update_event),
            ('/calendar/delete/1/', 'delete-event', 'write_requires_login', upstream_views.delete_event),
            ('/calendar/todo/create/1/', 'create-todo', 'write_requires_login', upstream_views.create_todo),
            ('/calendar/todo/update/1/', 'update-todo', 'write_requires_login', upstream_views.update_todo),
        ]
        for path, url_name, guard, upstream in rows:
            with self.subTest(path=path):
                match = resolve(path)

                self.assertEqual(match.app_name, 'calendar')
                self.assertEqual(match.namespace, 'calendar')
                self.assertEqual(match.url_name, url_name)
                self.assertIsNot(match.func, upstream)
                self.assertEqual(match.func.calendar_guard, guard)
                self.assertIs(inspect.unwrap(match.func), upstream)
