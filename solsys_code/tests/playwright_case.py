"""Shared base class for the browser-driven (Playwright) functional tests."""

import os

from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from playwright.sync_api import sync_playwright


class PlaywrightTestCase(StaticLiveServerTestCase):
    """Live-server test case with a headless Chromium ``self.page`` per test. Tag subclasses ``functional``."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Playwright's synchronous API keeps an asyncio event loop "running" (via greenlet-based
        # dispatch) in whichever thread calls sync_playwright().start(). Django's async-safety
        # guard (django.utils.asyncio.async_unsafe) sees that running loop and raises
        # SynchronousOnlyOperation on every subsequent DB access in this thread -- a false
        # positive, since nothing here is actually concurrent. This is a documented
        # Playwright/Django interaction; the standard fix is to opt this thread out of the
        # async-safety check, scoped to this class's lifetime so it doesn't mask real
        # async-safety bugs in other tests that share this process.
        cls._prev_async_unsafe = os.environ.get('DJANGO_ALLOW_ASYNC_UNSAFE')
        os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        if cls._prev_async_unsafe is None:
            os.environ.pop('DJANGO_ALLOW_ASYNC_UNSAFE', None)
        else:
            os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = cls._prev_async_unsafe
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()
        super().tearDown()
