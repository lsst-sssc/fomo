"""CR-01 regression tests (28-VERIFICATION.md / 28-REVIEW.md).

Why this module exists separately from ``test_campaign_attribution_views.py``: every check
here asserts **HTML structure**, never request behavior. The Django test client's POST
helper bypasses browser-side constraint validation entirely, so it cannot detect CR-01's
whole class of defect -- a rendered Confirm button that a real browser refuses to submit
because it shares a ``<form>`` with a ``required`` field meant only for Dismiss. Every one of
the 124 tests that existed before this plan submitted decisions through that same POST
helper and gave zero signal on this bug. Adding a test here that submits a decision through
the test client would defeat this module's purpose -- do not add one.
"""

import html.parser
import re
from collections import defaultdict

from django.template.loader import get_template
from django.test import SimpleTestCase
from django.urls import reverse

from solsys_code.tests.test_campaign_attribution_views import AttributionViewTestBase

TEMPLATE_NAME = 'campaigns/attribution_queue.html'


class _FormStructureParser(html.parser.HTMLParser):
    """Resolves HTML5 form ownership for ``input``/``button``/``select``/``textarea``
    elements against the rendered attribution-queue page.

    This page deliberately uses out-of-line forms: the High-band bulk-confirm checkboxes
    carry an explicit ``form="bulk-confirm-events"``/``form="bulk-confirm-records"``
    attribute and are rendered inside a table cell, not nested inside the ``<form>`` they
    submit into. A browser resolves a control's owning form as the value of its own ``form=``
    attribute when present, otherwise the innermost currently-open ``<form>`` -- this parser
    matches that resolution order exactly. Getting it wrong (e.g. naive nearest-enclosing-tag
    parsing) would attribute the High-band checkboxes to the wrong form and make the
    invariant below assert against the wrong controls.
    """

    OWNED_TAGS = {'input', 'button', 'select', 'textarea'}

    def __init__(self):
        super().__init__()
        self._form_stack: list[str] = []
        self._anon_counter = 0
        # form key -> {'required': [(tag, name), ...], 'submitters': [(name, value, has_formnovalidate), ...]}
        self.forms: dict[str, dict] = defaultdict(lambda: {'required': [], 'submitters': []})

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'form':
            form_id = attrs_dict.get('id')
            if form_id is None:
                self._anon_counter += 1
                form_id = f'_anon_{self._anon_counter}'
            self._form_stack.append(form_id)
            self.forms[form_id]  # noqa: B018 -- touch to ensure the key exists even if it owns nothing
            return

        if tag not in self.OWNED_TAGS:
            return

        owner = attrs_dict.get('form')
        if owner is None:
            if not self._form_stack:
                return  # An orphan control outside any form -- nothing to attribute it to.
            owner = self._form_stack[-1]

        if 'required' in attrs_dict:
            self.forms[owner]['required'].append((tag, attrs_dict.get('name')))

        is_submit_button = tag == 'button' and attrs_dict.get('type', 'submit') == 'submit'
        if is_submit_button:
            self.forms[owner]['submitters'].append(
                (attrs_dict.get('name'), attrs_dict.get('value'), 'formnovalidate' in attrs_dict)
            )

    def handle_endtag(self, tag):
        if tag == 'form' and self._form_stack:
            self._form_stack.pop()


class AttributionTemplateSourceTests(SimpleTestCase):
    """No database, no test client, no HTTP at all -- the evidence is the on-disk template
    source, resolved through Django's own template loader rather than a hardcoded path."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        template_path = get_template(TEMPLATE_NAME).origin.name
        with open(template_path, encoding='utf-8') as fh:
            cls.source = fh.read()

    def test_every_confirm_submitter_carries_formnovalidate(self):
        """Exactly 2 Confirm buttons (one per worklist), each with the validation opt-out --
        so a future third Confirm button that skips it cannot silently leave this test green."""
        button_tags = re.findall(r'<button[^>]*>', self.source)
        confirm_buttons = [b for b in button_tags if 'name="action"' in b and 'value="confirm"' in b]
        self.assertEqual(len(confirm_buttons), 2, f'expected exactly 2 Confirm buttons, found {len(confirm_buttons)}')
        for button in confirm_buttons:
            self.assertIn(
                'formnovalidate',
                button,
                f'Confirm button missing the validation opt-out attribute: {button!r}',
            )

    def test_dismiss_reason_input_is_still_required(self):
        """UI-SPEC's Copywriting Contract mandates `required` on the Dismiss reason input --
        a future "fix" that deletes this instead of exempting Confirm must fail here."""
        reason_inputs = re.findall(r'<input[^>]*name="reason"[^>]*>', self.source)
        self.assertEqual(len(reason_inputs), 2, f'expected exactly 2 reason inputs, found {len(reason_inputs)}')
        for reason_input in reason_inputs:
            self.assertIn(' required', reason_input, f'reason input lost its required attribute: {reason_input!r}')

    def test_no_form_level_novalidate_opt_out(self):
        """The literal `novalidate` attribute must never appear on a `<form>` tag itself --
        that would silently delete the Dismiss gate for every submitter in the form, not just
        Confirm. Searching only `<form` tags (not the whole file) matters: `formnovalidate` on
        a button contains the substring `novalidate`, so a naive whole-file search is wrong."""
        form_tags = re.findall(r'<form\b[^>]*>', self.source)
        self.assertTrue(form_tags, 'no <form> tags found in the template source')
        for form_tag in form_tags:
            self.assertNotIn(
                'novalidate', form_tag, f'a <form> tag carries a standalone novalidate opt-out: {form_tag!r}'
            )


class AttributionRenderedFormStructureTests(AttributionViewTestBase):
    """Renders the real page through a GET (never a POST simulating a submit) and parses the
    actual HTML the browser would receive, asserting the same invariant CR-01 broke: no
    Confirm submitter may share a form with a required control unless it opts out."""

    def setUp(self):
        self._make_event()
        self._make_record()
        self.client.force_login(self.staff_user)
        response = self.client.get(reverse('campaigns:attribution'))
        self.assertEqual(response.status_code, 200)
        parser = _FormStructureParser()
        parser.feed(response.content.decode())
        self.forms = parser.forms

    def test_no_confirm_submitter_is_gated_by_a_required_control(self):
        for form_id, data in self.forms.items():
            if not data['required']:
                continue
            for name, value, has_formnovalidate in data['submitters']:
                if value == 'dismiss':
                    continue
                self.assertTrue(
                    has_formnovalidate,
                    f'form {form_id!r} owns a required control and a non-dismiss submitter '
                    f'(name={name!r}, value={value!r}) that does not opt out of validation',
                )

    def test_the_invariant_was_actually_exercised(self):
        """Without this non-vacuity guard, the invariant test above would pass trivially if
        the fixture rendered no candidate rows -- the exact false-confidence failure mode
        CR-01 already demonstrated once. At least one form per worklist (events, records)
        must own both a required control and a Confirm submitter."""
        qualifying = [
            form_id
            for form_id, data in self.forms.items()
            if data['required'] and any(value == 'confirm' for _, value, _ in data['submitters'])
        ]
        self.assertGreaterEqual(
            len(qualifying), 2, f'expected at least 2 qualifying forms (one per worklist), found {qualifying}'
        )

    def test_dismiss_submitter_is_still_gated_client_side(self):
        found_dismiss = False
        for form_id, data in self.forms.items():
            for name, value, has_formnovalidate in data['submitters']:
                if value != 'dismiss':
                    continue
                found_dismiss = True
                self.assertFalse(
                    has_formnovalidate,
                    f'form {form_id!r} Dismiss submitter (name={name!r}) carries the validation opt-out, '
                    'which would defeat the UI-SPEC Dismiss gate',
                )
        self.assertTrue(found_dismiss, 'no Dismiss submitter was found -- fixture rendered no candidate rows')
