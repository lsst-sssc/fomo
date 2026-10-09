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
import inspect
import re
from collections import defaultdict

import django_tables2 as tables
from django.template.loader import get_template
from django.test import SimpleTestCase
from django.urls import reverse
from django.utils import timezone

from solsys_code import campaign_tables
from solsys_code.models import CampaignRunObservation, ObservationRecordDismissal
from solsys_code.tests.test_campaign_attribution_views import AttributionViewTestBase

TEMPLATE_NAME = 'campaigns/attribution_queue.html'
CONFIRMED_SECTION_ID = 'attribution-confirmed-section'
DISMISSED_SECTION_ID = 'attribution-dismissed-section'
SECTION_IDS = (CONFIRMED_SECTION_ID, DISMISSED_SECTION_ID)


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


class _CollapseSectionParser(html.parser.HTMLParser):
    """Records the attributes of the two collapsible section ``div``s (matched by id) and of
    every ``button`` carrying a ``data-bs-target``, so a test can assert each section's
    open/closed state and its trigger's matching ``aria-expanded`` and ``collapsed`` class."""

    def __init__(self):
        super().__init__()
        self.sections: dict[str, dict] = {}
        self.triggers: dict[str, dict] = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if attrs_dict.get('id') in SECTION_IDS:
            self.sections[attrs_dict['id']] = attrs_dict
        if tag == 'button' and 'data-bs-target' in attrs_dict:
            self.triggers[attrs_dict['data-bs-target']] = attrs_dict

    @staticmethod
    def classes(attrs: dict) -> set[str]:
        return set((attrs.get('class') or '').split())


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

    def test_collapse_triggers_use_bootstrap5_attributes(self):
        """G-37.1-1: ``tom_common/base.html`` loads Bootstrap 5, whose collapse plugin binds only
        ``data-bs-toggle``/``data-bs-target``. The Bootstrap 4 spelling (``data-toggle``,
        ``data-target``) is inert there, so the Confirmed and Dismissed headings opened nothing.
        Exactly two triggers, one per section, none using a Bootstrap 4 plugin attribute."""
        collapse_buttons = [b for b in re.findall(r'<button\b[^>]*>', self.source) if 'data-bs-toggle' in b]
        self.assertEqual(len(collapse_buttons), 2, f'expected exactly 2 collapse triggers, found {collapse_buttons}')
        targets = []
        for button in collapse_buttons:
            self.assertIn('data-bs-toggle="collapse"', button)
            match = re.search(r'data-bs-target="#([^"]+)"', button)
            self.assertIsNotNone(match, f'collapse trigger has no data-bs-target: {button!r}')
            targets.append(match.group(1))
        self.assertCountEqual(targets, SECTION_IDS)
        self.assertIsNone(
            re.search(r'\bdata-(toggle|target|dismiss)=', self.source),
            'the template still carries a Bootstrap 4 plugin data attribute',
        )

    def test_every_campaign_table_names_a_bootstrap_template(self):
        """G-37.1-1-pager: a django-tables2 table with no template_name falls back to the
        unstyled default pager ("12next"); every table in campaign_tables must name a Bootstrap one."""
        table_classes = [
            cls
            for _, cls in inspect.getmembers(campaign_tables, inspect.isclass)
            if issubclass(cls, tables.Table) and cls.__module__ == campaign_tables.__name__
        ]
        self.assertGreaterEqual(len(table_classes), 4)
        offenders = [
            f'{cls.__name__}: {cls._meta.template_name}'
            for cls in table_classes
            if not cls._meta.template_name.startswith('django_tables2/bootstrap')
        ]
        self.assertEqual(offenders, [], 'campaign tables without a Bootstrap template')


class AttributionPagerRenderTests(AttributionViewTestBase):
    """G-37.1-1-pager: the Confirmed and Dismissed tables paginate with Bootstrap 5 controls,
    not the unstyled default list. GET only -- see the module docstring."""

    def setUp(self):
        for offset in (0, 1):
            record = self._make_record(night_offset=offset)
            CampaignRunObservation.objects.create(
                run=self.campaign_run, observation_record=record, confirmed_at=timezone.now()
            )
        for offset in (2, 3):
            record = self._make_record(night_offset=offset)
            ObservationRecordDismissal.objects.create(
                observation_record=record,
                run=self.campaign_run,
                dismissed_by=self.staff_user,
                dismissed_at=timezone.now(),
                reason='Wrong night',
            )
        self.client.force_login(self.staff_user)

    def _get_html(self, data):
        response = self.client.get(reverse('campaigns:attribution'), data)
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_confirmed_pager_uses_bootstrap5_markup(self):
        content = self._get_html({'confirmed-per_page': '1'})
        section = content[content.index(f'id="{CONFIRMED_SECTION_ID}"') :]
        self.assertIn('class="page-link"', section)
        self.assertIn('page-item', section)
        self.assertIn('confirmed-page=2', section)
        self.assertIn('table-responsive', section)

    def test_dismissed_pager_uses_bootstrap5_markup(self):
        content = self._get_html({'dismissed-per_page': '1'})
        start = content.index(f'id="{DISMISSED_SECTION_ID}"')
        section = content[start : content.index(f'id="{CONFIRMED_SECTION_ID}"')]
        self.assertIn('class="page-link"', section)
        self.assertIn('page-item', section)
        self.assertIn('dismissed-page=2', section)
        self.assertIn('table-responsive', section)


class AttributionCollapseSectionRenderTests(AttributionViewTestBase):
    """G-37.1-1: renders the real page through a GET and parses the section markup a browser
    would receive. GET only -- see the module docstring."""

    def setUp(self):
        self.system_record = self._make_record()
        CampaignRunObservation.objects.create(
            run=self.campaign_run, observation_record=self.system_record, confirmed_at=timezone.now()
        )
        self.dismissed_record = self._make_record(night_offset=1)
        ObservationRecordDismissal.objects.create(
            observation_record=self.dismissed_record,
            run=self.campaign_run,
            dismissed_by=self.staff_user,
            dismissed_at=timezone.now(),
            reason='Wrong night',
        )
        self.client.force_login(self.staff_user)

    def _parse(self, data=None):
        response = self.client.get(reverse('campaigns:attribution'), data or {})
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        parser = _CollapseSectionParser()
        parser.feed(content)
        return content, parser

    def test_confirmed_section_renders_open_with_the_system_link_inside(self):
        content, parser = self._parse()
        section = parser.sections[CONFIRMED_SECTION_ID]
        self.assertEqual(parser.classes(section), {'collapse', 'show'})
        trigger = parser.triggers[f'#{CONFIRMED_SECTION_ID}']
        self.assertEqual(trigger.get('data-bs-toggle'), 'collapse')
        self.assertEqual(trigger.get('aria-expanded'), 'true')
        self.assertNotIn('collapsed', parser.classes(trigger))
        confirmed_start = content.index(f'id="{CONFIRMED_SECTION_ID}"')
        self.assertIn('System (exact match)', content[confirmed_start:])

    def test_dismissed_section_renders_folded_by_default(self):
        _, parser = self._parse()
        section = parser.sections[DISMISSED_SECTION_ID]
        self.assertIn('collapse', parser.classes(section))
        self.assertNotIn('show', parser.classes(section))
        trigger = parser.triggers[f'#{DISMISSED_SECTION_ID}']
        self.assertEqual(trigger.get('aria-expanded'), 'false')
        self.assertIn('collapsed', parser.classes(trigger))

    def test_dismissed_section_renders_open_while_paging_its_own_table(self):
        _, parser = self._parse({'dismissed-page': '1'})
        section = parser.sections[DISMISSED_SECTION_ID]
        self.assertEqual(parser.classes(section), {'collapse', 'show'})
        trigger = parser.triggers[f'#{DISMISSED_SECTION_ID}']
        self.assertEqual(trigger.get('aria-expanded'), 'true')
        self.assertNotIn('collapsed', parser.classes(trigger))


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
