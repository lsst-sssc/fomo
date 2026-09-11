"""Repo-level guard over the committed ``project_observation_calendar_demo.ipynb`` evidence.

This module makes the paired demo notebook's takeover demonstration checkable on every test
run, not only when someone re-executes the notebook by hand (G-34-3). It loads the committed
notebook once and asserts that its executed output still demonstrates what CLAUDE.md's
paired-docs rule and the PROJ-05 / TRIG-03 / SCHED-06 requirements need: the run says which
database it used, a scratch-routed run shows a real one-time takeover with a differing first
and second sweep, every run's second sweep converges to zero per facility, and the SCHED-06
baseline JSON stays in lockstep with the notebook's own printed ``captured_at``.

``FOMO_DEMO_NOTEBOOK_PATH`` overrides the notebook path this suite loads. This is the hook
that lets this plan prove the guard fails for the right reason, by pointing it at a copy of
the notebook whose takeover evidence (cell ``05528b38``'s outputs) has been deliberately
emptied.

No database access: every test here is a pure function of two files already on disk.
"""

import json
import os
import re
from pathlib import Path

from django.test import SimpleTestCase

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_NOTEBOOK_PATH = _REPO_ROOT / 'docs' / 'notebooks' / 'pre_executed' / 'project_observation_calendar_demo.ipynb'

NOTEBOOK_PATH = Path(os.environ.get('FOMO_DEMO_NOTEBOOK_PATH') or _DEFAULT_NOTEBOOK_PATH)
BASELINE_PATH = NOTEBOOK_PATH.parent / 'project_observation_calendar_demo.sched06-baseline.json'

# Cells are addressed by their nbformat id, never by position, so inserting a cell above one
# of these cannot silently retarget an assertion at the wrong output.
_CELL_ROUTING = '7022f987'
_CELL_TAKEOVER_DIFF = '05528b38'
_CELL_SECOND_SWEEP = '556d2a9f'
_CELL_SCHED06_BASELINE = '250b5d0b'
_REQUIRED_CELL_IDS = (_CELL_ROUTING, _CELL_TAKEOVER_DIFF, _CELL_SECOND_SWEEP, _CELL_SCHED06_BASELINE)


def _cell_output_text_by_id(notebook_path):
    """Map each code cell's nbformat id to its joined stream-output text.

    Args:
        notebook_path: Path to a `.ipynb` file to load.

    Returns:
        dict[str, str]: cell id -> concatenation of that cell's stream-output `text` entries.
        Non-stream output types (errors, rich display data) carry no top-level `text` key and
        are skipped.
    """
    with open(notebook_path) as fh:
        notebook = json.load(fh)
    text_by_id = {}
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        text_by_id[cell.get('id')] = ''.join(''.join(output.get('text', [])) for output in cell.get('outputs', []))
    return text_by_id


class TestProjectorDemoNotebookEvidence(SimpleTestCase):
    """Guards the committed notebook's takeover/convergence/baseline evidence (G-34-3)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cell_text = _cell_output_text_by_id(NOTEBOOK_PATH)
        missing = [cell_id for cell_id in _REQUIRED_CELL_IDS if cell_id not in cls.cell_text]
        if missing:
            raise AssertionError(
                f'Notebook at {NOTEBOOK_PATH} is missing expected cell id(s) {missing} -- addressed '
                'by nbformat id, not position, so a cell must have been removed or its id changed.'
            )
        with open(BASELINE_PATH) as fh:
            cls.baseline = json.load(fh)

    def test_notebook_names_which_database_the_run_used(self):
        """Cell 7022f987's output states which database this run used, one way or the other."""
        routing_text = self.cell_text[_CELL_ROUTING]
        self.assertTrue(
            'the developer database itself' in routing_text or 'routed to a scratch copy' in routing_text,
            f'Cell {_CELL_ROUTING} output names neither the developer database nor a scratch copy: '
            f'{routing_text!r}',
        )

    def test_scratch_routed_run_shows_a_real_takeover_and_diverging_sweeps(self):
        """When routed to a scratch copy, the takeover and second-sweep cells prove real work.

        An un-routed re-execution against an already-converged developer database is the
        legitimate SCHED-06 case (see the notebook's "What happens next" section) and must not
        be blocked by this guard -- so these assertions apply only when cell 7022f987 says the
        run was routed to a scratch copy.
        """
        if 'routed to a scratch copy' not in self.cell_text[_CELL_ROUTING]:
            self.skipTest('This run was not routed to a scratch copy -- takeover assertions do not apply.')

        diff_text = self.cell_text[_CELL_TAKEOVER_DIFF]
        match = re.search(r'^([1-9][0-9]*) of ([0-9]+) pre-existing', diff_text, re.M)
        self.assertIsNotNone(
            match, f'Cell {_CELL_TAKEOVER_DIFF} does not report a non-zero re-titled count: {diff_text!r}'
        )
        self.assertRegex(
            diff_text,
            re.compile(r'^\s+before: ', re.M),
            f'Cell {_CELL_TAKEOVER_DIFF} lists no before -> after title pair.',
        )

        sweep_text = self.cell_text[_CELL_SECOND_SWEEP]
        first_line = next(line for line in sweep_text.splitlines() if line.startswith('First sweep'))
        second_line = next(line for line in sweep_text.splitlines() if line.startswith('Second sweep'))
        self.assertNotEqual(
            first_line.split(':', 1)[1].strip(),
            second_line.split(':', 1)[1].strip(),
            f'Cell {_CELL_SECOND_SWEEP}: first and second sweep summaries are identical.',
        )

    def test_unrouted_run_states_the_run_was_already_converged(self):
        """When un-routed (against the developer database itself), the notebook says so.

        This is the legitimate SCHED-06 re-check case: the developer database is expected to
        already be converged by the receiver alone, so the second-sweep cell's own converged-run
        statement (added alongside the takeover guard) must be present instead of takeover
        evidence.
        """
        if 'the developer database itself' not in self.cell_text[_CELL_ROUTING]:
            self.skipTest('This run was routed to a scratch copy -- the converged-run statement does not apply.')

        self.assertIn(
            'found nothing to take over',
            self.cell_text[_CELL_SECOND_SWEEP],
            f'Cell {_CELL_SECOND_SWEEP} does not state that this un-routed run found nothing to take over.',
        )

    def test_second_sweep_reports_zero_per_facility_segment(self):
        """Every ' | '-separated facility segment of the Second sweep line converges to zero.

        This holds for ANY run, routed or not -- a second sweep over an already-projected
        database must always report no further work.
        """
        sweep_text = self.cell_text[_CELL_SECOND_SWEEP]
        second_line = next(line for line in sweep_text.splitlines() if line.startswith('Second sweep'))
        segments = second_line.split(' | ')[1:]
        self.assertTrue(segments, f'No facility segments found in the Second sweep line: {second_line!r}')
        for segment in segments:
            for token in ('created: 0', 'updated: 0', 'site_lookups: 0'):
                self.assertIn(token, segment, f'{token!r} missing from facility segment {segment!r}')

    def test_sched06_cell_quotes_the_committed_baseline_captured_at(self):
        """Cell 250b5d0b's output quotes the committed baseline JSON's own `captured_at`.

        Ties the notebook and the SCHED-06 evidence file together so a swapped or stale
        baseline is caught even though the two files are otherwise independent.
        """
        self.assertIn(
            self.baseline['captured_at'],
            self.cell_text[_CELL_SCHED06_BASELINE],
            f'Cell {_CELL_SCHED06_BASELINE} does not quote the committed baseline captured_at '
            f'({self.baseline["captured_at"]!r}).',
        )

    def test_no_credential_shaped_text_in_the_notebook_file(self):
        """Scans the raw notebook file, not the parsed outputs, so a cell source cannot hide it."""
        raw_text = NOTEBOOK_PATH.read_text()
        hits = sorted(set(re.findall(r'api_key|Authorization|token=', raw_text, re.I)))
        self.assertFalse(hits, f'Credential-shaped text found in {NOTEBOOK_PATH}: {hits}')
