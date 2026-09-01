"""Structural gate for the Phase-5 demo-notebook extension (quick-260619-f7u Task 1).

Asserts the notebook parses as valid JSON, demonstrates SELECT-02/03/04/05, includes
a SOAR fixture and the SELECT-05 test pointer, and has cleared code-cell outputs
(consistent with the repo's pre-commit output-clearing convention).
"""

import json
import sys

NB = 'docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb'

with open(NB) as f:
    nb = json.load(f)
cells = nb['cells']
src = '\n'.join(''.join(c.get('source', [])) for c in cells)

assert all(c.get('cell_type') in ('code', 'markdown') for c in cells), 'bad cell_type'
for c in cells:
    if c.get('cell_type') == 'code':
        assert c.get('outputs', []) == [], 'code cells must have empty outputs (pre-commit clears them)'

for token in ('SELECT-02', 'SELECT-03', 'SELECT-04', 'SELECT-05'):
    assert token in src, f'missing {token}'

assert 'test_select_05_soar_record_uses_soar_facility_instance' in src, 'missing SELECT-05 test pointer'
assert "'SOAR'" in src or '"SOAR"' in src, 'missing SOAR fixture'
assert src.count('PHASE5') >= 4, 'expected Phase-5 fixtures (PHASE5-* proposals)'
assert (
    'SiderealTargetFactory' not in src or 'NonSiderealTargetFactory' in src
), 'if a Target factory is used it must be NonSiderealTargetFactory'

print('notebook JSON valid; SELECT-02/03/04/05 + SOAR + SELECT-05 pointer present; code outputs clear')
sys.exit(0)
