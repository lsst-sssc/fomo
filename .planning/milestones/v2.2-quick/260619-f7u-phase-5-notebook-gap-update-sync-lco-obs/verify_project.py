"""Structural gate for the PROJECT.md Stage-vs-Phase note (quick-260619-f7u Task 2)."""

import sys

with open('.planning/PROJECT.md') as f:
    t = f.read()
low = t.lower()

assert 'Stage' in t and 'Phase' in t, 'missing Stage/Phase terms'
assert (
    'granularit' in low or 'different numbering' in low or 'numbering scheme' in low
), 'missing the intentional-different-granularity statement'
assert ('Phases 5-7' in t) or ('Phases 5–7' in t), 'missing Stage3 -> Phases 5-7 mapping'
assert ('Phases 2-3' in t) or ('Phase 2-3' in t) or ('Phases 2–3' in t), 'missing Stage2 -> Phases 2-3 mapping'
assert 'upsert' not in low, 'upsert jargon found (use plain English per CLAUDE.md)'

print('PROJECT.md Stage-vs-Phase note present; mappings correct; no upsert jargon')
sys.exit(0)
