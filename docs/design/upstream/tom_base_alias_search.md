## Title

General Search on the target list no longer matches aliases, and the documented `GENERAL_SEARCH_FUNCTIONS` key silently matches nothing

## Environment

- tomtoolkit 3.0.0
- Django 5.2.16, Python 3.11.13

## Summary

Two related problems in the new HTMX target list:

1. The **General Search** box matches `Target.name` only. A target that has been renamed keeps its old designation as a `TargetName` alias, and becomes unfindable under it.
2. The `GENERAL_SEARCH_FUNCTIONS` escape hatch that would let a project fix this is documented with a **key that never matches**, so the override is silently ignored.

## 1. General Search ignores aliases

`TargetFilterSet.general_search` (`tom_targets/filters.py:263-277`) ends with:

```python
return queryset.filter(Q(name__icontains=value))
```

The `TargetFilterSet` already contains a filter that does the right thing — `filter_name` at `tom_targets/filters.py:143-150`:

```python
q_set |= Q(name__icontains=term) | Q(aliases__name__icontains=term)
```

…but it carries the comment `# NOTE: this field is not displayed; the 'query' field is used instead`, and it is left out of the crispy `Layout` in the `form` property. So the alias-aware filter still exists but is no longer reachable from the UI, and the box that replaced it dropped the alias support. This reads like an oversight in the move to the HTMX filter set rather than an intentional narrowing.

Inheriting `HTMXTableFilterSet.general_search` would not help either: it sweeps model fields but explicitly skips `many_to_many`/`many_to_one`, and `aliases` is a reverse FK.

### Reproducer

```python
t = Target.objects.create(name='2026 RW1', type='NON_SIDEREAL')
t.aliases.create(name='CERNQ52')
```

- `/targets/?query=CERNQ52` → no results
- `/targets/?name=CERNQ52` → finds it (the hidden filter still works via querystring)

This is not a contrived case. It is the normal life cycle of a solar-system target: we ingest NEOCP candidates under their temporary tracklet designation and rename them when the MPC designates the object, keeping the trksub as an alias. After the rename, the designation everyone searched by for the first few days stops working. `TargetNameSearchView` handles this correctly, so the two search paths in the toolkit now disagree.

### Suggested fix

Restore alias matching in `TargetFilterSet.general_search`:

```python
return queryset.filter(
    Q(name__icontains=value) | Q(aliases__name__icontains=value)
).distinct()
```

The `.distinct()` matters — without it a target whose name *and* alias both match is returned twice by the join.

## 2. The documented `GENERAL_SEARCH_FUNCTIONS` key matches nothing

`HTMXTableFilterSet.get_general_search_function` (`tom_common/htmx_table.py`) builds its lookup key from the concrete model:

```python
model_label = f'{self.Meta.model._meta.app_label}.{self.Meta.model.__name__}'
```

For `TargetFilterSet`, `Meta.model` is `Target`, which — with no `TARGET_MODEL_CLASS` configured — *is* `tom_targets.base_models.BaseTarget`. So the computed key is:

```
tom_targets.BaseTarget
```

But the docstring immediately above it tells developers to write:

```python
GENERAL_SEARCH_FUNCTIONS = {
    'tom_targets.Target': 'custom_code.search.my_target_search',
}
```

`'tom_targets.Target'` never matches. `search_map.get(model_label)` returns `None`, the method falls back to `self.general_search`, and **no error or warning is raised** — the override just does nothing. We lost time to this: the settings entry looked correct, the app booted fine, and the search box silently kept its old behaviour. Only an end-to-end test through `TargetFilterSet(...).qs` caught it.

The same trap applies to any `HTMXTableFilterSet`, not just targets.

### Suggested fix

Either correct the docstring example to `'tom_targets.BaseTarget'`, or — better — make the lookup resilient and self-documenting:

- try the swappable label (`settings.TARGET_MODEL_CLASS`) and the concrete label, and/or
- `logger.warning(...)` when `GENERAL_SEARCH_FUNCTIONS` is non-empty but no key matched the model being filtered.

A silent no-op is the worst outcome for a settings-driven hook, since nothing in the running system indicates the setting was ignored.

## Workaround

For anyone hitting this before a fix lands, this works on 3.0.0:

```python
# settings.py  -- note BaseTarget, not Target
GENERAL_SEARCH_FUNCTIONS = {
    'tom_targets.BaseTarget': 'myapp.search.target_general_search',
}
```

```python
# myapp/search.py
from django.db.models import Q

def target_general_search(queryset, name, value):
    if not value:
        return queryset
    return queryset.filter(
        Q(name__icontains=value) | Q(aliases__name__icontains=value)
    ).distinct()
```

Happy to open a PR for either or both parts if that's useful.
