## Title

`TargetAdmin` defines no `search_fields`, so the Target admin changelist has no search box at all

## Environment

- tomtoolkit 3.0.0
- Django 5.2.16, Python 3.11.13

## Summary

`tom_targets/admin.py:12-14` registers `Target` with an essentially empty `ModelAdmin`:

```python
class TargetAdmin(admin.ModelAdmin):
    model = Target
    inlines = [TargetExtraInline]
```

Because `search_fields` is unset, Django renders **no search box** on `/admin/tom_targets/basetarget/`. With `list_display` and `list_filter` also unset, the changelist is a single unlabelled `__str__` column with no sidebar filters.

On a TOM of any real size the admin is effectively unusable for finding a specific target — you get pages of identical-looking rows and no way to narrow them. That's a shame, because the admin is exactly where you go when something looks wrong with the normal UI, which is how we ran into it.

## Suggested fix

```python
class TargetAdmin(admin.ModelAdmin):
    model = Target
    inlines = [TargetExtraInline]
    search_fields = ('name', 'aliases__name')
    list_display = ('name', 'type', 'created', 'modified')
    list_filter = ('type', 'scheme', 'created')
    ordering = ('-created',)
```

Including `aliases__name` in `search_fields` matters for the same reason as the companion draft `tom_base_alias_search.md` (substitute its issue number when filing): a renamed target keeps its previous designation only as a `TargetName`, and that is frequently the string an operator is searching for.

An inline for `TargetName` alongside the existing `TargetExtraInline` would also be a natural addition, since aliases are otherwise only editable from a separate changelist.

Happy to open a PR.
