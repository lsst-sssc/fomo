---
phase: quick-260722-uhh
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/admin.py
  - solsys_code/tests/test_admin.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "A superuser can open /admin/tom_targets/target/ and it returns 200 (no AlreadyRegistered/NotRegistered crash)."
    - "The Target admin change-list shows a 'By type' filter that restricts rows to SIDEREAL or NON_SIDEREAL."
    - "The Target admin change-list displays name, type, ra, dec columns."
  artifacts:
    - "solsys_code/admin.py registers a custom TargetAdmin for tom_targets.models.Target."
    - "solsys_code/tests/test_admin.py covers the new Target admin change-list + type filter."
  key_links:
    - "admin.site.unregister(Target) MUST run before admin.site.register(Target, TargetAdmin) — Target is already registered by tom_targets' own admin.py."
---

<objective>
Register a custom Django admin for tom_targets' `Target` model in `solsys_code/admin.py`
so staff can filter Targets by `type` (SIDEREAL vs NON_SIDEREAL) in the admin change-list.

tom_targets registers `Target` with a bare `ModelAdmin` (no `list_filter`/`list_display`),
so the /admin change-list is unfilterable and only shows `__str__`. FOMO overrides this by
unregistering the upstream admin and registering its own, following the exact
`CampaignRunAdmin`/`CalendarEventTelescopeLabelAdmin` pattern already in this file.

Purpose: give staff a filterable Target change-list in the admin.
Output: custom `TargetAdmin` + admin-test-client coverage.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

# The file being modified — mirror this exact ModelAdmin + register-at-bottom pattern:
@solsys_code/admin.py

# Admin-test-client precedent (quick task 260714-jpd) — mirror its TestCase + reverse() style:
@solsys_code/tests/test_admin.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Register custom TargetAdmin with list_filter on type</name>
  <files>solsys_code/admin.py</files>
  <action>
    Import `Target` from `tom_targets.models` (add alongside the existing
    `from solsys_code.models import CalendarEventTelescopeLabel, CampaignRun`).

    Add a `TargetAdmin(admin.ModelAdmin)` class matching the two existing admin classes in
    this file. Carry the same `# noqa: D101` inline comment the existing classes use (keep
    style consistent; if `ruff check .` shows D101 is not actually raised here, still match
    the existing two classes' convention). Set:
      - `list_display = ['name', 'type', 'ra', 'dec']` (cheap per-row scalar fields only —
        no relational/computed fields that would trigger N+1 queries).
      - `list_filter = ['type']` (the actual ask: sidereal vs non-sidereal). Do NOT add a
        targetlist/campaign filter — out of scope.
      - `search_fields = ['name']` (matches the existing admin classes' inclusion of
        search_fields).

    Do NOT replicate tom_targets' `inlines = [TargetExtraInline]` — losing the TargetExtra
    inline in the admin is an accepted tradeoff, not a bug to fix here.

    At the bottom of the file, alongside the two existing `admin.site.register(...)` calls,
    add `admin.site.unregister(Target)` immediately followed by
    `admin.site.register(Target, TargetAdmin)`. The unregister MUST come first and run exactly
    once — Target is already registered by tom_targets' own admin.py, so a bare re-register
    raises AlreadyRegistered, and a duplicate unregister raises NotRegistered.

    Run `ruff check . --fix` and `ruff format .` on the touched file before finishing.
  </action>
  <verify>
    <automated>DJANGO_SETTINGS_MODULE=src.fomo.settings python -c "import django; django.setup(); from django.contrib import admin; from tom_targets.models import Target; a=admin.site._registry[Target]; assert list(a.list_filter)==['type'], a.list_filter; assert list(a.list_display)==['name','type','ra','dec'], a.list_display; print('OK')"</automated>
  </verify>
  <done>
    admin.site registry maps Target to the custom TargetAdmin with list_filter==['type'] and
    list_display==['name','type','ra','dec']; Django setup imports without AlreadyRegistered/
    NotRegistered; ruff check/format clean.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add admin-test-client coverage for the Target change-list + type filter</name>
  <files>solsys_code/tests/test_admin.py</files>
  <action>
    Add a new TestCase (mirroring `AdminRegistrationAndGatingTests`' superuser + force_login +
    `reverse(...)` style) that asserts the new Target admin behaves. Because this test is
    specifically about the SIDEREAL vs NON_SIDEREAL distinction, it is legitimate here to
    fixture BOTH a sidereal and a non-sidereal Target — use
    `tom_targets.tests.factories.NonSiderealTargetFactory` for the non-sidereal one and
    `tom_targets.tests.factories.SiderealTargetFactory` for the sidereal one (per CLAUDE.md,
    a sidereal fixture is only acceptable because this test's whole point is verifying the
    type filter distinguishes the two).

    Cover:
      (a) `reverse('admin:tom_targets_target_changelist')` returns 200 for a logged-in
          superuser.
      (b) Requesting the change-list with `{'type__exact': 'SIDEREAL'}` returns 200 and the
          response contains the sidereal target's name but not the non-sidereal target's name;
          and symmetrically `{'type__exact': 'NON_SIDEREAL'}` contains the non-sidereal name
          but not the sidereal one. (Confirm the actual Target.type stored values via the
          factory-created objects — `.type` — rather than assuming the string literal, in
          case the choices differ.)

    Run `ruff check . --fix` and `ruff format .` on the touched file.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_admin -v2</automated>
  </verify>
  <done>
    `./manage.py test solsys_code.tests.test_admin` passes, including the new Target
    change-list-loads and type-filter tests; ruff clean.
  </done>
</task>

</tasks>

<verification>
- `./manage.py test solsys_code.tests.test_admin` passes (existing CampaignRun/label tests
  plus the new Target tests).
- `ruff check .` and `ruff format --check .` are clean.
- Manual sanity (optional): `./manage.py runserver`, log in as superuser, open
  /admin/tom_targets/target/, confirm the "By type" filter sidebar appears and filters rows.
</verification>

<success_criteria>
- solsys_code/admin.py registers a custom TargetAdmin (unregister-then-register) with
  list_filter==['type'], list_display==['name','type','ra','dec'], search_fields==['name'].
- The admin index / Target change-list loads without AlreadyRegistered or NotRegistered.
- New admin-test-client tests prove the change-list loads and the type filter separates
  sidereal from non-sidereal targets.
</success_criteria>

<output>
Create `.planning/quick/260722-uhh-register-a-custom-django-admin-for-tom-t/260722-uhh-SUMMARY.md` when done
</output>
