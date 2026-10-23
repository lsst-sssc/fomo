<!-- 
Thank you for your contribution to the repo :)

Pull Request (PR) Instructions:
Provide a general summary of your changes in the Title above. Fill out each section of the template, and replace the space with an `x` in all the boxes that apply. If you're unsure about any of these, don't hesitate to ask. We're here to help! Once you are satisfied with the pull request, click the "Create pull request" button to submit it for review.

How to link to a PR:
https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue 
-->

## Change Description
<!--- 
Describe your changes in detail. In your description, you should answer questions like "Why is this change required? What problem does it solve?".

If it fixes an open issue, please link to the issue here. If this PR closes an issue, put the word 'closes' before the issue link to auto-close the issue when the PR is merged, e.g.

Closes #???
-->
- [ ] My PR includes a link to the issue that I am addressing



## Solution Description
<!-- Please explain the technical solution that I have provided and how it addresses the issue or feature being implemented -->



## Code Quality
- [ ] I have read the Contribution Guide and agree to the Code of Conduct
- [ ] My code follows the code style of this project
- [ ] My code builds (or compiles) cleanly without any errors or warnings
- [ ] My code contains relevant comments and necessary documentation
- [ ] I have added or updated tests under `solsys_code/` for the change (`python manage.py test`)

## FOMO-Specific Checklist
<!--- Tick only the items that apply to this change; delete the rest. -->
- [ ] **Models:** I ran `python manage.py makemigrations` and the resulting migration(s) are committed
- [ ] **Settings:** Changes to `src/fomo/settings.py` that a deployment must mirror in `local_settings.py` are called out above
- [ ] **Ephemerides:** Changes to `solsys_code/ephem_utils.py` or the ephemeris views state whether the computed positions/rates/magnitudes change, and `test_ephem_utils` reference values are updated if so
- [ ] **Dependencies:** Changes to `pyproject.toml` dependencies are reflected in the Requirements list in `docs/installation.rst`
- [ ] **Docs:** New or changed user-facing behaviour is documented under `docs/` (and docstrings use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html))
