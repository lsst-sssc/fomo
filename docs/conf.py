# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib.metadata import version

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath('../src/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fomo'
copyright = '2024-%Y, SSSC Software'
author = 'SSSC Software'
release = version('fomo')
# for example take major/minor
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']

extensions.append('autoapi.extension')
extensions.append('nbsphinx')

# -- sphinx-copybutton configuration ----------------------------------------
extensions.append('sphinx_copybutton')
## sets up the expected prompt text from console blocks, and excludes it from
## the text that goes into the clipboard.
copybutton_exclude = '.linenos, .gp'
copybutton_prompt_text = '>> '

## lets us suppress the copy button on select code blocks.
copybutton_selector = 'div:not(.no-copybutton) > div.highlight > pre'

templates_path = []
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# This setting dates from a local pre-commit sphinx-build hook that overrode
# exclude_patterns to skip notebooks/* for speed, leaving docs/notebooks.rst's
# toctree entry pointing at an excluded document. That hook was removed in
# Phase 38 (D-07): the docs are now built only by the build-documentation CI
# workflow and ReadTheDocs, which do not exclude notebooks. The setting is kept
# because it is harmless there.
suppress_warnings = ['toc.excluded']

# This assumes that sphinx-build is called from the root directory
master_doc = 'index'
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False

autoapi_type = 'python'
autoapi_dirs = ['../src']
# CR-03 (36-REVIEW.md): 'local_settings.py' is the documented home of every credential
# this project's unattended path needs (LCO/SOAR api_key, EMAIL_HOST_PASSWORD,
# FOMO_HEARTBEAT_URL) -- autoapi (and viewcode) render source verbatim into generated
# HTML, so without this exclusion every configured host's real secrets end up in
# _readthedocs/html/ and docs/_build/html/ on every sphinx-build. ReadTheDocs itself
# builds from a checkout with no local_settings.py, so its published site was never
# affected -- but a build run and served from a configured host was.
autoapi_ignore = ['*/__main__.py', '*/local_settings.py']
autoapi_add_toc_tree_entry = False
autoapi_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
# Add following to allow notebook execution errors (e.g. interactive cells)
nbsphinx_allow_errors = True


def _skip_version_module(app, what, name, obj, skip, options):
    # fomo._version must stay parsed (not in autoapi_ignore) so that
    # `from ._version import __version__` in fomo/__init__.py resolves, but the
    # setuptools_scm-generated module doesn't merit its own API page.
    return True if name == 'fomo._version' else None


def setup(app):
    """Register the Sphinx event handlers."""
    app.connect('autoapi-skip-member', _skip_version_module)
