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

# The pre-commit sphinx-build hook overrides exclude_patterns to skip
# notebooks/* for speed (avoids executing/rendering .ipynb files on every
# commit). That intentionally leaves docs/notebooks.rst's toctree entry
# pointing at an excluded document during local pre-commit builds, which
# would otherwise emit a 'toctree contains reference to excluded document'
# warning on every commit. Full builds (ReadTheDocs, CI) don't apply that
# override, so this only ever suppresses the pre-commit-local false positive.
suppress_warnings = ['toc.excluded']

# This assumes that sphinx-build is called from the root directory
master_doc = 'index'
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False

autoapi_type = 'python'
autoapi_dirs = ['../src']
autoapi_ignore = ['*/__main__.py', '*/_version.py']
autoapi_add_toc_tree_entry = False
autoapi_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
# Add following to allow notebook execution errors (e.g. interactive cells)
nbsphinx_allow_errors = True
