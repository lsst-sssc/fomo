# Technology Stack

**Analysis Date:** 2026-06-12

## Languages

**Primary:**
- Python 3.10+ - Core application and TOM Toolkit backend (Django-based)

**Secondary:**
- HTML/CSS/JavaScript - Django templates and frontend components (Bootstrap 4 based)

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12 (tested across versions via GitHub Actions)

**Package Manager:**
- `pip` - Python package management
- Lockfile: `pyproject.toml` (PEP 517/518 compliant)

## Frameworks

**Core:**
- Django 2.1+ (via TOM Toolkit) - Web framework for TOM Toolkit-based TOM application
- TOM Toolkit 2.31.4+ - Target and Observation Manager framework for Solar System object follow-up
- Django REST Framework - REST API support (`rest_framework`, `rest_framework.authtoken`)
- Django Crispy Forms (`crispy_forms`, `crispy_bootstrap4`) - Form rendering with Bootstrap 4

**Frontend/UI:**
- Bootstrap 4 (`bootstrap4`) - CSS framework
- Plotly (configured in settings, `PLOTLY_THEME = 'plotly_white'`) - Interactive visualization
- Django HTMX (`django_htmx`) - HTMX middleware for AJAX interactions

**Database/ORM:**
- Django ORM (via TOM Toolkit) - Database abstraction and models
- SQLite3 (default development) - File-based database backend

**Testing:**
- pytest - Test runner
- pytest-cov - Code coverage reporting (`--cov` flags in GitHub workflows)

**Build/Dev:**
- setuptools 62+ - Package building
- setuptools_scm 6.2+ - Version management from git tags
- ruff 0.2.1+ - Linting and code formatting (via pre-commit)
- Sphinx - Documentation generation

## Key Dependencies

**Critical:**
- tomtoolkit>=2.31.4 - TOM Toolkit framework for observatory management and observations
- tom_fink>=1.0.0 - Fink alert stream integration
- tom_alertstreams - Alert stream handling framework
- sorcha - Solar System object simulation and planning

**Observatory/Catalog Integrations:**
- tom_eso - ESO (VLT) facility integration
- tom_observations - Core observation facilities (LCO, Gemini, SOAR)
- tom_catalogs - Catalog harvesters (JPL Horizons, MPC, SIMBAD, TNS)
- tom_registration - User registration and management

**Django Core Contrib:**
- django.contrib.auth - Authentication and authorization
- django.contrib.contenttypes - Content type framework
- django.contrib.sessions - Session management
- django.contrib.messages - Messaging framework
- django.contrib.sites - Multi-site framework
- django.contrib.admin - Django admin interface
- django.contrib.staticfiles - Static file serving

**Additional Libraries:**
- django-extensions - Management commands and utilities
- django-guardian - Object-level permissions
- django-comments - Commenting system
- django-filters - Filtering for querysets
- django-tables2 - Table rendering
- django-gravatar - Gravatar integration
- django_gravatar - Avatar display
- numpy>1.24 - Numerical computing (for photometry/data processing)

## Configuration

**Environment:**
- Environment variables via `os.getenv()` (see INTEGRATIONS.md for env var list)
- Django settings module: `src.fomo.settings`
- Local settings override via `local_settings.py` import (fallback: no error on missing)

**Build:**
- `pyproject.toml` - Main configuration (Python 3.10+ required, version dynamic via setuptools_scm)
- `.readthedocs.yml` - ReadTheDocs build configuration (Python 3.10, Sphinx)
- `.pre-commit-config.yaml` - Pre-commit hooks (ruff, pytest, Sphinx, validation)
- Ruff config in `pyproject.toml` - Format style (single quotes), line length 120

## Platform Requirements

**Development:**
- Python 3.10, 3.11, or 3.12
- Git (for setuptools_scm version management)
- SQLite3 support
- Pandoc (optional, for Jupyter notebook rendering in docs)

**Production:**
- Python 3.10+
- SQLite3 or PostgreSQL (configurable via `DATABASES` setting)
- Static file serving setup (via Django `STATIC_URL`, `STATIC_ROOT`, `MEDIA_ROOT`)
- WSGI application server (configured at `src.fomo.wsgi.application`)

**Documentation:**
- Sphinx 2.1+ - HTML documentation generation
- ReadTheDocs - Hosted documentation platform

---

*Stack analysis: 2026-06-12*
