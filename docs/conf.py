import pyqg

project = "pyqg"
copyright = "PyQG team"
author = "PyQG team"
version = pyqg.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.extlinks',
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_nb",
]

extlinks = {"issue": ("https://github.com/pyqg/pyqg/issues/%s", "GH%s")}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = ["css/pyqg-jax-fix-theme.css"]
suppress_warnings = ["epub.unknown_project_files"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

# Theme
html_theme = "sphinx_rtd_theme"

# Autosummary config
autosummary_generate = True
autoclass_content = "both"

# MyST-NB configuration
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_execution_mode = "off"
myst_enable_extensions = {"dollarmath"}
myst_dmath_double_inline = True

# Autodoc configuration
autodoc_mock_imports = []
autodoc_typehints = "none"
autodoc_member_order = "alphabetical"

# Napoleon configuration
napoleon_google_docstring = False
