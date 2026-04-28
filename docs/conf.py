"""Sphinx configuration for spectroxide documentation."""

import os
import sys

# Make the Python package importable
sys.path.insert(0, os.path.abspath("../python"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "spectroxide"
author = "Ethan Baker, Hongwan Liu, Siddharth Mishra-Sharma"
copyright = "2024, Ethan Baker, Hongwan Liu, Siddharth Mishra-Sharma"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings from Python source
    "sphinx.ext.napoleon",       # parse NumPy-style docstrings
    "sphinx.ext.linkcode",       # add [source] links to GitHub
    "sphinx.ext.intersphinx",    # cross-ref numpy, scipy docs
    "sphinx.ext.autosummary",    # generate summary tables
    "sphinx_design",             # grid cards, tab-sets, badges
    "nbsphinx",                  # render Jupyter notebooks
    "nbsphinx_link",             # reference notebooks outside source root
    "sphinx_copybutton",         # copy-button on code blocks
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,      # skip undocumented members
    "private-members": False,    # skip _ prefixed members
    "show-inheritance": True,
    "member-order": "bysource",  # preserve source order
}
autoclass_content = "class"      # use class docstring only; avoids dataclass field duplication
# Strip type hints from the signature line (NumPy docstrings already document
# parameter types in the Parameters section; duplicating in the signature is
# noisy, especially with ArrayLike / NDArray[np.float64]).
autodoc_typehints = "none"
autodoc_typehints_format = "short"
suppress_warnings = [
    "ref.duplicate",  # harmless dataclass field re-export duplicates
    "config.cache",   # nbsphinx_custom_formats holds a function and is unpicklable
]

# Define substitutions used in firas.py docstrings (|A|, |μ|, etc.)
# RST interprets |x| as substitution references; these define them as plain text.
rst_prolog = """
.. |A| replace:: *A*
.. |A_hat| replace:: *A*\\ :sub:`hat`
.. |μ| replace:: *\\u03bc*
.. |y| replace:: *y*
"""

# linkcode: point [source] links to GitHub
def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/bakerem/spectroxide/blob/main/python/{filename}.py"


# Result/data containers: dataclasses where the auto-generated __init__
# signature dumps every field into the class header. Users get these via
# solve() / load_or_build_*(), not by hand-constructing — so the signature
# is more noise than help. Napoleon renders the docstring's "Attributes"
# section as an :ivar: fieldlist below.
_SUPPRESS_CLASS_SIGNATURE = {
    "spectroxide.solver.SolverResult",
    "spectroxide.greens_table.GreensTable",
    "spectroxide.greens_table.PhotonGreensTable",
    "spectroxide.cosmology.Cosmology",
    "spectroxide.firas.FIRASData",
}


def _suppress_class_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and name in _SUPPRESS_CLASS_SIGNATURE:
        return ("", return_annotation)
    return None


def setup(app):
    app.connect("autodoc-process-signature", _suppress_class_signature)
# Nitpicky cross-reference suppressions. NumPy-style docstrings put free
# words ("optional", "callable", "array_like", set literals like
# {"pde", "table"}) in the type slot; Sphinx then tries to resolve each
# as a class and emits warnings under -n. Silence them — they are not
# real cross-references.
nitpick_ignore = [
    ("py:class", word)
    for word in [
        "optional",
        "array_like",
        "array-like",
        "callable",
        "Callable",
        "ndarray",
        "np.ndarray",
        "float64",
        "shape",
        "sequence",
        "Sequence",
        "Mapping",
        "Path",
        "os.PathLike",
    ]
]
nitpick_ignore_regex = [
    # NumPy literal-set type specs: {"pde", "greens_function", "table"} etc.
    ("py:class", r"^[\"\{].*[\"\}]$"),
    ("py:class", r"^.*\}$"),
    # Plain prose accidentally captured by Napoleon's type-slot parser.
    ("py:class", r"^[A-Z][a-z][a-zA-Z\- ]+$"),
    # Numeric literals (e.g. "43" from a shape spec).
    ("py:class", r"^[0-9]+$"),
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False       # rtype in Returns block, not separate field
# Render the "Attributes" section as a flat :ivar:/:vartype: fieldlist
# (one line per attribute, type inline) instead of one .. attribute::
# block per field with a stacked "Type:" sub-row. Matches the look of
# the "Parameters" section on functions like solve().
napoleon_use_ivar = True

# ---------------------------------------------------------------------------
# intersphinx: link to external docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# ---------------------------------------------------------------------------
# nbsphinx settings
# ---------------------------------------------------------------------------
nbsphinx_execute = "never"      # don't re-run notebooks on every build
nbsphinx_allow_errors = False
nbsphinx_timeout = 600

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "spectroxide"
html_theme_options = {
    "github_url": "https://github.com/bakerem/spectroxide",
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "icon_links": [],
    "use_edit_page_button": False,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "codex_suggestions.md",
    "cosmotherm_heating_rate_analysis.md",
    "heating_rate_conventions.md",
]
templates_path = ["_templates"]
