# Configuration file for the Sphinx documentation builder.

import os
import sys
import types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

# Provide lightweight stubs for optional bundled modules that may be absent
# from the repository (e.g., due to license restrictions).
def _register_stub(mod_name: str) -> None:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

_register_stub("MIR.models.fireants")
_register_stub("MIR.models.fireants.io")
_register_stub("MIR.models.fireants.registration")
_register_stub("MIR.models.fireants.utils")
_register_stub("MIR.models.fireants.losses")
_register_stub("MIR.models.fireants.interpolator")

project = "MIR"
author = "Junyu Chen"
copyright = "2026 Junyu Chen"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# Avoid importing heavy dependencies when building docs on CI.
autodoc_mock_imports = [
    "ants",
    "antspyx",
    "einops",
    "MIR.models.fireants",
    "MIR.models.fireants.io",
    "MIR.models.fireants.registration",
    "MIR.models.fireants.utils",
    "MIR.models.fireants.losses",
    "MIR.models.fireants.interpolator",
    "matplotlib",
    "ml_collections",
    "nibabel",
    "numpy",
    "pandas",
    "pymedio",
    "scipy",
    "seaborn",
    "SimpleITK",
    "statsmodels",
    "timm",
    "torch",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "MIR: Medical Image Registration toolkit"
html_short_title = "MIR"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "prev_next_buttons_location": "both",
}
html_context = {
    "display_github": True,
    "github_user": "junyuchen245",
    "github_repo": "MIR",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
pygments_style = "friendly"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# MIR docstrings describe the forward pass with "Forward inputs:" / "Forward outputs:" sections
napoleon_custom_sections = [("Forward inputs", "params_style"), ("Forward outputs", "returns_style"),
                            ("Inputs", "params_style"), ("Returns", "returns_style")]
