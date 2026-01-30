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
html_theme_options = {
    "style_nav_header_background": "#1f2937",
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}

html_css_files = [
    "custom.css",
]
