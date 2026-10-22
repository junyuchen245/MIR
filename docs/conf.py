# Configuration file for the Sphinx documentation builder.
# Build locally:  pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html

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


for _m in ("", ".io", ".registration", ".utils", ".losses", ".interpolator"):
    _register_stub("MIR.models.fireants" + _m)

# -- Project information -----------------------------------------------------
project = "MIR"
author = "Junyu Chen"
copyright = "2026, Junyu Chen"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_scripts", "_redirects", "Thumbs.db", ".DS_Store"]

# -- API documentation -------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# MIR docstrings describe the forward pass with "Forward inputs:" / "Forward outputs:" sections
napoleon_custom_sections = [("Forward inputs", "params_style"), ("Forward outputs", "returns_style"),
                            ("Inputs", "params_style"), ("Returns", "returns_style")]

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

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "MIR"
html_short_title = "MIR"
html_logo = "_static/mir_logo.svg"
html_favicon = "_static/mir_logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Old page names of the previous layout -> redirect stubs, so existing links keep working.
html_extra_path = ["_redirects"]
html_show_sourcelink = False
html_copy_source = False

_brand = {
    "light": {"primary": "#1f5f9e", "content": "#1b68b3", "accent": "#0f8a7e"},
    "dark": {"primary": "#6cb2ff", "content": "#7cbaff", "accent": "#4fd1c1"},
}
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view"],
    "source_repository": "https://github.com/junyuchen245/MIR/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": _brand["light"]["primary"],
        "color-brand-content": _brand["light"]["content"],
        "color-brand-visited": _brand["light"]["content"],
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace",
        "mir-accent": _brand["light"]["accent"],
    },
    "dark_css_variables": {
        "color-brand-primary": _brand["dark"]["primary"],
        "color-brand-content": _brand["dark"]["content"],
        "color-brand-visited": _brand["dark"]["content"],
        "mir-accent": _brand["dark"]["accent"],
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/junyuchen245/MIR",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                    '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53'
                    '.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
                    '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 '
                    '2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
                    '3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z">'
                    '</path></svg>',
            "class": "",
        },
    ],
}
pygments_style = "friendly"
pygments_dark_style = "github-dark"

# -- sphinx-copybutton: strip prompts when copying ---------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
