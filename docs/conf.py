# Configuration file for the Sphinx documentation builder.

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

project = "MIR"
author = "Junyu Chen"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Avoid importing heavy dependencies when building docs on CI.
autodoc_mock_imports = [
    "antspyx",
    "einops",
    "matplotlib",
    "ml_collections",
    "nibabel",
    "numpy",
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
