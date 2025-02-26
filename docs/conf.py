# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re

from tesseract_core import __version__

# Set the TESSERACT_API_PATH environment variable to the dummy Tesseract API
# This will be used to instantiate the Tesseract runtime so we can generate api docs
os.environ["TESSERACT_API_PATH"] = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tests", "dummy_tesseract", "tesseract_api.py"
    )
)

project = "Tesseract"
copyright = "2025, Pasteur Labs"
author = "The Tesseract Team @ Pasteur Labs + OSS contributors"

# The short X.Y version
parsed_version = re.match(r"(\d+\.\d+\.\d+)", __version__)
if parsed_version:
    version = parsed_version.group(1)
else:
    version = "0.0.0"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_click.ext",
    # Copy button for code blocks
    "sphinx_copybutton",
    # OpenGraph metadata for social media sharing
    "sphinxext.opengraph",
    # For tab-set directive
    "sphinx_design",
    # For nice rendering of Pydantic models
    "sphinxcontrib.autodoc_pydantic",
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

templates_path = []
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
}
html_css_files = ["custom.css"]


# -- Custom directives ----------------------------------------------------


def zip_examples_folder() -> None:
    """Zip a folder and save it to the specified path."""
    import shutil
    from pathlib import Path

    here = Path(__file__).parent

    root_dir = (here / "..").resolve()
    archive_path = here / "downloads" / "examples.zip"

    shutil.make_archive(archive_path.with_suffix(""), "zip", root_dir, "examples")
    assert archive_path.exists()


def setup(app) -> None:
    """Sphinx setup function. Used to register custom stuff."""
    # HACK: We zip the examples folder here so that it can be downloaded
    zip_examples_folder()
