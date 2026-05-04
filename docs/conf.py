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
import shutil
from pathlib import Path

from tesseract_core import __version__

# Set the TESSERACT_API_PATH environment variable to the dummy Tesseract API
# This will be used to instantiate the Tesseract runtime so we can generate api docs
os.environ["TESSERACT_API_PATH"] = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tests", "dummy_tesseract", "tesseract_api.py"
    )
)

project = "Tesseract"
copyright = "2026, Pasteur Labs"
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
    "myst_nb",
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
    # Sitemap for SEO
    "sphinx_sitemap",
]


myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "jupyter_execute", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The docs root is content/introduction/index.md, which owns all toctrees
# and drives the sidebar. The landing page (landing.md) is an orphan page
# that gets copied to index.html after build so it serves at the site root.
root_doc = "content/introduction/index"

html_theme = "furo"
html_static_path = ["static"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
}
html_css_files = ["custom.css"]
html_js_files = [
    ("https://buttons.github.io/buttons.js", {"async": "async"}),
    "external-links.js",
]
html_baseurl = "https://docs.pasteurlabs.ai/projects/tesseract-core/latest/"
sitemap_url_scheme = (
    "{link}"  # ReadTheDocs handles versioning; don't add language/version prefix
)


# -- OpenGraph metadata (social cards) ---------------------------------------

ogp_site_url = "https://docs.pasteurlabs.ai/projects/tesseract-core/latest/"
ogp_site_name = "Tesseract"
ogp_description_length = 200
ogp_type = "article"
ogp_social_cards = {
    "image": "static/logo-dark.png",
    "line_color": "#d946ef",
}


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


def _copy_landing_to_index(app, exception) -> None:
    """Copy landing.html to index.html so the landing page serves at /."""
    if exception or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    landing = outdir / "landing.html"
    index = outdir / "index.html"
    if landing.exists():
        shutil.copy2(landing, index)


def setup(app) -> None:
    """Sphinx setup function. Used to register custom stuff."""
    # HACK: We zip the examples folder here so that it can be downloaded
    zip_examples_folder()
    # Copy landing page to index.html after build
    app.connect("build-finished", _copy_landing_to_index)


# -- Handle Jupyter notebooks ------------------------------------------------

# Do not execute notebooks during build (just take existing output)
nb_execution_mode = "off"

# Copy example notebooks and their companion files to the docs folder on every build
_COMPANION_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".svg"}
for example_notebook in Path("../demo").glob("*/demo.ipynb"):
    # Copy the example notebook to the docs folder
    dest = (Path("content/demo") / example_notebook.parent.name).with_suffix(".ipynb")
    shutil.copyfile(example_notebook, dest)
    # Copy companion images so relative references in the notebook resolve
    for companion in example_notebook.parent.iterdir():
        if companion.suffix.lower() in _COMPANION_EXTS:
            shutil.copyfile(companion, Path("content/demo") / companion.name)
