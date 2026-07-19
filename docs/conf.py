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

project = "Tesseract Core"
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
    "sphinxcontrib.typer",
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

# jax_recipes imports optional JAX dependencies (jax, equinox) at module level
# that aren't installed in the docs environment; mock them so autodoc can
# introspect the module without importing the real packages.
autodoc_mock_imports = ["jax", "equinox"]

templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "jupyter_execute", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The docs root is content/introduction/index.md, which owns all toctrees
# and drives the sidebar. The landing page (index.md) is an orphan page
# that exists outside the main docs structure and is not included in the sidebar or toctrees.
root_doc = "content/introduction/index"

html_title = f"Tesseract Core {version}"
html_theme = "furo"
html_static_path = ["static"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
}
html_css_files = ["top-nav.css", "custom.css"]
html_js_files = []
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


def zip_examples_folder(_app) -> None:
    """Zip a folder and save it to the specified path."""
    import shutil
    from pathlib import Path

    here = Path(__file__).parent

    root_dir = (here / "..").resolve()
    archive_path = here / "downloads" / "examples.zip"

    shutil.make_archive(archive_path.with_suffix(""), "zip", root_dir, "examples")
    assert archive_path.exists()


def _collect_blog_posts() -> list[dict]:
    """Collect metadata from all blog posts for the blog index."""
    import logging
    from datetime import datetime

    import yaml

    logger = logging.getLogger("sphinx.ext.blog")

    here = Path(__file__).parent
    blog_dir = here / "blog"

    posts = []
    for md_file in sorted(blog_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text()
        if not text.startswith("---"):
            logger.warning(
                "blog post %s has no YAML frontmatter, skipping", md_file.name
            )
            continue
        end = text.index("---", 3)
        fm = yaml.safe_load(text[3:end])
        blog_date = fm.get("blog_date")
        if not blog_date:
            logger.warning(
                "blog post %s missing 'blog_date' in frontmatter, skipping",
                md_file.name,
            )
            continue
        title = fm.get("blog_title")
        if not title:
            logger.warning(
                "blog post %s missing 'blog_title' in frontmatter, skipping",
                md_file.name,
            )
            continue
        date = datetime.strptime(str(blog_date), "%Y-%m-%d")
        posts.append(
            {
                "file": md_file.stem,
                "title": title,
                "date": date.strftime("%b %d, %Y").replace(" 0", " "),
                "author": fm.get("blog_author", ""),
                "description": fm.get("blog_description", ""),
                "_sort_key": (date, md_file.name),
            }
        )

    posts.sort(key=lambda p: p["_sort_key"], reverse=True)
    return posts


def _inject_page_context(app, pagename, templatename, context, doctree):
    """Select special templates and inject context for blog/landing pages."""
    is_blog = pagename.startswith("blog/")
    is_landing = pagename == "index"

    # Override favicon and site title for blog + landing pages
    if is_blog or is_landing:
        pathto = context["pathto"]
        context["favicon_url"] = pathto("_static/favicon.ico", resource=True)

    if is_blog:
        context["docstitle"] = "Tesseract Blog"

    # Landing page: suppress docstitle so Furo renders just "Tesseract" as <title>
    if is_landing:
        context["docstitle"] = ""

    if not is_blog:
        return

    if pagename == "blog/index":
        posts = _collect_blog_posts()
        context["blog_posts"] = [
            {
                "url": app.builder.get_relative_uri(pagename, "blog/" + p["file"]),
                "title": p["title"],
                "date": p["date"],
                "author": p["author"],
                "description": p["description"],
            }
            for p in posts
        ]
        return "blog_index.html"

    return "blog_post.html"


def setup(app) -> None:
    """Sphinx setup function. Used to register custom stuff."""
    # We zip the examples folder here so that it can be downloaded
    app.connect("builder-inited", zip_examples_folder)
    # Inject blog post listing into blog index page context
    app.connect("html-page-context", _inject_page_context)


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
