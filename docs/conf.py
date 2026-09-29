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
import sys
from pathlib import Path

from tesseract_core import __version__

# Make the blog and forum helpers next to this file importable
sys.path.insert(0, os.path.dirname(__file__))

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
    "sphinx.ext.extlinks",
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
    # Redirect stubs for pages moved during the Diátaxis reorganization
    "sphinx_reredirects",
]

# The docs are served with the `dirhtml` builder (clean, extension-less URLs)
# and Read the Docs has an "HTML to clean URL" redirect configured, so old
# `.html` URLs are rewritten to their trailing-slash form at the hosting layer.
# The only thing left to preserve is pages that changed *location* during the
# Diátaxis reorganization: sphinx-reredirects writes a stub at each old page's
# (clean) URL that forwards to its new one.
#
# Targets are RELATIVE, not absolute. Under dirhtml each stub is written at
# `<old-docname>/index.html` — one directory deeper than the docname — but
# sphinx-reredirects relativizes leading-slash targets against the docname's
# depth, so an absolute target lands one `../` short. Version-pinned absolute
# URLs would also break RTD's per-version/PR-preview subpaths. Every old page
# lives at `content/<section>/<page>` (docname depth 3 → stub depth 4), so the
# route back to the output root is uniformly `../../../`.
_R = "../../../content"
redirects = {
    "content/introduction/get-started": f"{_R}/tutorials/get-started/",
    "content/creating-tesseracts/create": f"{_R}/tutorials/create/",
    "content/creating-tesseracts/design-patterns": f"{_R}/concepts/design-patterns/",
    "content/how-to/design-patterns": f"{_R}/concepts/design-patterns/",
    "content/creating-tesseracts/pipelines": f"{_R}/how-to/pipelines/",
    "content/creating-tesseracts/deploy": f"{_R}/how-to/deploy/",
    "content/creating-tesseracts/llm-assistance": f"{_R}/how-to/llm-assistance/",
    "content/creating-tesseracts/advanced": f"{_R}/how-to/defining-apis/",
    "content/using-tesseracts/use": f"{_R}/tutorials/interact/",
    "content/how-to/use": f"{_R}/tutorials/interact/",
    "content/using-tesseracts/advanced": f"{_R}/how-to/advanced-usage/",
    "content/using-tesseracts/array-encodings": f"{_R}/reference/array-encodings/",
    "content/misc/debugging": f"{_R}/how-to/debugging/",
    "content/misc/differentiable-programming": f"{_R}/concepts/differentiable-programming/",
    "content/misc/performance": f"{_R}/concepts/performance/",
    "content/api/config": f"{_R}/reference/config/",
    "content/api/endpoints": f"{_R}/reference/endpoints/",
    "content/api/tesseract-api": f"{_R}/reference/tesseract-api/",
    "content/api/tesseract-cli": f"{_R}/reference/tesseract-cli/",
    "content/api/tesseract-runtime-api": f"{_R}/reference/tesseract-runtime-api/",
    "content/api/tesseract-runtime-cli": f"{_R}/reference/tesseract-runtime-cli/",
}


# -- "View on GitHub" links --------------------------------------------------
# Link example source to the *ref the docs are being built from*, so a link to a
# brand-new example resolves on a PR/branch preview instead of 404ing against
# `main` before the branch is merged. Resolution order covers the three build
# contexts: Read the Docs (versioned + PR previews), the GitHub Actions docs job
# (which runs linkcheck on PRs), and local builds.
_repo_url = "https://github.com/pasteurlabs/tesseract-core"
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
    # RTD PR previews: READTHEDOCS_GIT_IDENTIFIER is the PR *number*, which is not
    # a valid GitHub ref. Use the PR head commit hash, which resolves under /tree/.
    _git_ref = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
else:
    _git_ref = (
        os.environ.get("READTHEDOCS_GIT_IDENTIFIER")  # RTD: tag or branch
        # GH Actions PR builds: the PR *head commit*, not GITHUB_HEAD_REF. A
        # fork PR's head branch lives in the fork, so `tree/<branch>` 404s
        # against this repo; the head commit is reachable here via refs/pull/N
        # and, unlike the branch, is guaranteed to contain any example the PR
        # adds. Set in build_docs.yml.
        or os.environ.get("DOCS_GIT_REF")
        or os.environ.get("GITHUB_REF_NAME")  # GH Actions: branch/tag on push
        or "main"  # local build
    )
extlinks = {
    # Usage in Markdown, with an explicit title:
    #   {gh-tree}`View on GitHub <examples/helloworld>`
    #   {gh-blob}`Dockerfile <tesseract_core/sdk/templates/Dockerfile.base>`
    # The path fills %s in the URL; the title is shown verbatim. Use `gh-tree`
    # for directories and `gh-blob` for single files.
    "gh-tree": (f"{_repo_url}/tree/{_git_ref}/%s", None),
    "gh-blob": (f"{_repo_url}/blob/{_git_ref}/%s", None),
}

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


def _emit_config_schema(app) -> None:
    """Write the tesseract_config.yaml JSON Schema into the build output root.

    Served at ``<site>/tesseract_config.schema.json``, this is the URL scaffolded into new
    configs and registered with SchemaStore, so IDEs can validate
    ``tesseract_config.yaml``. Generating it here (rather than committing it)
    keeps it in lockstep with the TesseractConfig model on every build.
    """
    import json

    from tesseract_core.sdk.api_parse import generate_config_schema

    out_dir = Path(app.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tesseract_config.schema.json").write_text(
        json.dumps(generate_config_schema(), indent=2)
    )


# Every blog post links to a forum topic as its comment thread. The workflow in
# .github/workflows/forum_topics.yml creates the topic when a post lands on main,
# unless the post names an existing one with `forum_topic: <topic ID>` in its
# frontmatter.
#
# The docs build and the workflow start together on merge, so the build waits
# this long for new topics before failing.
FORUM_TOPIC_TIMEOUT_SECONDS = 90
FORUM_TOPIC_POLL_SECONDS = 5

_forum_topic_urls: dict[str, str] = {}


def _resolve_forum_topics(_app) -> None:
    """Look up the forum topic of every blog post, failing the build if one is missing.

    Only runs for the production build (Read the Docs "latest"), so local builds
    and PR previews neither depend on the forum nor render the links. Set
    FORUM_TOPICS_OPTIONAL=1 to downgrade missing topics to a warning.
    """
    import logging
    import time

    from blog_posts import collect_blog_posts
    from forum_topics import FORUM_URL, blog_external_id, find_topic

    if os.environ.get("READTHEDOCS_VERSION") != "latest":
        return

    logger = logging.getLogger("sphinx.ext.blog")

    pending = {}
    for post in collect_blog_posts():
        if post["forum_topic"]:
            _forum_topic_urls[post["file"]] = f"{FORUM_URL}/t/{post['forum_topic']}"
        else:
            pending[post["file"]] = blog_external_id(post["file"])

    errors = {}
    deadline = time.monotonic() + FORUM_TOPIC_TIMEOUT_SECONDS
    while True:
        for post_file, external_id in list(pending.items()):
            try:
                url = find_topic(external_id)
            except (OSError, ValueError) as e:
                errors[post_file] = e
                continue
            if url:
                _forum_topic_urls[post_file] = url
                del pending[post_file]
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(FORUM_TOPIC_POLL_SECONDS)

    if pending:
        details = "\n".join(
            f"  {post_file}: "
            + (str(errors[post_file]) if post_file in errors else "not found")
            for post_file in pending
        )
        message = (
            f"No forum topic found after {FORUM_TOPIC_TIMEOUT_SECONDS}s for these blog "
            f"posts:\n{details}\nCheck the 'Create forum topics for blog posts' "
            "workflow run for this commit, or set FORUM_TOPICS_OPTIONAL=1 in the "
            "Read the Docs environment to build without the links."
        )
        if os.environ.get("FORUM_TOPICS_OPTIONAL"):
            logger.warning(message)
        else:
            raise RuntimeError(message)


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
        from blog_posts import collect_blog_posts

        posts = collect_blog_posts()
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

    context["forum_topic_url"] = _forum_topic_urls.get(pagename.removeprefix("blog/"))
    return "blog_post.html"


def _require_dirhtml(app) -> None:
    """Fail fast if the plain ``html`` builder is used.

    The site is served with ``dirhtml`` (clean, extension-less URLs) and the
    reredirect stubs are written assuming that layout. Building with ``html``
    produces subtly broken redirects and mismatched links, so steer devs to
    ``dirhtml`` instead of letting them ship a broken build.
    """
    if app.builder.name == "html":
        raise RuntimeError(
            "These docs are built with the 'dirhtml' builder, not 'html'. "
            "Use `make dirhtml` (or `sphinx-build -b dirhtml`) instead."
        )


def setup(app) -> None:
    """Sphinx setup function. Used to register custom stuff."""
    # Enforce the dirhtml builder (see _require_dirhtml for why)
    app.connect("builder-inited", _require_dirhtml)
    # We zip the examples folder here so that it can be downloaded
    app.connect("builder-inited", zip_examples_folder)
    # Emit the tesseract_config.yaml JSON Schema into the output root
    app.connect("build-finished", lambda app, exc: exc or _emit_config_schema(app))
    # Look up the forum topic of each blog post
    app.connect("builder-inited", _resolve_forum_topics)
    # Inject blog post listing into blog index page context
    app.connect("html-page-context", _inject_page_context)


# -- Options for the linkcheck builder ---------------------------------------
# `make linkcheck` (run in CI) validates every external URL and, crucially,
# every raw-HTML asset path — the one class of broken link that `-W` cannot
# catch, since Sphinx emits raw HTML verbatim without resolving it.

# URLs that linkcheck cannot validate but that are fine in a browser. Keep this
# list tight and annotated so it stays a set of known false positives, not a
# dumping ground for genuinely broken links.
linkcheck_ignore = [
    # Anti-bot / login walls return 403/redirects to headless requests.
    r"https://www\.mathworks\.com/",
    r"https://www\.linkedin\.com/",
    # Annual Reviews blocks headless requests (403); the DOI resolves in a browser.
    r"https://doi\.org/10\.1146/annurev-fluid-010518-040547",
    # MIT's shared script hosting (scripts-vhosts.mit.edu) intermittently refuses
    # HTTPS connections, causing flaky connect timeouts in CI even though the site
    # is fine in a browser. Retries don't help — it can be down for minutes.
    r"https://enzyme\.mit\.edu/.*",
    # The pasteurlabs.ai marketing site is fronted by Netlify, which throttles
    # bursts of requests from datacenter IPs like GitHub Actions runners, causing
    # read timeouts that outlast linkcheck_retries even though the links are fine
    # in a browser. The docs.pasteurlabs.ai subdomain is hosted elsewhere (Read
    # the Docs) and is left checked.
    r"https://pasteurlabs\.ai(/.*)?$",
]

# Pages whose in-page anchors are generated client-side (or are browser text
# fragments), so linkcheck's static anchor check yields false negatives.
linkcheck_anchors_ignore_for_url = [
    r"https://www\.ecmwf\.int/.*",
]

# Some CDNs (e.g. Netlify, which fronts pasteurlabs.ai) throttle bursts of
# concurrent requests from datacenter IPs like GitHub Actions runners, stalling
# the surplus connections until they time out — even though each link is fine in
# a browser. This surfaced as flaky `read timeout=30` failures in CI. Keeping the
# worker pool small shrinks those bursts, and a generous timeout plus retries
# lets a throttled connection recover. Timeouts are retried up to
# linkcheck_retries before being reported broken; genuine 4xx failures still fail
# fast, so real broken links are not masked.
linkcheck_workers = 2
linkcheck_timeout = 60
linkcheck_retries = 2


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
