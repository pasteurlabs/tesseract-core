#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror blog posts and release notes to the forum, where they get comment threads.

Each topic is tagged with a Discourse external ID derived from the post's
filename or the release version. This lets the script update existing topics
instead of duplicating them, and lets the docs build look up blog topics by the
same ID (see conf.py).

Usage:
    forum_topics.py blog MARKDOWN_DIR
    forum_topics.py release VERSION NOTES_FILE

MARKDOWN_DIR is the output of `sphinx-build -b markdown docs MARKDOWN_DIR`.

Environment:
    DISCOURSE_API_KEY       API key allowed to create topics and edit posts (required)
    DISCOURSE_CATEGORY_ID   Category that new topics are posted to (required)
"""

import argparse
import hashlib
import json
import os
import posixpath
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from blog_posts import collect_blog_posts

FORUM_URL = "https://si-tesseract.discourse.group"
DOCS_URL = "https://docs.pasteurlabs.ai/projects/tesseract-core/latest"
REPO_URL = "https://github.com/pasteurlabs/tesseract-core"
# Images are linked from GitHub, where they exist before the docs site is rebuilt.
# Discourse copies them when a post is created or edited, so they only need to
# exist on `main` at that point.
IMAGE_URL = "https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/docs"

# Discourse's default `max_post_length`, minus room for the sync marker. Longer
# posts are cut short with a link to the full text.
MAX_POST_LENGTH = 32_000 - 100

# Appended to every mirrored post to detect changes to the source. Comparing the
# posts themselves does not work, because Discourse rewrites image links in the
# post once it has copied the images.
SYNC_MARKER = "<!-- mirrored from source with digest {} -->"
SYNC_MARKER_REGEX = re.compile(r"<!-- mirrored from source with digest (\w+) -->")

# Fenced code blocks and inline code, which rewrites must leave alone
CODE_REGEX = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if api_key := os.environ.get("DISCOURSE_API_KEY"):
        headers["Api-Key"] = api_key
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        FORUM_URL + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        # Discourse explains validation failures (e.g. duplicate titles) in the body
        e.msg = f"{e.msg}: {e.read().decode(errors='replace')}"
        raise


def blog_external_id(post_file: str) -> str:
    """Return the external ID of the topic for the blog post with this filename stem."""
    return f"blog-{post_file}"


def _get_topic(external_id: str) -> dict | None:
    try:
        # include_raw adds the Markdown source of each post, which holds the sync marker
        return _request("GET", f"/t/external_id/{external_id}.json?include_raw=true")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def find_topic(external_id: str) -> str | None:
    """Return the URL of the topic with this external ID, if it exists."""
    topic = _get_topic(external_id)
    return f"{FORUM_URL}/t/{topic['slug']}/{topic['id']}" if topic else None


def sync_topic(external_id: str, title: str, body: str) -> str:
    """Create or update the topic with this external ID, and return its URL."""
    # Discourse only routes lookups for external IDs of this form
    if not re.fullmatch(r"[\w-]+", external_id):
        raise ValueError(f"Invalid Discourse external ID: {external_id!r}")

    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    body = f"{body.rstrip()}\n\n{SYNC_MARKER.format(digest)}\n"

    topic = _get_topic(external_id)
    if topic is None:
        post = _request(
            "POST",
            "/posts.json",
            {
                "title": title,
                "raw": body,
                "category": int(os.environ["DISCOURSE_CATEGORY_ID"]),
                "external_id": external_id,
            },
        )
        url = f"{FORUM_URL}/t/{post['topic_slug']}/{post['topic_id']}"
        print(f"created:   {external_id} -> {url}", file=sys.stderr)
        return url

    url = f"{FORUM_URL}/t/{topic['slug']}/{topic['id']}"
    first_post = topic["post_stream"]["posts"][0]
    current = SYNC_MARKER_REGEX.search(first_post["raw"])
    if current and current[1] == digest:
        print(f"unchanged: {external_id} -> {url}", file=sys.stderr)
    else:
        _request(
            "PUT",
            f"/posts/{first_post['id']}.json",
            {"post": {"raw": body, "edit_reason": "Synced from the original"}},
        )
        print(f"updated:   {external_id} -> {url}", file=sys.stderr)
    return url


def _outside_code(text: str, rewrite: Callable[[str], str]) -> str:
    """Apply `rewrite` to everything in a Markdown text except code."""
    parts = CODE_REGEX.split(text)
    # re.split puts the captured code spans at the odd indices
    return "".join(part if i % 2 else rewrite(part) for i, part in enumerate(parts))


def _truncate(body: str, full_text_url: str, where: str) -> str:
    """Cut a post that exceeds the forum's length limit at a paragraph break."""
    if len(body) <= MAX_POST_LENGTH:
        return body
    notice = f"\n\n*This post continues [on {where}]({full_text_url}).*\n"
    # Leave room for the notice and for closing a code block that the cut splits
    limit = MAX_POST_LENGTH - len(notice) - len("\n```")
    cut = body.rfind("\n\n", 0, limit)
    if cut <= 0:
        cut = body.rfind("\n", 0, limit)
    if cut <= 0:
        cut = limit
    body = body[:cut]
    if body.count("```") % 2:
        body += "\n```"
    return body + notice


def render_blog_post(markdown: str, post_file: str) -> str:
    """Turn the Markdown build of a blog post into a forum post."""
    post_url = f"{DOCS_URL}/blog/{post_file}/"

    def absolutize(text: str) -> str:
        # Image paths are relative to the docs source root
        text = re.sub(
            r"(!\[[^\]]*\]\()(?!https?://)([^)\s]+)\)",
            lambda m: f"{m[1]}{IMAGE_URL}/{m[2]})",
            text,
        )

        # Links to other pages are relative to the post and point at Markdown sources
        def page_url(m: re.Match) -> str:
            page = posixpath.normpath(posixpath.join("blog", m[2]))
            # dirhtml serves `foo/index.md` at `foo/`
            page = "" if page == "index" else page.removesuffix("/index")
            url = f"{DOCS_URL}/{page}/" if page else f"{DOCS_URL}/"
            return f"{m[1]}{url}{m[3] or ''})"

        return re.sub(
            r"(?<!!)(\[[^\]]*\]\()(?!https?://|#|mailto:)([^)\s]+?)\.md(#[^)\s]*)?\)",
            page_url,
            text,
        )

    # The topic title replaces the post's heading
    markdown = re.sub(r"\A# .*\n+", "", markdown)
    body = (
        f"*Originally published on the [Tesseract Blog]({post_url}).*\n\n"
        + _outside_code(markdown, absolutize).strip()
        + "\n"
    )
    return _truncate(body, post_url, "the Tesseract Blog")


def render_release_notes(notes: str, version: str) -> str:
    """Turn GitHub release notes into a forum post."""
    release_url = f"{REPO_URL}/releases/tag/{version}"

    def link_github_refs(text: str) -> str:
        # Discourse does not link #123 references automatically
        text = re.sub(r"(?<![\w/\[&])#(\d+)\b", rf"[#\1]({REPO_URL}/pull/\1)", text)
        # On the forum, @name would notify whichever forum user has that name
        return re.sub(
            r"(?<![\w/\[`])@([A-Za-z0-9][A-Za-z0-9-]*)",
            r"[@\1](https://github.com/\1)",
            text,
        )

    # The topic title replaces the "# Release ..." heading
    notes = re.sub(r"\A# .*\n+", "", notes.strip())
    body = (
        f"*Originally published on [GitHub]({release_url}).*\n\n"
        + _outside_code(notes, link_github_refs)
        + "\n"
    )
    return _truncate(body, release_url, "GitHub")


def sync_blog_topics(markdown_dir: Path) -> None:
    """Mirror every blog post, unless it names an existing topic via `forum_topic`."""
    for post in collect_blog_posts():
        if post["forum_topic"]:
            continue
        markdown = (markdown_dir / "blog" / f"{post['file']}.md").read_text()
        body = render_blog_post(markdown, post["file"])
        sync_topic(blog_external_id(post["file"]), post["title"], body)


def sync_release_topic(version: str, notes_file: Path) -> None:
    """Mirror the notes of a release and print the topic URL."""
    body = render_release_notes(notes_file.read_text(), version)
    external_id = "tesseract-core-release-" + version.replace(".", "-")
    print(sync_topic(external_id, f"Tesseract Core {version} released", body))


def main() -> None:
    """Parse arguments and dispatch to the blog or release command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    blog = subparsers.add_parser("blog", help="mirror every blog post")
    blog.add_argument(
        "markdown_dir", type=Path, help="output of the Sphinx markdown builder"
    )

    release = subparsers.add_parser("release", help="mirror the notes of a release")
    release.add_argument("version", help="release tag, e.g. v1.14.0")
    release.add_argument("notes_file", type=Path, help="release notes in Markdown")

    args = parser.parse_args()
    if args.command == "blog":
        sync_blog_topics(args.markdown_dir)
    else:
        sync_release_topic(args.version, args.notes_file)


if __name__ == "__main__":
    main()
