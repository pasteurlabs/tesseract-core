#!/usr/bin/env python3
# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the forum topics that serve as comment threads for blog posts and releases.

Each topic is tagged with a Discourse external ID derived from the post's
filename or the release version. This keeps the script idempotent, since
existing topics are left alone, and lets the docs build look up blog topics by
the same ID (see conf.py).

Usage:
    forum_topics.py blog
    forum_topics.py release VERSION NOTES_FILE

Environment:
    DISCOURSE_API_KEY       API key allowed to create topics (required)
    DISCOURSE_CATEGORY_ID   Category that new topics are posted to (required)
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from blog_posts import collect_blog_posts

FORUM_URL = "https://si-tesseract.discourse.group"
BLOG_URL = "https://docs.pasteurlabs.ai/projects/tesseract-core/latest/blog"
RELEASE_URL = "https://github.com/pasteurlabs/tesseract-core/releases/tag"


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


def find_topic(external_id: str) -> str | None:
    """Return the URL of the topic with this external ID, if it exists."""
    try:
        topic = _request("GET", f"/t/external_id/{external_id}.json")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return f"{FORUM_URL}/t/{topic['slug']}/{topic['id']}"


def ensure_topic(external_id: str, title: str, body: str) -> str:
    """Return the URL of the topic with this external ID, creating it if needed."""
    # Discourse only routes lookups for external IDs of this form
    if not re.fullmatch(r"[\w-]+", external_id):
        raise ValueError(f"Invalid Discourse external ID: {external_id!r}")

    if url := find_topic(external_id):
        print(f"exists:  {external_id} -> {url}", file=sys.stderr)
        return url

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
    print(f"created: {external_id} -> {url}", file=sys.stderr)
    return url


def ensure_blog_topics() -> None:
    """Make sure every blog post has a topic, unless it names one via `forum_topic`."""
    for post in collect_blog_posts():
        if post["forum_topic"]:
            continue
        # A bare URL on its own line renders as a preview card on Discourse
        body = f"{post['description']}\n\n{BLOG_URL}/{post['file']}/\n"
        ensure_topic(blog_external_id(post["file"]), post["title"], body)


def ensure_release_topic(version: str, notes_file: Path) -> None:
    """Make sure a release has a topic and print its URL."""
    notes = notes_file.read_text().strip()
    if not notes:
        notes = f"Tesseract Core {version} is out."
    body = f"{notes}\n\n{RELEASE_URL}/{version}\n"
    external_id = "tesseract-core-release-" + version.replace(".", "-")
    print(ensure_topic(external_id, f"Tesseract Core {version} released", body))


def main() -> None:
    """Parse arguments and dispatch to the blog or release command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("blog", help="ensure every blog post has a topic")

    release = subparsers.add_parser("release", help="ensure a release has a topic")
    release.add_argument("version", help="release tag, e.g. v1.14.0")
    release.add_argument(
        "notes_file", type=Path, help="hand-written release notes for the topic"
    )

    args = parser.parse_args()
    if args.command == "blog":
        ensure_blog_topics()
    else:
        ensure_release_topic(args.version, args.notes_file)


if __name__ == "__main__":
    main()
