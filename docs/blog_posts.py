# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blog post discovery, shared by the docs build and forum_topics.py."""

import functools
import logging
from datetime import datetime, timezone
from pathlib import Path

BLOG_DIR = Path(__file__).parent / "blog"

logger = logging.getLogger("sphinx.ext.blog")


@functools.cache
def collect_blog_posts() -> list[dict]:
    """Collect metadata from all blog posts, newest first."""
    import yaml

    posts = []
    for md_file in sorted(BLOG_DIR.glob("*.md")):
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
        date = datetime.strptime(str(blog_date), "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        posts.append(
            {
                "file": md_file.stem,
                "title": title,
                "date": date.strftime("%b %d, %Y").replace(" 0", " "),
                "author": fm.get("blog_author", ""),
                "description": fm.get("blog_description", ""),
                "forum_topic": fm.get("forum_topic"),
                "_sort_key": (date, md_file.name),
            }
        )

    posts.sort(key=lambda p: p["_sort_key"], reverse=True)
    return posts
