# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling utilities for Tesseract runtime.

Uses Python's built-in cProfile module to avoid external dependencies.
Profiling is controlled by the TESSERACT_PROFILING config flag.
"""

import cProfile
import pstats
import re
from io import StringIO
from typing import Any

from .config import get_config

# Patterns to exclude from profiling output (framework internals)
EXCLUDED_PATTERNS = [
    r"/starlette/",
    r"/uvicorn/",
    r"/fastapi/",
    r"/anyio/",
    r"/httptools/",
    r"/h11/",
]


class Profiler:
    """A simple profiler wrapper around cProfile.

    Only profiles when the profiling config flag is enabled.
    Can be used as a context manager.

    Example:
        with Profiler() as profiler:
            # code to profile
            pass
        if profiler.enabled:
            print(profiler.get_stats())
    """

    def __init__(self, enabled: bool | None = None) -> None:
        """Initialize the profiler.

        Args:
            enabled: Whether profiling is enabled. If None, uses the
                TESSERACT_PROFILING config flag.
        """
        self._profiler: cProfile.Profile | None = None
        self._enabled = get_config().profiling if enabled is None else enabled

    @property
    def enabled(self) -> bool:
        """Whether profiling is enabled."""
        return self._enabled

    def start(self) -> None:
        """Start profiling if enabled."""
        if not self._enabled:
            return
        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self) -> None:
        """Stop profiling if it was started."""
        if self._profiler is not None:
            self._profiler.disable()

    def __enter__(self) -> "Profiler":
        """Start profiling and return self."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop profiling."""
        self.stop()

    def _filter_stats(self, stats: pstats.Stats) -> pstats.Stats:
        """Filter out framework internals from profiling stats.

        Removes entries matching EXCLUDED_PATTERNS (starlette, uvicorn, fastapi, etc.)
        to focus on user code.
        """
        # Build combined regex pattern
        combined_pattern = "|".join(EXCLUDED_PATTERNS)
        exclude_regex = re.compile(combined_pattern)

        # Filter the stats dictionary
        # Keys are tuples of (filename, lineno, funcname)
        filtered_stats = {
            key: value
            for key, value in stats.stats.items()
            if not exclude_regex.search(key[0])
        }

        # Replace stats dict with filtered version
        stats.stats = filtered_stats

        # Recalculate total calls and time
        stats.total_calls = sum(v[0] for v in filtered_stats.values())
        stats.prim_calls = sum(v[0] for v in filtered_stats.values())
        stats.total_tt = sum(v[2] for v in filtered_stats.values())

        return stats

    def get_stats(self, limit: int = 30) -> str:
        """Get profiling statistics as a formatted string.

        Returns two reports:
        1. Sorted by cumulative time (time including sub-calls)
        2. Sorted by total time (time in function itself)

        Framework internals (starlette, uvicorn, fastapi, etc.) are excluded
        to focus on user code.

        Args:
            limit: Maximum number of entries to include per report.

        Returns:
            Formatted profiling statistics string, or empty string if profiling
            was not enabled.
        """
        if self._profiler is None:
            return ""

        output_parts = []

        # Report 1: Cumulative time (includes time in sub-calls)
        stream1 = StringIO()
        stats1 = pstats.Stats(self._profiler, stream=stream1)
        stats1 = self._filter_stats(stats1)
        stats1.sort_stats("cumulative")
        stats1.print_stats(limit)
        output_parts.append("=== By Cumulative Time (includes sub-calls) ===")
        output_parts.append(stream1.getvalue())

        # Report 2: Total time (time in function itself, excluding sub-calls)
        stream2 = StringIO()
        stats2 = pstats.Stats(self._profiler, stream=stream2)
        stats2 = self._filter_stats(stats2)
        stats2.sort_stats("tottime")
        stats2.print_stats(limit)
        output_parts.append("=== By Total Time (excluding sub-calls) ===")
        output_parts.append(stream2.getvalue())

        return "\n".join(output_parts)
