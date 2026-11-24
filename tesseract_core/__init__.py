# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from ._version import __version__ as scm_version
except ImportError:
    import warnings

    warnings.warn(
        "Unable to import version information from _version.py. "
        "This is likely due to the package not being installed. "
        "Using default version '0.0.0+unknown'.",
        ImportWarning,
        stacklevel=1,
    )
    scm_version = "0.0.0+unknown"

__version__ = scm_version

# Oldest SDK and runtime release this one is tested against, in both directions
# (tests/compat_tests). Raised by hand, only for a change that cannot keep it working.
__oldest_supported_version__ = "1.13.0"

# import public API of the package
# from . import <obj>
from .sdk.engine import build_tesseract, run_tesseract, serve, teardown
from .sdk.tesseract import Tesseract

# add public API as strings here, for example __all__ = ["obj"]
__all__ = ["Tesseract", "build_tesseract", "run_tesseract", "serve", "teardown"]
