# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .finite_differences import (
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
)
from .gradient_endpoint_derivation import (
    jacobian_from_jvp,
    jacobian_from_vjp,
    jvp_from_jacobian,
    vjp_from_jacobian,
)
from .lazy_sequence import (
    LazySequence,
)
from .mpa import (
    log_artifact,
    log_metric,
    log_parameter,
)
from .paths import (
    InputFileReference,
    InputPath,
    OutputFileReference,
    OutputPath,
    require_file,
)
from .tesseract_reference import TesseractReference
from .vjp_cache import set_jax_vjp_cache_size

# Flag is modified by runtime.cli based on arguments or during build time.
# It lives here (rather than in .paths) so that existing code that mutates
# ``tesseract_core.runtime.experimental.SKIP_REQUIRED_FILE_CHECK`` keeps working;
# ``require_file`` reads it back through this module.
SKIP_REQUIRED_FILE_CHECK = False

# Re-point the canonical module of the re-exported objects back to this package.
# This keeps the public, documented path stable (e.g. Sphinx autodoc cross-refs
# such as ``tesseract_core.runtime.experimental.LazySequence``) after the split
# from a single module into submodules.
for _obj in (
    LazySequence,
    InputPath,
    OutputPath,
    TesseractReference,
    require_file,
    set_jax_vjp_cache_size,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
    jacobian_from_jvp,
    jacobian_from_vjp,
    jvp_from_jacobian,
    vjp_from_jacobian,
    log_artifact,
    log_metric,
    log_parameter,
):
    _obj.__module__ = __package__
del _obj

__all__ = [
    "InputFileReference",
    "InputPath",
    "LazySequence",
    "OutputFileReference",
    "OutputPath",
    "TesseractReference",
    "finite_difference_jacobian",
    "finite_difference_jvp",
    "finite_difference_vjp",
    "jacobian_from_jvp",
    "jacobian_from_vjp",
    "jvp_from_jacobian",
    "log_artifact",
    "log_metric",
    "log_parameter",
    "require_file",
    "set_jax_vjp_cache_size",
    "vjp_from_jacobian",
]
