# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    Strict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError


class _ApiObject(NamedTuple):
    name: str
    expected_type: type
    num_args: int | None = None
    arg_names: tuple[str, ...] | None = None
    optional: bool = False


ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

EXPECTED_OBJECTS = (
    _ApiObject("apply", ast.FunctionDef, 1, arg_names=("inputs",)),
    _ApiObject("InputSchema", ast.ClassDef),
    _ApiObject("OutputSchema", ast.ClassDef),
    _ApiObject(
        "jacobian",
        ast.FunctionDef,
        3,
        arg_names=("inputs", "jac_inputs", "jac_outputs"),
        optional=True,
    ),
    _ApiObject(
        "jacobian_vector_product",
        ast.FunctionDef,
        4,
        arg_names=("inputs", "jvp_inputs", "jvp_outputs", "tangent_vector"),
        optional=True,
    ),
    _ApiObject(
        "vector_jacobian_product",
        ast.FunctionDef,
        4,
        arg_names=("inputs", "vjp_inputs", "vjp_outputs", "cotangent_vector"),
        optional=True,
    ),
    _ApiObject(
        "abstract_eval",
        ast.FunctionDef,
        1,
        arg_names=("abstract_inputs",),
        optional=True,
    ),
)


def assert_relative_path(value: str) -> str:
    """Assert that a string encodes a relative path."""
    from pathlib import PurePosixPath, PureWindowsPath

    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"value must be a relative path (got {value})")
    return value


RelativePath = Annotated[str, AfterValidator(assert_relative_path)]
StrictStr = Annotated[str, Strict()]


def _normalize_provider(value: Any) -> Any:
    """Accept the deprecated ``python-pip`` provider name as an alias for ``uv-pip``."""
    if value == "python-pip":
        # Emit through the logger rather than warnings.warn: a library-emitted
        # DeprecationWarning is suppressed under Python's default filters, so a
        # real CLI user would never see it.
        # Scheduled for removal in 1.13.0; see tesseract_core/_deprecations.py.
        logging.getLogger("tesseract").warning(
            "The 'python-pip' requirements provider has been renamed to 'uv-pip' "
            "(the build has always used uv under the hood). Set `provider: uv-pip` "
            "in tesseract_config.yaml; 'python-pip' still works but will be "
            "removed in Tesseract 1.13.0."
        )
        return "uv-pip"
    return value


class HostCredential(BaseModel):
    """Credentials for authenticating HTTPS access to a host during the build.

    The credential is keyed by host and applies to everything fetched from that
    host at build time -- package indices (``--extra-index-url``), PEP 508 direct
    references (``pkg @ https://host/...whl``), conda channels, and
    ``git+https://host/...`` dependencies alike. The token is supplied
    out-of-band via a build secret and assembled into netrc and git-credential
    entries inside the build stage; it never lands in the config or an image layer.
    """

    host: StrictStr = Field(
        ...,
        description=(
            "Host the credential authenticates against (e.g. ``pkgs.dev.azure.com`` "
            "or ``github.com``). Just the host, not a full URL."
        ),
    )
    secret: StrictStr = Field(
        ...,
        description=(
            "Name of the build secret carrying the token/password for this host. "
            "Supply it at build time with ``tesseract build --secret "
            "id=<name>,env=<VAR>`` (or ``,src=<file>``)."
        ),
    )
    username: StrictStr = Field(
        "__token__",
        description=(
            "Username paired with the secret. Defaults to ``__token__``, which "
            "suits PAT-style tokens; set it for hosts that require a real username."
        ),
    )
    model_config: ConfigDict = ConfigDict(extra="forbid")

    @field_validator("host", "secret", "username")
    @classmethod
    def _no_whitespace(cls, value: str, info: "ValidationInfo") -> str:
        # These values are written verbatim into the tab-separated credentials
        # file and, at build time, into netrc and git-credential entries. Rejecting
        # whitespace keeps them on a single field and prevents a crafted value from
        # injecting extra credential lines.
        if value != value.strip() or any(c.isspace() for c in value):
            raise ValueError(
                f"{info.field_name} must not contain whitespace (got {value!r})"
            )
        return value


class PipRequirements(BaseModel):
    """Configuration options for Python environments built via uv."""

    provider: Annotated[Literal["uv-pip"], BeforeValidator(_normalize_provider)]
    python_version: StrictStr | None = Field(
        None,
        description=(
            "Python version to use inside the Tesseract (e.g., '3.12'). "
            "When set, ``uv python install`` is used to install the specified version, "
            "decoupling the Python version from the base image. "
            "When unset, the system Python from the base image is used."
        ),
    )
    _filename: Literal["tesseract_requirements.txt"] = "tesseract_requirements.txt"
    _build_script: Literal["build_pip_venv.sh"] = "build_pip_venv.sh"
    model_config: ConfigDict = ConfigDict(extra="forbid")


class CondaRequirements(BaseModel):
    """Configuration options for Python environments built via conda."""

    provider: Literal["conda"]
    _filename: Literal["tesseract_environment.yaml"] = "tesseract_environment.yaml"
    _build_script: Literal["build_conda_venv.sh"] = "build_conda_venv.sh"
    model_config: ConfigDict = ConfigDict(extra="forbid")


PythonRequirements = PipRequirements | CondaRequirements


class TesseractBuildConfig(BaseModel, validate_assignment=True):
    """Configuration options for building a Tesseract."""

    base_image: StrictStr = Field(
        "debian:bookworm-slim",
        description="Base Docker image for the build. Must be Debian-based.",
    )
    target_platform: StrictStr = Field(
        "native",
        description=(
            "Target platform for the Docker build. Must be a valid Docker platform, "
            "or 'native' to build for the host platform. "
            "In general, images built for one platform will not run on another."
        ),
    )
    extra_packages: tuple[StrictStr, ...] = Field(
        (), description="Extra packages to install during build via apt-get."
    )
    package_data: tuple[tuple[RelativePath, StrictStr], ...] | None = Field(
        (),
        description=(
            "Additional files to copy into the Docker image, in the format ``(source, destination)``. "
            "Source paths are relative to the directory containing `tesseract_api.py`. "
            "Destination paths are relative or absolute paths within the Docker image (e.g., ``/app/shared_code.py``) "
            "and must be unique."
        ),
    )
    custom_build_steps: tuple[StrictStr, ...] | None = Field(
        (),
        description=(
            "Custom steps to run during ``docker build`` (after everything else is installed). "
            "Example: ``[\"RUN echo 'Hello, world!'\"]``"
        ),
    )
    python_version: StrictStr | None = Field(
        None,
        description=(
            "Deprecated alias for ``build_config.requirements.python_version``. "
            "Kept for backwards compatibility; set the version under the provider "
            "settings instead. Removed in Tesseract 1.13.0."
        ),
    )

    inherit_base_image_packages: bool = Field(
        False,
        description=(
            "If True, create the Python virtual environment with --system-site-packages "
            "so it inherits Python packages pre-installed in the base image "
            "(e.g. Firedrake, FEniCS, OpenFOAM). Cannot be combined with python_version."
        ),
    )

    requirements: PythonRequirements = PipRequirements(provider="uv-pip")

    host_credentials: tuple[HostCredential, ...] = Field(
        (),
        description=(
            "Credentials for authenticated hosts accessed during the build. Each "
            "entry maps a host to a build secret supplied out-of-band at build time "
            "(see ``HostCredential`` and ``tesseract build --secret``). Applies to "
            "package indices, direct-reference wheels, conda channels, and "
            "``git+https`` dependencies on that host."
        ),
    )

    build_env: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description=(
            "Environment variables to set during the build stage only (not in the "
            "final image). Useful for configuring the package resolver, e.g. "
            "``{UV_INDEX_STRATEGY: unsafe-best-match}``. "
            "Do not put secrets here: values are written into the build context. "
            "Use ``tesseract build --secret`` for credentials instead."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def effective_python_version(self) -> str | None:
        """Python version requested for the build, or None to use the base image's.

        Only the uv-pip provider supports pinning the version; conda pins it via
        ``tesseract_environment.yaml`` instead.
        """
        if isinstance(self.requirements, PipRequirements):
            return self.requirements.python_version
        return None

    @model_validator(mode="after")
    def _validate_python_version_provider(self):
        # Forward the deprecated build_config.python_version onto the provider so
        # the rest of the code only reads requirements.python_version (via
        # effective_python_version). Scheduled for removal in 1.13.0; see
        # tesseract_core/_deprecations.py.
        if self.python_version is not None:
            # Emit through the logger rather than relying on the Field's
            # deprecated= warning: a library-emitted DeprecationWarning is
            # suppressed under Python's default filters, so a real CLI user would
            # never see it (mirrors _normalize_provider).
            logging.getLogger("tesseract").warning(
                "build_config.python_version has moved to the uv-pip provider "
                "settings. Set `build_config.requirements.python_version` in "
                "tesseract_config.yaml; the old location still works but will be "
                "removed in Tesseract 1.13.0."
            )
            if not isinstance(self.requirements, PipRequirements):
                raise ValueError(
                    "python_version cannot be used with conda requirements. "
                    "Set the Python version in tesseract_environment.yaml instead."
                )
            if (
                self.requirements.python_version is not None
                and self.requirements.python_version != self.python_version
            ):
                raise ValueError(
                    "python_version is set both on build_config and on "
                    "build_config.requirements. Set it only once, under "
                    "build_config.requirements.python_version."
                )
            self.requirements.python_version = self.python_version

        if (
            self.effective_python_version is not None
            and self.inherit_base_image_packages
        ):
            raise ValueError(
                "python_version cannot be used with inherit_base_image_packages. "
                "inherit_base_image_packages exposes the base image's system Python "
                "packages, which belong to a different interpreter than the one "
                "installed by python_version. Set only one of the two."
            )
        return self

    skip_checks: bool = Field(
        False,
        description=(
            "If True, skip build-time checks of Tesseract API module. "
            "This can be useful when such a check cannot succeed (e.g. when building for a "
            "different platform), but may lead to runtime errors if the Tesseract API is not valid."
        ),
    )


# Allow None to be passed as a valid value for build_config, for example in YAML that comments out all options.
OptionalBuildConfig = Annotated[
    TesseractBuildConfig,
    BeforeValidator(lambda v: TesseractBuildConfig() if v is None else v),
]


class TesseractConfig(BaseModel, validate_assignment=True):
    """Configuration options for Tesseracts. Defines valid options in ``tesseract_config.yaml``."""

    name: StrictStr = Field(..., description="Name of the Tesseract.", min_length=1)
    version: StrictStr = Field(
        "unknown", description="Version of the Tesseract.", min_length=1, max_length=128
    )
    description: StrictStr = Field(
        "",
        description="Free-text description of what the Tesseract does.",
    )
    build_config: OptionalBuildConfig = Field(
        default_factory=TesseractBuildConfig,
        description="Configuration options for building the Tesseract.",
    )
    env: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description=(
            "Environment variables to set in the Docker image. "
            "Rendered as ``ENV`` lines in the Dockerfile. "
            "Example: ``{XLA_PYTHON_CLIENT_PREALLOCATE: 'false'}``"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary user-defined metadata. "
        "This will be stored as a Docker label (ai.pasteurlabs.tesseract.metadata).",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that the version string is a valid semantic version."""
        version_pattern = r"^\d+\.\d+\.\d+[a-zA-Z-0-9]*$"

        if (not re.match(version_pattern, v)) and v != "unknown":
            raise ValueError(
                f"Version '{v}' is not a valid version number for a Tesseract. "
                "You can only use three dot-separated digits (e.g. 1.2.3), to which "
                "optionally you can append a hyphen - and a string (e.g. 1.2.3-nightly)"
            )

        return v


class ValidationError(Exception):
    """Raised when inputs needed to build a tesseract are invalid."""

    pass


def _get_func_argnames(func: ast.FunctionDef) -> tuple[str, ...]:
    """Get the names of the arguments of a function node.

    See:
    https://docs.python.org/3/library/ast.html#ast.FunctionDef
    https://docs.python.org/3/library/ast.html#ast.arguments
    """
    func_args = func.args
    if func_args.kwonlyargs:
        raise ValidationError(
            f"Function {func.name} must not have keyword-only arguments"
        )
    if func_args.posonlyargs:
        raise ValidationError(
            f"Function {func.name} must not have positional-only arguments"
        )
    return tuple(arg.arg for arg in func_args.args)


def validate_tesseract_api(src_dir: Path) -> None:
    """Check that given folder contains a Tesseract API that satisfies our constraints.

    This function does not return anything, but it raises `ValidationError` in
    case something goes wrong. In particular, we are checking that:
      *  The mandatory endpoints needed for a tesseract are actually
         implemented
      *  The implemented functions have the correct signature
      *  Both InputSchema and OutputSchema are `pydantic.BaseModel`s.

    Args:
        src_dir (Path): Path to the directory containing tesseract_api.py and tesseract_config.yaml.
    """
    tesseract_api_location = src_dir / "tesseract_api.py"
    config_location = src_dir / "tesseract_config.yaml"

    if not tesseract_api_location.exists():
        raise ValidationError(f"No file found at {tesseract_api_location}")

    if not config_location.exists():
        raise ValidationError(f"No file found at {config_location}")

    # Validate config
    try:
        get_config(src_dir)
    except PydanticValidationError as err:
        raise ValidationError(
            f"Invalid configuration in {config_location}: {err}"
        ) from err

    # Parse Tesseract API
    with open(tesseract_api_location) as f:
        tesseract_api_code = f.read()

    try:
        tesseract_api = ast.parse(tesseract_api_code)
    except SyntaxError as err:
        raise ValidationError(
            f"Syntax error in {tesseract_api_location}: {err}"
        ) from err

    # Check if expected attributes are defined
    toplevel_objects = {
        node.name: node for node in tesseract_api.body if hasattr(node, "name")
    }

    for obj in EXPECTED_OBJECTS:
        if obj.name not in toplevel_objects:
            if obj.optional:
                continue

            raise ValidationError(f"{obj.name} not defined in {tesseract_api_location}")

        if not isinstance(toplevel_objects[obj.name], obj.expected_type):
            raise ValidationError(
                f"{obj.name} is not a {obj.expected_type.__name__} in {tesseract_api_location}"
            )

        if obj.num_args is not None:
            func_argnames = _get_func_argnames(toplevel_objects[obj.name])
            func_argnums = len(func_argnames)
            if func_argnums != obj.num_args:
                raise ValidationError(
                    f"{obj.name} must have {obj.num_args} arguments: {', '.join(obj.arg_names)}.\n"
                    f"However, {tesseract_api_location} specifies {func_argnums} "
                    f"arguments: {', '.join(func_argnames)}."
                )
            msgs = []
            for i in range(obj.num_args):
                if func_argnames[i] != obj.arg_names[i]:
                    msgs.append(
                        f"The {ORDINALS[i]} argument (argument {i}) of {obj.name} must be named {obj.arg_names[i]}, "
                        f"but {tesseract_api_location} has named it {func_argnames[i]}."
                    )
            if msgs:
                raise ValidationError("\n".join(msgs))

    # Check InputSchema and OutputSchema are pydantic BaseModels
    for schema in ("InputSchema", "OutputSchema"):
        obj = toplevel_objects[schema]
        if not obj.bases:
            raise ValidationError(f"{schema} must inherit from pydantic.BaseModel")


def get_config(src_dir: Path) -> TesseractConfig:
    """Get configuration options from a tesseract_config.yaml file."""
    config_file = src_dir / "tesseract_config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"No file found at {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    try:
        return TesseractConfig(**config)
    except PydanticValidationError as err:
        raise ValidationError(f"Invalid configuration: {err}") from err


def get_submodel_fields_in_tesseract_config() -> list[tuple[str, type]]:
    """Gets fields in TesseractConfig that are Pydantic sub-models."""
    non_base_fields = []
    for field_name, field_info in TesseractConfig.model_fields.items():
        origin = getattr(field_info.annotation, "__origin__", field_info.annotation)
        if isinstance(origin, type) and issubclass(origin, BaseModel):
            non_base_fields.append((field_name, field_info.annotation))
    return non_base_fields


def _submodels_in_annotation(annotation: Any) -> list[type[BaseModel]]:
    """Return the directly-settable ``BaseModel`` subclasses in a field annotation.

    Handles bare models, optionals (``Model | None``), and unions of models
    (e.g. the ``requirements`` field, which is ``PipRequirements | CondaRequirements``).
    Models nested inside collections (``tuple[HostCredential, ...]``) are ignored:
    their elements cannot be reached by a dotted ``--config-override`` keypath.
    """
    import types
    import typing

    origin = typing.get_origin(annotation)
    # Only descend through bare annotations and unions -- not list/tuple/dict.
    if origin not in (None, typing.Union, types.UnionType):
        return []

    args = typing.get_args(annotation)
    candidates = args if args else (annotation,)
    seen: dict[type[BaseModel], None] = {}
    for arg in candidates:
        # ``isinstance(arg, type)`` is not enough to guarantee ``issubclass`` won't
        # raise: some parameterized generics report as types on older pydantic/typing
        # versions but reject ``issubclass``. Guard the check so such args are simply
        # treated as non-models.
        try:
            is_model = isinstance(arg, type) and issubclass(arg, BaseModel)
        except TypeError:
            is_model = False
        if is_model:
            seen.setdefault(arg, None)
    return list(seen)


def get_config_keypaths(model: type[BaseModel] = TesseractConfig) -> list[str]:
    """Enumerate dot-separated config keypaths a ``--config-override`` may target.

    Recurses through nested sub-models (including union members like the
    requirements providers) so nested attributes such as
    ``build_config.requirements.python_version`` are surfaced, not just top-level
    fields. Both intermediate sub-model paths and leaf paths are returned.
    """
    # dict preserves insertion order and dedupes paths shared across union members
    # (e.g. `provider` on both requirements providers).
    keypaths: dict[str, None] = {}

    def _walk(current: type[BaseModel], prefix: str) -> None:
        for field_name, field_info in current.model_fields.items():
            path = f"{prefix}{field_name}"
            keypaths.setdefault(path, None)
            for submodel in _submodels_in_annotation(field_info.annotation):
                _walk(submodel, f"{path}.")

    _walk(model, "")
    return list(keypaths)
