# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import (
    Any,
)

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema


class TesseractReference:
    """Allows passing a reference to another Tesseract as input."""

    def __init__(self, tesseract: Any) -> None:
        self._tesseract = tesseract

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Tesseract instance."""
        return getattr(self._tesseract, name)

    @classmethod
    def _get_tesseract_class(cls) -> type:
        """Lazy import of Tesseract class. Avoids hard dependency of Tesseract runtime on Tesseract SDK."""
        try:
            from tesseract_core import Tesseract

            return Tesseract
        except ImportError:
            raise ImportError(
                "Tesseract class not found. Ensure tesseract_core is installed and configured correctly."
            ) from ImportError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for TesseractReference."""

        def validate_tesseract_reference(v: Any) -> "TesseractReference":
            if isinstance(v, cls):
                return v

            if not (isinstance(v, dict) and "type" in v and "ref" in v):
                raise ValueError(
                    f"Expected dict with 'type' and 'ref' keys, got {type(v)}"
                )

            tesseract_type = v["type"]
            ref = v["ref"]

            if tesseract_type not in ("api_path", "image", "url"):
                raise ValueError(
                    f"Invalid tesseract type '{tesseract_type}'. Expected 'api_path', 'image' or 'url'."
                )

            Tesseract = cls._get_tesseract_class()
            if tesseract_type == "api_path":
                tesseract = Tesseract.from_tesseract_api(ref)
            elif tesseract_type == "image":
                tesseract = Tesseract.from_image(ref)
                tesseract.serve()
            elif tesseract_type == "url":
                tesseract = Tesseract.from_url(ref)

            return cls(tesseract)

        return core_schema.no_info_plain_validator_function(
            validate_tesseract_reference
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Generate JSON schema for OpenAPI."""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["api_path", "image", "url"],
                    "description": "Type of tesseract reference",
                },
                "ref": {
                    "type": "string",
                    "description": "URL or file path to the tesseract",
                },
            },
            "required": ["type", "ref"],
            "additionalProperties": False,
        }
