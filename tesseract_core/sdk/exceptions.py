# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class UserError(Exception):
    """Exception raised for anything that is the user's fault."""

    pass


class ValidationError(UserError):
    """Exception raised for input validation errors."""

    pass
