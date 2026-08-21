#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Assemble host credentials from any entries declared in tesseract_config.yaml,
# for use by the dependency install step. Meant to be sourced (not executed) so
# the exported NETRC persists into the caller's environment.
#
# Each line of host_credentials.txt is "<host>\t<secret_id>\t<username>"; the
# token is read from the BuildKit secret mount at /run/secrets/<id>. For every
# credential we set up two consumers, so a single entry authenticates the whole
# host:
#   - netrc (via NETRC env var)      -> uv, pip, and conda HTTPS fetches
#   - git-credentials (helper=store) -> git+https dependencies
#
# Credentials only ever live on the mounted tmpfs and in these files inside the
# build stage -- never in a layer or in host_credentials.txt.

if [ -f host_credentials.txt ]; then
    netrc_file=$(mktemp)
    chmod 600 "$netrc_file"

    # git reads ~/.git-credentials via the "store" helper; it does not honor the
    # NETRC env var, so we write both formats from the same credential.
    git_credentials_file="$HOME/.git-credentials"
    git config --global credential.helper store

    while IFS=$'\t' read -r host secret_id username; do
        [ -z "$host" ] && continue
        [ -f "/run/secrets/$secret_id" ] || continue
        password=$(cat "/run/secrets/$secret_id")
        printf 'machine %s login %s password %s\n' \
            "$host" "$username" "$password" >> "$netrc_file"
        printf 'https://%s:%s@%s\n' \
            "$username" "$password" "$host" >> "$git_credentials_file"
    done < host_credentials.txt

    chmod 600 "$git_credentials_file"
    export NETRC="$netrc_file"
fi
