#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Set up build-time credentials for authenticated hosts. Sourced (not executed)
# by the provider build scripts so the exported NETRC reaches the install step.
#
# Each line of host_credentials.txt is "<host>\t<secret_id>\t<username>". The
# token is read from its BuildKit secret mount at /run/secrets/<id> and written
# to two credential stores, so one entry covers every fetch from that host:
#   - netrc, via the NETRC env var: read by uv, pip, and conda over HTTPS
#   - git's "store" helper: read by git+https dependencies
#
# Both stores are written under /tmp/tesseract-credentials, which the Dockerfile
# mounts as a tmpfs. The plaintext token therefore lives only on the secret mount
# and this tmpfs, neither of which is part of a build layer. Nothing reaches an
# image layer, the build cache, or host_credentials.txt.

# Percent-encode a string per RFC 3986, leaving unreserved characters as-is.
# Encodes one byte at a time (via `od`) so multi-byte UTF-8 tokens are handled
# correctly; a naive character loop mis-encodes bytes >= 0x80.
_urlencode() {
    local hex byte
    while read -r hex; do
        [ -z "$hex" ] && continue
        byte=$(printf "\\x$hex")
        case "$byte" in
            [a-zA-Z0-9.~_-]) printf '%s' "$byte" ;;
            *) printf '%%%s' "$(printf '%s' "$hex" | tr '[:lower:]' '[:upper:]')" ;;
        esac
    done < <(printf '%s' "$1" | od -An -tx1 | tr -s ' ' '\n')
}

if [ -f host_credentials.txt ]; then
    credentials_dir=/tmp/tesseract-credentials
    netrc_file="$credentials_dir/netrc"
    git_credentials_file="$credentials_dir/git-credentials"
    : > "$netrc_file"
    : > "$git_credentials_file"
    chmod 600 "$netrc_file" "$git_credentials_file"

    # git does not read the NETRC env var, so the same credential is written in
    # git's own format. The store file is scoped to this build via --file rather
    # than the default ~/.git-credentials, keeping it on the tmpfs.
    git config --global credential.helper "store --file=$git_credentials_file"

    while IFS=$'\t' read -r host secret_id username; do
        [ -z "$host" ] && continue
        [ -f "/run/secrets/$secret_id" ] || continue
        # Read only the first line of the secret: a trailing newline is common and
        # harmless, but an embedded newline would forge additional credential
        # entries, so anything past the first line is dropped. `read` returns
        # non-zero when the line has no trailing newline even though it sets the
        # variable, so the failure is tolerated to stay compatible with `set -e`.
        IFS= read -r password < "/run/secrets/$secret_id" || true
        printf 'machine %s login %s password %s\n' \
            "$host" "$username" "$password" >> "$netrc_file"
        # The git-credentials URL form requires percent-encoded userinfo so that
        # a token containing ":", "@", "/" etc. cannot break the URL.
        printf 'https://%s:%s@%s\n' \
            "$(_urlencode "$username")" "$(_urlencode "$password")" "$host" \
            >> "$git_credentials_file"
    done < host_credentials.txt

    export NETRC="$netrc_file"
fi
