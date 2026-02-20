#!/bin/sh

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Append the current uid:gid to the pre-populated passwd/group files so that
# NSS lookups work for any uid the container is run as. libnss_wrapper
# intercepts those lookups via LD_PRELOAD — no privilege escalation required,
# so this works even with --security-opt no-new-privileges.
# /tmp/passwd and /tmp/group are pre-seeded from /etc at image build time.
grep -q "^tesseract-user:x:$(id -u):" /tmp/passwd || \
    echo "tesseract-user:x:$(id -u):$(id -g)::/tesseract:/bin/bash" >> /tmp/passwd
grep -q "^tesseract-group:x:$(id -g):" /tmp/group || \
    echo "tesseract-group:x:$(id -g):" >> /tmp/group

exec "$@"
