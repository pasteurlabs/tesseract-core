#!/bin/sh

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Append the current uid:gid to the pre-populated passwd/group files so that
# NSS lookups work for any uid the container is run as. libnss_wrapper
# intercepts those lookups via LD_PRELOAD — no privilege escalation required,
# so this works even with --security-opt no-new-privileges.
#
# /tmp/passwd and /tmp/group are pre-seeded from /etc at image build time. Under
# Apptainer, however, /tmp is a fresh tmpfs (from --writable-tmpfs) that shadows
# the image's pre-seeded files, so they are missing here. Create them if absent
# so nss_wrapper always has a file to read, and silence the lookup either way.
[ -f /tmp/passwd ] || touch /tmp/passwd
[ -f /tmp/group ] || touch /tmp/group
grep -q "^tesseract-user:x:$(id -u):" /tmp/passwd 2>/dev/null || \
    echo "tesseract-user:x:$(id -u):$(id -g)::/home/tesseract-user:/bin/bash" >> /tmp/passwd
grep -q "^tesseract-group:x:$(id -g):" /tmp/group 2>/dev/null || \
    echo "tesseract-group:x:$(id -g):" >> /tmp/group

exec "$@"
