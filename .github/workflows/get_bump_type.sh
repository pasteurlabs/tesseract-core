#!/bin/bash

set -e

new_version=$(git-cliff --bumped-version)

# Prevent bumping major versions, downgrade to minor in this case
current_major=$(cut -d '.' -f 1 <<< "$current_version")
new_major=$(cut -d '.' -f 1 <<< "$new_version")
if [[ "$current_major" != "$new_major" ]]; then
    bump_type=$(git-cliff --bumped-version --bump minor)
else
    bump_type="auto"
fi

echo $bump_type
