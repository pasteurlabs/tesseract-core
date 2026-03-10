#!/bin/bash

set -e

bump_type="auto"
current_version=$(gh release view --json tagName --jq '.tagName' || echo "v0.0.0")

# Prevent bumping major versions, downgrade to minor in this case
new_version=$(git-cliff --bumped-version --bump $bump_type)
current_major=$(cut -d '.' -f 1 <<< "$current_version")
new_major=$(cut -d '.' -f 1 <<< "$new_version")
if [[ "$current_major" != "$new_major" ]]; then
    bump_type="minor"
fi

echo $bump_type
