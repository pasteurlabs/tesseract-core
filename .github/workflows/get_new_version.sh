#!/bin/bash

set -e

new_version=$(git-cliff --bumped-version)

# Prevent bumping major versions, downgrade to minor in this case
current_major=$(cut -d '.' -f 1 <<< "$current_version")
new_major=$(cut -d '.' -f 1 <<< "$new_version")
if [[ "$current_major" != "$new_major" ]]; then
    echo "GIT_CLIFF__BUMP__BUMP_TYPE=minor" >> $GITHUB_ENV
    new_version=$(git-cliff --bumped-version)
fi

echo $new_version
