#!/bin/bash
set -e

# Release script: bump version, commit, tag, and push
# Usage: ./scripts/release.sh [patch|minor|major]

BUMP_TYPE="${1:-patch}"
TOML_FILE="pyproject.toml"

# Get the last tag
LAST_TAG=$(git tag --list --sort=-v:refname | head -1)

if [ -z "$LAST_TAG" ]; then
    echo "No existing tags found. Starting from v0.0.0"
    LAST_TAG="v0.0.0"
fi

# Extract version numbers (strip 'v' prefix)
VERSION="${LAST_TAG#v}"
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

# Bump version based on type
case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Invalid bump type: $BUMP_TYPE"
        echo "Usage: $0 [patch|minor|major]"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
NEW_TAG="v$NEW_VERSION"

echo "Current version: $LAST_TAG"
echo "New version: $NEW_TAG"
echo ""

# Update version in pyproject.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$TOML_FILE"
else
    # Linux
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$TOML_FILE"
fi

echo "Updated $TOML_FILE"

# Stage, commit, tag, and push
git add "$TOML_FILE"
git commit -m "Bump version to $NEW_VERSION"
git tag -a "$NEW_TAG" -m "Release $NEW_TAG"

echo ""
echo "Pushing commit and tag..."
git push && git push origin "$NEW_TAG"

echo ""
echo "✅ Released $NEW_TAG successfully!"
