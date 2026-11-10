#!/bin/sh
# Build the public duck's image from one git ref of this repository (never the working tree's edits).
#   build.sh [REF] [PLATFORM] [TAG]
#   REF       default HEAD
#   PLATFORM  default linux/amd64 (Cloudflare Containers); linux/arm64 to measure on an Apple-silicon Mac
#   TAG       default duckmind-public:<short ref>-<arch>
# Needs docker with buildx (BuildKit). Pushes nothing: the image is loaded into the local docker only.
set -eu
REF="${1:-HEAD}"
PLATFORM="${2:-linux/amd64}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SHORT="$(git -C "$ROOT" rev-parse --short "$REF^{commit}")"
TAG="${3:-duckmind-public:$SHORT-${PLATFORM#linux/}}"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git -C "$ROOT" archive "$REF" | tar -x -C "$CTX"
docker buildx build --platform "$PLATFORM" --load \
  --label "org.opencontainers.image.revision=$(git -C "$ROOT" rev-parse "$REF^{commit}")" \
  -f "$CTX/deploy/public/image/Dockerfile" -t "$TAG" "$CTX"
echo "$TAG"
