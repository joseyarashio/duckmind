#!/bin/sh
# Build the duck image from one ref of the repository on this host and give it to the front for the
# next duck (the playing duck keeps its own image to the end).
#   deploy/public/host/refresh-image.sh <ref>     e.g. origin/main, a tag, or a commit
# Run it as the checkout's owner (in the docker group); it uses sudo only to write the front's
# settings and reload it. Fetches the repository's origin first. Keeps the new image and the one it
# replaces; older duckmind-public images not in use are removed.
set -eu
REF="${1:?usage: refresh-image.sh <ref>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
CONF=/etc/duckmind-public/front.env
case "$(uname -m)" in
  aarch64 | arm64) PLATFORM=linux/arm64 ;;
  x86_64) PLATFORM=linux/amd64 ;;
  *) echo "unknown machine $(uname -m)" >&2; exit 1 ;;
esac

git -C "$ROOT" fetch --quiet origin
COMMIT="$(git -C "$ROOT" rev-parse --verify "$REF^{commit}")"
TAG="$(sh "$ROOT/deploy/public/image/build.sh" "$COMMIT" "$PLATFORM" | tail -n 1)"
docker image inspect "$TAG" >/dev/null

OLD="$(sed -n 's/^DUCK_HOST_IMAGE=//p' "$CONF" | tail -n 1)"
if grep -q '^DUCK_HOST_IMAGE=' "$CONF"; then
  sudo sed -i "s|^DUCK_HOST_IMAGE=.*|DUCK_HOST_IMAGE=$TAG|" "$CONF"
else
  printf 'DUCK_HOST_IMAGE=%s\n' "$TAG" | sudo tee -a "$CONF" >/dev/null
fi
if systemctl is-active --quiet duckmind-front; then sudo systemctl reload duckmind-front; fi
echo "the next duck runs $TAG (was ${OLD:-none})"

for image in $(docker image ls --format '{{.Repository}}:{{.Tag}}' duckmind-public); do
  [ "$image" = "$TAG" ] || [ "$image" = "$OLD" ] || docker image rm "$image" >/dev/null 2>&1 || true
done
