#!/bin/sh
# Install or update the public play front on this host (Amazon Linux 2023 with docker). Idempotent:
# settings and secrets that exist are kept; code, fonts, pictures and units are replaced.
#   sudo deploy/public/host/install.sh         (from a checkout of the repository on the host)
# It does NOT start the front or the tunnel, and touches no running duck; at the end it prints what
# to do next. Installs (once): Node.js 22 and cloudflared from the distribution's and Cloudflare's
# package repositories.
set -eu
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PREFIX=/opt/duckmind-front
ETC=/etc/duckmind-public

# Node.js 22.18 or later runs the front's TypeScript as it is (no build step, no packages).
node_ok() { "$1" -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=18)?0:1)' 2>/dev/null; }
NODE=""
for n in /usr/bin/node-22 /usr/bin/node; do
  if [ -x "$n" ] && node_ok "$n"; then NODE=$n; break; fi
done
if [ -z "$NODE" ]; then
  dnf install -y -q nodejs22
  for n in /usr/bin/node-22 /usr/bin/node; do
    if [ -x "$n" ] && node_ok "$n"; then NODE=$n; break; fi
  done
fi
[ -n "$NODE" ] || { echo "no Node.js 22.18+ found" >&2; exit 1; }

# cloudflared 2025.4.0 or later (--token-file), from Cloudflare's repository.
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL https://pkg.cloudflare.com/cloudflared.repo -o /etc/yum.repos.d/cloudflared.repo
  dnf install -y -q cloudflared
fi
cloudflared tunnel run --help 2>/dev/null | grep -q -- '--token-file' \
  || { echo "cloudflared is too old for --token-file: dnf upgrade cloudflared" >&2; exit 1; }

# The front's user: no login, no home, in the docker group (it starts the duck containers).
getent group duckfront >/dev/null || groupadd --system duckfront
id duckfront >/dev/null 2>&1 \
  || useradd --system --gid duckfront --no-create-home --home-dir /nonexistent --shell /sbin/nologin duckfront
usermod -a -G docker duckfront

# Code (the shared front modules and the host's), the simulator's fonts, the pages' pictures, the
# firewall script.
install -d -m 0755 "$PREFIX"
NEW="$PREFIX/app.new"
rm -rf "$NEW" "$PREFIX/fonts.new" "$PREFIX/images.new"
install -d -m 0755 "$NEW"
cp -R "$ROOT/deploy/public/edge/src/core" "$ROOT/deploy/public/edge/src/host" "$NEW/"
printf '{ "private": true, "type": "module" }\n' >"$NEW/package.json"
git -C "$ROOT" rev-parse HEAD >"$NEW/REVISION"
cp -R "$ROOT/src/duckmind/simweb/static/fonts" "$PREFIX/fonts.new"
cp -R "$ROOT/deploy/public/edge/static/img" "$PREFIX/images.new"
chown -R root:root "$NEW" "$PREFIX/fonts.new" "$PREFIX/images.new"
chmod -R a+rX,go-w "$NEW" "$PREFIX/fonts.new" "$PREFIX/images.new"
rm -rf "$PREFIX/app" "$PREFIX/fonts" "$PREFIX/images"
mv "$NEW" "$PREFIX/app"
mv "$PREFIX/fonts.new" "$PREFIX/fonts"
mv "$PREFIX/images.new" "$PREFIX/images"
install -m 0755 "$HERE/firewall.sh" "$PREFIX/firewall.sh"
ln -sfn "$NODE" "$PREFIX/node"

# Settings (kept once written) and secrets (root only; systemd hands them to the front).
install -d -m 0755 "$ETC"
[ -f "$ETC/front.env" ] || install -m 0644 "$HERE/front.env.example" "$ETC/front.env"
# Settings added since an earlier install (the rest of a kept front.env stays as it is).
grep -q '^DUCK_HOST_IMAGES=' "$ETC/front.env" || echo 'DUCK_HOST_IMAGES=/opt/duckmind-front/images' >>"$ETC/front.env"
if [ ! -f "$ETC/secrets.env" ]; then
  (umask 077; printf 'TURNSTILE_SECRET=\nIP_HASH_KEY=%s\n' "$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')" >"$ETC/secrets.env")
fi
chown root:root "$ETC/secrets.env"
chmod 0600 "$ETC/secrets.env"
[ ! -f "$ETC/tunnel-token" ] || { chown root:root "$ETC/tunnel-token"; chmod 0600 "$ETC/tunnel-token"; }

# Units: the firewall now; the front and the tunnel are enabled (start at boot) but started by hand.
install -m 0644 "$HERE/duckmind-firewall.service" "$HERE/duckmind-front.service" \
  "$HERE/duckmind-cloudflared.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now duckmind-firewall.service
systemctl restart duckmind-firewall.service
systemctl enable duckmind-front.service duckmind-cloudflared.service >/dev/null 2>&1

echo "installed $(cat "$PREFIX/app/REVISION") with $("$NODE" --version) and $(cloudflared --version 2>&1 | head -1)"
echo "next: settings in $ETC/front.env, TURNSTILE_SECRET in $ETC/secrets.env, the tunnel token in $ETC/tunnel-token,"
echo "      an image: $HERE/refresh-image.sh <ref> (as the checkout's owner); then: systemctl restart duckmind-front && systemctl start duckmind-cloudflared"
