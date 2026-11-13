#!/bin/sh
# Put one secret where the services read it, from standard input (never on a command line):
#   ... | sudo deploy/public/host/put-secret.sh turnstile   TURNSTILE_SECRET in /etc/duckmind-public/secrets.env
#   ... | sudo deploy/public/host/put-secret.sh tunnel      /etc/duckmind-public/tunnel-token
# Files stay root:root 0600; the other lines of secrets.env are kept. Restart the service afterwards.
set -eu
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
ETC=/etc/duckmind-public
umask 077
value="$(head -n 1 | tr -d '\r\n')"
case "$value" in
  '' | *[!A-Za-z0-9._=+/-]*) echo "expected one line of letters, digits and ._=+/-" >&2; exit 2 ;;
esac
case "${1:-}" in
  turnstile)
    touch "$ETC/secrets.env"
    { grep -v '^TURNSTILE_SECRET=' "$ETC/secrets.env" || true; printf 'TURNSTILE_SECRET=%s\n' "$value"; } >"$ETC/secrets.env.new"
    mv "$ETC/secrets.env.new" "$ETC/secrets.env"
    chown root:root "$ETC/secrets.env"
    chmod 0600 "$ETC/secrets.env"
    ;;
  tunnel)
    printf '%s\n' "$value" >"$ETC/tunnel-token.new"
    mv "$ETC/tunnel-token.new" "$ETC/tunnel-token"
    chown root:root "$ETC/tunnel-token"
    chmod 0600 "$ETC/tunnel-token"
    ;;
  *) echo "usage: put-secret.sh turnstile|tunnel  (the value on standard input)" >&2; exit 2 ;;
esac
echo "stored ${1} ($(printf '%s' "$value" | wc -c | tr -d ' ') characters)"
