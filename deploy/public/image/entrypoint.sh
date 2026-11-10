#!/bin/sh
# The public duck's container: one duck for one visitor (deploy/public/image/Dockerfile).
#   DUCKMIND_PUBLIC_SEED     the duck's seed (a non-negative integer; the edge picks it)    required
#   DUCKMIND_PUBLIC_MINUTES  the visit's length (simweb --public reads it)                  default 15
# Everything the duck writes goes under /tmp; the app files are read-only.
set -eu
seed="${DUCKMIND_PUBLIC_SEED:-}"
case "$seed" in
  '' | *[!0-9]*) echo "DUCKMIND_PUBLIC_SEED must be a non-negative integer" >&2; exit 2 ;;
esac
mkdir -p "$HOME" "$DUCKMIND_HOME"
U=/opt/duckmind/upstream
set -- --host 0.0.0.0 --port 8080 --live \
  --microduck "$U/microduck" --microduck-rl "$U/microduck_rl" \
  --policies "$DUCKMIND_POLICY_DIR" --robotd-bin /opt/duckmind/bin/robotd \
  --brain neuro --home real_home --no-nest --tof
if python -m duckmind.simweb --help 2>/dev/null | grep -q -- '--public'; then
  exec python -m duckmind.simweb --public "$@"
fi
# A duckmind without the public mode serves every endpoint: never for visitors. Only an explicit
# measurement run may start it, with what --public would choose (seed, a fresh life, growth day 32).
if [ "${DUCKMIND_IMAGE_MEASURE:-}" != "1" ]; then
  echo "this image's duckmind has no --public mode; refusing to serve visitors" >&2
  exit 70
fi
echo "MEASUREMENT RUN: duckmind without --public (all endpoints open); never expose this" >&2
exec python -m duckmind.simweb "$@" --seed "$seed" --life "$DUCKMIND_HOME/life" --life-backdate-days 32
