#!/bin/sh
# Realtime of 1, 2 and 3 public ducks running at once on one Linux Docker host (e.g. an EC2 t3.large).
#   measure_many.sh IMAGE [MINUTES] [OUTDIR] [COUNTS]
#   IMAGE    the public duck image for THIS machine's architecture (build.sh REF linux/amd64)
#   MINUTES  per count, default 10
#   OUTDIR   default ./measure-many-<date>
#   COUNTS   default "1 2 3"
# For each count N it starts N ducks (127.0.0.1:18081.., no CPU limit: they share the host as they would
# in production, 3 GiB memory each), probes every duck at once with realtime_probe.py --page-load (each duck
# polled as if a visitor had its page open), and records the host's CPU use and steal time from /proc/stat
# (on a burstable t3, steal rising means the CPU credits ran out and the host is held at its baseline).
# Prints one line per duck and one per count; exit status 0 only when every duck of every count kept
# realtime (median >= 0.95, >= 95 % of samples at or above it). Runs measurement containers only: they bind
# 127.0.0.1 and, when the image has no --public yet, serve every endpoint (DUCKMIND_IMAGE_MEASURE=1).
# Needs docker and python3 on a Linux host. Sends nothing anywhere.
set -eu
IMAGE=$1
MINUTES=${2:-10}
OUT=${3:-./measure-many-$(date +%Y%m%d-%H%M%S)}
COUNTS=${4:-1 2 3}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
cpu() { awk '/^cpu /{print $2+$3+$4+$7+$8, $9, $2+$3+$4+$5+$6+$7+$8+$9}' /proc/stat; }  # busy steal total
names=""
cleanup() { for n in $names; do docker stop -t 5 "$n" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT
all_ok=0
for N in $COUNTS; do
  names=""; i=1
  while [ "$i" -le "$N" ]; do
    name="dm-many-$$-$N-$i"; port=$((18080 + i))
    docker run -d --rm --name "$name" --memory 3g --memory-swap 3g -p "127.0.0.1:$port:8080" \
      -e DUCKMIND_PUBLIC_SEED=$((4240 + i)) -e DUCKMIND_IMAGE_MEASURE=1 \
      --read-only --tmpfs /tmp:rw,exec,size=1g "$IMAGE" >/dev/null
    names="$names $name"; i=$((i + 1))
  done
  set -- $(cpu); b0=$1; s0=$2; t0=$3
  i=1; pids=""
  while [ "$i" -le "$N" ]; do
    python3 "$HERE/realtime_probe.py" "http://127.0.0.1:$((18080 + i))" --minutes "$MINUTES" --page-load \
      --out "$OUT/n$N-duck$i.jsonl" >"$OUT/n$N-duck$i.summary.json" 2>&1 &
    pids="$pids $!"; i=$((i + 1))
  done
  for p in $pids; do wait "$p" || all_ok=1; done
  set -- $(cpu); b1=$1; s1=$2; t1=$3
  cores=$(getconf _NPROCESSORS_ONLN)
  for n in $names; do
    docker exec "$n" cat /sys/fs/cgroup/memory.peak 2>/dev/null | awk -v n="$n" '{printf "%s memory.peak %.2f GiB\n", n, $1/2^30}'
  done >"$OUT/n$N-memory.txt"
  cleanup
  awk -v N="$N" -v c="$cores" -v b=$((b1 - b0)) -v s=$((s1 - s0)) -v t=$((t1 - t0)) 'BEGIN {
    printf "ducks=%d host: busy %.2f of %d cores, steal %.1f %%\n", N, c * b / t, c, 100 * s / t }' \
    | tee "$OUT/n$N-host.txt"
  i=1
  while [ "$i" -le "$N" ]; do
    printf 'ducks=%d duck %d: %s\n' "$N" "$i" "$(cat "$OUT/n$N-duck$i.summary.json")"; i=$((i + 1))
  done
  cat "$OUT/n$N-memory.txt"
  names=""
done
exit "$all_ok"
