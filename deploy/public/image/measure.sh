#!/bin/sh
# Measure one public duck locally under a CPU / memory limit (deploy/public/image/realtime_probe.py).
#   measure.sh IMAGE CPUS MEMORY MINUTES OUTDIR [SEED]      e.g. measure.sh duckmind-public:abc-arm64 1 6g 10 /tmp/m
# Runs the container on 127.0.0.1:18080 only, with DUCKMIND_IMAGE_MEASURE=1 when the image's duckmind has no
# --public yet (never expose such a container). $MEASURE_DOCKER_ARGS adds docker run options (e.g. -e NAME=value).
# Writes OUTDIR/<cpus>cpu-<mem>/{probe.jsonl,summary.json,stats.tsv,log.txt}.
set -eu
IMAGE=$1; CPUS=$2; MEM=$3; MINUTES=$4; OUT=$5; SEED=${6:-4242}
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$OUT/${CPUS}cpu-$MEM"; mkdir -p "$DIR"
NAME="dm-measure-$$"
docker run -d --rm --name "$NAME" --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM" \
  -p 127.0.0.1:18080:8080 -e DUCKMIND_PUBLIC_SEED="$SEED" -e DUCKMIND_IMAGE_MEASURE=1 \
  --read-only --tmpfs /tmp:rw,exec,size=1g ${MEASURE_DOCKER_ARGS:-} "$IMAGE" >/dev/null
finish() {
  docker exec "$NAME" cat /sys/fs/cgroup/memory.peak /sys/fs/cgroup/cpu.stat >"$DIR/cgroup.txt" 2>&1 || true
  docker logs "$NAME" >"$DIR/log.txt" 2>&1 || true
  docker stop -t 5 "$NAME" >/dev/null 2>&1 || true
}
trap finish EXIT
( while docker inspect "$NAME" >/dev/null 2>&1; do
    docker stats --no-stream --format '{{.CPUPerc}}\t{{.MemUsage}}' "$NAME" 2>/dev/null | sed "s/^/$(date +%s)\t/"
    sleep 30
  done ) >"$DIR/stats.tsv" &
python3 "$HERE/realtime_probe.py" http://127.0.0.1:18080 --minutes "$MINUTES" --out "$DIR/probe.jsonl" \
  | tee "$DIR/summary.json" || true
