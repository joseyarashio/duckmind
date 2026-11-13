#!/bin/sh
# The ducks' own Docker network and the host firewall that keeps them off the internet.
#   firewall.sh [NETWORK] [BRIDGE]     (run as root; duckmind-firewall.service runs it after docker)
# Idempotent: makes the network if it is missing and adds each rule only once.
#   - every packet a duck sends that the host would route on (the internet, the VPC, the instance
#     metadata service) is dropped: DOCKER-USER, the chain Docker leaves to the owner;
#   - a duck cannot open a connection to the host itself (sshd or anything else listening): INPUT;
#   - ducks cannot talk to each other (icc off) and are never masqueraded.
# What still works: the front's own connections to a duck's published port on 127.0.0.1, and the
# duck's answers to them (they are not routed, and not new).
set -eu
NET="${1:-duckmind-public}"
BRIDGE="${2:-dm-duck0}"

if ! docker network inspect "$NET" >/dev/null 2>&1; then
  docker network create --driver bridge \
    --opt com.docker.network.bridge.name="$BRIDGE" \
    --opt com.docker.network.bridge.enable_icc=false \
    --opt com.docker.network.bridge.enable_ip_masquerade=false \
    --label duckmind.front.network=1 \
    "$NET" >/dev/null
fi
actual="$(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "$NET")"
if [ "$actual" != "$BRIDGE" ]; then
  echo "network $NET uses bridge '$actual', not $BRIDGE: remove it (docker network rm $NET) and run again" >&2
  exit 1
fi

rule() {  # rule CHAIN ARGS...: add at the top unless it is there
  chain=$1; shift
  iptables -C "$chain" "$@" 2>/dev/null || iptables -I "$chain" 1 "$@"
}
iptables -N DOCKER-USER 2>/dev/null || true
rule DOCKER-USER -i "$BRIDGE" -j DROP
rule INPUT -i "$BRIDGE" -m conntrack --ctstate NEW -j DROP
echo "ducks' network $NET ($BRIDGE): no route out, no new connections to the host"
