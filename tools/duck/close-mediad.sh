#!/bin/sh
# Close Pollen's unauthenticated LAN doors on a Microduck while Duck Mind's bridge is in use.
#
# mediad listens on every interface, TCP 8443 (WebRTC signalling) and 8080 (console page), with no
# password; netpadd takes a gamepad on UDP 4210 when [netpad] is enabled. Anyone on the home
# network can drive the duck through them, around the bridge and its deadman. This puts one
# nftables table ("inet duckmind") in front of them:
#   - TCP 8443, 8080 and UDP 4210 are dropped unless they come over loopback TO a loopback
#     address (the duck's own processes); a connection from the duck to its own LAN address is
#     dropped too, which is how the bridge's start check sees from the duck that they are closed;
#   - the bridge's port (8797) takes only the brain computer's address.
# ssh (22), Bluetooth, duckctl and the duck's updates are untouched.
#
# Run it ON THE DUCK, as root. By default the table lives in memory: a reboot, or --undo, gives
# Pollen's setup back. --at-boot also installs a systemd unit that loads it at every start (an
# OS update that replaces /etc removes it: run this again; `remotebody check` says so).
#
#   sudo sh close-mediad.sh --brain 10.0.0.20            close (in memory)
#   sudo sh close-mediad.sh --brain 10.0.0.20 --at-boot  close now and at every boot
#   sudo sh close-mediad.sh --status                        is the table in? is it kept at boot?
#   sudo sh close-mediad.sh --undo                          open again (and forget --at-boot)
#   sh close-mediad.sh --brain 10.0.0.20 --print         only print the table; change nothing
#
# Not verified on a duck yet (the hardware is awaiting delivery): check first that `nft` exists
# on the factory image (`command -v nft`) and that `sudo nft list ruleset` answers.
set -eu

TABLE="inet duckmind"
BRIDGE_PORT=8797
BRAIN=""
MODE=close
AT_BOOT=0
# (DESTDIR: a staging root for the boot files, as install scripts take it; empty on a duck)
RULES_FILE="${DESTDIR:-}/etc/duckmind/close-mediad.nft"
UNIT_NAME="duckmind-close-mediad.service"
UNIT_FILE="${DESTDIR:-}/etc/systemd/system/$UNIT_NAME"

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "close-mediad: $*" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --brain) [ $# -ge 2 ] || die "--brain needs the brain computer's LAN address"; BRAIN="$2"; shift 2 ;;
        --bridge-port) [ $# -ge 2 ] || die "--bridge-port needs a port"; BRIDGE_PORT="$2"; shift 2 ;;
        --at-boot) AT_BOOT=1; shift ;;
        --undo) MODE=undo; shift ;;
        --status) MODE=status; shift ;;
        --print) MODE=print; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

# An IPv4 address, four numbers of 0..255 and nothing else (it goes into the rules as it is).
valid_ipv4() {
    case "$1" in
        ""|*[!0-9.]*|.*|*.|*..*) return 1 ;;
    esac
    old_ifs=$IFS
    IFS=.
    # shellcheck disable=SC2086
    set -- $1
    IFS=$old_ifs
    [ $# -eq 4 ] || return 1
    for part in "$@"; do
        [ ${#part} -le 3 ] && [ "$part" -le 255 ] || return 1
    done
}

valid_port() {
    case "$1" in
        ""|*[!0-9]*) return 1 ;;
    esac
    [ ${#1} -le 5 ] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

rules() {
    cat <<EOF
table $TABLE {
  chain input {
    type filter hook input priority 0; policy accept;
    iif "lo" ip daddr 127.0.0.0/8 accept
    iif "lo" ip6 daddr ::1 accept
    tcp dport { 8443, 8080 } drop
    udp dport 4210 drop
    meta nfproto ipv4 tcp dport $BRIDGE_PORT ip saddr != $BRAIN drop
    meta nfproto ipv6 tcp dport $BRIDGE_PORT drop
  }
}
EOF
}

# The table replaced in one transaction (declared, deleted, defined again): never half in.
transaction() {
    echo "table $TABLE"
    echo "delete table $TABLE"
    rules
}

need_nft() {
    command -v nft >/dev/null 2>&1 || die "nft is not on this duck (try the iptables lines in docs/deploy/board.md, section 5)"
}

table_in() {
    nft list table $TABLE >/dev/null 2>&1
}

case "$MODE" in
    print)
        valid_ipv4 "$BRAIN" || die "--brain needs the brain computer's IPv4 address (got '$BRAIN')"
        valid_port "$BRIDGE_PORT" || die "--bridge-port is not a port: '$BRIDGE_PORT'"
        rules
        ;;
    close)
        valid_ipv4 "$BRAIN" || die "--brain needs the brain computer's IPv4 address (got '$BRAIN'); the bridge's link_up event logs it as \"peer\""
        valid_port "$BRIDGE_PORT" || die "--bridge-port is not a port: '$BRIDGE_PORT'"
        need_nft
        transaction | nft -f - || die "nft refused the table (are you root? sudo sh $0 ...); nothing was changed"
        echo "close-mediad: closed to the LAN: TCP 8443, 8080 and UDP 4210; TCP $BRIDGE_PORT only from $BRAIN"
        if [ "$AT_BOOT" -eq 1 ]; then
            mkdir -p "$(dirname "$RULES_FILE")" "$(dirname "$UNIT_FILE")"
            transaction >"$RULES_FILE"
            cat >"$UNIT_FILE" <<EOF
[Unit]
Description=Duck Mind: close mediad's and netpadd's LAN doors (tools/duck/close-mediad.sh)
Before=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$(command -v nft) -f /etc/duckmind/close-mediad.nft
ExecStop=$(command -v nft) delete table $TABLE

[Install]
WantedBy=multi-user.target
EOF
            systemctl daemon-reload
            systemctl enable "$UNIT_NAME" >/dev/null
            echo "close-mediad: kept at boot ($UNIT_NAME; --undo removes it)"
        else
            echo "close-mediad: in memory only: a reboot or --undo opens them again (--at-boot keeps it)"
        fi
        echo "close-mediad: now check from the computer: python -m duckmind.remotebody check remote://<duck>:$BRIDGE_PORT"
        ;;
    undo)
        need_nft
        if table_in; then
            nft delete table $TABLE || die "nft refused (are you root?)"
            echo "close-mediad: the table is out: Pollen's doors are as shipped"
        else
            echo "close-mediad: the table was not in (or this user may not look: use sudo)"
        fi
        if [ -e "$UNIT_FILE" ] || [ -e "$RULES_FILE" ]; then
            systemctl disable "$UNIT_NAME" >/dev/null 2>&1 || true
            rm -f "$UNIT_FILE" "$RULES_FILE"
            systemctl daemon-reload
            echo "close-mediad: no longer kept at boot"
        fi
        ;;
    status)
        need_nft
        if table_in; then
            echo "close-mediad: the table is in:"
            nft list table $TABLE
        else
            echo "close-mediad: the table is NOT in (or this user may not look: use sudo)"
        fi
        if [ -e "$UNIT_FILE" ]; then
            echo "close-mediad: kept at boot ($UNIT_FILE)"
        else
            echo "close-mediad: not kept at boot"
        fi
        ;;
esac
