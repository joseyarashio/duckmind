#!/bin/sh
# Select an installed application and its matching unit; never restores the DB.
set -eu
umask 022
if [ "$#" -ne 1 ] || [ "$(id -u)" -ne 0 ]; then
    echo "usage: sudo rollback-board.sh RELEASE_ID" >&2
    exit 2
fi
exec 9>/run/duckmind-install.lock
if ! flock -n 9; then
    echo "another Duck Mind install/rollback is running" >&2
    exit 1
fi
release=$1
case "$release" in
    ''|[!a-zA-Z0-9]*|*[!a-zA-Z0-9._-]*) echo "invalid release ID" >&2; exit 2 ;;
esac
if [ "${#release}" -gt 64 ]; then
    echo "invalid release ID" >&2; exit 2
fi
root=/opt/duckmind
unit=/etc/systemd/system/duckmind.service
target="$root/releases/$release"
if [ -L "$target" ]; then
    echo "refusing a symlinked release directory: $release" >&2
    exit 1
fi
if [ ! -x "$target/venv/bin/duckmindd" ] || [ ! -f "$target/duckmind.service" ] ||
   [ ! -x "$target/board-boot-guard.sh" ]; then
    echo "release application, boot guard or archived unit is missing: $release" >&2
    exit 1
fi
if [ ! -f "$target/RELEASE_ID" ] || [ -L "$target/RELEASE_ID" ] ||
   [ "$(cat "$target/RELEASE_ID")" != "$release" ]; then
    echo "release identity record is missing or mismatched: $release" >&2
    exit 1
fi
if [ ! -L "$root/current" ] || [ ! -f "$unit" ] || [ -L "$unit" ]; then
    echo "current must be a release symlink with a regular installed unit" >&2; exit 1
fi
previous=$(readlink "$root/current")
case "$previous" in
    releases/*) previous_id=${previous#releases/} ;;
    *) echo "current must point to a local Duck Mind release" >&2; exit 1 ;;
esac
case "$previous_id" in
    ''|.*|*[!a-zA-Z0-9._-]*) echo "invalid current release pointer" >&2; exit 1 ;;
esac
if [ ! -x "$root/$previous/venv/bin/duckmindd" ]; then
    echo "current release is incomplete" >&2; exit 1
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ! /bin/sh "$script_dir/board-boot-guard.sh"; then
    echo "existing Duck Mind installation failed boot guard; inspect before rollback" >&2
    exit 1
fi
sudo -u duckmind "$target/venv/bin/duckmindd" --help >/dev/null
sudo -u duckmind "$target/venv/bin/duckmindctl" --help >/dev/null
recovery=$(mktemp -d "$root/.rollback-recovery.XXXXXX")
cp -p "$unit" "$recovery/duckmind.service"
was_active=0
if systemctl is-active --quiet duckmind.service; then was_active=1; fi
committed=0

healthy() {
    systemctl is-active --quiet duckmind.service || return 1
    response=$(sudo -u duckmind "$root/current/venv/bin/duckmindctl" status) || return 1
    printf '%s\n' "$response" | python3 -c 'import json,sys; r=json.load(sys.stdin); sys.exit(0 if r.get("ok") is True and r.get("result", {}).get("database_integrity")=="ok" and r.get("result", {}).get("play_enabled") is False else 1)' 2>/dev/null
}

wait_healthy() {
    attempt=0
    while [ "$attempt" -lt 10 ]; do
        if healthy; then return 0; fi
        attempt=$((attempt + 1))
        sleep 1
    done
    return 1
}

recover() {
    status=$?
    trap - 0 HUP INT TERM
    if [ "$committed" -eq 0 ]; then
        set +e
        if ! systemctl stop duckmind.service; then
            echo "automatic recovery incomplete: cannot stop service; configuration retained" >&2
            exit "$status"
        fi
        restore_ok=1
        ln -s "$previous" "$root/.current-restore.$$" &&
            mv -Tf "$root/.current-restore.$$" "$root/current" || restore_ok=0
        install -m 0644 "$recovery/duckmind.service" "$unit" || restore_ok=0
        systemctl daemon-reload || restore_ok=0
        if [ "$was_active" -eq 1 ] && [ "$restore_ok" -eq 1 ]; then
            systemctl start duckmind.service || restore_ok=0
            wait_healthy || restore_ok=0
        fi
        if [ "$restore_ok" -ne 1 ]; then
            echo "automatic recovery incomplete; inspect the unit and journal before play" >&2
        fi
        echo "rollback failed; prior application and unit restored where possible; DB unchanged" >&2
    fi
    exit "$status"
}
trap recover 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl stop duckmind.service
install -m 0644 "$target/duckmind.service" "$unit.next.$$"
mv -Tf "$unit.next.$$" "$unit"
ln -s "releases/$release" "$root/.current-rollback.$$"
mv -Tf "$root/.current-rollback.$$" "$root/current"
systemctl daemon-reload
if systemctl start duckmind.service && wait_healthy; then
    committed=1
    echo "Duck Mind rolled back to $release; state database preserved."
    exit 0
fi
echo "rollback health gate failed" >&2
exit 1
