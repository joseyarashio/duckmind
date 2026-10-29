#!/bin/sh
# Install a prebuilt Duck Mind wheel on an official-provisioned Linux board.
# Usage: sudo ./deploy/install-board.sh WHEEL SHA256 RELEASE_ID
set -eu
umask 022

if [ "$#" -ne 3 ]; then
    echo "usage: install-board.sh WHEEL SHA256 RELEASE_ID" >&2
    exit 2
fi
wheel=$1
expected=$(printf '%s' "$2" | tr 'A-F' 'a-f')
release=$3
case "$expected" in
    *[!a-fA-F0-9]*|'') echo "SHA256 must be 64 hexadecimal characters" >&2; exit 2 ;;
esac
if [ "${#expected}" -ne 64 ]; then
    echo "SHA256 must be 64 hexadecimal characters" >&2
    exit 2
fi
case "$release" in
    ''|[!a-zA-Z0-9]*|*[!a-zA-Z0-9._-]*) echo "invalid release ID" >&2; exit 2 ;;
esac
if [ "${#release}" -gt 64 ]; then
    echo "invalid release ID" >&2; exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "run as root on the board" >&2
    exit 2
fi
exec 9>/run/duckmind-install.lock
if ! flock -n 9; then
    echo "another Duck Mind install/rollback is running" >&2
    exit 1
fi
if [ ! -f "$wheel" ]; then
    echo "wheel not found: $wheel" >&2
    exit 2
fi
actual=$(sha256sum "$wheel" | cut -d ' ' -f 1)
if [ "$actual" != "$expected" ]; then
    echo "wheel SHA256 mismatch" >&2
    exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ required")'
for required in sudo systemctl install mktemp mv cp ln readlink cmp; do
    command -v "$required" >/dev/null || { echo "missing command: $required" >&2; exit 1; }
done
if ! getent group robot >/dev/null; then
    echo "official robot group is missing; provision Microduck first" >&2
    exit 1
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=/opt/duckmind
unit=/etc/systemd/system/duckmind.service
target="$root/releases/$release"
if [ -e "$target" ] || [ -L "$target" ]; then
    echo "release already exists; refusing to overwrite: $target" >&2
    exit 1
fi
previous=''
if [ -L "$root/current" ]; then
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
elif [ -e "$root/current" ]; then
    echo "current is not a release symlink" >&2; exit 1
fi
if [ -L "$unit" ]; then
    echo "refusing to replace a symlinked or masked Duck Mind unit" >&2; exit 1
fi
if [ -n "$previous" ]; then
    if ! /bin/sh "$script_dir/board-boot-guard.sh"; then
        echo "existing Duck Mind installation failed boot guard; inspect before upgrade" >&2
        exit 1
    fi
elif [ -e "$unit" ]; then
    echo "Duck Mind unit exists without a current release pointer; inspect before install" >&2
    exit 1
fi
if ! getent group duckmind >/dev/null; then
    groupadd --system duckmind
fi
if ! id duckmind >/dev/null 2>&1; then
    useradd --system --gid duckmind --groups robot --home-dir /var/lib/duckmind --shell /usr/sbin/nologin duckmind
fi
if ! id -nG duckmind | tr ' ' '\n' | grep -qx robot; then
    usermod -aG robot duckmind
fi

# venv shebangs include their absolute path, so build at the final location.
# Explicit modes also make root-created releases traversable by User=duckmind.
install -d -m 0755 "$root" "$root/releases" "$target"
install -m 0644 "$script_dir/../README.md" "$target/README.md"
install -m 0644 "$script_dir/duckmind.service" "$target/duckmind.service"
install -m 0755 "$script_dir/board-boot-guard.sh" "$target/board-boot-guard.sh"
wheel_name=$(basename -- "$wheel")
install -m 0644 -- "$wheel" "$target/$wheel_name"
printf '%s\n' "$release" > "$target/RELEASE_ID"
actual=$(sha256sum "$target/$wheel_name" | cut -d ' ' -f 1)
if [ "$actual" != "$expected" ]; then
    echo "copied wheel SHA256 mismatch" >&2; exit 1
fi
python3 -m venv "$target/venv"
"$target/venv/bin/pip" install --no-index --no-deps "$target/$wheel_name"
sudo -u duckmind "$target/venv/bin/duckmindd" --help >/dev/null
sudo -u duckmind "$target/venv/bin/duckmindctl" --help >/dev/null

# Preserve both application and unit before stopping anything. Failures during
# backup, unit publication, start or health restore the prior configuration.
recovery="$target/install-recovery"
install -d -m 0700 "$recovery"
had_unit=0
if [ -f "$unit" ]; then
    cp -p "$unit" "$recovery/duckmind.service"
    had_unit=1
fi
was_active=0
was_enabled=0
if systemctl is-active --quiet duckmind.service; then was_active=1; fi
if systemctl is-enabled --quiet duckmind.service; then was_enabled=1; fi
transaction=0
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
    if [ "$transaction" -eq 1 ] && [ "$committed" -eq 0 ]; then
        set +e
        echo "install failed; restoring previous application and unit" >&2
        if [ -f "$unit" ] && ! systemctl stop duckmind.service; then
            echo "automatic recovery incomplete: cannot stop service; configuration retained" >&2
            exit "$status"
        fi
        restore_ok=1
        if [ "$was_enabled" -eq 0 ] && systemctl is-enabled --quiet duckmind.service; then
            systemctl disable duckmind.service || restore_ok=0
        fi
        if [ -n "$previous" ]; then
            ln -s "$previous" "$root/.current-restore.$$" &&
                mv -Tf "$root/.current-restore.$$" "$root/current" || restore_ok=0
        elif [ -L "$root/current" ]; then
            rm -f -- "$root/current" || restore_ok=0
        fi
        if [ "$had_unit" -eq 1 ]; then
            install -m 0644 "$recovery/duckmind.service" "$unit" || restore_ok=0
        else
            rm -f -- "$unit" || restore_ok=0
        fi
        systemctl daemon-reload || restore_ok=0
        if [ "$was_enabled" -eq 1 ]; then
            systemctl enable duckmind.service || restore_ok=0
        fi
        if [ "$was_active" -eq 1 ] && [ "$restore_ok" -eq 1 ]; then
            systemctl start duckmind.service || restore_ok=0
            if [ -n "$previous" ]; then wait_healthy || restore_ok=0; fi
        fi
        if [ "$restore_ok" -ne 1 ]; then
            echo "automatic recovery incomplete; inspect the unit and journal before play" >&2
        fi
        echo "release files retained; state database was not restored or removed" >&2
    fi
    exit "$status"
}
trap recover 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
transaction=1
if [ "$had_unit" -eq 1 ] || [ "$was_active" -eq 1 ]; then
    systemctl stop duckmind.service
fi
state_db=/var/lib/duckmind/duckmind.sqlite3
if [ -f "$state_db" ]; then
    backup_dir=/var/lib/duckmind/backups
    install -d -m 0700 -o duckmind -g duckmind "$backup_dir"
    backup=$(mktemp "$backup_dir/before-${release}.XXXXXX.sqlite3")
    python3 -c 'import pathlib,sqlite3,sys; src=sqlite3.connect(pathlib.Path(sys.argv[1]).as_uri()+"?mode=ro", uri=True); dst=sqlite3.connect(sys.argv[2]); src.backup(dst); dst.close(); src.close()' "$state_db" "$backup"
    chown duckmind:duckmind "$backup"
    chmod 0600 "$backup"
    echo "pre-upgrade SQLite backup: $backup"
fi

install -m 0644 "$target/duckmind.service" "$unit.next.$$"
mv -Tf "$unit.next.$$" "$unit"
ln -s "releases/$release" "$root/.current-next.$$"
mv -Tf "$root/.current-next.$$" "$root/current"
systemctl daemon-reload
if systemctl enable duckmind.service && systemctl start duckmind.service && wait_healthy; then
    committed=1
    echo "Duck Mind $release installed; passive and healthy. State database preserved."
    exit 0
fi

echo "health gate failed" >&2
exit 1
