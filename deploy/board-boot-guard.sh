#!/bin/sh
# Read-only boot gate: never start cognition with a split application/unit pair.
# systemd ExecStartPre failure prevents ExecStart (systemd.service(5)):
# https://github.com/systemd/systemd/blob/main/man/systemd.service.xml
set -eu

root=/opt/duckmind
unit=/etc/systemd/system/duckmind.service
if [ ! -L "$root/current" ] || [ ! -f "$unit" ] || [ -L "$unit" ]; then
    echo "Duck Mind boot guard: current pointer or installed unit is invalid" >&2
    exit 1
fi
selected=$(readlink "$root/current")
case "$selected" in
    releases/*) release=${selected#releases/} ;;
    *) echo "Duck Mind boot guard: current points outside releases" >&2; exit 1 ;;
esac
case "$release" in
    ''|[!a-zA-Z0-9]*|*[!a-zA-Z0-9._-]*)
        echo "Duck Mind boot guard: invalid release identity" >&2; exit 1 ;;
esac
if [ "${#release}" -gt 64 ]; then
    echo "Duck Mind boot guard: invalid release identity" >&2
    exit 1
fi
target="$root/$selected"
if [ ! -d "$target" ] || [ -L "$target" ] ||
   [ ! -x "$target/venv/bin/duckmindd" ] ||
   [ ! -x "$target/board-boot-guard.sh" ] ||
   [ ! -f "$target/RELEASE_ID" ] || [ -L "$target/RELEASE_ID" ] ||
   [ ! -f "$target/duckmind.service" ] || [ -L "$target/duckmind.service" ]; then
    echo "Duck Mind boot guard: selected release is incomplete" >&2
    exit 1
fi
if ! printf '%s\n' "$release" | cmp -s - "$target/RELEASE_ID"; then
    echo "Duck Mind boot guard: selected release ID does not match" >&2
    exit 1
fi
if ! cmp -s "$unit" "$target/duckmind.service"; then
    echo "Duck Mind boot guard: installed unit differs from selected release" >&2
    exit 1
fi
