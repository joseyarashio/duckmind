#!/bin/sh
# Build the SIMULATOR-ONLY robotd that answers robot.poseJoints (play dead lies down: docs/design/neural-duck/PLAY-DEAD-2026-10-11.md).
#
#   ./build-robotd-pose-joints.sh UPSTREAM_MICRODUCK_CHECKOUT OUTPUT_DIR
#
# Copies the upstream microduck checkout to OUTPUT_DIR/microduck-pj (the checkout is NEVER modified), applies the patch
# for the checkout's daemon version there and builds robotd:
#   daemon 0.15.0 (a9ec4b2)          robotd-poseJoints.patch
#   daemon 0.18.5 (54e0789, the tag) robotd-poseJoints-0.18.5.patch (the same change rebased: UPSTREAM-0185-2026-11-05)
#   daemon 0.18.8 (4a763dd, the tag) robotd-poseJoints-0.18.8.patch (the same change again: only duck-ipc-proto's count of
#                                    Call variants differs, 78 -> 79, since API 50 dropped robot.remoteSessionActive)
# The binary is OUTPUT_DIR/microduck-pj/target/aarch64-apple-darwin/debug/robotd (pass it to the simulator as --robotd-bin;
# the official binary stays as it is and is what a real duck runs).
# The patch ports Pollen Robotics' workshop patch 19 (robot.poseJoints, per-servo torque off), Apache-2.0, plus the simulator
# op torque_joints (the sim's body host hangs the named servos: duckmind/bodyhost.py). It is not for a real robot as it is.
set -eu
UP="${1:?upstream microduck checkout}"; OUT="${2:?output dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="$(sed -n 's/^version = "\([0-9][0-9.]*\)"$/\1/p' "$UP/Cargo.toml" | head -1)"
case "$VERSION" in
  0.15.0) PATCH="$HERE/robotd-poseJoints.patch" ;;
  0.18.5) PATCH="$HERE/robotd-poseJoints-0.18.5.patch" ;;
  0.18.8) PATCH="$HERE/robotd-poseJoints-0.18.8.patch" ;;
  *) echo "no poseJoints patch for daemon version '$VERSION' (have 0.15.0, 0.18.5, 0.18.8)"; exit 1 ;;
esac
mkdir -p "$OUT"
[ -e "$OUT/microduck-pj" ] && { echo "$OUT/microduck-pj exists; remove it first"; exit 1; }
cp -Rc "$UP" "$OUT/microduck-pj" 2>/dev/null || cp -R "$UP" "$OUT/microduck-pj"
cd "$OUT/microduck-pj"
git apply --check "$PATCH"
git apply "$PATCH"
cargo build --target aarch64-apple-darwin -p robotd
ls -l target/aarch64-apple-darwin/debug/robotd
