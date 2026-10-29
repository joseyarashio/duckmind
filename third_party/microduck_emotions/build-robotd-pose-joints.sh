#!/bin/sh
# Build the SIMULATOR-ONLY robotd that answers robot.poseJoints (play dead lies down: docs/design/neural-duck/PLAY-DEAD-2026-10-11.md).
#
#   ./build-robotd-pose-joints.sh UPSTREAM_MICRODUCK_CHECKOUT OUTPUT_DIR
#
# Copies the upstream microduck checkout (daemon 0.15.0, a9ec4b2) to OUTPUT_DIR/microduck-pj (the checkout is NEVER modified),
# applies robotd-poseJoints.patch there and builds robotd. The binary is
# OUTPUT_DIR/microduck-pj/target/aarch64-apple-darwin/debug/robotd (pass it to the simulator as --robotd-bin; the official
# binary stays as it is and is what a real duck runs).
# The patch ports Pollen Robotics' workshop patch 19 (robot.poseJoints, per-servo torque off), Apache-2.0, plus the simulator
# op torque_joints (the sim's body host hangs the named servos: duckmind/bodyhost.py). It is not for a real robot as it is.
set -eu
UP="${1:?upstream microduck checkout}"; OUT="${2:?output dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
[ -e "$OUT/microduck-pj" ] && { echo "$OUT/microduck-pj exists; remove it first"; exit 1; }
cp -Rc "$UP" "$OUT/microduck-pj" 2>/dev/null || cp -R "$UP" "$OUT/microduck-pj"
cd "$OUT/microduck-pj"
git apply --check "$HERE/robotd-poseJoints.patch"
git apply "$HERE/robotd-poseJoints.patch"
cargo build --target aarch64-apple-darwin -p robotd
ls -l target/aarch64-apple-darwin/debug/robotd
