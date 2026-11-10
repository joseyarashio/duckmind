"""``python -m duckmind.setup_cmd``: fetch and build what duckmind uses but does not ship.

What and where is pinned in :mod:`duckmind.pins` (``upstream.env`` and the ``*.sha256`` files), the
same files the public play image is built from:

1. Pollen's ``microduck`` checkout (Apache-2.0: robotd, tofd, the sounds tool) at the
   ``daemon-v0.18.5`` commit, into ``duckmind.paths.microduck_dir()``, with duckmind's poseJoints
   patch applied (``third_party/microduck_emotions``); then robotd, tofd and sounds are built with
   Rust's ``cargo`` (dev profile, ``target/debug``). Without ``cargo`` setup says how to install
   Rust and goes on: the page then runs without robotd.
2. Pollen's ``microduck_rl`` checkout at the pinned commit, into
   ``duckmind.paths.microduck_rl_dir()``. It holds the 3D meshes (``.stl``), which are
   CC BY-NC-SA 4.0, not Apache-2.0: setup prints that licence and downloads this
   repository only after an explicit "yes" (or ``--accept-mesh-licence``).
3. Pollen's policy set v9 from Hugging Face ``pollen-robotics/microduck-policies`` (Apache-2.0),
   into ``duckmind.paths.v9_policy_dir()`` (what the duck walks with), and the v5 set into
   ``duckmind.paths.v5_policy_dir()`` (what the tests and the training tools were measured with;
   when ``duckmind.paths.policy_dir()`` does not exist yet it becomes a link to it).
4. duckmind's seven trained motions from Hugging Face ``joseyarashio/duckmind-motion-skills``
   (Apache-2.0) and the motion table built on v9 (``skills.json``), into
   ``duckmind.paths.motions_dir()``.

Every downloaded file must match its pinned sha256. Nothing that is already there is changed: a
checkout at another commit, a file with another sha256 or another motion table is reported and
left alone (exit status 1). Then setup prints the command that starts the duck with your own
paths. Only ``git``, ``cargo`` and HTTPS downloads from GitHub and Hugging Face are used;
nothing is uploaded.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from duckmind import paths, pins
from duckmind.pins import hub


@dataclass(frozen=True)
class Upstream:
    name: str
    url: str
    commit: str


@dataclass(frozen=True)
class HubSet:
    """One pinned set of files on Hugging Face (a repo, a revision and its pin file)."""

    label: str
    repo: str
    revision: str
    pin_file: Path


PINS = pins.UPSTREAM
MICRODUCK = Upstream("microduck", PINS["MICRODUCK_URL"], PINS["MICRODUCK_COMMIT"])
MICRODUCK_RL = Upstream("microduck_rl", PINS["MICRODUCK_RL_URL"], PINS["MICRODUCK_RL_COMMIT"])
MICRODUCK_TAG = PINS["MICRODUCK_TAG"]
# the poseJoints patch ships with duckmind's source (a clone of the repository), not in the package
PATCH = Path(__file__).resolve().parents[2] / PINS["MICRODUCK_PATCH"]
POLICIES_V9 = HubSet(
    "policies v9", PINS["POLICIES_REPO"], PINS["POLICIES_REV"], pins.POLICIES_V9_FILE
)
POLICIES_V5 = HubSet(
    "policies v5", PINS["POLICIES_REPO"], PINS["POLICIES_V5_REV"], pins.POLICIES_V5_FILE
)
MOTIONS = HubSet("motions", PINS["MOTIONS_REPO"], PINS["MOTIONS_REV"], pins.MOTIONS_FILE)
# (cargo package, binary): robotd, the ToF daemon, and the tool the duck's voice bank is rendered with
BUILDS = (("robotd", "robotd"), ("tof", "tofd"), ("sounds", "sounds"))
PORT = 8765

MESH_NOTICE = """\
------------------------------------------------------------------------------
The Microduck 3D model files (the .stl meshes in microduck_rl) are made by
Pollen Robotics and licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0):
  https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
They are NOT covered by duckmind's Apache-2.0 licence and are not part of
duckmind. In short: non-commercial use only, credit Pollen Robotics, and share
any adaptation under the same licence. The rest of microduck_rl (the code and
the MJCF .xml robot descriptions) is Apache-2.0.
setup downloads the whole microduck_rl repository from GitHub, meshes included.
Without it the page shows a 2D duck and the physics simulator cannot run.

Microduck 的 3D 模型檔（microduck_rl 裡的 .stl 網格）是 Pollen Robotics 做的，
授權是 CC BY-NC-SA 4.0（姓名標示—非商業性—相同方式分享）。它們不在 duckmind
的 Apache-2.0 授權裡，也不是 duckmind 的一部分。簡單說：只能非商業使用、要註明
Pollen Robotics、改作要用同樣的授權分享。microduck_rl 其餘的部分（程式和 MJCF
.xml 機器人描述檔）是 Apache-2.0。setup 會從 GitHub 下載整個 microduck_rl，
包括這些網格。不下載的話，頁面只會顯示 2D 的鴨子，物理模擬也不能跑。
------------------------------------------------------------------------------"""

PROMPT = 'Type "yes" to accept the CC BY-NC-SA 4.0 licence and download / 輸入 yes 表示同意並下載: '

Runner = Callable[[Sequence[str]], None]
Cargo = Callable[[Sequence[str], Path], None]
Ask = Callable[[str], str]


def say(en: str, zh: str) -> None:
    print(f"{en}\n  {zh}", flush=True)


def run_git(args: Sequence[str]) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True)


def run_cargo(args: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(args), check=True, cwd=cwd)


def git_output(folder: Path, args: Sequence[str]) -> str | None:
    """``git -C folder <args>``'s output, or None when git fails there."""
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def head_commit(folder: Path) -> str | None:
    """The checked-out commit of ``folder``, or None when it is not a git checkout."""
    out = git_output(folder, ["rev-parse", "HEAD"])
    return out.strip() if out is not None else None


def is_clean(folder: Path) -> bool:
    """True when ``folder``'s checkout has no changed or new files."""
    return git_output(folder, ["status", "--porcelain"]) == ""


def clone(upstream: Upstream, dest: Path, run: Runner = run_git) -> bool:
    """Check out ``upstream`` at its pinned commit into ``dest``; True when ``dest`` is right."""
    if dest.exists():
        found = head_commit(dest)
        if found == upstream.commit:
            say(
                f"{upstream.name}: already at {upstream.commit[:7]} in {dest}",
                f"{upstream.name}：{dest} 已經是 {upstream.commit[:7]}",
            )
            return True
        say(
            f"{upstream.name}: {dest} exists but is not a checkout of {upstream.commit[:7]}"
            f" (found {found or 'no git checkout'}); left unchanged. To get"
            f" {upstream.commit[:7]}, move that folder aside and run setup again",
            f"{upstream.name}：{dest} 已經存在，但不是 {upstream.commit[:7]}（看到的是"
            f" {found or '不是 git checkout'}），沒有動它。要換成 {upstream.commit[:7]}，"
            "把那個資料夾移開，再跑一次 setup",
        )
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f".{dest.name}.partial-{os.getpid()}")
    shutil.rmtree(partial, ignore_errors=True)
    say(
        f"{upstream.name}: downloading {upstream.url} at {upstream.commit[:7]}",
        f"{upstream.name}：正在下載 {upstream.url}（{upstream.commit[:7]}）",
    )
    try:
        run(["init", "-q", str(partial)])
        run(["-C", str(partial), "remote", "add", "origin", upstream.url])
        run(["-C", str(partial), "fetch", "-q", "--depth", "1", "origin", upstream.commit])
        run(["-C", str(partial), "checkout", "-q", "--detach", "FETCH_HEAD"])
    except (OSError, subprocess.CalledProcessError) as err:
        shutil.rmtree(partial, ignore_errors=True)
        detail = (getattr(err, "stderr", None) or "").strip() or err
        say(
            f"{upstream.name}: download failed ({detail})", f"{upstream.name}：下載失敗（{detail}）"
        )
        return False
    partial.rename(dest)
    say(f"{upstream.name}: ready in {dest}", f"{upstream.name}：已放在 {dest}")
    return True


def patch_microduck(dest: Path, run: Runner = run_git, patch: Path = PATCH) -> bool:
    """Apply duckmind's poseJoints patch to the microduck checkout once; True when it is in."""
    if not patch.is_file():
        say(
            f"robotd: the poseJoints patch {PINS['MICRODUCK_PATCH']} is not here (it comes with"
            " duckmind's source): run setup from a clone of duckmind",
            f"robotd：找不到 poseJoints 修補檔 {PINS['MICRODUCK_PATCH']}（它在 duckmind 的原始碼裡），"
            "請在 duckmind 的 clone 裡跑 setup",
        )
        return False

    def applies(*how: str) -> bool:
        try:
            run(["-C", str(dest), "apply", *how, str(patch)])
        except (OSError, subprocess.CalledProcessError):
            return False
        return True

    if applies("--reverse", "--check"):
        say("robotd: the poseJoints patch is already in", "robotd：poseJoints 修補已經在了")
        return True
    if not is_clean(dest) or not applies("--check"):
        say(
            f"robotd: {dest} has other changes and the poseJoints patch does not apply; left unchanged",
            f"robotd：{dest} 有別的修改，poseJoints 修補放不進去，沒有動它",
        )
        return False
    if not applies():
        say("robotd: applying the poseJoints patch failed", "robotd：poseJoints 修補失敗")
        return False
    say(
        f"robotd: applied the poseJoints patch ({patch.name}) in {dest}",
        f"robotd：已在 {dest} 放進 poseJoints 修補（{patch.name}）",
    )
    return True


def find_cargo() -> str | None:
    """Rust's ``cargo``: on the PATH, else where rustup puts it (a new terminal finds it there)."""
    found = shutil.which("cargo")
    if found:
        return found
    home = os.environ.get("CARGO_HOME")
    cargo = (Path(home) if home else Path.home() / ".cargo") / "bin" / "cargo"
    return str(cargo) if cargo.is_file() else None


def cargo_target() -> str | None:
    """The Rust target robotd is built for on macOS: this Python's own architecture (robotd loads
    the ONNX Runtime library of this Python, and a Rust toolchain installed for the other one
    would build a binary that cannot load it). None elsewhere: cargo's own host."""
    if sys.platform != "darwin":
        return None
    return "aarch64-apple-darwin" if platform.machine() == "arm64" else "x86_64-apple-darwin"


def robotd_bin(checkout: Path) -> Path:
    """Where cargo's dev profile puts robotd (``target[/<target>]/debug/robotd``)."""
    triple = cargo_target()
    return checkout / "target" / (f"{triple}/debug" if triple else "debug") / "robotd"


def build_robotd(checkout: Path, cargo: str | None, run: Cargo = run_cargo) -> bool:
    """Build robotd, tofd and sounds in the checkout (dev profile); True when all three built."""
    if cargo is None:
        say(
            "robotd: Rust's cargo was not found, so robotd (the live physics) was not built."
            " Install Rust with rustup (https://rustup.rs), open a new terminal and run setup"
            " again. Until then the page runs without robotd: no live physics.",
            "robotd：找不到 Rust 的 cargo，所以沒有建 robotd（即時物理）。用 rustup"
            "（https://rustup.rs）裝 Rust，開一個新的終端機，再跑一次 setup。"
            "在那之前，頁面可以開，但沒有 robotd，也就沒有即時物理。",
        )
        return False
    say(
        "robotd: building robotd, tofd and sounds with cargo (the first time takes a few minutes)",
        "robotd：用 cargo 建 robotd、tofd 和 sounds（第一次要幾分鐘）",
    )
    triple = cargo_target()
    for package, binary in BUILDS:
        args = [cargo, "build", "--locked", "--profile", "dev", "-p", package, "--bin", binary]
        if triple:
            args += ["--target", triple]
        try:
            run(args, checkout)
        except (OSError, subprocess.CalledProcessError) as err:
            say(f"robotd: building {binary} failed ({err})", f"robotd：建 {binary} 失敗（{err}）")
            if triple:
                say(
                    f"If cargo said the target may not be installed: rustup target add {triple}",
                    f"如果 cargo 說沒有裝這個 target：rustup target add {triple}",
                )
            return False
    if not robotd_bin(checkout).is_file():
        say(
            f"robotd: cargo finished but {robotd_bin(checkout)} is not there",
            f"robotd：cargo 跑完了，但找不到 {robotd_bin(checkout)}",
        )
        return False
    say(f"robotd: ready at {robotd_bin(checkout)}", f"robotd：已經建好，在 {robotd_bin(checkout)}")
    return True


def fetch_set(hubset: HubSet, folder: Path, fetch: hub.Get = hub.get) -> bool:
    """Every pinned file of ``hubset`` into ``folder``; True when all of them are there and right."""
    wanted = hub.read_pins(hubset.pin_file)
    say(
        f"{hubset.label}: {len(wanted)} files from Hugging Face {hubset.repo}"
        f" (revision {hubset.revision[:12]}) into {folder}",
        f"{hubset.label}：從 Hugging Face {hubset.repo}（版本 {hubset.revision[:12]}）取得"
        f" {len(wanted)} 個檔案，放在 {folder}",
    )
    _kept, bad = hub.fetch_all(hubset.repo, hubset.revision, wanted, folder, fetch)
    if bad:
        for line in bad:
            print(f"  {line}", flush=True)
        say(
            f"{hubset.label}: {len(bad)} of {len(wanted)} files are not right (above); a download"
            " that does not match its pinned sha256 is not kept",
            f"{hubset.label}：{len(wanted)} 個檔案裡有 {len(bad)} 個不對（見上面）；"
            "和釘住的 sha256 不同的下載不會留下",
        )
        return False
    say(
        f"{hubset.label}: all {len(wanted)} files match their pinned sha256",
        f"{hubset.label}：{len(wanted)} 個檔案的 sha256 全部相符",
    )
    return True


def link_current(target: Path) -> None:
    """Point ``duckmind.paths.policy_dir()`` at ``target`` when it does not exist yet."""
    current = paths.policy_dir()
    if current.exists() or current.is_symlink():
        say(
            f"policies: {current} is already there; left unchanged",
            f"政策：{current} 已經存在，沒有動它",
        )
        return
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(os.path.relpath(target, current.parent), target_is_directory=True)
    say(f"policies: {current} now points at {target}", f"政策：{current} 現在指向 {target}")


def motion_table_ok(out: Path) -> bool:
    from duckmind.skilltrain import motion_table as MT

    table = out / MT.DEFAULT_TABLE
    return table.is_file() and not MT.check(table)


def install_motions(fetch: hub.Get = hub.get) -> bool:
    """duckmind's seven motions and their table on v9 into ``duckmind.paths.motions_dir()``."""
    from duckmind.pins import motions

    out = paths.motions_dir()
    if out.exists():
        if motion_table_ok(out):
            say(f"motions: the table in {out} is there", f"動作：{out} 的動作表已經在了")
            return True
        say(
            f"motions: {out} exists but is not a motion table whose files all match; left unchanged",
            f"動作：{out} 已經存在，但不是檔案都對的動作表，沒有動它",
        )
        return False
    staging = out.with_name(f".motions-fetched-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if not fetch_set(MOTIONS, staging, fetch):
            return False
        table, problems = motions.install(staging, paths.v9_policy_dir(), out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if table is None:
        shutil.rmtree(out, ignore_errors=True)
        say(
            f"motions: the table was not built ({'; '.join(problems)})",
            f"動作：動作表沒有建好（{'; '.join(problems)}）",
        )
        return False
    say(f"motions: the table is in {table}", f"動作：動作表在 {table}")
    return True


def start_command(port: int = PORT) -> tuple[str, bool]:
    """The command that starts the duck with this machine's files, and whether it has live physics.

    Live physics through robotd on v9 (and the trained motions when their table is right) when
    robotd, microduck_rl and v9 are here; else the page alone, with the 3D robot model when it is
    here (a 2D duck without it). The Python is the one setup runs in (it has duckmind)."""
    q = shlex.quote
    python = f"{q(sys.executable)} -m duckmind.simweb --port {port}"
    md, rl, v9 = paths.microduck_dir(), paths.microduck_rl_dir(), paths.v9_policy_dir()
    live = (
        robotd_bin(md).is_file()
        and (rl / "src/mjlab_microduck/sim/body_server.py").is_file()
        and (v9 / "velstand.onnx").is_file()
    )
    if live:
        from duckmind.skilltrain import motion_table as MT
        from duckmind.skilltrain import table as T

        env = ""
        if motion_table_ok(paths.motions_dir()):
            env = f"{T.SKILLS_ENV}={q(str(paths.motions_dir() / MT.DEFAULT_TABLE))} \\\n    "
        return (
            f"{env}{python} --live --brain neuro \\\n"
            f"    --microduck {q(str(md))} --microduck-rl {q(str(rl))} \\\n"
            f"    --policies {q(str(v9))} --robotd-bin {q(str(robotd_bin(md)))}",
            True,
        )
    mjcf = paths.microduck_mjcf_dir()
    if mjcf.is_dir():
        return f"{python} --microduck-mjcf {q(str(mjcf))}", False
    return python, False


def mesh_consent(accepted: bool, ask: Ask | None) -> bool:
    """Print the mesh licence; True only after an explicit yes."""
    print(MESH_NOTICE, flush=True)
    if accepted:
        say("Accepted with --accept-mesh-licence.", "已用 --accept-mesh-licence 同意。")
        return True
    if ask is None:
        say(
            "Not a terminal: microduck_rl skipped. Run again with --accept-mesh-licence to accept.",
            "不是互動終端機：略過 microduck_rl。要同意的話，加上 --accept-mesh-licence 再跑一次。",
        )
        return False
    try:
        answer = ask(PROMPT)
    except EOFError:
        answer = ""
    if answer.strip().lower() == "yes":
        return True
    say("Not accepted: microduck_rl skipped.", "沒有同意：略過 microduck_rl。")
    return False


def setup(
    *,
    accept_mesh_licence: bool = False,
    skip_policies: bool = False,
    ask: Ask | None = None,
    run: Runner = run_git,
    fetch: hub.Get | None = None,
    build: Cargo | None = None,
) -> int:
    """Run every step; 0 when everything asked for is in place, else 1. ``fetch`` downloads
    (default :func:`duckmind.pins.hub.get`), ``build`` runs cargo (default :func:`run_cargo`)."""
    fetch = fetch or hub.get
    md = paths.microduck_dir()
    microduck = clone(MICRODUCK, md, run)
    # (the licence question comes before the robotd build, which takes minutes)
    rl_dir = paths.microduck_rl_dir()
    rl = True
    if rl_dir.exists() or mesh_consent(accept_mesh_licence, ask):
        rl = clone(MICRODUCK_RL, rl_dir, run)
    if microduck:
        robotd = patch_microduck(md, run) and build_robotd(md, find_cargo(), build or run_cargo)
    else:
        say("robotd: not built (microduck is not ready)", "robotd：沒有建（microduck 沒有準備好）")
        robotd = False
    ok = microduck and rl and robotd
    if not skip_policies:
        v9 = fetch_set(POLICIES_V9, paths.v9_policy_dir(), fetch)
        v5 = fetch_set(POLICIES_V5, paths.v5_policy_dir(), fetch)
        if v5:
            link_current(paths.v5_policy_dir())
        ok = ok and v9 and v5
        if v9:
            ok = install_motions(fetch) and ok
    command, live = start_command()
    if live:
        say(
            "Start the duck (live physics through robotd, Pollen's v9 walk, duckmind's motions):",
            "啟動小鴨（透過 robotd 的即時物理、Pollen 的 v9 走路、duckmind 的動作）：",
        )
    else:
        say(
            "Start the simulator page (no live physics: robotd, microduck_rl or the v9 policies"
            " are missing):",
            "啟動模擬器頁面（沒有即時物理：缺 robotd、microduck_rl 或 v9 政策）：",
        )
    print(f"  {command}\n  # open http://127.0.0.1:{PORT}/", flush=True)
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.setup_cmd", description=__doc__.split("\n\n", 1)[0]
    )
    parser.add_argument(
        "--accept-mesh-licence",
        action="store_true",
        help="accept the CC BY-NC-SA 4.0 licence of the 3D meshes without asking",
    )
    parser.add_argument(
        "--skip-policies",
        action="store_true",
        help="do not fetch the policies and duckmind's motions",
    )
    args = parser.parse_args(argv)
    ask = input if sys.stdin.isatty() else None
    return setup(
        accept_mesh_licence=args.accept_mesh_licence,
        skip_policies=args.skip_policies,
        ask=ask,
    )


if __name__ == "__main__":
    raise SystemExit(main())
