"""``python -m duckmind.setup_cmd``: fetch what duckmind uses but does not ship.

1. The upstream ``microduck`` checkout (Pollen Robotics, Apache-2.0: robotd, the sounds
   crate) at the pinned commit, into ``duckmind.paths.microduck_dir()``.
2. The upstream ``microduck_rl`` checkout at the pinned commit, into
   ``duckmind.paths.microduck_rl_dir()``. It holds the 3D meshes (``.stl``), which are
   CC BY-NC-SA 4.0, not Apache-2.0: setup prints that licence and downloads this
   repository only after an explicit "yes" (or ``--accept-mesh-licence``).
3. The v5 policy set from Hugging Face ``pollen-robotics/microduck-policies`` (Apache-2.0),
   into ``duckmind.paths.v5_policy_dir()``; every file must match its pinned sha256. When
   ``duckmind.paths.policy_dir()`` does not exist yet it becomes a link to that set.

Nothing that is already there is changed: a checkout at another commit, or a policy file
with another sha256, is reported and left alone (exit status 1). Only ``git`` and HTTPS
downloads from GitHub and Hugging Face are used; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from duckmind import paths
from duckmind.policy_check import CONTRACT_COMMIT as MICRODUCK_COMMIT
from duckmind.skilltrain.motion_table import ROULADE_SHA256
from duckmind.skilltrain.rl_contract import V5_HF_URL, V5_SHA256
from duckmind.training import UPSTREAM_COMMIT as MICRODUCK_RL_COMMIT


@dataclass(frozen=True)
class Upstream:
    name: str
    url: str
    commit: str


MICRODUCK = Upstream(
    "microduck", "https://github.com/pollen-robotics/microduck.git", MICRODUCK_COMMIT
)
MICRODUCK_RL = Upstream(
    "microduck_rl", "https://github.com/pollen-robotics/microduck_rl.git", MICRODUCK_RL_COMMIT
)

# The whole v5 release (revision ``v5`` of pollen-robotics/microduck-policies), sha256 of each file.
V5_FILES = {
    "manifest.json": "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94",
    "alpha_walking.onnx": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    **V5_SHA256,  # velstand, alpha_sitstand, alpha_stand
    "roller.onnx": "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd",
    "alpha_ground_pick.onnx": "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
    "roller_crouch.onnx": "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
    "roulade.onnx": ROULADE_SHA256,
    "ball_kick_left.onnx": "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
}

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
Opener = Callable[[str], BinaryIO]
Ask = Callable[[str], str]


def say(en: str, zh: str) -> None:
    print(f"{en}\n  {zh}", flush=True)


def run_git(args: Sequence[str]) -> None:
    subprocess.run(["git", *args], check=True, stdout=subprocess.DEVNULL)


def open_url(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)  # noqa: S310 (fixed https URLs)


def head_commit(folder: Path) -> str | None:
    """The checked-out commit of ``folder``, or None when it is not a git checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


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
            f" (found {found or 'no git checkout'}); left unchanged",
            f"{upstream.name}：{dest} 已經存在，但不是 {upstream.commit[:7]}（看到的是"
            f" {found or '不是 git checkout'}），沒有動它",
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
        say(f"{upstream.name}: download failed ({err})", f"{upstream.name}：下載失敗（{err}）")
        return False
    partial.rename(dest)
    say(f"{upstream.name}: ready in {dest}", f"{upstream.name}：已放在 {dest}")
    return True


def sha256_of(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_policy(name: str, sha256: str, folder: Path, opener: Opener = open_url) -> bool:
    """Download one v5 file into ``folder``; it is kept only when its sha256 matches."""
    dest = folder / name
    if dest.exists():
        if sha256_of(dest) == sha256:
            return True
        say(
            f"policies: {dest} exists with another sha256; left unchanged",
            f"政策：{dest} 已經存在，但 sha256 不同，沒有動它",
        )
        return False
    partial = folder / f".{name}.partial-{os.getpid()}"
    digest = hashlib.sha256()
    try:
        with opener(V5_HF_URL.format(name=name)) as response, partial.open("wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
    except OSError as err:
        partial.unlink(missing_ok=True)
        say(f"policies: downloading {name} failed ({err})", f"政策：{name} 下載失敗（{err}）")
        return False
    if digest.hexdigest() != sha256:
        partial.unlink()
        say(
            f"policies: {name} does not match its pinned sha256 {sha256[:12]}…; not kept",
            f"政策：{name} 和釘住的 sha256 {sha256[:12]}… 不同，沒有留下",
        )
        return False
    partial.replace(dest)
    return True


def fetch_policies(opener: Opener = open_url) -> bool:
    folder = paths.v5_policy_dir()
    folder.mkdir(parents=True, exist_ok=True)
    say(
        f"policies: v5 set from Hugging Face pollen-robotics/microduck-policies into {folder}",
        f"政策：從 Hugging Face pollen-robotics/microduck-policies 取得 v5，放在 {folder}",
    )
    ok = all([fetch_policy(name, sha, folder, opener) for name, sha in V5_FILES.items()])
    if not ok:
        return False
    say(
        f"policies: all {len(V5_FILES)} v5 files match their sha256",
        f"政策：v5 的 {len(V5_FILES)} 個檔案 sha256 全部相符",
    )
    current = paths.policy_dir()
    if current.exists() or current.is_symlink():
        say(
            f"policies: {current} is already there; left unchanged",
            f"政策：{current} 已經存在，沒有動它",
        )
    else:
        current.parent.mkdir(parents=True, exist_ok=True)
        current.symlink_to(os.path.relpath(folder, current.parent), target_is_directory=True)
        say(f"policies: {current} now points at the v5 set", f"政策：{current} 現在指向 v5")
    return True


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
    opener: Opener = open_url,
) -> int:
    """Run the three steps; 0 when everything asked for is in place, else 1."""
    ok = clone(MICRODUCK, paths.microduck_dir(), run)
    rl_dir = paths.microduck_rl_dir()
    if rl_dir.exists() or mesh_consent(accept_mesh_licence, ask):
        ok = clone(MICRODUCK_RL, rl_dir, run) and ok
    if not skip_policies:
        ok = fetch_policies(opener) and ok
    mjcf = paths.microduck_mjcf_dir()
    if mjcf.is_dir():
        say("Start the simulator page:", "啟動模擬器頁面：")
        print(f"  python -m duckmind.simweb --port 8765 --microduck-mjcf {mjcf}", flush=True)
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
    parser.add_argument("--skip-policies", action="store_true", help="do not fetch the v5 policies")
    args = parser.parse_args(argv)
    ask = input if sys.stdin.isatty() else None
    return setup(
        accept_mesh_licence=args.accept_mesh_licence,
        skip_policies=args.skip_policies,
        ask=ask,
    )


if __name__ == "__main__":
    raise SystemExit(main())
