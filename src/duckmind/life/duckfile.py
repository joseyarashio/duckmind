"""One duck, one file: a duck's life packed to move it to another computer (docs/duck-file.md).

A **duck file** (``<name>-<date>.duck``) is a zip that holds what makes the duck itself:

    manifest.json        what is inside: the format and its version, the duck (id, name, seed, birth,
                         growth day, stage, branch, counts), the build that wrote it, and the SHA-256
                         and size of every other member
    duck/life.json       who: name, seed, birth, nature, the hardware it hatched with (as on disk)
    duck/core.json       Duck Core's save: its whole growth record, every input, byte for byte
    duck/neural.json     what the neural duck learnt: the people it knows (names and looks), the cues,
                         the tricks and how well it does each, its habits and drives. **Without the
                         home**: the places, the things, the nest spot and the lasting map stay behind
    voice/<key>/...      the voice library its lineage sounds like now (the learned calls, a folder of
                         wav clips and their plan), so it sounds the same on the other computer

What stays behind (owner's spec: a new home or a new body is learnt again, which is natural): the
home map and the furniture (the snapshot's places, things, nest spot and map), the eye's floor model
(``eye/``: what one home's floor looks like), the layouts the editor applied (``layouts/``), the
snapshot before the newest one, the bank of the duck's species voice (rendered again from its seed).

Import never overwrites a duck silently: it makes a **new** life folder. A file of a duck that this
computer already has is refused unless the caller asks to replace it; the folder it replaces is
first moved aside as a backup. The duck the server lives is never replaced. Every member is checked
against the manifest; a file from a newer build is refused; an older duck-file or snapshot version
is read through its migrations. Duck Core's save must replay on this build (a duck grown on another
engine cannot be raised on by this one).

    python -m duckmind.life.duckfile export LIFE_FOLDER [-o FILE.duck]
    python -m duckmind.life.duckfile inspect FILE.duck
    python -m duckmind.life.duckfile import FILE.duck [--into FOLDER] [--replace]

The file holds people's names and looks: it is private (written 0600) and is never sent anywhere by
this program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.i18n import msg
from duckmind.life import snapshot as snap
from duckmind.life.clock import TPE
from duckmind.life.record import LIFE_FORMAT, LIFE_VERSION, LifeError, LifeRecord, write_private

FORMAT = "duckmind.duckfile"
VERSION = 1
SUFFIX = ".duck"
MANIFEST = "manifest.json"
LIFE = "duck/life.json"
CORE = "duck/core.json"
NEURAL = "duck/neural.json"
VOICE = "voice/"
MAX_FILE_BYTES = (
    64 * 1024 * 1024
)  # a duck file is ~1-2 MB (core.json ~0.5 MB, a voice library ~1 MB)
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 4000
# a fixed time on every member but the manifest, so the same duck packs to the same bytes
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
# what of the neural snapshot is the home (the body's odometry frame of one home): left behind
HOME_LEARNER_KEYS = ("places", "things", "dock")


class DuckFileError(ValueError):
    """The duck file cannot be made, read or put in (said in plain words; nothing was written)."""


# the duck file's own migrations: version → function(manifest, members) -> (manifest, members) of the
# next version. Version 1 is the first; the snapshot inside has its own (snapshot.MIGRATIONS).
Members = dict[str, bytes]
MIGRATIONS: dict[int, Callable[[dict[str, Any], Members], tuple[dict[str, Any], Members]]] = {}


# ── who the duck is ──────────────────────────────────────────────────────────────


def duck_id(seed: int, birth: datetime) -> str:
    """The duck's identity across computers: its seed and its birth (a duck is hatched once).
    The same on every copy of it; two ducks of the same seed hatched at different moments differ."""
    text = f"{int(seed)}|{birth.astimezone(TPE).isoformat()}"
    return "duck-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def record_id(record: LifeRecord) -> str:
    return duck_id(record.seed, record.birth)


def lives_dir() -> Path:
    """Where this computer keeps its ducks' folders (``<DUCKMIND_HOME>/lives``)."""
    return paths.duckmind_home() / "lives"


def lineage_root() -> Path:
    """Where the voice libraries live (as the server keeps them: ``$DUCKMIND_LINEAGE_ROOT``, else
    beside the voice banks in the data folder)."""
    raw = os.environ.get("DUCKMIND_LINEAGE_ROOT")
    return Path(raw).expanduser() if raw else paths.data_dir() / "voices" / "lineage"


def bank_root() -> Path:
    return paths.data_dir() / "voices"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _app_version() -> str:
    try:
        from importlib.metadata import version

        return version("duckmind")
    except Exception:  # (a checkout run from src/ without an install)
        return "unknown"


def _replay(core_bytes: bytes) -> Any:
    """Duck Core's save replayed on this build (DuckFileError: it does not replay here)."""
    import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
    from duckmind.duckcore import api

    try:
        saved = json.loads(core_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise DuckFileError(msg("web.duckfile.bad_part", name=CORE)) from None
    try:
        return api.restore(saved)
    except (api.SaveContentMismatch, api.SaveBridgeMismatch):
        raise DuckFileError(msg("web.duckfile.engine")) from None
    except (ValueError, KeyError, TypeError):
        raise DuckFileError(msg("web.duckfile.bad_part", name=CORE)) from None


def home_free(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The snapshot at this build's version without the home (places, things, nest spot, lasting
    map, the home it was saved in), and what was left behind, in counts."""
    out = copy.deepcopy(snap.upgrade(snapshot))
    learner = dict(out.get("learner") or {})
    saved_map = out.pop("map", None) or {}
    left = {
        "places": len(learner.get("places") or ()),
        "things": len(learner.get("things") or ()),
        "nest_spot": learner.get("dock") is not None,
        "map_cells": len(saved_map.get("cells") or ()),
    }
    learner["places"], learner["things"], learner["dock"] = [], [], None
    out["learner"] = learner
    out["context"] = {}  # (no home: the next boot's home is never "the same home")
    return out, left


def _summary(life: dict[str, Any], session: Any, neural: dict[str, Any] | None) -> dict[str, Any]:
    """The duck in numbers (the manifest's ``duck``; no people's names or looks)."""
    core = session.core
    learner = (neural or {}).get("learner") or {}
    people = learner.get("people") or []
    skills = ((learner.get("tricks") or {}).get("skills")) or {}
    regrow = (neural or {}).get("regrow") or {}
    birth = datetime.fromisoformat(str(life["birth"]))
    return {
        "id": duck_id(int(life["seed"]), birth),
        "name": str(life["name"]),
        "seed": int(life["seed"]),
        "birth": birth.astimezone(TPE).isoformat(),
        "nature": str(life.get("nature", "")),
        "growth_day": int(core.s.gd),
        "stage": core.stage().stage_id,
        "branch": core.s.branch,
        "body_branch": regrow.get("branch"),
        "core_head": core.log.head,
        "core_events": len(core.log.events),
        "people": len(people),
        "named_people": sum(1 for p in people if p.get("name")),
        "tricks": {t: round(float(s.get("p", 0.0)), 4) for t, s in sorted(skills.items())},
    }


# ── the voice library ───────────────────────────────────────────────────────────


def library_for(session: Any, seed: int, *, root: Path | None = None) -> Path | None:
    """The complete voice library this duck's lineage sounds like now, on this computer (None: the
    bank or the library is not here; the other computer then builds it again from the lineage)."""
    from duckmind.life import voice_lineage as VL
    from duckmind.simweb import lineage_voice as LV
    from duckmind.simweb import voice

    vseed = voice.voice_seed(int(seed))
    bank = bank_root() / f"v{voice.BANK_VERSION}" / f"seed{vseed}"
    if not bank.is_dir():
        return None
    plan = VL.plan_for(VL.view_of(session.core.s), vseed, LV.bank_letters(bank))
    folder = (root or lineage_root()) / f"seed{vseed}" / plan.key
    return folder if _library_complete(folder) else None


def _library_complete(folder: Path) -> bool:
    from duckmind.simweb import lineage_voice as LV

    try:
        key = (folder / LV.MARKER).read_text().strip()
        data = json.loads((folder / LV.PLAN_FILE).read_text())
        return key == folder.name and all(
            (folder / name / file).is_file()
            for name, files in data["lengths"].items()
            for file in files
        )
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def _library_members(folder: Path) -> Members:
    out: Members = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            out[f"{VOICE}{folder.name}/{path.relative_to(folder).as_posix()}"] = path.read_bytes()
    return out


def _wav_hash(files: dict[str, bytes]) -> str:
    """lineage_voice.small_hash of a library, from its members (names relative to the library)."""
    digest = hashlib.sha256()
    for name in sorted(n for n in files if n.endswith(".wav")):
        digest.update(name.encode())
        digest.update(files[name])
    return digest.hexdigest()[:16]


# ── export ──────────────────────────────────────────────────────────────────────


def pack(
    folder: Path,
    *,
    library: Path | None | bool = True,
    now: datetime | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """The duck file of the life in ``folder``, and its manifest. ``library``: the voice library to
    put in (a folder), True to find the one its lineage sounds like now, False/None for none."""
    from duckmind.duckcore.api import content_version
    from duckmind.life import voice_lineage as VL
    from duckmind.simweb import voice

    folder = Path(folder).expanduser()
    try:
        record = LifeRecord.load(folder)
    except LifeError:
        raise DuckFileError(msg("web.duckfile.no_duck", folder=str(folder))) from None
    life_bytes = record.path.read_bytes()
    try:
        core_bytes = record.core_path.read_bytes()
    except FileNotFoundError:
        raise DuckFileError(msg("web.duckfile.no_duck", folder=str(folder))) from None
    session = _replay(core_bytes)
    neural: dict[str, Any] | None = None
    left: dict[str, Any] = {"places": 0, "things": 0, "nest_spot": False, "map_cells": 0}
    try:
        saved = snap.read(record.neural_path)
    except snap.SnapshotUnusable:
        raise DuckFileError(msg("web.duckfile.bad_part", name=NEURAL)) from None
    if saved is not None:
        try:
            neural, left = home_free(saved)
        except snap.SnapshotUnusable:
            raise DuckFileError(msg("web.duckfile.bad_part", name=NEURAL)) from None
    left["eye"] = record.eye_dir.is_dir()
    left["layouts"] = (folder / "layouts").is_dir()

    members: Members = {LIFE: life_bytes, CORE: core_bytes}
    if neural is not None:
        members[NEURAL] = json.dumps(neural, ensure_ascii=False, separators=(",", ":")).encode()
    lib = library_for(session, record.seed) if library is True else (library or None)
    vseed = voice.voice_seed(record.seed)
    voice_part: dict[str, Any] = {"voice_seed": vseed, "embedded": False, "key": None, "hash": None}
    if isinstance(lib, Path) and _library_complete(lib):
        files = _library_members(lib)
        members.update(files)
        prefix = f"{VOICE}{lib.name}/"
        voice_part.update(
            embedded=True,
            key=lib.name,
            hash=_wav_hash({n[len(prefix) :]: b for n, b in files.items()}),
            files=len(files),
        )

    life = json.loads(life_bytes.decode("utf-8"))
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "created": (now or datetime.now(TPE)).astimezone(TPE).isoformat(),
        "app": {
            "duckmind": _app_version(),
            "life_version": LIFE_VERSION,
            "snapshot_version": snap.SNAPSHOT_VERSION,
            "engine": content_version(),
            "voice_library_version": VL.LIBRARY_VERSION,
        },
        "duck": _summary(life, session, neural),
        "voice": voice_part,
        "left_behind": left,
        "parts": {
            name: {"sha256": _sha(data), "bytes": len(data)}
            for name, data in sorted(members.items())
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(MANIFEST, date_time=ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=1).encode())
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buffer.getvalue(), manifest


def file_name(manifest: dict[str, Any]) -> str:
    """``<name>-<date>.duck`` (the name kept as the family wrote it; only path characters go)."""
    duck = manifest["duck"]
    day = str(manifest.get("created", ""))[:10] or time.strftime("%Y-%m-%d")
    return f"{_safe(duck['name'])}-{day}{SUFFIX}"


def export(folder: Path, out: Path | None = None, **kw: Any) -> tuple[Path, dict[str, Any]]:
    """Write the duck file of ``folder`` (default: ``<name>-<date>.duck`` here). Never over a file."""
    data, manifest = pack(folder, **kw)
    target = Path(out).expanduser() if out is not None else Path.cwd() / file_name(manifest)
    if target.exists():
        raise DuckFileError(msg("web.duckfile.dest_taken", folder=str(target)))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return target, manifest


# ── reading a duck file ───────────────────────────────────────────────────────────


@dataclass
class DuckFile:
    """A duck file read and checked: every member matched its checksum, the life record and the
    snapshot are of this build's versions (migrated when older), Duck Core's save replays here."""

    manifest: dict[str, Any]
    members: Members
    life: dict[str, Any]
    neural: dict[str, Any] | None
    session: Any = field(repr=False)

    @property
    def id(self) -> str:
        return str(self.manifest["duck"]["id"])

    @property
    def name(self) -> str:
        return str(self.life["name"])

    def library(self) -> dict[str, bytes]:
        """The voice library's files, by their names inside the library folder."""
        key = (self.manifest.get("voice") or {}).get("key")
        if not key:
            return {}
        prefix = f"{VOICE}{key}/"
        return {n[len(prefix) :]: b for n, b in self.members.items() if n.startswith(prefix)}


def _member_ok(name: str) -> bool:
    if name in (MANIFEST, LIFE, CORE, NEURAL):
        return True
    parts = name.split("/")
    return (
        name.startswith(VOICE)
        and len(parts) >= 3
        and all(p and p not in (".", "..") for p in parts)
        and "\\" not in name
    )


def read(data: bytes) -> DuckFile:
    """Open and check a duck file (DuckFileError: refused, in plain words)."""
    if len(data) > MAX_FILE_BYTES:
        raise DuckFileError(msg("web.duckfile.too_big"))
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        raise DuckFileError(msg("web.duckfile.not_duckfile")) from None
    if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
        raise DuckFileError(msg("web.duckfile.too_big"))
    names = [i.filename for i in infos]
    if MANIFEST not in names:
        raise DuckFileError(msg("web.duckfile.not_duckfile"))
    for info in infos:
        if info.is_dir():
            continue
        if not _member_ok(info.filename) or names.count(info.filename) > 1:
            raise DuckFileError(msg("web.duckfile.bad_member", name=info.filename))
    try:
        raw = {i.filename: zf.read(i) for i in infos if not i.is_dir()}
    except (zipfile.BadZipFile, OSError, RuntimeError, EOFError, ValueError):
        raise DuckFileError(msg("web.duckfile.not_duckfile")) from None
    try:
        manifest = json.loads(raw.pop(MANIFEST).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise DuckFileError(msg("web.duckfile.not_duckfile")) from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise DuckFileError(msg("web.duckfile.not_duckfile"))
    try:
        version = int(manifest.get("version", 0))
    except (TypeError, ValueError):
        raise DuckFileError(msg("web.duckfile.not_duckfile")) from None
    if version > VERSION:
        raise DuckFileError(msg("web.duckfile.newer", found=version, max=VERSION))
    # every member is the one the manifest lists, byte for byte (checked before any migration)
    parts = manifest.get("parts")
    if not isinstance(parts, dict):
        raise DuckFileError(msg("web.duckfile.not_duckfile"))
    for name in sorted(set(parts) | set(raw)):
        if name not in raw:
            raise DuckFileError(msg("web.duckfile.missing", name=name))
        if name not in parts:
            raise DuckFileError(msg("web.duckfile.bad_member", name=name))
        want = parts[name]
        if (
            not isinstance(want, dict)
            or want.get("sha256") != _sha(raw[name])
            or want.get("bytes") != len(raw[name])
        ):
            raise DuckFileError(msg("web.duckfile.checksum", name=name))
    while version < VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise DuckFileError(msg("web.duckfile.no_migration", found=version))
        manifest, raw = step(manifest, raw)
        version = int(manifest["version"])
    for name in (LIFE, CORE):
        if name not in raw:
            raise DuckFileError(msg("web.duckfile.missing", name=name))
    # the life record (this build reads up to LIFE_VERSION)
    try:
        life = json.loads(raw[LIFE].decode("utf-8"))
        if not isinstance(life, dict) or life.get("format") != LIFE_FORMAT:
            raise ValueError("not a life record")
        birth = datetime.fromisoformat(str(life["birth"]))
        seed = int(life["seed"])
        str(life["name"])
    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
        raise DuckFileError(msg("web.duckfile.bad_part", name=LIFE)) from None
    if int(life.get("version", 0)) > LIFE_VERSION:
        raise DuckFileError(msg("web.duckfile.newer", found=int(life["version"]), max=LIFE_VERSION))
    # the snapshot (older versions migrated; a newer one is refused)
    neural = None
    if NEURAL in raw:
        try:
            loaded = json.loads(raw[NEURAL].decode("utf-8"))
            found = int(loaded.get("version", 0)) if isinstance(loaded, dict) else 0
        except (UnicodeDecodeError, ValueError, TypeError):
            raise DuckFileError(msg("web.duckfile.bad_part", name=NEURAL)) from None
        if found > snap.SNAPSHOT_VERSION:
            raise DuckFileError(msg("web.duckfile.newer", found=found, max=snap.SNAPSHOT_VERSION))
        try:
            neural = snap.upgrade(loaded)
        except snap.SnapshotUnusable:
            raise DuckFileError(msg("web.duckfile.bad_part", name=NEURAL)) from None
        if int((neural.get("duck") or {}).get("seed", -1)) != seed:
            raise DuckFileError(msg("web.duckfile.mismatch"))
    session = _replay(raw[CORE])
    if int(session.core.s.seed) != seed:
        raise DuckFileError(msg("web.duckfile.mismatch"))
    duck = manifest.get("duck") or {}
    if duck.get("id") != duck_id(seed, birth):
        raise DuckFileError(msg("web.duckfile.mismatch"))
    return DuckFile(manifest=manifest, members=raw, life=life, neural=neural, session=session)


def read_path(path: Path) -> DuckFile:
    path = Path(path).expanduser()
    try:
        size = path.stat().st_size
    except OSError:
        raise DuckFileError(msg("web.duckfile.not_duckfile")) from None
    if size > MAX_FILE_BYTES:
        raise DuckFileError(msg("web.duckfile.too_big"))
    return read(path.read_bytes())


# ── import ──────────────────────────────────────────────────────────────────────


def _safe(name: str) -> str:
    """A folder or file name from a duck's name: letters and digits of any script stay."""
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-.")
    return out[:40] or "duck"


def find_same(duck: str, where: Path | None = None) -> list[Path]:
    """The life folders under ``where`` (default: this computer's lives) that hold this duck."""
    root = where or lives_dir()
    found = []
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []
    for child in children:
        try:
            if record_id(LifeRecord.load(child)) == duck:
                found.append(child)
        except (LifeError, OSError, ValueError, KeyError):
            continue
    return found


def summary(
    df: DuckFile, *, where: Path | None = None, running: Path | None = None
) -> dict[str, Any]:
    """What the page (or the CLI) shows of a checked file before it is put in: the duck in numbers,
    what stays behind, and whether this computer has the same duck already."""
    same = find_same(df.id, where)
    running_same = running is not None and _is_duck(running, df.id)
    if running_same and running is not None and running not in same:
        same.append(running)
    return {
        "format": df.manifest["format"],
        "version": df.manifest["version"],
        "created": df.manifest.get("created"),
        "app": df.manifest.get("app"),
        "duck": df.manifest["duck"],
        "voice": {k: v for k, v in (df.manifest.get("voice") or {}).items()},
        "left_behind": df.manifest.get("left_behind"),
        "same_here": [str(p) for p in same],
        "running_here": running_same,
    }


def _is_duck(folder: Path, duck: str) -> bool:
    try:
        return record_id(LifeRecord.load(folder)) == duck
    except (LifeError, OSError, ValueError, KeyError):
        return False


def _new_folder(root: Path, name: str, duck: str) -> Path:
    base = f"{_safe(name)}-{duck[len('duck-') :][:8]}"
    target = root / base
    n = 2
    while target.exists():
        target = root / f"{base}-{n}"
        n += 1
    return target


def install(
    data: bytes | DuckFile,
    *,
    into: Path | None = None,
    where: Path | None = None,
    replace: bool = False,
    running: Path | None = None,
    voices: Path | None = None,
    stamp: str | None = None,
) -> dict[str, Any]:
    """Put the duck of a duck file on this computer as a life folder; returns what was done.

    ``where``: the lives folder (default this computer's). ``into``: the folder to make instead of
    a new one there (it must not exist). A duck this computer has already is refused unless
    ``replace``: the folder it replaces is moved aside first (``<folder>.replaced-<stamp>``).
    ``running``: the life folder of the server that asks (never replaced). ``voices``: the root of
    the voice libraries (default as the server keeps them)."""
    df = data if isinstance(data, DuckFile) else read(data)
    root = Path(where).expanduser() if where is not None else lives_dir()
    same = find_same(df.id, root)
    if running is not None and _is_duck(running, df.id):
        raise DuckFileError(msg("web.duckfile.running", name=df.name))
    if into is not None:
        into = Path(into).expanduser()
        if into.exists():
            raise DuckFileError(msg("web.duckfile.dest_taken", folder=str(into)))
        same += [p for p in find_same(df.id, into.parent) if p not in same]
    if same and not replace:
        raise DuckFileError(msg("web.duckfile.exists", name=df.name, folder=str(same[0])))
    backup = None
    if same and replace:
        target = into or same[0]
        old = same[0]
        backup = old.with_name(f"{old.name}.replaced-{stamp or time.strftime('%Y%m%d-%H%M%S')}")
        if backup.exists():
            raise DuckFileError(msg("web.duckfile.dest_taken", folder=str(backup)))
    else:
        target = into or _new_folder(root, df.name, df.id)

    # everything is written in a staging folder beside the target, then put in place in one rename
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        write_private(staging / "life.json", df.members[LIFE].decode("utf-8"))
        write_private(staging / "core.json", df.members[CORE].decode("utf-8"))
        if df.neural is not None:
            snap.write(staging / "neural.json", df.neural)
        record = LifeRecord.load(staging)  # (the record reads back on this build)
        if backup is not None:
            os.replace(same[0], backup)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    try:
        voice = _install_voice(df, voices or lineage_root())
    except OSError:
        # (the duck is in place; its server builds the same library again from the lineage)
        voice = {
            "installed": False,
            "key": (df.manifest.get("voice") or {}).get("key"),
            "same": None,
        }
    return {
        "ok": True,
        "folder": str(target),
        "id": df.id,
        "name": record.name,
        "seed": record.seed,
        "growth_day": df.manifest["duck"].get("growth_day"),
        "replaced": backup is not None,
        "backup": None if backup is None else str(backup),
        "voice": voice,
        "start": f"--life {target}",
    }


def _install_voice(df: DuckFile, root: Path) -> dict[str, Any]:
    """Put the file's voice library where the server looks for it (``<root>/seed<v>/<key>``), unless a
    complete one of that key is there already (then it is compared, never overwritten)."""
    part = df.manifest.get("voice") or {}
    files = df.library()
    if not part.get("embedded") or not files:
        return {"installed": False, "key": None, "same": None}
    key = str(part["key"])
    target = root / f"seed{int(part['voice_seed'])}" / key
    if _library_complete(target):
        from duckmind.simweb import lineage_voice as LV

        return {"installed": False, "key": key, "same": LV.small_hash(target) == part.get("hash")}
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{key}.", suffix=".new", dir=target.parent))
    try:
        staging.chmod(0o755)
        for name, data in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)  # (an incomplete one: the builder's rule)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"installed": True, "key": key, "same": True}


# ── the command line ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m duckmind.life.duckfile",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="pack a life folder into a duck file")
    ex.add_argument("folder", type=Path)
    ex.add_argument(
        "-o", "--out", type=Path, default=None, help="the file (default <name>-<date>.duck)"
    )
    ex.add_argument("--no-voice", action="store_true", help="leave the voice library out")
    ins = sub.add_parser("inspect", help="check a duck file and say what is in it")
    ins.add_argument("file", type=Path)
    im = sub.add_parser("import", help="put the duck of a duck file on this computer")
    im.add_argument("file", type=Path)
    im.add_argument("--into", type=Path, default=None, help="the new life folder (must not exist)")
    im.add_argument(
        "--replace",
        action="store_true",
        help="this computer has the same duck: replace it (its folder is kept as a backup)",
    )
    args = ap.parse_args(argv)
    try:
        if args.cmd == "export":
            path, manifest = export(args.folder, args.out, library=not args.no_voice)
            out: dict[str, Any] = {"file": str(path), "bytes": path.stat().st_size, **manifest}
            out.pop("parts", None)
        elif args.cmd == "inspect":
            out = summary(read_path(args.file))
        else:
            out = install(read_path(args.file), into=args.into, replace=args.replace)
    except DuckFileError as error:
        print(str(error.args[0] if error.args else error), file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
