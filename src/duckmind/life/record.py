"""The duck record: who the duck is, on disk, in one folder.

    <life folder>/
        life.json     who: name, seed, birth, the hardware it hatched with, the engine it hatched on
        core.json     Duck Core's save (config + world header + every input; replayed on restart)
        neural.json   the neural duck's slow state (the snapshot, :mod:`duckmind.life.snapshot`)
        neural.prev.json   the snapshot before the newest one
        eye/          what its eye learnt of the floor (kept with the duck)

One folder is one duck. A server given ``--life FOLDER`` lives that duck: it hatches it if the
folder is new (the birth is that moment), and otherwise brings it back as it was. The folder holds
people's names and looks (the snapshot), so it is made private (mode 0700, files 0600) and nothing
here ever leaves the machine.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from duckmind.life.clock import TPE

LIFE_FORMAT = "duckmind.life"
LIFE_VERSION = 1


class LifeError(ValueError):
    """The folder is not a duck record this build can use (said in plain words)."""


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (a temp file, fsync, rename), readable by the owner
    only. A reader never sees half a file, and a crash leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


@dataclass(frozen=True)
class LifeRecord:
    folder: Path
    name: str
    seed: int
    birth: datetime  # aware, TPE
    nature: str  # the neural duck's character preset it grows from (explorer for 4107)
    nest: bool
    tof: bool
    engine: dict[str, str]  # api.content_version() when it hatched
    # A backdated hatch (owner decision 2026-10-10: 蛋黃 starts at growth day 48): the duck was given
    # a birth this many days before the real moment and Duck Core lived those days, awake, at creation
    # (SimState.grow(lived=True)). Real growth goes on from there. 0: hatched now.
    backdated_days: int = 0
    # Growth G2: whose branch the neural duck's body follows at its branch point. False (a record
    # made before G2, or any record until its family decides): the branch its seed's nature draws
    # (explorer for 蛋黃). True: the branch Duck Core holds, which is nature plus the care the neural
    # duck really got (a life hatched attached from its first moment). Written to life.json only
    # when True, so a G1 record is unchanged.
    branch_from_core: bool = False

    # ── where things live ────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self.folder / "life.json"

    @property
    def core_path(self) -> Path:
        return self.folder / "core.json"

    @property
    def neural_path(self) -> Path:
        return self.folder / "neural.json"

    @property
    def neural_prev_path(self) -> Path:
        return self.folder / "neural.prev.json"

    @property
    def eye_dir(self) -> Path:
        return self.folder / "eye"

    # ── making and reading ───────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        folder: Path,
        *,
        name: str,
        seed: int,
        birth: datetime | None = None,
        nature: str = "",
        nest: bool = False,
        tof: bool = True,
        eye_from: Path | None = None,
        backdate_days: int = 0,
        branch_from_core: bool = False,
    ) -> LifeRecord:
        """Hatch a new duck in ``folder`` (it must not hold one already). ``birth``: now by
        default. ``eye_from``: a folder of an eye's floor model to copy in (copied, never moved)."""
        from duckmind.duckcore.api import content_version

        folder = Path(folder)
        if (folder / "life.json").exists():
            raise LifeError(f"{folder} already holds a duck: not hatching another over it")
        when = (birth or datetime.now(TPE)).astimezone(TPE)
        if backdate_days:
            when = when - timedelta(days=int(backdate_days))
        record = cls(
            folder=folder,
            name=name,
            seed=int(seed),
            birth=when,
            nature=nature,
            nest=bool(nest),
            tof=bool(tof),
            engine=content_version(),
            backdated_days=int(backdate_days),
            branch_from_core=bool(branch_from_core),
        )
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(folder, 0o700)
        except OSError:
            pass
        if eye_from is not None and Path(eye_from).is_dir() and not record.eye_dir.exists():
            shutil.copytree(eye_from, record.eye_dir)
        write_private(record.path, json.dumps(record.to_json(), ensure_ascii=False, indent=1))
        return record

    @classmethod
    def load(cls, folder: Path) -> LifeRecord:
        folder = Path(folder)
        path = folder / "life.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LifeError(f"{folder} holds no duck (no life.json)") from None
        except (OSError, ValueError) as error:
            raise LifeError(f"{path} cannot be read: {error}") from error
        if data.get("format") != LIFE_FORMAT:
            raise LifeError(f"{path} is not a duck record (format {data.get('format')!r})")
        if int(data.get("version", 0)) > LIFE_VERSION:
            raise LifeError(
                f"{path} was written by a newer build (record version {data.get('version')}, "
                f"this build reads up to {LIFE_VERSION}); it is left as it is"
            )
        hardware = data.get("hardware") or {}
        return cls(
            folder=folder,
            name=str(data["name"]),
            seed=int(data["seed"]),
            birth=datetime.fromisoformat(str(data["birth"])).astimezone(TPE),
            nature=str(data.get("nature", "")),
            nest=bool(hardware.get("nest", False)),
            tof=bool(hardware.get("tof", True)),
            engine=dict(data.get("engine") or {}),
            backdated_days=int(data.get("backdated_days", 0)),
            branch_from_core=bool(data.get("branch_from_core", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": LIFE_FORMAT,
            "version": LIFE_VERSION,
            "name": self.name,
            "seed": self.seed,
            "birth": self.birth.isoformat(),
            "nature": self.nature,
            "hardware": {"nest": self.nest, "tof": self.tof},
            "engine": dict(self.engine),
            "backdated_days": self.backdated_days,
            **({"branch_from_core": True} if self.branch_from_core else {}),
            "files": {
                "core": self.core_path.name,
                "neural": self.neural_path.name,
                "eye": self.eye_dir.name,
            },
        }

    def view(self) -> dict[str, Any]:
        """The record without the folder path (for a status answer)."""
        out = self.to_json()
        out.pop("files", None)
        return out
