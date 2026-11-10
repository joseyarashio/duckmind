"""Where duckmind keeps things outside the repo: one place for every default folder.

Each folder is chosen in this order: its environment variable, else the machine's
``paths.toml``, else a folder under ``DUCKMIND_HOME``.

* ``duckmind_home()`` — ``$DUCKMIND_HOME``, else ``~/.duckmind`` (the ducks' lives and saves
  are here too).
* ``data_dir()`` — caches, live records and measurement records (the voice banks, the motion
  table, the voice-stream helper build): ``$DUCKMIND_DATA_DIR``, else ``data`` in
  ``paths.toml``, else ``<home>/data``. ``motions_dir()`` is the trained-motion table's folder
  inside it (``<data>/live/motions``).
* ``loops_dir(life)`` — a server's loop-watcher captures: ``$DUCKMIND_LOOP_DIR``, else
  ``<life>/loops``, else a temporary folder of the process's own (never the live duck's).
* ``policy_root()`` — the ONNX policy sets: ``$DUCKMIND_POLICY_ROOT``, else ``policies`` in
  ``paths.toml``, else ``<home>/policies``. Inside it, ``policy_dir()`` (the shipped set:
  ``$DUCKMIND_POLICY_DIR``, else ``<policy root>/current``), ``v5_policy_dir()`` (the v5
  release: ``$DUCKMIND_V5_DIR``, else ``<policy root>/releases/seed-v5``) and ``v9_policy_dir()``
  (the v9 release the live duck walks with: ``$DUCKMIND_V9_DIR``, else
  ``<policy root>/releases/v9``).
* ``upstream_root()`` — the folder holding the upstream ``microduck/`` and ``microduck_rl/``
  checkouts (``python -m duckmind.setup_cmd`` clones them there): ``$DUCKMIND_UPSTREAM``, else
  ``upstream`` in ``paths.toml``, else ``<home>/upstream``. ``microduck_mjcf_dir()`` is the robot model
  folder inside it, unless ``$DUCKMIND_MICRODUCK_MJCF`` names another.

``paths.toml`` is optional and lives at ``<home>/paths.toml`` (``$DUCKMIND_PATHS_FILE`` names
another). It holds flat string keys, ``~`` is expanded, unknown keys are an error::

    data = "~/duckmind-records"
    policies = "~/policies"
    upstream = "~/src/pollen"

Every function reads the environment and the file when called, so tests and tools may set
the variables at any time.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

HOME_ENV = "DUCKMIND_HOME"
PATHS_FILE_ENV = "DUCKMIND_PATHS_FILE"
DATA_ENV = "DUCKMIND_DATA_DIR"
POLICY_ROOT_ENV = "DUCKMIND_POLICY_ROOT"
POLICY_ENV = "DUCKMIND_POLICY_DIR"
V5_ENV = "DUCKMIND_V5_DIR"
V9_ENV = "DUCKMIND_V9_DIR"
UPSTREAM_ENV = "DUCKMIND_UPSTREAM"
MJCF_ENV = "DUCKMIND_MICRODUCK_MJCF"
LOOP_ENV = "DUCKMIND_LOOP_DIR"
KEYS = ("data", "policies", "upstream")
MJCF_IN_UPSTREAM = Path("microduck_rl/src/mjlab_microduck/robot/microduck")


class PathsFileError(ValueError):
    """``paths.toml`` is there but is not a flat table of the known string keys."""


def duckmind_home() -> Path:
    """``$DUCKMIND_HOME``, else ``~/.duckmind``."""
    raw = os.environ.get(HOME_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".duckmind"


def paths_file() -> Path:
    """``$DUCKMIND_PATHS_FILE``, else ``<home>/paths.toml`` (it need not exist)."""
    raw = os.environ.get(PATHS_FILE_ENV)
    return Path(raw).expanduser() if raw else duckmind_home() / "paths.toml"


def configured() -> dict[str, Path]:
    """The folders ``paths.toml`` names (empty when there is no file)."""
    file = paths_file()
    try:
        raw = file.read_bytes()
    except FileNotFoundError:
        return {}
    try:
        table = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as err:
        raise PathsFileError(f"{file}: {err}") from err
    out: dict[str, Path] = {}
    for key, value in table.items():
        if key not in KEYS or not isinstance(value, str) or not value:
            raise PathsFileError(f"{file}: {key!r} is not one of {', '.join(KEYS)} with a folder")
        out[key] = Path(value).expanduser()
    return out


def _pick(env: str, key: str, fallback: str) -> Path:
    raw = os.environ.get(env)
    if raw:
        return Path(raw).expanduser()
    return configured().get(key) or duckmind_home() / fallback


def data_dir() -> Path:
    """Caches, live records and measurement records (see the module docstring)."""
    return _pick(DATA_ENV, "data", "data")


def motions_dir() -> Path:
    """The trained-motion table (``skills.json`` and its networks): ``<data>/live/motions``."""
    return data_dir() / "live" / "motions"


def policy_root() -> Path:
    """The folder of ONNX policy sets."""
    return _pick(POLICY_ROOT_ENV, "policies", "policies")


def policy_dir() -> Path:
    """The shipped policy set: ``$DUCKMIND_POLICY_DIR``, else ``<policy root>/current``."""
    raw = os.environ.get(POLICY_ENV)
    return Path(raw).expanduser() if raw else policy_root() / "current"


def v5_policy_dir() -> Path:
    """The v5 release set: ``$DUCKMIND_V5_DIR``, else ``<policy root>/releases/seed-v5``."""
    raw = os.environ.get(V5_ENV)
    return Path(raw).expanduser() if raw else policy_root() / "releases" / "seed-v5"


def v9_policy_dir() -> Path:
    """The v9 release set: ``$DUCKMIND_V9_DIR``, else ``<policy root>/releases/v9``."""
    raw = os.environ.get(V9_ENV)
    return Path(raw).expanduser() if raw else policy_root() / "releases" / "v9"


def loops_dir(life: Path | None = None) -> Path:
    """Where a server writes its loop-watcher captures (:mod:`duckmind.simweb.loopwatch`): ``$DUCKMIND_LOOP_DIR``
    (the live launch names its own folder this way: ``<data>/live/loops``), else ``<life>/loops`` for a server that
    lives the duck in the folder ``life``, else a folder of this process's own in the system's temporary folder
    (``duckmind-loops-<pid>``). By default a server never writes into another server's folder, the live one's
    included."""
    raw = os.environ.get(LOOP_ENV)
    if raw:
        return Path(raw).expanduser()
    if life is not None:
        return Path(life).expanduser() / "loops"
    return Path(tempfile.gettempdir()) / f"duckmind-loops-{os.getpid()}"


def upstream_root() -> Path:
    """The folder holding the upstream ``microduck/`` and ``microduck_rl/`` checkouts."""
    return _pick(UPSTREAM_ENV, "upstream", "upstream")


def microduck_dir() -> Path:
    return upstream_root() / "microduck"


def microduck_rl_dir() -> Path:
    return upstream_root() / "microduck_rl"


def microduck_mjcf_dir() -> Path:
    """The upstream robot model folder: ``$DUCKMIND_MICRODUCK_MJCF``, else the one in
    ``microduck_rl`` under :func:`upstream_root`."""
    raw = os.environ.get(MJCF_ENV)
    return Path(raw).expanduser() if raw else upstream_root() / MJCF_IN_UPSTREAM
