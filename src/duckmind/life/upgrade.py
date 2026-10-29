"""Attach a duck's record to the neural duck (growth integration G2): the upgrade of a life made
before ``bridge-v1`` (docs/design/GROWTH-G2-2026-10-10.md, section 7).

A record's ``core.json`` is Duck Core's inputs, replayed on every start. A life made in G1 holds the
shadow life it lived (its Duck Core simulated its own naps, walks and babbles). This keeps all of
it and adds one ``bridge`` input at the end: the seam from which Duck Core is given what the neural
duck lives and simulates nothing of its own. Before it writes anything it

* backs ``core.json`` up (``core.json.pre-g2-<stamp>``, mode 0600, never overwritten);
* replays the old record, attaches a copy, replays the new record, and checks that the old events
  are the first events of the new chain, event for event (same ids, same chain), and that the new
  record restores to the very same state.

  python -m duckmind.life.upgrade ~/.duckmind/lives/danhuang [--dry-run]
  python -m duckmind.life.upgrade ~/.duckmind/lives/danhuang --branch-from-core on|off

The second form is the family's (the planner's) decision, not part of the upgrade: whether the neural
duck's body follows the branch Duck Core holds (True) or the branch its seed's nature draws (False, the
default of a record made before G2). It rewrites ``life.json`` only (backed up first).

The server does the same at its first start with the new build (``SimState.attach_neural``), so
running this first is for the backup and the proof, not a required step.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api
from duckmind.duckcore.bridge import BRIDGE_API
from duckmind.life.record import LifeRecord, write_private


class UpgradeError(ValueError):
    """The record cannot be upgraded (said in plain words); nothing was written."""


def backup(path: Path, label: str) -> Path | None:
    """Copy ``path`` to ``<path>.<label>`` (private; None when that backup exists already)."""
    target = path.with_name(f"{path.name}.{label}")
    if target.exists() or not path.is_file():
        return None
    shutil.copy2(path, target)
    target.chmod(0o600)
    return target


def upgrade(folder: Path, *, dry_run: bool = False, stamp: str | None = None) -> dict[str, Any]:
    folder = Path(folder).expanduser()
    record = LifeRecord.load(folder)
    text = record.core_path.read_text(encoding="utf-8")
    saved = json.loads(text)
    if isinstance(saved.get("bridge"), dict):
        return {"upgraded": False, "why": "already attached", "bridge": saved["bridge"]}
    try:
        old = api.restore(copy.deepcopy(saved))
    except (api.SaveContentMismatch, api.SaveBridgeMismatch, ValueError) as error:
        raise UpgradeError(f"the record does not replay on this build: {error}") from error
    if not isinstance(old, api.Session):
        raise UpgradeError("this record is a v4_0 duck: it has no neural-duck bridge")
    new = api.restore(copy.deepcopy(saved))
    assert isinstance(new, api.Session)
    if new.core.s.bridged:
        raise UpgradeError("the record is attached already (its inputs hold a bridge)")
    api.dispatch(new, "bridge", {"api": BRIDGE_API})
    payload = {**api.save_json(new), "realtime": bool(saved.get("realtime", False))}
    again = api.restore(json.loads(json.dumps(payload)))
    assert isinstance(again, api.Session)
    old_ids = [event.event_id for event in old.core.log.events]
    new_ids = [event.event_id for event in again.core.log.events]
    same_prefix = new_ids[: len(old_ids)] == old_ids
    same_state = dataclasses.asdict(again.core.s) == dataclasses.asdict(new.core.s)
    same_head = again.core.log.head == new.core.log.head
    kept_inputs = payload["inputs"][: len(saved["inputs"])] == saved["inputs"]
    proof = {
        "old_events_are_the_first_events": same_prefix,
        "old_inputs_are_the_first_inputs": kept_inputs,
        "new_record_restores_to_the_same_state": same_state and same_head,
    }
    if not all(proof.values()):
        raise UpgradeError(f"the upgrade did not prove itself, nothing written: {proof}")
    result: dict[str, Any] = {
        "upgraded": not dry_run,
        "dry_run": dry_run,
        "growth_day": old.core.s.gd,
        "old": {"inputs": len(saved["inputs"]), "events": len(old_ids), "head": old.core.log.head},
        "new": {
            "inputs": len(payload["inputs"]),
            "events": len(new_ids),
            "head": again.core.log.head,
        },
        "proof": proof,
        "backup": None,
    }
    if not dry_run:
        label = "pre-g2-" + (stamp or datetime.now().strftime("%Y%m%d-%H%M%S"))
        kept = backup(record.core_path, label)
        if kept is None:
            raise UpgradeError(f"a backup {label} exists already: nothing written")
        write_private(record.core_path, json.dumps(payload, ensure_ascii=False))
        result["backup"] = str(kept)
    return result


def set_branch_from_core(folder: Path, on: bool) -> dict[str, Any]:
    """Whose branch the body follows (``LifeRecord.branch_from_core``): life.json only."""
    record = LifeRecord.load(Path(folder).expanduser())
    new = dataclasses.replace(record, branch_from_core=on)
    kept = backup(record.path, "pre-branch-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    write_private(record.path, json.dumps(new.to_json(), ensure_ascii=False, indent=1))
    return {"branch_from_core": on, "was": record.branch_from_core, "backup": str(kept)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", type=Path, help="the life folder (life.json, core.json…)")
    ap.add_argument("--dry-run", action="store_true", help="prove it, write nothing")
    ap.add_argument(
        "--branch-from-core", choices=("on", "off"), help="set life.json's branch_from_core"
    )
    args = ap.parse_args(argv)
    try:
        if args.branch_from_core:
            print(
                json.dumps(
                    set_branch_from_core(args.folder, args.branch_from_core == "on"), indent=1
                )
            )
            return 0
        result = upgrade(args.folder, dry_run=args.dry_run)
    except (UpgradeError, ValueError) as error:
        print(f"沒有改動：{error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
