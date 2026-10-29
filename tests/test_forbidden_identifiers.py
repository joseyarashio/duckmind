"""God's-eye names in Duck Core (physical-design spec §2.2 rule R1, §8.10; survey_godseye.md
G01–G71, W10): what a real duck in an unknown home cannot know, and outcome dice the mind must
not roll (rule R4).

``forbidden_identifiers_baseline.json`` records how often each name below is used today, per
file under ``src/duckmind/duckcore/``. A count that grows, or a file that starts using a name,
fails. A count that shrinks passes with a note: lower the baseline then, so it cannot grow back.
Phase 5 ends with every count at zero (the spec's "blindfold test").

What counts is read with ``ast``, so comments and docstrings never do:

* ``name``: an identifier spelled exactly so (a variable, attribute, def, parameter, keyword or
  import), or a string equal to it (a dict or payload key, ``getattr``);
* ``attr``: an attribute only (``s.heading``);
* ``dice``: an outcome draw, ``_rand("<purpose>")``;
* ``text``: words inside a string (the duck's own words claiming a wall or the door).
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DUCKCORE = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "duckcore"
BASELINE = Path(__file__).with_name("forbidden_identifiers_baseline.json")


@dataclass(frozen=True)
class Forbidden:
    label: str
    spec: str
    why: str
    how: str = "name"  # name | attr | dice | text
    regex: str | None = None  # a name pattern instead of the label itself


W = "against the known walls"
CATALOGUE: tuple[Forbidden, ...] = (
    # The room is a known 6 × 6 m square (G01–G08); the nose-to-the-wall check is G04/G05.
    Forbidden(
        "ROOM_*", "G01", "the room is a known square around the mat", regex="ROOM_[A-Z0-9_]+"
    ),
    Forbidden("WALL_MARGIN_MM", "G01", "it turns back short of walls it never sensed"),
    Forbidden("PROBE_MM", "G01, G03", f"800 mm clear ahead, measured {W}"),
    Forbidden("TURN_ROOM_MM", "G01, G04", f"room left ahead at a walk's end, measured {W}"),
    Forbidden("_blocked", "G02", "path points beyond the known walls are refused"),
    Forbidden("_blocker", "G02", "the wall, leash and spot test, prepared once per path"),
    Forbidden("_turn_away", "G03", "turns toward open floor of the known room, facing its middle"),
    Forbidden("_room_to_turn", "G04, G05", f"the nose-to-the-wall check, {W}"),
    Forbidden("_plan_walk", "G05, G06", f"wander and turn plans probed {W}"),
    Forbidden("_arc_turn", "G07", f"arc turns probed {W}"),
    Forbidden("_plan_leg_toward", "G08", f"legs toward a target probed {W}"),
    Forbidden(
        "_turn_home_fits", "G18, G22", "a turn home fits the home radius, from the exact pose"
    ),
    Forbidden("_pose_turns_home", "G18, G22", "the same test for a planned pose"),
    # The fixed 12 × 12 grid (G09, G10, G20, G51).
    Forbidden("CELL_MM", "G09", "the room cut into a fixed grid around the nest"),
    Forbidden("CELLS_PER_SIDE", "G09", "the fixed grid's size"),
    Forbidden("cell_mm", "G09", "the fixed grid's cell size in the content table"),
    Forbidden("cell_of", "G09", "exact position to grid cell, clamped into the room"),
    Forbidden("cell_center", "G09", "grid cell to room coordinates"),
    Forbidden("CENTERS", "G09, G10", "every cell centre of the known room"),
    Forbidden("pick_target", "G10, G20", "targets in never-seen cells, by exact heading and range"),
    Forbidden("mark_seen", "G51", "the cell under its exact pose counts as seen"),
    # The pose is the plan, never estimated (G12–G17), and reads of that exact pose.
    Forbidden("pose_along", "G12", "the predicted path becomes where it is"),
    Forbidden("_along", "G12, G14", "path integration used as its position"),
    Forbidden("pose_now", "G15", "the page draws the plan as the duck"),
    Forbidden("pos_x", "G12–G21", "its exact x (the plan), read as truth"),
    Forbidden("pos_y", "G12–G21", "its exact y (the plan), read as truth"),
    Forbidden(
        ".heading", "G12–G21", "its exact heading, read as truth", how="attr", regex="heading"
    ),
    Forbidden("_pose", "G12–G21", "the exact pose"),
    Forbidden("_pose_list", "G15, G16", "the exact pose written into its events"),
    Forbidden("pose_err_mm", "G17", "uncertainty as a formula, reset by made-up sightings"),
    Forbidden("dist_from_nest_mm", "G16", "a walk's exact distance from the nest, logged as fact"),
    Forbidden("dist_to_mm", "G16", "a walk's exact distance to its target, logged as fact"),
    # Leash, arrival, places and fall spots by exact distance (G18–G21, G48).
    Forbidden("leash_mm", "G18", "how far it may go, from its exact distance to the nest"),
    Forbidden("_end_check", "G18", "the leash checked on the exact pose"),
    Forbidden("_arrived", "G19", "arrival by exact distance"),
    Forbidden("good_here", "G21", "a liked place at its exact cell"),
    Forbidden("active_spots", "G48", "fall spots in exact coordinates"),
    Forbidden("AVOID_RADIUS_MM", "G48", "fall spots avoided by exact distance"),
    # The nest is at (0, 0) facing +x, known at all times (G22–G28).
    Forbidden("dist_mm", "G22", "distance to the nest taken as distance to the origin"),
    Forbidden("backoff_mm", "G23", "the way home staged at fixed coordinates"),
    Forbidden("approach_mm", "G23, G29", "the nest 'seen' within a fixed exact distance"),
    Forbidden("funnel_mm", "G24", "close enough to dock, by exact distance"),
    Forbidden("_home_for_battery", "G24, G28", "the battery kept for the exact way home"),
    Forbidden("_place_on_nest", "G25", "on the nest it is at (0, 0) facing the opening"),
    Forbidden("FRONT_MM", "G26", "straight out of the opening at +x"),
    Forbidden("ACTUAL_SPEED_MM", "G28, G52", "a fixed ground speed instead of measured odometry"),
    Forbidden("ACTUAL_YAW_RATIO", "G52–G54", "a fixed turn ratio instead of measured yaw"),
    Forbidden("_actual_speed_mm", "G52", "predicted speed used as the outcome"),
    # People, the door, walls in words (G11, G38, G39, G42).
    Forbidden("door_xy", "G38", "the front door's position is a constant"),
    Forbidden("family_spot", "G38, G39", "where the family is, from exact cells or the door"),
    Forbidden("前面是牆", "G11", "narrates a wall taken from the map", how="text"),
    Forbidden("看著門口", "G42", "claims it watches a door it cannot locate", how="text"),
    # Picked up, put down, powered off: back at the mat (G43–G46); darkness from the clock (G49).
    Forbidden("_place_on_mat", "G43, G44, G46", "set down or switched off, it is at the mat pose"),
    Forbidden("_night", "G49", "darkness read from the clock, not from the camera"),
    # Outcome dials in the duck's own config (G57–G61, G71) and its battery model (G62, G63).
    Forbidden("fall_ppm_per_s", "G57, G71", "fall odds in the duck's own config"),
    Forbidden("fall_ppm_per_start", "G57, G71", "fall odds in the duck's own config"),
    Forbidden("righting_ppm", "G58, G71", "self-righting odds in the duck's own config"),
    Forbidden("stand_ppm", "G59, G71", "stand-up odds in the duck's own config"),
    Forbidden("dock_ppm", "G61, G71", "docking odds in the duck's own config"),
    Forbidden("BATTERY_DRAIN_PER_MIN", "G62", "battery drain as a constant"),
    Forbidden("CHARGE_PER_MIN", "G63", "charging as a constant"),
    Forbidden("nest_charge_per_min", "G63", "charging on the nest as a constant"),
    Forbidden("_charge_per_min", "G63", "charging as a constant"),
    # Outcome dice rolled by the mind (rule R4).
    Forbidden("fall", "G31, G57", "a fall drawn when a walk starts", how="dice"),
    Forbidden("fall.at", "G31, G57", "the second the fall happens, drawn", how="dice"),
    Forbidden("righting", "G32, G58", "whether a try to get up works, drawn", how="dice"),
    Forbidden("stand.try", "G33, G59", "whether standing up works, drawn", how="dice"),
    Forbidden("dock.try", "G30, G61", "whether the contacts touched, drawn", how="dice"),
    # (``struggle`` left this list in phase 4: lying stuck, *trying* to struggle is Duck Core's
    # own decision — an output it draws like any other, recorded as ``duck.struggle`` signed
    # self — not an outcome of the world. Design §4 G34, §5 row 7.)
    Forbidden("play.meters", "G35, G69", "metres walked in gamepad play, drawn", how="dice"),
    Forbidden("play.skill", "G35, G69", "skills played in gamepad play, drawn", how="dice"),
    Forbidden("false_alarm.tv", "G36", "its own ear's false alarm, drawn", how="dice"),
    Forbidden("detect.touch", "G67", "whether its own touch detector noticed, drawn", how="dice"),
    Forbidden("detect.hold", "G68", "whether its own head-hold detector noticed", how="dice"),
    # Physics truth, for the page and the evaluators only (W10); none in the mind today.
    Forbidden("oracle", "W10", "the simulator's truth port"),
    Forbidden("trunk", "W10", "the body server's true trunk pose"),
    Forbidden("trunk_z", "W10", "the body server's true trunk height"),
    Forbidden("qpos", "W10", "MuJoCo joint and free-body positions"),
    Forbidden("xpos", "W10", "MuJoCo body positions in the world"),
)


def _docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _uses(tree: ast.AST) -> Iterator[tuple[str, str]]:
    """(how, spelling) of every identifier, string and draw in the tree."""
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield "name", node.id
        elif isinstance(node, ast.Attribute):
            yield "name", node.attr
            yield "attr", node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "name", node.name
        elif isinstance(node, ast.arg):
            yield "name", node.arg
        elif isinstance(node, ast.keyword) and node.arg is not None:
            yield "name", node.arg
        elif isinstance(node, ast.alias):
            yield "name", node.name.rsplit(".", 1)[-1]
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            yield from (("name", name) for name in node.names)
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
        ):
            yield "name", node.value
            yield "text", node.value
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func, purpose = node.func, node.args[0].value
            callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if callee == "_rand" and isinstance(purpose, str):
                yield "dice", purpose


def _count(tree: ast.AST) -> Counter[str]:
    """How often each catalogue label is used in the tree."""
    exact: dict[tuple[str, str], list[str]] = {}
    patterns: list[tuple[str, re.Pattern[str], str]] = []
    texts = [(entry.label, entry.label) for entry in CATALOGUE if entry.how == "text"]
    for entry in CATALOGUE:
        if entry.how == "text":
            continue
        if entry.regex is not None and re.escape(entry.regex) != entry.regex:
            patterns.append((entry.how, re.compile(entry.regex), entry.label))
        else:
            exact.setdefault((entry.how, entry.regex or entry.label), []).append(entry.label)
    found: Counter[str] = Counter()
    for how, spelling in _uses(tree):
        for label in exact.get((how, spelling), ()):
            found[label] += 1
        for kind, pattern, label in patterns:
            if kind == how and pattern.fullmatch(spelling):
                found[label] += 1
        if how == "text":
            found.update(label for words, label in texts if words in spelling)
    return found


@cache
def _counts() -> dict[str, dict[str, int]]:
    """{label: {file under duckcore/: uses}} for today's tree, every label present."""
    table: dict[str, dict[str, int]] = {entry.label: {} for entry in CATALOGUE}
    for path in sorted(DUCKCORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for label, uses in sorted(_count(tree).items()):
            table[label][path.relative_to(DUCKCORE).as_posix()] = uses
    return table


def _baseline() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = json.loads(BASELINE.read_text(encoding="utf-8"))["counts"]
    return counts


def test_the_catalogue_and_its_baseline_agree() -> None:
    labels = [entry.label for entry in CATALOGUE]
    assert len(labels) == len(set(labels)), "a catalogue label is used twice"
    assert all(entry.how in {"name", "attr", "dice", "text"} for entry in CATALOGUE)
    baseline = _baseline()
    assert set(baseline) == set(labels), (
        "forbidden_identifiers_baseline.json and CATALOGUE list different names: "
        f"missing {sorted(set(labels) - set(baseline))}, extra {sorted(set(baseline) - set(labels))}"
    )


def test_no_god_s_eye_name_spreads_in_duck_core() -> None:
    now, baseline = _counts(), _baseline()
    spec = {entry.label: entry.spec for entry in CATALOGUE}
    grew = [
        f"{label} ({spec[label]}) in {file}: {baseline.get(label, {}).get(file, 0)} → {uses}"
        + ("  (a file that did not use it)" if file not in baseline.get(label, {}) else "")
        for label, files in now.items()
        for file, uses in files.items()
        if uses > baseline.get(label, {}).get(file, 0)
    ]
    assert not grew, (
        "Duck Core uses more god's-eye names than tests/forbidden_identifiers_baseline.json "
        "allows (spec R1/R4, §8.10). Read it from a sense instead:\n" + "\n".join(grew)
    )
    # 只准變少 (spec §17): a use that goes away is locked in the same change, so it can never
    # quietly come back up to the old count.
    shrank = [
        f'"{label}": "{file}": {was} → {now.get(label, {}).get(file, 0)}'
        + ("  (remove the file's entry)" if now.get(label, {}).get(file, 0) == 0 else "")
        for label, files in baseline.items()
        for file, was in files.items()
        if now.get(label, {}).get(file, 0) < was
    ]
    assert not shrank, (
        "Fewer god's-eye names in Duck Core — good. Lower them in "
        "tests/forbidden_identifiers_baseline.json in the same change (the baseline only goes "
        "down):\n  " + "\n  ".join(shrank)
    )


# Phase 4 (design §5): the ten outcome dice left for LegacyWorld (the same key, the world's
# own counters) and the five outcome rates for its header; old saves' rates are converted on
# the world side, never in Duck Core.
PHASE4_GONE = (
    "fall",
    "fall.at",
    "righting",
    "stand.try",
    "dock.try",
    "detect.touch",
    "detect.hold",
    "false_alarm.tv",
    "play.meters",
    "play.skill",
    "fall_ppm_per_s",
    "fall_ppm_per_start",
    "righting_ppm",
    "stand_ppm",
    "dock_ppm",
)


def test_phase4_moved_the_outcome_dice_and_rates_out_of_duck_core() -> None:
    now = _counts()
    left = {label: files for label in PHASE4_GONE if (files := now.get(label))}
    assert not left, f"outcome dice or rates still in Duck Core (they belong to the world): {left}"
    assert not any(_baseline().get(label) for label in PHASE4_GONE)
    assert "struggle" not in {entry.label for entry in CATALOGUE}  # a decision, not a die


def test_the_counter_reads_code_not_comments() -> None:
    tree = ast.parse(
        textwrap.dedent(
            '''
            """ROOM_HALF_MM in a docstring does not count, nor 前面是牆 here."""
            from x import PROBE_MM  # PROBE_MM in a comment does not count
            ROOM_HALF_MM = 3000
            ROOM_LIMIT_MM = ROOM_HALF_MM - 250

            def _room_to_turn(self, pose_err_mm=0):
                self._room_to_turn(pose_err_mm=1)
                self.s.heading += 1
                heading = self._rand("fall")
                return getattr(self, "door_xy"), "（前面是牆，轉開）", self._rand("fall.at")
            '''
        )
    )
    found = _count(tree)
    assert found["ROOM_*"] == 3 and found["PROBE_MM"] == 1
    assert found["_room_to_turn"] == 2 and found["pose_err_mm"] == 2
    assert found[".heading"] == 1  # the attribute, not the local name
    assert found["fall"] == 1 and found["fall.at"] == 1
    assert found["door_xy"] == 1 and found["前面是牆"] == 1
