"""robotd 0.16.0+ runs one move at a time: nothing cuts into a running skill, so a held skill (quick turn, panic
flight, freeze, lie low, turn on the spot) ends only at the end of a window once the body link stops asking for it —
up to the chain window plus a whole window later (0.4–1.0 s windows: 0.55–1.15 s). On such a robotd the simulator's
table gives every held entry a short renewal slot, so a run ends ≤ 0.3 s after the brain lets go, and a drop stop
ends a flight as soon. robotd 0.15.0 keeps the table's own windows (UPSTREAM-0185-2026-11-05 §10)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from test_neuro_bodylink import DT, state

from duckmind import livebody as LB
from duckmind.duckbody import protocol as P
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import gt_wiring as W
from duckmind.skilltrain import guard as G
from duckmind.skilltrain import motion_table as M
from duckmind.skilltrain.table import CHAIN_WINDOW_S, SKILLS_ENV

HELD = ("panic_run", "quick_turn_left", "quick_turn_right", "freeze_2", "freeze_3", "freeze_4", "lie_low",
        "turn_left", "turn_right")  # fmt: skip
LIMIT_S = 0.3


def table(tmp_path: Path) -> dict[str, str]:
    """A motion table installed the way the launcher finds it (networks are placeholder bytes)."""
    src, walk = tmp_path / "src", tmp_path / "walk"
    src.mkdir()
    walk.mkdir()
    for role, name in M.FILES.items():
        (src / name).write_bytes(role.encode())
    for file, _why in M.WALK_ROLES.values():
        (walk / file).write_bytes(file.encode())
    sources = {role: str(src / name) for role, name in M.FILES.items()}
    path = M.install(tmp_path / "motions", sources, walk_dir=walk)
    return {SKILLS_ENV: str(path)}


def checkout(tmp_path: Path, version: str) -> Path:
    root = tmp_path / f"microduck-{version}"
    root.mkdir()
    (root / "Cargo.toml").write_text(
        f'[workspace]\nmembers = []\n\n[workspace.package]\nversion = "{version}"\n'
    )
    return root


def windows(tmp_path: Path, env: dict[str, str], version: str) -> dict[str, tuple[float, bool]]:
    """name → (duration, chain) as the launcher writes robotd.toml for this checkout's robotd."""
    one = LB.robotd_one_move_at_a_time(checkout(tmp_path, version))
    doc = tomllib.loads(W.verified_section(env, one_move_at_a_time=one))
    return {s["name"]: (float(s["duration"]), bool(s["chain"])) for s in doc["policy"]["skill"]}


class OneMove:
    """robotd 0.18.5's skill windows (control.rs ``start_skill`` / ``step``): a request for the running chaining skill
    refreshes its chain counter; any other request while a skill runs is refused; a window's end restarts a chaining
    skill whose counter is still positive, else hands back to the walk. Windows expire before the tick, advance after."""

    def __init__(self, defs: dict[str, tuple[float, bool]]) -> None:
        self.defs = defs
        self.active: list | None = None  # [name, remaining s, chain s]

    def request(self, name: str) -> None:
        duration, chains = self.defs[name]
        if self.active is not None:
            if self.active[0] == name and chains:
                self.active[2] = CHAIN_WINDOW_S
            return
        self.active = [name, duration, 0.0]

    def step(self) -> str:
        a = self.active
        if a is not None and a[1] <= 0.0:
            duration, chains = self.defs[a[0]]
            self.active = [a[0], duration, 0.0] if chains and a[2] > 0.0 else None
        if self.active is None:
            return "walk"
        self.active[1] -= DT
        self.active[2] = max(0.0, self.active[2] - DT)
        return self.active[0]


def _link(names) -> BodyLink:
    link = BodyLink()
    link.skills = G.SkillGuard(list(names), interrupt=False)
    return link


def _tail(
    defs, ask: BodyRequest, held_ticks: int, *, let_go: BodyRequest, drop: bool = False, skill: str
) -> float:
    """Walk 1 s, ask for ``ask`` for ``held_ticks`` ticks, then ``let_go`` (with a drop under the nose when
    ``drop``): the seconds robotd still runs ``skill`` after the first tick the link no longer wants it."""
    robotd = OneMove(defs)
    link = _link(defs)
    t, label, seen = 0.0, "walk", False
    for k in range(50 + held_ticks + 100):
        t += DT
        let = k >= 50 + held_ticks
        req = ask if 50 <= k and not let else (let_go if let else BodyRequest(vx=0.4))
        depth = None
        if drop and let:
            depth = P.DepthView(state(t).t_ns, None, None, None, 4, False, 0)
        out = link.tick(state(t, policy=label), req, depth=depth)
        for i in out.intents:
            if i.method == "robot.do":
                robotd.request(i.params["skill"])
        label = robotd.step()
        if label == skill:
            seen = True
        if let and label != skill:
            assert seen, f"{skill} never ran"
            return t - (50 + held_ticks) * DT - DT
    raise AssertionError(f"{skill} still running 2 s after the link let go")


def _worst(defs, ask, let_go, skill, *, drop=False) -> float:
    # every phase of the let-go against robotd's window (held 1.0-1.5 s)
    return max(_tail(defs, ask, n, let_go=let_go, drop=drop, skill=skill) for n in range(50, 76))


def test_a_drop_stop_ends_a_flight_within_the_bound_on_a_one_move_robotd(tmp_path) -> None:
    defs = windows(tmp_path, table(tmp_path), "0.18.5")
    flee = BodyRequest(vx=0.5, panic=True)
    # the brain still flees; the drop under the nose is the reason the flight has to end
    assert _worst(defs, flee, flee, "panic_run", drop=True) <= LIMIT_S


def test_held_skills_end_soon_after_the_brain_lets_go_on_a_one_move_robotd(tmp_path) -> None:
    defs = windows(tmp_path, table(tmp_path), "0.18.5")
    walk = BodyRequest(vx=0.4)
    assert (
        _worst(defs, BodyRequest(vx=0.5, panic=True), walk, "panic_run") <= LIMIT_S
    )  # the fright gone
    assert (
        _worst(defs, BodyRequest(vx=0.3, pivot=1.0), walk, "quick_turn_left") <= LIMIT_S
    )  # the turn reached
    for skill in ("freeze_3", "lie_low", "turn_left"):
        assert _worst(defs, BodyRequest(skill=skill), walk, skill) <= LIMIT_S, skill


def test_every_held_entry_renews_and_a_one_shot_keeps_its_window(tmp_path) -> None:
    env = table(tmp_path)
    new, old = windows(tmp_path, env, "0.18.5"), windows(tmp_path, env, "0.15.0")
    for name in HELD:
        assert new[name][1] and new[name][0] < CHAIN_WINDOW_S, (
            name
        )  # every window ends inside a request's reach
    for name, (duration, chain) in old.items():
        if name not in HELD:
            assert new[name] == (duration, chain), (
                name
            )  # one-shots (bow, roulade, startles …) as they were


def test_the_015_robotd_gets_the_table_as_it_was(tmp_path) -> None:
    env = table(tmp_path)
    old = LB.robotd_one_move_at_a_time(checkout(tmp_path, "0.15.0"))
    assert old is False
    assert W.verified_section(env, one_move_at_a_time=old) == W.verified_section(env)
    assert LB.robotd_one_move_at_a_time(
        tmp_path / "no-checkout"
    )  # a robotd that does not say: the newer kind
