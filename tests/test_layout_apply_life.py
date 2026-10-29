"""The layout editor's Apply under a life (docs/design/GROWTH-G2-2026-10-10.md, §11 and
docs/design/REAL-HOME-2026-10-xx.md): Apply takes a life snapshot and reloads the physics, and after it
the life duck is still on screen, Duck Core is still attached to it, the periodic save timer still runs
and what the duck lives still reaches Duck Core. No physics here: a stand-in for the live bridge."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_life_bridge_host import BIRTH, Life, Wall, live_a_pat
from test_life_snapshot import taught
from test_neuro_senses import Doors, World, contact

from duckmind.simweb import homeview, layouts
from duckmind.simweb.neuro_live import Lanes, NeuroDriver, NeuroSpec
from duckmind.simweb.server import SimServer, make_handler
from duckmind.world import neuro_senses as ns


class StubLive:
    """The live bridge as far as Apply touches it: where the duck is, and a restart that gives the
    neural duck a new body (detach, then attach, as :meth:`LiveBridge._start` does)."""

    ready = True
    restoring = False
    restored = None

    def __init__(self, driver: NeuroDriver) -> None:
        self.neuro = driver
        self.restarts: list[tuple[str | None, Any]] = []

    def take_fresh(self) -> bool:
        return False

    def set_clock(self, *args: Any, **kwargs: Any) -> None:
        pass

    def truth_now(self) -> dict[str, Any]:
        return {"x_mm": 1500, "y_mm": -700, "yaw_mrad": 300}

    def restart(self, side: Any = None, *, home: str | None = None, restore: Any = None) -> None:
        self.restarts.append((home, restore))
        self.neuro.detach()
        self.neuro.attach(SimpleNamespace(passive=False, name=f"body {len(self.restarts)}"))


class Rig:
    """A life whose neural driver is the real one (stand-in lanes, a stub body), a host that
    saves its mind, and a server with the layout editor on top."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("_run", "_aux", "_tof"):  # (the tick threads need a robotd: not here)
            monkeypatch.setattr(NeuroDriver, name, lambda self, body, stop: None)
        self.wall = Wall(BIRTH)
        self.life = Life(folder, self.wall, backdate=14)
        self.senses = ns.SenseHub(seed=4107)
        lanes = Lanes(
            brain=taught(),
            senses=self.senses,
            names={"brain": "stub", "senses": "real", "motor": "stub"},
        )
        self.driver = NeuroDriver(NeuroSpec(seed=4107, life=True), lanes=lanes)
        self.host = self.life.host
        self.host.attach(self.driver)
        self.host.every_s = 600.0
        self.host.watch_s = 0.01
        self.driver.attach(SimpleNamespace(passive=False, name="body 0"))
        self.state = self.life.state
        self.state.live = StubLive(self.driver)
        self.state.layouts = layouts.LayoutStore(folder / "layouts")
        self.state.layout_default = "real_home"
        homeview.set_home("real_home")
        self.httpd = SimServer(("127.0.0.1", 0), make_handler(self.state))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.host._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        homeview.set_home(None)

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base + path, timeout=20) as r:
            return dict(json.loads(r.read()))

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return dict(json.loads(r.read()))

    def life_view(self) -> dict[str, Any]:
        return dict(self.get("/api/life")["life"])

    def apply_edited(self, tag: str = "a", x_mm: int = -2500, y_mm: int = -1300) -> dict[str, Any]:
        layout = self.get("/api/layout")["layout"]
        layout["placements"].append(
            {"id": f"cu_{tag}", "lib": "cushion", "x_mm": x_mm, "y_mm": y_mm}
        )
        layout["id"], layout["zh"] = f"apply_{tag}", "套用測試"
        out = self.post("/api/layout/apply", {"layout": layout})
        assert out["ok"] is True, out
        return out

    def wait_saves(self, n: int, seconds: float = 5.0) -> None:
        deadline = time.time() + seconds
        while self.host.saves < n and time.time() < deadline:
            time.sleep(0.005)
        assert self.host.saves >= n, (self.host.saves, n, self.life_view()["snapshot"])


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    r = Rig(tmp_path / "duck", monkeypatch)
    yield r
    r.close()


def test_apply_keeps_the_life_duck_on_screen_attached_and_saved(rig: Rig) -> None:
    rig.host.start()
    before = rig.life_view()
    assert before["on_screen"] and before["bridge"]["attached"]
    saves = rig.host.saves
    out = rig.apply_edited()
    live: StubLive = rig.state.live
    assert [home for home, _ in live.restarts] == [out["home"]]
    assert live.restarts[0][1] == pytest.approx((1.5, -0.7, 0.3))  # the duck is put back
    after = rig.life_view()
    snapshot = after["snapshot"]
    # the mind was saved before the reload, as the life duck's ("layout"), and is still on screen
    assert rig.host.saves == saves + 1 and snapshot["last_why"] == "layout"
    assert snapshot["on_screen"] is True and snapshot["problem"] is None
    assert snapshot["watching"] is True and snapshot["saved_calendar"]
    assert after["on_screen"] is True and after["bridge"]["attached"] is True
    assert after["bridge"]["refused"] == 0 and after["bridge"]["problem"] is None
    assert rig.state.session.core.s.bridged  # Duck Core still lives on the neural duck's news
    assert rig.driver.stats.attaches == 2  # (the new body was taken by the same driver)
    assert rig.driver.life is rig.host and rig.host._brain() is not None


def test_apply_does_not_look_like_a_demo_duck_or_a_missed_save(rig: Rig) -> None:
    rig.host.start()
    rig.apply_edited("a")
    rig.apply_edited("b", -2500, -2000)  # (twice: the second one reloads an applied layout)
    for cycle in range(8):  # well over three periods of ten minutes after the Apply
        saves = rig.host.saves
        rig.wall.t += 601.0
        rig.wait_saves(saves + 1)
        snapshot = rig.life_view()["snapshot"]
        assert snapshot["problem"] is None, (cycle, snapshot)
        assert snapshot["on_screen"] is True and snapshot["watching"] is True


def test_the_periodic_timer_runs_on_after_apply(rig: Rig) -> None:
    rig.host.start()
    rig.apply_edited()
    first = rig.host.saves
    for step in range(1, 5):
        rig.wall.t += 601.0
        rig.wait_saves(first + step)
        assert rig.host.last_why in {"periodic", "night"}
        assert rig.life_view()["snapshot"]["problem"] is None
    on_disk = json.loads((rig.life.record.neural_path).read_text())
    assert on_disk["why"] in {"periodic", "night"}


def test_what_the_duck_lives_after_apply_still_reaches_duck_core(rig: Rig) -> None:
    rig.life.second()
    live_a_pat(rig.life)
    rig.life.second()
    assert rig.life.core.s.touches == 1 and rig.host.delivered == 1
    rig.apply_edited()
    delivered = rig.host.delivered
    live_a_pat(rig.life, base_ms=20_000)
    rig.life.second()
    assert rig.host.delivered == delivered + 1 and rig.host.refused == 0
    assert rig.life.core.s.touches == 2
    status = rig.life_view()["bridge"]
    assert status["delivered"] == delivered + 1 and status["seen"]["touch"] == 2


def test_a_new_body_is_not_a_charged_pack_put_in(rig: Rig) -> None:
    """The new body's battery model starts full: that is not a `feed` (a person putting a charged
    pack in), and a contact that was open on the old body ends where the old body ended."""
    from duckmind.neuro import contract as C

    def frame(t_ms: int, battery: float, site: str | None = None) -> C.SensorFrame:
        return C.SensorFrame(
            t_ms=t_ms,
            power=C.PowerSense(battery=battery, on_dock=False, charging=False),
            touch=C.TouchSense(site=site, pressure=0.3 if site else 0.0, stroke=0.6),
        )

    host = rig.host
    for k in range(40):
        rig.wall.t += 0.1
        host.on_tick(k * 100, frame(k * 100, 0.31), None, None)
    for k in range(10):  # a hand on its head when the body is replaced
        rig.wall.t += 0.1
        host.on_tick(4_000 + k * 100, frame(4_000 + k * 100, 0.31, "head"), None, None)
    rig.apply_edited()
    for k in range(40):
        rig.wall.t += 0.1
        host.on_tick(10_000 + k * 100, frame(10_000 + k * 100, 1.0), None, None)
    with rig.state.lock:
        host.deliver()
    seen = rig.life_view()["bridge"]["seen"]
    assert "feed" not in seen
    assert rig.life.core.s.touches == 1  # (the open contact was closed once, at the swap)


# ── the senses of the new world ──────────────────────────────────────────────────────────


def _feel(hub: ns.SenseHub, brain_start_ms: int, seed_world: World) -> bool:
    """Run a new physics world (its clock from 0) under a brain clock that goes on from
    ``brain_start_ms``: was the touch at physics second 1.0 felt?"""
    felt = False
    for k in range(250):
        sim = k * 0.01
        doors = Doors(t_ms=brain_start_ms + k * 10, world=seed_world.section(sim))
        felt = felt or hub(doors).touch.site is not None
    return felt


def _old_world(hub: ns.SenseHub, seconds: float) -> None:
    for k in range(int(seconds * 100)):
        hub(Doors(t_ms=k * 10, world=World().section(k * 0.01)))


def test_senses_of_a_reloaded_world_are_not_deaf_to_it() -> None:
    """Without `new_world`, the brain clock that went on while the physics clock restarted at 0 put
    every contact of the new world in the past for as long as the old world had run."""
    world = World(touches=[contact(1.0, 2.0, "head")])
    felt_without, felt_with = 0, 0
    for seed in range(12):
        stale = ns.SenseHub(seed=seed)
        _old_world(stale, 60.0)
        felt_without += _feel(stale, 60_000, world)
        fresh = ns.SenseHub(seed=seed)
        _old_world(fresh, 60.0)
        fresh.new_world()
        felt_with += _feel(fresh, 60_000, world)
    assert felt_without == 0  # (the bug this guards: nothing is felt)
    assert felt_with >= 8  # (a light contact is missed now and then, as on a real duck)


def test_new_world_keeps_what_counts_and_resets_the_models() -> None:
    hub = ns.SenseHub(seed=3, vision_hz=5.0)
    _old_world(hub, 5.0)
    hub.looks, hub.looks_refused = 4, 1
    frames, offset = hub.frames, hub.offset_ms
    assert frames > 0 and offset is not None
    hub.new_world()
    assert (hub.seed, hub.vision_hz, hub.frames, hub.looks, hub.looks_refused) == (
        3,
        5.0,
        frames,
        4,
        1,
    )
    assert hub.offset_ms is None and hub.last_t_ms is None and not hub.touch.episodes
    assert hub.next_look_t == -1.0 and not hub.pending


def test_the_driver_starts_the_senses_afresh_for_a_second_body_not_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("_run", "_aux", "_tof"):
        monkeypatch.setattr(NeuroDriver, name, lambda self, body, stop: None)
    calls: list[str] = []

    class Senses:
        def __call__(self, doors: Any) -> Any:
            raise AssertionError("no tick here")

        def new_world(self) -> None:
            calls.append("senses")

    class LifeStub:
        def body_replaced(self) -> None:
            calls.append("life")

    driver = NeuroDriver(
        NeuroSpec(seed=1),
        lanes=Lanes(brain=object(), senses=Senses(), names={"brain": "stub"}),
    )
    driver.life = LifeStub()
    driver.attach(SimpleNamespace(passive=False))
    assert calls == []  # (the first body: boot, nothing to start over)
    driver.attach(SimpleNamespace(passive=False))
    assert calls == ["senses", "life"]
