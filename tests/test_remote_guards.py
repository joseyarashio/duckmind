"""The bridge's guards for a real duck (duckmind.remotebody.bridge): the deadman's sit is never
toggled back into a stand, and an address that keeps offering a wrong token is locked out.

The seat here is a model of robotd 0.18.5's ``Controller::sit_toggle`` (robotd/src/control.rs):
``robot.do sit_toggle`` is answered ``accepted`` when it is *queued* (robotd/src/main.rs
``queue_skill``), the control loop then sits a standing duck, refuses a toggle during the seat's
ramp (only in its own log) and STANDS a seated duck on a toggle after the ramp.
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.remotebody import bridge as B
from duckmind.remotebody import link as L
from duckmind.remotebody.bridge import Bridge, Channel, SafetyKeeper
from duckmind.remotebody.client import RemoteLink
from duckmind.tmpdir import short_tmp

TOKEN = b"0123456789abcdef0123456789abcdef0123456789abcdef"
RAMP_S = 2.0  # robotd-params DEFAULT_SITSTAND_RAMP_S
RISE_S = 1.0
TICK_S = 0.1  # the keeper's robot.state rate (10 Hz)


class SlowSeat:
    """A robotd with a seat, on a clock the test steps. ``label_lag_s``: how long its
    ``robot.state`` label goes on reading the old posture after the seat changed (a seat slow to
    show). ``drop_first``: this many toggles are accepted and then not taken (robotd says so only
    in its log). ``labels``: (from, until, label) spans in which its label reads something else (a
    move that is running). ``rate``: robotd's own clock per second of the bridge's."""

    def __init__(
        self,
        *,
        label_lag_s: float = 0.0,
        drop_first: int = 0,
        labels: tuple[tuple[float, float, str], ...] = (),
        rate: float = 1.0,
    ) -> None:
        self.wall = 0.0
        self.rate = rate
        self.label_lag_s = label_lag_s
        self.drop = drop_first
        self.labels = labels
        self.seat = "up"  # up / sitting (since) / rising (since), as robotd's Sit
        self.since = -1e9
        self.shown: list[tuple[float, str]] = [(-1e9, "stand")]  # (robotd time, label) changes
        self.toggles: list[float] = []  # the bridge's time of every sit_toggle that arrived
        self.stops = 0

    @property
    def t(self) -> float:
        return self.wall * self.rate  # robotd's own clock

    def _show(self, label: str) -> None:
        self.shown.append((self.t + self.label_lag_s, label))

    def advance(self, dt: float) -> None:
        self.wall += dt
        if self.seat == "rising" and self.t - self.since >= RISE_S:
            self.seat = "up"
            self._show("stand")

    def label(self) -> str:
        for begun, until, label in self.labels:
            if begun <= self.t < until:
                return label
        return [label for at, label in self.shown if at <= self.t][-1]

    # ── the port the keeper holds (RobotdBodyPort's four calls) ──

    def observe(self) -> Any:
        return SimpleNamespace(
            policy=self.label(), safety=SimpleNamespace(fallen=False), robot_time_s=self.t
        )

    def stop(self) -> Any:
        self.stops += 1
        return SimpleNamespace(accepted=True)

    def dispatch_intent(self, intent: Any) -> Any:
        assert intent.method == "robot.do" and intent.params == {"skill": "sit_toggle"}
        self.toggles.append(round(self.wall, 3))
        if self.drop > 0:
            self.drop -= 1
        elif self.seat == "up":
            self.seat, self.since = "sitting", self.t
            self._show("sit")
        elif self.seat == "sitting" and self.t - self.since >= RAMP_S:
            self.seat, self.since = "rising", self.t  # a toggle after the ramp stands it up
            self._show("rise")
        return SimpleNamespace(accepted=True)  # (queued: a refusal is only in robotd's log)

    def close(self) -> None:
        pass


def lose_the_link(seat: SlowSeat, seconds: float) -> tuple[SafetyKeeper, list[str]]:
    """The link is lost at 0 and stays lost for ``seconds``; the keeper sees every frame."""
    events: list[str] = []
    keeper = SafetyKeeper(
        lambda: seat, log=lambda event, **fields: events.append(event), clock=lambda: seat.wall
    )
    keeper.engage("test: link lost")
    for _ in range(round(seconds / TICK_S)):
        assert keeper.step()
        seat.advance(TICK_S)
    return keeper, events


def test_a_seat_slow_to_show_is_not_toggled_back_up() -> None:
    """robotd took the sit, and its label went on reading ``stand`` for 3.5 s. A second toggle
    after the 2 s ramp is a stand: the duck must stay down."""
    seat = SlowSeat(label_lag_s=3.5)
    _, events = lose_the_link(seat, 15.0)
    assert seat.stops == 1
    assert seat.toggles == [0.0], seat.toggles
    assert seat.seat == "sitting" and "sat" in events


def test_a_sit_robotd_did_not_take_is_asked_again_after_the_ramp_and_a_margin() -> None:
    seat = SlowSeat(drop_first=1)
    _, events = lose_the_link(seat, 12.0)
    assert len(seat.toggles) == 2, seat.toggles
    again = seat.toggles[1]
    assert B.SIT_RAMP_S + B.SIT_MARGIN_S <= again <= B.SIT_RAMP_S + B.SIT_MARGIN_S + 2 * TICK_S
    assert again > RAMP_S + 1.5  # (well clear of the window in which a toggle stands a seated duck)
    assert seat.seat == "sitting" and "sat" in events


def test_no_second_toggle_until_the_label_has_stood_still_for_the_whole_wait() -> None:
    """The first toggle was not taken; a move ran from 2 s to 3 s (the label read its name). The
    wait starts again when the label is back to ``stand``: the posture was changing."""
    seat = SlowSeat(drop_first=1, labels=((2.0, 3.0, "kick_left"),))
    lose_the_link(seat, 12.0)
    assert len(seat.toggles) == 2, seat.toggles
    assert seat.toggles[1] >= 3.0 + B.SIT_RAMP_S + B.SIT_MARGIN_S - 1e-6
    assert seat.seat == "sitting"


def test_the_wait_runs_on_robotds_own_clock() -> None:
    """A robotd running at a third of real time (the simulator did, for a week): its 2 s ramp
    lasts 6 s here. The second toggle waits for robotd's seconds, not the bridge's."""
    seat = SlowSeat(drop_first=1, rate=1 / 3)
    lose_the_link(seat, 20.0)
    assert len(seat.toggles) == 2, seat.toggles
    assert seat.toggles[1] * seat.rate >= B.SIT_RAMP_S + B.SIT_MARGIN_S - 1e-6
    assert seat.seat == "sitting"


def test_a_duck_that_was_standing_up_is_sat_once_it_stands() -> None:
    seat = SlowSeat(labels=((0.0, 1.0, "rise"),))
    lose_the_link(seat, 8.0)
    assert len(seat.toggles) == 1 and 1.0 <= seat.toggles[0] <= 1.0 + 2 * TICK_S
    assert seat.seat == "sitting"


def test_a_duck_stood_up_again_while_the_link_is_lost_is_sat_again() -> None:
    """It sat; someone stood it up with the gamepad; the link is still lost: down again."""
    seat = SlowSeat()
    keeper, _ = lose_the_link(seat, 5.0)
    assert seat.toggles == [0.0] and seat.seat == "sitting"
    seat.seat, seat.since = "rising", seat.t  # (the pad's own toggle, after the ramp)
    seat._show("rise")
    for _ in range(60):
        assert keeper.step()
        seat.advance(TICK_S)
    assert len(seat.toggles) == 2 and seat.seat == "sitting"


# ── an address that keeps offering a wrong token ─────────────────────────────


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(B, "AUTH_FAIL_DELAY_S", 0.01)
    with tempfile.TemporaryDirectory(prefix="dm-rg-", dir=short_tmp()) as directory:
        events: list[tuple[str, dict[str, Any]]] = []
        made = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={"robotd": Channel("robotd", unix=os.path.join(directory, "none.sock"))},
            keeper=None,
            body_info={"backend": "simulator"},
            log=lambda event, **fields: events.append((event, fields)),
        )
        made.start()
        yield made, events
        made.close()


def offer_a_wrong_token(port: int) -> dict[str, Any]:
    """One handshake with a proof made without the token: the bridge's last line."""
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
        sock.settimeout(2.0)
        hello = L.read_json_line(sock)
        if "nonce" not in hello:
            return hello
        L.send_json_line(sock, {"protocol": L.PROTOCOL, "nonce": "ab" * 16, "proof": "00" * 32})
        return L.read_json_line(sock)


def dial(port: int, timeout_s: float = 1.0) -> RemoteLink:
    link = RemoteLink(f"remote://127.0.0.1:{port}", TOKEN)
    try:
        link.start(timeout_s=timeout_s)
    except BaseException:
        link.close()
        raise
    return link


def test_an_address_that_keeps_offering_a_wrong_token_is_locked_out_for_a_while(
    bridge: tuple[Bridge, list[tuple[str, dict[str, Any]]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    made, events = bridge
    monkeypatch.setattr(B, "AUTH_LOCK_S", 1.5)
    for _ in range(B.AUTH_LOCK_AFTER):
        assert offer_a_wrong_token(made.port) == {"ok": False, "error": "refused"}
    locked = offer_a_wrong_token(made.port)
    assert locked.get("locked_s", 0) >= 1 and "nonce" not in locked  # (no handshake is begun)
    # the right token from that address is not heard either while the lock holds, and says why
    with pytest.raises(ConnectionError, match="locked this computer out"):
        dial(made.port, timeout_s=0.5)
    assert not made.link_up
    assert [f["peer"] for e, f in events if e == "peer_locked"] == ["127.0.0.1"]  # (said once)
    assert len([e for e, _ in events if e == "auth_failed"]) == B.AUTH_LOCK_AFTER
    time.sleep(1.6)
    link = dial(made.port, timeout_s=3.0)  # the lock is temporary
    assert made.link_up
    link.close()


def test_a_right_token_forgets_the_wrong_ones_before_it(
    bridge: tuple[Bridge, list[tuple[str, dict[str, Any]]]],
) -> None:
    made, events = bridge
    for _ in range(B.AUTH_LOCK_AFTER - 1):
        offer_a_wrong_token(made.port)
    dial(made.port, timeout_s=3.0).close()
    for _ in range(B.AUTH_LOCK_AFTER - 1):
        assert offer_a_wrong_token(made.port) == {"ok": False, "error": "refused"}
    assert not [e for e, _ in events if e == "peer_locked"]
    dial(made.port, timeout_s=3.0).close()


def test_wrong_tokens_spread_over_a_long_time_do_not_lock(
    bridge: tuple[Bridge, list[tuple[str, dict[str, Any]]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    made, events = bridge
    monkeypatch.setattr(B, "AUTH_LOCK_WINDOW_S", 0.2)
    for _ in range(B.AUTH_LOCK_AFTER + 2):
        assert offer_a_wrong_token(made.port) == {"ok": False, "error": "refused"}
        time.sleep(0.25)
    assert not [e for e, _ in events if e == "peer_locked"]
