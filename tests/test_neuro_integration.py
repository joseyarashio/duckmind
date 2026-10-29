"""The default social/play/sound brain is one deterministic NeuroBrain composition.

Run the process-wide installer in a child process so its intentionally permanent registry cannot
change the meaning of legacy reversible skill fixtures in this pytest process.
"""

from __future__ import annotations

import subprocess
import sys


def test_default_skill_registry_and_integrated_brain_in_isolated_process() -> None:
    code = r"""
from duckmind.neuro import contract as C
from duckmind.neuro import genome as G
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.integration import IntegratedBrain, install_default_skills, make_brain

old = G.preset("explorer")
registry = install_default_skills()
features_ref = S.F
assert install_default_skills() is registry
assert S.F is features_ref

first = make_brain(old, old.seed)
assert isinstance(first, IntegratedBrain)
assert first.genome.seed == old.seed and first.genome.gd == old.gd
assert first.genome.branch == old.branch and first.genome.stage == old.stage
assert set(first.genome.gains) >= {"follow_mom", "play_ball", "theremin", "dance"}
assert first.genome.unlocked >= G._unlocked(old.gd, old.pace, old.branch, old.stage_days)

second_genome = G.preset("snuggler")
second = make_brain(second_genome, second_genome.seed)
assert S.F is features_ref
assert first.sel is not second.sel
assert first.drives is not second.drives
assert first.imprint is not second.imprint
assert first.play is not second.play
assert first.sound_skills is not second.sound_skills
assert IntegratedBrain.mro().count(NeuroBrain) == 1

# Installation after a genome already exists upgrades the new channels while retaining age,
# stage and identity at both ends of the unlock curve.
for preset_name in ("hatchling", "snuggler", "explorer", "dreamer"):
    genome = G.preset(preset_name)
    integrated = make_brain(genome, genome.seed)
    assert integrated.genome.seed == genome.seed
    assert integrated.genome.gd == genome.gd
    assert integrated.genome.stage == genome.stage
    assert integrated.genome.unlocked >= G._unlocked(
        genome.gd, genome.pace, genome.branch, genome.stage_days
    )
    assert set(integrated.genome.gains) >= {"play_ball", "carry_toy", "sunbathe", "theremin", "dance"}

# The cooperative hooks reach the one base brain tick exactly once.
calls = []
original = NeuroBrain.step
def counted(self, t_ms, frame):
    calls.append(id(self))
    return original(self, t_ms, frame)
NeuroBrain.step = counted

a = make_brain(G.preset("explorer"), 404)
b = make_brain(G.preset("explorer"), 404)
for t in range(0, 1000, 50):
    frame = C.SensorFrame(t_ms=t, body=C.BodySense(posture="stand"))
    ca, ta = a.step(t, frame)
    cb, tb = b.step(t, frame)
    assert ca.to_json() == cb.to_json()
    assert ta.behaviour == tb.behaviour
    assert set(ta.learned) >= {"social", "play", "sound"}
assert calls.count(id(a)) == 20 and calls.count(id(b)) == 20

# A new loud, classified event is still passed into the core reflex path.
frame = C.SensorFrame(
    t_ms=2000,
    body=C.BodySense(posture="stand"),
    sound=C.SoundSense(level_db=90.0, onset=1.0, kind="bang"),
)
cmds = [a.step(2000, frame)[0]]
for t in range(2050, 3000, 50):
    cmds.append(a.step(t, C.SensorFrame(t_ms=t, body=C.BodySense(posture="stand")))[0])
assert any(cmd.reflex is not None or cmd.behaviour in {"startle", "freeze", "withdraw"} for cmd in cmds)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registry_refuses_to_capture_reversible_legacy_fixtures() -> None:
    code = r"""
from duckmind.neuro import select as S
from duckmind.neuro import skill_play as P
from duckmind.neuro import skill_social_brain as B
from duckmind.neuro import skill_sound as D
from duckmind.neuro.integration import install_default_skills

for context in (P.play_installed, B.registered):
    before = tuple(S.FEATURES)
    with context():
        try:
            install_default_skills()
        except RuntimeError as error:
            assert "legacy reversible skill fixtures" in str(error)
        else:
            raise AssertionError("permanent install must not capture a reversible context")
        assert P.legacy_context_active() or B.legacy_context_active()
    assert tuple(S.FEATURES) == before
    assert not P.legacy_context_active() and not B.legacy_context_active()

D.install()
try:
    try:
        install_default_skills()
    except RuntimeError as error:
        assert "legacy reversible skill fixtures" in str(error)
    else:
        raise AssertionError("permanent install must not capture legacy sound tables")
finally:
    D.uninstall()
assert not D.installed()
registry = install_default_skills()
assert install_default_skills() is registry
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_integrated_senses_expose_only_observed_known_one_shot_policies() -> None:
    from types import SimpleNamespace

    from duckmind.world.neuro_integration import live_senses
    from duckmind.world.neuro_play_rig import play_senses

    hubs = (live_senses(seed=13), play_senses(seed=13))

    def sample(hub, policy: str):
        state = SimpleNamespace(
            policy=policy,
            applied=(0.0, 0.0, 0.0),
            down=(0.0, 0.0, -1.0),
            tilt_rad=0.0,
            fallen=False,
            limp=False,
            leg_load=None,
            gyro=(0.0, 0.0, 0.0),
            odom_pos=(0.0, 0.0, 0.0),
            odom_yaw=0.0,
            head=(0.349, 0.349, 0.0, 0.0),
            velocities=None,
            joints=None,
            t_ns=1,
        )
        doors = SimpleNamespace(
            t_ms=1,
            state=state,
            link=None,
            world={},
            tof=None,
            depth=None,
            pg=None,
            events=(),
            day_minute=600,
        )
        return hub.step(doors).body.active_skill

    for hub in hubs:
        assert sample(hub, "ground_pick") == "ground_pick"
        assert sample(hub, "ground_pick:unwind") == "ground_pick"
        assert sample(hub, "kick_left") == "kick_left"
        assert sample(hub, "kick_left:unwind") == "kick_left"
        assert sample(hub, "kick_right") == "kick_right"
        assert sample(hub, "kick_right:unwind") == "kick_right"
        for other in (
            "walk",
            "stand",
            "theremin",
            "unknown",
            "duck_ground_pick",
            "kick_left:other",
        ):
            assert sample(hub, other) is None
