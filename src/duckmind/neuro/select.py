"""Action selection among the discrete repertoire: a basal-ganglia-like competition.

Every behaviour is a channel. Its **salience** is a weighted sum of what the brain feels and
notices (drives, emotions, stimulus traces, learned values — the "cortical" input; the weights
below are the duck's wiring) × its temperament/stage gain (``genome.gains``) × what the body can
do now (posture, sleep, battery, dark) × how satiated the channel is, plus slow seeded noise.
Channels integrate their salience (leaky accumulators); the selected channel keeps a
commitment bonus (self-excitation through the thalamic loop) that fades as its bout satiates,
so a challenger must stay clearly stronger for a while to take over: no dithering.

A bout ends only when its satiation (consummatory feedback: doing it satisfies it; walking also
drains the urge to walk, investigating drains curiosity, a turn is satisfied by the angle
turned) lets a competitor win — never on a timer. Satiation recovers when the behaviour rests
(a behaviour just done is less likely again soon). A bout that is not done (its satiation has not
used up half of it) is not taken over by a rival that only crept up — "finishing what it started",
see ``FINISH_SAT`` below; something that *happens* (a surge), a reflex or a low pack still do.
A person's live answer to the duck (``set_extra(..., claims=...)``) is such an event: it claims the
body from a bout that is not done, see ``CLAIM_MIN`` below.
Reflexes interrupt from outside (``reflex.py``) and their after-effects (``appraisal.py``) are
transient inputs here (a startle feeds freeze/flee/tuck_away, then alert scanning, a shake, a
preen…).

A fresh bout also holds on to its first motor unit (``LOCK`` — the thalamo-cortical loop's
self-excitation, strongest right after the switch and fading over the unit's own time
``UNIT_S``): a challenger must be clearly stronger to cut an act before the body has rendered
it (review 2026-09-28: 18–27 act changes a minute, a preen cut before the nibbling began).

**Moments** (:data:`~duckmind.neuro.contract.MOMENTS`: a glance, a tilt, a double take, a look
down, a moment of stillness, a head shake…) are not channels of the main competition: they
compete among themselves (:meth:`Selector.moment_step`) and fire as point events *inside* the
current act when their activation crosses :data:`MOMENT_FIRE`, then rest (satiation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duckmind.i18n import join, lst, msg
from duckmind.neuro.contract import BEHAVIOURS, MOMENTS, MOMENTS_WALKING
from duckmind.neuro.genome import Genome
from duckmind.neuro.signals import Stream

FEATURES: tuple[str, ...] = (
    "bias",
    "curiosity",
    "boredom",
    "restless",
    "energy",
    "fatigue",
    "sleepy",
    "fear",
    "social",
    "comfort",
    "play",
    "hunger",
    "arousal",
    "calm",
    "happy",
    "heard",
    "sound_nodir",
    "voice",
    "name",
    "music",
    "tv",
    "startled",
    "recovering",
    "alert",
    "rhythm_slow",
    "rhythm_fast",
    "petted",
    "touch_now",
    "touched_sharp",
    "person",
    "person_near",
    "person_moving",
    "person_new",
    "greet_up",
    "settle_in",
    "thing",
    "thing_near",
    "novel",
    "remembered_thing",
    "motion",
    "looming",
    "dark",
    "light_change",
    "light_on",
    "nest",
    "on_dock",
    "just_woke",
    "stiff",
    "after_fall",
    "righted",
    "bump",
    "turn_away",
    "obstacle",
    "stumble",
    "set_down",
    "alone",
    "good_cue",
    "bad_cue",
    "double_take",
    "liked_place",
    "night",
    "legs_tired",
    "dream",
    "preen_next",
    "charging",
    "groom",
    "peckish",
    "vigil",
    "called_unseen",
    "called_seen",
    "switching",
    "walked",
    "rested",
    "reached",
    "place_new",
    "cornered",
    # (dock:build 2026-09-28) a docking under way (its own channel holds on through the
    # manoeuvre), and sleepy away from its nest (the pull home)
    "docking",
    "home_pull",
    "battery_low",
    "homing",
    "on_contacts",
)
F = {name: index for index, name in enumerate(FEATURES)}

FEATURE_ZH = {
    "docking": msg("neuro.select.pending.1e4a75"),
    "home_pull": msg("neuro.why.home_pull"),
    "battery_low": msg("neuro.why.battery_low"),
    "homing": msg("neuro.why.homing"),
    "on_contacts": msg("neuro.why.on_contacts"),
    "cornered": msg("neuro.why.cornered"),
    "switching": msg("neuro.why.switching"),
    "walked": msg("neuro.why.walked"),
    "rested": msg("neuro.why.rested"),
    "reached": msg("neuro.why.reached"),
    "place_new": msg("neuro.why.place_new"),
    "called_unseen": msg("neuro.why.called_unseen"),
    "called_seen": msg("neuro.why.called_seen"),
    "groom": msg("neuro.why.groom"),
    "peckish": msg("neuro.why.peckish"),
    "vigil": msg("neuro.why.vigil"),
    "bias": msg("neuro.why.bias"),
    "curiosity": msg("neuro.why.curiosity"),
    "boredom": msg("neuro.why.boredom"),
    "restless": msg("neuro.why.restless"),
    "energy": msg("neuro.why.energy"),
    "fatigue": msg("neuro.why.fatigue"),
    "sleepy": msg("neuro.why.sleepy"),
    "fear": msg("neuro.why.fear"),
    "social": msg("neuro.why.social"),
    "comfort": msg("neuro.why.comfort"),
    "play": msg("neuro.why.play"),
    "hunger": msg("neuro.why.hunger"),
    "arousal": msg("neuro.why.arousal"),
    "calm": msg("neuro.why.calm"),
    "happy": msg("neuro.why.happy"),
    "heard": msg("neuro.why.heard"),
    "sound_nodir": msg("neuro.why.sound_nodir"),
    "voice": msg("neuro.why.voice"),
    "name": msg("neuro.why.name"),
    "music": msg("neuro.why.music"),
    "tv": msg("neuro.why.tv"),
    "startled": msg("neuro.why.startled"),
    "recovering": msg("neuro.why.recovering"),
    "rhythm_slow": msg("neuro.why.rhythm_slow"),
    "rhythm_fast": msg("neuro.why.rhythm_fast"),
    "alert": msg("neuro.why.alert"),
    "petted": msg("neuro.why.petted"),
    "touch_now": msg("neuro.why.touch_now"),
    "touched_sharp": msg("neuro.why.touched_sharp"),
    "person": msg("neuro.why.person"),
    "person_near": msg("neuro.why.person_near"),
    "person_moving": msg("neuro.why.person_moving"),
    "person_new": msg("neuro.why.person_new"),
    "greet_up": msg("neuro.why.greet_up"),
    "settle_in": msg("neuro.why.settle_in"),
    "thing": msg("neuro.why.thing"),
    "thing_near": msg("neuro.why.thing_near"),
    "novel": msg("neuro.why.novel"),
    "remembered_thing": msg("neuro.why.remembered_thing"),
    "motion": msg("neuro.why.motion"),
    "looming": msg("neuro.why.looming"),
    "dark": msg("neuro.why.dark"),
    "light_change": msg("neuro.why.light_change"),
    "light_on": msg("neuro.why.light_on"),
    "nest": msg("neuro.why.nest"),
    "on_dock": msg("neuro.why.on_dock"),
    "just_woke": msg("neuro.why.just_woke"),
    "stiff": msg("neuro.why.stiff"),
    "after_fall": msg("neuro.why.after_fall"),
    "righted": msg("neuro.why.righted"),
    "bump": msg("neuro.why.bump"),
    "turn_away": msg("neuro.why.turn_away"),
    "obstacle": msg("neuro.why.obstacle"),
    "stumble": msg("neuro.why.stumble"),
    "set_down": msg("neuro.why.set_down"),
    "alone": msg("neuro.why.alone"),
    "good_cue": msg("neuro.why.good_cue"),
    "bad_cue": msg("neuro.why.bad_cue"),
    "double_take": msg("neuro.why.double_take"),
    "liked_place": msg("neuro.why.liked_place"),
    "night": msg("neuro.why.night"),
    "legs_tired": msg("neuro.why.legs_tired"),
    "dream": msg("neuro.why.dream"),
    "preen_next": msg("neuro.why.preen_next"),
    "charging": msg("neuro.why.charging"),
}

# The wiring: behaviour → {feature: weight}. Negative weights inhibit.
W: dict[str, dict[str, float]] = {
    "look_around": {
        "curiosity": 0.3,
        "boredom": 0.22,
        "called_unseen": 0.8,  # someone called and it cannot see them: it looks for them
        "sound_nodir": 0.6,
        "heard": 0.15,
        "just_woke": 0.3,
        "set_down": 0.8,
        "light_change": 0.4,
        "bias": 0.08,
        "recovering": 0.45,
    },
    "perk_tilt": {
        "sound_nodir": 0.45,
        "voice": 0.3,
        "name": 0.5,
        "novel": 0.3,
        "curiosity": 0.12,
        "good_cue": 0.3,
    },
    # (fix 2026-09-28, pet:etho F1/F4 comfort grammar: the comfort acts have their own causes in
    # an ordinary day — a stretch after resting, a shake after a walk or being handled, a head
    # shake as the intention movement before a change of act, a proud pose on arriving, a double
    # take when a known thing reappears, a look down in a new place)
    "stretch_yawn": {"stiff": 1.1, "sleepy": 0.2, "fatigue": 0.1, "rested": 0.9},
    # (integration 2026-09-28: preening came mostly from being calm and comfortable, 19–29 %
    # of the day; pet:etho's comfort budget is 10–20 % — now the itch (groom), a disturbance
    # and being handled drive it, calm only a little)
    "preen": {
        "calm": 0.12,
        "comfort": 0.1,
        "startled": 0.22,
        "petted": 0.2,
        "bias": 0.03,
        "boredom": 0.04,
        "preen_next": 1.2,
        "recovering": 0.15,
        "set_down": 0.2,
        "sleepy": 0.12,
    },
    "happy_sway": {"happy": 0.42, "music": 0.7, "play": 0.22},
    "tuck_away": {
        "fear": 0.4,
        "touched_sharp": 0.5,
        "startled": 0.3,
        "bad_cue": 0.3,
        "person_near": 0.15,  # (a shy one, when the family comes very close: a little tuck)
    },
    "nap": {
        "sleepy": 1.3,
        "comfort": 0.12,
        "on_dock": 0.15,
        "night": 0.3,
        "fear": -0.6,
        "restless": -0.2,
        "voice": -0.1,
        "charging": 0.45,
        "recovering": -0.5,
        # (dock:build) away from its nest a sleepy duck goes home first (evening headless, 8 per
        # home: at −0.5 two of three ducks in default_home napped where they stood)
        "home_pull": -0.9,
    },
    "wake_ritual": {"just_woke": 1.05},
    "nod": {"voice": 0.42, "person_near": 0.25, "social": 0.12, "good_cue": 0.2},
    "peekaboo": {"person_near": 0.42, "play": 0.5, "petted": 0.15},
    "call_response": {"voice": 0.55, "name": 0.55, "social": 0.22},
    "snuggle": {
        "touch_now": 0.8,
        "petted": 0.5,
        "social": 0.28,
        "person_near": 0.32,
        "comfort": 0.14,
    },
    "little_dance": {"music": 0.9, "happy": 0.38, "play": 0.33},
    "not_now": {"touched_sharp": 0.6, "sleepy": 0.2, "touch_now": 0.25, "bad_cue": 0.3},
    "little_song": {"happy": 0.42, "music": 0.35, "voice": 0.2, "social": 0.12, "bias": 0.05},
    "self_duet": {"happy": 0.28, "boredom": 0.3, "alone": 0.25},
    "survey": {
        "curiosity": 0.35,
        "boredom": 0.28,
        "set_down": 0.3,
        "bias": 0.05,
        "recovering": 0.3,
    },
    "peekaboo_master": {"person_near": 0.45, "play": 0.6},
    "melt": {"touch_now": 0.8, "settle_in": 0.5, "petted": 0.6, "sleepy": 0.35, "comfort": 0.3},
    "nuzzle_tilt": {"touch_now": 0.5, "person_near": 0.45, "social": 0.33},
    "moon_gaze": {"calm": 0.32, "night": 0.22, "boredom": 0.2, "sleepy": 0.12, "light_on": 0.2},
    "sleep_talk": {"dream": 0.9, "social": 0.15},
    "stand_up": {
        "restless": 0.55,
        "curiosity": 0.18,
        "person_new": 0.2,
        "greet_up": 0.6,
        "alert": 0.8,
        "name": 0.3,
        "good_cue": 0.3,
        "fear": -0.2,
        "sleepy": -0.4,
        "legs_tired": -0.5,
    },
    "sit_down": {
        "cornered": 0.5,
        "fatigue": 0.45,
        "sleepy": 0.45,
        "comfort": 0.15,
        "legs_tired": 0.75,
        "settle_in": 0.35,
        "restless": -0.45,
        "curiosity": -0.1,
    },
    "wander": {
        "cornered": -0.7,
        "restless": 0.95,
        "curiosity": 0.3,
        "boredom": 0.28,
        "social": 0.14,
        "alone": 0.2,  # a duckling left alone goes looking for its family (and calls: C2)
        "liked_place": 0.15,
        "fear": -0.35,
        "sleepy": -0.45,
        "hunger": -0.3,
        "dark": -0.4,
        "legs_tired": -0.2,
        "charging": -0.6,
        "rhythm_fast": 0.35,
    },
    "turn_around": {
        "turn_away": 0.8,
        "bump": 0.45,
        "called_unseen": 1.6,  # called and nobody in sight: it turns round to look
        "sound_nodir": 0.3,
        "boredom": 0.12,
        "restless": 0.1,
        "obstacle": 0.3,
    },
    "peek_down": {
        "stumble": 0.9,
        "after_fall": 0.4,
        "thing_near": 0.18,
        "bias": 0.03,
        "place_new": 0.45,
    },
    "glance": {"heard": 0.32, "motion": 0.45, "bias": 0.05},
    "tilt": {"novel": 0.33, "voice": 0.22, "name": 0.25, "sound_nodir": 0.2},
    "dabble": {"play": 0.28, "happy": 0.22, "boredom": 0.28, "thing_near": 0.15},
    "double_take": {"double_take": 1.0, "motion": 0.22, "novel": 0.22},
    "proud_puff": {"happy": 0.3, "righted": 0.7, "good_cue": 0.2, "reached": 0.55},
    "still": {
        "startled": 0.35,
        "sound_nodir": 0.2,
        "heard": 0.25,  # a sound out of nowhere: a moment's stillness, listening
        "bias": 0.02,
        "rhythm_slow": 0.4,
    },
    "freeze": {"fear": 0.5, "looming": 0.6, "startled": 0.4, "rhythm_slow": 0.9},
    # the tricks (neuro.tricks): no wiring of their own — a bow, a spin, a lie-low are taken up when a
    # person's command claims them (neuro.command_brain), never by a drift of the duck's own wants
    "bow": {},
    "lie_low": {},
    "spin": {},
    "flip": {},
    "glance_away": {"person_near": 0.18, "bad_cue": 0.3, "boredom": 0.1, "touch_now": 0.1},
    "peck": {"thing_near": 0.45, "curiosity": 0.22, "boredom": 0.28, "bias": 0.05},
    "forage": {
        "cornered": -0.6,
        "boredom": 0.38,
        "curiosity": 0.28,
        "restless": 0.35,
        "energy": 0.1,
        "fear": -0.3,
        "sleepy": -0.3,
        "charging": -0.5,
    },
    "nibble": {"thing_near": 0.9, "novel": 0.5, "curiosity": 0.2},
    "head_shake": {
        "righted": 0.7,
        "petted": 0.18,
        "startled": 0.15,
        "stiff": 0.2,
        "bias": 0.02,
        "recovering": 0.25,
        "set_down": 0.3,
    },
    "body_shake": {
        "righted": 0.6,
        "after_fall": 0.3,
        "petted": 0.22,
        "stiff": 0.2,
        "recovering": 0.2,
        "set_down": 0.35,
    },
    "investigate": {
        # A newly seen tracked thing must be able to interrupt an ordinary wander.
        # This raises only object-directed salience; it does not increase wandering,
        # weaken reflexes, or alter temperament/commitment parameters.
        "novel": 0.9,
        "thing": 0.3,
        "remembered_thing": 0.5,
        "curiosity": 0.42,
        "fear": -0.4,
        "sleepy": -0.3,
    },
    # (fix 2026-09-28: the family barely registered — a social act followed 1 of 28 family
    # cues; now its name or a hello brings it over, the more social the duck the more surely)
    "follow": {
        "person_moving": 0.8,
        "social": 0.5,
        "person": 0.22,
        "called_seen": 0.95,  # its name: it goes to them …
        "called_unseen": 0.45,  # … or goes looking for them
        "person_near": -0.15,
        "fear": -0.3,
    },
    "greet": {
        "person_new": 0.9,
        "called_seen": 0.7,
        "voice": 0.12,
        "social": 0.3,
        "good_cue": 0.35,
        "rhythm_fast": 0.3,
    },
    "call_out": {
        "cornered": 0.8,  # stuck where no arc gets it out: it calls for its family
        "alone": 0.55,
        "social": 0.35,
        "dark": 0.12,
        "bias": -0.12,
        "rhythm_fast": 0.6,
        "name": 0.3,
    },
    "watch": {
        "motion": 0.55,
        "person": 0.22,
        "tv": 0.45,
        "music": 0.1,
        "calm": 0.14,
        "recovering": 0.2,
    },
    "retreat": {
        "fear": 0.5,
        "touched_sharp": 0.45,
        "obstacle": 0.2,
        "bad_cue": 0.35,
        "looming": 0.35,
    },
    "flee": {"fear": 0.75, "looming": 0.75, "startled": 0.45},
    "doze": {
        "sleepy": 0.95,
        "comfort": 0.2,
        "calm": 0.18,
        "fear": -0.5,
        "restless": -0.2,
        "charging": 0.4,
        "recovering": -0.4,
        "home_pull": -0.5,  # (dock:build)
    },
    "rest_sit": {
        "fatigue": 0.5,
        "calm": 0.22,
        "comfort": 0.18,
        "restless": -0.3,
        "bias": 0.05,
        "charging": 0.9,
        "recovering": -0.3,
    },
    "return_dock": {
        "hunger": 1.25,
        "night": 0.3,
        "fear": 0.15,
        "dark": 0.22,
        "nest": 0.15,
        # (dock:build) only its contacts closed (PG) end the wish to go home: in its dock
        # without contact it still wants to be charged — it sits down again (the re-seat)
        "on_contacts": -1.2,
        # (dock:build) it sees a docking through, and goes home to sleep when sleepy
        "docking": 0.8,
        "home_pull": 1.1,
        # … and home it goes when the pack runs low, whatever its temperament (a homeostatic
        # urgency: the explorer's love of wandering must not strand it)
        "homing": 2.5,
    },
    "zoomies": {"play": 0.5, "happy": 0.32, "arousal": 0.22, "energy": 0.18, "fear": -0.4},
}
# Consummatory time scale per behaviour (s): how long doing it takes to satisfy it.
# The maintenance needs (integration 2026-09-28, drives.py: groom, peckish, vigil): each is
# satisfied by its own acts, and walking on gives way to them — a duck walks, stops to look
# around, pecks, preens, shakes, walks again (pet:etho D1/E1/F1).
for _behaviour, _extra in {
    "preen": {"groom": 1.0},
    "body_shake": {"groom": 0.45, "walked": 0.45},
    "head_shake": {"groom": 0.12, "switching": 0.55},
    "stretch_yawn": {"groom": 0.25},
    "peck": {"peckish": 0.8},
    "forage": {"peckish": 0.5},
    "nibble": {"peckish": 0.3},
    "dabble": {"peckish": 0.35},
    "look_around": {"vigil": 0.8},
    "survey": {"vigil": 0.3},
    "glance": {"vigil": 0.35},
    "perk_tilt": {"vigil": 0.2},
    "watch": {"vigil": 0.2},
    # (walking builds the urge to stop and look around; the itch to preen or peck competes
    # through preen/peck/forage themselves, not by holding the walk back)
    "wander": {"vigil": -0.35},
    "zoomies": {"vigil": -0.3},
}.items():
    W[_behaviour] = {**W[_behaviour], **_extra}
# (dock:build 2026-09-28) charging on its contacts it stays until charged (Duck Core's nest
# hysteresis, NEST.leave): the walks that would take it off (wander −0.6 and forage −0.5 had it)
for _behaviour, _w in {
    "investigate": -0.6,
    "follow": -0.4,
    "zoomies": -0.8,
    "turn_around": -0.4,
}.items():
    W[_behaviour] = {**W[_behaviour], "charging": W[_behaviour].get("charging", 0.0) + _w}
# (dock:build 2026-09-28) the pack running low damps the walks that lead away from home
for _behaviour, _w in {
    "wander": -1.0,
    "forage": -0.8,
    "investigate": -0.6,
    "zoomies": -1.0,
    "follow": -0.5,
    "turn_around": -0.4,
}.items():
    W[_behaviour] = {**W[_behaviour], "battery_low": _w}

# (review 2026-09-28: comfort and rest acts ended before they became recognisable — preen median
# 1.8 s where pet:etho has bouts of 10 s–5 min, rest in long bouts; the acts tied to a need now
# end mostly when the need is used up (brain._consumed), these are the slow backstops)
SATIATE_S: dict[str, float] = {
    "look_around": 12.0,
    "perk_tilt": 2.5,
    "stretch_yawn": 6.0,
    "preen": 90.0,
    "happy_sway": 8.0,
    "tuck_away": 6.0,
    "nap": 900.0,
    "wake_ritual": 8.0,
    "nod": 4.0,
    "peekaboo": 10.0,
    "call_response": 12.0,
    "snuggle": 40.0,
    "little_dance": 10.0,
    "not_now": 3.0,
    "little_song": 10.0,
    "self_duet": 10.0,
    "survey": 14.0,
    "peekaboo_master": 12.0,
    "melt": 90.0,
    "nuzzle_tilt": 6.0,
    "moon_gaze": 20.0,
    "sleep_talk": 2.0,
    "stand_up": 2.2,
    "sit_down": 2.2,
    "wander": 90.0,
    "turn_around": 30.0,
    "peek_down": 2.0,
    "glance": 1.5,
    "tilt": 1.8,
    "dabble": 8.0,
    "double_take": 2.0,
    "proud_puff": 3.0,
    "still": 2.0,
    "freeze": 5.0,
    "bow": 4.0,
    "lie_low": 5.0,
    "spin": 6.0,
    "flip": 6.0,
    "glance_away": 2.0,
    "peck": 12.0,
    "forage": 60.0,
    "nibble": 12.0,
    "head_shake": 1.5,
    "body_shake": 4.0,
    "investigate": 40.0,
    "follow": 45.0,
    "greet": 6.0,
    "call_out": 10.0,
    "watch": 20.0,
    "retreat": 8.0,
    "flee": 6.0,
    "doze": 300.0,
    "rest_sit": 150.0,
    # going home ends on its contacts (brain._consumed); (dock:build, live check 2026-09-28: the
    # slow backstop gave a homing duck up mid-docking after ~2 min of trying — while its docking
    # program has the legs the backstop now waits, brain.step; 600 s here instead made the
    # docked duck restless in its dock, 4 falls in 30 forced stand/sits and exits)
    "return_dock": 240.0,
    "zoomies": 10.0,
}
# A fresh bout's hold on its first motor unit: + LOCK × exp(−age / UNIT_S) on its channel.
LOCK = 0.5
# ── Finishing what it started ──────────────────────────────────────────────────
# (M5, owner 2026-10-04: 「行為常常還沒做完就換成別的」. The machine check found that only 54–75 % of
# the bouts a stimulus started ended with half their satiation used, the rest taken over by a rival
# that had only crept up — and a fear act reopened three times in ten seconds.)
# A bout is *done* once its satiation has used up FINISH_SAT of it. Until then a challenger has to
# beat the most the bout has been wanted (``_peak``: a smoothed peak of its salience before its own
# satiation, what it would want if it were not tired of itself) by a margin, FINISH_MARGIN, that holds
# until the bout is done and then lapses over FINISH_FADE, and that grows with the bout's progress
# (FINISH_RISE: the nearer to done, the harder to cut) — so a slowly drifting rival (noise, a drive
# that rose a little) does not take a half-done bout, and a bout ends where its own satiation ends it.
# What still takes a bout at once: a reflex that halts the body (``interrupted``), a released
# commitment (a new thing to look at), the pack running low (HOMING_RELEASE), a challenger that has
# *surged* — its salience is SURGE above where it was a moment ago (``_base``, a slow trace) and where
# it was when this bout began (``_ref``): something happened, a thing appeared, it was touched, its
# name was called (the surge needed grows with the bout's progress too: SURGE_RISE) — a challenger
# that is urgent business (URGENT_FEATURES: a threat builds up slowly too) when the bout is not, and
# a challenger stronger than the margin itself.
COMMIT_BOOST = 0.5  # what ``commit_next`` adds to the strongest act's activation
FINISH_SAT = 0.5
FINISH_FADE = 0.15
FINISH_MARGIN = 0.6
FINISH_RISE = 0.5
WANT_TAU_S = 3.0
SURGE = 0.35
SURGE_RISE = 2.0
SURGE_TAU_S = 4.0
HOMING_RELEASE = 0.15
# What an *urgent* challenger is made of: threat and emergency inputs. A channel whose present input is
# mostly these (``URGENT_SHARE``) is urgent business — it takes a bout that is not done, unless the bout
# is itself urgent business (the fear acts do not take each other over half-done: flee, freeze, retreat
# and tuck away in turn, three times in ten seconds, was the shy duck's reopening).
URGENT_FEATURES = (
    "fear",
    "startled",
    "looming",
    "touched_sharp",
    "after_fall",
    "rhythm_slow",
)
URGENT_SHARE = 0.5
# Two acts are not made to wait for their satiation: going home ends on arriving (or a failed try) —
# its own program and ``consumed`` end it, and a sleepy duck in a room without its dock gives the
# search up for its sleeping circle (test_neuro_skill_circle) as before — and answering a call lasts
# as long as the call does, ended by what else wants the duck as before (a person who speaks over the
# call stops it and the duck does not call again on its own, live voice LV2's barge-in). The acts of
# sleep (``spec.sleep``: nap, doze, sleep_talk) are not made to wait either: with the rule the longest
# nap of two hours went from 88 s to 762 s; sleep is left by waking (``wakefulness``, a sound, the
# morning), and a duck that sleeps on through the day walks too little.
UNFINISHED_OK = frozenset({"return_dock", "call_response"})
# ── A person's answer claims the body ───────────────────────────────────────────
# (live check 2026-10-04: with a speaker who stands in view the cloud's replies to an unnamed person
# voiced 2 of 6 in the combined check and 2 of 11 / 1 of 12 for the explorer and the snuggler on the
# tree before this change, 29 of 32 before the speaker had a body. A person the family has not named
# gets none of the bridge's urge to answer — LV4's reserve, kept — so the answer stands at 0.1–1.0
# (the lazy and the shy 0.1–0.25, the explorer 1.0) against a walk that holds the body at 2.0–2.6:
# the plain competition never gave it the body, and the finishing rule's "urgent" only lifted the
# margin, never the hold.) What a person says to the duck is an event, not a drift: an answer the
# bridge puts on a channel (``claims``: a live reply whose input is at least CLAIM_MIN: any input
# the bridge has not taken away — its fear and sleepiness ramps take it to nothing for a frightened
# or a drowsy duck; a lazy duck's reluctance toward someone it does not know is a small input, 0.1,
# not nothing; 0.15 left four of the dreamer's ten proposals unanswered in the live check, their
# inputs 0.09–0.15) takes the body from the bout in hand, not done or done, and
# keeps it until the answer is given or lapses. It never takes it from what is urgent business (a
# fear act, the fear acts' own rule), from sleep (the plain competition: ``wakefulness``), from a
# duck that is afraid (fear ≥ CLAIM_FEAR_MAX, the bridge's own mute), from going home or a docking, or
# from a skill's act in the middle of something it would spoil by stopping (BUSY_FEATURES: waited for
# up to BUSY_WAIT_S, never by a one-shot claim); and a reflex never waits for it.
CLAIM_MIN = 0.02
CLAIM_FEAR_MAX = 0.5
# a docking is never broken off; a skill's busy phase (a kick, a grab: ``ball_busy``, ``toy_busy``) is
# waited for, but only as long as such a phase lasts (live, tree ea6aa77: two of 49 replies lost to a
# line-up on the ball that held ``ball_busy`` for some 15 s, longer than a reply lives); a one-shot claim
# (``claim_next``) never breaks one off
BUSY_FEATURES = ("docking", "ball_busy", "toy_busy")
BUSY_ABSOLUTE = ("docking",)
BUSY_WAIT_S = 2.0
UNIT_S_MAX = 6.0
# A moment fires when its activation crosses this (then its satiation rests it).
MOMENT_FIRE = 0.2
MOMENT_REST_S = 25.0  # its satiation recovers over about this (× its own SATIATE_S / 2)
SLEEP = frozenset(b for b, spec in BEHAVIOURS.items() if spec.sleep)
REFLEX_ONLY = frozenset(b for b, spec in BEHAVIOURS.items() if spec.family == "reflex")
# The channels with no wiring of their own (the tricks a person can teach): a command's claim is
# all that ever gives them salience. They are inert for the duck's own day — no noise on them (and
# none drawn for them: the select stream gives every other channel exactly the draws it had before
# they were there), so a duck that is never told is the duck it was.
TOLD_ONLY = frozenset(b for b, w in W.items() if not w)
PLAY = frozenset(
    {
        "happy_sway",
        "little_dance",
        "zoomies",
        "peekaboo",
        "peekaboo_master",
        "dabble",
        "self_duet",
        "little_song",
    }
)
LOCOMOTES = frozenset(b for b, spec in BEHAVIOURS.items() if spec.locomotes)
INVESTIGATIVE = frozenset(
    {"investigate", "nibble", "peck", "forage", "survey", "look_around", "watch"}
)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(slots=True)
class Context:
    """What the body can do now (gates, not motivations)."""

    sitting: bool
    standing: bool
    asleep: bool
    can_walk: bool  # floor, battery ≥ 30 %, not held/fallen, walking unlocked
    wakefulness: float  # 0 deep asleep … 1 awake (how much awake channels get through)
    posture_tired: float = 0.0  # 0..1 it changed posture a moment ago (a leaky trace): the
    # next change costs more — a duck does not sit down and stand up again every minute
    recharging: bool = False  # at the dock, below the genome's existing leave level
    charge_escape: bool = False  # a real fright/reflex, not noise selecting a defensive name
    nest_rest: bool = False  # its search for its nest ended without it: going home is let go for now (brain.NEST_REST_S)


class Selector:
    """The competition. ``step`` returns the winning behaviour (and whether a new bout began)."""

    __slots__ = (
        "_g",
        "names",
        "_extra",
        "_claim",
        "_cmd_extra",
        "_cmd_claim",
        "_claim_age",
        "_once",
        "_forced",
        "_weights",
        "_gain",
        "_sat_rate",
        "_rec_tau",
        "_unit_s",
        "act",
        "sal",
        "sat",
        "_noise",
        "_noise_t",
        "_told_only",
        "_stream",
        "current",
        "bout",
        "bout_age",
        "_contrib",
        "_base",
        "_ref",
        "_pre",
        "_urgent",
        "_want",
        "_peak",
        "m_names",
        "_m_weights",
        "_m_gain",
        "_m_rec",
        "m_act",
        "m_sal",
        "m_sat",
        "_m_noise",
    )

    def __init__(self, genome: Genome, seed: int) -> None:
        self._g = genome
        self.names: list[str] = [
            b
            for b in BEHAVIOURS
            if b not in REFLEX_ONLY and b not in MOMENTS and b in genome.unlocked
        ]
        self.m_names: list[str] = [b for b in BEHAVIOURS if b in MOMENTS and b in genome.unlocked]
        self._m_weights: list[list[tuple[int, float]]] = [
            [(F[f], w) for f, w in W[b].items()] for b in self.m_names
        ]
        self._m_gain = [genome.gains.get(b, 1.0) for b in self.m_names]
        self._m_rec = [max(8.0, MOMENT_REST_S * SATIATE_S[b] / 2.0) for b in self.m_names]
        self.m_act = [0.0] * len(self.m_names)
        self.m_sal = [0.0] * len(self.m_names)
        self.m_sat = [0.0] * len(self.m_names)
        self._weights: list[list[tuple[int, float]]] = [
            [(F[f], w) for f, w in W[b].items()] for b in self.names
        ]
        self._gain = [genome.gains.get(b, 1.0) for b in self.names]
        scale = genome.p.satiation_scale
        self._sat_rate = [scale / SATIATE_S[b] for b in self.names]
        self._rec_tau = [min(900.0, max(20.0, 4.0 * SATIATE_S[b])) for b in self.names]
        self._unit_s = [min(UNIT_S_MAX, max(2.5, 0.25 * SATIATE_S[b])) for b in self.names]
        n = len(self.names)
        self._extra = [0.0] * n
        self._claim = [0.0] * n
        # what a person told the duck to do (``set_command``): its own channel, so the live bridge's
        # ``set_extra`` (which replaces ``_extra``/``_claim`` every tick) never erases it
        self._cmd_extra = [0.0] * n
        self._cmd_claim = [0.0] * n
        self._once: int | None = None  # a one-shot claim for the next tick (``claim_next``)
        self._forced: int | None = None  # a one-shot switch for the next tick (``commit_next``)
        self._claim_age = 0.0  # how long an answer has been waiting for the body, s
        self.act = [0.0] * n
        self.sal = [0.0] * n
        self.sat = [0.0] * n
        self._stream = Stream(seed, "select")
        self._told_only = [b in TOLD_ONLY for b in self.names]
        self._noise = [
            0.0 if told else genome.p.sel_noise * self._stream.normal() for told in self._told_only
        ]
        self._m_noise = [genome.p.sel_noise * self._stream.normal() for _ in self.m_names]
        self._noise_t = 0.0
        start = self.names.index("rest_sit") if "rest_sit" in self.names else 0
        self.current = start
        self.act[start] = 0.3
        self.bout = 0
        self.bout_age = 0.0
        self._contrib: list[float] = [0.0] * len(FEATURES)
        self._base = [0.0] * n  # a slow trace of each channel's salience (the surge's yardstick)
        self._ref: list[float] = [0.0] * n  # the salience of each channel when this bout began
        self._pre = [0.0] * n  # each channel's salience before its satiation (what it wants)
        urgent = frozenset(F[f] for f in URGENT_FEATURES)
        # per channel: the weights of its input that are urgent business, and all its positive weights
        self._urgent: list[tuple[list[tuple[int, float]], list[tuple[int, float]]]] = [
            (
                [(f, w) for f, w in ws if f in urgent and w > 0.0],
                [(f, w) for f, w in ws if w > 0.0],
            )
            for ws in self._weights
        ]
        self._want = 0.0  # the current bout's want (its act without its satiation), smoothed
        self._peak = 0.0  # … and the most it reached in this bout

    def index(self, behaviour: str) -> int:
        return self.names.index(behaviour)

    def adopt_state(self, old: Selector) -> None:
        """(Growth G3, a regrow.) This selector is built from the duck's genome *now*; ``old`` is
        the one that was running, built from the genome before. Its state goes across by behaviour
        name: how active, salient and satiated each act was, the noise it was in, which act holds
        the body and how long its bout has run. An act the new genome adds (unlocked since) starts
        as a fresh selector has it; an act it drops (a branch's signature that is no longer its)
        is forgotten. The noise stream goes on from where it was."""
        new_index = {name: i for i, name in enumerate(self.names)}
        for i, name in enumerate(old.names):
            j = new_index.get(name)
            if j is None:
                continue
            for attr in (
                "act",
                "sal",
                "sat",
                "_noise",
                "_base",
                "_ref",
                "_pre",
                "_extra",
                "_claim",
                "_cmd_extra",
                "_cmd_claim",
            ):
                getattr(self, attr)[j] = getattr(old, attr)[i]
        moment_index = {name: i for i, name in enumerate(self.m_names)}
        for i, name in enumerate(old.m_names):
            j = moment_index.get(name)
            if j is None:
                continue
            for attr in ("m_act", "m_sal", "m_sat", "_m_noise"):
                getattr(self, attr)[j] = getattr(old, attr)[i]
        held = old.names[old.current]
        if held in new_index:
            self.current = new_index[held]
        for attr in ("bout", "bout_age", "_claim_age", "_noise_t", "_want", "_peak"):
            setattr(self, attr, getattr(old, attr))
        self._contrib = list(old._contrib)
        for attr in ("_once", "_forced"):
            index = getattr(old, attr)
            pending = None if index is None else old.names[index]
            setattr(self, attr, None if pending is None else new_index.get(pending))
        self._stream = old._stream

    def set_extra(
        self, inputs: dict[str, float], *, claims: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Outside candidates (live voice, LV2: a proposal of the cloud brain that the grounding
        gate let through): extra input on a channel, **in the units of a feature × its weight**,
        replacing last tick's. It is summed with the wired inputs *before* the temperament gain, the
        body's gates, the satiation and the noise, so it competes like any other reason: a duck that
        is afraid, asleep, busy or tired of the act can lose it, and a reflex (which never waits for
        this) still takes the body first. A name that is not a selectable channel (not unlocked at
        this growth stage, a reflex-only act) is ignored and returned in the result (``{name:
        value}`` of what was not applied); an empty mapping clears everything.

        ``claims`` (the same units; replaces last tick's too): the part of ``inputs`` that is a person's
        answer to the duck, an event the duck is to answer and not a drift of its own wants (its own
        wishes, which the bridge asks for by itself, are not claims). A claim of at least
        :data:`CLAIM_MIN` takes the body from the bout in hand (see there for what it never takes
        it from)."""
        names = self.names
        ignored: dict[str, float] = {}
        extra = [0.0] * len(names)
        for behaviour, value in inputs.items():
            if behaviour in names and value > 0.0:
                extra[names.index(behaviour)] = float(value)
            elif value > 0.0:
                ignored[behaviour] = float(value)
        self._extra = extra
        claim = [0.0] * len(names)
        for behaviour, value in (claims or {}).items():
            if behaviour in names and value > 0.0:
                claim[names.index(behaviour)] = float(value)
        self._claim = claim
        return ignored

    def set_command(
        self, inputs: dict[str, float], *, claims: dict[str, float] | None = None
    ) -> dict[str, float]:
        """What a person told the duck to do (neuro.command_brain): the same units and the same rules
        as :meth:`set_extra` (a claim of at least :data:`CLAIM_MIN` takes the body on the terms of
        a person's answer: never from urgent business, sleep, a duck that is afraid, going home or a
        docking, and a reflex never waits for it), on a channel of its own that **adds to** whatever
        ``set_extra`` holds instead of replacing it. Replaces last tick's command; an empty mapping
        clears it. Returns what was not applied (a name that is not a selectable channel)."""
        names = self.names
        ignored: dict[str, float] = {}
        extra = [0.0] * len(names)
        for behaviour, value in inputs.items():
            if behaviour in names and value > 0.0:
                extra[names.index(behaviour)] = float(value)
            elif value > 0.0:
                ignored[behaviour] = float(value)
        self._cmd_extra = extra
        claim = [0.0] * len(names)
        for behaviour, value in (claims or {}).items():
            if behaviour in names and value > 0.0:
                claim[names.index(behaviour)] = float(value)
        self._cmd_claim = claim
        return ignored

    def claim_next(self, behaviour: str) -> None:
        """A one-shot claim for the next tick: ``behaviour`` takes the body from the bout in hand, on
        the same terms as a person's answer (see CLAIM_MIN: never from urgent business, sleep, a duck
        that is afraid, going home or a docking, a skill in the middle of something). The brain makes
        it when a thing has just come into view, for the curiosity act: something new is a surge for
        it (a rival that happens), and the bout it starts is then held on to by the finishing rule
        like any other."""
        self._once = self.names.index(behaviour) if behaviour in self.names else None

    def commit_next(self, behaviour: str) -> None:
        """A one-shot switch for the next tick: ``behaviour`` takes the body from whatever bout is in
        hand (its gate still applies: a walk it cannot make is not made). The brain makes it for a
        flight the fright has decided — a startle whose threat has not gone (2026-10-04, owner:
        「嚇一跳也是全速逃離」) — which the competition would only reach after seconds of freezing."""
        self._forced = self.names.index(behaviour) if behaviour in self.names else None

    # ── the step ─────────────────────────────────────────────────────────────

    def step(
        self,
        dt_s: float,
        feat: list[float],
        ctx: Context,
        interrupted: bool,
        consumed: float,
        *,
        release_commitment: bool = False,
        walking_to: bool = False,
    ) -> tuple[str, bool]:
        """One tick. ``consumed``: extra satiation the current bout got from its consequences
        this tick (a turn's angle, reaching the nest, curiosity used up…). ``walking_to``: the bout
        is a walk to a thing it has in mind and has not got there — not done, whatever its
        satiation says (see FINISH_SAT): a walk is finished by arriving."""
        p = self._g.p
        n = len(self.names)
        self._noise_t += dt_s
        if self._noise_t >= 0.25:
            decay = math.exp(-self._noise_t / p.sel_noise_tau_s)
            spread = p.sel_noise * math.sqrt(1.0 - decay * decay)
            stream = self._stream
            self._noise = [
                0.0 if told else x * decay + spread * stream.normal()
                for x, told in zip(self._noise, self._told_only)
            ]
            self._m_noise = [x * decay + spread * stream.normal() for x in self._m_noise]
            self._noise_t = 0.0
        self.bout_age += dt_s
        tau_a = 0.45 if p.hatchling_wobble < 0.5 else 0.33
        k_act = 1.0 - math.exp(-dt_s / tau_a)
        cur = self.current
        names = self.names
        for i in range(n):
            total = self._extra[i] + self._cmd_extra[i]
            for f, w in self._weights[i]:
                total += w * feat[f]
            s = self._gain[i] * total if total > 0.0 else 0.0
            s *= self._gate(names[i], ctx)
            self._pre[i] = s
            s *= 1.0 - 0.85 * self.sat[i]
            s += self._noise[i] * (0.4 + 0.6 * _clip(s * 3.0))
            s = s if s > 0.0 else 0.0
            self.sal[i] = s
            self.act[i] += k_act * (s - self.act[i])
            if i != cur:
                self.sat[i] *= math.exp(-dt_s / self._rec_tau[i])
        # the current bout satiates (faster when it is only weakly wanted)
        drive = self.sal[cur]
        rate = self._sat_rate[cur] * (1.25 - 0.5 * _clip(drive))
        self.sat[cur] = _clip(
            self.sat[cur] + (1.0 - self.sat[cur]) * (1.0 - math.exp(-dt_s * rate)) + consumed
        )
        if interrupted:
            self.act[cur] *= 0.5
            self.bout_age = max(self.bout_age, 3.0 * self._unit_s[cur])  # a reflex cut it
        # commitment: the selected channel holds on while its bout is fresh — and holds on to
        # its first motor unit (the lock fades over the unit's own time)
        # (the loop's self-excitation: as strong as the act is active — a half-hearted rest
        # does not lock out its name being called)
        lock = (
            0.0
            if release_commitment
            else (
                LOCK * min(1.0, self.act[cur] / 0.6) * math.exp(-self.bout_age / self._unit_s[cur])
            )
        )
        hold = self.act[cur] * (1.0 + p.commitment * (1.0 - self.sat[cur])) + p.switch_margin
        hold += lock
        if not self._gate_ok(names[cur], ctx):
            hold = -1.0
        # finishing what it started (see FINISH_SAT): the bout's peak want × a margin that holds
        # until it is done is what a challenger has to beat, unless it has surged
        sat_cur = self.sat[cur]
        self._want += (self._pre[cur] - self._want) * (1.0 - math.exp(-dt_s / WANT_TAU_S))
        self._peak = max(self._peak, self._want)
        unfinished = _clip((FINISH_SAT + FINISH_FADE - sat_cur) / FINISH_FADE)
        if walking_to:
            unfinished = 1.0
        committed = (
            unfinished > 0.0
            and hold >= 0.0
            and names[cur] not in UNFINISHED_OK
            and not BEHAVIOURS[names[cur]].sleep
            and not release_commitment
            and not interrupted
            and feat[F["homing"]] < HOMING_RELEASE
        )
        progress = min(1.0, sat_cur / FINISH_SAT)
        finish = (
            self._peak
            * (1.0 + FINISH_MARGIN * unfinished)
            * (1.0 + FINISH_RISE * progress * unfinished)
            if committed
            else 0.0
        )
        surge = SURGE * (1.0 + SURGE_RISE * progress)
        sal, base, ref = self.sal, self._base, self._ref
        best = cur
        best_act = hold
        cur_urgent: bool | None = None
        claims = self._claims()
        self._claim_age = self._claim_age + dt_s if any(c >= CLAIM_MIN for c in claims) else 0.0
        claimant = self._claimant(cur, feat, ctx, interrupted)
        self._once = None
        # the answer a person is waiting for is kept once it has the body: nothing but urgent
        # business takes the act that voices it
        protected = claimant == cur
        for i in range(n):
            if i == cur or self.act[i] <= best_act or not self._gate_ok(names[i], ctx):
                continue
            if protected and not self._is_urgent(i, feat):
                continue
            if committed and self.act[i] <= finish and sal[i] - max(base[i], ref[i]) < surge:
                # a rival that crept up does not take a bout that is not done — unless it is urgent
                # business and the bout is not
                if not self._is_urgent(i, feat):
                    continue
                if cur_urgent is None:
                    cur_urgent = self._is_urgent(cur, feat)
                if cur_urgent:
                    continue
            best, best_act = i, self.act[i]
        if (
            claimant is not None
            and claimant != cur
            and (best == cur or not self._is_urgent(best, feat))
        ):
            best = claimant
        forced, self._forced = self._forced, None
        if forced is not None and forced != cur and self._gate_ok(names[forced], ctx):
            best = forced
            # the switch is a decision, not a vote: the act is made the strongest, so it holds
            # the body as any winner does instead of giving it back to the bout it took it from
            self.act[forced] = max(self.act) + COMMIT_BOOST
            self._pre[forced] = max(self._pre[forced], max(self._pre))
        k_base = 1.0 - math.exp(-dt_s / SURGE_TAU_S)
        for i in range(n):
            base[i] += k_base * (sal[i] - base[i])
        if best != cur:
            self.current = best
            self.sat[best] *= 0.5
            self.bout += 1
            self.bout_age = 0.0
            self._want = self._peak = self._pre[best]
            self._ref = list(sal)
            return names[best], True
        return names[cur], False

    def _claims(self) -> list[float]:
        """The claims of the live bridge and of a command, summed per channel."""
        own, told = self._claim, self._cmd_claim
        return [a + b for a, b in zip(own, told)] if any(told) else own

    def _claimant(self, cur: int, feat: list[float], ctx: Context, interrupted: bool) -> int | None:
        """The channel a person's answer claims the body for this tick (the strongest claim of at
        least CLAIM_MIN), ``cur`` itself when the bout in hand is the one that voices it, else None:
        see CLAIM_MIN for what a claim never takes the body from."""
        claim = self._claims()
        names = self.names
        pick: int | None = None
        if claim[cur] >= CLAIM_MIN and self._gate_ok(names[cur], ctx):
            # (the act in hand already answers: the three answering acts do not trade it)
            pick = cur
        else:
            for i in range(len(names)):
                if claim[i] >= CLAIM_MIN and self._gate_ok(names[i], ctx):
                    if pick is None or self.sal[i] > self.sal[pick]:
                        pick = i
        once = self._once
        one_shot = False
        if pick is None and once is not None and self._gate_ok(names[once], ctx):
            pick, one_shot = once, True  # (a one-shot claim: ``claim_next``)
        if pick is None or interrupted or ctx.asleep or feat[F["fear"]] >= CLAIM_FEAR_MAX:
            return None
        if feat[F["homing"]] >= HOMING_RELEASE:
            return None
        if pick == cur:
            return cur
        if BEHAVIOURS[names[cur]].sleep:
            return None
        if names[cur] == "return_dock" and self._cmd_claim[pick] < CLAIM_MIN:
            # a walk home with a full pack yields to a person's command (never to a live answer;
            # a pack running low has already kept the body above: HOMING_RELEASE)
            return None
        if any(_feature(feat, b) > 0.5 for b in BUSY_ABSOLUTE):
            return None
        if (one_shot or self._claim_age < BUSY_WAIT_S) and any(
            _feature(feat, b) > 0.5 for b in BUSY_FEATURES
        ):
            return None
        if self._is_urgent(cur, feat):
            return None
        return pick

    def urgent_now(self, feat: list[float]) -> bool:
        """Whether the bout in hand is urgent business (a fear act: see ``URGENT_FEATURES``)."""
        return self._is_urgent(self.current, feat)

    def _is_urgent(self, i: int, feat: list[float]) -> bool:
        """Whether most of channel ``i``'s present (positive) input is urgent business."""
        urgent, positive = self._urgent[i]
        # (what a person says to the duck, the live voice's proposals, is an event, not a drift)
        said = self._extra[i] + self._cmd_extra[i]
        part = said + sum(w * feat[f] for f, w in urgent)
        return part > 0.0 and part >= URGENT_SHARE * (said + sum(w * feat[f] for f, w in positive))

    def moment_step(
        self,
        dt_s: float,
        feat: list[float],
        ctx: Context,
        *,
        busy: bool,
        stepping: bool,
        exclude: frozenset[str] = frozenset(),
    ) -> str | None:
        """The moments' own competition this tick: leaky accumulators of their salience (the
        same wiring and gains as the acts). Returns the moment that fires now (its activation
        crossed :data:`MOMENT_FIRE`; not while ``busy`` — a moment or a reflex is playing —
        and while ``stepping`` only :data:`MOMENTS_WALKING`), which then rests (satiation).
        ``exclude`` (fid:fix): moments the head has no room for from where it is now."""
        if not self.m_names:
            return None
        p = self._g.p
        k_act = 1.0 - math.exp(-dt_s / 0.45)
        best, best_act = -1, MOMENT_FIRE
        for i, name in enumerate(self.m_names):
            total = 0.0
            for f, w in self._m_weights[i]:
                total += w * feat[f]
            s = self._m_gain[i] * total if total > 0.0 else 0.0
            s *= self._gate(name, ctx)
            s *= 1.0 - self.m_sat[i]
            s += 0.5 * self._m_noise[i] * (0.4 + 0.6 * _clip(s * 3.0))
            s = s if s > 0.0 else 0.0
            self.m_sal[i] = s
            self.m_act[i] += k_act * (s - self.m_act[i])
            self.m_sat[i] *= math.exp(-dt_s / self._m_rec[i])
            allowed = (not stepping or name in MOMENTS_WALKING) and name not in exclude
            if allowed and self.m_act[i] > best_act:
                best, best_act = i, self.m_act[i]
        if busy or best < 0 or p is None:
            return None
        self.m_sat[best] = 1.0
        self.m_act[best] = 0.0
        return self.m_names[best]

    def _gate(self, behaviour: str, ctx: Context) -> float:
        spec = BEHAVIOURS[behaviour]
        if spec.sleep:
            if behaviour == "sleep_talk" and not ctx.asleep:
                return 0.0
            return 1.0
        g = ctx.wakefulness if ctx.asleep else 1.0
        if spec.locomotes and not ctx.can_walk:
            return 0.0
        if ctx.recharging and behaviour != "return_dock":
            escape = behaviour in ("flee", "retreat") and ctx.charge_escape
            if not escape and (
                spec.locomotes or spec.posture == "stand" or behaviour == "stand_up"
            ):
                return 0.0
        if behaviour == "return_dock":
            if ctx.nest_rest:
                return 0.0
            return g  # (dock:build) going home ends seated in the dock: both postures are its own
        # changing posture costs effort: a behaviour that needs the other posture is less
        # attractive (fewer sit/stand flip-flops)
        if spec.posture == "stand" and ctx.sitting:
            g *= 0.72 * (1.0 - 0.6 * ctx.posture_tired)
        elif spec.posture == "sit" and ctx.standing:
            g *= 0.8 * (1.0 - 0.6 * ctx.posture_tired)
        if behaviour in ("stand_up", "sit_down"):
            g *= 1.0 - 0.6 * ctx.posture_tired
        if behaviour == "stand_up" and ctx.standing:
            return 0.0  # (while getting up it is neither: the bout carries on)
        if behaviour == "sit_down" and ctx.sitting:
            return 0.0
        if behaviour == "body_shake" and not ctx.standing:
            return 0.0
        return g

    def _gate_ok(self, behaviour: str, ctx: Context) -> bool:
        return self._gate(behaviour, ctx) > 0.0

    # ── reasons ──────────────────────────────────────────────────────────────

    def why(self, behaviour: str, feat: list[float], top: int = 2) -> str:
        """The strongest positive inputs of a channel now, in plain Chinese."""
        weights = W.get(behaviour, {})
        parts = sorted(
            ((w * feat[F[f]], f) for f, w in weights.items() if w > 0 and feat[F[f]] > 0.05),
            reverse=True,
        )
        words = [
            _feature_zh(f, feat[F[f]]) for value, f in parts[:top] if value > 0.02 and f != "bias"
        ]
        if not words:
            return msg("neuro.why.nothing")
        return join(msg("neuro.why.because.prefix"), lst(words, "fmt.sep.and_then"))

    def ranking(self, k: int = 6) -> list[tuple[str, float, float]]:
        order = sorted(range(len(self.names)), key=lambda i: -self.act[i])[:k]
        return [(self.names[i], self.sal[i], self.act[i]) for i in order]


class CommandChannel:
    """One owner of the selector's command channel (:meth:`Selecter.set_command`): what a person told
    the duck (:mod:`duckmind.neuro.command_brain`) and what an errand holds (:mod:`duckmind.neuro.errand`)
    are composed here and written once per tick, so neither overwrites the other. A command's ask
    lasts until it is released; an errand's hold until it arrives or gives up."""

    def __init__(self) -> None:
        self._ask: dict[str, float] = {}
        self._hold: dict[str, float] = {}

    def ask(self, behaviour: str, weight: float) -> None:
        """A command's claim on a channel, until :meth:`drop_ask`."""
        self._ask[behaviour] = float(weight)

    def drop_ask(self) -> None:
        self._ask.clear()

    def hold(self, behaviour: str, weight: float) -> None:
        """An errand's hold on a channel, until :meth:`drop_hold`."""
        self._hold[behaviour] = float(weight)

    def drop_hold(self) -> None:
        self._hold.clear()

    def apply(self, sel: Selector) -> dict[str, float]:
        """Write the composed channel to ``sel``: the strongest of the two per channel."""
        merged: dict[str, float] = {}
        for name, weight in (*self._ask.items(), *self._hold.items()):
            merged[name] = max(merged.get(name, 0.0), weight)
        return sel.set_command(merged, claims=merged)



def _feature_zh(feature: str, value: float) -> str:
    """A feature in plain words; how new a thing is depends on how new it still is."""
    if feature == "novel":
        if value >= 0.75:
            return msg("neuro.why.novel.new")
        if value >= 0.4:
            return msg("neuro.why.novel.half_known")
        return msg("neuro.why.novel.known")
    return FEATURE_ZH[feature]


def _feature(feat: list[float], name: str) -> float:
    """A feature's value, 0 where this brain has no such feature (a skill's, before it is installed)."""
    index = F.get(name)
    return feat[index] if index is not None and index < len(feat) else 0.0


def empty_features() -> list[float]:
    feat = [0.0] * len(FEATURES)
    feat[F["bias"]] = 1.0
    return feat
