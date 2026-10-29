"""聽音樂跳舞 — dancing to music: beat entrainment from what the microphone hears (a:sound).

Owner, 2026-09-28 (「A B 全數」, behaviour 4): with music playing the duck bobs its head to the beat
and a songbird sings along — emerging from the neural brain, performed by the body, decided only
from its own senses.

**What it hears.** Only the microphone model's sound sense (:class:`~duckmind.neuro.contract.
SoundSense`): a level, an onset strength each tick (a sudden rise over the background — a kick or
a snare in the music is one) and, after the classifier's delay, a class (``music``). No beat
times, no tempo, no world clock: the world plays music with a tempo, the microphone model gives
onsets (late 20–40 ms, some missed, some extra), and the brain must find the beat itself.

**Beat induction and entrainment** (:class:`BeatOscillator`). An adaptive phase oscillator — the
"dynamic attending" oscillator of Large & Jones (1999) with the linear phase and period
correction of Repp's two-process model: the first few onsets set its period (the median
inter-onset interval, allowing a missed beat); after that every onset near a predicted beat
pulls the phase (``PHASE_GAIN`` of the error) and the period (``PERIOD_GAIN``) toward it, and the
phase-locking value of the recent onsets is how sure it is (``lock``). Onsets far from a
predicted beat (off-beats) barely move it; beats that pass with no onset near them wear the lock
down, and after ``LOST_BEATS`` of them the beat is lost. An onset that falls where a beat was
expected is flagged ``expected``: the brain's reflex layer should not orient to it (predictive
coding — the beat is anticipated, not a surprise; see the integration note in skill_sound).

**The dance** (:class:`DanceSkill`): its urge (the selection feature ``beat``) is the lock while it
hears music, and the motor overlay (:func:`bob_pose`) nods the head DOWN on the beat — sized
above the gearboxes' deadband, and at a rate the head can follow in its posture (the head is a
low-pass filter: A100 yaw sines at head_alpha 1.0 standing gain 0.77 at 1 Hz, 0.50 at 2 Hz;
seated ~1.0 up to 2 Hz), so a fast tempo is danced every second (or fourth) beat, the way
animals entrain at integer ratios; commanded ahead of the beat by the head's own lag (a forward
model) so the body is down ON the beat. A songbird (its branch, or its nature before GD 45) sings
along: rhythmic chirps timed to land on the beat (the call is started ``CHIRP_LEAD_MS`` early).

Standard library only (the brain never imports the world); floats in the prototype, phase 5
converts (the oscillator is a phase accumulator and a few multiply-adds).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from duckmind.neuro.contract import HeadPose

DANCE_VERSION = "neuro-dance-1"

# ── beat induction and entrainment ──────────────────────────────────────────────

BPM_MIN = 50.0
BPM_MAX = 180.0
ONSET_MIN = 0.08  # the reflex layer's own onset threshold (reflex.py: snd.onset ≥ 0.08)
INDUCE_ONSETS = 4  # onsets before a beat is induced
INDUCE_CV = 0.2  # the intervals, each over its nearest multiple of the median, this regular
INDUCE_KEEP_S = 6.0  # onsets kept for the induction
PHASE_GAIN = 0.4  # α: share of an on-beat phase error corrected at once (Repp: 0.3–0.6)
PERIOD_GAIN = 0.12  # β: share of it that goes into the period
OFF_BEAT_GAIN = 0.05  # an onset far from any predicted beat pulls this little
ON_BEAT = 0.25  # |phase error| (periods) that counts as "at a beat"
EXPECT = 0.12  # |phase error| that was expected: no surprise, no orienting
LOCK_K = 0.22  # the phase-locking value's update per onset (≈ the last 4–5 onsets)
LOCK_MISS = 0.8  # a beat that passes with no onset near it: the lock × this
LOCK_LOST = 0.2  # a lock this low: the beat is lost (found again from the onsets)
LOCK_DOUBT = 0.6  # a lock under this for DOUBT_BEATS beats in a row: the tempo is doubted
DOUBT_BEATS = 8
ATTEND = 0.1  # the attentional pulse's width around an expected beat (periods, σ)
LOST_BEATS = 5  # this many beats in a row with nothing near them: the beat is lost
MUSIC_MEMORY_S = 12.0  # it heard ``music`` this recently: it is still the music


def _wrap_phase(x: float) -> float:
    """A phase error in periods, in (−0.5, 0.5]."""
    x = x - math.floor(x)
    return x - 1.0 if x > 0.5 else x


@dataclass(frozen=True, slots=True)
class BeatTick:
    """The oscillator after one tick (``phase`` 0 = on the beat, rising to 1)."""

    active: bool
    phase: float
    hz: float
    lock: float
    beats: int  # beats counted since it locked on (the bar position is beats % 4)
    beat_edge: bool  # a beat fell inside this tick
    onset: bool  # an onset was heard this tick
    expected: bool  # … and it fell where a beat was predicted
    error: float | None  # that onset's phase error (periods; + = after the predicted beat)

    @property
    def bpm(self) -> float:
        return 60.0 * self.hz

    def until_beat_s(self) -> float:
        return (1.0 - self.phase) / self.hz if self.hz > 0.0 else math.inf


@dataclass(slots=True)
class BeatOscillator:
    """The beat-finding oscillator (see the module docstring). ``step(t_s, onset)`` each tick."""

    phase: float = 0.0
    hz: float = 0.0
    active: bool = False
    lock: float = 0.0
    beats: int = 0
    plv_c: float = 0.0
    plv_s: float = 0.0
    onsets: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    last_t: float | None = None
    # Was each beat heard? An onset may come a little before a beat or (the microphone is late)
    # a little after it: a beat is judged once its window (±ON_BEAT) has passed.
    pre_hit: bool = False  # an onset came in the window before the coming beat
    window_hit: bool = False  # … in the window around the beat that just passed
    judging: bool = False  # that beat's window is still open
    silent_beats: int = 0
    doubt: int = 0  # beats in a row judged with a weak lock (a wrong tempo: induce again)
    induced: int = 0  # how many times a beat was induced (a new song, a re-lock)

    def reset(self) -> None:
        self.active = False
        self.lock = 0.0
        self.plv_c = self.plv_s = 0.0
        self.hz = 0.0
        self.phase = 0.0
        self.silent_beats = 0
        self.doubt = 0
        self.pre_hit = self.window_hit = self.judging = False

    def step(self, t_s: float, onset: float) -> BeatTick:
        dt = 0.0 if self.last_t is None else min(0.25, max(0.0, t_s - self.last_t))
        self.last_t = t_s
        edge = False
        if self.active:
            self.phase += self.hz * dt
            while self.phase >= 1.0:
                self.phase -= 1.0
                self.beats += 1
                edge = True
                if self.judging:  # (two beats in one tick: the older one was not heard)
                    self._judge(self.window_hit)
                self.window_hit, self.pre_hit, self.judging = self.pre_hit, False, True
        heard = onset >= ONSET_MIN
        expected = False
        error: float | None = None
        if heard:
            self.onsets.append((t_s, onset))
            while self.onsets and t_s - self.onsets[0][0] > INDUCE_KEEP_S:
                self.onsets.popleft()
            if not self.active:
                self._induce(t_s)
                if self.active:
                    error = 0.0
            else:
                error = _wrap_phase(self.phase)
                weight = min(1.0, onset / 0.5)
                # the attentional pulse (Large & Jones): an onset near the expected time counts
                # most, one at the window's edge little
                focus = math.exp(-0.5 * (error / ATTEND) ** 2)
                if abs(error) <= ON_BEAT:
                    period = 1.0 / self.hz
                    pull = PHASE_GAIN * focus * weight
                    self.phase = min(0.999999, max(0.0, self.phase - pull * error))
                    period *= 1.0 + PERIOD_GAIN * focus * weight * error
                    self.hz = min(BPM_MAX, max(BPM_MIN, 60.0 / period)) / 60.0
                    c = math.cos(2.0 * math.pi * error)
                    s = math.sin(2.0 * math.pi * error)
                    k = LOCK_K * (0.5 + 0.5 * weight)
                    self.plv_c += k * (c - self.plv_c)
                    self.plv_s += k * (s - self.plv_s)
                    self.lock = min(1.0, math.hypot(self.plv_c, self.plv_s))
                    if error < 0.0:
                        self.pre_hit = True  # the coming beat is heard (a little early)
                    else:
                        self.window_hit = True  # the beat that just passed is heard
                    expected = abs(error) <= EXPECT
                else:
                    pull = OFF_BEAT_GAIN * focus * weight
                    self.phase = min(0.999999, max(0.0, self.phase - pull * error))
                if self.lock < LOCK_LOST:
                    self.reset()  # it has lost the beat: find it again from the onsets
        if self.active and self.judging and self.phase >= ON_BEAT:
            self._judge(self.window_hit)
        return BeatTick(
            active=self.active,
            phase=self.phase,
            hz=self.hz,
            lock=self.lock if self.active else 0.0,
            beats=self.beats,
            beat_edge=edge,
            onset=heard,
            expected=expected,
            error=error,
        )

    def _judge(self, heard: bool) -> None:
        """A beat's window has passed: heard, or one more silent beat (the lock wears down; after
        ``LOST_BEATS`` of them the beat is lost)."""
        self.judging = False
        self.doubt = self.doubt + 1 if self.lock < LOCK_DOUBT else 0
        if self.doubt >= DOUBT_BEATS:
            self.reset()  # a tempo that never fits well: find the beat again
            return
        if heard:
            self.silent_beats = 0
            return
        self.silent_beats += 1
        self.lock *= LOCK_MISS
        self.plv_c *= LOCK_MISS
        self.plv_s *= LOCK_MISS
        if self.silent_beats >= LOST_BEATS:
            self.reset()

    def _induce(self, t_s: float) -> None:
        """A beat from the recent onsets (beat induction from the inter-onset intervals, as
        Dixon's): every interval between two onsets, or a half or a third of it, in the dance
        range is a candidate period; the one whose grid through the newest of its pair catches
        the most onset strength (within ±``EXPECT`` × 1.5 of a period), less half an average
        onset for every empty beat between them and a little for its distance from 120 a minute
        (the preferred tempo), wins if it catches ``INDUCE_ONSETS`` onsets. Its period is then
        fitted to those onsets (least squares) and the newest onset is taken as a beat. Weak
        stray onsets (a clap, a knock) do not outweigh the music's; a missed beat is allowed."""
        heard = list(self.onsets)
        if len(heard) < INDUCE_ONSETS:
            return
        times = [t for t, _ in heard]
        mean_strength = sum(v for _, v in heard) / len(heard)
        lo, hi = 60.0 / BPM_MAX, 60.0 / BPM_MIN
        tol = 1.5 * EXPECT
        best: tuple[float, float, list[tuple[int, float]]] | None = None
        for j in range(len(times) - 1, 0, -1):
            for i in range(j):
                gap = times[j] - times[i]
                for n in (1, 2, 3):
                    period = gap / n
                    if not lo <= period <= hi:
                        continue
                    fit: dict[int, tuple[float, float]] = {}
                    for t, v in heard:
                        m = (t - times[j]) / period
                        k = round(m)
                        if abs(m - k) <= tol and (k not in fit or v > fit[k][1]):
                            fit[k] = (t, v)
                    if len(fit) < INDUCE_ONSETS:
                        continue
                    # beats of the grid with nothing on them count against it (so a grid at
                    # twice the tempo, every other slot empty, loses to the true one)
                    empty = (max(fit) - min(fit) + 1) - len(fit)
                    score = sum(v for _, v in fit.values()) - 0.5 * mean_strength * empty
                    score -= 0.3 * mean_strength * abs(math.log2(period / 0.5))
                    if best is None or score > best[0] + 1e-9:
                        best = (score, period, [(k, fit[k][0]) for k in sorted(fit)])
        if best is None:
            return
        _, period, fit_list = best
        seen = dict(fit_list)
        ks = sorted(seen)
        mk = sum(ks) / len(ks)
        mt = sum(seen[k] for k in ks) / len(ks)
        var = sum((k - mk) ** 2 for k in ks)
        if var > 0.0:
            period = sum((k - mk) * (seen[k] - mt) for k in ks) / var
        if not lo <= period <= hi:
            return
        residual = math.sqrt(sum((seen[k] - (mt + (k - mk) * period)) ** 2 for k in ks) / len(ks))
        if residual > INDUCE_CV * period:
            return
        self.hz = 1.0 / period
        # the newest onset on the grid is a beat; it is now (t_s − that) later
        newest = mt + (max(ks) - mk) * period
        self.phase = min(0.999999, max(0.0, (t_s - newest) * self.hz))
        self.active = True
        self.lock = 0.5
        self.plv_c, self.plv_s = 0.5, 0.0
        self.beats = 0
        self.silent_beats = 0
        self.doubt = 0
        self.window_hit, self.pre_hit, self.judging = True, False, True
        self.induced += 1


# ── the body's dance ────────────────────────────────────────────────────────────

# The head nods DOWN on the beat (head pitch, clip units: 0.35 rad per unit) with a small roll
# sway at half the nod rate. The nod rate is the beat's divided by 1, 2 or 4, whichever is the
# fastest the head can follow in its posture (the head is a low-pass filter: see the module
# docstring); the command leads the beat by the head's lag in that posture.
BOB_MAX_HZ = {"sit": 2.2, "stand": 1.25}
BOB_LEAD_S = {"sit": 0.15, "stand": 0.23}  # the head pitch lag (a:sound headless, alpha 1.0)
BOB_HP = {"sit": 0.42, "stand": 0.55}  # nod half-amplitude, clip units of head pitch (± rad×0.35)
BOB_HR = {"sit": 0.35, "stand": 0.45}  # roll sway, clip units of head roll (×0.25 rad)
BOB_MIN_LOCK = 0.35  # below this lock it only sways, softly, without a nod
# Its own hearing is late: an onset is known ~45 ms after the sound began (the microphone's
# detector, 20–40 ms, and a tick) — a fixed property of its own nervous system, which it
# anticipates (as the brain already does the camera's capture delay): its beat is placed that much
# earlier than it heard it. (On the board: the audio pipeline's own latency, phase 8.)
HEARING_LAG_S = 0.045
CHIRP_LEAD_MS = 30  # a chirp is started this early so its attack lands on the beat (the wav's
# attack; on the board add robotd's aplay start-up, phase 8)


def beats_per_bob(hz: float, posture: str) -> int:
    """Beats per nod (1, 2 or 4): the head nods at the fastest rate it can follow."""
    limit = BOB_MAX_HZ.get("sit" if posture == "sit" else "stand", 1.25)
    for n in (1, 2, 4):
        if hz / n <= limit:
            return n
    return 4


def bob_pose(
    beat: BeatTick, posture: str, amount: float, *, lead: bool = True, adjust: float = 0.0
) -> HeadPose:
    """The dance's head offset now (clip units, to ADD to the program's pose): down on the beat,
    up between, a sway from side to side every other nod. ``amount`` 0..1 is how much it dances
    (engagement × the stage's amplitude cap); nothing without an active beat. ``adjust``: the
    timing it learned from its own felt nods (:class:`NodTimer`, s, + = earlier)."""
    if not beat.active or beat.hz <= 0.0 or amount <= 0.0:
        return HeadPose()
    key = "sit" if posture == "sit" else "stand"
    n = beats_per_bob(beat.hz, key)
    ahead = (BOB_LEAD_S[key] + adjust + HEARING_LAG_S) * beat.hz if lead else 0.0
    # where the head should be `ahead` beats from now: the nod's own phase (0 = down)
    total = (beat.beats % (2 * n)) + beat.phase + ahead
    nod_phase = (total / n) % 1.0
    sway_phase = (total / (2 * n)) % 1.0
    locked = max(0.0, min(1.0, (beat.lock - BOB_MIN_LOCK) / (0.8 - BOB_MIN_LOCK)))
    nod = -BOB_HP[key] * amount * locked * math.cos(2.0 * math.pi * nod_phase)
    sway = BOB_HR[key] * amount * (0.4 + 0.6 * locked) * math.sin(2.0 * math.pi * sway_phase)
    return HeadPose(hp=nod, hr=sway, hy=0.35 * sway)


# Its own nods, felt: the lead above is its innate guess of its neck's delay; the delay it
# really has (another body, another controller, the real robot) it learns by comparing the nods
# it sent (the efference copy) with the nods its joint sense feels (a forward model of its own
# neck: Wolpert & Kawato 1998; Repp & Su 2013 for movement timing to a beat).
PROPRIO_LAG_S = 0.015  # its joint sense is this much older than the head (live: 15 ms measured)
TIMING_GAIN = 0.3  # the share of each estimate's difference it takes on
TIMING_MAX_S = 0.12  # the learned correction stays within this of the innate lead
TIMING_MIN_LOCK = 0.6  # it judges its timing only against a beat it is sure of
TIMING_MIN_CORR = 0.6  # a felt nod this much like the sent one (else: not judged)
TIMING_MAX_LAG_S = 0.5
TIMING_TICK_S = 0.02


@dataclass(slots=True)
class NodTimer:
    """Its neck's delay, learned by feel: the nods it sent (``sent``, one per tick: the nod's
    head-pitch overlay as its joint sense reads pitch, + = nose down) against what its joint
    sense felt (``got``, the measured head pitch), every nod, over the last two nods: the lag of
    best correlation is how late its head follows its commands (plus its joint sense's own age,
    innate: :data:`PROPRIO_LAG_S`). ``adjust``: the correction to the innate lead (s, + =
    earlier) that puts its felt nods on the beat. Only while it dances to a beat it is sure of;
    a nod that does not look like the one it sent (a reflex, a stumble) is not judged."""

    adjust: float = 0.0
    felt: int = 0  # estimates made
    delay: float | None = None  # its latest estimate of the command → felt delay (s)
    corr: float = 0.0
    sent: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    got: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    next_t: float = 0.0

    def reset(self) -> None:
        self.sent.clear()
        self.got.clear()

    def sent_nod(self, nod: float) -> None:
        """This tick's nod as sent (after :meth:`step` of the same tick)."""
        if self.got:
            self.sent.append(float(nod))

    def step(
        self, t_s: float, pitch: float | None, beat: BeatTick, posture: str, *, dancing: bool
    ) -> float | None:
        """One tick of its joint sense while it dances; the delay it just estimated (s), else
        None."""
        if (
            not dancing
            or pitch is None
            or not beat.active
            or beat.hz <= 0.0
            or beat.lock < TIMING_MIN_LOCK
        ):
            self.reset()
            return None
        if len(self.got) > len(self.sent):
            self.got.pop()  # (the last tick sent nothing: keep them paired)
        self.got.append(float(pitch))
        key = "sit" if posture == "sit" else "stand"
        period = beats_per_bob(beat.hz, key) / beat.hz  # one nod
        max_lag = int(min(TIMING_MAX_LAG_S, 0.9 * period) / TIMING_TICK_S)
        window = int(2.0 * period / TIMING_TICK_S)
        if t_s < self.next_t or len(self.sent) < window + max_lag + 1:
            return None
        self.next_t = t_s + period
        got = list(self.got)[-window - 1 : -1]  # (the felt samples paired with sent ones)
        sent = list(self.sent)
        n = len(sent)
        gm = sum(got) / len(got)
        g = [v - gm for v in got]
        gn = math.sqrt(sum(v * v for v in g))
        if gn < 1e-6:
            return None
        scores = []
        for lag in range(max_lag + 1):
            seg = sent[n - window - lag : n - lag]
            sm = sum(seg) / len(seg)
            c = [v - sm for v in seg]
            cn = math.sqrt(sum(v * v for v in c))
            scores.append(sum(a * b for a, b in zip(g, c, strict=True)) / (gn * cn + 1e-12))
        best = max(range(len(scores)), key=lambda i: scores[i])
        self.corr = scores[best]
        if scores[best] < TIMING_MIN_CORR:
            return None
        frac = 0.0
        if 0 < best < len(scores) - 1:
            left, mid, right = scores[best - 1], scores[best], scores[best + 1]
            den = left - 2.0 * mid + right
            if den < 0.0:
                frac = 0.5 * (left - right) / den
        self.delay = (best + frac) * TIMING_TICK_S
        target = self.delay - PROPRIO_LAG_S - BOB_LEAD_S[key]
        self.adjust += TIMING_GAIN * (target - self.adjust)
        self.adjust = max(-TIMING_MAX_S, min(TIMING_MAX_S, self.adjust))
        self.felt += 1
        return self.delay


@dataclass(frozen=True, slots=True)
class DanceTick:
    """What the dance skill says this tick: the selection feature, the motor overlay, a chirp."""

    beat: BeatTick
    feature: float  # the selection feature ``beat``: a beat it can dance to, in music (0..1)
    music: bool  # it heard music recently (its classifier said so)
    chirp: str | None  # a call to start now (a songbird singing along), else None
    chirp_beat: int  # the beat that call is for (−1: none)
    recurring: bool = False  # an off-beat onset where the pattern had one bars ago: expected


# The rhythm's own pattern: an onset off the beat that came at the same place one, two or four
# bars ago (4, 8 or 16 beats back, within RECUR_TOL of a beat) is expected too — a song's
# syncopation surprises it the first time, not every time (metrical expectancy builds within a
# few cycles of a pattern: Large & Jones 1999; Vuust & Witek 2014)
RECUR_BEATS = (4, 8, 16)
RECUR_TOL = 0.12  # of a beat
RECUR_MIN_LOCK = 0.5
TEMPO_PREF_BPM = (80.0, 125.0)  # its favourite tempo: a calm duck's … a lively duck's
TEMPO_WIDTH_OCT = 0.5  # how far from it (octaves) the pull falls to ~60 %
LIKE_RATE = 0.2  # a dance it kept at (≥ LIKE_MIN_S): its liking moves this share toward +1
LIKE_MIN_S = 8.0
DISLIKE_MAX_S = 3.0  # a dance it gave up this soon: a little less liking


@dataclass(slots=True)
class DanceSkill:
    """One duck's ear for the beat and its way of dancing to it.

    ``songbird``: how much it is a songbird (1 for its grown branch; its nature's share before
    the branch) — it sings along; ``activity`` / ``sociability`` (0..1) shape how much it
    wants to dance (the lively and the sociable more: genome._gains' playful acts), and a lively
    duck's favourite tempo is faster (:data:`TEMPO_PREF_BPM`): the pull is strongest near it.
    What it learned: ``liking`` (−1..1) rises with every dance it kept at and falls a little with
    one it gave up at once — a duck that enjoyed dancing comes to the next beat sooner."""

    songbird: float = 0.0
    activity: float = 0.5
    sociability: float = 0.5
    seed: int = 0
    osc: BeatOscillator = field(default_factory=BeatOscillator)
    music_t: float = -1e9
    chirped_beat: int = -1
    pattern: int = 2  # beats between chirps this bout (chosen at the bout's start)
    last: BeatTick | None = None
    liking: float = 0.0
    dances: int = 0
    timer: NodTimer = field(default_factory=NodTimer)  # its nods' timing, learned by feel
    heard: deque[float] = field(default_factory=lambda: deque(maxlen=128))  # its onsets (s)

    @property
    def preferred_bpm(self) -> float:
        lo, hi = TEMPO_PREF_BPM
        return lo + (hi - lo) * max(0.0, min(1.0, self.activity))

    def tempo_fit(self, bpm: float) -> float:
        """1 at its favourite tempo, falling off by octaves (and never to nothing)."""
        if bpm <= 0.0:
            return 0.0
        off = math.log2(bpm / self.preferred_bpm) / TEMPO_WIDTH_OCT
        return math.exp(-0.5 * off * off)

    def learn(self, seconds: float) -> None:
        """A dance bout ended after ``seconds``: kept at, it was enjoyed; dropped at once, not."""
        self.dances += 1
        if seconds >= LIKE_MIN_S:
            self.liking += LIKE_RATE * (1.0 - self.liking)
        elif seconds < DISLIKE_MAX_S:
            self.liking += 0.25 * LIKE_RATE * (-1.0 - self.liking)
        self.liking = max(-1.0, min(1.0, self.liking))

    def start_bout(self, bout: int, hz: float) -> None:
        """A new dance bout: its singing pattern (every beat at a slow tempo, every second or
        fourth beat when fast), drawn from the bout's own number (no two bouts alike)."""
        options = (1, 2, 2, 4) if hz <= 1.6 else (2, 2, 4, 4)
        self.pattern = options[(bout * 2654435761 + self.seed) % len(options)]
        self.chirped_beat = -1

    def step(
        self,
        t_ms: int,
        onset: float,
        kind: str | None,
        *,
        dancing: bool,
        asleep: bool = False,
        arousal: float = 0.4,
    ) -> DanceTick:
        t_s = t_ms / 1000.0
        if kind == "music":
            self.music_t = t_s
        beat = self.osc.step(t_s, 0.0 if asleep else onset)
        self.last = beat
        recurring = False
        if not asleep and onset >= ONSET_MIN:
            if beat.active and beat.hz > 0.0 and beat.lock >= RECUR_MIN_LOCK and not beat.expected:
                period = 1.0 / beat.hz
                for m in RECUR_BEATS:
                    back = t_s - m * period
                    if any(abs(h - back) <= RECUR_TOL * period for h in self.heard):
                        recurring = True
                        break
            self.heard.append(t_s)
        music = t_s - self.music_t <= MUSIC_MEMORY_S
        # a beat in music is the pull; a steady rhythm without the music label pulls a little;
        # more near its favourite tempo, and the more it has come to like dancing
        feature = beat.lock if beat.active else 0.0
        feature *= 1.0 if music else 0.35
        if beat.active:
            feature *= (0.55 + 0.45 * self.tempo_fit(beat.bpm)) * (0.8 + 0.4 * self.liking)
        feature = max(0.0, min(1.0, feature))
        chirp: str | None = None
        chirp_beat = -1
        if dancing and beat.active and beat.lock >= 0.5 and self.songbird >= 0.35:
            # the call is started CHIRP_LEAD_MS before the beat it is for
            due = (beat.until_beat_s() - HEARING_LAG_S) * 1000.0 <= CHIRP_LEAD_MS + 10.0
            target = beat.beats + 1  # the coming beat
            if due and target != self.chirped_beat and target % max(1, self.pattern) == 0:
                self.chirped_beat = target
                chirp_beat = target
                bar_end = target % 16 == 15
                chirp = "greet" if bar_end and arousal > 0.45 else "chirp"
        return DanceTick(beat, feature, music, chirp, chirp_beat, recurring)


__all__ = [
    "BOB_HP",
    "BOB_LEAD_S",
    "BOB_MAX_HZ",
    "BeatOscillator",
    "BeatTick",
    "CHIRP_LEAD_MS",
    "DANCE_VERSION",
    "DanceSkill",
    "DanceTick",
    "NodTimer",
    "PROPRIO_LAG_S",
    "beats_per_bob",
    "bob_pose",
]
