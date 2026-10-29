"""The frozen numbers and word lists of MACHINE-VERIFY-v3.md (2026-10-02), beside the v1 and v2 ones.

``spec.py`` (v1) and ``spec_v2.py`` (v2) are untouched and stay what v1 and v2 judged with. Numbers the
v3 spec leaves as they were are read from them (not copied); numbers it changes or adds are here, each
naming the sentence of the spec it comes from. The operational choices the spec leaves open are in
``docs/design/pet-behavior-v1/MACHINE-VERIFY-v3-CHOICES.md`` (committed before any v3 result).
"""

from __future__ import annotations

from duckmind.verify import spec as v1
from duckmind.verify import spec_v2 as v2

# ── unchanged ────────────────────────────────────────────────────────────────────
WINDOW_S = v2.WINDOW_S
AUDIBLE_DB = v2.AUDIBLE_DB
LOUD_HEAD_DB_MIN = v2.LOUD_HEAD_DB_MIN
L2_BANG_DB1M = v2.L2_BANG_DB1M
HOLD_REFLEXES = v2.HOLD_REFLEXES
LIFE_STIM_ORDER = v2.LIFE_STIM_ORDER
RUNS = v2.RUNS  # a seed's life: a day run and a night run
M6_ACC_MIN = v2.M6_ACC_MIN
M6_EACH_MIN = v2.M6_EACH_MIN
M6_SIG_MIN = v2.M6_SIG_MIN
M1_STIM_MIN = v2.M1_STIM_MIN  # 有刺激組 ≥ 70%
M1_DIFF_MIN = v2.M1_DIFF_MIN  # 有刺激組 − 對照組 ≥ 0.5
FALL_GRACE_S = v1.FALL_GRACE_S  # 寬限時間 0.5 s (CHOICES §七.3)

# ── M1 注意：有方向只留滾球、放新東西；其餘是事件 ─────────────────────────────────────
DIRECTIONAL = frozenset({"roll_ball", "novel_object"})
EVENT = frozenset({"clap", "bang", "call_name", "hello", "loom", "pet_head", "poke"})
STIMULI = (
    "roll_ball",
    "novel_object",
    "clap",
    "bang",
    "call_name",
    "hello",
    "pet_head",
    "poke",
    "loom",
)
L1_CELL_PAIRS = 16  # 腦層：一個個性 × 一種刺激，種子 1–16
L2_CELL_PAIRS = 32  # 物理層：一種刺激、合併 4 個個性 × 種子 1–8

# ── 物理層 ───────────────────────────────────────────────────────────────────────
L2_SEEDS = tuple(range(1, 9))  # 種子 1–8
PUSH = "push"
L2_STIMULI = (*STIMULI, PUSH)
# every stimulus has its own second (so the shams of one seed are not one trial many times);
# the first five are v2's, the new ones follow at +6 s (CHOICES §六.3)
L2_ONSET_S = {
    "roll_ball": 10.0,
    "novel_object": 16.0,
    "clap": 22.0,
    "call_name": 28.0,
    "bang": 34.0,
    "hello": 40.0,
    "pet_head": 46.0,
    "poke": 52.0,
    "loom": 58.0,
    PUSH: 64.0,
}
# the spec's name -> the physics world's own name (neuro_world.STIMULI)
PHYSICS_NAME = {"loom": "hand_loom"}
SOUNDS = ("clap", "call_name", "bang", "hello")  # level at the head is the truth of "heard"
CONTACT = ("pet_head", "poke")  # contact is always "touchable" (v1, v2)
LOOM_SEE_S = 1.5  # L2 loom: the hand is in the head camera's sample within this long after it comes

# ── M7 推一下 ────────────────────────────────────────────────────────────────────
PUSH_STRENGTH = 1.0  # the world's push interface at its top: 7 N (neuro_world.PUSH_N)
PUSH_WINDOW_S = WINDOW_S  # 倒下要在推了之後 3 秒內
PUSH_MIN_TIPPED = 4  # 推倒的次數不到 4 次 → 未完成
PUSH_BLOCKED_MAX = 0  # 被擋的次數必須是 0
# the physics world's own stimulus names the spec's nine map to, and what is left over
OTHER_PHYSICS_STIMULI = (
    "voice",
    "door",
    "knock",
    "music",
    "tv",
    "wave",
    "stroke_back",
    "toy_drop",
    "walk_by",
    "sit_near",
    "lights_off",
    "lights_on",
    "pick_up",
    "set_down",
    "say",  # (live voice: a scripted line of the family, at a place; not one of the nine: 未測)
)

# ── M6 z 分數 ────────────────────────────────────────────────────────────────────
Z_SD_MIN = 1e-12  # 標準差 ≤ 這個 ＝ 0：那一維拿掉

# ── 真實耳朵延遲版 ───────────────────────────────────────────────────────────────
EAR_LAYER = "L1耳"
