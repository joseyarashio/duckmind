"""The frozen numbers and word lists of MACHINE-VERIFY-v2.md (2026-10-02), beside the v1 ones.

``spec.py`` (v1) is untouched and stays what v1 judged with. Numbers the v2 spec leaves as they
were are read from it (not copied); numbers it changes or adds are here, each naming the sentence
of the spec it comes from. The operational choices the spec leaves open are in
``docs/design/pet-behavior-v1/MACHINE-VERIFY-v2-CHOICES.md`` (committed before any v2 result).
"""

from __future__ import annotations

import math

from duckmind.verify import spec as v1

# ── unchanged (same sentence as v1) ──────────────────────────────────────────────
WINDOW_S = v1.WINDOW_S
M1_TURN_MIN_DEG = v1.M1_TURN_MIN_DEG
M1_ERR_SHRINK = v1.M1_ERR_SHRINK
AUDIBLE_DB = v1.AUDIBLE_DB
BEAK_REACH_M = v1.BEAK_REACH_M
M4_ARRIVE_MARGIN_M = v1.M4_ARRIVE_MARGIN_M
NATURAL_SATIATION = v1.NATURAL_SATIATION  # 滿足度 ≥ 0.5
STARTLE_NAMES = v1.STARTLE_NAMES

# ── M1 注意（改）─────────────────────────────────────────────────────────────────
M1_STIM_MIN = 0.70  # 有刺激組 ≥ 70%
M1_DIFF_MIN = 0.5  # 有刺激組 − 對照組 ≥ 0.5（和 M2 同一把尺）
# 有方向的刺激（滾球、放新東西、快速逼近、戳、摸頭等有位置的）
DIRECTIONAL = frozenset({"roll_ball", "novel_object", "loom", "poke", "pet_head"})
# 沒有方向的刺激（拍手、巨響、叫名字、打招呼）：3 秒內有沒有反射
NON_DIRECTIONAL = frozenset({"clap", "bang", "call_name", "hello"})
# L1 poke (SynthWorld ``poke(site="left")``): the world's truth is "the left side was poked", so
# the direction is the duck's left, 0.4 m from the trunk, fixed in the world (the distance only
# places the point; the direction is what is judged). Choice: MACHINE-VERIFY-v2-CHOICES.md §二.3.
POKE_SITE_BEARING_RAD = {"left": math.pi / 2.0, "right": -math.pi / 2.0, "back": math.pi}
POKE_SOURCE_DIST_M = 0.4
# the rule can be met only if a turn of at least M1_TURN_MIN_DEG exists that also leaves at most
# M1_ERR_SHRINK of the onset error: |turn - e0| <= 0.5 |e0| needs turn <= 1.5 |e0|. Below that
# the rule is undefined for the trial (unknown), not "did not notice".
M1_MIN_ONSET_ERR_RAD = math.radians(M1_TURN_MIN_DEG) / (1.0 + M1_ERR_SHRINK)

# ── M3（不變）；L2 的很大聲 ───────────────────────────────────────────────────────
LOUD_HEAD_DB_MIN = 90.0  # 頭的位置 ≥ 90 dB
L2_BANG_DB1M = 100.0  # the world's own ``stimulate`` path, its ``db1m`` parameter

# ── M4 說真話（補）──────────────────────────────────────────────────────────────
# a remembered place has no width: the touchable distance is the reach + 0.1 m
PLACE_ARRIVE_BOUND_M = BEAK_REACH_M + M4_ARRIVE_MARGIN_M

# ── M5 收尾（定義寫進規格）──────────────────────────────────────────────────────
# the reflexes that take hold of the body (neuro.reflex.HOLDS); read from the ``reflex`` column
HOLD_REFLEXES = frozenset(
    {"righting", "lifted", "drop_stop", "startle", "withdraw", "bump_stop", "freeze"}
)

# ── M6 同一隻鴨（改）────────────────────────────────────────────────────────────
LIFE_SECONDS = 7200.0  # 模擬 2 小時
LIFE_HZ = 5  # 5 Hz
STIM_EVERY_S = 600.0  # 每 10 分鐘一次刺激
LIFE_STIM_ORDER = (  # 9 種輪流
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
STIMULI_PER_RUN = int(LIFE_SECONDS // STIM_EVERY_S) - 1  # 10, 20, ..., 110 min
DAY_MINUTE0 = 540.0  # 09:00: SynthWorld's default, the existing personality test's start
NIGHT_MINUTE0 = 1350.0  # 22:30: Duck Core RHYTHM["asleep_from_min"]
DAY_WORLD_SEED0 = 100  # the existing personality test's world seed is 100 + the brain seed
NIGHT_WORLD_SEED0 = 200
RUNS = ("day", "night")
M6_ACC_MIN = 0.75  # 猜對 ≥ 75%
M6_EACH_MIN = 0.50  # 每個個性自己被猜對 ≥ 50%
M6_SIG_MIN = 0.75  # 每個個性 ≥ 75% 的種子要出現招牌動作
LATENCY_CAP_S = WINDOW_S  # no response inside the window counts as this many seconds
