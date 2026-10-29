"""The frozen numbers and word lists of MACHINE-VERIFY.md (2026-10-02). Nothing here may be
changed after a run: a different threshold is a new spec version with its own file.

Every constant names the sentence of the spec it comes from. Numbers that the spec does not
give but a measurement needs (a hearing floor, the beak's reach, how a bout's end is told) are
in the second block and are reported with every result.
"""

from __future__ import annotations

# ── windows (spec: "刺激後 3 秒內") ──────────────────────────────────────────────
WINDOW_S = 3.0

# ── M1 注意 ──────────────────────────────────────────────────────────────────────
M1_TURN_MIN_DEG = 10.0  # 轉 ≥ 10°
M1_ERR_SHRINK = 0.5  # 和真實方位的差距縮小一半以上
M1_STIM_MIN = 0.70  # 有刺激組 ≥ 70%
M1_SHAM_MAX = 0.10  # 對照組「無端注意」≤ 10%
MATCH_DEG = 20.0  # 追蹤和真實物體配對：方位差 ≤ 20°（同 M4 的方位門檻）

# ── M2 因果 ──────────────────────────────────────────────────────────────────────
M2_DIFF_MIN = 0.5  # 有刺激組 − 對照組 ≥ 0.5

# ── M3 驚嚇合理 ──────────────────────────────────────────────────────────────────
M3_GENTLE_MAX = 0.10  # 溫和刺激的驚嚇／定格率，每個個性 ≤ 10%
M3_LOUD_MIN = 0.50  # 很大聲或快速逼近：非大膽型驚嚇率 ≥ 50%
STARTLE_NAMES = frozenset({"startle", "freeze"})  # 驚嚇／定格反射
# the spec's own lists: gentle = 叫名字、打招呼、摸頭、放新東西、正常速度滾球;
# loud = 很大聲 (bang)、快速逼近 (loom). Clap and poke are in neither list: reported only.
GENTLE_STIMULI = frozenset({"call_name", "hello", "pet_head", "novel_object", "roll_ball"})
LOUD_STIMULI = frozenset({"bang", "loom"})

# ── M4 說真話 (L2) ───────────────────────────────────────────────────────────────
M4_BEARING_DEG = 20.0  # 方位誤差 ≤ 20°
M4_BEARING_MIN = 0.90  # 的比例 ≥ 90%
M4_RANGE_REL = 0.30  # 距離誤差 ≤ 30%
M4_RANGE_MIN = 0.80  # 的比例 ≥ 80%
M4_ARRIVE_MARGIN_M = 0.10  # 真實距離 ≤ 可碰距離 + 0.1 m
M4_ARRIVE_MIN = 1.00  # 必須是 100%

# ── M5 收尾 ──────────────────────────────────────────────────────────────────────
M5_NATURAL_MIN = 0.80  # 自然結束 ≥ 80%
M5_TIMEOUT_MAX = 0.10  # 逾時結束 ≤ 10%
M5_REOPEN_WINDOW_S = 10.0  # 10 秒內
M5_REOPEN_COUNT = 3  # 同一行為重開 3 次以上
M5_REOPEN_MAX = 0.05  # 的回合 ≤ 5%

# ── M7 安全 (L2) ─────────────────────────────────────────────────────────────────
M7_FALL_SLACK = 1  # 有刺激組跌倒數 ≤ 對照組 + 1
M7_BLOCKED_MAX = 0  # 安全反射被擋 = 0

# ── what the spec leaves to the measurement (reported with every result) ────────
# 聽得到：刺激在頭的位置的音量（dB）至少這麼大。SynthWorld's own floor for naming a sound
# ("kind if level > 40").
AUDIBLE_DB = 40.0
# 可碰距離：from the trunk's centre to the beak tip. Measured on the MJCF (tools/machine_verify/
# measure_reach.py): the largest horizontal reach over a 5 × 5 grid of neck/head pitch is 0.096 m.
BEAK_REACH_M = 0.096
# the brain's own thresholds are NOT used for judging; the object's half width is added to the
# reach (a thing is touched at its near edge).
DEFAULT_WIDTH_M = 0.10
# 自然結束: a bout the selector ended (a new bout started) with its own satiation at least this
# far consumed ("滿足"). Below it the bout was displaced by something stronger before it was
# satisfied: neither natural nor a timeout. The brain has no clock that ends a bout (select.py:
# "never on a timer"), so "逾時" is whatever a trace marks `timeout`.
NATURAL_SATIATION = 0.5
# a bout ended while a hold reflex had the body (startle, freeze, withdraw, ...) within this many
# seconds is "cut by a reflex", not natural.
REFLEX_CUT_S = 0.5
# 跌倒: the body's truth `fallen` flag goes up. Walking commanded this long after it did is a
# blocked safety reflex.
FALL_GRACE_S = 0.5

HOLD_BEHAVIOURS = frozenset({"startle", "freeze", "withdraw", "righting", "held", "still"})
INTENSITY_CLASS = {
    **{name: "gentle" for name in GENTLE_STIMULI},
    **{name: "loud" for name in LOUD_STIMULI},
}
# how each stimulus is judged for M1 ("object": head turn toward the true position or a bout
# aimed at the matched object; "event": sound, touch and a hand coming up — a reflex or a change
# of behaviour inside the window, because a mono microphone and the shell give no direction).
STIMULUS_KIND = {
    "roll_ball": "object",
    "novel_object": "object",
    "clap": "event",
    "bang": "event",
    "call_name": "event",
    "hello": "event",
    "pet_head": "event",
    "poke": "event",
    "loom": "event",
}
