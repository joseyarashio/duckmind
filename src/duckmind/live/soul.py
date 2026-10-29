"""The duck's "soul" for the cloud brain: the system instruction and the periodic status text.

Both are written in 繁體中文 (the duck's own language of thought, and the one the family speaks)
and built from data the duck already has: its genome summary (stage, nature, temperament axes,
favourite call, habit), the learner's view (places and things it has learnt), its drives and
what it is doing now. The model reads them; nothing here changes the model.

What the soul tells the model, in short: it is the duck's brain, not a person; the duck never
speaks human words (its mouth is the official voice bank); it cannot move the body, only
propose (the five tools of :mod:`duckmind.live.tools`); its own calls come back through its
microphone as an echo and are not people; it never invents what it did not hear or see; and its
own ear is reliable for its own name where the transcript of a name is not: when the ear note
(:data:`EAR_NAME_HINT`) arrives the person is calling the duck, whatever name the transcript shows.

``SOUL_VERSION`` goes into the cassette key: change the template, bump it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.live.tools import BANK_TAGS

SOUL_VERSION = "soul-2"

# What the duck itself tells the model when its own ear (the hearing model's on-device keyword
# spotting, ``SoundSense.kind == "name"``) reports that its name was called. The model's transcript
# of the name is unreliable (蛋黃 comes back as 郑华 / 大华 / 段华…), so the duck's ear says it. One short
# text update, sent like the status (``turn_complete`` False). The system instruction (soul-2) tells
# the model what this text means (the exact words are put into it from here, so they cannot drift).
EAR_NAME_HINT = "（我自己的耳朵：剛剛有人叫我的名字）"

DRIVE_ZH: dict[str, str] = {
    "social": "想有人陪",
    "talk": "想說話",
    "bond": "親近",
    "curiosity": "好奇",
    "fear": "害怕",
    "energy": "精神",
    "sleep_pressure": "想睡",
    "play": "想玩",
    "comfort": "舒服",
    "boredom": "無聊",
    "restless": "坐不住",
    "alertness": "警覺",
}
DRIVE_ORDER: tuple[str, ...] = tuple(DRIVE_ZH)

NATURE_ZH: dict[str, str] = {
    "songbird": "愛唱歌型：高興、想說話的時候愛用叫聲表達，叫得比較多",
    "explorer": "好奇型：對新東西、新聲音很有興趣，愛用疑問的叫聲（inquire）去探問",
    "snuggler": "依偎型：愛靠近家人，安心的時候常發咕咕聲（coo），遇到陌生的聲音會先躲一躲",
    "dreamer": "悠哉型：慢慢來，不常主動叫，常常只是靜靜看著",
}
AXIS_WORDS: dict[str, tuple[str, str]] = {  # (low, high)
    "boldness": ("膽子小，陌生的聲音和東西要先看一看、熟了才肯靠近", "膽子大，不太怕新的聲音"),
    "activity": ("不太愛動，大部分時間安安靜靜", "很愛動，常常想去走走"),
    "sociability": ("不太黏人，不常主動找人說話", "很黏人，愛找人聊"),
    "exploration": ("對新東西沒那麼好奇", "對新東西特別好奇"),
}
LOW, HIGH = 0.3, 0.7

CLAUSES = (
    "你是一隻小鴨的「腦」，牠的名字叫{name}。你不是這隻鴨，你是牠耳朵和眼睛後面幫牠想事情的那一塊。",
    "",
    "【最重要的規則】",
    "1. 小鴨不會說人話。牠的嘴只有 Pollen 的鴨叫聲庫（{tags}）。所以你絕對不要用聲音或文字回答人"
    "類：不要講話、不要說明、不要打招呼，連一個字都不要。你想表達，只能呼叫工具；沒有事要做，"
    "就完全安靜，什麼都不輸出。",
    "2. 你不能直接控制小鴨的身體。你只能提議：叫（vocalize）、注意（attend）、認人（recognize）、"
    "記住（remember）、告訴身體你的感覺（feel）。要不要照做，由小鴨的身體自己決定。",
    "3. 小鴨自己叫的時候，聲音會從喇叭傳回麥克風，那是回音，不是人。聽起來像鴨叫聲、或是小鴨"
    "剛叫完馬上出現的聲音，一律當作回音：不要回應，不要呼叫任何工具。",
    "4. 只根據你真的聽到、看到的做事，不要編造。沒有人說話、聽不清楚、或是人在說別人的事，就"
    "不要呼叫工具。",
    "5. 名字是家人給的資料：照家人說的寫（繁體中文），不要自己改，不認識的人不要猜名字。",
    "6. 你聽到的文字常常不準，尤其是名字：{name}常被聽成別的名字（例如「大華」「鄭華」「段華」）。"
    "小鴨自己的耳朵聽自己的名字很準，比你聽到的文字可靠。",
    "",
    "【怎麼回應】",
    "- 有人叫{name}：通常用 vocalize 回一個 greet（打招呼）；個性害羞就小聲一點、慢一點。",
    "- 如果我傳來「{ear}」這句話，那是小鴨自己的耳朵說的：有人在叫{name}。收到它，就當作"
    "是在叫{name}，不管你聽到的文字顯示的是什麼名字；像一隻小鴨那樣回應：呼叫 vocalize，"
    "回一個短短的叫聲（greet 打招呼，或 chirp、inquire 輕輕問一聲，看小鴨的個性），強度"
    "輕一點、不要重複。這句話比你聽到的文字早到或晚到都一樣算數。",
    "- 有人說你好、嗨：vocalize 一個 greet，強度看小鴨的個性；也可以 attend（person）看向說話的人。",
    "- 家人說「這是某某」介紹一個人：呼叫 recognize 記下名字，也可以 remember 一句話。",
    "- 有人問「你看到什麼」：只用 attend 指出畫面裡真的有的東西；畫面裡沒有就不要編，"
    "用 feel 表示有點疑惑就好。",
    "- 有人叫小鴨不要叫了或安靜：不要再 vocalize，呼叫 feel（talk 設 0）。",
    "- 沒有收到「{ear}」這句話，而且聽到的明顯是另一個名字（例如叫小美吃飯了）、聊天、電視聲音："
    "不是在叫{name}，不要 vocalize，最多 attend（sound）。",
    "- 不用每一句話都叫。叫的時候，強度（level）和次數要像一隻小鴨：短、輕、不重複。",
    "- 呼叫工具時只用上面給的工具和欄位。",
    "",
    "【小鴨是誰】",
)
TAIL = (
    "",
    "之後我會不時用文字告訴你小鴨最新的狀態（心情、驅力、正在做什麼、認得誰、眼睛看到什麼）。"
    "那只是通知，不需要回應；它會影響你怎麼提議，但你還是只呼叫工具。只有上面「怎麼回應」"
    f"說的那句「{EAR_NAME_HINT}」要照那裡說的處理。",
)


@dataclass(frozen=True)
class SoulInputs:
    """What the soul is built from (all plain data; names are data and are never translated)."""

    name: str
    genome: Mapping[str, Any] = field(default_factory=dict)  # neuro.genome.Genome.summary()
    learner: Mapping[str, Any] = field(default_factory=dict)  # neuro.learn.Learner.view()
    drives: Mapping[str, float] = field(default_factory=dict)  # neuro.drives.Drives.as_dict()
    known_people: tuple[str, ...] = ()
    behaviour: str = ""


def _level(value: float) -> int:
    return max(0, min(10, int(round(10.0 * float(value)))))


def drives_line(drives: Mapping[str, float]) -> str:
    """Drives as plain words and 0–10 numbers, in a fixed order (so equal states are equal text)."""
    parts = [
        f"{DRIVE_ZH[key]} {_level(drives[key])}"
        for key in DRIVE_ORDER
        if key in drives and isinstance(drives[key], (int, float))
    ]
    return "、".join(parts)


def _mood_word(valence: float) -> str:
    if valence >= 0.5:
        return "很開心"
    if valence >= 0.15:
        return "心情不錯"
    if valence > -0.15:
        return "心情普通"
    if valence > -0.5:
        return "有點不舒服"
    return "很不舒服"


def temperament_lines(genome: Mapping[str, Any]) -> list[str]:
    """The duck's temperament in words: its nature, the axes that stand out, favourite call, habit."""
    lines: list[str] = []
    primary = str(genome.get("primary") or "")
    stage = str(genome.get("stage_zh") or "")
    branch = str(genome.get("branch_zh") or "")
    head = "、".join(
        part for part in (stage, branch or str(genome.get("primary_zh") or "")) if part
    )
    if head:
        lines.append(f"目前是{head}。")
    if primary in NATURE_ZH:
        lines.append(f"天性是{NATURE_ZH[primary]}。")
    axes = genome.get("axes")
    if isinstance(axes, Mapping):
        for axis, (low, high) in AXIS_WORDS.items():
            value = axes.get(axis)
            if not isinstance(value, (int, float)):
                continue
            if value <= LOW:
                lines.append(f"個性：{low}。")
            elif value >= HIGH:
                lines.append(f"個性：{high}。")
    favourite = genome.get("favorite_tag")
    if isinstance(favourite, str) and favourite in BANK_TAGS:
        lines.append(f"最常用的叫法是 {favourite}。")
    habit = genome.get("habit_zh")
    if habit:
        lines.append(f"小習慣：{habit}。")
    return lines


def learner_lines(view: Mapping[str, Any]) -> list[str]:
    """What the duck has learnt, in a line or two (counts and kinds; no coordinates)."""
    lines: list[str] = []
    places = view.get("places")
    if isinstance(places, Sequence) and not isinstance(places, str) and places:
        liked = sum(1 for p in places if isinstance(p, Mapping) and (p.get("value") or 0) > 0.2)
        lines.append(f"牠記得家裡 {len(places)} 個地方，其中 {liked} 個牠喜歡。")
    things = view.get("things")
    if isinstance(things, Sequence) and not isinstance(things, str) and things:
        kinds = sorted(
            {str(t.get("kind")) for t in things if isinstance(t, Mapping) and t.get("kind")}
        )
        if kinds:
            lines.append("牠見過的東西：" + "、".join(kinds) + "。")
    return lines


def build_soul(inputs: SoulInputs) -> str:
    """The system instruction. Equal inputs give the same text."""
    out = [
        clause.format(name=inputs.name, tags="、".join(BANK_TAGS), ear=EAR_NAME_HINT)
        if "{" in clause
        else clause
        for clause in CLAUSES
    ]
    out += temperament_lines(inputs.genome)
    out += learner_lines(inputs.learner)
    if inputs.known_people:
        out.append("牠已經認得的人：" + "、".join(inputs.known_people) + "。")
    else:
        out.append("牠還沒有認得的人（家人還沒有介紹過）。")
    line = drives_line(inputs.drives)
    if line:
        out.append("剛開始的驅力（0 到 10）：" + line + "。")
    if inputs.behaviour:
        out.append(f"剛開始正在：{inputs.behaviour}。")
    out += TAIL
    return "\n".join(out)


@dataclass(frozen=True)
class ContextSnapshot:
    """What the duck tells the model every few seconds (a notice, not a question)."""

    behaviour: str = ""
    valence: float | None = None
    drives: Mapping[str, float] = field(default_factory=dict)
    known_people: tuple[str, ...] = ()
    in_view: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    # how close the duck feels to each person it knows (name → 0..1: LV4's 親近); shown as 0–10
    bonds: Mapping[str, float] = field(default_factory=dict)

    def text(self) -> str:
        lines = ["【小鴨的狀態通知，不需要回應】"]
        lines.append(f"正在做：{self.behaviour or '還沒決定'}")
        if self.valence is not None:
            lines.append(f"心情：{_mood_word(float(self.valence))}")
        drive = drives_line(self.drives)
        if drive:
            lines.append(f"驅力（0 到 10）：{drive}")
        known = [
            f"{name}（親近 {_level(self.bonds[name])}）" if name in self.bonds else name
            for name in self.known_people
        ]
        lines.append("認得的人：" + ("、".join(known) if known else "還沒有"))
        lines.append(
            "眼睛現在看到：" + ("、".join(self.in_view) if self.in_view else "沒有特別的東西")
        )
        lines += [f"另外：{note}" for note in self.notes]
        return "\n".join(lines)


__all__ = [
    "DRIVE_ZH",
    "EAR_NAME_HINT",
    "SOUL_VERSION",
    "ContextSnapshot",
    "SoulInputs",
    "build_soul",
    "drives_line",
    "learner_lines",
    "temperament_lines",
]
