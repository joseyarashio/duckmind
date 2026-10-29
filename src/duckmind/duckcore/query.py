"""Querying the duck's distilled memory, with every answer tied to its evidence."""

from __future__ import annotations

from typing import Any

from duckmind.duckcore import content
from duckmind.duckcore.engine import DuckCore
from duckmind.duckcore.fixed import SCALE
from duckmind.duckcore.memory import Fact
from duckmind.duckcore.narrate import describe

SYNONYMS: dict[str, tuple[str, ...]] = {
    "摸": ("摸", "搔", "頭頂"),
    "搔": ("摸", "搔", "頭頂"),
    "抱": ("抱",),
    "玩": ("玩", "手把", "遊戲", "公尺"),
    "手把": ("手把", "玩"),
    "歌": ("歌", "片語"),
    "唱": ("歌", "片語", "唱"),
    "片語": ("片語", "歌"),
    "主人": ("主人", "招呼"),
    "招呼": ("招呼", "主人"),
    "名字": ("名字", "叫我"),
    "叫": ("叫我", "名字"),
    "怕": ("怕", "縮", "手"),
    "手": ("手",),
    "電視": ("電視", "聲音"),
    "聲音": ("聲音", "電視"),
    "喜歡": ("喜歡",),
    "個性": ("型", "個性"),
    "型": ("型", "個性"),
    "怪癖": ("怪癖", "總是", "歪"),
    "歪頭": ("歪",),
    "第一": ("第一",),
    "晚上": ("晚上",),
    "早上": ("早上",),
    "時段": ("時段", "晚上", "早上", "下午"),
    "紀念": ("紀念", "公尺"),
    "驚喜": ("稀有", "噴嚏", "打嗝", "咻"),
    "噴嚏": ("噴嚏",),
    "長大": ("里程碑", "幼鴨", "少年鴨", "成鴨", "長成"),
    "里程碑": ("里程碑",),
    "走": ("走", "公尺", "探索"),
    "散步": ("走", "公尺"),
    "探索": ("走", "探索", "到處"),
    "到處": ("走", "到處"),
    "站": ("站", "走"),
    "跌倒": ("跌倒", "翻回來"),
    "跌": ("跌倒", "翻回來"),
    "摔": ("跌倒", "翻回來"),
    "翻": ("翻回來", "跌倒"),
    "小心": ("不敢", "跌倒"),
    "不敢": ("不敢", "不適合走"),
    "角落": ("角落", "不敢"),
    "地方": ("不適合走", "不敢", "角落", "牆邊"),
}
OTHER_PEOPLE = (
    "小孩",
    "孩子",
    "兒子",
    "女兒",
    "訪客",
    "客人",
    "伴侶",
    "太太",
    "老婆",
    "先生",
    "老公",
    "媽媽",
    "爸爸",
    "朋友",
)
CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "song": ("歌", "唱", "片語"),
    "like": ("摸", "搔", "抱", "喜歡", "玩"),
    "bond": ("主人", "招呼"),
    "trait": ("個性", "型", "性格"),
    "quirk": ("怪癖", "習慣", "總是"),
    "keepsake": ("紀念", "公尺"),
    "rare": ("驚喜", "稀有", "噴嚏", "打嗝"),
    "milestone": ("長大", "里程碑", "第一"),
    "pattern": ("時段", "晚上", "早上", "電視", "聲音"),
    "habit": ("怕", "手", "跌", "摔", "小心", "不敢", "地方", "角落"),
}
CATEGORY_ZH = {
    "like": "喜好",
    "bond": "關係",
    "habit": "習慣",
    "trait": "個性",
    "quirk": "怪癖",
    "milestone": "里程碑",
    "song": "聲音",
    "pattern": "生活規律",
    "rare": "稀有時刻",
    "keepsake": "紀念",
}


def _terms(question: str) -> list[str]:
    terms: list[str] = []
    for key, expansions in SYNONYMS.items():
        if key in question:
            terms.extend(expansions)
    stripped = question
    for noise in (
        "你記得",
        "記得",
        "查詢",
        "查",
        "關於",
        "有關",
        "的事",
        "嗎",
        "？",
        "?",
        "什麼",
        "我",
        "你",
    ):
        stripped = stripped.replace(noise, " ")
    for chunk in stripped.split():
        if len(chunk) >= 1 and chunk not in terms:
            terms.append(chunk)
    return terms


def fact_json(core: DuckCore, fact: Fact, *, cites: int = 5) -> dict[str, Any]:
    evidence = []
    for event_id in fact.evidence[-cites:]:
        event = core.log.get(event_id)
        if event is not None:
            evidence.append(
                {
                    "event_id": event_id,
                    "at": event.occurred_at.strftime("%m/%d %H:%M"),
                    "text": describe(event),
                }
            )
    return {
        "fact_id": fact.fact_id,
        "category": fact.category,
        "category_zh": CATEGORY_ZH.get(fact.category, fact.category),
        "subject": fact.subject,
        "predicate": fact.predicate,
        "text": fact.text,
        "strength": fact.strength * 100 // SCALE,
        "recall": fact.recall(),
        "evidence_count": len(fact.evidence),
        "evidence": evidence,
        "first_at": fact.first_at.strftime("%m/%d") if fact.first_at else None,
        "last_at": fact.last_at.strftime("%m/%d") if fact.last_at else None,
        "half_life_days": fact.half_life_days,
        "generator": "reducer:distill-v1（確定性，不是 LLM）",
    }


def search(core: DuckCore, question: str, limit: int = 8) -> dict[str, Any]:
    memory = core.memory()
    terms = _terms(question)
    scored: list[tuple[int, Fact]] = []
    for fact in memory.facts.values():
        haystack = f"{fact.text}{fact.predicate}{fact.subject}{CATEGORY_ZH.get(fact.category, '')}"
        hits = sum(1 for term in terms if term and term in haystack)
        if hits or not terms:
            boost = (
                SCALE * 3 // 2
                if any(word in question for word in CATEGORY_HINTS.get(fact.category, ()))
                else 0
            )
            scored.append((hits * SCALE + boost + fact.strength, fact))
    scored.sort(key=lambda item: (-item[0], item[1].fact_id))
    facts = [fact_json(core, fact) for _, fact in scored[:limit]]
    diary_hits = []
    for day in reversed(memory.diary):
        for line in day.lines:
            if terms and any(term in line.text for term in terms):
                diary_hits.append(
                    {"date": day.date, "gd": day.gd, "text": line.text, "cites": line.cites[:5]}
                )
    phrases = [
        {"code": code, "text": item["text"], "gen": item["gen"], "promoted": item["promoted"]}
        for code, item in sorted(core.s.phrases.items())
        if any(term in item["text"] or term == code for term in terms)
    ]
    mentions_other = [word for word in OTHER_PEOPLE if word in question]
    if mentions_other and not facts:
        household = sorted(
            (
                fact
                for fact in memory.facts.values()
                if fact.subject in {"家人", "我們家", "家裡", "我們"}
            ),
            key=lambda fact: (-fact.strength, fact.fact_id),
        )
        facts = [fact_json(core, fact) for fact in household[:3]]
    if mentions_other:
        answer = (
            f"我分不出誰是「{mentions_other[0]}」。我只知道「家裡有人」摸我、陪我玩；"
            "只有主人用手機跟我說話時，我才知道那是主人。"
        )
        if facts:
            answer += f" 跟這有關、我記得的是：{facts[0]['text']}"
    elif facts and (not terms or scored[0][0] >= SCALE):
        top = facts[0]
        answer = f"我記得：{top['text']}（{top['recall']}，記憶強度 {top['strength']}%，有 {top['evidence_count']} 件事支持）"
    else:
        answer = "我不記得有這件事。（我只記得真的發生過、而且有紀錄的事。）"
    return {
        "question": question,
        "terms": terms,
        "answer": answer,
        "facts": facts,
        "diary": diary_hits[:limit],
        "phrases": phrases[:limit],
        "honesty": "每一句都連回 L1 事件；記憶強度是確定性的公式，不是 LLM 猜的。",
    }


def diary(core: DuckCore, limit: int = 30) -> list[dict[str, Any]]:
    memory = core.memory()
    days = []
    for day in memory.diary[-limit:]:
        days.append(
            {
                "date": day.date,
                "gd": day.gd,
                "positive": day.positive,
                "negative": day.negative,
                "lines": [
                    {"text": line.text, "cites": line.cites[:8], "tone": line.tone}
                    for line in day.lines
                ],
            }
        )
    return list(reversed(days))


def why(core: DuckCore, question: str) -> dict[str, Any]:
    for verb in content.VERBS:
        if verb.zh in question:
            result = core.why_verb(verb.verb_id)
            result["kind"] = "verb"
            return result
    if "走" in question:
        return {"kind": "walk", "found": True, "text": core.walk_reason()}
    code, text = core.still_reason()
    return {"kind": "still", "code": code, "found": True, "text": text}
