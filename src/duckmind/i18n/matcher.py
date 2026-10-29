"""Map the frozen engine's real output back to {template id, params} (from i18n:inv-core's
prototype, measured there: 100 % of 8.9 M page strings of the golden corpus and 蛋黃's copy
matched, 0 of 37,740 traced strings contradicted).

The frozen v4_0 Duck Core may never be edited (蛋黃 runs on it; phase-4 gate G1 wants identical
outputs), so the page cannot get message keys from it. This matcher sits at the API boundary:
it is built ONLY from the engine's own templates (``extract_templates.py`` over
``src/duckmind/engines/v4_0``) and turns each Traditional Chinese string the engine emits into

    {"id": template id, "params": {name: value | nested match}, "segments": [...]}

so a catalog can render it in any language. Steps, most specific first:

  1. exact     — the string is one literal of the engine (a table value, a note, a title);
  2. sounds    — the duck's own babble: its sound words (TAG_ZH) joined by "·";
  3. template  — one f-string / str.format / "+" template matches the whole string;
  4. segments  — the string is several templates glued together (``text += …``: banners,
                 diary lines), found by dynamic programming over the string;
  5. glue      — a template without words of its own (``f"{at} {summary}"``) whose every
                 field resolves;
  6. params    — every placeholder value is resolved: a number, a clock time or date, the
                 duck's name (data), babble, a literal alternative of the placeholder, a cut
                 piece of a literal (``zh[:10]``, ``removeprefix``), a list, or — recursively —
                 another template.

A result is ``full`` when every part and parameter is accounted for, ``partial`` when the
shape matched but a parameter is free text, ``none`` otherwise. Nothing here reads the event
payload: the string alone (plus the duck's name, which is data) decides.
"""

from __future__ import annotations

import bisect
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿＀-￯]")
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?%?")
TIMEISH = re.compile(
    r"(?:\d{1,2}:\d{2}(?::\d{2})?|\d{2}/\d{2}(?: \d{2}:\d{2})?|\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)"
)
LIST_SEPARATORS = ("、", " ／ ", "；", "，", " / ", ", ")
MAX_DEPTH = 99  # (recursion is bounded by the cycle guard in match)
CACHE_SIZE = 200_000  # distinct strings remembered (least recently used ones are dropped)
NUMERIC_SPECS = re.compile(r"^[<>=^]?[+ -]?#?0?\d*,?(?:\.\d+)?[dfeg%]$")
CUT_EXPR = re.compile(r"\[:\s*\d+\s*\]|removeprefix|removesuffix|rstrip|lstrip|strip\(")
# A placeholder's value: any text whose brackets pair up (two levels deep), so the regex
# engine itself looks for the split where every field is well formed.
_FLAT = r"[^（）「」()]"
_IN1 = rf"(?:（{_FLAT}*）|「{_FLAT}*」|\({_FLAT}*\))"
BALANCED = rf"(?:{_FLAT}|（(?:{_FLAT}|{_IN1})*）|「(?:{_FLAT}|{_IN1})*」|\((?:{_FLAT}|{_IN1})*\))*?"


def has_cjk(text: str) -> bool:
    return bool(CJK.search(text))


PAIRS = {"（": "）", "「": "」", "(": ")", "『": "』", "【": "】"}
CLOSERS = {v: k for k, v in PAIRS.items()}


def balanced(text: str) -> bool:
    """A placeholder's value never closes a bracket it did not open (nor leaves one open):
    otherwise the match ran across the end of one sentence into the next."""
    stack: list[str] = []
    for ch in text:
        if ch in PAIRS:
            stack.append(ch)
        elif ch in CLOSERS:
            if not stack or stack[-1] != CLOSERS[ch]:
                return False
            stack.pop()
    return not stack


def anchored(param: Any) -> int:
    """How many characters of a parameter are explained by something typed (a literal of
    the engine, a number, a time, a name, babble) rather than taken on trust."""
    if not isinstance(param, dict):
        return 0
    kind = param.get("kind")
    if kind in {"number", "time", "name", "sounds", "alt", "cut"}:
        return len(str(param.get("value", "")))
    if kind == "list":
        return sum(anchored(item) for item in param.get("items") or []) + len(
            param.get("sep", "")
        ) * max(0, len(param.get("items") or []) - 1)
    if kind == "nested":
        return int(param.get("anchored", 0))
    return 0


def complexity(param: Any) -> int:
    if not isinstance(param, dict):
        return 0
    if param.get("kind") == "list":
        return sum(complexity(item) for item in param.get("items") or [])
    if param.get("kind") == "nested":
        return int(param.get("complexity", 0))
    return 0


@dataclass
class Template:
    tid: str
    kind: str
    text: str  # rendered with {placeholders}
    parts: list[dict[str, Any]]
    site: str
    wordless: bool = False
    lits: list[str] = field(default_factory=list)  # literal chunks, in order
    regex: re.Pattern[str] | None = None
    display_glue: bool = False  # glue that joins words for people (" ", "・", "（…）"), not an id
    first: str = ""  # first literal chunk when the template starts with a literal
    anchor: str = ""  # the longest literal chunk (prefilter)
    lit_len: int = 0

    def compile(self) -> None:
        pattern = []
        for index, part in enumerate(self.parts):
            if "lit" in part:
                pattern.append(re.escape(part["lit"]))
                self.lits.append(part["lit"])
            else:
                name = f"p{index}"
                spec = part.get("spec") or ""
                alts = part.get("alts")
                if alts:
                    pieces = []
                    for alt in alts:
                        if isinstance(alt, str):
                            pieces.append(re.escape(alt))
                        else:  # an f-string shape: its words, its fields as balanced text
                            pieces.append(
                                "".join(
                                    re.escape(q["lit"]) if "lit" in q else BALANCED
                                    for q in alt["parts"]
                                )
                            )
                    pieces.sort(key=len, reverse=True)
                    pattern.append(f"(?P<{name}>{'|'.join(f'(?:{x})' for x in pieces)})")
                elif spec and NUMERIC_SPECS.match(spec):
                    pattern.append(f"(?P<{name}>[+-]?[\\d.,]+%?)")
                else:
                    pattern.append(f"(?P<{name}>{BALANCED})")
        self.regex = re.compile("".join(pattern), re.S)
        self.first = self.parts[0]["lit"] if self.parts and "lit" in self.parts[0] else ""
        self.anchor = max(self.lits, key=len) if self.lits else ""
        self.lit_len = sum(len(x) for x in self.lits)
        glue = "".join(self.lits)
        self.display_glue = bool(glue) and not re.fullmatch(r"[A-Za-z0-9_:./#=\-]+", glue)


@dataclass
class Match:
    status: str  # full | partial | none
    how: str  # exact | sounds | template | segments | glue | none
    tid: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    anchored: int = 0
    complexity: int = 0  # glue and nested splits used: the simpler reading wins a tie

    def json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "how": self.how,
            "anchored": self.anchored,
            "complexity": self.complexity,
        }
        if self.tid:
            out["id"] = self.tid
        if self.params:
            out["params"] = self.params
        if self.segments:
            out["segments"] = self.segments
        if self.unresolved:
            out["unresolved"] = self.unresolved
        return out


class Matcher:
    def __init__(
        self,
        sites: list[dict[str, Any]],
        *,
        names: tuple[str, ...] = (),
        sound_tokens: tuple[str, ...] = (),
    ) -> None:
        self.names = set(names)
        self.sound_tokens = set(sound_tokens)
        self.exact: dict[str, list[str]] = {}
        self.templates: list[Template] = []
        self.glue: list[Template] = []
        self.by_first: dict[str, list[Template]] = {}
        self.leading_ph: list[Template] = []
        self.literals_by_first: dict[str, list[str]] = {}
        for site in sites:
            if site["kind"] == "separator":
                continue
            tid = site["tid"]
            if site["kind"] == "literal":
                text = site["parts"][0]["lit"]
                if text:
                    self.exact.setdefault(text, []).append(tid)
                continue
            template = Template(
                tid,
                site["kind"],
                site["zh_hant"],
                site["parts"],
                site["site"],
                bool(site.get("wordless")),
            )
            lits = "".join(p.get("lit", "") for p in site["parts"])
            is_error = (site.get("slot") or {}).get("type") == "raise" and len(lits.strip()) >= 6
            if not is_error and (template.wordless or not has_cjk(lits)):
                template.wordless = True
                template.compile()
                self.glue.append(template)
                continue
            template.compile()
            self.templates.append(template)
        # derived literals: each character of an indexed constant ("一二三四五六日"[weekday])
        for site in sites:
            if site.get("indexed_chars"):
                for ch in site["parts"][0]["lit"]:
                    self.exact.setdefault(ch, []).append(site["tid"] + f"[{ch}]")
        for template in self.templates:
            if template.first:
                self.by_first.setdefault(template.first[0], []).append(template)
            else:
                self.leading_ph.append(template)
        for text in self.exact:
            self.literals_by_first.setdefault(text[0], []).append(text)
        # most specific first
        self.templates.sort(key=lambda t: -t.lit_len)
        for group in self.by_first.values():
            group.sort(key=lambda t: -t.lit_len)
        self.leading_ph.sort(key=lambda t: -t.lit_len)
        self.glue.sort(key=lambda t: -t.lit_len)
        # cut pieces of literals (zh[:10], removeprefix("我")): one big string to search in
        self._literal_list = sorted(self.exact, key=len, reverse=True)
        self._big = "\x00".join(self._literal_list)
        self._starts: list[int] = []
        pos = 0
        for literal in self._literal_list:
            self._starts.append(pos)
            pos += len(literal) + 1
        self._cache: OrderedDict[str, Match] = OrderedDict()
        self._busy: set[str] = set()
        self._cut_cache: dict[str, dict[str, Any] | None] = {}
        self.cache_size = CACHE_SIZE
        self.lock = threading.RLock()

    def set_names(self, names: tuple[str, ...] | list[str] | set[str]) -> None:
        """The names that are data (the duck's name): a changed set clears the cache."""
        new = {n for n in names if n}
        if new != self.names:
            with self.lock:
                self.names = new
                self._cache.clear()

    # ---------------------------------------------------------------- parameters
    def _param(self, value: str, part: dict[str, Any], depth: int) -> tuple[Any, bool]:
        """(resolved param, ok)."""
        if value == "":
            return {"kind": "empty"}, True
        alts = part.get("alts")
        if alts and value in [a for a in alts if isinstance(a, str)]:
            alt_ids = self.exact.get(value)
            return {
                "kind": "alt",
                "value": value,
                **({"id": self._pick(alt_ids, part)} if alt_ids else {}),
            }, True
        if NUMBER.fullmatch(value) or re.fullmatch(r"[+-]?[\d.,]+%?", value):
            return {"kind": "number", "value": value}, True
        if TIMEISH.fullmatch(value):
            return {"kind": "time", "value": value}, True
        if value in self.names:
            return {"kind": "name", "value": value}, True
        if self.is_sounds(value):
            return {"kind": "sounds", "value": value}, True
        if (
            not has_cjk(value)
            and len(value) <= 40
            and re.fullmatch(r"[\w.:+\-/=°]+", value)
            and re.search(r"[A-Za-z]", value)
        ):
            return {"kind": "code", "value": value}, True  # an id, a unit, a tag (not words)
        if depth < MAX_DEPTH:
            nested = self.match(value, depth + 1)
            if nested.status == "full":
                out = nested.json()
                if nested.how == "exact":
                    ids = self.exact.get(value) or []
                    if len(ids) > 1:
                        out["id"] = self._pick(ids, part)
                return {"kind": "nested", **out}, True
            items = self._list(value, depth)
            if items is not None:
                return items, True
            expr = str(part.get("expr") or "")
            if CUT_EXPR.search(expr):
                # a word taken off the front (text.removeprefix("我")): put it back and try again
                for removed in re.findall(r"removeprefix\(\s*[\"']([^\"']+)[\"']\s*\)", expr):
                    again = self.match(removed + value, depth + 1)
                    if again.status == "full":
                        return {
                            "kind": "cut",
                            "op": f"removeprefix {removed}",
                            "value": value,
                            **again.json(),
                        }, True
                cut = self._cut(value)
                if cut is not None:
                    return cut, True
            if nested.status == "partial":
                return {"kind": "nested", **nested.json()}, False
        return {"kind": "text", "value": value}, False

    def _pick(self, ids: list[str], part: dict[str, Any] | None) -> str:
        """Among identical words written at several sites, the one a placeholder most likely
        holds: a table row when the field reads a table or a .zh (a verb's name, not a banner
        title that happens to be the same words)."""
        if len(ids) == 1 or part is None:
            return ids[0]
        expr = str(part.get("expr") or "")
        if re.search(
            r"_ZH|\.zh\b|'zh'|\"zh\"|\.short\b|content\.|TAG|VERB|MOMENT|STAGE|BRANCH", expr
        ):
            for tid in ids:
                if re.search(r":[a-z_.]+\.[A-Z][A-Z_]+\.", tid):
                    return tid
        return ids[0]

    def _cut(self, value: str) -> dict[str, Any] | None:
        """A literal cut short or with a word taken off (``zh[:10]``, ``removeprefix("我")``)."""
        if value in self._cut_cache:
            return self._cut_cache[value]
        found = None
        at = self._big.find(value)
        while at >= 0:
            index = bisect.bisect_right(self._starts, at) - 1
            literal = self._literal_list[index]
            if len(literal) > len(value):
                op = "prefix" if literal.startswith(value) else "substring"
                found = {
                    "kind": "cut",
                    "op": op,
                    "id": self.exact[literal][0],
                    "of": literal,
                    "value": value,
                }
                if op == "prefix":
                    break
            at = self._big.find(value, at + 1)
        self._cut_cache[value] = found
        return found

    def is_sounds(self, value: str) -> bool:
        """The duck's babble: sound words (啾, 嗯？, …) joined by the middle dot."""
        if not self.sound_tokens:
            return False
        pieces = re.split(r"[·・]", value)
        return bool(pieces) and all(piece in self.sound_tokens for piece in pieces)

    def _list(self, value: str, depth: int) -> dict[str, Any] | None:
        for sep in LIST_SEPARATORS:
            if sep in value:
                items = value.split(sep)
                resolved = []
                for item in items:
                    param, ok = self._param(item.strip(), {}, depth + 1)
                    if not ok:
                        break
                    resolved.append(param)
                else:
                    return {"kind": "list", "sep": sep, "items": resolved}
        return None

    # ---------------------------------------------------------------- whole strings
    def _try_template(
        self, template: Template, text: str, depth: int, pos: int = 0, end: int | None = None
    ) -> Match | None:
        assert template.regex is not None
        found = template.regex.fullmatch(text, pos, len(text) if end is None else end)
        if not found:
            return None
        values = [found.group(f"p{i}") for i, part in enumerate(template.parts) if "ph" in part]
        if not all(balanced(value) for value in values):
            return None
        params: dict[str, Any] = {}
        unresolved: list[str] = []
        score = template.lit_len
        cost = 1 if template.wordless else 0
        for index, part in enumerate(template.parts):
            if "ph" not in part:
                continue
            value = found.group(f"p{index}")
            param, ok = self._param(value, part, depth)
            params[part["ph"]] = param
            score += anchored(param)
            cost += complexity(param)
            if not ok:
                unresolved.append(part["ph"])
        how = "glue" if template.wordless else "template"
        return Match(
            "partial" if unresolved else "full",
            how,
            template.tid,
            params,
            unresolved=unresolved,
            anchored=score,
            complexity=cost,
        )

    def match(self, text: str, depth: int = 0) -> Match:
        with self.lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached
            if text in self._busy or depth > 8:
                return Match("none", "none")  # (a template inside itself)
            self._busy.add(text)
            try:
                result = self._match(text, 0)
            finally:
                self._busy.discard(text)
            self._cache[text] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            if len(self._cut_cache) > self.cache_size:
                self._cut_cache.clear()
            return result

    def _match(self, text: str, depth: int) -> Match:
        if text in self.exact:
            return Match("full", "exact", self.exact[text][0], anchored=len(text))
        if self.is_sounds(text):
            return Match(
                "full",
                "sounds",
                None,
                {"sounds": {"kind": "sounds", "value": text}},
                anchored=len(text),
            )
        best_partial: Match | None = None
        best_full: Match | None = None
        for template in self.templates:
            if template.anchor and template.anchor not in text:
                continue
            if not all(lit in text for lit in template.lits):
                continue
            found = self._try_template(template, text, depth)
            if found is None:
                continue
            if found.status == "full":
                if best_full is None or (found.anchored, -found.complexity) > (
                    best_full.anchored,
                    -best_full.complexity,
                ):
                    best_full = found
                if best_full.anchored >= len(text) and best_full.complexity == 0:
                    break
            elif best_partial is None:
                best_partial = found
        if best_full is not None and best_full.anchored >= len(text) and best_full.complexity == 0:
            return best_full
        seg = self._segments(text, depth)
        if (
            seg is not None
            and seg.status == "full"
            and (
                best_full is None
                or (seg.anchored, -seg.complexity) > (best_full.anchored, -best_full.complexity)
            )
        ):
            return seg
        if best_full is not None:
            return best_full
        glue = self._glue(text, depth)
        if glue is not None:
            return glue
        items = self._list(text, depth) if depth == 0 else None
        if items is not None:
            # a list the engine joined itself (the song: phrases joined by " ／ ")
            return Match("full", "list", None, {"items": items}, anchored=anchored(items))
        if best_partial is not None:
            return best_partial
        if seg is not None:
            return seg
        return Match("none", "none")

    def _glue(self, text: str, depth: int) -> Match | None:
        if depth >= MAX_DEPTH:
            return None
        for template in self.glue:
            if not template.display_glue:
                continue
            if not all(lit in text for lit in template.lits):
                continue
            found = self._try_template(template, text, depth)
            if found is not None and found.status == "full":
                kinds = {p.get("kind") for p in found.params.values() if isinstance(p, dict)}
                if kinds & {"nested", "list", "sounds", "cut", "alt"}:  # words somewhere in it
                    return found
        return None

    # ---------------------------------------------------------------- concatenations
    def _segments(self, text: str, depth: int) -> Match | None:
        """Split ``text`` into literals/templates glued together (DP from the right)."""
        n = len(text)
        if n > 600 or depth >= MAX_DEPTH:
            return None
        # best[i]: (quality, pieces) for text[i:]; quality = (unresolved count, piece count)
        best: list[tuple[tuple[int, int, int, int], list[dict[str, Any]]] | None] = [None] * (n + 1)
        best[n] = ((0, 0, 0, 0), [])
        for i in range(n - 1, -1, -1):
            options: list[tuple[tuple[int, int, int, int], list[dict[str, Any]]]] = []
            for lit in self.literals_by_first.get(text[i], ()):
                j = i + len(lit)
                if j <= n and best[j] is not None and text.startswith(lit, i):
                    q, rest = best[j]  # type: ignore[misc]
                    options.append(
                        (
                            (q[0], q[1] - len(lit), q[2], q[3] + 1),
                            [
                                {
                                    "id": self.exact[lit][0],
                                    "how": "exact",
                                    "text": lit,
                                    "anchored": len(lit),
                                },
                                *rest,
                            ],
                        )
                    )
            candidates = [t for t in self.by_first.get(text[i], ()) if text.startswith(t.first, i)]
            candidates += self.leading_ph
            tail = text[i:]
            for template in candidates:
                if not all(lit in tail for lit in template.lits):
                    continue
                partial = None
                for j in range(n, i, -1):
                    if best[j] is None:
                        continue
                    found = self._try_template(template, text, depth + 1, i, j)
                    if found is None:
                        continue
                    q, rest = best[j]  # type: ignore[misc]
                    piece = {
                        "id": template.tid,
                        "how": "template",
                        "text": text[i:j],
                        "params": found.params,
                        "anchored": found.anchored,
                    }
                    option = (
                        (
                            q[0] + len(found.unresolved),
                            q[1] - found.anchored,
                            q[2] + found.complexity,
                            q[3] + 1,
                        ),
                        [piece, *rest],
                    )
                    if not found.unresolved:
                        options.append(option)
                        break  # the longest full reach for this template
                    if partial is None:
                        partial = option
                else:
                    if partial is not None:
                        options.append(partial)
            if options:
                best[i] = min(options, key=lambda item: item[0])
        if best[0] is None:
            return None
        quality, pieces = best[0]
        if len(pieces) < 2:
            return None  # a single piece would have matched whole
        unresolved = [
            p["id"]
            for p in pieces
            if any(
                isinstance(v, dict) and v.get("kind") == "text"
                for v in (p.get("params") or {}).values()
            )
        ]
        status = "full" if quality[0] == 0 else "partial"
        return Match(
            status,
            "segments",
            None,
            segments=pieces,
            unresolved=unresolved,
            anchored=-quality[1],
            complexity=quality[2] + (1 if depth else 0),
        )
