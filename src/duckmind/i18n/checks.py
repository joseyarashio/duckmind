"""The i18n checks: catalogs, context, the manifest, and words left hard-coded in the page.

The tests (tests/test_i18n_*.py) and ``python -m duckmind.i18n check`` both run these, so a
developer sees the same list the tests fail on. Every function returns problems as plain
sentences; an empty list means fine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from . import icu
from .catalog import (
    DOMAINS,
    STATIC,
    CatalogError,
    catalog_path,
    context,
    domain_of,
    manifest_data,
    parse_file,
    read_catalog,
    read_manifest,
    src_hash,
    valid_key,
)
from .locales import CODES, SOURCE

HERE = Path(__file__).resolve().parent
ALLOWLIST_FILE = HERE / "allowlist.json"
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


# ------------------------------------------------------------------ catalogs
def load_all(static: Path | None = None) -> dict[str, dict[str, str]]:
    """lang → key → message, for every language that has files (missing files skipped)."""
    out: dict[str, dict[str, str]] = {}
    for code in CODES:
        merged: dict[str, str] = {}
        found = False
        for domain in DOMAINS:
            messages = read_catalog(code, domain, static)
            if messages is not None:
                found = True
                merged.update(messages)
        if found:
            out[code] = merged
    return out


def check_files(static: Path | None = None) -> list[str]:
    """Every catalog file parses, is canonical, holds only its own domain's valid keys."""
    from .catalog import dump_file

    problems: list[str] = []
    for code in CODES:
        for domain in DOMAINS:
            path = catalog_path(code, domain, static)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            try:
                got_lang, got_domain, messages = parse_file(text, str(path))
            except CatalogError as error:
                problems.append(str(error))
                continue
            if (got_lang, got_domain) != (code, domain):
                problems.append(f"{path.name}: says {got_lang}/{got_domain}")
            for key in messages:
                if not valid_key(key):
                    problems.append(f"{path.name}: bad key {key!r}")
                elif domain_of(key) != domain:
                    problems.append(f"{path.name}: {key} belongs in the {domain_of(key)} file")
            if text != dump_file(code, domain, messages):
                problems.append(
                    f"{path.name}: not in canonical form (run `python -m duckmind.i18n format`)"
                )
    return problems


def arg_kinds(message: str) -> dict[str, str]:
    """Arguments of a message, number-like kinds folded together (a translation may show a
    number plainly, formatted, or with plural forms)."""
    kinds = icu.arguments(icu.parse(message))
    return {
        name: ("num" if kind in {"number", "plural", "selectordinal"} else kind)
        for name, kind in kinds.items()
    }


def check_language(
    code: str, messages: dict[str, str], source: dict[str, str]
) -> dict[str, list[str]]:
    """One language against the source: missing / extra keys, broken messages, values that
    differ from the source's, plural forms the language does not have."""
    out: dict[str, list[str]] = {
        "missing": [],
        "extra": [],
        "broken": [],
        "values": [],
        "plural": [],
    }
    for key in source:
        if key not in messages:
            out["missing"].append(key)
    for key, message in messages.items():
        if key not in source:
            out["extra"].append(key)
            continue
        problems = icu.check(message, code)
        parse_errors = [
            p for p in problems if "not a plural category" not in p and "has no" not in p
        ]
        if parse_errors and not _parses(message):
            out["broken"].append(f"{key}: {parse_errors[0]}")
            continue
        out["plural"].extend(f"{key}: {p}" for p in problems if "plural" in p or "has no" in p)
        try:
            want = arg_kinds(source[key])
            got = arg_kinds(message)
        except icu.MessageError as error:
            out["broken"].append(f"{key}: {error}")
            continue
        if set(want) != set(got):
            out["values"].append(f"{key}: values {sorted(got)} ≠ source {sorted(want)}")
        else:
            for name, kind in want.items():
                if kind == "select" and got[name] != "select":
                    out["values"].append(f"{key}: {{{name}}} must stay a select")
    return out


def _parses(message: str) -> bool:
    try:
        icu.parse(message)
        return True
    except icu.MessageError:
        return False


def check_context(source: dict[str, str], ctx: dict[str, Any] | None = None) -> list[str]:
    """Every key has translator context, and the context describes today's source words."""
    problems: list[str] = []
    ctx = context() if ctx is None else ctx
    for key, message in source.items():
        entry = ctx.get(key)
        if entry is None:
            problems.append(f"{key}: no context (add it with `python -m duckmind.i18n add`)")
        elif entry.get("src") != src_hash(message):
            problems.append(
                f"{key}: its 繁體中文 changed after it was translated/described — give the new words a new key "
                "(and translate them), or for a same-meaning fix run `python -m duckmind.i18n restamp "
                + key
                + "`"
            )
    for key in ctx:
        if key not in source:
            problems.append(f"{key}: context for a key the source catalog does not have")
    return problems


def check_manifest(static: Path | None = None) -> list[str]:
    want = manifest_data(static)
    have = read_manifest(static)
    if have is None:
        return ["i18n.manifest.js is missing (run `python -m duckmind.i18n manifest`)"]
    if have != want:
        return [
            "i18n.manifest.js does not match the catalog files (run `python -m duckmind.i18n manifest`)"
        ]
    return []


# ------------------------------------------------------------------ hard-coded words in the page
@dataclass
class Leftover:
    file: str
    line: int
    text: str
    kind: str  # string | template | regex | html-text | html-attr | css

    def __str__(self) -> str:
        return f"{self.file}:{self.line} [{self.kind}] {self.text[:80]}"


def allowlist() -> dict[str, Any]:
    if ALLOWLIST_FILE.is_file():
        data = json.loads(ALLOWLIST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {}


def _allowed(text: str, allow: dict[str, Any]) -> bool:
    words = set(allow.get("names", [])) | set(allow.get("strings", []))
    stripped = text.strip()
    if stripped in words:
        return True
    # a string that is only allowed names / symbols once they are taken out
    rest = stripped
    for word in sorted(words, key=len, reverse=True):
        rest = rest.replace(word, "")
    return not CJK.search(rest)


_REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_WORDS = {
    "return",
    "typeof",
    "case",
    "do",
    "else",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "throw",
    "yield",
    "await",
}


def js_strings(src: str) -> list[tuple[int, str, str]]:
    """(line, text, kind) of every string, template-literal text and regex literal in JS
    source, comments skipped. A small lexer: good enough for this project's page scripts."""
    out: list[tuple[int, str, str]] = []
    n = len(src)

    def scan(i: int, line: int, until_brace: bool) -> tuple[int, int]:
        depth = 0
        last = ""  # last significant token (for / as regex or division)
        while i < n:
            c = src[i]
            if c == "\n":
                line += 1
                i += 1
                continue
            if c in " \t\r":
                i += 1
                continue
            if src.startswith("//", i):
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src.startswith("/*", i):
                j = src.find("*/", i + 2)
                j = n if j < 0 else j + 2
                line += src.count("\n", i, j)
                i = j
                continue
            if c in "\"'":
                start_line = line
                j = i + 1
                buf: list[str] = []
                while j < n and src[j] != c:
                    if src[j] == "\\" and j + 1 < n:
                        buf.append(src[j : j + 2])
                        if src[j + 1] == "\n":
                            line += 1
                        j += 2
                        continue
                    if src[j] == "\n":
                        break
                    buf.append(src[j])
                    j += 1
                out.append((start_line, "".join(buf), "string"))
                i = j + 1
                last = "str"
                continue
            if c == "`":
                i, line = template(i + 1, line)
                last = "str"
                continue
            if c == "/" and (last == "" or last in _REGEX_BEFORE or last in _REGEX_WORDS):
                start_line = line
                j = i + 1
                in_class = False
                while j < n:
                    ch = src[j]
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        break
                    elif ch == "\n":
                        break
                    j += 1
                out.append((start_line, src[i + 1 : j], "regex"))
                i = j + 1
                while i < n and src[i].isalpha():
                    i += 1
                last = "regex"
                continue
            if until_brace:
                if c == "{":
                    depth += 1
                elif c == "}":
                    if depth == 0:
                        return i + 1, line
                    depth -= 1
            if c.isalnum() or c in "_$":
                j = i
                while j < n and (src[j].isalnum() or src[j] in "_$"):
                    j += 1
                last = src[i:j] if not src[i].isdigit() else "num"
                i = j
                continue
            last = c
            i += 1
        return i, line

    def template(i: int, line: int) -> tuple[int, int]:
        buf: list[str] = []
        start_line = line
        while i < n:
            c = src[i]
            if c == "\\" and i + 1 < n:
                buf.append(src[i : i + 2])
                i += 2
                continue
            if c == "`":
                out.append((start_line, "".join(buf), "template"))
                return i + 1, line
            if src.startswith("${", i):
                out.append((start_line, "".join(buf), "template"))
                buf = []
                i, line = scan(i + 2, line, True)
                start_line = line
                continue
            if c == "\n":
                line += 1
            buf.append(c)
            i += 1
        out.append((start_line, "".join(buf), "template"))
        return i, line

    scan(0, 1, False)
    return out


class _HtmlScan(HTMLParser):
    VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, source: dict[str, str]) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.stack: list[dict[str, Any]] = []
        self.found: list[Leftover] = []
        self.skip = 0
        self.file = "index.html"

    def _covered_text(self) -> dict[str, Any] | None:
        for item in reversed(self.stack):
            if item["key"]:
                return item
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        line = self.getpos()[0]
        if tag in {"script", "style"}:
            self.skip += 1
        key = values.get("data-i18n") or values.get("data-i18n-html")
        covered_attrs: dict[str, str] = {}
        for pair in (values.get("data-i18n-attr") or "").split(";"):
            if ":" in pair:
                attr_name, _, attr_key = pair.partition(":")
                covered_attrs[attr_name.strip()] = attr_key.strip()
        for name, value in values.items():
            if not CJK.search(value) or name.startswith("data-i18n"):
                continue
            akey = covered_attrs.get(name)
            if akey is None:
                self.found.append(Leftover(self.file, line, f"{name}={value}", "html-attr"))
            elif _zh(self.source, akey) != value:
                self.found.append(
                    Leftover(self.file, line, f"{name}={value} ≠ {akey}", "html-attr")
                )
        if tag not in self.VOID:
            self.stack.append({"tag": tag, "key": key, "text": [], "line": line})

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        while self.stack:
            item = self.stack.pop()
            if item["key"]:
                words = re.sub(r"\s+", " ", "".join(item["text"])).strip()
                want = re.sub(r"<[^>]+>", "", _zh(self.source, item["key"]) or "")
                want = re.sub(r"\s+", " ", want).strip()
                if words != want:
                    self.found.append(
                        Leftover(self.file, item["line"], f"{words} ≠ {item['key']}", "html-text")
                    )
            if item["tag"] == tag:
                break

    def handle_data(self, data: str) -> None:
        if self.skip or not CJK.search(data):
            if self.stack:
                self.stack[-1]["text"].append(data)
            return
        owner = self._covered_text()
        for item in self.stack:
            if item["key"]:
                item["text"].append(data)
        if owner is None:
            self.found.append(Leftover(self.file, self.getpos()[0], data.strip(), "html-text"))


def _zh(source: dict[str, str], key: str) -> str | None:
    message = source.get(key)
    if message is None:
        return None
    try:
        return icu.plain_text(icu.parse(message))
    except icu.MessageError:
        return message


def is_catalog_file(name: str) -> bool:
    return bool(re.fullmatch(r"i18n\.[A-Za-z-]+\.[a-z]+\.js|i18n\.manifest\.js", name))


def scan_static(static: Path | None = None, source: dict[str, str] | None = None) -> list[Leftover]:
    """Words in the page files that do not come from a catalog (comments ignored)."""
    static = static or STATIC
    if source is None:
        source = {}
        for domain in DOMAINS:
            source.update(read_catalog(SOURCE, domain, static) or {})
    allow = allowlist()
    found: list[Leftover] = []
    for path in sorted(static.iterdir()):
        name = path.name
        if not path.is_file() or is_catalog_file(name):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if name.endswith(".js"):
            for line, words, kind in js_strings(text):
                if CJK.search(words) and not _allowed(words, allow):
                    found.append(Leftover(name, line, words, kind))
        elif name.endswith(".html"):
            scanner = _HtmlScan(source)
            scanner.file = name
            scanner.feed(
                re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
            )
            scanner.close()
            found.extend(
                f
                for f in scanner.found
                if not _allowed(f.text.split("=", 1)[-1].split(" ≠ ")[0], allow)
            )
        elif name.endswith(".css"):
            clean = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
            for match in re.finditer(r"([\"'])((?:\\.|(?!\1).)*)\1", clean):
                if CJK.search(match.group(2)) and not _allowed(match.group(2), allow):
                    found.append(
                        Leftover(
                            name, clean.count("\n", 0, match.start()) + 1, match.group(2), "css"
                        )
                    )
    return found


def page_uses_i18n(static: Path | None = None) -> bool:
    """The page has the i18n system: index.html loads i18n.js (docs/i18n.md, CLAUDE.md)."""
    index = (static or STATIC) / "index.html"
    return index.is_file() and bool(
        re.search(r"<script[^>]+src=\"/?i18n\.js\"", index.read_text(encoding="utf-8"))
    )


KEY_CALL_JS = re.compile(
    r"""\b(?:I18N\.)?(?:t|tHtml|text|html)\(\s*(?:[A-Za-z_$][\w$.]*\s*,\s*)?["']([a-z][\w.|<>-]+)["']"""
)
KEY_ATTR_JS = re.compile(
    r"""\bI18N\.attr\(\s*[^,()]+,\s*["'][^"']+["']\s*,\s*["']([a-z][\w.|<>-]+)["']"""
)
KEY_ATTR_HTML = re.compile(r"""data-i18n(?:-html)?=["']([^"']+)["']""")
KEY_ATTR_PAIRS = re.compile(r"""data-i18n-attr=["']([^"']+)["']""")
KEY_CALL_PY = re.compile(r"""\bmsg\(\s*["']([a-z][\w.|<>-]+)["']""")


def used_keys(root: Path) -> dict[str, list[str]]:
    """Keys the code names literally (JS t()/text()/html()/attr(), data-i18n attributes,
    Python msg()): key → places."""
    out: dict[str, list[str]] = {}
    static = root / "src/duckmind/simweb/static"
    for path in sorted(static.glob("*")):
        if not path.is_file() or is_catalog_file(path.name) or path.suffix not in {".js", ".html"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in KEY_CALL_JS.finditer(text):
            if "." in match.group(1):
                out.setdefault(match.group(1), []).append(
                    f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}"
                )
        for match in KEY_ATTR_JS.finditer(text):
            out.setdefault(match.group(1), []).append(
                f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}"
            )
        for match in KEY_ATTR_HTML.finditer(text):
            out.setdefault(match.group(1), []).append(
                f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}"
            )
        for match in KEY_ATTR_PAIRS.finditer(text):
            for pair in match.group(1).split(";"):
                if ":" in pair:
                    out.setdefault(pair.split(":", 1)[1].strip(), []).append(
                        f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}"
                    )
    for path in sorted((root / "src/duckmind").rglob("*.py")):
        if "engines" in path.parts or "__pycache__" in path.parts or path.parent.name == "i18n":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "msg(" not in text:
            continue
        for match in KEY_CALL_PY.finditer(text):
            out.setdefault(match.group(1), []).append(
                f"{path.relative_to(root)}:{text.count(chr(10), 0, match.start()) + 1}"
            )
    return out
