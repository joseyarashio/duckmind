"""Engine packs: the text templates of one Duck Core engine, keyed like the catalogs.

蛋黃 runs on the FROZEN engine ``engines/v4_0`` (never edited: tests/test_engine_frozen.py and
phase-4 gate G1). Its words cannot carry message keys, so the API boundary matches each string
it sends back to the engine's own templates (``matcher.py``) and names the catalog key. The
templates are read from the engine's source (``extract.py``) the first time a pack is needed
(about a second per engine; ``warm()`` does it in the background at server start).

``v4_1`` is the current Duck Core (``duckcore/``, new ducks): the same boundary works for it
while its words are still plain text (92 % of its templates are the frozen engine's).

A template's key is the catalog key the i18n inventory gave it (i18n:inv-core): ``core.<module>.
<TABLE>.<row>`` for a table row, ``core.<module>.<function>.<digest>`` otherwise — the same in
both engines when the words and their place are the same.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import icu
from .catalog import Catalogs, default
from .extract import extract
from .locales import SOURCE
from .matcher import Matcher

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # src/duckmind
ENGINES: dict[str, list[str]] = {"v4_0": ["engines/v4_0"], "v4_1": ["duckcore"]}
MATCH_VISIBILITY = {"visible", "raw", "glue"}

# ------------------------------------------------------------------ what reaches the page
# (i18n:inv-core traced every frozen-engine string through api.py, narrate.describe, the
# journal, the notes and the query/graph tabs: this is its visibility rule.)
INTERNAL_FUNCS = {
    "provenance": "invariant",
    "rng": "invariant",
    "fixed": "invariant",
    "expression_sweep.envelope": "tooling",
    "expression_sweep.clips": "invariant",
    "expression_sweep.looks": "tooling",
}
GRAMMAR_TABLES = {"SYNONYMS", "OTHER_PEOPLE", "WORDS_OF", "PATTERNS"}


def short_module(module: str) -> str:
    """engines.v4_0.duckcore.narrate → narrate; duckcore.narrate → narrate."""
    parts = module.split(".")
    if parts[:3] == ["engines", "v4_0", "duckcore"]:
        return ".".join(parts[3:])
    if parts[:1] == ["duckcore"]:
        return ".".join(parts[1:])
    return ".".join(parts[2:])


def visibility(site: dict[str, Any]) -> tuple[str, str]:
    """Whether an engine string reaches the page: visible | raw | glue | input | logic | data |
    internal (the i18n core inventory's rule, for both engines)."""
    module = short_module(site["module"])
    table = site.get("table")
    last = str(site["table_path"][-1]) if site.get("table_path") else ""
    if site.get("wordless"):
        return "glue", "no words of its own: joins worded parts"
    if site["kind"] == "separator":
        return "visible", "list separator"
    slot = site["slot"]["type"]
    func = site["function"]
    for prefix, why in INTERNAL_FUNCS.items():
        if module == prefix:
            return "internal", why
    if slot in {"compare", "subscript_key", "condition"}:
        return "logic", "compared or used as a key, never shown"
    if slot == "dict_key":
        if module in {"query", "commands"}:
            return "input", "input grammar"
        return "logic", "a dict key"
    if module == "commands":
        if slot == "raise":
            return "visible", "error toast (bad command)"
        if func in {"parse", "_parse_clause", "_split"} or table:
            return "input", "command grammar"
    if module == "query" and (
        table in GRAMMAR_TABLES or slot in {"arg"} and "startswith" in str(site["slot"].get("func"))
    ):
        return "input", "memory-query grammar"
    if module in {"query", "commands"} and (slot == "loop_values" or func == "_terms"):
        return "input", "words looked for in what the family typed"
    if (
        module == "api"
        and func == "dispatch"
        and slot == "arg"
        and str(site["slot"].get("func")) == "params.get"
    ):
        return "input", "a default question the page never relies on"
    if table == "Config.name" or (
        module == "engine" and func == "DuckCore.__init__" and slot.startswith("stmt")
    ):
        return "data", "the default duck name (a name is data once hatched)"
    if slot == "raise":
        return "visible", "error toast"
    if table == "VERBS" and last.endswith(".body"):
        return "internal", "design note on each move (never sent)"
    if table == "BRANCHES" and last.endswith(".touch_reaction"):
        return "internal", "design note (never sent)"
    if table == "VERBS" and last.endswith(".validated"):
        return "raw", "copied into event payloads: shown only in the raw JSON"
    return "visible", "reaches the page"


def _flat(item: Any) -> str:
    if isinstance(item, list):
        return "|".join(_flat(x) for x in item)
    return re.sub(r"[^A-Za-z0-9_|<>-]", "_", str(item))


def key_for(site: dict[str, Any]) -> str:
    """The catalog key of an engine template (i18n:inv-core's rule, the same in both engines)."""
    module = site["module"]
    module = re.sub(r"^engines\.v4_0\.duckcore\.", "core.", module)
    module = re.sub(r"^engines\.v4_0\.", "core.", module)
    module = re.sub(r"^duckcore\.", "core.", module)
    if site.get("table") and site["kind"] in {"literal", "format"}:
        path = ".".join(_flat(x) for x in site["table_path"]) or "_"
        return f"{module}.{site['table']}.{path}"
    shape = re.sub(r"\{[A-Za-z0-9_]+\}", "{}", site["zh_hant"])
    digest = hashlib.sha1(f"{site['function']}|{site['kind']}|{shape}".encode()).hexdigest()[:8]
    return f"{module}.{site['function']}.{digest}"


def assign_ids(sites: list[dict[str, Any]], engine: str) -> None:
    """Template ids for the matcher (unique within one engine)."""
    seen: Counter[str] = Counter()
    for site in sites:
        module = short_module(site["module"])
        if site.get("table") and site["kind"] in {"literal", "format"}:
            path = ".".join(_flat(x) for x in site["table_path"]) or "_"
            base = f"{engine}:{module}.{site['table']}.{path}"
        else:
            base = f"{engine}:{module}.{site['function']}:{site['digest']}"
        seen[base] += 1
        site["tid"] = base if seen[base] == 1 else f"{base}#{seen[base]}"


@dataclass
class Pack:
    engine: str
    sites: list[dict[str, Any]]
    matcher: Matcher
    key_of: dict[str, str] = field(default_factory=dict)  # tid → catalog key
    parts_of: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rename: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # tid → {engine name: catalog name}
    sound_key: dict[str, str] = field(default_factory=dict)  # babble token → key
    missing: list[str] = field(default_factory=list)  # keys this engine emits that no catalog has

    def key(self, tid: str) -> str:
        """The catalog key of a template id (``tid[ch]`` = one character of an indexed literal)."""
        if tid in self.key_of:
            return self.key_of[tid]
        base, bracket, rest = tid.partition("[")
        if bracket and base in self.key_of:
            text = "".join(p.get("lit", "") for p in self.parts_of.get(base, []))
            ch = rest[:-1]
            if ch in text:
                return f"{self.key_of[base]}.{text.index(ch)}"
        return tid


def build(engine: str, root: Path | None = None, catalogs: Catalogs | None = None) -> Pack | None:
    """The pack of one engine, or None when its source is not in this tree (the neural-duck
    worktree has no frozen engine)."""
    root = root or PACKAGE_ROOT
    targets = ENGINES[engine]
    if not all((root / t).exists() for t in targets):
        return None
    sites = extract(root, targets)["sites"]
    assign_ids(sites, engine)
    keep: list[dict[str, Any]] = []
    for site in sites:
        vis, _why = visibility(site)
        site["visibility"] = vis
        if vis in MATCH_VISIBILITY:
            keep.append(site)
    sounds = tuple(
        s["parts"][0]["lit"] for s in sites if s.get("table") == "TAG_ZH" and s["kind"] == "literal"
    )
    matcher = Matcher(keep, sound_tokens=sounds)
    pack = Pack(engine, keep, matcher)
    cats = catalogs or default()
    for site in keep:
        key = key_for(site)
        pack.key_of[site["tid"]] = key
        pack.parts_of[site["tid"]] = site["parts"]
        if site.get("table") == "TAG_ZH" and site["kind"] == "literal":
            pack.sound_key[site["parts"][0]["lit"]] = key
        if site["kind"] == "separator":
            continue
        message = cats.get(key, SOURCE)
        if message is None:
            if site.get("visibility") == "visible" or site.get("wordless"):
                pack.missing.append(key)
            continue
        names = [p["ph"] for p in site["parts"] if "ph" in p]
        wanted = _arg_order(message)
        if (
            names
            and names != wanted
            and len(set(names)) == len(set(wanted))
            and set(names) != set(wanted)
        ):
            pack.rename[site["tid"]] = dict(zip(_unique(names), _unique(wanted), strict=False))
    return pack


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _arg_order(message: str) -> list[str]:
    """The arguments of a message in the order they appear."""
    out: list[str] = []

    def walk(nodes: tuple[icu.Node, ...]) -> None:
        for node in nodes:
            if isinstance(node, (icu.Arg, icu.Number, icu.Choice)) and node.name not in out:
                out.append(node.name)
            if isinstance(node, icu.Choice):
                for branch in node.options.values():
                    walk(branch)

    walk(icu.parse(message))
    return out


_PACKS: dict[str, Pack | None] = {}
_LOCK = threading.Lock()


def get(engine: str) -> Pack | None:
    """The pack of ``engine`` (built once, thread safe)."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r} (known: {sorted(ENGINES)})")
    with _LOCK:
        if engine not in _PACKS:
            _PACKS[engine] = build(engine)
        return _PACKS[engine]


def warm(engines: tuple[str, ...] = ("v4_0", "v4_1")) -> threading.Thread:
    """Build the packs in a background thread (call at server start)."""
    thread = threading.Thread(
        target=lambda: [get(e) for e in engines], name="i18n-packs", daemon=True
    )
    thread.start()
    return thread
