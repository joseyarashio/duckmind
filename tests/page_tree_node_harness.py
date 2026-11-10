"""The simulator page's own scripts, run in Node over the page's own markup.

``run`` reads ``static/index.html`` with ``html.parser`` into a plain tree (tag, attributes, parent,
text) and hands it to ``page_tree_node_harness.js``, which builds a small fake page from it, loads
the named scripts in the order the page itself loads them, plays one scenario (clicks, answers,
time) and prints what happened as JSON. So a test says what the page does with its real markup:
which panel a tab shows, what a button sends, what is drawn — not which words are in a file.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

NODE = shutil.which("node")
STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
HARNESS = Path(__file__).with_name("page_tree_node_harness.js")

VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: list[dict[str, Any]] = []
        self.scripts: list[dict[str, str]] = []
        self._open: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self._open[-1] if self._open else None
        self.nodes.append({"tag": tag, "attrs": {k: v or "" for k, v in attrs}, "parent": parent})
        if tag == "script":
            given = dict(attrs)
            if given.get("src"):
                self.scripts.append({"src": given["src"] or "", "type": given.get("type") or ""})
        if tag not in VOID:
            self._open.append(len(self.nodes) - 1)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self._open[-1] if self._open else None
        self.nodes.append({"tag": tag, "attrs": {k: v or "" for k, v in attrs}, "parent": parent})

    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self._open) - 1, -1, -1):
            if self.nodes[self._open[depth]]["tag"] == tag:
                del self._open[depth:]
                return

    def handle_data(self, data: str) -> None:
        if not self._open or not data.strip():
            return
        parent = self._open[-1]
        if self.nodes[parent]["tag"] in ("script", "style"):
            return
        self.nodes.append({"text": data, "parent": parent})


@functools.cache
def page() -> _Tree:
    """The page's markup as a tree, and the scripts it loads, in its order."""
    tree = _Tree()
    tree.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return tree


def _within(index: int | None, inside: str | None) -> bool:
    nodes = page().nodes
    while inside is not None and index is not None:
        if nodes[index].get("attrs", {}).get("id") == inside:
            return True
        index = nodes[index]["parent"]
    return inside is None


def elements_with(name: str, inside: str | None = None) -> list[dict[str, str]]:
    """The attributes of every element of the page that has the attribute ``name``, in the
    page's order (``inside``: only in the element with that id)."""
    return [
        node["attrs"]
        for node in page().nodes
        if name in node.get("attrs", {}) and _within(node["parent"], inside)
    ]


def element_texts(tag: str, inside: str | None = None) -> list[str]:
    """The text of every ``tag`` element of the page, in its order (``inside``: only in the
    element with that id)."""
    nodes = page().nodes
    out = []
    for index, node in enumerate(nodes):
        if node.get("tag") == tag and _within(node["parent"], inside):
            out.append(
                "".join(n["text"] for n in nodes if n.get("parent") == index and "text" in n)
            )
    return out


def run(scenario: str, scripts: set[str], data: Any = None) -> Any:
    """Play ``scenario`` (a name in the harness) with ``scripts`` (file names) loaded in the
    order the page runs them (its classic scripts as it names them, then its modules, as a
    browser does); ``data`` is the scenario's input. Returns what it printed."""
    named = [s for s in page().scripts if s["src"].lstrip("/") in scripts]
    assert {s["src"].lstrip("/") for s in named} == set(scripts), "a script the page does not load"
    named.sort(key=lambda s: s["type"] == "module")
    order = [{"name": s["src"].lstrip("/"), "module": s["type"] == "module"} for s in named]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "input.json"
        path.write_text(
            json.dumps(
                {"tree": page().nodes, "scripts": order, "scenario": scenario, "data": data},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        done = subprocess.run(
            [str(NODE), str(HARNESS), str(STATIC), str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)
