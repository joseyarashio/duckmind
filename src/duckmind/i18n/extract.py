"""Extract every text template from Python source (ported from i18n:inv-core's prototype).

The API boundary (``boundary.py``) matches the frozen engine's words back to the templates that
wrote them; this module finds those templates in the engine's own source. It walks the AST of
the given files and records each string that can carry words:
  * plain string constants (``"開機"``),
  * f-strings (``f"孵化：{p.get('name')}（種子 {p.get('seed')}）"``), with their placeholders,
  * ``str.format`` templates (constants holding ``{name}`` fields),
  * ``+`` chains that glue several of these together (kept as one composite template too),
  * list separators (``"、".join(...)``: locale-specific punctuation).
For each: file, line, enclosing function, the slot it sits in (dict key, table key, keyword,
call argument, return, assignment, raise, print, comparison…), and a stable key.

Read-only: it only parses source text (``python -m duckmind.i18n extract`` prints a summary).
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿＀-￯]")
FORMAT_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:![rsa])?(?::[^{}]*)?\}")
SEPARATOR_MAX = 4  # a "、".join separator is short


def has_cjk(text: str) -> bool:
    return bool(CJK.search(text))


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    return parent


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    out.add(id(value))
    # bare string statements elsewhere (comments written as strings) count as docs too
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                out.add(id(node.value))
    return out


def _dataclass_fields(tree: ast.Module) -> dict[str, list[str]]:
    """Field order of every class in the module (dataclass positional arguments)."""
    fields: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names = [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
            if names:
                fields[node.name] = names
    return fields


def _func_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{func.attr}"
        if isinstance(base, ast.Constant) and isinstance(base.value, str):
            return f"<str>.{func.attr}"
        return f"….{func.attr}"
    return "…"


def _const_key(node: ast.AST | None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return [_const_key(item) for item in node.elts]
    if isinstance(node, ast.Name):
        return f"<{node.id}>"
    if isinstance(node, ast.Attribute):
        return f"<{ast.unparse(node)}>"
    return None


def _ph_name(expr: ast.AST, used: set[str]) -> str:
    """A readable placeholder name for an f-string field."""
    name = ""
    node = expr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant):
            name = str(node.args[0].value)
        elif node.func.attr in {"strftime"}:
            name = "time"
        elif node.func.attr == "join":
            name = "list"
    if not name and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"round", "int", "abs", "max", "min", "float", "num"} and node.args:
            return _ph_name(node.args[0], used)
        if node.func.id in {"str"} and node.args:
            return _ph_name(node.args[0], used)
        name = node.func.id
    if not name and isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            name = key.value
        else:
            name = _ph_name(node.value, used)
    if not name and isinstance(node, ast.Attribute):
        base = node.value
        name = (
            f"{base.id}_{node.attr}"
            if isinstance(base, ast.Name) and base.id not in {"s", "p", "self", "core", "payload"}
            else node.attr
        )
    if not name and isinstance(node, ast.Name):
        name = node.id
    if not name and isinstance(node, ast.BinOp):
        name = _ph_name(node.left, used)
    if not name and isinstance(node, ast.IfExp):
        name = "choice"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name or "p").strip("_") or "p"
    stem, n = name, 2
    while name in used:
        name = f"{stem}{n}"
        n += 1
    used.add(name)
    return name


def _spec_text(spec: ast.AST | None) -> str:
    if spec is None:
        return ""
    if isinstance(spec, ast.JoinedStr):
        return "".join(str(v.value) for v in spec.values if isinstance(v, ast.Constant))
    return ""


def fstring_parts(
    node: ast.JoinedStr, src: str, used: set[str] | None = None
) -> list[dict[str, Any]]:
    used = used if used is not None else set()
    parts: list[dict[str, Any]] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append({"lit": str(value.value)})
        elif isinstance(value, ast.FormattedValue):
            expr = value.value
            if isinstance(expr, ast.JoinedStr):  # f"{f'...'}" — inline it
                parts.extend(fstring_parts(expr, src, used))
                continue
            parts.append(
                {
                    "ph": _ph_name(expr, used),
                    "expr": ast.get_source_segment(src, expr) or ast.unparse(expr),
                    "spec": _spec_text(value.format_spec),
                    "conv": chr(value.conversion)
                    if value.conversion and value.conversion > 0
                    else "",
                    "alts": _alternatives(expr, src),
                }
            )
    return _merge(parts)


def _alternatives(expr: ast.AST, src: str = "") -> list[Any] | None:
    """The alternatives of a conditional placeholder: literals ({'機智' if … else '呆萌'}) and
    f-string shapes ({f'；現在 {clock}' if … else ''}, kept as {"parts": [...]}), so two optional
    fields side by side ({sound}{coda}) cannot swallow each other."""
    if isinstance(expr, ast.IfExp):
        out: list[Any] = []
        for branch in (expr.body, expr.orelse):
            if isinstance(branch, ast.Constant) and isinstance(branch.value, str):
                out.append(branch.value)
            elif isinstance(branch, ast.JoinedStr) and src:
                parts = fstring_parts(branch, src, set())
                if not any("lit" in part and part["lit"] for part in parts):
                    return None  # a shape with no words of its own tells nothing
                out.append({"parts": parts})
            elif isinstance(branch, ast.IfExp):
                more = _alternatives(branch, src)
                if more is None:
                    return None
                out.extend(more)
            else:
                return None
        return out
    return None


def _merge(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in parts:
        if "lit" in part and out and "lit" in out[-1]:
            out[-1] = {"lit": out[-1]["lit"] + part["lit"]}
        elif "lit" in part and part["lit"] == "":
            continue
        else:
            out.append(part)
    return out


def format_parts(text: str) -> list[dict[str, Any]]:
    """A str.format template's parts ({name} fields)."""
    parts: list[dict[str, Any]] = []
    pos = 0
    for match in FORMAT_FIELD.finditer(text):
        if match.start() > pos:
            parts.append({"lit": text[pos : match.start()]})
        parts.append(
            {"ph": match.group(1), "expr": match.group(0), "spec": "", "conv": "", "alts": None}
        )
        pos = match.end()
    if pos < len(text):
        parts.append({"lit": text[pos:]})
    return _merge(parts)


def render(parts: list[dict[str, Any]]) -> str:
    out = []
    for part in parts:
        if "lit" in part:
            out.append(part["lit"].replace("{", "{{").replace("}", "}}"))
        else:
            out.append("{" + part["ph"] + "}")
    return "".join(out)


def _concat_operands(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concat_operands(node.left) + _concat_operands(node.right)
    return [node]


def _is_texty(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    return isinstance(node, ast.JoinedStr)


class Extractor:
    def __init__(self, path: Path, rel: str, module: str) -> None:
        self.path = path
        self.rel = rel
        self.module = module
        self.src = path.read_text()
        self.tree = ast.parse(self.src)
        self.parent = _parents(self.tree)
        self.docs = _docstring_ids(self.tree)
        self.fields = _dataclass_fields(self.tree)
        self.sites: list[dict[str, Any]] = []
        self.docstrings = 0
        self._alts_cache: dict[int, dict[str, list[str] | None]] = {}

    # ------------------------------------------------------------------ context
    def enclosing(self, node: ast.AST) -> str:
        names = []
        cur: ast.AST | None = node
        while cur is not None:
            cur = self.parent.get(id(cur))
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(cur.name)
        return ".".join(reversed(names)) or "<module>"

    def table_of(self, node: ast.AST) -> tuple[str | None, list[Any]]:
        """For a string inside a module-level table (dict/tuple/list/call literal): the table's
        name and the key path to it."""
        path: list[Any] = []
        child = node
        cur = self.parent.get(id(node))
        while cur is not None:
            if isinstance(cur, ast.Dict):
                if child in cur.values:
                    path.append(_const_key(cur.keys[cur.values.index(child)]))
                elif child in cur.keys:
                    path.append("<key>")
            elif isinstance(cur, (ast.Tuple, ast.List, ast.Set)):
                first = _const_key(cur.elts[0]) if cur.elts else None
                if (
                    isinstance(child, ast.Call)
                    and child.args
                    and isinstance(child.args[0], ast.Constant)
                    and isinstance(child.args[0].value, str)
                ):
                    pass  # a row that names itself (Verb("look_around", …))
                elif (
                    isinstance(child, ast.Tuple)
                    and len(child.elts) == 2
                    and isinstance(_const_key(child.elts[0]), str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(_const_key(child.elts[0])))
                ):
                    pass  # an (id, text) pair inside a list: the pair names itself
                elif (
                    isinstance(cur, ast.Tuple)
                    and len(cur.elts) == 2
                    and child is cur.elts[1]
                    and isinstance(first, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first)
                ):
                    path.append(first)  # an (id, text) pair: named by its id
                else:
                    path.append(cur.elts.index(child) if child in cur.elts else "?")
            elif isinstance(cur, ast.Call):
                fname = _func_name(cur)
                label = None
                if child in cur.args:
                    index = cur.args.index(child)
                    names = self.fields.get(fname)
                    label = names[index] if names and index < len(names) else f"arg{index}"
                else:
                    for kw in cur.keywords:
                        if kw is child or kw.value is child:
                            label = kw.arg
                # name a dataclass row by its first (id) argument
                row = _const_key(cur.args[0]) if cur.args else None
                path.append(f"{row}.{label}" if isinstance(row, str) else f"{fname}.{label}")
            elif isinstance(cur, (ast.Assign, ast.AnnAssign)):
                targets = cur.targets if isinstance(cur, ast.Assign) else [cur.target]
                owner = self.parent.get(id(cur))
                if isinstance(owner, (ast.Module, ast.ClassDef)) and targets:
                    target = targets[0]
                    if isinstance(target, ast.Name):
                        name = (
                            f"{owner.name}.{target.id}"
                            if isinstance(owner, ast.ClassDef)
                            else target.id
                        )
                        return name, list(reversed(path))
                return None, []
            elif isinstance(
                cur,
                (
                    ast.JoinedStr,
                    ast.FormattedValue,
                    ast.BinOp,
                    ast.IfExp,
                    ast.BoolOp,
                    ast.Compare,
                    ast.Subscript,
                    ast.Attribute,
                    ast.Starred,
                    ast.keyword,
                    ast.DictComp,
                    ast.ListComp,
                    ast.GeneratorExp,
                    ast.comprehension,
                    ast.Lambda,
                ),
            ):
                if isinstance(cur, (ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Lambda)):
                    return None, []
                if isinstance(cur, ast.keyword):
                    pass
                else:
                    pass
            else:
                return None, []
            child = cur
            cur = self.parent.get(id(cur))
        return None, []

    def slot(self, node: ast.AST) -> dict[str, Any]:
        """Where the string goes: the nearest meaningful syntactic role."""
        child = node
        cur = self.parent.get(id(node))
        in_fexpr = False
        via: list[str] = []
        while cur is not None:
            if isinstance(cur, ast.FormattedValue):
                in_fexpr = True
            if isinstance(cur, ast.Dict):
                if child in cur.values:
                    key = _const_key(cur.keys[cur.values.index(child)])
                    return {"type": "dict_value", "key": key, "via": via, "in_fexpr": in_fexpr}
                return {"type": "dict_key", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, ast.keyword):
                call = self.parent.get(id(cur))
                fname = _func_name(call) if isinstance(call, ast.Call) else "?"
                return {
                    "type": "kwarg",
                    "func": fname,
                    "name": cur.arg,
                    "via": via,
                    "in_fexpr": in_fexpr,
                }
            if isinstance(cur, ast.Call):
                fname = _func_name(cur)
                if child is cur.func:
                    # "、".join(...) — the string is the receiver
                    return {"type": "receiver", "func": fname, "via": via, "in_fexpr": in_fexpr}
                if child in cur.args:
                    index = cur.args.index(child)
                    if fname in {"print"}:
                        return {"type": "print", "via": via, "in_fexpr": in_fexpr}
                    exc = self._raise_of(cur)
                    if exc:
                        return {"type": "raise", "exc": exc, "via": via, "in_fexpr": in_fexpr}
                    return {
                        "type": "arg",
                        "func": fname,
                        "index": index,
                        "via": via,
                        "in_fexpr": in_fexpr,
                    }
            if isinstance(cur, ast.Return):
                return {"type": "return", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = cur.targets if isinstance(cur, ast.Assign) else [cur.target]
                target = targets[0] if targets else None
                name = ast.unparse(target) if target is not None else "?"
                aug = isinstance(cur, ast.AugAssign)
                return {
                    "type": "augassign" if aug else "assign",
                    "target": name,
                    "via": via,
                    "in_fexpr": in_fexpr,
                }
            if isinstance(cur, ast.Compare):
                return {"type": "compare", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, ast.Subscript) and child is cur.slice:
                return {"type": "subscript_key", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, ast.Raise):
                return {"type": "raise", "exc": "?", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, (ast.If, ast.While, ast.Assert)):
                return {"type": "condition", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, (ast.For, ast.AsyncFor)) and child is cur.iter:
                # for kind, label in (("clever", "本週機智精選"), …): the loop's values
                return {"type": "loop_values", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, ast.Expr):
                return {"type": "expr", "via": via, "in_fexpr": in_fexpr}
            if isinstance(cur, ast.stmt):
                return {"type": f"stmt:{type(cur).__name__}", "via": via, "in_fexpr": in_fexpr}
            if isinstance(
                cur,
                (
                    ast.IfExp,
                    ast.BoolOp,
                    ast.BinOp,
                    ast.JoinedStr,
                    ast.Tuple,
                    ast.List,
                    ast.Set,
                    ast.FormattedValue,
                    ast.Starred,
                    ast.ListComp,
                    ast.GeneratorExp,
                    ast.DictComp,
                    ast.comprehension,
                    ast.Lambda,
                    ast.Await,
                    ast.UnaryOp,
                    ast.Subscript,
                    ast.Attribute,
                    ast.NamedExpr,
                    ast.Yield,
                ),
            ):
                via.append(type(cur).__name__)
            child = cur
            cur = self.parent.get(id(cur))
        return {"type": "?", "via": via, "in_fexpr": in_fexpr}

    def stmt_line(self, node: ast.AST) -> int:
        """The first line of the statement the string is in (a folded constant tuple runs as
        one instruction on that line)."""
        cur: ast.AST | None = node
        while cur is not None and not isinstance(cur, ast.stmt):
            cur = self.parent.get(id(cur))
        return getattr(cur, "lineno", getattr(node, "lineno", 0))

    def name_alternatives(self, node: ast.AST, name: str) -> list[str] | None:
        """The literal values a local name can hold (every assignment to it in the enclosing
        function is a string constant or a conditional of string constants), or None."""
        func: ast.AST | None = node
        while func is not None and not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = self.parent.get(id(func))
        if func is None:
            return None
        cache = self._alts_cache.setdefault(id(func), {})
        if name in cache:
            return cache[name]
        values: list[str] = []
        seen_any = False
        ok = True
        for sub in ast.walk(func):
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(sub, ast.Assign):
                targets, value = sub.targets, sub.value
            elif isinstance(sub, ast.AnnAssign):
                targets, value = [sub.target], sub.value
            elif (
                isinstance(sub, ast.AugAssign)
                and isinstance(sub.target, ast.Name)
                and sub.target.id == name
            ):
                ok = False
                break
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    seen_any = True
                    alts = _alternatives(value, self.src) if value is not None else None
                    if value is not None and isinstance(value, ast.JoinedStr):
                        parts = fstring_parts(value, self.src, set())
                        alts = (
                            [{"parts": parts}]
                            if any("lit" in q and q["lit"] for q in parts)
                            else None
                        )
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        alts = [value.value]
                    if alts is None:
                        ok = False
                    else:
                        values.extend(alts)
                elif isinstance(target, (ast.Tuple, ast.List)) and any(
                    isinstance(e, ast.Name) and e.id == name for e in target.elts
                ):
                    ok = False
            # a for-loop or with-target binding the name: not a constant
            if isinstance(sub, (ast.For, ast.comprehension)) and any(
                isinstance(e, ast.Name) and e.id == name for e in ast.walk(sub.target)
            ):
                ok = False
        result = None
        if seen_any and ok and values:
            unique: list[Any] = []
            for item in values:
                if item not in unique:
                    unique.append(item)
            result = sorted(
                unique, key=lambda v: len(v) if isinstance(v, str) else 1000, reverse=True
            )
        cache[name] = result
        return result

    def with_name_alts(self, node: ast.AST, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for part in parts:
            if "ph" in part and not part.get("alts"):
                expr = str(part.get("expr") or "")
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
                    alts = self.name_alternatives(node, expr)
                    if alts:
                        part["alts"] = alts
                        part["alts_from"] = "local name"
        return parts

    def calls_around(self, node: ast.AST) -> list[str]:
        """The calls this string is an argument of (nearest first), up to its statement:
        ``self._emit`` means it is written into an event payload (the hash chain)."""
        out: list[str] = []
        cur = self.parent.get(id(node))
        while cur is not None and not isinstance(cur, ast.stmt):
            if isinstance(cur, ast.Call):
                out.append(_func_name(cur))
            cur = self.parent.get(id(cur))
        # a value assigned to a name that a later _emit uses is not followed (see core.json)
        return out

    def _raise_of(self, call: ast.Call) -> str | None:
        up = self.parent.get(id(call))
        if isinstance(up, ast.Raise):
            return _func_name(call)
        return None

    # ------------------------------------------------------------------ walking
    def run(self) -> list[dict[str, Any]]:
        fstring_values: set[int] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    fstring_values.add(id(value))
                    if isinstance(value, ast.FormattedValue) and value.format_spec is not None:
                        fstring_values.add(id(value.format_spec))
                        for sub in ast.walk(value.format_spec):
                            fstring_values.add(id(sub))
        # composite "+" chains
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                up = self.parent.get(id(node))
                if isinstance(up, ast.BinOp) and isinstance(up.op, ast.Add):
                    continue  # only the whole chain
                operands = _concat_operands(node)
                texty = [op for op in operands if _is_texty(op)]
                if len(operands) < 2 or not texty:
                    continue
                if not any(has_cjk(self._flat_literal(op)) for op in operands):
                    continue
                self._add_concat(node, operands)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.JoinedStr):
                if id(node) in fstring_values:
                    continue
                parts = fstring_parts(node, self.src)
                lit = "".join(p.get("lit", "") for p in parts)
                slot = self.slot(node)
                if not has_cjk(lit) and slot["type"] != "raise":
                    # A template without words (f"{at} {summary}", f"{cue} {count}"): it glues
                    # worded parts together, so the matcher may use it to split a string whose
                    # every piece it then resolves. Needs a literal between its fields.
                    if any("ph" in q for q in parts) and lit:
                        self._add(node, "fstring", parts, slot, wordless=True)
                    continue
                self._add(node, "fstring", parts, slot)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in fstring_values:
                    continue
                if id(node) in self.docs:
                    if has_cjk(node.value):
                        self.docstrings += 1
                    continue
                slot = self.slot(node)
                text = node.value
                if slot["type"] == "receiver" and slot["func"].endswith(".join"):
                    if len(text) <= SEPARATOR_MAX:
                        self._add(node, "separator", [{"lit": text}], slot)
                    continue
                if not has_cjk(text) and slot["type"] != "raise":
                    continue
                if FORMAT_FIELD.search(text) and has_cjk(text):
                    self._add(node, "format", format_parts(text), slot)
                else:
                    up = self.parent.get(id(node))
                    indexed = isinstance(up, ast.Subscript) and up.value is node
                    self._add(node, "literal", [{"lit": text}], slot, indexed_chars=indexed)
        return self.sites

    def _flat_literal(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(str(v.value) for v in node.values if isinstance(v, ast.Constant))
        return ""

    def _add_concat(self, node: ast.BinOp, operands: list[ast.AST]) -> None:
        used: set[str] = set()
        parts: list[dict[str, Any]] = []
        for op in operands:
            if isinstance(op, ast.Constant) and isinstance(op.value, str):
                parts.append({"lit": op.value})
            elif isinstance(op, ast.JoinedStr):
                parts.extend(fstring_parts(op, self.src, used))
            else:
                parts.append(
                    {
                        "ph": _ph_name(op, used),
                        "expr": ast.get_source_segment(self.src, op) or ast.unparse(op),
                        "spec": "",
                        "conv": "",
                        "alts": _alternatives(op, self.src),
                    }
                )
        self._add(node, "concat", _merge(parts), self.slot(node))

    def _add(
        self,
        node: ast.AST,
        kind: str,
        parts: list[dict[str, Any]],
        slot: dict[str, Any],
        indexed_chars: bool = False,
        wordless: bool = False,
    ) -> None:
        parts = self.with_name_alts(node, parts)
        table, path = self.table_of(node)
        func = self.enclosing(node)
        text = render(parts)
        digest = hashlib.sha1(f"{self.module}|{func}|{kind}|{text}".encode()).hexdigest()[:8]
        lineno = int(getattr(node, "lineno", 0))
        col = int(getattr(node, "col_offset", 0))
        site = {
            "site": f"{self.rel}:{lineno}:{col}",
            "file": self.rel,
            "line": lineno,
            "col": col,
            "end_line": getattr(node, "end_lineno", lineno),
            "module": self.module,
            "function": func,
            "kind": kind,
            "zh_hant": text,
            "parts": parts,
            "placeholders": [p["ph"] for p in parts if "ph" in p],
            "slot": slot,
            "stmt_line": self.stmt_line(node),
            "calls": self.calls_around(node),
            "table": table,
            "table_path": path,
            "has_cjk": has_cjk(text),
            "digest": digest,
        }
        if indexed_chars:
            site["indexed_chars"] = True  # "一二三四五六日"[weekday]: each character is shown alone
        if wordless:
            site["wordless"] = True  # glue only (no words of its own): used to split strings
        self.sites.append(site)


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def extract(root: Path, targets: list[str], prefix: str = "src/duckmind") -> dict[str, Any]:
    files: list[Path] = []
    for target in targets:
        path = root / target
        files.extend([path] if path.is_file() else sorted(path.rglob("*.py")))
    sites: list[dict[str, Any]] = []
    docstrings: dict[str, int] = {}
    for path in files:
        if "__pycache__" in path.parts:
            continue
        rel = f"{prefix}/{path.relative_to(root)}"
        ex = Extractor(path, rel, module_name(root, path))
        sites.extend(ex.run())
        if ex.docstrings:
            docstrings[rel] = ex.docstrings
    return {
        "root": str(root),
        "targets": targets,
        "sites": sites,
        "docstrings_with_cjk": docstrings,
    }
