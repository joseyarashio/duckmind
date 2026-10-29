"""``python -m duckmind.i18n`` — the catalog tools (docs/i18n.md).

check                      everything the i18n tests check, as a list
stats                      keys per domain and how much of each language is done
find WORDS|KEY             search the 繁體中文 source and the keys
add KEY "繁體中文" --where "…" [--tone …] [--latin N] [--one-line] [--file F --line N]
                           a new source string (catalog + context); then translate it:
set LANG KEY "words"       one translation (checked: parses, same values, valid plurals)
export LANG [--missing] [--domain D] [-o FILE]
                           a translator's file: key → {zh, where, tone, room, values, now}
import LANG FILE           translations back in ({key: words} or the export's shape)
restamp KEY                after a same-meaning fix of a 繁體中文 text (translations stay valid)
format                     rewrite every catalog file canonically, then the manifest
manifest                   rewrite i18n.manifest.js (which catalog files exist)
pseudo KEY [values JSON]   a message in the pseudo-locale
scan [--static DIR]        page words that do not come from a catalog (hard-coded)
pack ENGINE                the engine pack: templates, keys, keys missing from the catalog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import checks, icu
from .catalog import (
    CONTEXT_FILE,
    DOMAINS,
    STATIC,
    Catalogs,
    context_file,
    domain_of,
    read_catalog,
    src_hash,
    valid_key,
    write_catalog,
    write_manifest,
)
from .locales import CODES, PSEUDO, SOURCE
from .message import render


def _source(static: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for domain in DOMAINS:
        out.update(read_catalog(SOURCE, domain, static) or {})
    return out


def _lang_messages(code: str, static: Path) -> dict[str, dict[str, str]]:
    return {d: dict(read_catalog(code, d, static) or {}) for d in DOMAINS}


def _write_context(data: dict[str, Any], path: Path = CONTEXT_FILE) -> None:
    entries = data.get("entries", {})
    lines = ["{", f'"@": {json.dumps(data.get("@", {}), ensure_ascii=False)},', '"entries": {']
    items = sorted(entries.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append(
            f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}{comma}"
        )
    lines += ["}", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_check(args: argparse.Namespace) -> int:
    static = Path(args.static)
    problems = checks.check_files(static)
    source = _source(static)
    problems += checks.check_context(source, context_file(Path(args.context)).get("entries", {}))
    problems += checks.check_manifest(static)
    for code, messages in checks.load_all(static).items():
        if code == SOURCE:
            for key, message in messages.items():
                problems += [f"{SOURCE} {key}: {p}" for p in icu.check(message, SOURCE)]
            continue
        report = checks.check_language(code, messages, source)
        for kind, items in report.items():
            if kind == "missing":
                if items:
                    problems.append(
                        f"{code}: {len(items)} keys not translated yet (e.g. {items[0]})"
                    )
                continue
            problems += [f"{code} {kind}: {item}" for item in items]
    used = checks.used_keys(Path(args.root))
    for key, places in sorted(used.items()):
        if key not in source:
            problems.append(f"{key}: used at {places[0]} but not in the source catalog")
    if checks.page_uses_i18n(static):
        problems += [f"hard-coded: {x}" for x in checks.scan_static(static, source)]
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_stats(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    by_domain: dict[str, int] = {}
    for key in source:
        by_domain[domain_of(key)] = by_domain.get(domain_of(key), 0) + 1
    print(f"source {SOURCE}: {len(source)} keys", json.dumps(by_domain))
    all_langs = checks.load_all(static)
    for code in CODES:
        if code == SOURCE:
            continue
        have = all_langs.get(code, {})
        done = sum(1 for k in source if k in have)
        print(f"{code}: {done}/{len(source)} ({100 * done / max(1, len(source)):.1f} %)")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    ctx = context_file(Path(args.context)).get("entries", {})
    needle = args.words
    hits = [k for k, v in source.items() if needle in k or needle in v]
    for key in sorted(hits)[: args.limit]:
        entry = ctx.get(key, {})
        print(f"{key}\n    {source[key]}\n    where: {str(entry.get('where', ''))[:140]}")
        for origin in entry.get("origins", [])[:3]:
            print(f"    from: {origin}")
    print(f"{len(hits)} match(es)")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    static = Path(args.static)
    key = args.key
    if not valid_key(key):
        print(
            f"bad key {key!r}: lowercase namespace (page/web/core/world/neuro/fmt …) then dotted segments"
        )
        return 2
    domain = domain_of(key)
    messages = dict(read_catalog(SOURCE, domain, static) or {})
    if key in messages:
        print(f"{key} already exists: {messages[key]!r} (a changed meaning needs a new key)")
        return 2
    problems = icu.check(args.text, SOURCE)
    if problems:
        print("\n".join(problems))
        return 2
    messages[key] = args.text
    write_catalog(SOURCE, domain, messages, static)
    data = context_file(Path(args.context))
    entry: dict[str, Any] = {
        "where": args.where,
        "kind": args.kind,
        "src": src_hash(args.text),
        "zh_len": len(icu.plain_text(icu.parse(args.text))),
    }
    if args.tone:
        entry["tone"] = args.tone
    if args.latin or args.one_line:
        entry["room"] = {"latin": args.latin, "one_line": bool(args.one_line)}
    if args.file:
        entry["origins"] = [f"added {args.file}:{args.line or ''}".strip(":")]
    if args.note:
        entry["notes"] = [args.note]
    names = icu.arguments(icu.parse(args.text))
    if names:
        entry["params"] = {n: {"type": kind} for n, kind in names.items()}
    data.setdefault("entries", {})[key] = entry
    _write_context(data, Path(args.context))
    write_manifest(static)
    print(f"added {key} to i18n.{SOURCE}.{domain}.js — now add it in en, ja, es, zh-Hans, ko:")
    print(f'  python -m duckmind.i18n set en {key} "…"   (and the four others)')
    return 0


def _check_translation(code: str, key: str, text: str, source: dict[str, str]) -> list[str]:
    report = checks.check_language(code, {key: text}, {key: source[key]})
    return (
        report["broken"]
        + report["values"]
        + report["plural"]
        + [f"{key}: not in the source catalog" for _ in report["extra"]]
    )


def cmd_set(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    if args.lang not in CODES or args.lang == SOURCE:
        print(
            f"LANG must be one of {', '.join(c for c in CODES if c != SOURCE)} (the source is edited with add)"
        )
        return 2
    if args.key not in source:
        print(f"{args.key}: not in the source catalog")
        return 2
    problems = _check_translation(args.lang, args.key, args.text, source)
    if problems:
        print("\n".join(problems))
        return 2
    domain = domain_of(args.key)
    messages = dict(read_catalog(args.lang, domain, static) or {})
    messages[args.key] = args.text
    write_catalog(args.lang, domain, messages, static)
    write_manifest(static)
    print(f"{args.lang} {args.key} = {args.text}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    ctx = context_file(Path(args.context)).get("entries", {})
    have: dict[str, str] = {}
    if args.lang in CODES:
        for domain in DOMAINS:
            have.update(read_catalog(args.lang, domain, static) or {})
    out: dict[str, Any] = {}
    for key in sorted(source):
        if args.domain and domain_of(key) != args.domain:
            continue
        if args.missing and key in have:
            continue
        entry = ctx.get(key, {})
        item = {
            "zh": source[key],
            **{
                k: entry[k]
                for k in (
                    "where",
                    "tone",
                    "kind",
                    "room",
                    "params",
                    "notes",
                    "flags",
                    "same_words_as",
                )
                if k in entry
            },
        }
        if key in have:
            item["now"] = have[key]
        out[key] = item
    text = (
        json.dumps(
            {"lang": args.lang, "source": SOURCE, "messages": out}, ensure_ascii=False, indent=1
        )
        + "\n"
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"{len(out)} message(s) → {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    if args.lang not in CODES or args.lang == SOURCE:
        print("LANG must be a translation language")
        return 2
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    incoming: dict[str, str] = {}
    for key, value in data.items():
        text = value.get("text") or value.get("now") if isinstance(value, dict) else value
        if isinstance(text, str) and text:
            incoming[key] = text
    problems: list[str] = []
    for key, text in incoming.items():
        if key not in source:
            problems.append(f"{key}: not in the source catalog")
            continue
        problems += _check_translation(args.lang, key, text, source)
    if problems and not args.force:
        print("\n".join(problems[:200]))
        print(
            f"{len(problems)} problem(s): nothing written (fix them, or --force to write the good ones)"
        )
        return 2
    bad = {p.split(":", 1)[0] for p in problems}
    per_domain = _lang_messages(args.lang, static)
    written = 0
    for key, text in incoming.items():
        if key in source and key not in bad:
            per_domain[domain_of(key)][key] = text
            written += 1
    for domain, messages in per_domain.items():
        if messages:
            write_catalog(args.lang, domain, messages, static)
    write_manifest(static)
    print(f"{args.lang}: {written} message(s) written, {len(problems)} problem(s)")
    return 0


def cmd_restamp(args: argparse.Namespace) -> int:
    static = Path(args.static)
    source = _source(static)
    data = context_file(Path(args.context))
    entries = data.get("entries", {})
    if args.key not in source or args.key not in entries:
        print(f"{args.key}: not in the source catalog / context")
        return 2
    entries[args.key]["src"] = src_hash(source[args.key])
    _write_context(data, Path(args.context))
    print(f"{args.key}: context now describes {source[args.key]!r}")
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    static = Path(args.static)
    for code in CODES:
        for domain in DOMAINS:
            messages = read_catalog(code, domain, static)
            if messages is not None:
                write_catalog(code, domain, messages, static)
    print(write_manifest(static))
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    print(write_manifest(Path(args.static)))
    return 0


def cmd_pseudo(args: argparse.Namespace) -> int:
    params = json.loads(args.values) if args.values else {}
    print(render({"k": args.key, "p": params}, PSEUDO, catalogs=Catalogs(Path(args.static))))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    found = checks.scan_static(Path(args.static))
    for item in found:
        print(item)
    print(f"{len(found)} hard-coded text(s)")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    from . import packs

    pack = packs.build(args.engine)
    if pack is None:
        print(f"{args.engine}: not in this tree")
        return 1
    print(
        f"{args.engine}: {len(pack.sites)} templates the page can show, {len(set(pack.key_of.values()))} keys, "
        f"{len(pack.rename)} renamed, {len(pack.missing)} missing from the catalog"
    )
    for key in pack.missing[:50]:
        print("  missing:", key)
    return 1 if pack.missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.i18n",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--static", default=str(STATIC), help="the page's static directory (catalog files)"
    )
    parser.add_argument(
        "--root", default=str(STATIC.parents[3]), help="the tree root (for used keys)"
    )
    parser.add_argument("--context", default=str(CONTEXT_FILE), help="the translator context file")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("find")
    p.add_argument("words")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("add")
    p.add_argument("key")
    p.add_argument("text")
    p.add_argument("--where", required=True, help="where the words show (element, card, field)")
    p.add_argument("--tone", default="")
    p.add_argument(
        "--kind",
        default="template",
        choices=["static", "attr", "canvas", "template", "server", "format"],
    )
    p.add_argument(
        "--latin", type=int, default=0, help="Latin letters that fit on one line at 390 px"
    )
    p.add_argument("--one-line", action="store_true", help="the element never wraps")
    p.add_argument("--note", default="")
    p.add_argument("--file", default="")
    p.add_argument("--line", default="")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("set")
    p.add_argument("lang")
    p.add_argument("key")
    p.add_argument("text")
    p.set_defaults(fn=cmd_set)
    p = sub.add_parser("export")
    p.add_argument("lang")
    p.add_argument("--missing", action="store_true")
    p.add_argument("--domain", choices=list(DOMAINS))
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("import")
    p.add_argument("lang")
    p.add_argument("file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("restamp")
    p.add_argument("key")
    p.set_defaults(fn=cmd_restamp)
    sub.add_parser("format").set_defaults(fn=cmd_format)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    p = sub.add_parser("pseudo")
    p.add_argument("key")
    p.add_argument("values", nargs="?")
    p.set_defaults(fn=cmd_pseudo)
    sub.add_parser("scan").set_defaults(fn=cmd_scan)
    p = sub.add_parser("pack")
    p.add_argument("engine", choices=["v4_0", "v4_1"])
    p.set_defaults(fn=cmd_pack)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
