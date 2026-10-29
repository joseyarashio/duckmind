"""i18n: the catalog tools translators and developers use (python -m duckmind.i18n), on a copy.

add → set → export → import → check, with the checks that keep five translators honest: the
same values as the 繁體中文 source, only the plural forms a language has, no changed source
words under an old key.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from duckmind.i18n.__main__ import main
from duckmind.i18n.catalog import CONTEXT_FILE, STATIC, read_catalog


@pytest.fixture()
def copy(tmp_path: Path) -> list[str]:
    static = tmp_path / "static"
    static.mkdir()
    for path in STATIC.glob("i18n.zh-Hant.*.js"):
        shutil.copy(path, static / path.name)
    shutil.copy(CONTEXT_FILE, tmp_path / "context.json")
    return [
        "--static",
        str(static),
        "--context",
        str(tmp_path / "context.json"),
        "--root",
        str(tmp_path),
    ]


def test_add_set_export_import(
    copy: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = "page.test.dock_hint"
    assert (
        main(
            [
                *copy,
                "add",
                key,
                "牠自己倒退進充電座了（{n} 次）",
                "--where",
                "status banner",
                "--latin",
                "41",
            ]
        )
        == 0
    )
    assert main([*copy, "add", key, "別的字", "--where", "x"]) == 2  # a key keeps its words
    assert main([*copy, "add", "Bad Key", "字", "--where", "x"]) == 2
    good = "It backed into its dock by itself ({n, plural, one {# time} other {# times}})"
    assert main([*copy, "set", "en", key, good]) == 0
    assert (
        main(
            [
                *copy,
                "set",
                "es",
                key,
                "Entró solo ({n, plural, one {# vez} few {# veces} other {# veces}})",
            ]
        )
        == 2
    )
    assert main([*copy, "set", "ja", key, "自分で入った"]) == 2  # lost {n}
    assert main([*copy, "set", "ja", "page.nothing.here", "x"]) == 2
    static = Path(copy[1])
    assert read_catalog("en", "page", static) == {key: good}
    out = tmp_path / "ko.json"
    assert main([*copy, "export", "ko", "--missing", "--domain", "page", "-o", str(out)]) == 0
    exported = json.loads(out.read_text(encoding="utf-8"))["messages"]
    assert (
        exported[key]["zh"] == "牠自己倒退進充電座了（{n} 次）"
        and exported[key]["room"]["latin"] == 41
    )
    exported[key]["text"] = "혼자 충전대에 들어갔어요 ({n}번)"
    (tmp_path / "ko_in.json").write_text(
        json.dumps({"messages": {key: exported[key]}}, ensure_ascii=False), encoding="utf-8"
    )
    assert main([*copy, "import", "ko", str(tmp_path / "ko_in.json")]) == 0
    assert read_catalog("ko", "page", static) == {key: "혼자 충전대에 들어갔어요 ({n}번)"}
    capsys.readouterr()
    assert main([*copy, "check"]) == 1  # the other keys are not translated yet
    report = capsys.readouterr().out
    assert re.search(
        r"ko: \d+ keys not translated yet", report
    )  # (was the prefix "ko: 4": the count crossed 5000)
    assert "values" not in report and "broken" not in report


def test_changed_source_words_need_a_new_key_or_a_restamp(
    copy: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    static = Path(copy[1])
    messages = read_catalog("zh-Hant", "page", static) or {}
    messages["page.html.btnPlay.text"] = "▶ 讓時間走"
    from duckmind.i18n.catalog import write_catalog

    write_catalog("zh-Hant", "page", messages, static)
    capsys.readouterr()
    main([*copy, "check"])
    assert "page.html.btnPlay.text: its 繁體中文 changed" in capsys.readouterr().out
    assert main([*copy, "restamp", "page.html.btnPlay.text"]) == 0
    capsys.readouterr()
    main([*copy, "check"])
    assert "btnPlay.text: its" not in capsys.readouterr().out
