"""i18n: the catalogs — six languages, every key, the same values, valid plural forms.

* The 繁體中文 source catalog parses, is canonical, and every key has translator context whose
  fingerprint matches today's words (a changed text under an old key fails: see docs/i18n.md).
* Every other language that has catalog files has every key, no extra key, the same values
  ({name}s) as the source and only the plural forms the language has. A language without
  files is reported as not translated yet; once the page loads i18n.js, all six must exist.
* The manifest matches the files; keys named in the code exist; the glossary is complete;
  the page runtime's language table matches Python's.
"""

from __future__ import annotations

import json
import re

import pytest

from duckmind.i18n import checks, icu, locales
from duckmind.i18n.catalog import STATIC, catalog_path, glossary, read_catalog

SOURCE = locales.SOURCE
TRANSLATIONS = [code for code in locales.CODES if code != SOURCE]


@pytest.fixture(scope="module")
def source() -> dict[str, str]:
    out: dict[str, str] = {}
    for domain in ("page", "web", "core", "neuro"):
        out.update(read_catalog(SOURCE, domain) or {})
    return out


def test_catalog_files_are_valid_and_canonical() -> None:
    assert checks.check_files() == []


def test_source_catalog_is_complete_and_parses(source: dict[str, str]) -> None:
    assert len(source) > 3000, "the source catalog lost most of its keys"
    for domain in ("page", "web", "core", "neuro"):
        assert catalog_path(SOURCE, domain).is_file(), f"no i18n.zh-Hant.{domain}.js"
    problems = [
        f"{key}: {p}" for key, message in source.items() for p in icu.check(message, SOURCE)
    ]
    assert problems == []
    for sep in (
        "fmt.sep.enum",
        "fmt.sep.babble",
        "fmt.sep.slash",
        "page.i18n.picker.label",
        "page.i18n.load_failed",
    ):
        assert sep in source


def test_every_key_has_context_describing_todays_words(source: dict[str, str]) -> None:
    assert checks.check_context(source) == []


def test_the_manifest_matches_the_files() -> None:
    assert checks.check_manifest() == []


@pytest.mark.parametrize("lang", TRANSLATIONS)
def test_translation_is_complete_with_the_same_values(lang: str, source: dict[str, str]) -> None:
    messages: dict[str, str] = {}
    found = False
    for domain in ("page", "web", "core", "neuro"):
        part = read_catalog(lang, domain)
        if part is not None:
            found = True
            messages.update(part)
    if not found:
        if checks.page_uses_i18n():
            pytest.fail(f"the page offers {lang} but it has no catalog files (i18n.{lang}.*.js)")
        pytest.skip(f"{lang}: not translated yet (no i18n.{lang}.*.js files)")
    report = checks.check_language(lang, messages, source)
    assert report["broken"] == [], f"{lang}: messages that do not parse"
    assert report["values"] == [], f"{lang}: values that differ from the 繁體中文 source"
    assert report["plural"] == [], f"{lang}: plural forms {lang} does not have (or no 'other')"
    assert report["extra"] == [], f"{lang}: keys the source does not have"
    assert report["missing"] == [], (
        f"{lang}: {len(report['missing'])} keys not translated (python -m duckmind.i18n export {lang} --missing)"
    )


def test_keys_named_in_code_exist(source: dict[str, str]) -> None:
    used = checks.used_keys(STATIC.parents[3])
    unknown = {key: places[:3] for key, places in used.items() if key not in source}
    assert unknown == {}


def test_glossary_names_every_term_in_all_five_languages() -> None:
    data = glossary()
    terms = data.get("terms") or []
    assert len(terms) >= 50
    for term in terms:
        assert term.get("zh") and term.get("meaning"), term
        suggest = term.get("suggest") or {}
        assert set(suggest) == set(TRANSLATIONS), term["zh"]
        assert all(isinstance(v, str) and v.strip() for v in suggest.values()), term["zh"]
    kept = {t["zh"] for t in terms if t.get("keep_as_is")}
    assert "蛋黃" in kept


def test_page_runtime_has_the_same_language_table() -> None:
    js = (STATIC / "i18n.js").read_text(encoding="utf-8")
    rows = re.findall(
        r'\{ code: "([^"]+)", native: "([^"]+)", script: "([^"]+)", font: (?:"([^"]+)"|null), decimal: "([^"]+)" \}',
        js,
    )
    assert [(c, n, s, f or None, d) for c, n, s, f, d in rows] == [
        (lang.code, lang.native, lang.script, lang.font, lang.decimal) for lang in locales.LANGS
    ]
    assert 'const SOURCE = "zh-Hant";' in js
    assert 'const DOMAINS = ["page", "web", "core", "neuro"];' in js


def test_context_file_has_its_legend() -> None:
    raw = json.loads((STATIC.parents[1] / "i18n" / "context.json").read_text(encoding="utf-8"))
    assert raw["@"]["source_lang"] == SOURCE
    assert "flags" in raw["@"] and "fields" in raw["@"]
