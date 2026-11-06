"""Run with:  python -m pytest test_i18n.py
Guards the translation table: no missing keys, no mismatched {placeholders}."""
import re

import i18n

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def test_both_languages_have_the_same_keys():
    en, zh = set(i18n.STRINGS["en"]), set(i18n.STRINGS["zh"])
    assert not (en - zh), f"missing in zh: {sorted(en - zh)}"
    assert not (zh - en), f"missing in en: {sorted(zh - en)}"


def test_placeholders_match_between_languages():
    for key, en_text in i18n.STRINGS["en"].items():
        en_ph = set(PLACEHOLDER.findall(en_text))
        zh_ph = set(PLACEHOLDER.findall(i18n.STRINGS["zh"][key]))
        assert en_ph == zh_ph, f"{key}: en {en_ph} vs zh {zh_ph}"


def test_no_empty_texts():
    for lang, table in i18n.STRINGS.items():
        for key, text in table.items():
            assert text.strip(), f"{lang}/{key} is empty"


def test_resolve():
    assert i18n.resolve("English") == "en"
    assert i18n.resolve("Simplified Chinese") == "zh"
