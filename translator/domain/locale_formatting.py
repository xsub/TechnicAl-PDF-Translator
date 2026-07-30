from __future__ import annotations

import re

from translator.schemas import JobConfig, TranslationResult


DECIMAL_COMMA_TARGETS = {
    "polish",
    "polski",
    "pl",
}

_DECIMAL_DOT_NUMBER_RE = re.compile(
    r"(?<![\w./-])(?P<integer>\d{1,3}(?:[ ,]\d{3})+|\d+)\.(?P<fraction>\d+)(?![\w./-])"
)
_STRUCTURAL_NUMBER_PREFIX_RE = re.compile(
    r"\b(?:section|sec\.|chapter|clause|table|figure|fig\.|appendix|annex|"
    r"sekcja|rozdział|tabela|rys\.|rysunek|punkt|pkt\.|ust\.|art\.)\s*$",
    re.IGNORECASE,
)


def apply_locale_formatting(translation: TranslationResult, config: JobConfig) -> TranslationResult:
    """Apply target-locale formatting that is safe and deterministic.

    The first supported rule localizes English decimal-dot notation to Polish
    decimal-comma notation, e.g. ``0.01`` -> ``0,01`` and
    ``1,234.56`` -> ``1 234,56``. Structural references such as
    ``section 1.2`` or ``table 2.1`` are intentionally preserved.
    """

    if not _target_uses_decimal_comma(config.target_language):
        return translation

    localized_text = localize_decimal_separator(translation.translated_text)
    if localized_text == translation.translated_text:
        return translation

    note = "Localized decimal separator for target locale."
    notes = list(translation.translator_notes)
    if note not in notes:
        notes.append(note)

    return translation.model_copy(
        update={
            "translated_text": localized_text,
            "translator_notes": notes,
        }
    )


def localize_decimal_separator(text: str) -> str:
    def replace_decimal(match: re.Match[str]) -> str:
        if _looks_like_structural_reference(text, match.start()):
            return match.group(0)

        integer = match.group("integer").replace(",", " ")
        fraction = match.group("fraction")
        return f"{integer},{fraction}"

    return _DECIMAL_DOT_NUMBER_RE.sub(replace_decimal, text or "")


def _target_uses_decimal_comma(target_language: str | None) -> bool:
    normalized = " ".join((target_language or "").strip().lower().split())
    return normalized in DECIMAL_COMMA_TARGETS


def _looks_like_structural_reference(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32):start]
    return bool(_STRUCTURAL_NUMBER_PREFIX_RE.search(prefix))
