from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from translator.schemas import DocumentSegment, ProtectedToken, TranslationResult, ValidationIssue


@dataclass(frozen=True)
class _PatternSpec:
    kind: str
    pattern: re.Pattern[str]


_PATTERNS: list[_PatternSpec] = [
    _PatternSpec("cas", re.compile(r"\b\d{2,7}-\d{2}-\d\b")),
    _PatternSpec("regulation", re.compile(r"\bRegulation\s*\((?:EU|EC)\)\s*No\.?\s*\d+/\d+\b", re.IGNORECASE)),
    _PatternSpec("standard", re.compile(r"\b(?:EN|ISO|DIN|ASTM)\s*\d+(?:[-:/]\d+)*(?:\s*[:/]\s*\d+)?\b", re.IGNORECASE)),
    _PatternSpec("regulation", re.compile(r"\b(?:EU|EC)\s*No\.?\s*\d+/\d+\b", re.IGNORECASE)),
    _PatternSpec("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    _PatternSpec("abbreviation", re.compile(r"\b(?:ND|N/A|NA|LOQ|LOD|SML|OML)\b", re.IGNORECASE)),
    _PatternSpec("formula", re.compile(r"\b(?=[A-Z0-9]*\d)(?:[A-Z][a-z]?\d*){2,}\b")),
    _PatternSpec("comparator", re.compile(r"(?:≤|≥|<|>|=|±)")),
    _PatternSpec(
        "unit",
        re.compile(
            r"\b(?:mg/dm(?:2|²)|mg/kg|µg/kg|ug/kg|g/m(?:2|²)|kg|mg|µg|ug|ppm|ppb|days?|hours?|hrs?|min|s)\b|°C|%",
            re.IGNORECASE,
        ),
    ),
    _PatternSpec("number", re.compile(r"(?<![A-Za-z0-9/-])(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,]\d+)?(?![A-Za-z0-9/-])")),
]

_ISSUE_BY_KIND = {
    "number": "missing_number",
    "unit": "missing_unit",
    "comparator": "changed_comparator",
    "cas": "missing_reference",
    "standard": "missing_reference",
    "regulation": "missing_reference",
    "date": "missing_reference",
    "abbreviation": "missing_reference",
    "formula": "missing_reference",
}

_CRITICAL_KINDS = {"number", "unit", "comparator", "cas", "standard", "regulation", "date"}


def extract_protected_tokens(text: str, prefix: str = "TOKEN") -> list[ProtectedToken]:
    matches: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    for spec in _PATTERNS:
        for match in spec.pattern.finditer(text or ""):
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue
            occupied.append((start, end))
            matches.append((start, end, spec.kind, match.group(0)))

    matches.sort(key=lambda item: (item[0], item[1]))
    return [
        ProtectedToken(
            token_id=f"{prefix}_{index + 1}",
            kind=kind,  # type: ignore[arg-type]
            text=value,
            normalized=normalize_token(kind, value),
            start=start,
            end=end,
        )
        for index, (start, end, kind, value) in enumerate(matches)
    ]


def protect_segment(segment: DocumentSegment) -> DocumentSegment:
    return segment.model_copy(
        update={
            "protected_tokens": extract_protected_tokens(
                segment.source_text,
                prefix=segment.segment_id.replace("-", "_").upper(),
            )
        }
    )


def validate_segment_invariants(segment: DocumentSegment, translation: TranslationResult) -> list[ValidationIssue]:
    source_tokens = segment.protected_tokens or extract_protected_tokens(segment.source_text, prefix="SOURCE")
    target_tokens = extract_protected_tokens(translation.translated_text, prefix="TARGET")
    issues: list[ValidationIssue] = []

    source_counter = _counter_by_kind(source_tokens)
    target_counter = _counter_by_kind(target_tokens)

    for (kind, normalized), source_count in source_counter.items():
        target_count = target_counter.get((kind, normalized), 0)
        if target_count >= source_count:
            continue

        issue_type = _ISSUE_BY_KIND.get(kind, "missing_reference")
        severity = "critical" if kind in _CRITICAL_KINDS else "warning"
        source_value = _first_text(source_tokens, kind, normalized)
        issues.append(
            ValidationIssue(
                segment_id=segment.segment_id,
                severity=severity,  # type: ignore[arg-type]
                issue_type=issue_type,  # type: ignore[arg-type]
                source_value=source_value,
                translated_value=None,
                message=f"Chroniony element '{source_value}' ({kind}) nie występuje w tłumaczeniu.",
            )
        )

    for (kind, normalized), target_count in target_counter.items():
        source_count = source_counter.get((kind, normalized), 0)
        if target_count <= source_count:
            continue
        if kind not in {"number", "unit", "comparator"}:
            continue
        translated_value = _first_text(target_tokens, kind, normalized)
        issues.append(
            ValidationIssue(
                segment_id=segment.segment_id,
                severity="critical" if kind in {"number", "unit", "comparator"} else "warning",
                issue_type="unexpected_addition",
                source_value=None,
                translated_value=translated_value,
                message=f"Tłumaczenie dodało chroniony element '{translated_value}' ({kind}), którego nie było w źródle.",
            )
        )

    issues.extend(_validate_negation(segment, translation))
    return issues


def normalize_token(kind: str, value: str) -> str:
    cleaned = value.strip()
    if kind == "number":
        cleaned = cleaned.replace(" ", "").replace(",", ".")
    if kind == "unit":
        cleaned = cleaned.replace("µ", "u").replace("²", "2").lower()
    if kind in {"standard", "regulation", "abbreviation", "formula"}:
        cleaned = re.sub(r"\s+", "", cleaned).upper()
    if kind == "comparator":
        cleaned = {"≤": "<=", "≥": ">=", "=": "="}.get(cleaned, cleaned)
    return cleaned


def _counter_by_kind(tokens: list[ProtectedToken]) -> Counter[tuple[str, str]]:
    return Counter((token.kind, token.normalized) for token in tokens)


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _first_text(tokens: list[ProtectedToken], kind: str, normalized: str) -> str | None:
    for token in tokens:
        if token.kind == kind and token.normalized == normalized:
            return token.text
    return None


def _validate_negation(segment: DocumentSegment, translation: TranslationResult) -> list[ValidationIssue]:
    source = segment.source_text.lower()
    target = translation.translated_text.lower()
    source_has_negation = bool(re.search(r"\b(?:no|not|none|without|does\s+not|did\s+not|was\s+not|were\s+not)\b", source))
    target_has_negation = bool(re.search(r"\b(?:nie|brak|bez|żaden|zadna|żadna|niewykryto|nie\s+wykryto)\b", target))

    if source_has_negation and not target_has_negation:
        return [
            ValidationIssue(
                segment_id=segment.segment_id,
                severity="critical",
                issue_type="missing_negation",
                source_value=segment.source_text,
                translated_value=translation.translated_text,
                message="Źródło zawiera negację, której nie widać w tłumaczeniu.",
            )
        ]
    return []
