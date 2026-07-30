from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from translator.languages import is_polish_language
from translator.schemas import ValidationIssue


class GlossaryTerm(BaseModel):
    key: str
    source: str
    preferred_pl: str
    alternatives: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    retain_english: bool = False
    retain_english_on_first_use: bool = False
    preserve_abbreviation: str | None = None
    short_pl: str | None = None


class DomainGlossary(BaseModel):
    terms: list[GlossaryTerm] = Field(default_factory=list)

    def terms_for_prompt(self, target_language: str = "Polish") -> str:
        lines = []
        for term in self.terms:
            extras = []
            if term.alternatives:
                extras.append(f"alternatives={', '.join(term.alternatives)}")
            if term.forbidden:
                extras.append(f"forbidden={', '.join(term.forbidden)}")
            if term.retain_english or term.retain_english_on_first_use:
                extras.append("retain English when useful")
            suffix = f" ({'; '.join(extras)})" if extras else ""
            if is_polish_language(target_language):
                lines.append(f"- {term.source} -> {term.preferred_pl}{suffix}")
            else:
                lines.append(
                    f"- source term: {term.source}; approved Polish equivalent: {term.preferred_pl}; "
                    f"use as a domain concept anchor only for target language '{target_language}'{suffix}"
                )
        return "\n".join(lines)

    def relevant_terms(self, source_text: str) -> list[GlossaryTerm]:
        lowered = source_text.lower()
        return [
            term for term in self.terms
            if term.source.lower() in lowered or term.key.replace("_", " ") in lowered
        ]

    def apply_mock_replacements(self, text: str) -> str:
        result = text
        for term in sorted(self.terms, key=lambda item: len(item.source), reverse=True):
            result = re.sub(
                re.escape(term.source),
                term.preferred_pl,
                result,
                flags=re.IGNORECASE,
            )
        return result

    def validate_translation(
        self,
        segment_id: str,
        source_text: str,
        translated_text: str,
        target_language: str = "Polish",
    ) -> list[ValidationIssue]:
        if not is_polish_language(target_language):
            return []

        issues: list[ValidationIssue] = []
        translated_lower = translated_text.lower()
        for term in self.relevant_terms(source_text):
            accepted = [term.preferred_pl, *term.alternatives]
            if term.short_pl:
                accepted.append(term.short_pl)
            if term.retain_english or term.retain_english_on_first_use:
                accepted.append(term.source)

            if not any(candidate and candidate.lower() in translated_lower for candidate in accepted):
                issues.append(
                    ValidationIssue(
                        segment_id=segment_id,
                        severity="warning",
                        issue_type="untranslated_fragment",
                        source_value=term.source,
                        translated_value=translated_text,
                        message=f"Termin '{term.source}' nie używa zatwierdzonego odpowiednika '{term.preferred_pl}'.",
                    )
                )

            for forbidden in term.forbidden:
                if forbidden.lower() in translated_lower:
                    issues.append(
                        ValidationIssue(
                            segment_id=segment_id,
                            severity="warning",
                            issue_type="forbidden_term",
                            source_value=term.source,
                            translated_value=forbidden,
                            message=f"Termin '{forbidden}' jest zabroniony dla '{term.source}'.",
                        )
                    )
        return issues


def load_glossary(path: str | Path) -> DomainGlossary:
    glossary_path = Path(path)
    if not glossary_path.exists():
        return DomainGlossary()

    text = glossary_path.read_text(encoding="utf-8")
    data = _load_yaml_with_fallback(text)
    raw_terms = data.get("terms", {}) if isinstance(data, dict) else {}
    terms = []
    for key, raw in raw_terms.items():
        if not isinstance(raw, dict):
            continue
        source = raw.get("source") or key.replace("_", " ")
        preferred = raw.get("preferred_pl") or source
        terms.append(
            GlossaryTerm(
                key=key,
                source=source,
                preferred_pl=preferred,
                alternatives=list(raw.get("alternatives", []) or []),
                forbidden=list(raw.get("forbidden", []) or []),
                retain_english=bool(raw.get("retain_english", False)),
                retain_english_on_first_use=bool(raw.get("retain_english_on_first_use", False)),
                preserve_abbreviation=raw.get("preserve_abbreviation"),
                short_pl=raw.get("short_pl"),
            )
        )
    return DomainGlossary(terms=terms)


def _load_yaml_with_fallback(text: str) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _parse_simple_terms_yaml(text)


def _parse_simple_terms_yaml(text: str) -> dict:
    result: dict[str, dict] = {"terms": {}}
    current_key: str | None = None
    current_list: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line == "terms:":
            continue

        if indent == 2 and line.endswith(":"):
            current_key = line[:-1]
            current_list = None
            result["terms"][current_key] = {}
            continue

        if current_key is None:
            continue

        if indent == 4 and ":" in line:
            name, value = line.split(":", 1)
            name = name.strip()
            value = value.strip()
            if not value:
                result["terms"][current_key][name] = []
                current_list = name
            else:
                result["terms"][current_key][name] = _parse_scalar(value)
                current_list = None
            continue

        if indent >= 6 and line.startswith("- ") and current_list:
            result["terms"][current_key].setdefault(current_list, []).append(_parse_scalar(line[2:].strip()))

    return result


def _parse_scalar(value: str) -> str | bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value.strip('"').strip("'")
