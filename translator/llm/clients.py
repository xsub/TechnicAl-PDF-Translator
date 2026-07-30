from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from translator.debug import log_debug, log_exception, text_preview
from translator.domain.glossary import DomainGlossary
from translator.domain.locale_formatting import apply_locale_formatting
from translator.schemas import (
    DocumentSegment,
    JobConfig,
    ReviewFinding,
    ReviewResult,
    TokenUsage,
    TranslationResult,
)


class TranslatorClient(Protocol):
    def translate(self, segment: DocumentSegment, glossary: DomainGlossary, config: JobConfig) -> TranslationResult:
        ...

    def revise(
        self,
        segment: DocumentSegment,
        current: TranslationResult,
        findings: list[ReviewFinding],
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> TranslationResult:
        ...


class ReviewerClient(Protocol):
    def review(
        self,
        segment: DocumentSegment,
        translation: TranslationResult,
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> ReviewResult:
        ...


class MockTechnicalTranslator:
    """Deterministic demo translator.

    This is intentionally modest. It exists so the full PDF workflow can be
    tested without vendor API keys; production translation should use the LLM
    adapters below.
    """

    _phrase_replacements = [
        (r"\bNo specific migration was detected\b", "Nie wykryto migracji specyficznej"),
        (r"\bNo migration was detected\b", "Nie wykryto migracji"),
        (r"\bwas below\b", "była poniżej"),
        (r"\bwere below\b", "były poniżej"),
        (r"\bat\b", "w temperaturze"),
        (r"\bfor\b", "przez"),
        (r"\bdays\b", "dni"),
        (r"\bday\b", "dzień"),
        (r"\bhours\b", "godzin"),
        (r"\bhour\b", "godzinę"),
        (r"\bResult\b", "Wynik"),
        (r"\bLimit\b", "Limit"),
        (r"\bMethod\b", "Metoda"),
        (r"\bSubstance\b", "Substancja"),
        (r"\bTest conditions\b", "Warunki badania"),
        (r"\bDeclaration of compliance\b", "Deklaracja zgodności"),
        (r"\bCompliant\b", "Zgodny"),
        (r"\bNot detected\b", "Nie wykryto"),
    ]

    def translate(self, segment: DocumentSegment, glossary: DomainGlossary, config: JobConfig) -> TranslationResult:
        log_debug(
            "mock.translate.start",
            segment_id=segment.segment_id,
            source_preview=text_preview(segment.source_text),
        )
        translated = glossary.apply_mock_replacements(segment.source_text)
        for pattern, replacement in self._phrase_replacements:
            translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

        uncertain_terms = []
        notes = ["mock translator - use real provider for production documents"]
        confidence = "high"

        if _looks_english(translated) and translated.strip().lower() == segment.source_text.strip().lower():
            uncertain_terms.append("untranslated_or_unknown_segment")
            confidence = "low"

        result = TranslationResult(
            segment_id=segment.segment_id,
            translated_text=translated,
            uncertain_terms=uncertain_terms,
            translator_notes=notes,
            confidence=confidence,  # type: ignore[arg-type]
        )
        result = apply_locale_formatting(result, config)
        log_debug(
            "mock.translate.done",
            segment_id=segment.segment_id,
            confidence=result.confidence,
            translated_preview=text_preview(result.translated_text),
        )
        return result

    def revise(
        self,
        segment: DocumentSegment,
        current: TranslationResult,
        findings: list[ReviewFinding],
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> TranslationResult:
        for finding in findings:
            if finding.proposed_translation:
                return apply_locale_formatting(
                    current.model_copy(
                        update={
                            "translated_text": finding.proposed_translation,
                            "translator_notes": [*current.translator_notes, "Applied reviewer proposal."],
                            "confidence": "medium",
                        }
                    ),
                    config,
                )
        return self.translate(segment, glossary, config)


class MockTechnicalReviewer:
    def review(
        self,
        segment: DocumentSegment,
        translation: TranslationResult,
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> ReviewResult:
        log_debug(
            "mock.review.start",
            segment_id=segment.segment_id,
            source_preview=text_preview(segment.source_text),
            translation_preview=text_preview(translation.translated_text),
        )
        findings: list[ReviewFinding] = []

        if translation.confidence == "low":
            findings.append(
                ReviewFinding(
                    segment_id=segment.segment_id,
                    severity="major",
                    category="untranslated_fragment",
                    source_evidence=segment.source_text,
                    translation_evidence=translation.translated_text,
                    explanation="Tłumaczenie wygląda jak nieprzetłumaczony fragment źródłowy.",
                    proposed_translation=None,
                    confidence="high",
                )
            )

        if segment.source_text.strip().lower() == translation.translated_text.strip().lower() and _looks_english(segment.source_text):
            findings.append(
                ReviewFinding(
                    segment_id=segment.segment_id,
                    severity="major",
                    category="unchanged_source",
                    source_evidence=segment.source_text,
                    translation_evidence=translation.translated_text,
                    explanation="Fragment angielski został pozostawiony bez tłumaczenia.",
                    proposed_translation=None,
                    confidence="high",
                )
            )

        verdict = "accept"
        if any(finding.severity in {"critical", "major"} for finding in findings):
            verdict = "human_review"
        elif findings:
            verdict = "revise"

        result = ReviewResult(segment_id=segment.segment_id, verdict=verdict, findings=findings)
        log_debug(
            "mock.review.done",
            segment_id=segment.segment_id,
            verdict=result.verdict,
            findings=len(result.findings),
        )
        return result


class OpenAITranslator:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-5-mini")

    def translate(self, segment: DocumentSegment, glossary: DomainGlossary, config: JobConfig) -> TranslationResult:
        prompt = _read_prompt("translator.txt")
        payload = _segment_payload(segment, glossary, config)
        log_debug(
            "openai.translate.prepare",
            segment_id=segment.segment_id,
            model=self.model_name,
            source_chars=len(segment.source_text),
            protected_tokens=len(segment.protected_tokens),
            source_preview=text_preview(segment.source_text),
        )
        result = _invoke_openai_structured(self.model_name, prompt, payload, TranslationResult, operation="translate")
        if result.segment_id != segment.segment_id:
            result = result.model_copy(update={"segment_id": segment.segment_id})
        return apply_locale_formatting(result, config)

    def revise(
        self,
        segment: DocumentSegment,
        current: TranslationResult,
        findings: list[ReviewFinding],
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> TranslationResult:
        prompt = _read_prompt("translator.txt")
        payload = _segment_payload(segment, glossary, config)
        payload["current_translation"] = current.model_dump(mode="json")
        payload["review_findings"] = [finding.model_dump(mode="json") for finding in findings]
        payload["instruction"] = "Revise only the material issues. Preserve protected tokens."
        log_debug(
            "openai.revise.prepare",
            segment_id=segment.segment_id,
            model=self.model_name,
            findings=len(findings),
            current_preview=text_preview(current.translated_text),
        )
        result = _invoke_openai_structured(self.model_name, prompt, payload, TranslationResult, operation="revise")
        if result.segment_id != segment.segment_id:
            result = result.model_copy(update={"segment_id": segment.segment_id})
        return apply_locale_formatting(result, config)


class OpenAIReviewer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OPENAI_REVIEW_MODEL", "gpt-5-mini")

    def review(
        self,
        segment: DocumentSegment,
        translation: TranslationResult,
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> ReviewResult:
        prompt = _read_prompt("reviewer.txt")
        payload = _review_payload(segment, translation, glossary, config)
        log_debug(
            "openai.review.prepare",
            segment_id=segment.segment_id,
            model=self.model_name,
            source_chars=len(segment.source_text),
            translation_chars=len(translation.translated_text),
            source_preview=text_preview(segment.source_text),
            translation_preview=text_preview(translation.translated_text),
        )
        result = _invoke_openai_structured(self.model_name, prompt, payload, ReviewResult, operation="review")
        if result.segment_id != segment.segment_id:
            result = result.model_copy(update={"segment_id": segment.segment_id})
        return result


class AnthropicReviewer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("ANTHROPIC_REVIEW_MODEL", "claude-sonnet-4-5")

    def review(
        self,
        segment: DocumentSegment,
        translation: TranslationResult,
        glossary: DomainGlossary,
        config: JobConfig,
    ) -> ReviewResult:
        prompt = _read_prompt("reviewer.txt")
        payload = _review_payload(segment, translation, glossary, config)
        log_debug(
            "anthropic.review.prepare",
            segment_id=segment.segment_id,
            model=self.model_name,
            source_chars=len(segment.source_text),
            translation_chars=len(translation.translated_text),
            source_preview=text_preview(segment.source_text),
            translation_preview=text_preview(translation.translated_text),
        )
        result = _invoke_anthropic_structured(self.model_name, prompt, payload, ReviewResult, operation="review")
        if result.segment_id != segment.segment_id:
            result = result.model_copy(update={"segment_id": segment.segment_id})
        return result


def build_translator(config: JobConfig) -> TranslatorClient:
    log_debug("llm.build_translator", provider=config.translator_provider)
    if config.translator_provider == "openai":
        return OpenAITranslator()
    return MockTechnicalTranslator()


def build_reviewer(config: JobConfig) -> ReviewerClient:
    log_debug("llm.build_reviewer", provider=config.reviewer_provider)
    if config.reviewer_provider == "anthropic":
        return AnthropicReviewer()
    if config.reviewer_provider == "openai":
        return OpenAIReviewer()
    return MockTechnicalReviewer()


def estimate_translation_request_tokens(
    segment: DocumentSegment,
    glossary: DomainGlossary,
    config: JobConfig,
) -> int:
    return _estimate_structured_request_tokens(
        _read_prompt("translator.txt"),
        _segment_payload(segment, glossary, config),
        TranslationResult,
    )


def estimate_revision_request_tokens(
    segment: DocumentSegment,
    current: TranslationResult,
    findings: list[ReviewFinding],
    glossary: DomainGlossary,
    config: JobConfig,
) -> int:
    payload = _segment_payload(segment, glossary, config)
    payload["current_translation"] = current.model_dump(mode="json")
    payload["review_findings"] = [finding.model_dump(mode="json") for finding in findings]
    payload["instruction"] = "Revise only the material issues. Preserve protected tokens."
    return _estimate_structured_request_tokens(_read_prompt("translator.txt"), payload, TranslationResult)


def estimate_review_request_tokens(
    segment: DocumentSegment,
    translation: TranslationResult,
    glossary: DomainGlossary,
    config: JobConfig,
) -> int:
    return _estimate_structured_request_tokens(
        _read_prompt("reviewer.txt"),
        _review_payload(segment, translation, glossary, config),
        ReviewResult,
    )


def _invoke_openai_structured(
    model_name: str,
    system_prompt: str,
    payload: dict,
    schema: type,
    *,
    operation: Literal["translate", "review", "revise"],
):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Brakuje pakietu langchain-openai. Uruchom: python -m pip install -e .") from exc

    timeout = _llm_timeout_seconds()
    max_retries = _llm_max_retries()
    started_at = time.perf_counter()
    log_debug(
        "llm.openai.invoke.start",
        model=model_name,
        schema=schema.__name__,
        timeout_s=timeout,
        max_retries=max_retries,
        system_prompt_chars=len(system_prompt),
        payload_summary=_payload_summary(payload),
    )

    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        timeout=timeout,
        max_retries=max_retries,
    )
    structured = model.with_structured_output(schema, include_raw=True)
    try:
        result = structured.invoke(
            [
                ("system", system_prompt),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        parsed, raw = _parse_structured_result(result, schema)
        usage = _extract_token_usage(raw, provider="openai", model=model_name, operation=operation)
        if hasattr(parsed, "model_copy"):
            parsed = parsed.model_copy(update={"token_usage": usage})
        log_debug(
            "llm.openai.invoke.done",
            model=model_name,
            schema=schema.__name__,
            duration_s=round(time.perf_counter() - started_at, 3),
            token_usage=usage.model_dump(mode="json") if usage else None,
            result_summary=_result_summary(parsed),
        )
        return parsed
    except Exception as exc:
        log_exception(
            "llm.openai.invoke.error",
            model=model_name,
            schema=schema.__name__,
            duration_s=round(time.perf_counter() - started_at, 3),
            error_type=type(exc).__name__,
            error=str(exc),
            payload_summary=_payload_summary(payload),
        )
        raise


def _invoke_anthropic_structured(
    model_name: str,
    system_prompt: str,
    payload: dict,
    schema: type,
    *,
    operation: Literal["translate", "review", "revise"],
):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError("Brakuje pakietu langchain-anthropic. Uruchom: python -m pip install -e .") from exc

    timeout = _llm_timeout_seconds()
    max_retries = _llm_max_retries()
    started_at = time.perf_counter()
    log_debug(
        "llm.anthropic.invoke.start",
        model=model_name,
        schema=schema.__name__,
        timeout_s=timeout,
        max_retries=max_retries,
        system_prompt_chars=len(system_prompt),
        payload_summary=_payload_summary(payload),
    )

    model = ChatAnthropic(
        model_name=model_name,
        temperature=0,
        timeout=timeout,
        max_retries=max_retries,
    )
    structured = model.with_structured_output(schema, include_raw=True)
    try:
        result = structured.invoke(
            [
                ("system", system_prompt),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        parsed, raw = _parse_structured_result(result, schema)
        usage = _extract_token_usage(raw, provider="anthropic", model=model_name, operation=operation)
        if hasattr(parsed, "model_copy"):
            parsed = parsed.model_copy(update={"token_usage": usage})
        log_debug(
            "llm.anthropic.invoke.done",
            model=model_name,
            schema=schema.__name__,
            duration_s=round(time.perf_counter() - started_at, 3),
            token_usage=usage.model_dump(mode="json") if usage else None,
            result_summary=_result_summary(parsed),
        )
        return parsed
    except Exception as exc:
        log_exception(
            "llm.anthropic.invoke.error",
            model=model_name,
            schema=schema.__name__,
            duration_s=round(time.perf_counter() - started_at, 3),
            error_type=type(exc).__name__,
            error=str(exc),
            payload_summary=_payload_summary(payload),
        )
        raise


def _parse_structured_result(result: Any, schema: type) -> tuple[Any, Any | None]:
    if isinstance(result, dict) and "parsed" in result:
        parsing_error = result.get("parsing_error")
        if parsing_error:
            raise RuntimeError(f"Structured output parsing failed: {parsing_error}")
        parsed = result.get("parsed")
        raw = result.get("raw")
        if parsed is None:
            raise RuntimeError("Structured output parsing returned no parsed result.")
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed), raw

    return result if isinstance(result, schema) else schema.model_validate(result), None


def _extract_token_usage(
    raw_message: Any | None,
    *,
    provider: Literal["openai", "anthropic", "mock"],
    model: str,
    operation: Literal["translate", "review", "revise"],
) -> TokenUsage | None:
    if raw_message is None:
        return None

    usage_metadata = _metadata_value(raw_message, "usage_metadata")
    response_metadata = _metadata_value(raw_message, "response_metadata")
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}

    usage_candidates = [
        usage_metadata,
        _metadata_value(raw_message, "token_usage"),
        _metadata_value(raw_message, "usage"),
        response_metadata.get("token_usage"),
        response_metadata.get("usage"),
    ]

    for usage in usage_candidates:
        if not isinstance(usage, dict):
            continue

        input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens", "input_token_count")
        output_tokens = _usage_int(usage, "output_tokens", "completion_tokens", "output_token_count")
        total_tokens = _usage_int(usage, "total_tokens", "total_token_count")
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        if input_tokens or output_tokens or total_tokens:
            return TokenUsage(
                provider=provider,
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

    log_debug(
        "llm.token_usage.unavailable",
        provider=provider,
        model=model,
        operation=operation,
        raw_type=type(raw_message).__name__,
        response_metadata=getattr(raw_message, "response_metadata", None),
        usage_metadata=getattr(raw_message, "usage_metadata", None),
    )
    return None


def _metadata_value(raw_message: Any, key: str) -> Any:
    if isinstance(raw_message, dict):
        return raw_message.get(key)
    return getattr(raw_message, key, None)


def _usage_int(usage: dict, *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _segment_payload(segment: DocumentSegment, glossary: DomainGlossary, config: JobConfig) -> dict:
    return {
        "source_language": config.source_language,
        "target_language": config.target_language,
        "domain": config.domain,
        "mode": config.mode,
        "segment": segment.model_dump(mode="json"),
        "glossary": glossary.terms_for_prompt(config.target_language),
    }


def _review_payload(
    segment: DocumentSegment,
    translation: TranslationResult,
    glossary: DomainGlossary,
    config: JobConfig,
) -> dict:
    payload = _segment_payload(segment, glossary, config)
    payload["translation"] = translation.model_dump(mode="json")
    return payload


def _read_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "domain" / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


def _looks_english(text: str) -> bool:
    return bool(re.search(r"\b(the|and|of|was|were|for|with|migration|detected|compliance)\b", text, re.IGNORECASE))


def _llm_timeout_seconds() -> float:
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


def _llm_max_retries() -> int:
    return int(os.getenv("LLM_MAX_RETRIES", "1"))


def _payload_summary(payload: dict) -> dict:
    segment = payload.get("segment") or {}
    translation = payload.get("translation") or payload.get("current_translation") or {}
    review_findings = payload.get("review_findings") or []
    source_text = segment.get("source_text", "")
    translated_text = translation.get("translated_text", "") if isinstance(translation, dict) else ""
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)

    return {
        "payload_chars": len(payload_json),
        "payload_keys": sorted(payload.keys()),
        "segment_id": segment.get("segment_id"),
        "page_number": segment.get("page_number"),
        "block_type": segment.get("block_type"),
        "source_chars": len(source_text),
        "source_preview": text_preview(source_text),
        "translated_chars": len(translated_text),
        "translation_preview": text_preview(translated_text),
        "protected_tokens": len(segment.get("protected_tokens", []) or []),
        "review_findings": len(review_findings),
    }


def _result_summary(result: object) -> dict:
    if isinstance(result, TranslationResult):
        return {
            "segment_id": result.segment_id,
            "translated_chars": len(result.translated_text),
            "confidence": result.confidence,
            "uncertain_terms": result.uncertain_terms,
            "translated_preview": text_preview(result.translated_text),
        }
    if isinstance(result, ReviewResult):
        return {
            "segment_id": result.segment_id,
            "verdict": result.verdict,
            "findings": len(result.findings),
            "finding_severities": [finding.severity for finding in result.findings],
        }
    return {"type": type(result).__name__}


def _estimate_structured_request_tokens(system_prompt: str, payload: dict, schema: type) -> int:
    schema_text = ""
    if hasattr(schema, "model_json_schema"):
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, default=str)
    payload_text = json.dumps(payload, ensure_ascii=False, default=str)
    request_text = "\n\n".join([system_prompt, payload_text, schema_text])
    return _estimate_text_tokens(request_text)


def _estimate_text_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # pragma: no cover - dependency fallback
        return max(1, len(text) // 4)
