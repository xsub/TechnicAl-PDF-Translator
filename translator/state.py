from __future__ import annotations

from typing import Any, TypedDict

from translator.schemas import (
    DocumentSegment,
    JobConfig,
    OperatorDecision,
    OutputVerification,
    ReviewResult,
    TranslationResult,
    ValidationIssue,
)


class TranslationState(TypedDict, total=False):
    job_id: str
    config: JobConfig
    source_pdf_path: str
    source_language: str
    target_language: str
    document_metadata: dict[str, Any]
    segments: list[DocumentSegment]
    translations: dict[str, TranslationResult]
    deterministic_issues: list[ValidationIssue]
    review_results: dict[str, ReviewResult]
    unresolved_segments: list[str]
    revision_required_segments: list[str]
    revision_attempts: dict[str, int]
    translation_memory_hits: int
    persistent_translation_cache_hits: int
    translation_memory_misses: int
    user_phrase_memory_hits: int
    translation_cache_scope: dict[str, Any]
    llm_inflight: dict[str, Any] | None
    operator_decisions: dict[str, OperatorDecision]
    output_pdf_path: str | None
    output_verification: OutputVerification | None
    report_path: str | None
    status: str
