from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BlockType = Literal[
    "heading",
    "paragraph",
    "table_cell",
    "list_item",
    "footnote",
    "header",
    "footer",
    "caption",
]

IssueSeverity = Literal["info", "warning", "critical"]
ReviewSeverity = Literal["critical", "major", "minor", "style"]
Confidence = Literal["high", "medium", "low"]
LLMOperation = Literal["translate", "review", "revise"]


class ProtectedToken(BaseModel):
    token_id: str
    kind: Literal[
        "number",
        "unit",
        "comparator",
        "cas",
        "standard",
        "regulation",
        "date",
        "abbreviation",
        "formula",
    ]
    text: str
    normalized: str
    start: int
    end: int


class DocumentSegment(BaseModel):
    segment_id: str
    page_number: int
    order_index: int
    block_type: BlockType
    source_text: str
    bbox: tuple[float, float, float, float] | None = None
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    column_header: str | None = None
    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    preceding_context: str | None = None
    following_context: str | None = None
    protected_tokens: list[ProtectedToken] = Field(default_factory=list)


class TokenUsage(BaseModel):
    provider: Literal["openai", "anthropic", "mock"]
    model: str
    operation: LLMOperation
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TranslationResult(BaseModel):
    segment_id: str
    translated_text: str
    source_terms_retained: list[str] = Field(default_factory=list)
    uncertain_terms: list[str] = Field(default_factory=list)
    translator_notes: list[str] = Field(default_factory=list)
    confidence: Confidence = "medium"
    token_usage: TokenUsage | None = None


class ValidationIssue(BaseModel):
    segment_id: str
    severity: IssueSeverity
    issue_type: Literal[
        "missing_number",
        "changed_number",
        "missing_unit",
        "changed_unit",
        "missing_reference",
        "changed_comparator",
        "missing_negation",
        "table_alignment",
        "untranslated_fragment",
        "unexpected_addition",
        "forbidden_term",
    ]
    source_value: str | None = None
    translated_value: str | None = None
    message: str


class ReviewFinding(BaseModel):
    segment_id: str
    severity: ReviewSeverity
    category: str
    source_evidence: str
    translation_evidence: str
    explanation: str
    proposed_translation: str | None = None
    confidence: Confidence = "medium"


class ReviewResult(BaseModel):
    segment_id: str
    verdict: Literal["accept", "revise", "human_review"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class OperatorDecision(BaseModel):
    segment_id: str
    action: Literal["accept", "edit", "keep_source"]
    text: str | None = None
    note: str | None = None


class OutputVerification(BaseModel):
    ok: bool
    extracted_text_chars: int
    missing_segments: list[str] = Field(default_factory=list)
    placeholder_leaks: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class JobConfig(BaseModel):
    source_pdf_path: str
    output_dir: str = "storage/output"
    source_language: str = "English"
    target_language: str = "Polish"
    domain: str = "Food-contact packaging, printing materials, chemical migration tests"
    mode: Literal["standard", "high_assurance", "strict_regulatory"] = "standard"
    translator_provider: Literal["mock", "openai"] = "mock"
    reviewer_provider: Literal["mock", "anthropic", "openai"] = "mock"
    glossary_path: str = "translator/domain/glossary.yaml"
    require_human_review: bool = True
    max_revision_attempts: int = 1
    translation_concurrency: int = Field(default=4, ge=1, le=16)
    review_concurrency: int = Field(default=4, ge=1, le=16)
    debug: bool = False
    job_id: str | None = None


class JobReport(BaseModel):
    job_id: str
    source_pdf_path: str
    output_pdf_path: str | None
    report_path: str | None = None
    source_language: str = "English"
    target_language: str = "Polish"
    source_sha256: str | None = None
    output_sha256: str | None = None
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    deterministic_issues: list[ValidationIssue] = Field(default_factory=list)
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    output_verification: OutputVerification | None = None
