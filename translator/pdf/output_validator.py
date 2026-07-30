from __future__ import annotations

import re
from pathlib import Path

from translator.debug import DebugTimer, log_debug
from translator.schemas import OutputVerification, TranslationResult
from translator.utils import normalize_ws


def extract_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    log_debug("pdf.output.extract_text.start", path=str(path))
    try:
        import pdfplumber

        parts = []
        with DebugTimer("pdf.output.extract_text.pdfplumber", path=str(path)):
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
        text = "\n".join(parts)
        if text.strip():
            log_debug("pdf.output.extract_text.done", path=str(path), chars=len(text), engine="pdfplumber")
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        with DebugTimer("pdf.output.extract_text.pypdf", path=str(path)):
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        log_debug("pdf.output.extract_text.done", path=str(path), chars=len(text), engine="pypdf")
        return text
    except Exception as exc:
        raise RuntimeError(f"Nie udało się wyekstrahować tekstu z PDF-a wynikowego: {path}") from exc


def verify_output_pdf(
    output_pdf_path: str | Path,
    translations: dict[str, TranslationResult],
) -> OutputVerification:
    text = extract_pdf_text(output_pdf_path)
    normalized_output = normalize_ws(text).lower()
    missing_segments = []
    placeholder_leaks = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))

    for segment_id, translation in translations.items():
        normalized_segment = normalize_ws(translation.translated_text).lower()
        if not normalized_segment:
            continue
        if len(normalized_segment) <= 2:
            continue
        if normalized_segment not in normalized_output:
            # Extraction can split table cells or wrap long text differently.
            # For longer segments, also accept a prefix/suffix signal.
            prefix = normalized_segment[: min(40, len(normalized_segment))]
            if prefix not in normalized_output:
                missing_segments.append(segment_id)

    messages = []
    if placeholder_leaks:
        messages.append("PDF wynikowy zawiera niepodmienione placeholdery.")
    if missing_segments:
        messages.append("Nie wszystkie zatwierdzone segmenty odnaleziono w ekstrakcji PDF-a wynikowego.")

    result = OutputVerification(
        ok=not placeholder_leaks and not missing_segments,
        extracted_text_chars=len(text),
        missing_segments=missing_segments,
        placeholder_leaks=placeholder_leaks,
        messages=messages,
    )
    log_debug(
        "pdf.output.verify.done",
        path=str(output_pdf_path),
        ok=result.ok,
        extracted_text_chars=result.extracted_text_chars,
        missing_segments=len(result.missing_segments),
        placeholder_leaks=len(result.placeholder_leaks),
        messages=result.messages,
    )
    return result
