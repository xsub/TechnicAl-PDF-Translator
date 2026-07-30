from __future__ import annotations

from translator.pdf.parser import PDFParser
from translator.state import TranslationState


def extract_pdf(state: TranslationState) -> TranslationState:
    parser = PDFParser()
    segments, metadata = parser.extract(state["source_pdf_path"])
    return {
        **state,
        "segments": segments,
        "document_metadata": metadata,
        "status": "pdf_extracted",
    }

