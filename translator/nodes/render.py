from __future__ import annotations

from pathlib import Path

from translator.pdf.output_validator import verify_output_pdf
from translator.pdf.renderer import render_translated_pdf
from translator.state import TranslationState
from translator.utils import ensure_dir


def render_pdf(state: TranslationState) -> TranslationState:
    config = state["config"]
    output_dir = ensure_dir(config.output_dir)
    output_path = output_dir / f"{state['job_id']}_translated.pdf"
    render_translated_pdf(
        state.get("segments", []),
        state.get("translations", {}),
        output_path,
        title=f"Translated {Path(config.source_pdf_path).name}",
    )
    return {**state, "output_pdf_path": str(output_path), "status": "pdf_rendered"}


def verify_output(state: TranslationState) -> TranslationState:
    output_path = state.get("output_pdf_path")
    if not output_path:
        return {**state, "status": "missing_output_pdf"}
    verification = verify_output_pdf(output_path, state.get("translations", {}))
    return {
        **state,
        "output_verification": verification,
        "status": "completed" if verification.ok else "completed_with_output_warnings",
    }

