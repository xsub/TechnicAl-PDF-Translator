from __future__ import annotations

import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import build_translator
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import ReviewFinding
from translator.state import TranslationState


def translate_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    client = build_translator(config)
    translations = dict(state.get("translations", {}))
    segments = state.get("segments", [])
    total = len(segments)

    for index, segment in enumerate(segments, start=1):
        if segment.segment_id in translations:
            emit_progress(
                progress_callback,
                stage="translate",
                message="Pomijam już przetłumaczony segment",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue
        started_at = time.perf_counter()
        log_debug(
            "segment.translate.start",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            page_number=segment.page_number,
            block_type=segment.block_type,
            source_chars=len(segment.source_text),
            protected_tokens=len(segment.protected_tokens),
            source_preview=text_preview(segment.source_text),
        )
        emit_progress(
            progress_callback,
            stage="translate",
            message="Tłumaczę segment",
            current=index,
            total=total,
            segment_id=segment.segment_id,
        )
        translations[segment.segment_id] = client.translate(segment, glossary, config)
        translation = translations[segment.segment_id]
        partial_state = {
            **state,
            "translations": translations,
            "status": f"translating {index}/{total}",
        }
        if checkpoint_callback:
            checkpoint_callback(partial_state)
        log_debug(
            "segment.translate.done",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            duration_s=round(time.perf_counter() - started_at, 3),
            translated_chars=len(translation.translated_text),
            confidence=translation.confidence,
            uncertain_terms=translation.uncertain_terms,
            translated_preview=text_preview(translation.translated_text),
        )
        emit_progress(
            progress_callback,
            stage="translate",
            message="Segment przetłumaczony",
            current=index,
            total=total,
            segment_id=segment.segment_id,
        )

    return {**state, "translations": translations, "status": "segments_translated"}


def revise_flagged_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    client = build_translator(config)
    translations = dict(state.get("translations", {}))
    attempts = dict(state.get("revision_attempts", {}))
    review_results = state.get("review_results", {})
    segment_ids = state.get("revision_required_segments", [])
    total = len(segment_ids)

    for index, segment_id in enumerate(segment_ids, start=1):
        segment = next(segment for segment in state.get("segments", []) if segment.segment_id == segment_id)
        current = translations[segment_id]
        findings: list[ReviewFinding] = review_results.get(segment_id).findings if segment_id in review_results else []
        started_at = time.perf_counter()
        log_debug(
            "segment.revise.start",
            segment_id=segment_id,
            index=index,
            total=total,
            findings=len(findings),
            current_preview=text_preview(current.translated_text),
        )
        emit_progress(
            progress_callback,
            stage="revise",
            message="Poprawiam zakwestionowany segment",
            current=index,
            total=total,
            segment_id=segment_id,
        )
        translations[segment_id] = client.revise(segment, current, findings, glossary, config)
        attempts[segment_id] = attempts.get(segment_id, 0) + 1
        partial_state = {
            **state,
            "translations": translations,
            "revision_attempts": attempts,
            "status": f"revising {index}/{total}",
        }
        if checkpoint_callback:
            checkpoint_callback(partial_state)
        log_debug(
            "segment.revise.done",
            segment_id=segment_id,
            index=index,
            total=total,
            duration_s=round(time.perf_counter() - started_at, 3),
            translated_preview=text_preview(translations[segment_id].translated_text),
        )

    return {
        **state,
        "translations": translations,
        "revision_attempts": attempts,
        "revision_required_segments": [],
        "status": "flagged_segments_revised",
    }
