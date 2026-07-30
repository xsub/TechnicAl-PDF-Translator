from __future__ import annotations

import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import build_translator
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import DocumentSegment, ReviewFinding, TranslationResult
from translator.state import TranslationState


TranslationMemory = dict[str, tuple[str, TranslationResult]]


def translate_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    client = None
    translations = dict(state.get("translations", {}))
    segments = state.get("segments", [])
    total = len(segments)
    memory = _build_translation_memory(segments, translations)
    memory_hits = int(state.get("translation_memory_hits", 0))
    memory_misses = int(state.get("translation_memory_misses", 0))

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
        memory_key = _translation_memory_key(segment.source_text)
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
        if memory_key and memory_key in memory:
            source_segment_id, cached_translation = memory[memory_key]
            translations[segment.segment_id] = _copy_translation_from_memory(
                segment,
                cached_translation,
                source_segment_id=source_segment_id,
            )
            translation = translations[segment.segment_id]
            memory_hits += 1
            partial_state = {
                **state,
                "translations": translations,
                "translation_memory_hits": memory_hits,
                "translation_memory_misses": memory_misses,
                "status": f"translating {index}/{total}",
            }
            if checkpoint_callback:
                checkpoint_callback(partial_state)
            log_debug(
                "segment.translate.memory_hit",
                segment_id=segment.segment_id,
                source_segment_id=source_segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(translation.translated_text),
                translated_preview=text_preview(translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="translate",
                message="Używam zapisanego tłumaczenia identycznego segmentu",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue

        emit_progress(
            progress_callback,
            stage="translate",
            message="Tłumaczę segment",
            current=index,
            total=total,
            segment_id=segment.segment_id,
        )
        if client is None:
            client = build_translator(config)
        translations[segment.segment_id] = client.translate(segment, glossary, config)
        translation = translations[segment.segment_id]
        memory_misses += 1
        if memory_key:
            memory.setdefault(memory_key, (segment.segment_id, translation))
        partial_state = {
            **state,
            "translations": translations,
            "translation_memory_hits": memory_hits,
            "translation_memory_misses": memory_misses,
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

    return {
        **state,
        "translations": translations,
        "translation_memory_hits": memory_hits,
        "translation_memory_misses": memory_misses,
        "status": "segments_translated",
    }


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


def _build_translation_memory(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
) -> TranslationMemory:
    memory: TranslationMemory = {}
    for segment in segments:
        translation = translations.get(segment.segment_id)
        if not translation:
            continue

        key = _translation_memory_key(segment.source_text)
        if key:
            memory.setdefault(key, (segment.segment_id, translation))

    return memory


def _translation_memory_key(source_text: str) -> str:
    return " ".join(source_text.split())


def _copy_translation_from_memory(
    segment: DocumentSegment,
    cached_translation: TranslationResult,
    *,
    source_segment_id: str,
) -> TranslationResult:
    note = f"Reused exact-match translation from segment {source_segment_id}."
    notes = list(cached_translation.translator_notes)
    if note not in notes:
        notes.append(note)

    return cached_translation.model_copy(
        update={
            "segment_id": segment.segment_id,
            "translator_notes": notes,
        }
    )
