from __future__ import annotations

import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import build_translator
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import DocumentSegment, ReviewFinding, TranslationResult
from translator.state import TranslationState
from translator.translation_cache import TranslationCache, copy_translation_from_cache, normalize_translation_source


TranslationMemory = dict[str, tuple[str, TranslationResult]]


def translate_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    translation_cache = TranslationCache.for_config(config)
    client = None
    translations = dict(state.get("translations", {}))
    segments = state.get("segments", [])
    total = len(segments)
    memory = _build_translation_memory(segments, translations)
    memory_hits = int(state.get("translation_memory_hits", 0))
    persistent_cache_hits = int(state.get("persistent_translation_cache_hits", 0))
    memory_misses = int(state.get("translation_memory_misses", 0))
    cache_scope = state.get("translation_cache_scope") or translation_cache.scope

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
        memory_key = normalize_translation_source(segment.source_text)
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
            translations[segment.segment_id] = copy_translation_from_cache(
                segment,
                cached_translation,
                source_label=f"segment {source_segment_id}",
            )
            translation = translations[segment.segment_id]
            memory_hits += 1
            partial_state = {
                **state,
                "translations": translations,
                "translation_memory_hits": memory_hits,
                "persistent_translation_cache_hits": persistent_cache_hits,
                "translation_memory_misses": memory_misses,
                "translation_cache_scope": cache_scope,
                "llm_inflight": None,
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

        cached_translation = translation_cache.lookup(segment)
        if cached_translation:
            translations[segment.segment_id] = cached_translation
            memory.setdefault(memory_key, (segment.segment_id, cached_translation))
            persistent_cache_hits += 1
            partial_state = {
                **state,
                "translations": translations,
                "translation_memory_hits": memory_hits,
                "persistent_translation_cache_hits": persistent_cache_hits,
                "translation_memory_misses": memory_misses,
                "translation_cache_scope": cache_scope,
                "llm_inflight": None,
                "status": f"translating {index}/{total}",
            }
            if checkpoint_callback:
                checkpoint_callback(partial_state)
            log_debug(
                "segment.translate.persistent_cache_hit",
                segment_id=segment.segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(cached_translation.translated_text),
                translated_preview=text_preview(cached_translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="translate",
                message="Używam trwałego cache tłumaczenia",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue

        if client is None:
            client = build_translator(config)
        if checkpoint_callback:
            checkpoint_callback(
                {
                    **state,
                    "translations": translations,
                    "translation_memory_hits": memory_hits,
                    "persistent_translation_cache_hits": persistent_cache_hits,
                    "translation_memory_misses": memory_misses,
                    "translation_cache_scope": cache_scope,
                    "llm_inflight": {
                        "operation": "translate",
                        "segment_id": segment.segment_id,
                        "index": index,
                        "total": total,
                    },
                    "status": f"translating {index}/{total}",
                }
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
        memory_misses += 1
        if memory_key:
            memory.setdefault(memory_key, (segment.segment_id, translation))
        translation_cache.store(segment, translation, job_id=state["job_id"])
        partial_state = {
            **state,
            "translations": translations,
            "translation_memory_hits": memory_hits,
            "persistent_translation_cache_hits": persistent_cache_hits,
            "translation_memory_misses": memory_misses,
            "translation_cache_scope": cache_scope,
            "llm_inflight": None,
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
        "persistent_translation_cache_hits": persistent_cache_hits,
        "translation_memory_misses": memory_misses,
        "translation_cache_scope": cache_scope,
        "llm_inflight": None,
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
        if checkpoint_callback:
            checkpoint_callback(
                {
                    **state,
                    "translations": translations,
                    "revision_attempts": attempts,
                    "llm_inflight": {
                        "operation": "revise",
                        "segment_id": segment_id,
                        "index": index,
                        "total": total,
                    },
                    "status": f"revising {index}/{total}",
                }
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
            "llm_inflight": None,
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
        "llm_inflight": None,
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

        key = normalize_translation_source(segment.source_text)
        if key:
            memory.setdefault(key, (segment.segment_id, translation))

    return memory
